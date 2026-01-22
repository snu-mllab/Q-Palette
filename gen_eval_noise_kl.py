import hashlib
import torch
import random
from lib.config import DEVDIR, MODEL_KEYS
import argparse
import random

import os
import torch
from tqdm import tqdm
import lib.utils as utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils.unsafe_import import model_from_hf_path
import math
import torch.nn.functional as F
torch.set_grad_enabled(False)

NOISE_LEVELS = [
    0.0625 * (16-R) / 16 for R in range(0,16)
]

def get_sha256_hash(input_string: str) -> str:
    return hashlib.sha256(input_string.encode('utf-8')).hexdigest()

def eval_model(model, devset, orig_logits):
    """
        model: Llama
        devset: list of torch.Tensor of size (1, ctx_size_i)
        orig_logits: list of torch.Tensor of size (1, ctx_size_i, vocab_size)
    """
    model = model.half()
    progress = tqdm(range(len(devset)))
    tot_kl = 0.0
    tot_tok = 0
    for ii in progress:
        with torch.inference_mode():
            input = devset[ii].cuda().view(1, -1)
            output = model(input,
                        use_cache=False,
                        output_hidden_states=False,
                        output_attentions=False)[0]      # (1, S, V)
            log_q = F.log_softmax(output, dim=-1)
            log_p = F.log_softmax(orig_logits[ii].to(log_q.device),
                                dim=-1)

            kl = F.kl_div(log_q, log_p,
                        log_target=True,
                        reduction="none").sum(-1)      # (1, S)

            tot_kl  += kl.sum().item()
            tot_tok += kl.numel()
            utils.clean()

    avg_loss = tot_kl / tot_tok
    kl = torch.exp(torch.tensor(avg_loss)).item()
    print(f"kl: {kl}, avg_loss: {avg_loss}")
    return kl, avg_loss

def main(args):
    model_key = MODEL_KEYS[args.model]
    print(f"Loading model {args.model} ({model_key})")
    model, model_str = model_from_hf_path(args.model, max_mem_ratio=args.max_mem_ratio)
    print(f"Loaded model {args.model} ({model_key})")

    print(f"Loading devset and logits for {args.model}")
    dev_path = f"{DEVDIR}/{MODEL_KEYS[args.model]}/rnd_devset_{args.devset_size}_{args.ctx_size}_{args.min_ctx_size}.pt"
    logit_path = f"{DEVDIR}/{MODEL_KEYS[args.model]}/rnd_logits_{args.devset_size}_{args.ctx_size}_{args.min_ctx_size}.pt"
    devset = torch.load(dev_path)
    orig_logits = torch.load(logit_path)
    print(f"Loaded devset from {dev_path} and logits from {logit_path}")


    nlayers = len(model.model.layers)
    group_size = (nlayers // args.group_num)  if args.group_num != -1 else 1

    gidxs = [args.group_idx] if args.group_idx != -1 else list(range(args.group_num)) if args.group_num != -1 else list(range(nlayers))
    for group_idx in gidxs:
        result_dir = f"{DEVDIR}/{model_key}/inject_dev_loss_data/rnd_dev{args.devset_size}_gs{args.group_num}/{group_idx}_{args.partition}/"
        print(f"cur_group_idx: {group_idx}, cur_group_size: {group_size}")
        print(f"Result directory: {result_dir}")

        os.makedirs(result_dir, exist_ok=True)
        if len(os.listdir(result_dir)) == 16:
            print(f"Result directory {result_dir} already exists. Exiting.")
            continue

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        tokenizer.pad_token = tokenizer.eos_token

        mkey, skey = args.partition.split(".")
        orig_weight_group, orig_scale_group = [], []
        assert nlayers % args.group_num == 0, f"nlayers = {nlayers} is not divisible by group_num = {args.group_num}"
        sidx = group_size * group_idx
        eidx = min(sidx + group_size, nlayers)

        for lidx in range(sidx, eidx):
            orig_weight = getattr(getattr(model.model.layers[lidx], mkey), skey).weight.data
            orig_scale = orig_weight.float().pow(2).mean().sqrt().item()
            orig_weight_group.append(orig_weight.clone())
            orig_scale_group.append(orig_scale)
            utils.clean()

        orig_ppl, orig_avg_loss = eval_model(model, devset, orig_logits)

        for nidx, noise_level in enumerate(NOISE_LEVELS):
            result_path = os.path.join(result_dir, f"{nidx}_{noise_level}.pt")
            if os.path.exists(result_path):
                print(f"Result file {result_path} already exists. Skipping.")
                continue

            print("="*100)
            print(f"Sampling {noise_level} noise in group {group_idx} ({sidx} - {eidx}) partition {args.partition}")

            errs = []
            print(f"[{nidx}/{len(NOISE_LEVELS)}] inject noise ({noise_level}) to layers from {sidx} to {eidx}")
            for lidx in range(len(orig_weight_group)):
                orig_weight = orig_weight_group[lidx]
                orig_scale = orig_scale_group[lidx]
                noise = torch.randn_like(orig_weight) * math.sqrt(noise_level)
                noised_weight = orig_weight + noise * orig_scale
                getattr(getattr(model.model.layers[int(sidx + lidx)], mkey), skey).weight.data = noised_weight
                
                errs.append(noise.float().pow(2).mean().cpu().item())

                del H
                utils.clean()

            ppl, avg_loss = eval_model(model, devset, orig_logits)

            results = {
                "noise_level": noise_level,
                "err": sum(errs) / len(errs),
                "err_list": errs,
                "orig_ppl": orig_ppl,
                "ppl": ppl,
                "orig_avg_loss": orig_avg_loss,
                "avg_loss": avg_loss,
            }
            torch.save(results, result_path)

            print("noise_level: ", noise_level, "err: ", noise.float().pow(2).mean().cpu().item())
            print(f"ppl: {ppl}, avg_loss: {avg_loss}")
            print(f"Saved results to {result_path}")
            print("="*100)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--devset_size', default=32, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--ctx_size', default=4096, type=int)
    parser.add_argument('--min_ctx_size', default=256, type=int)
    parser.add_argument('--partition', type=str, required=True)
    parser.add_argument('--group_idx', type=int, required=True)
    parser.add_argument('--group_num', type=int, default=-1)
    parser.add_argument('--max_mem_ratio', type=float, default=0.8)
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
