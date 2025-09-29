import torch
import random
from lib.utils import clean
import argparse
import random

import os
import glog
import torch
from tqdm import tqdm
from lib.linear.incoherent_linear import IncoherentLinear
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.utils import gptq_data_utils
from lib.config import MODEL_KEYS
torch.set_grad_enabled(False)

def eval_model(model, devset):
    model = model.to("cuda", dtype=torch.float16)
    loss_fct = torch.nn.CrossEntropyLoss().cuda()
    acc_loss = 0.0
    progress = tqdm(range(devset.shape[0]))
    for ii in progress:
        input = devset[ii, :].cuda().view(1, -1)
        output = model(input,
                    use_cache=False,
                    output_hidden_states=False,
                    output_attentions=False)[0]
        shift_logits = output[:, :-1, :].contiguous()
        shift_labels = input[:, 1:]
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1))
        acc_loss += loss.item()
        progress.set_description(f"avg_loss = {acc_loss/(ii+1)}")

    avg_loss = acc_loss / devset.shape[0]
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    glog.info(f'perplexity: {ppl}')
    return ppl, avg_loss

from quantize_layer import quantize
def load_model(model, qdict, save_dir, model_key, seed=0):
    quant_dir = f"{save_dir}/{model_key}/left_only_seed{seed}_cache"

    for i, layer in enumerate(model.model.layers):
        for key in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            ]:
            mkey, skey = key.split(".")
            if isinstance(qdict, str):
                quantizer_str = qdict
                simt = False
            elif isinstance(qdict[f"{i}_{key}"], tuple):
                quantizer_str, simt_str = qdict[f"{i}_{key}"]
                assert simt_str in ["0", "1"], f"simt should be 0 or 1, but got {simt_str}"
                simt = True if simt_str == "1" else False
            else:
                quantizer_str = qdict[f"{i}_{key}"]
                simt = False
            if not os.path.exists(f"{quant_dir}/{quantizer_str}/{i}_{key}.pt"):
                target_linear = getattr(getattr(layer, mkey), skey)
                quantize(target_linear, model_key, i, key, quantizer_str, seed, left_only=True, save_dir=save_dir)
            setattr(getattr(layer, mkey), skey, IncoherentLinear.gen_layer_from_quantizer_str_and_key(
                model.config, quant_dir, quantizer_str, f"{i}_{key}", merge_layers=True, use_simt=simt))
        clean()
    return model

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", low_cpu_mem_usage=True)#
    model_str = args.model
    model_key = MODEL_KEYS[args.model]

    datasets = ['wikitext2']
    result_path = args.qdict_path.replace(".pt", f"_result.pt")
    if os.path.exists(result_path) and not args.re_eval:
        results = torch.load(result_path)
        print(f"Loaded results from {result_path}")
        return
    if args.quantizer_str is None:
        qdict = torch.load(args.qdict_path)
    else:
        qdict = args.quantizer_str
        args.qdict_path = f"msq_results/{model_key}/{args.quantizer_str}.pt"

    print("="*100)
    model = load_model(model, qdict, args.save_dir, model_key, seed=args.seed).cuda().to(torch.float16)

    results = {}
    for dataset in datasets:
        input_tok = gptq_data_utils.get_test_tokens(dataset,
                                                    seed=args.seed,
                                                    seqlen=args.ctx_size,
                                                    model=model_str)
        nsamples = input_tok.numel() // args.ctx_size
        input_tok = input_tok[0, :(args.ctx_size * nsamples)].view(
            nsamples, args.ctx_size)
        ppl, avg_loss = eval_model(model, input_tok)

        results[dataset] = {
            "ppl": ppl,
            "avg_loss": avg_loss,
            "quant_info_dict": qdict,
        }
        print(f"ppl: {ppl}, avg_loss: {avg_loss}")
        print("="*100)

    result_path = args.qdict_path.replace(".pt", f"_result.pt")
    torch.save(results, result_path)
    print(f"Saved results to {result_path}")
    str_ = ""
    for dataset in datasets:
        str_ += f"{dataset}, {results[dataset]['ppl']}, {results[dataset]['avg_loss']}\n"
    result_path = args.qdict_path.replace(".pt", f"_result.txt") 
    with open(result_path, "w") as f:
        f.write(str_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--qdict_path', default='./msq_results/figure1d/0.0_8.0bit_1.17.pt', type=str)
    parser.add_argument('--quantizer_str', default=None, type=str)
    parser.add_argument('--left_only', action='store_true')
    parser.add_argument('--ctx_size', default=8192, type=int)
    parser.add_argument('--save_dir', default='quant_results/', type=str)
    parser.add_argument('--re_eval', default=False, action='store_true')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
