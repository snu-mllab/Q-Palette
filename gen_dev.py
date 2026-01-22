import os
import torch
from lib.config import DEVDIR, MODEL_KEYS
import lib.utils as utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.utils.unsafe_import import model_from_hf_path
import argparse

def gen_data_from_model(model, tokenizer, device="cuda", ctx_size=4096, devset_size=64, batch_size=1, min_ctx_size=2048):
    devset = []

    bos_id = tokenizer.bos_token_id
    input_ids = torch.full((batch_size, 1),
                        bos_id,
                        dtype=torch.long,
                        device=device)
    total_len = 0
    while total_len < devset_size * ctx_size:
        with torch.inference_mode():
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=ctx_size-1,
                do_sample=True,
                temperature=1.0,
                top_k=50,             
                top_p=0.98,
            )
            # detokenize
            print(out.shape)
            out_text = tokenizer.decode(out[0], skip_special_tokens=True)
            print(out_text[:100])
        if out.shape[1] > min_ctx_size:
            tokens = tokenizer([out_text],
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=ctx_size)
            print(tokens.input_ids[0].shape)
            devset.append(tokens.input_ids[0])
            total_len += tokens.input_ids[0].shape[0]
        print(f"Current length: {total_len / ctx_size} * {ctx_size}")
        utils.clean()
    print(f"\n\nTotal length: {total_len / ctx_size} * {ctx_size}")
    return devset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--devset_size", type=int, default=32)
    parser.add_argument("--ctx_size", type=int, default=4096)
    parser.add_argument("--min_ctx_size", type=int, default=256)
    args = parser.parse_args()

    if DEVDIR == "set appropriate path":
        raise ValueError("DEVDIR is not set")

    os.makedirs(f"{DEVDIR}/{MODEL_KEYS[args.model]}", exist_ok=True)
    model, model_str = model_from_hf_path(args.model, max_mem_ratio=0.8)
    model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    if "Qwen" in args.model:
        tokenizer.bos_token = tokenizer.eos_token
        tokenizer.bos_token_id = tokenizer.eos_token_id
    devset = gen_data_from_model(model, tokenizer, devset_size=args.devset_size, ctx_size=args.ctx_size, min_ctx_size=args.min_ctx_size)

    # cache logits
    logits = []
    for toks in devset:
        input = toks.view(1, -1).cuda()
        with torch.inference_mode():
            output = model(input, use_cache=False, output_hidden_states=False, output_attentions=False)[0]
            logits.append(output.cpu())
            print(logits[-1].shape)
        utils.clean()
    torch.save(logits, f"{DEVDIR}/{MODEL_KEYS[args.model]}/rnd_logits_{args.devset_size}_{args.ctx_size}_{args.min_ctx_size}.pt")
    torch.save(devset, f"{DEVDIR}/{MODEL_KEYS[args.model]}/rnd_devset_{args.devset_size}_{args.ctx_size}_{args.min_ctx_size}.pt")

if __name__ == "__main__":
    main()