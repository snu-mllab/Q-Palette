# This script is based off of the generation script in https://github.com/chu-tianxiang/QuIP-for-all
import os
import time
from typing import Optional

import torch
from transformers import AutoTokenizer
from model.cache_utils import StaticCache

from lib.utils.unsafe_import import model_from_hf_path

torch.set_grad_enabled(False)
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 

SKEY_DICT = {
    "self_attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp": ["gate_proj", "up_proj", "down_proj"],
}

def load_quant_model(model, args):
    from lib.linear.incoherent_linear import IncoherentLinear, IncoherentMLP, IncoherentSdpaAttention
    quant_dir_key = f"left_only_seed{args.seed}_cache"

    if args.hf_path == "meta-llama/Llama-3.1-8B":
        quant_dir = f"quant_results/3_8b/{quant_dir_key}"
    else:
        raise ValueError(f"Model name {args.hf_path} not supported")
    
    nlayers = len(model.model.layers)
    if args.quantizer_str is not None:
        qdict = {}
        for i in range(nlayers):
            for key in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                qdict[f"{i}_{key}"] = args.quantizer_str
    else:
        qdict = torch.load(args.qdict_path)
        for i in range(nlayers):
            for key in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                print(i, key, qdict[f"{i}_{key}"])
    
    if args.merge_info_path != "":
        merge_info = torch.load(args.merge_info_path)
    else:
        merge_info = None
        
    for i, layer in enumerate(model.model.layers):
        if merge_info is not None:
            print(merge_info[i]) 
        merge_ug = 'merge_ug' in merge_info[i] if merge_info is not None else args.merge_ug
        merge_qk = 'merge_qk' in merge_info[i] if merge_info is not None else args.merge_qk
        merge_kv = 'merge_kv' in merge_info[i] if merge_info is not None else args.merge_kv
        merge_qv = 'merge_qv' in merge_info[i] if merge_info is not None else args.merge_qv
        merge_qkv = 'merge_qkv' in merge_info[i] if merge_info is not None else args.merge_qkv
        for mkey in ["self_attn", "mlp"]:
            
            if mkey == "mlp" and args.use_inc_mlp:
                mlp = IncoherentMLP.gen_layer_from_quantizer_str_and_key(
                                                    model.config, 
                                                    quant_dir, 
                                                    quantizer_str_up=qdict[f"{i}_mlp.up_proj"], 
                                                    quantizer_str_gate=qdict[f"{i}_mlp.gate_proj"], 
                                                    quantizer_str_down=qdict[f"{i}_mlp.down_proj"], 
                                                    key_up=f"{i}_mlp.up_proj", 
                                                    key_gate=f"{i}_mlp.gate_proj", 
                                                    key_down=f"{i}_mlp.down_proj", 
                                                    merge_ug=merge_ug,
                                                    dummy=args.dummy,
                                                    use_simt=args.use_simt
                                                )
                setattr(layer, mkey, mlp)
            elif mkey == "self_attn" and args.use_inc_attn:
                attn = IncoherentSdpaAttention.gen_layer_from_quantizer_str_and_key(
                                                    model.config, 
                                                    i,
                                                    quant_dir, 
                                                    quantizer_str_q=qdict[f"{i}_self_attn.q_proj"], 
                                                    quantizer_str_k=qdict[f"{i}_self_attn.k_proj"], 
                                                    quantizer_str_v=qdict[f"{i}_self_attn.v_proj"], 
                                                    quantizer_str_o=qdict[f"{i}_self_attn.o_proj"], 
                                                    key_q=f"{i}_self_attn.q_proj", 
                                                    key_k=f"{i}_self_attn.k_proj", 
                                                    key_v=f"{i}_self_attn.v_proj", 
                                                    key_o=f"{i}_self_attn.o_proj", 
                                                    merge_qk=merge_qk, 
                                                    merge_kv=merge_kv,
                                                    merge_qv=merge_qv,
                                                    merge_qkv=merge_qkv,
                                                    dummy=args.dummy,
                                                    use_simt=args.use_simt,
                                                )
                setattr(layer, mkey, attn)
            else:
                for skey in SKEY_DICT[mkey]:
                    key = f"{mkey}.{skey}"
                    setattr(getattr(layer, mkey), skey, IncoherentLinear.gen_layer_from_quantizer_str_and_key(
                        model.config, quant_dir, qdict[f"{i}_{key}"], f"{i}_{key}", merge_layers=True, use_simt=args.use_simt, dummy=args.dummy))
    return model.cuda().to(torch.float16)

def multinomial_sample_one_no_sync(
        probs_sort
):  # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1,
                        keepdim=True).to(dtype=torch.int)


def logits_to_probs(logits,
                    temperature: float = 1.0,
                    top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

@torch.compile
def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs


@torch.no_grad()
def decode_one_tokens(model, cur_token, past_kv, cache_position):
    logits = model(cur_token,
                   past_key_values=past_kv,
                   cache_position=cache_position)[0]
    new_token = sample(logits, temperature=0.6, top_k=5)[0]
    return new_token, logits

@torch.no_grad()
def generate(model, inputs, max_new_tokens, callback, past_kv, enable_flash=True):
    batch_size, seq_length = inputs.shape
    cache_position = torch.arange(seq_length, device=0)

    T = inputs.size(-1)
    T_new = T + max_new_tokens
    empty = torch.empty(batch_size, T_new, dtype=inputs.dtype, device=inputs.device)
    empty[:, :T] = inputs
    seq = empty
    generated_ids = []

    cur_token = inputs.clone()
    for _ in range(max_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=enable_flash,
                                            enable_mem_efficient=False,
                                            enable_math=True):
            next_token, logits = decode_one_tokens(model, cur_token.clone(),
                                                   past_kv, cache_position)
        cache_position += 1
        generated_ids.append(next_token.clone())
        callback(generated_ids[-1])
        cur_token = next_token.clone()
    torch.cuda.synchronize()
    seq[:, 1:] = torch.cat(generated_ids, dim=-1)
    return seq

def _get_model_size(model):
    import itertools
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            print(name)
            size = sum(p.numel() * p.dtype.itemsize for p in itertools.chain(child.parameters(), child.buffers()))
            print(size/1024/1024/1024)

    return model_size, params

def main(hf_path, compile, max_tokens, enable_flash, num_samples, args, print_result=False):
    model, model_str = model_from_hf_path(hf_path)
    if not args.num_hidden_layers == -1:
        model.model.layers = model.model.layers[:args.num_hidden_layers]
    model = model.to(torch.float16)
    
    if args.quantizer_str is not None or args.qdict_path is not None:
        model = load_quant_model(model, args)
                                                   
    
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    tokenizer.pad_token = tokenizer.eos_token
    past_kv = StaticCache(model.config,
                              args.batch_size,
                              args.max_new_tokens*2,
                              device=0,
                              dtype=model.dtype)

    text = "This is a test of this large language model"
    inputs = tokenizer(text, return_tensors="pt").to(0)
    inputs = inputs['input_ids'][:, -1:].expand(args.batch_size, -1)
    print(inputs)
    prompt_length = inputs.size(-1)

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)

    callback = lambda x: x
    ids = generate(model, inputs, 8, callback, past_kv, enable_flash=enable_flash)
    print(ids)

    if compile:
        print('Capturing CUDA graphs, may take some time. If you are running a model over multiple GPUs, the first generation will be very slow due to compiling the model.')
        global decode_one_tokens
        decode_one_tokens = torch.compile(decode_one_tokens,
                                            mode="max-autotune",
                                            fullgraph=True)


    text = "This is a test of this large language model"
    ids = generate(model, inputs, 16, callback, past_kv, enable_flash=enable_flash)
    print(ids)
    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0
    for i in range(start, num_samples):
        print(i)
        callback = lambda x : x

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        import contextlib
        prof = contextlib.nullcontext()

        with prof:
            y = generate(
                model,
                inputs,
                max_tokens,
                callback=callback,
                past_kv=past_kv,
                enable_flash=enable_flash
            )

        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds", flush=True)
            continue

        torch.cuda.synchronize()
        time_elapsed = time.perf_counter() - t0

        if print_result and i <= 1:
            print(tokenizer.decode(y[0].tolist()))
        
        tokens_generated = (y.size(-1) - prompt_length)
        generated_tokens_sec = tokens_generated / time_elapsed
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec  * args.batch_size)
        print(f"Time for inference {i + 1}: {time_elapsed:.02f} sec total, {generated_tokens_sec * args.batch_size:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        total_tokens_sec = y.numel() / time_elapsed
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print(flush=True)
    print("==========")
    print("model size: ", model_size / 1e9, "GB")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    # print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

    result_dict = {
        "average_tokens_per_sec": torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item(),
        "generated_tokens": max_tokens,
        "quantizer_str": args.quantizer_str,
        "qdict_path": args.qdict_path,
        "use_inc_mlp": args.use_inc_mlp,
        "use_inc_attn": args.use_inc_attn,
        "merge_ug": args.merge_ug,
        "merge_qk": args.merge_qk,
        "merge_qkv": args.merge_qkv,
        "model_size": model_size / 1e9,
        "prompt_length": prompt_length,
    }
    if args.save_key != "": 
        save_dir = f"./eval_results/latency/{args.hf_path}/{args.save_key}.pt"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        torch.save(result_dict, save_dir)

        # save text
        save_dir = f"./eval_results/latency/{args.hf_path}/{args.save_key}.txt"
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        str_ = ""
        for k, v in result_dict.items():
            str_ += f"{k}: {v}\n"
        with open(save_dir, "w") as f:
            f.write(str_)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--hf_path', type=str, help="Path to checkpoint")
    parser.add_argument('--qdict_path', type=str, help="Path to qdict", default=None)
    parser.add_argument('--max_new_tokens',
                        type=int,
                        default=64,
                        help='Maximum number of new tokens.')
    parser.add_argument('--top_k',
                        type=int,
                        default=32,
                        help='Top-k for sampling.')
    parser.add_argument('--disable_tf32',
                        action='store_true',
                        help='Whether to disable TF32 for FP32 matmuls.')
    parser.add_argument('--enable_flash',
                        action='store_true',
                        help='Whether to enable flash attention.')
    parser.add_argument('--use_inc_mlp',
                        action='store_true',
                        help='Whether to use incoherent mlp.')
    parser.add_argument('--use_inc_attn',
                        action='store_true',
                        help='Whether to use incoherent attention.')
    parser.add_argument('--merge_ug',
                        action='store_true',
                        help='Whether to merge ug.')
    parser.add_argument('--merge_qk',
                        action='store_true',
                        help='Whether to merge qk.')
    parser.add_argument('--merge_kv',
                        action='store_true',
                        help='Whether to merge kv.')
    parser.add_argument('--merge_qv',
                        action='store_true',
                        help='Whether to merge qv.')
    parser.add_argument('--merge_qkv',
                        action='store_true',
                        help='Whether to merge qkv.')
    parser.add_argument('--merge_info_path',
                        type=str,
                        default="",
                        help='Merge info.')
    parser.add_argument('--num_samples',
                        type=int,
                        default=5,
                        help='Number of samples to run.')
    parser.add_argument('--print_result',
                        action='store_true',
                        help='Whether to print the result.')
    parser.add_argument('--quantizer_str', 
                        type=str,
                        default=None,
                        help='Quantizer string.')
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help='Seed for the random number generator.')
    parser.add_argument('--save_key',
                        type=str,
                        default="",
                        help='Save key.')
    parser.add_argument('--no_compile',
                        action='store_true',
                        help='Whether to compile the model.')
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help='Batch size.')
    parser.add_argument('--use_simt',
                        action='store_true',
                        help='Whether to use simt.')
    parser.add_argument('--dummy',
                        action='store_true',
                        help='Whether to use dummy quantizers.')
    parser.add_argument('--num_hidden_layers',
                        type=int,
                        default=-1,
                        help='Number of hidden layers.')
    args = parser.parse_args()

    if not args.disable_tf32:
        torch.set_float32_matmul_precision('high')

    main(args.hf_path, not args.no_compile, args.max_new_tokens, args.enable_flash, args.num_samples, args, args.print_result)
