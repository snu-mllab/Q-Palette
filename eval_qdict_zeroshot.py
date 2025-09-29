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
import lib.utils as utils
from transformers import AutoModelForCausalLM, AutoTokenizer
from lib.config import MODEL_KEYS
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
torch.set_grad_enabled(False)

def load_model(model, qdict, quant_dir):
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
            quantizer_str = qdict[f"{i}_{key}"]
            if isinstance(quantizer_str, tuple):
                quantizer_str, simt_str = quantizer_str
                simt = True if simt_str == "1" else False
            else:
                simt = False
            setattr(getattr(layer, mkey), skey, IncoherentLinear.gen_layer_from_quantizer_str_and_key(
                                        model.config, quant_dir, quantizer_str, f"{i}_{key}", merge_layers=True, use_simt=simt))
        clean()
    return model

def main(args):
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto", low_cpu_mem_usage=True).cuda().to(torch.float16)
    model_str = args.model
    model_key = MODEL_KEYS[args.model]

    datasets = ['wikitext2', 'c4']
    quant_dir = f"quant_results/{model_key}"
    if args.left_only:
        quant_dir = f"quant_results/{model_key}/left_only_seed{args.seed}_cache"
    else:
        raise ValueError(f"Unsupported model: {args.model}")

    qdict = torch.load(args.qdict_path)

    print("="*100)
    model = load_model(model, qdict, quant_dir)

    model = model.to("cuda", dtype=torch.float16)
    results = {}

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    glog.info('loaded model!')
    tokenizer.pad_token = tokenizer.eos_token

    task_names = args.tasks.split(",")

    lm_eval_model = HFLM(model,
                         tokenizer=tokenizer,
                         batch_size=args.batch_size)

    results = evaluator.simple_evaluate(
        model=lm_eval_model,
        tasks=task_names,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn)

    for key in results['results']:
        print(key)
        print()
        print(results['results'][key])
        print()
        print()

    accs = [results['results'][key]['acc,none'] for key in results['results']]
    print("avg_acc", sum(accs) / len(accs))

    str_ = f"avg, {sum(accs) / len(accs)}\n"
    for key in results['results']:
        str_ += f"{key}, {results['results'][key]['acc,none']}\n"

    result_path = args.qdict_path.replace(".pt", f"_zeroshot.txt")
    with open(result_path, "w") as f:
        f.write(str_)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='meta-llama/Llama-3.1-8B', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--qdict_path', default='./msq_results/figure1d/0.0_8.0bit_1.17.pt', type=str)
    parser.add_argument('--left_only', action='store_true')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')

    parser.add_argument("--tasks", type=str, default="arc_challenge,arc_easy,piqa,winogrande,hellaswag")
    parser.add_argument('--num_fewshot', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--apply_chat_template', action='store_true')
    parser.add_argument('--fewshot_as_multiturn', action='store_true')
    torch.set_grad_enabled(False)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    main(args)
