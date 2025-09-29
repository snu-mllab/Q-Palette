import torch
import os 
import argparse
import numpy as np
QDICT = {
    "default": {
        "tcq_3_none_0.9":1.5,
        "tcq_4_none_0.9":2.0,
        "tcq_5_none_0.9":2.5,
        "tcq_6_none_0.9":3.0,
        "tcq_7_none_0.9":3.5,
        "tcq_8_none_0.9":4.0,
        "tcq_9_none_0.9":4.5,
        "tcq_10_none_0.9":5.0,
        "tcomb_3_4_0.5_none_0.9":1.75,
        "tcomb_4_5_0.5_none_0.9":2.25,
        "tcomb_5_6_0.5_none_0.9":2.75,
        "tcomb_6_7_0.5_none_0.9":3.25,
        "tcomb_7_8_0.5_none_0.9":3.75,
        "tcomb_8_9_0.5_none_0.9":4.25,
        "tcomb_9_10_0.5_none_0.9":4.75,
        "ldlq_1_2_none_1.0":2.0,
        "ldlq_1_3_none_1.0":3.0,
        "ldlq_1_4_none_1.0":4.0,
        "ldlq_1_5_none_1.0":5.0,
        "ldlq_1_6_none_1.0":6.0,
        "ldlq_1_7_none_1.0":7.0,
        "ldlq_1_8_none_1.0":8.0,
        "ldlq_2_3_none_1.0":1.5,
        "ldlq_2_4_none_1.0":2.0,
        "ldlq_2_5_none_1.0":2.5,
        "ldlq_2_6_none_1.0":3.0,
        "ldlq_2_7_none_1.0":3.5,
        "ldlq_2_8_none_1.0":4.0,
        "ldlq_2_9_none_1.0":4.5,
        "ldlq_2_10_none_1.0":5.0,
        "ldlq_2_11_none_1.0":5.5,
        "ldlq_2_12_none_1.0":6.0,
    },
}

from ortools.linear_solver import pywraplp
# solve optimal quantizers for x
SIMPLEKEY_TO_KEY = {
    "q": "self_attn.q_proj",
    "k": "self_attn.k_proj",
    "v": "self_attn.v_proj",
    "o": "self_attn.o_proj",
    "u": "mlp.up_proj",
    "g": "mlp.gate_proj",
    "d": "mlp.down_proj",
}

def solve_optimal_quantizer(qlist, err_coeff_dict, lat_coeff_dict, mem_err_val_dict, lat_limit, imp_key, nlayers, use_cc=False, mem_limit=None):
    """
        qlist: list of str 
        err_coeff_dict: dict 
        lat_coeff_dict: dict
        mem_err_val_dict: dict
    """

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SetNumThreads(24)
    solver.SetTimeLimit(60000)
    if not solver:
        return
    
    # assignment P
    P = []
    for lidx in range(nlayers):
        P.append({})
        for qidx in range(len(qlist)):
            if qlist[qidx].startswith("ldlq") and use_cc:
                simts = [1, 0]
            else:
                simts = [0]
            for simt in simts:
                P[lidx][f"q_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_q_{qidx}_{simt}")
                P[lidx][f"k_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_k_{qidx}_{simt}")
                P[lidx][f"v_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_v_{qidx}_{simt}")
                P[lidx][f"o_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_o_{qidx}_{simt}")
                P[lidx][f"u_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_u_{qidx}_{simt}")
                P[lidx][f"g_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_g_{qidx}_{simt}")
                P[lidx][f"d_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_d_{qidx}_{simt}")    
                if not args.no_fuse:
                    P[lidx][f"qk_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_qk_{qidx}_{simt}")
                    P[lidx][f"kv_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_kv_{qidx}_{simt}")
                    P[lidx][f"qv_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_qv_{qidx}_{simt}")
                    P[lidx][f"qkv_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_qkv_{qidx}_{simt}")    
                    P[lidx][f"ug_{qidx}_{simt}"] = solver.IntVar(0.0, 1.0, f"P_{lidx}_ug_{qidx}_{simt}")
    print("Number of variables =", solver.NumVariables())

    # 1. assignment constraint
    for lidx in range(nlayers):
        for key in ["q", "k", "v", "o", "u", "g", "d"]:
            var_has_key = [P[lidx][name] for name in P[lidx] if key in name]
            solver.Add(sum(var_has_key) == 1)
    
    # 2. memory constraint
    if mem_limit is not None:
        mexpr = 0
        for lidx in range(nlayers):
            for name in P[lidx]:
                simple_key, qidx, simt = name.split("_")
                qidx = int(qidx)
                for skey in ["q", "k", "v", "o", "u", "g", "d"]:
                    if skey in simple_key:
                        dictkey = f"{lidx}_{SIMPLEKEY_TO_KEY[skey]}"
                        mexpr += P[lidx][name] * mem_err_val_dict[dictkey][qlist[qidx]]["mem"]
        solver.Add(mexpr <= mem_limit)

    # 3. latency constraint
    lexpr = float(lat_coeff_dict['constant'].item())
    for lidx in range(nlayers):
        for name in P[lidx]:
            simple_key, qidx, simt = name.split("_")
            qidx = int(qidx)
            quantizer_str = qlist[qidx]
            simt_str = "True" if simt == 1 else "False"
            lat_coeff = lat_coeff_dict[f"{simple_key}_{quantizer_str}_{simt_str}"] 
            lexpr += P[lidx][name] * lat_coeff
    solver.Add(lexpr <= lat_limit)

    print("Number of constraints =", solver.NumConstraints())
    # 4. objective
    obj = 0
    for lidx in range(nlayers):
        for name in P[lidx]:
            simple_key, qidx, simt = name.split("_")
            qidx = int(qidx)
            for skey in ["q", "k", "v", "o", "u", "g", "d"]:
                if skey in simple_key:
                    dictkey = f"{lidx}_{SIMPLEKEY_TO_KEY[skey]}"
                    obj += P[lidx][name] * mem_err_val_dict[dictkey][qlist[qidx]][imp_key] * float(err_coeff_dict[dictkey])
    solver.Maximize(-obj)
    status = solver.Solve()

    print("Objective value =", -solver.Objective().Value())
    quantizer_dict = {}
    merge_info = []
    for lidx in range(nlayers):
        merge_info.append([])
        for name in P[lidx]:
            if P[lidx][name].solution_value() > 0.5:
                simple_key, qidx, simt = name.split("_")
                quantizer_str = qlist[int(qidx)]
                for skey in ["q", "k", "v", "o", "u", "g", "d"]:
                    if skey in simple_key:
                        dictkey = f"{lidx}_{SIMPLEKEY_TO_KEY[skey]}"
                        quantizer_dict[dictkey] = (quantizer_str, simt)
                if len(simple_key) > 1:
                    if simple_key == "qk":
                        merge_info[lidx].append(f"merge_qk")
                    elif simple_key == "kv":
                        merge_info[lidx].append(f"merge_kv")
                    elif simple_key == "qv":
                        merge_info[lidx].append(f"merge_qv")
                    elif simple_key == "qkv":
                        merge_info[lidx].append(f"merge_qkv")
                    elif simple_key == "ug":
                        merge_info[lidx].append(f"merge_ug")
    print("\nAdvanced usage:")
    print(f"Problem solved in {solver.wall_time():d} milliseconds")
    print(f"Problem solved in {solver.iterations():d} iterations")
    print(f"Problem solved in {solver.nodes():d} branch-and-bound nodes")

    return quantizer_dict, merge_info

def lidx_to_key(lidx):
    bidx = lidx // 7
    lkey = [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
            ][lidx % 7]
    return f"{bidx}_{lkey}"

def load_mem_err_val_dict(quantizer_type, nlayers):
    from lib.utils.mem_op import get_layer_mem, get_constant_mem
    mem_err_val_dict = {}
    qlist = list(QDICT[quantizer_type].keys())
    for key in [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
        ]:
        cached_errs = torch.load("assets/quant_err.pt")
        for i in range(nlayers):
            mem_err_val_dict[f"{i}_{key}"] = {}
            for quantizer_str in qlist:
                mem = get_layer_mem(model_key, f"{key}", quantizer_str)
                orig_mem = get_layer_mem(model_key, f"{key}", "default")
                mem_err_val_dict[f"{i}_{key}"][quantizer_str] = {
                    "err": cached_errs[quantizer_str],
                    "mem": mem,
                    "orig_mem": orig_mem,
                }
    mem_err_val_dict["constant"] = {
        "err": 0.0,
        "mem": get_constant_mem(model_key) * nlayers,
        "orig_mem": 0.0,
    }
    return mem_err_val_dict


def err_coeff_routine(model_key):
    err_coeffs = torch.load(f'assets/{model_key}_err_coeffs.pt')
    return err_coeffs

def lat_coeff_routine(model_key, nodename):
    lat_coeffs = torch.load(f"assets/{model_key}_latency_coeffs_{nodename}.pt")
    return lat_coeffs

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--quantizer_type", type=str, default="default", choices=["default"])
parser.add_argument('--imp_key', type=str, default="err", choices=["err"])
parser.add_argument('--nodename', type=str, default="4090_cc")
parser.add_argument('--no_fuse', action="store_true")
parser.add_argument('--target_thp', type=float, default=200)
parser.add_argument('--use_cc', action="store_true")

args = parser.parse_args()
if args.model == "meta-llama/Llama-3.1-8B":
    model_key, nlayers = "3_8b", 32
elif args.model == "meta-llama/Llama-3.2-1B":
    model_key, nlayers = "3_1b", 16
elif args.model == "meta-llama/Llama-3.2-3B":
    model_key, nlayers = "3_3b", 28
else:
    raise ValueError(f"Unsupported model: {args.model}")

# 1. aggregate quantization results
mem_err_val_dict = load_mem_err_val_dict(args.quantizer_type, nlayers)
nlinears = nlayers * 7 
assert len(mem_err_val_dict) == nlinears + 1 # 1 for constant
print(len(mem_err_val_dict), len(mem_err_val_dict["0_self_attn.q_proj"]))

# 2. get coeff data
err_coeff_dict = err_coeff_routine(model_key)
lat_coeff_dict = lat_coeff_routine(model_key, args.nodename)

if not args.no_fuse:
    save_dir = f"msq_results/{model_key}/lat_constrained/{args.nodename}/{args.quantizer_type}_{args.imp_key}"
else:
    save_dir = f"msq_results/{model_key}/lat_constrained_no_fuse/{args.nodename}/{args.quantizer_type}_{args.imp_key}"
os.makedirs(save_dir, exist_ok=True)

lat_limit = 1.0 / args.target_thp
qlist = list(QDICT[args.quantizer_type].keys())
quantizer_dict, merge_info = solve_optimal_quantizer(qlist, err_coeff_dict, lat_coeff_dict, mem_err_val_dict, lat_limit, args.imp_key, nlayers, use_cc=args.use_cc)

torch.save(quantizer_dict, f"{save_dir}/{args.target_thp}thp{'_cc' if args.use_cc else ''}.pt")
torch.save(merge_info, f"{save_dir}/{args.target_thp}thp{'_cc' if args.use_cc else ''}_merge_info.pt")