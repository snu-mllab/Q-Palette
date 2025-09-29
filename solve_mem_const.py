import torch
import os 
import argparse
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
    },
}

from ortools.linear_solver import pywraplp

# solve optimal quantizers for x
def solve_optimal_quantizer(qlist, err_coeff_dict, mem_err_val_dict, mem_limit, imp_key, nlinears, per_layer=False):
    """
        qlist: list of str 
        err_coeff_dict: dict 
        mem_err_val_dict: dict
    """

    # Create the mip solver with the SCIP backend.
    solver = pywraplp.Solver.CreateSolver("SCIP")
    solver.SetNumThreads(12)
    solver.SetTimeLimit(60000)
    if not solver:
        return
    
    # assignment P
    P = []
    for lidx in range(nlinears):
        P.append([])
        for qidx in range(len(qlist)):
            P[lidx].append(solver.IntVar(0.0, 1.0, f"P_{lidx}_{qidx}"))

    print("Number of variables =", solver.NumVariables())

    # 1. assignment constraint
    for lidx in range(nlinears):
        solver.Add(sum(P[lidx]) == 1)
    
    # 2. memory constraint
    if not per_layer:
        expr = 0
        for lidx in range(nlinears):
            for qidx in range(len(qlist)):
                expr += P[lidx][qidx] * mem_err_val_dict[lidx_to_key(lidx)][qlist[qidx]]["mem"]
        solver.Add(expr <= mem_limit)
    else:
        for lidx in range(nlinears // 7):
            expr = 0
            for ridx in range(7):
                for qidx in range(len(qlist)):
                    expr += P[lidx * 7 + ridx][qidx] * mem_err_val_dict[lidx_to_key(lidx * 7 + ridx)][qlist[qidx]]["mem"]
            solver.Add(expr <= mem_limit)
    print("Number of constraints =", solver.NumConstraints())

    # 3. objective
    obj = 0
    for lidx in range(nlinears):
        for qidx in range(len(qlist)):
            obj += P[lidx][qidx] * (float(err_coeff_dict[lidx_to_key(lidx)]) * mem_err_val_dict[lidx_to_key(lidx)][qlist[qidx]][imp_key])
    solver.Minimize(obj)
    status = solver.Solve()

    print("Objective value =", solver.Objective().Value())
    # solution to quantizer_dict
    quantizer_dict = {}
    for lidx in range(nlinears):
        for qidx in range(len(qlist)):
            if P[lidx][qidx].solution_value() > 0.5:
                quantizer_dict[lidx_to_key(lidx)] = qlist[qidx]

    print("\nAdvanced usage:")
    print(f"Problem solved in {solver.wall_time():d} milliseconds")
    print(f"Problem solved in {solver.iterations():d} iterations")
    print(f"Problem solved in {solver.nodes():d} branch-and-bound nodes")

    return quantizer_dict

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

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B")
parser.add_argument("--quantizer_type", type=str, default="default", choices=["default"])
parser.add_argument('--imp_key', type=str, default="err", choices=["err"])
parser.add_argument('--target_bitwidth', type=float, default=3.25)

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

# 2. aggregate random data
err_coeff_dict = err_coeff_routine(model_key)

# 3. solve optimal quantizers
total_mem = 0
qlist = list(QDICT[args.quantizer_type].keys())
for i in range(nlinears):
    key = lidx_to_key(i)
    total_mem += mem_err_val_dict[key][qlist[0]]["orig_mem"]
print("total_mem", total_mem / 1024 / 1024 / 1024, "GB") # total mem for model.model.layers

save_dir = f"msq_results/{model_key}/mem_constrained/{args.quantizer_type}"

os.makedirs(save_dir, exist_ok=True)
from lib.utils.mem_op import calc_avg_bits, QuantConfig

mem_budget = total_mem * ((args.target_bitwidth) / 16)
qlist = list(QDICT[args.quantizer_type].keys()) 
quantizer_dict = solve_optimal_quantizer(qlist, err_coeff_dict, mem_err_val_dict, mem_budget, args.imp_key, nlinears)
avg_bits = calc_avg_bits(QuantConfig(model_key, quantizer_dict, []))
print(f"avg_bits: {round(avg_bits, 3)} / {args.target_bitwidth}bit")

torch.save(quantizer_dict, f"{save_dir}/{args.target_bitwidth}bit.pt")