import torch
import os
import argparse
from lib.utils.mem_op import LAYER_INFO
from lib.config import DEVDIR, MODEL_KEYS

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

def err_coeff_routine_noise(model_key, devset_key, nlinears):
    dir = f"{DEVDIR}/{model_key}/inject_dev_loss_data/{devset_key}/"
    coeff = torch.zeros(nlinears)
    err_coeff = {}
    err_coeff["constant"] = 0
    for i in range(nlinears):
        key = lidx_to_key(i)
        dirname = f"{dir}/{key}/"
        ppls = []
        errs = []
        for file in os.listdir(dirname):
            if file.endswith(".pt"):
                results = torch.load(f"{dirname}/{file}")
                ppls.append(results["ppl"] - results["orig_ppl"])
                errs.append(results["err"])
        assert len(ppls) == 16, f"len(ppls) = {len(ppls)} not 16."
        ppls = torch.tensor(ppls).view(-1)
        errs = torch.tensor(errs).view(-1, 1)
        coeff[i] = torch.linalg.lstsq(errs, ppls)[0].item()
        err_coeff[key] = coeff[i]

        if coeff[i] < 0 :
            print(i, lidx_to_key(i), coeff[i])
            print("orig_ppl", results["orig_ppl"])
            print(ppls)
            print(errs)

    return err_coeff

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--devset_key', type=str, required=True)
    args = parser.parse_args()
    if DEVDIR == "set appropriate path":
        raise ValueError("DEVDIR is not set")
    model_key = MODEL_KEYS[args.model]
    nlinears = LAYER_INFO[model_key]["nlayers"] * 7 # num blocks * num layers per block
    err_coeff = err_coeff_routine_noise(model_key, args.devset_key, nlinears)
    torch.save(err_coeff, f"{DEVDIR}/{model_key}/err_coeff.pt")

if __name__ == "__main__":
    main()