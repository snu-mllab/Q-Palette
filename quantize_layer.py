import os
from lib.quantizer.tcq_quant import linear_to_tcq_linear
from lib.quantizer.vq_quant import linear_to_vq_linear
from lib.quantizer.comb_quant import linear_to_comb_linear, linear_to_combt_linear
from lib.codebook.bitshift import bitshift_codebook
from lib.codebook.vq_codebook import vq_codebook
from lib.quantizer.vq_quant_ldlq import linear_to_vq_linear_ldlq
from lib.utils import clean
import torch._dynamo
HESSKEY = {
    "self_attn.q_proj": "qkv",
    "self_attn.k_proj": "qkv",
    "self_attn.v_proj": "qkv",
    "self_attn.o_proj": "o",
    "mlp.up_proj": "up",
    "mlp.gate_proj": "up",
    "mlp.down_proj": "down",
}
PARTITION = {
    "self_attn": [
        "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"
    ],
    "mlp": [
        "mlp.up_proj", "mlp.gate_proj", "mlp.down_proj"
    ]
}

def quantize_linear(linear, hess_path, quantizer_str, layer_key, save_path, hadU, hadV, SU=None, SV=None, lnorm=None, rot_info=None, left_only=False):
    if quantizer_str.startswith("tcq"):
        qname, KV, hess_op, scale_override = quantizer_str.split("_")
        KV = int(KV)
        if KV <= 8: tlut_bits = 9
        elif KV == 9: tlut_bits = 10
        elif KV == 10: tlut_bits = 11
        use_hess = hess_op == "hess"
        scale_override = float(scale_override)
        cb = bitshift_codebook(L=16, KV=KV, V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        new_layer, quant_info = linear_to_tcq_linear(linear, hess_path, cb, KV=KV, V=2,scale_override=scale_override, use_hess=use_hess, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
        del cb
        torch._dynamo.reset()
    elif quantizer_str.startswith("comb"):
        qname, KV1, KV2, ratio, hess_op, scale_override = quantizer_str.split("_")
        KV = (int(KV1), int(KV2))
        if max(KV) <= 8: tlut_bits = 9
        elif max(KV) == 9: tlut_bits = 10
        elif max(KV) == 10: tlut_bits = 11
        use_hess = hess_op == "hess"
        scale_override = float(scale_override)
        cb1 = bitshift_codebook(L=16, KV=KV[0], V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        cb2 = bitshift_codebook(L=16, KV=KV[1], V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        ratio = float(ratio)
        out_part = (int(linear.out_features * ratio), int(linear.out_features * (1 - ratio)))
        new_layer, quant_info = linear_to_comb_linear(linear, hess_path, cb1, cb2, out_part=out_part, KV=KV, V=2,scale_override=scale_override, use_hess=use_hess, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
        del cb1, cb2
        torch._dynamo.reset()
    elif quantizer_str.startswith("tcomb"):
        qname, KV1, KV2, ratio, hess_op, scale_override = quantizer_str.split("_")
        KV = (int(KV1), int(KV2))
        if max(KV) <= 8: tlut_bits = 9
        elif max(KV) == 9: tlut_bits = 10
        elif max(KV) == 10: tlut_bits = 11
        use_hess = hess_op == "hess"
        scale_override = float(scale_override)
        cb1 = bitshift_codebook(L=16, KV=KV[0], V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        cb2 = bitshift_codebook(L=16, KV=KV[1], V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        ratio = float(ratio)
        in_part = (int(linear.in_features * ratio), int(linear.in_features * (1 - ratio)))
        new_layer, quant_info = linear_to_combt_linear(linear, hess_path, cb1, cb2, in_part=in_part, KV=KV, V=2,scale_override=scale_override, use_hess=use_hess, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
        del cb1, cb2
        torch._dynamo.reset()
    elif quantizer_str.startswith("ldlq"):
        qname, vec_sz, lut_bits, hess_op, scale_override = quantizer_str.split("_")
        vec_sz = int(vec_sz)
        lut_bits = int(lut_bits)
        use_hess = hess_op == "hess"
        scale_override = float(scale_override)
        cb = vq_codebook(vec_sz=vec_sz, lut_bits=lut_bits)
        new_layer, quant_info = linear_to_vq_linear_ldlq(linear, hess_path, cb=cb, scale_override=scale_override, use_hess=use_hess, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    elif quantizer_str.startswith("sq"):
        qname, lut_bits, hess_op, scale_override = quantizer_str.split("_")
        lut_bits = int(lut_bits)
        use_hess = hess_op == "hess"
        scale_override = float(scale_override)
        new_layer, quant_info = linear_to_vq_linear(linear, hess_path, lut_bits=lut_bits, vec_sz=1, scale_override=scale_override, use_hess=use_hess, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    elif quantizer_str.startswith("vq2"):
        qname, lut_bits, hess_op, scale_override = quantizer_str.split("_")
        lut_bits = int(lut_bits)
        use_hess = hess_op == "hess"
        scale_override = float(scale_override)
        new_layer, quant_info = linear_to_vq_linear(linear, hess_path, lut_bits=lut_bits, vec_sz=2, scale_override=scale_override, use_hess=use_hess, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    else:
        raise ValueError(f"Quantizer {quantizer_str} not supported")
    quant_info["layer_key"] = layer_key
    quant_info["quantizer_str"] = quantizer_str
    quant_info["save_path"] = save_path
    quant_info["rot_info"] = rot_info
    new_layer.save_info(save_path, quant_info)

    del new_layer, quant_info
    clean()

def get_random_sign(length, dtype=torch.float32):
    return (torch.randn(length, dtype=dtype) > 0.0) * 2.0 - 1.0

def quantize(linear, model_key, layer_idx, layer_key, quantizer_str, seed, left_only=True, save_dir="quant_results", hess_dir=None):
    SU_SV = torch.load(f'assets/{model_key}_SU_SV_{seed}.pt')
    SU_qkvs, SV_q, SV_k, SV_v, SU_o, SV_o, SU_upgates, SV_up, SV_gate, SU_dp, SV_dp = SU_SV
    print(f"Loaded SU, SV for {model_key} from assets/{model_key}_SU_SV_{seed}.pt")

    mkey, skey = layer_key.split(".")
    hadU = linear.in_features
    hadV = linear.out_features
    
    lnorm = None

    if layer_key in ["self_attn.o_proj"]:
        SU = SU_o[layer_idx].clone()
    elif layer_key in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
        SU = SU_qkvs[layer_idx].clone()
    elif layer_key in ["mlp.gate_proj", "mlp.up_proj"]:
        SU = SU_upgates[layer_idx].clone()
    elif layer_key in ["mlp.down_proj"]:
        SU = SU_dp[layer_idx].clone()
    else:
        raise ValueError(f"Unknown key: {layer_key}")
    if left_only:
        rot_info = "skip_r"
        SV = torch.ones(hadV).to(SU.device)
    else:
        raise ValueError(f"We don't support non-left-only quant for submission")
    
    SU = SU.cuda()
    SV = SV.cuda()
    full_key = f"model.layers.{str(layer_idx)}.{layer_key}"
    print("="*50)
    print(f"Quantizing {full_key} with {quantizer_str}")
    print("="*50)

    save_path = f"{save_dir}/{model_key}/left_only_seed{seed}_cache/{quantizer_str}/{layer_idx}_{layer_key}.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if os.path.exists(save_path):
        print(f"Skipping {full_key} because it already exists")
        return

    hess_path = f"{hess_dir}/{str(layer_idx)}_{HESSKEY[layer_key]}.pt" if hess_dir is not None else None

    quantize_linear(linear.cuda(), hess_path, quantizer_str, full_key, save_path, SU=SU, SV=SV, lnorm=lnorm, rot_info=rot_info, hadU=hadU, hadV=hadV, left_only=left_only)


def cache_random_signs(model, model_key, seed):
    import random
    torch.set_grad_enabled(False)
    random.seed(seed)
    torch.random.manual_seed(seed)
    num_heads = model.config.num_attention_heads
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    print(f"Generating random SU, SV for {model_key}")
    kv_size = hidden_size // model.config.num_attention_heads * model.config.num_key_value_heads

    SU_qkvs = [get_random_sign(hidden_size) for _ in range(len(model.model.layers))]
    SV_o = [get_random_sign(hidden_size) for _ in range(len(model.model.layers))]
    SU_upgates = [get_random_sign(hidden_size) for _ in range(len(model.model.layers))]
    SV_dp = [get_random_sign(hidden_size) for _ in range(len(model.model.layers))]

    SV_q = [get_random_sign(hidden_size) for _ in range(len(model.model.layers))]
    SV_k = [get_random_sign(kv_size) for _ in range(len(model.model.layers))]

    SV_v = [get_random_sign(kv_size) for _ in range(len(model.model.layers))]
    SU_o = [get_random_sign(hidden_size) for _ in range(len(model.model.layers))]

    SV_up = [get_random_sign(intermediate_size) for _ in range(len(model.model.layers))]
    SV_gate = [get_random_sign(intermediate_size) for _ in range(len(model.model.layers))]

    SU_dp = [get_random_sign(intermediate_size) for _ in range(len(model.model.layers))]

    # save all SU, SV
    total = [SU_qkvs, SV_q, SV_k, SV_v, SU_o, SV_o, SU_upgates, SV_up, SV_gate, SU_dp, SV_dp]
    torch.save(total, f'assets/{model_key}_SU_SV_{seed}.pt')
    print(f"Saved SU, SV for {model_key} in assets/{model_key}_SU_SV_{seed}.pt")

