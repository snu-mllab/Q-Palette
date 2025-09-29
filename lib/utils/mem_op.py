import torch
LAYER_INFO = {
    "2_7b" : {
        "nlayers" : 32,
        "self_attn.q_proj": {
            "in_features" : 4096,
            "out_features" : 4096,
        },
        "self_attn.k_proj": {
            "in_features" : 4096,
            "out_features" : 4096,
        },
        "self_attn.v_proj": {
            "in_features" : 4096,
            "out_features" : 4096,
        },
        "self_attn.o_proj": {
            "in_features" : 4096,
            "out_features" : 4096,
        },
        "mlp.gate_proj": {
            "in_features" : 4096,
            "out_features" : 11008,
        },
        "mlp.up_proj": {
            "in_features" : 4096,
            "out_features" : 11008,
        },
        "mlp.down_proj": {
            "in_features" : 11008,
            "out_features" : 4096,
        },
    },
    "2_13b": {
        "nlayers" : 40,
        "self_attn.q_proj": {
            "in_features" : 5120,
            "out_features" : 5120,
        },
        "self_attn.k_proj": {
            "in_features" : 5120,
            "out_features" : 5120,
        },
        "self_attn.v_proj": {
            "in_features" : 5120,
            "out_features" : 5120,
        },
        "self_attn.o_proj": {
            "in_features" : 5120,
            "out_features" : 5120,
        },
        "mlp.gate_proj": {
            "in_features" : 5120,
            "out_features" : 13824,
        },
        "mlp.up_proj": {
            "in_features" : 5120,
            "out_features" : 13824,
        },
        "mlp.down_proj": {
            "in_features" : 13824,
            "out_features" : 5120,
        },
    },
    "2_70b": {
        "nlayers" : 80,
        "self_attn.q_proj": {
            "in_features" : 8192,
            "out_features" : 8192,
        },
        "self_attn.k_proj": {
            "in_features" : 8192,
            "out_features" : 1024,
        },
        "self_attn.v_proj": {
            "in_features" : 8192,
            "out_features" : 1024,
        },
        "self_attn.o_proj": {
            "in_features" : 8192,
            "out_features" : 8192,
        },
        "mlp.gate_proj": {
            "in_features" : 8192,
            "out_features" : 28672,
        },
        "mlp.up_proj": {
            "in_features" : 8192,
            "out_features" : 28672,
        },
        "mlp.down_proj": {
            "in_features" : 28672,
            "out_features" : 8192,
        },
    },
    "3_8b": {
        "nlayers": 32,
        "self_attn.q_proj": {
            "in_features" : 4096,
            "out_features" : 4096,
        },
        "self_attn.k_proj": {
            "in_features" : 4096,
            "out_features" : 1024,
        },
        "self_attn.v_proj": {
            "in_features" : 4096,
            "out_features" : 1024,
        },
        "self_attn.o_proj": {
            "in_features" : 4096,
            "out_features" : 4096,
        },
        "mlp.gate_proj": {
            "in_features" : 4096,
            "out_features" : 14336,
        },
        "mlp.up_proj": {
            "in_features" : 4096,
            "out_features" : 14336,
        },
        "mlp.down_proj": {
            "in_features" : 14336,
            "out_features" : 4096,
        },
    },
    "3_3b": {
        "nlayers": 28,
        "self_attn.q_proj": {
            "in_features" : 3072,
            "out_features" : 3072,
        },
        "self_attn.k_proj": {
            "in_features" : 3072,
            "out_features" : 1024,
        },
        "self_attn.v_proj": {
            "in_features" : 3072,
            "out_features" : 1024,
        },
        "self_attn.o_proj": {
            "in_features" : 3072,
            "out_features" : 3072,
        },
        "mlp.gate_proj": {
            "in_features" : 3072,
            "out_features" : 8192,
        },
        "mlp.up_proj": {
            "in_features" : 3072,
            "out_features" : 8192,
        },
        "mlp.down_proj": {
            "in_features" : 8192,
            "out_features" : 3072,
        },
    },
    "3_1b": {
        "nlayers": 16,
        "self_attn.q_proj": {
            "in_features" : 2048,
            "out_features" : 2048,
        },
        "self_attn.k_proj": {
            "in_features" : 2048,
            "out_features" : 512,
        },
        "self_attn.v_proj": {
            "in_features" : 2048,
            "out_features" : 512,
        },
        "self_attn.o_proj": {
            "in_features" : 2048,
            "out_features" : 2048,
        },
        "mlp.gate_proj": {
            "in_features" : 2048,
            "out_features" : 8192,
        },
        "mlp.up_proj": {
            "in_features" : 2048,
            "out_features" : 8192,
        },
        "mlp.down_proj": {
            "in_features" : 8192,
            "out_features" : 2048,
        },
    },
}


def get_layer_info(model_key):
    if model_key == "3_8b_0":
        return LAYER_INFO["3_8b"]
    else:
        return LAYER_INFO[model_key]

def get_dummy_quant_results(model_key, layer_key, quantizer_str):
    import math
    layer_info = get_layer_info(model_key)
    in_features = layer_info[layer_key]["in_features"]
    out_features = layer_info[layer_key]["out_features"]

    quant_info = get_quant_info(quantizer_str)
    info = {
        "quant_info": quant_info,
    }
    if quantizer_str.startswith("ldlq"):
        linear_info = {
            "in_features": in_features,
            "out_features": out_features,
            "lut_bits": quant_info["lut_bits"],
            "dtype": torch.float16,
            "vec_sz": quant_info["vec_sz"],
            "qweight": torch.randint(0, int(2**30), (out_features, quant_info["lut_bits"] * in_features // 32 // quant_info["vec_sz"]), dtype=torch.int32, device="cuda"),
            "lut": torch.randn((2 ** quant_info["lut_bits"], quant_info["vec_sz"]), dtype=torch.float16, device="cuda"),
            "bias": None,
        }
    elif quantizer_str.startswith("tcq"):
        linear_info = {
            "in_features": in_features,
            "out_features": out_features,
            "td_x": 16,
            "td_y": 16,
            "L": 16,
            "KV": quant_info["KV"],
            "V": quant_info["V"],
            "tlut_bits": quant_info["tlut_bits"],
            "dtype": torch.float16,
            "trellis": torch.randint(0, int(2**14), ((out_features // 16) * (in_features // 16), math.ceil((16 * 16) * quant_info["KV"] / 16 / quant_info["V"])), dtype=torch.int16, device="cuda"),
            "tlut": torch.randn((2 ** quant_info["tlut_bits"], quant_info["V"]), dtype=torch.float16, device="cuda"),
            "bias": None,
        }
    elif quantizer_str.startswith("tcomb"):
        assert quant_info["ratio"] == 0.5, "only support ratio = 0.5 for now"
        in_part = (in_features // 2, in_features // 2)
        linear_info = {
            "in_features": in_features,
            "out_features": out_features,
            "td_x": 16,
            "td_y": 16,
            "in_part": in_part,
            "L": 16,
            "KV": quant_info["KV"],
            "V": quant_info["V"],
            "tlut_bits": quant_info["tlut_bits"],
            "dtype": torch.float16,
            "trellis1": torch.randint(0, int(2**14), ((out_features // 16) * (in_part[0] // 16), math.ceil((16 * 16) * quant_info["KV"][0] / 16 / quant_info["V"])), dtype=torch.int16, device="cuda"),
            "trellis2": torch.randint(0, int(2**14), ((out_features // 16) * (in_part[1] // 16), math.ceil((16 * 16) * quant_info["KV"][1] / 16 / quant_info["V"])), dtype=torch.int16, device="cuda"),
            "tlut": torch.randn((2 ** quant_info["tlut_bits"], quant_info["V"]), dtype=torch.float16, device="cuda"),
            "bias": None,
        }
    elif quantizer_str == "default":
        info["in_features"] = in_features
        info["out_features"] = out_features
        linear_info = {
            "in_features": in_features,
            "out_features": out_features,
            "dtype": torch.float16,
            "bias": None,
        }
    else:
        raise ValueError(f"Unknown quantizer: {quantizer_str}")
    info["linear_info"] = linear_info
    info["in_features"] = in_features
    info["out_features"] = out_features
    info["dtype"] = torch.float16
    info["bias"] = None
    return info

def get_quant_info(quantizer_str):
    if quantizer_str.startswith("tcq"):
        qname, kv, hess, scale = quantizer_str.split("_")
        tlut_bits = 9 if int(kv) <= 8 else int(int(kv) + 1)
        quant_info = {
            "quantizer_str": quantizer_str,
            "quantizer": "tcq_ldlq",
            "KV": int(kv),
            "V": 2,
            "tlut_bits": tlut_bits,
        }
    elif quantizer_str.startswith("tcomb"):
        qname, kv1, kv2, ratio, hess, scale = quantizer_str.split("_")
        tlut_bits = 9 if int(kv2) <= 8 else int(int(kv2) + 1)
        quant_info = {
            "quantizer_str": quantizer_str,
            "quantizer": "combt_ldlq",
            "KV": [int(kv1), int(kv2)],
            "V": 2,
            "tlut_bits": tlut_bits,
            "ratio": float(ratio),
        }
    elif quantizer_str.startswith("ldlq"):
        qname, vec_sz, lut_bits, hess, scale = quantizer_str.split("_")
        quant_info = {
            "quantizer_str": quantizer_str,
            "quantizer": "vq_ldlq",
            "vec_sz": int(vec_sz),
            "lut_bits": int(lut_bits),
        }
    elif quantizer_str == "default":
        quant_info = {
            "quantizer_str": quantizer_str,
        }
    else:
        raise ValueError(f"Unknown quantizer: {quantizer_str}")
    return quant_info

def get_layer_mem(model_key, layer_key, quantizer_str="default"):
    layer_info = get_layer_info(model_key)
    in_features = layer_info[layer_key]["in_features"]
    out_features = layer_info[layer_key]["out_features"]

    quant_info = get_quant_info(quantizer_str)
    if quantizer_str == "default":
        return in_features * out_features * 16 / 8
    else:
        if quant_info["quantizer"] == "vq_ldlq":
            mem = in_features * out_features * quant_info["lut_bits"] / quant_info["vec_sz"] / 8 + (2 ** quant_info["lut_bits"]) * quant_info["vec_sz"] * 16 / 8 
        elif quant_info["quantizer"] == "tcq_ldlq": 
            mem = in_features * out_features * quant_info["KV"] / quant_info["V"] / 8 + (2 ** quant_info["tlut_bits"]) * 2 * 16 / 8
        elif quant_info["quantizer"] == "combt_ldlq":
            mem = in_features * out_features * (quant_info["KV"][0] + quant_info["KV"][1]) / 2 / quant_info["V"] / 8 + (2 ** quant_info["tlut_bits"]) * quant_info["V"] * 16 / 8
        else:
            raise ValueError(f"Unknown quantizer: {quant_info['quantizer']}")
    return mem

def get_constant_mem(model_key):
    layer_info = get_layer_info(model_key)
    constant_mem = 0
    for key in ["self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]:
        in_features = layer_info[key]["in_features"]
        constant_mem += in_features / 8 # boolean SU, SV
    return constant_mem

def cache_quantizer_err(quantizer_str):
    assert "hess" not in quantizer_str, "we only compute err for data-free quantizers"
    Wr = torch.randn(4096, 4096).cuda()
    Wscale = torch.ones(4096).cuda()
    HRr = torch.eye(4096).cuda().double().unsqueeze(-1)
    if quantizer_str.startswith("tcq"):
        from lib.quantizer.tcq_quant import qtip_quantize_mat
        from lib.codebook.bitshift import bitshift_codebook
        qname, kv, hess, scale = quantizer_str.split("_")
        scale = float(scale)
        tlut_bits = 9 if int(kv) <= 8 else int(int(kv) + 1)
        cb = bitshift_codebook(L=16, KV=int(kv), V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        packed, hatWr, quant_info = qtip_quantize_mat(Wr * scale, HRr, Wscale / scale, cb, td_x=16, td_y=16, KV=int(kv), V=2, use_hess=False)
        
    elif quantizer_str.startswith("tcomb"):
        from lib.quantizer.comb_quant import combt_quantize_mat
        from lib.codebook.bitshift import bitshift_codebook
        qname, kv1, kv2, ratio, hess, scale = quantizer_str.split("_")
        scale = float(scale)
        tlut_bits = 9 if int(kv2) <= 8 else int(int(kv2) + 1)
        cb1 = bitshift_codebook(L=16, KV=int(kv1), V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        cb2 = bitshift_codebook(L=16, KV=int(kv2), V=2, tlut_bits=tlut_bits, decode_mode='quantlut_sym')
        packed, packed2, hatWr, quant_info = combt_quantize_mat(Wr * scale, HRr, Wscale / scale, cb1, cb2, td_x=16, td_y=16, KV=(int(kv1), int(kv2)), V=2, use_hess=False)
    
    elif quantizer_str.startswith("ldlq"):
        from lib.quantizer.vq_quant_ldlq import vq_quantize_mat_ldlq
        from lib.codebook.vq_codebook import vq_codebook
        qname, vec_sz, lut_bits, hess, scale = quantizer_str.split("_")
        scale = float(scale)
        cb = vq_codebook(vec_sz=int(vec_sz), lut_bits=int(lut_bits))
        packed, hatWr, quant_info = vq_quantize_mat_ldlq(Wr * scale, HRr, Wscale / scale, cb, use_hess=False)

    else:
        raise ValueError(f"Unknown quantizer: {quantizer_str}")
    
    return quant_info["err"]

def cache_quant_errors():
    err_dict = {}
    for quantizer_str in [
        "tcq_2_none_0.9",
        "tcq_3_none_0.9",
        "tcq_4_none_0.9",
        "tcq_5_none_0.9",
        "tcq_6_none_0.9",
        "tcq_7_none_0.9",
        "tcq_8_none_0.9",
        "tcq_9_none_0.9",
        "tcq_10_none_0.9",
        "tcomb_2_3_0.5_none_0.9",
        "tcomb_3_4_0.5_none_0.9",
        "tcomb_4_5_0.5_none_0.9",
        "tcomb_5_6_0.5_none_0.9",
        "tcomb_6_7_0.5_none_0.9",
        "tcomb_7_8_0.5_none_0.9",
        "tcomb_8_9_0.5_none_0.9",
        "tcomb_9_10_0.5_none_0.9",
        "ldlq_1_2_none_1.0",
        "ldlq_1_3_none_1.0",
        "ldlq_1_4_none_1.0",
        "ldlq_1_5_none_1.0",
        "ldlq_1_6_none_1.0",
        "ldlq_1_7_none_1.0",
        "ldlq_1_8_none_1.0",
        "ldlq_2_3_none_1.0",
        "ldlq_2_4_none_1.0",
        "ldlq_2_5_none_1.0",
        "ldlq_2_6_none_1.0",
        "ldlq_2_7_none_1.0",
        "ldlq_2_8_none_1.0",
        "ldlq_2_9_none_1.0",
        "ldlq_2_10_none_1.0",
        "ldlq_2_11_none_1.0",
        "ldlq_2_12_none_1.0",
        # "ldlq_4_6_none_1.0",
        # "ldlq_4_7_none_1.0",
        # "ldlq_4_8_none_1.0",
        # "ldlq_4_9_none_1.0",
        # "ldlq_4_10_none_1.0",
        # "ldlq_4_11_none_1.0",
        # "ldlq_4_12_none_1.0",
    ]:
        err = cache_quantizer_err(quantizer_str)
        err_dict[quantizer_str] = err
        print(quantizer_str, err)

        torch._dynamo.reset()
    for key, err in err_dict.items():
        print(f"{key}: {err}")

    torch.save(err_dict, "./eval_results/coeffs/quant_err.pt")

def cache_uniform_error(nbits):
    Wr = torch.randn(4096, 4096).cuda()
    Wr_max = Wr.max(-1).values # max 
    Wr_min = Wr.min(-1).values # min 
    Q = (Wr - Wr_min) / (Wr_max - Wr_min) * (2 ** (nbits) - 1)
    Q = Q.round().clamp(0, 2 ** (nbits) - 1)
    What = Q / (2 ** (nbits) - 1) * (Wr_max - Wr_min) + Wr_min
    return (What - Wr).pow(2).mean()

class QuantConfig:
    def __init__(self, model_key, quant_dict, merge_infos):
        self.model_key = model_key
        self.quant_dict = quant_dict
        self.merge_infos = merge_infos
        self.layer_infos = get_layer_info(model_key)

        self.nlayers = self.layer_infos["nlayers"]

def calc_avg_bits(quant_config):
    total_mem = 0
    default_mem = 0
    for i in range(quant_config.nlayers):
        for layer_key in [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]:
            if isinstance(quant_config.quant_dict[f"{i}_{layer_key}"], tuple):
                quantizer_str, _ = quant_config.quant_dict[f"{i}_{layer_key}"]
            else:
                quantizer_str = quant_config.quant_dict[f"{i}_{layer_key}"]
            total_mem += get_layer_mem(quant_config.model_key, layer_key, quantizer_str)
            default_mem += get_layer_mem(quant_config.model_key, layer_key, "default")

            if layer_key in ["self_attn.q_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"]:
                total_mem += quant_config.layer_infos[layer_key]['in_features'] / 8 # SU
    avg_bits = total_mem / default_mem * 16
    return avg_bits


if __name__ == "__main__":
    for nbits in [2,3,4,5,6]:
        err = cache_uniform_error(nbits)
        print(f"{nbits}, {err}")

    # cache_quant_errors()
    # for model_key in ["2_7b"]:
    # # for model_key in ["3_8b", "3_3b", "3_1b"]:
    #     for quantizer_str in ["tcomb_6_7_0.5_none_0.9", "tcq_8_none_0.9", "tcomb_8_9_0.5_none_0.9"]:
    #         quant_dict = {}
    #         for i in range(32):
    #             for key in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
    #                 quant_dict[f"{i}_{key}"] = quantizer_str
    #         quant_config = QuantConfig(model_key, quant_dict, [])
    #         print(model_key, quantizer_str, calc_avg_bits(quant_config))
