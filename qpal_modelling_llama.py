import torch
import torch.nn as nn
from typing import Any, Dict, Tuple

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as _BaseLlamaForCausalLM,
)
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2ForCausalLM as _BaseQwenForCausalLM,
)

from lib.linear.incoherent_linear import (
    IncoherentLinear,
    IncoherentMLP,
    IncoherentSdpaAttention,
)
from lib.linear.vq_linear import VQLinearPackTensorCore, VQLinearPackSIMT
from lib.linear.tcq_linear import QTIPLinearTCQ
from lib.linear.comb_linear import CombLinearTCQ, CombtLinearTCQ

DTYPE_MAP: Dict[str, torch.dtype] = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
}

MODEL_TYPE_TO_QPAL_CLS: Dict[str, str] = {
    "llama": "QPalLlamaForCausalLM",
    "qwen2": "QPalQwenForCausalLM",
    "qwen2_moe": "QPalQwenForCausalLM",
}


def _dtype_from_str(name: Any) -> torch.dtype:
    if isinstance(name, torch.dtype):
        return name
    key = str(name).replace("torch.", "")
    if key not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {name}")
    return DTYPE_MAP[key]


def _dtype_to_str(dtype: torch.dtype) -> str:
    for key, value in DTYPE_MAP.items():
        if dtype == value:
            return key
    return str(dtype).replace("torch.", "")


def _get_child(module: Any, name: str) -> Any:
    if name.isdigit():
        return module[int(name)]
    return getattr(module, name)


def _resolve_parent(root: nn.Module, path: str) -> Tuple[nn.Module, str]:
    parts = path.split(".")
    module = root
    for name in parts[:-1]:
        module = _get_child(module, name)
    return module, parts[-1]


def _assign_child(module: Any, name: str, value: Any) -> None:
    if name.isdigit():
        module[int(name)] = value
    else:
        setattr(module, name, value)


def _serialize_quant_linear(linear: nn.Module) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "linear_cls": linear.__class__.__name__,
        "in_features": getattr(linear, "in_features", None),
        "out_features": getattr(linear, "out_features", None),
        "bias": getattr(linear, "bias", None) is not None,
    }
    if hasattr(linear, "dtype"):
        meta["linear_dtype"] = _dtype_to_str(getattr(linear, "dtype"))
    if isinstance(linear, VQLinearPackTensorCore):
        meta.update({
            "lut_bits": linear.lut_bits,
            "vec_sz": linear.vec_sz,
        })
    elif isinstance(linear, VQLinearPackSIMT):
        meta.update({
            "lut_bits": linear.lut_bits,
            "vec_sz": linear.vec_sz,
        })
    elif isinstance(linear, QTIPLinearTCQ):
        meta.update({
            "td_x": linear.td_x,
            "td_y": linear.td_y,
            "L": linear.L,
            "KV": linear.KV,
            "V": linear.V,
            "tlut_bits": linear.tlut_bits,
        })
    elif isinstance(linear, CombLinearTCQ):
        meta.update({
            "td_x": linear.td_x,
            "td_y": linear.td_y,
            "out_part": list(linear.out_part),
            "L": linear.L,
            "KV": list(linear.KV),
            "V": linear.V,
            "tlut_bits": linear.tlut_bits,
        })
    elif isinstance(linear, CombtLinearTCQ):
        meta.update({
            "td_x": linear.td_x,
            "td_y": linear.td_y,
            "in_part": list(linear.in_part),
            "L": linear.L,
            "KV": list(linear.KV),
            "V": linear.V,
            "tlut_bits": linear.tlut_bits,
        })
    return meta


def _serialize_incoherent_linear(layer: IncoherentLinear) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "module_type": "IncoherentLinear",
        "in_features": layer.in_features,
        "out_features": layer.out_features,
        "hadU": layer.hadU,
        "hadV": layer.hadV,
        "bias": layer.bias is not None,
        "dtype": _dtype_to_str(layer.dtype),
        "rot_info": layer.rot_info,
        "scale": float(layer.scale),
    }
    if layer.linear is not None:
        meta["linear"] = _serialize_quant_linear(layer.linear)
    return meta


def _serialize_incoherent_mlp(layer: IncoherentMLP) -> Dict[str, Any]:
    projections: Dict[str, Dict[str, Any]] = {}
    if layer.merge_ug and layer.ug_proj is not None:
        projections["ug_proj"] = _serialize_quant_linear(layer.ug_proj)
    else:
        if layer.up_proj is not None:
            projections["up_proj"] = _serialize_quant_linear(layer.up_proj)
        if layer.gate_proj is not None:
            projections["gate_proj"] = _serialize_quant_linear(layer.gate_proj)
    if layer.down_proj is not None:
        projections["down_proj"] = _serialize_quant_linear(layer.down_proj)

    return {
        "module_type": "IncoherentMLP",
        "dtype": _dtype_to_str(layer.dtype),
        "merge_ug": layer.merge_ug,
        "scale": float(layer.scale),
        "projections": projections,
    }


def _serialize_incoherent_attention(layer: IncoherentSdpaAttention) -> Dict[str, Any]:
    projections: Dict[str, Dict[str, Any]] = {}
    for name in [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qk_proj",
        "qv_proj",
        "kv_proj",
        "qkv_proj",
    ]:
        proj = getattr(layer, name, None)
        if proj is not None:
            projections[name] = _serialize_quant_linear(proj)

    return {
        "module_type": "IncoherentSdpaAttention",
        "dtype": _dtype_to_str(layer.dtype),
        "scale": float(layer.scale),
        "merge_qk": layer.merge_qk,
        "merge_kv": layer.merge_kv,
        "merge_qv": layer.merge_qv,
        "merge_qkv": layer.merge_qkv,
        "layer_idx": layer.layer_idx,
        "projections": projections,
    }


def export_qpal_quant_config(model: nn.Module) -> Dict[str, Any]:
    modules: Dict[str, Any] = {}
    for name, module in model.named_modules():
        if isinstance(module, IncoherentLinear):
            modules[name] = _serialize_incoherent_linear(module)
        elif isinstance(module, IncoherentMLP):
            modules[name] = _serialize_incoherent_mlp(module)
        elif isinstance(module, IncoherentSdpaAttention):
            modules[name] = _serialize_incoherent_attention(module)
    return {"modules": modules}


def _build_quant_linear(meta: Dict[str, Any], dtype: torch.dtype, bias: bool) -> nn.Module:
    cls_name = meta["linear_cls"]
    if cls_name == "VQLinearPackTensorCore":
        return VQLinearPackTensorCore(
            meta["in_features"],
            meta["out_features"],
            meta["lut_bits"],
            meta.get("vec_sz", 2),
            bias=bias,
            dtype=_dtype_from_str(meta.get("linear_dtype", dtype)),
        )
    if cls_name == "VQLinearPackSIMT":
        return VQLinearPackSIMT(
            meta["in_features"],
            meta["out_features"],
            meta["lut_bits"],
            meta.get("vec_sz", 1),
            bias=bias,
            dtype=_dtype_from_str(meta.get("linear_dtype", dtype)),
        )
    if cls_name == "QTIPLinearTCQ":
        return QTIPLinearTCQ(
            meta["in_features"],
            meta["out_features"],
            meta["td_x"],
            meta["td_y"],
            meta["L"],
            meta["KV"],
            meta["V"],
            meta["tlut_bits"],
            bias=bias,
            dtype=_dtype_from_str(meta.get("linear_dtype", dtype)),
        )
    if cls_name == "CombLinearTCQ":
        return CombLinearTCQ(
            meta["in_features"],
            meta["out_features"],
            meta["td_x"],
            meta["td_y"],
            meta["out_part"],
            meta["L"],
            meta["KV"],
            meta["V"],
            meta["tlut_bits"],
            bias=bias,
            dtype=_dtype_from_str(meta.get("linear_dtype", dtype)),
        )
    if cls_name == "CombtLinearTCQ":
        return CombtLinearTCQ(
            meta["in_features"],
            meta["out_features"],
            meta["td_x"],
            meta["td_y"],
            meta["in_part"],
            meta["L"],
            meta["KV"],
            meta["V"],
            meta["tlut_bits"],
            bias=bias,
            dtype=_dtype_from_str(meta.get("linear_dtype", dtype)),
        )
    if cls_name == "Linear":
        return nn.Linear(
            meta["in_features"],
            meta["out_features"],
            bias=bias,
            dtype=_dtype_from_str(meta.get("linear_dtype", dtype)),
        )
    raise ValueError(f"Unsupported quantized linear class: {cls_name}")


def _instantiate_incoherent_linear(meta: Dict[str, Any]) -> IncoherentLinear:
    dtype = _dtype_from_str(meta.get("dtype", "float16"))
    bias = bool(meta.get("bias", False))
    hadU = meta.get("hadU", meta["in_features"])
    hadV = meta.get("hadV", meta["out_features"])
    linear_meta = meta.get("linear")
    linear_cls = linear_meta["linear_cls"] if linear_meta else None
    use_linear = linear_cls is None
    layer = IncoherentLinear(
        meta["in_features"],
        meta["out_features"],
        hadU,
        hadV,
        bias=bias,
        dtype=dtype,
        use_linear=use_linear,
    )
    if linear_meta is not None:
        layer.linear = _build_quant_linear(
            linear_meta,
            dtype,
            bias=linear_meta.get("bias", False),
        )
    layer.rot_info = meta.get("rot_info", "all")
    layer.scale = float(meta.get("scale", layer.scale))
    layer.apply_rot_info()
    return layer


def _instantiate_incoherent_mlp(meta: Dict[str, Any], config) -> IncoherentMLP:
    dtype = _dtype_from_str(meta.get("dtype", "float16"))
    mlp = IncoherentMLP(
        config.hidden_size,
        config.intermediate_size,
        config.hidden_act,
        merge_ug=meta.get("merge_ug", False),
        dtype=dtype,
    )
    mlp.scale = float(meta.get("scale", mlp.scale))

    projections = meta.get("projections", {})
    if mlp.merge_ug:
        proj_meta = projections.get("ug_proj")
        if proj_meta is not None:
            mlp.ug_proj = _build_quant_linear(
                proj_meta,
                _dtype_from_str(proj_meta.get("linear_dtype", dtype)),
                bias=proj_meta.get("bias", False),
            )
    else:
        up_meta = projections.get("up_proj")
        gate_meta = projections.get("gate_proj")
        if up_meta is not None:
            mlp.up_proj = _build_quant_linear(
                up_meta,
                _dtype_from_str(up_meta.get("linear_dtype", dtype)),
                bias=up_meta.get("bias", False),
            )
        if gate_meta is not None:
            mlp.gate_proj = _build_quant_linear(
                gate_meta,
                _dtype_from_str(gate_meta.get("linear_dtype", dtype)),
                bias=gate_meta.get("bias", False),
            )

    down_meta = projections.get("down_proj")
    if down_meta is not None:
        mlp.down_proj = _build_quant_linear(
            down_meta,
            _dtype_from_str(down_meta.get("linear_dtype", dtype)),
            bias=down_meta.get("bias", False),
        )

    return mlp


def _instantiate_incoherent_attention(meta: Dict[str, Any], config) -> IncoherentSdpaAttention:
    dtype = _dtype_from_str(meta.get("dtype", "float16"))
    attn = IncoherentSdpaAttention(
        config,
        merge_qk=meta.get("merge_qk", False),
        merge_kv=meta.get("merge_kv", False),
        merge_qv=meta.get("merge_qv", False),
        merge_qkv=meta.get("merge_qkv", False),
        layer_idx=meta.get("layer_idx"),
        dtype=dtype,
    )
    attn.scale = float(meta.get("scale", attn.scale))

    projections = meta.get("projections", {})
    for name, proj_meta in projections.items():
        setattr(
            attn,
            name,
            _build_quant_linear(
                proj_meta,
                _dtype_from_str(proj_meta.get("linear_dtype", dtype)),
                bias=proj_meta.get("bias", False),
            ),
        )

    return attn


def apply_qpal_quantization(model: nn.Module, quant_config: Dict[str, Any]) -> None:
    modules = quant_config.get("modules", {})
    for path, meta in modules.items():
        parent, attr = _resolve_parent(model, path)
        module_type = meta.get("module_type", "IncoherentLinear")
        if module_type == "IncoherentLinear":
            replacement = _instantiate_incoherent_linear(meta)
        elif module_type == "IncoherentMLP":
            replacement = _instantiate_incoherent_mlp(meta, model.config)
        elif module_type == "IncoherentSdpaAttention":
            replacement = _instantiate_incoherent_attention(meta, model.config)
        else:
            raise ValueError(f"Unsupported module_type: {module_type} for {path}")
        _assign_child(parent, attr, replacement)
        _assign_child(parent, attr, replacement)


class QPalLlamaForCausalLM(_BaseLlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        quant_config = getattr(config, "qpal_quant_config", None)
        if quant_config:
            apply_qpal_quantization(self, quant_config)


class QPalQwenForCausalLM(_BaseQwenForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        quant_config = getattr(config, "qpal_quant_config", None)
        if quant_config:
            apply_qpal_quantization(self, quant_config)


__all__ = [
    "QPalLlamaForCausalLM",
    "QPalQwenForCausalLM",
    "export_qpal_quant_config",
    "apply_qpal_quantization",
    "MODEL_TYPE_TO_QPAL_CLS",
]
