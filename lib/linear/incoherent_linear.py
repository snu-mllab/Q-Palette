import torch
import torch.nn as nn

from lib.utils import (get_hadK, matmul_hadU_cuda, matmul_hadUt_cuda, matmul_hadUt, matmul_hadU, matmul_hadUt_head, matmul_hadU_head, matmul_hadU_head_cuda, matmul_hadUt_head_cuda)
from lib.linear.tcq_linear import QTIPLinearTCQ
from lib.linear.vq_linear import VQLinearPackTensorCore, VQLinearPackSIMT
from lib.linear.comb_linear import CombLinearTCQ, CombtLinearTCQ
from transformers.activations import ACT2FN
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import Optional, Tuple
from model.llama import LlamaRotaryEmbedding, repeat_kv, apply_rotary_pos_emb, Cache

def make_linear(info, use_simt=False):
    if "tcq" in info["quant_info"]["quantizer_str"]:
        linear = QTIPLinearTCQ.gen_layer_from_info(info["linear_info"])
    elif use_simt and ("sq" in info["quant_info"]["quantizer_str"] or "vq" in info["quant_info"]["quantizer_str"] or "ldlq" in info["quant_info"]["quantizer_str"]):
        linear = VQLinearPackSIMT.gen_layer_from_info(info["linear_info"])
    elif "sq" in info["quant_info"]["quantizer_str"] or "vq" in info["quant_info"]["quantizer_str"] or "ldlq" in info["quant_info"]["quantizer_str"]:
        linear = VQLinearPackTensorCore.gen_layer_from_info(info["linear_info"])
    elif "tcomb" in info["quant_info"]["quantizer_str"]:
        linear = CombtLinearTCQ.gen_layer_from_info(info["linear_info"])
    elif "comb" in info["quant_info"]["quantizer_str"]:
        linear = CombLinearTCQ.gen_layer_from_info(info["linear_info"])
    else:
        linear = nn.Linear(info["in_features"], info["out_features"], bias=False)
    return linear

class IncoherentSdpaAttention(nn.Module):
    def __init__(self, config, merge_qk=False, merge_kv=False, merge_qv=False, merge_qkv=False, layer_idx=None, dtype=torch.float16):
        super().__init__()
        self.config = config
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim",
                                self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.kv_out = self.hidden_size * self.num_key_value_heads // self.num_heads
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.o_proj = None
        self.qk_proj = None
        self.qkv_proj = None
        self.kv_proj = None
        self.dtype=dtype
        self.layer_idx = layer_idx
        self.register_buffer("SU_qkv", torch.ones(config.hidden_size, dtype=self.dtype))
        self.register_buffer("SU_o", torch.ones(config.hidden_size, dtype=self.dtype))

        hidden_had, hidden_K = get_hadK(config.hidden_size)

        hidden_had_T = hidden_had.T.contiguous().cuda() if hidden_had is not None else None

        self.register_buffer('Wscale_qkv', torch.ones(config.hidden_size + 2 * self.kv_out, dtype=self.dtype), persistent=False)
        self.register_buffer('Wscale_o', torch.ones(config.hidden_size, dtype=self.dtype), persistent=False)
        self.register_buffer('had_left_qkv_T', hidden_had_T, persistent=False)
        self.register_buffer('had_left_o_T', hidden_had_T, persistent=False)

        self.hidden_K = hidden_K
        self.scale = 64.0
        self.rotary_emb = LlamaRotaryEmbedding(config=self.config)

        self.merge_qk = merge_qk
        self.merge_kv = merge_kv
        self.merge_qv = merge_qv
        self.merge_qkv = merge_qkv

        assert sum([self.merge_qk, self.merge_kv, self.merge_qv, self.merge_qkv]) <= 1, "Only one of merge_qk, merge_kv, merge_qv, merge_qkv can be True"

    def compute_qkv(self, input):

        n = len(self.SU_qkv)
        x = input.view(-1, n).half()
        
        x = matmul_hadU_cuda(x * self.SU_qkv, self.had_left_qkv_T, self.hidden_K) / self.scale
        if self.merge_qkv:
            qkv = self.qkv_proj(x.half()) * self.Wscale_qkv * self.scale
            q, k, v = qkv.split([self.hidden_size, self.kv_out, self.kv_out], dim=-1)
        elif self.merge_qk:
            qk = self.qk_proj(x.half()) * self.Wscale_qkv[:self.hidden_size + self.kv_out] * self.scale
            q, k = qk.split([self.hidden_size, self.kv_out], dim=-1)
            v = self.v_proj(x.half()) * self.Wscale_qkv[self.hidden_size + self.kv_out:] * self.scale
        elif self.merge_kv:
            kv = self.kv_proj(x.half()) * self.Wscale_qkv[self.hidden_size:] * self.scale
            k, v = kv.split([self.kv_out, self.kv_out], dim=-1)
            q = self.q_proj(x.half()) * self.Wscale_qkv[:self.hidden_size] * self.scale
        elif self.merge_qv:
            qv = self.qv_proj(x.half()) * self.Wscale_qkv[:self.hidden_size + self.kv_out] * self.scale
            q, v = qv.split([self.hidden_size, self.kv_out], dim=-1)
            k = self.k_proj(x.half()) * self.Wscale_qkv[self.hidden_size + self.kv_out:] * self.scale
        else:
            q = self.q_proj(x.half()) * self.Wscale_qkv[:self.hidden_size] * self.scale
            k = self.k_proj(x.half()) * self.Wscale_qkv[self.hidden_size:self.hidden_size + self.kv_out] * self.scale
            v = self.v_proj(x.half()) * self.Wscale_qkv[self.hidden_size + self.kv_out:] * self.scale
        return q.view(*input.shape[:-1], n), k.view(*input.shape[:-1], self.kv_out), v.view(*input.shape[:-1], self.kv_out)
    
    def compute_o(self, input):
        n = len(self.SU_o)
        x = input.view(-1, n).half()
        x = matmul_hadU_cuda(x * self.SU_o, self.had_left_o_T, self.hidden_K) / self.scale
        x = self.o_proj(x.half()) * self.Wscale_o * self.scale
        return x.view(*input.shape[:-1], n)
            
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor,
                  torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor],
               Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        # query_states = self.q_proj(hidden_states)
        # key_states = self.k_proj(hidden_states)
        # value_states = self.v_proj(hidden_states)
        query_states, key_states, value_states = self.compute_qkv(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads,
                                         self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                         self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin)

        
        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position
            }
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states.to(query_states.device),
            value_states.to(query_states.device),
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, -1)

        # attn_output = self.o_proj(attn_output)
        attn_output = self.compute_o(attn_output)
        return attn_output, None, past_key_value

    @staticmethod
    def gen_layer_from_info(config, layer_idx, info_q, info_k, info_v, info_o, merge_qk=False, merge_qv=False, merge_kv=False, merge_qkv=False, dummy=False, use_simt=False, use_simt_q=None, use_simt_k=None, use_simt_v=None, use_simt_o=None):
        attn = IncoherentSdpaAttention(config, merge_qk=merge_qk, merge_qv=merge_qv, merge_kv=merge_kv, merge_qkv=merge_qkv, layer_idx=layer_idx)
        if not dummy:
            attn.SU_qkv.data.copy_(info_q["SU"])
            attn.SU_o.data.copy_(info_o["SU"])
            if not merge_qv:
                attn.Wscale_qkv.data.copy_(torch.cat([info_q["Wscale"], info_k["Wscale"], info_v["Wscale"]], dim=-1))
            else:
                attn.Wscale_qkv.data.copy_(torch.cat([info_q["Wscale"], info_v["Wscale"], info_k["Wscale"]], dim=-1))
            attn.Wscale_o.data.copy_(info_o["Wscale"])
        
        use_simt_q = use_simt if use_simt_q is None else use_simt_q
        use_simt_k = use_simt if use_simt_k is None else use_simt_k
        use_simt_v = use_simt if use_simt_v is None else use_simt_v
        use_simt_o = use_simt if use_simt_o is None else use_simt_o

        if merge_qk:
            to_merged, rest, target_proj, rest_proj = [info_q, info_k], [info_v], "qk_proj", ["v_proj"]
        elif merge_kv:
            to_merged, rest, target_proj, rest_proj = [info_k, info_v], [info_q], "kv_proj", ["q_proj"]
        elif merge_qv:
            to_merged, rest, target_proj, rest_proj = [info_q, info_v], [info_k], "qv_proj", ["k_proj"]
        elif merge_qkv:
            to_merged, rest, target_proj, rest_proj = [info_q, info_k, info_v], [], "qkv_proj", []
        else: 
            to_merged, rest, target_proj, rest_proj = [], [info_q, info_k, info_v], "", ["q_proj", "k_proj", "v_proj"]

        if merge_qk or merge_kv or merge_qv or merge_qkv:
            if merge_kv: use_simt_merge = use_simt_k
            elif merge_qk or merge_qv or merge_qkv: use_simt_merge = use_simt_q
            else: raise ValueError

            if "tcq" in to_merged[0]["quant_info"]["quantizer_str"]:
                merged_linear = QTIPLinearTCQ
            elif "sq" in to_merged[0]["quant_info"]["quantizer_str"] or "vq" in to_merged[0]["quant_info"]["quantizer_str"] or "ldlq" in to_merged[0]["quant_info"]["quantizer_str"]:
                merged_linear = VQLinearPackTensorCore if not use_simt_merge else VQLinearPackSIMT
            elif "tcomb" in to_merged[0]["quant_info"]["quantizer_str"]:
                merged_linear = CombtLinearTCQ
            elif "comb" in to_merged[0]["quant_info"]["quantizer_str"]:
                merged_linear = CombLinearTCQ
            merged = to_merged[0]['linear_info']
            for info in to_merged[1:]:
                merged = merged_linear.merge_infos(merged, info['linear_info'])
            setattr(attn, target_proj, merged_linear.gen_layer_from_info(merged))
        for info, proj in zip(rest, rest_proj):
            if proj == "q_proj": cur_use_simt = use_simt_q
            elif proj == "k_proj": cur_use_simt = use_simt_k
            elif proj == "v_proj": cur_use_simt = use_simt_v
            else: raise ValueError
            setattr(attn, proj, make_linear(info, use_simt=cur_use_simt))
        attn.o_proj = make_linear(info_o, use_simt=use_simt_o)
        return attn

    @staticmethod
    def gen_layer_from_quantizer_str_and_key(config, layer_idx, quant_dir, quantizer_str_q, quantizer_str_k, quantizer_str_v, quantizer_str_o, key_q, key_k, key_v, key_o, merge_qk=False, merge_qv=False, merge_kv=False, merge_qkv=False, dummy=False, use_simt=False, use_simt_q=None, use_simt_k=None, use_simt_v=None, use_simt_o=None):
        if not dummy:
            info_q = torch.load(f"{quant_dir}/{quantizer_str_q}/{key_q}.pt")
            info_k = torch.load(f"{quant_dir}/{quantizer_str_k}/{key_k}.pt")
            info_v = torch.load(f"{quant_dir}/{quantizer_str_v}/{key_v}.pt")
            info_o = torch.load(f"{quant_dir}/{quantizer_str_o}/{key_o}.pt")
        else:
            from lib.utils.mem_op import get_dummy_quant_results
            from lib.config import MODEL_KEYS
            model_key = MODEL_KEYS[config._name_or_path]
            info_q = get_dummy_quant_results(model_key, f"self_attn.q_proj", quantizer_str_q)
            info_k = get_dummy_quant_results(model_key, f"self_attn.k_proj", quantizer_str_k)
            info_v = get_dummy_quant_results(model_key, f"self_attn.v_proj", quantizer_str_v)
            info_o = get_dummy_quant_results(model_key, f"self_attn.o_proj", quantizer_str_o)

        return IncoherentSdpaAttention.gen_layer_from_info(config, layer_idx, info_q, info_k, info_v, info_o, merge_qk=merge_qk, merge_qv=merge_qv, merge_kv=merge_kv, merge_qkv=merge_qkv, dummy=dummy, use_simt=use_simt, use_simt_q=use_simt_q, use_simt_k=use_simt_k, use_simt_v=use_simt_v, use_simt_o=use_simt_o)




class IncoherentMLP(nn.Module):
    """
        only support left only and unified SU for upgates.
    """
    def __init__(self, hidden_size, intermediate_size, hidden_act, merge_ug=False, bias=False, dtype=torch.float16):
        super().__init__()
        assert bias is False, "bias is not supported"
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dtype = dtype

        self.up_proj = None
        self.gate_proj = None
        self.ug_proj = None
        self.down_proj = None

        self.register_buffer("SU_ug", torch.ones(hidden_size, dtype=self.dtype))
        self.register_buffer("SU_dp", torch.ones(intermediate_size, dtype=self.dtype))

        hidden_had, hidden_K = get_hadK(hidden_size)
        inter_had, inter_K = get_hadK(intermediate_size)

        inter_had_T = inter_had.T.contiguous().cuda() if inter_had is not None else None
        hidden_had_T = hidden_had.T.contiguous().cuda() if hidden_had is not None else None

        self.register_buffer('Wscale_ug', torch.ones(intermediate_size * 2, dtype=self.dtype), persistent=False)
        self.register_buffer('Wscale_dp', torch.ones(hidden_size, dtype=self.dtype), persistent=False)
        self.register_buffer('had_left_ug_T', hidden_had_T, persistent=False)
        self.register_buffer('had_left_dp_T', inter_had_T, persistent=False)

        self.hidden_K = hidden_K
        self.inter_K = inter_K

        self.scale = 64.0

        self.act_fn = ACT2FN[hidden_act]
        self.merge_ug = merge_ug

    def forward(self, input):
        n = len(self.SU_ug)
        x = input.view(-1, n).half()
        x = self.compute_ug(x)
        x = self.compute_dp(x)
        return x.view(*input.shape[:-1], n).to(input.dtype)
    
    def compute_ug(self, x):
        x = matmul_hadU_cuda(x * self.SU_ug, self.had_left_ug_T, self.hidden_K) / self.scale
        if self.merge_ug:
            x = self.ug_proj(x.half()) * self.Wscale_ug * self.scale
            x_up, x_gate = x.split(self.intermediate_size, dim=-1)
        else:
            x_up = self.up_proj(x.half()) * self.Wscale_ug[:self.intermediate_size] * self.scale
            x_gate = self.gate_proj(x.half()) * self.Wscale_ug[self.intermediate_size:] * self.scale
        x = self.act_fn(x_gate) * x_up
        return x
    
    def compute_dp(self, x):
        x = matmul_hadU_cuda(x * self.SU_dp, self.had_left_dp_T, self.inter_K) / self.scale
        x = self.down_proj(x.half()) * self.Wscale_dp * self.scale
        return x

    @staticmethod
    def gen_layer_from_info(config, info_up, info_gate, info_down, merge_ug=False, dummy=False, use_simt=False, use_simt_u=None, use_simt_g=None, use_simt_d=None):
        mlp = IncoherentMLP(
                    hidden_size=config.hidden_size, 
                    intermediate_size=config.intermediate_size, 
                    hidden_act=config.hidden_act,
                    merge_ug=merge_ug
                )
        if not dummy:
            mlp.SU_ug.data.copy_(info_up["SU"])
            mlp.SU_dp.data.copy_(info_down["SU"])
            mlp.Wscale_ug.data.copy_(torch.cat([info_up["Wscale"], info_gate["Wscale"]], dim=-1))
            mlp.Wscale_dp.data.copy_(info_down["Wscale"])

        use_simt_u = use_simt if use_simt_u is None else use_simt_u
        use_simt_g = use_simt if use_simt_g is None else use_simt_g
        use_simt_d = use_simt if use_simt_d is None else use_simt_d

        if merge_ug:
            if "tcq" in info_up["quant_info"]["quantizer_str"]:
                linear_info_ug = QTIPLinearTCQ.merge_infos(info_up['linear_info'], info_gate['linear_info'])
                mlp.ug_proj = QTIPLinearTCQ.gen_layer_from_info(linear_info_ug)
            elif "vq" in info_up["quant_info"]["quantizer_str"] or "sq" in info_up["quant_info"]["quantizer_str"] or "ldlq" in info_up["quant_info"]["quantizer_str"]:
                if use_simt_u:
                    linear_info_ug = VQLinearPackSIMT.merge_infos(info_up['linear_info'], info_gate['linear_info'])
                    mlp.ug_proj = VQLinearPackSIMT.gen_layer_from_info(linear_info_ug)
                else:
                    linear_info_ug = VQLinearPackTensorCore.merge_infos(info_up['linear_info'], info_gate['linear_info'])
                    mlp.ug_proj = VQLinearPackTensorCore.gen_layer_from_info(linear_info_ug)
            elif "tcomb" in info_up["quant_info"]["quantizer_str"]:
                linear_info_ug = CombtLinearTCQ.merge_infos(info_up['linear_info'], info_gate['linear_info'])
                mlp.ug_proj = CombtLinearTCQ.gen_layer_from_info(linear_info_ug)
            elif "comb" in info_up["quant_info"]["quantizer_str"]:
                linear_info_ug = CombLinearTCQ.merge_infos(info_up['linear_info'], info_gate['linear_info'])
                mlp.ug_proj = CombLinearTCQ.gen_layer_from_info(linear_info_ug)
        else:
            mlp.up_proj = make_linear(info_up, use_simt=use_simt_u)
            mlp.gate_proj = make_linear(info_gate, use_simt=use_simt_g)
        mlp.down_proj = make_linear(info_down, use_simt=use_simt_d)
        return mlp
    
    @staticmethod
    def gen_layer_from_quantizer_str_and_key(config, quant_dir, quantizer_str_up, quantizer_str_gate, quantizer_str_down, key_up, key_gate, key_down, merge_ug=False, dummy=False, use_simt=False, use_simt_u=None, use_simt_g=None, use_simt_d=None):
        if not dummy:
            info_up = torch.load(f"{quant_dir}/{quantizer_str_up}/{key_up}.pt")
            info_gate = torch.load(f"{quant_dir}/{quantizer_str_gate}/{key_gate}.pt") 
            info_down = torch.load(f"{quant_dir}/{quantizer_str_down}/{key_down}.pt")
        else:
            from lib.utils.mem_op import get_dummy_quant_results
            from lib.config import MODEL_KEYS
            model_key = MODEL_KEYS[config._name_or_path]
            info_up = get_dummy_quant_results(model_key, f"mlp.up_proj", quantizer_str_up)
            info_gate = get_dummy_quant_results(model_key, f"mlp.gate_proj", quantizer_str_gate)
            info_down = get_dummy_quant_results(model_key, f"mlp.down_proj", quantizer_str_down)
        return IncoherentMLP.gen_layer_from_info(config, info_up, info_gate, info_down, merge_ug, dummy=dummy, use_simt=use_simt, use_simt_u=use_simt_u, use_simt_g=use_simt_g, use_simt_d=use_simt_d)



class IncoherentLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        hadU,
        hadV,
        bias=False,
        dtype=torch.float16,
        use_linear=True,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dtype = dtype

        if use_linear:
            self.linear = nn.Linear(in_features, out_features, bias=False, dtype=dtype)
        else:
            self.linear = None

        if bias:
            self.register_buffer('bias', torch.ones(out_features))
        else:
            self.bias = None

        self.register_buffer("SU", torch.ones(in_features, dtype=self.dtype))
        self.register_buffer("SV", torch.ones(out_features, dtype=self.dtype))

        self.hadU = hadU
        self.hadV = hadV
        had_left, K_left = get_hadK(hadU)
        had_right, K_right = get_hadK(hadV)
        if had_left is not None:
            had_left_T = had_left.T.contiguous().cuda()
        else:
            had_left_T = None
        if had_right is not None:
            had_right = had_right.cuda()
        self.register_buffer('Wscale', torch.ones(out_features, dtype=self.dtype), persistent=False)
        self.register_buffer('had_right', had_right, persistent=False)
        self.register_buffer('had_left_T', had_left_T, persistent=False)
        self.K_left = K_left
        self.K_right = K_right
        self.scale = 32.0

        self.rot_info = "all"

        self.skip_l = False
        self.skip_r = False

    def apply_rot_info(self):
        if self.rot_info == "all":
            self.skip_l = False
            self.skip_r = False
        elif self.rot_info == "skip_l":
            self.skip_l = True
            self.skip_r = False
        elif self.rot_info == "skip_r":
            self.skip_l = False
            self.skip_r = True
        elif self.rot_info == "skip_lr":
            self.skip_l = True
            self.skip_r = True
        else:
            raise ValueError(f"Invalid rot_info: {self.rot_info}")
            

    def save_info(self, path, quant_info=None):
        linear_info = self.linear._info()
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "hadU": self.hadU,
            "hadV": self.hadV,
            "dtype": self.dtype,
            "scale": self.scale,
            "Wscale": self.Wscale.detach().cpu(),
            "rot_info": self.rot_info,
            "linear_info": linear_info,
            "bias": self.bias.detach().cpu() if self.bias is not None else None,
            "SU": self.SU.detach().cpu(),
            "SV": self.SV.detach().cpu(),
            "quant_info": quant_info,
        }
        torch.save(info, path)

    def forward(self, input):
        n, m = len(self.SU), len(self.SV)
        x = input.view(-1, n).half()#.to(torch.float32)
        if not self.skip_l:
            x = x * self.SU
            x = matmul_hadU_head_cuda(x, self.had_left_T, self.K_left, self.hadU) / self.scale
        else: 
            # x = x * self.SU
            x = x / self.scale
        x = self.linear(x.half()) * self.Wscale#.float() 
        if not self.skip_r:
            x = matmul_hadU_head_cuda(x, self.had_right, self.K_right, self.hadV) 
            x = x.to(self.SV.device) * (self.SV * self.scale)
        else:
            # x = x.to(self.SV.device) * (self.SV * self.scale)
            x = x * self.scale

        x = x.view(*input.shape[:-1], m).to(input.dtype)
        if self.bias is not None:
            x = x + self.bias
        return x
    
    @staticmethod
    def gen_layer_from_info(info, merge_layers=False, dummy=False, use_simt=False):
        layer = IncoherentLinear(
                    in_features=info["in_features"], 
                    out_features=info["out_features"], 
                    hadU=info["hadU"] if "hadU" in info else info["in_features"],
                    hadV=info["hadV"] if "hadV" in info else info["out_features"],
                    bias=info["bias"] is not None, 
                    dtype=info["dtype"], 
                    use_linear=False,
                )
        if not dummy:
            if info["bias"] is not None:
                layer.bias.data.copy_(info["bias"])
            layer.SU.data.copy_(info["SU"])
            layer.SV.data.copy_(info["SV"])
            layer.Wscale.data.copy_(info["Wscale"])
        if info["quant_info"] is not None:
            if "tcq" in info["quant_info"]["quantizer_str"]:
                layer.linear = QTIPLinearTCQ.gen_layer_from_info(info["linear_info"])
            elif "sq" in info["quant_info"]["quantizer_str"] or "vq" in info["quant_info"]["quantizer_str"] or "ldlq" in info["quant_info"]["quantizer_str"]:
                if use_simt:
                    layer.linear = VQLinearPackSIMT.gen_layer_from_info(info["linear_info"])
                else:
                    layer.linear = VQLinearPackTensorCore.gen_layer_from_info(info["linear_info"])
            elif "tcomb" in info["quant_info"]["quantizer_str"]:
                layer.linear = CombtLinearTCQ.gen_layer_from_info(info["linear_info"])
            elif "comb" in info["quant_info"]["quantizer_str"]:
                layer.linear = CombLinearTCQ.gen_layer_from_info(info["linear_info"])
        if "rot_info" in info["quant_info"]:
            layer.rot_info = info["quant_info"]["rot_info"]
        elif "rot_info" in info:
            layer.rot_info = info["rot_info"]
        else:
            layer.rot_info = "all"
        if merge_layers:
            layer.apply_rot_info()
        return layer
    
    @staticmethod
    def gen_layer_from_quantizer_str_and_key(config, quant_dir, quantizer_str, key, merge_layers=False, dummy=False, use_simt=False):
        if not dummy:
            info = torch.load(f"{quant_dir}/{quantizer_str}/{key}.pt")
        else:
            from lib.utils.mem_op import get_dummy_quant_results
            from lib.config import MODEL_KEYS
            model_key = MODEL_KEYS[config._name_or_path]
            layer_id = key.split("_")[0]
            layer_key = key.replace(f"{layer_id}_", "")
            info = get_dummy_quant_results(model_key, f"{layer_key}", quantizer_str)
        return IncoherentLinear.gen_layer_from_info(info, merge_layers=merge_layers, dummy=dummy, use_simt=use_simt)


def calc_kurtosis(W):
    # W: (-1, n), ||W[i]|| = 1
    W = W.to(torch.float64)
    return W.pow(4).mean(-1) - 3.0

def calc_skewness(W):
    # W: (-1, n), ||W[i]|| = 1
    W = W.to(torch.float64)
    return W.pow(3).mean(-1)

def linear_to_incoherent(linear, hadU, hadV, SU=None, SV=None, lnorm=None, rot_info="all"):
    dtype_ = torch.float32
    dtype = linear.weight.data.dtype
    device = linear.weight.device
    inc_linear = IncoherentLinear(linear.in_features, linear.out_features, hadU, hadV, linear.bias is not None, dtype)
    if SU is None:
        SU = ((torch.randn(linear.in_features, dtype=dtype_) > 0.0) * 2.0 - 1.0).to(device)
    if SV is None:
        SV = ((torch.randn(linear.out_features, dtype=dtype_) > 0.0) * 2.0 - 1.0).to(device)
    if lnorm is not None:
        lnorm = lnorm.to(device).to(dtype_)

    if linear.bias is not None:
        inc_linear.bias.data.copy_(linear.bias)

    W = linear.weight.data.to(dtype_)
    Wr = (W.to(torch.float64).to(device) @ torch.diag(lnorm).to(torch.float64)).to(dtype_).to(device) if lnorm is not None else W
    if hadU != linear.in_features or hadV != linear.out_features:
        Wr = matmul_hadUt_head(matmul_hadUt_head(Wr.T.to(device) * SV, hadV).T * SU, hadU)
    else:
        Wr = matmul_hadUt(matmul_hadUt(Wr.T.to(device) * SV).T * SU)
    # Wscale = Wr.square().mean().sqrt()
    Wscale = Wr.to(torch.float64).square().mean(-1).sqrt().view(-1, 1).to(dtype_)

    Wr = Wr / Wscale

    inc_linear.SU.data.copy_(SU.to(inc_linear.SU.dtype))
    # inc_linear.SV.data.copy_((SV * Wscale).to(inc_linear.SV.dtype))
    inc_linear.SV.data.copy_((SV).to(inc_linear.SV.dtype))
    inc_linear.Wscale.data.copy_(Wscale.view(-1))
    inc_linear.linear.weight.data.copy_(Wr.to(inc_linear.linear.weight.dtype))
    inc_linear = inc_linear.to(dtype).to(device)
    inc_linear.rot_info = rot_info
    inc_linear.apply_rot_info()


    # anal weight
    kurt = calc_kurtosis(inc_linear.linear.weight.data)
    skew = calc_skewness(inc_linear.linear.weight.data)
    # print(kurt.pow(2).mean(), kurt.mean(), kurt.std(), kurt.max(), kurt.min())
    # print pretty
    print(f"E[kurt^2]: {kurt.pow(2).mean():.4f}, E[kurt]: {kurt.mean():.4f}, std[kurt]: {kurt.std():.4f}, max[kurt]: {kurt.max():.4f}, min[kurt]: {kurt.min():.4f}")
    print(f"E[skew^2]: {skew.pow(2).mean():.4f}, E[skew]: {skew.mean():.4f}, std[skew]: {skew.std():.4f}, max[skew]: {skew.max():.4f}, min[skew]: {skew.min():.4f}")
    kurt_stats = {
        "kurt_pow2_mean": kurt.pow(2).mean(),
        "kurt_mean": kurt.mean(),
        "kurt_std": kurt.std(),
        "kurt_max": kurt.max(),
        "kurt_min": kurt.min(),
        "skew_pow2_mean": skew.pow(2).mean(),
        "skew_mean": skew.mean(),
        "skew_std": skew.std(),
        "skew_max": skew.max(),
        "skew_min": skew.min(),
    }
    return inc_linear, kurt_stats

if __name__ == "__main__":
    linear = nn.Linear(4096, 4096, bias=True, dtype=torch.float16).cuda()
    # linear.weight.data = linear.weight.data * 5 + 4
    inc_linear = linear_to_incoherent(linear)
    
    ran = torch.randn(4096, 4096, dtype=torch.float16).cuda()
    orig = linear(ran)
    inc = inc_linear(ran)

    print((orig - inc).pow(2).mean() / orig.pow(2).mean())
    
    