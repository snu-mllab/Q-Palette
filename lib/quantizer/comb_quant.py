import torch
from lib.quantizer.quant_op import load_hessian, load_group_hessian
from lib.quantizer.tcq_quant import qtip_quantize_mat, linear_to_incoherent_for_tcq, Args
from lib.linear import CombLinearTCQ, CombtLinearTCQ
from lib.codebook.bitshift import bitshift_codebook
from lib.utils import clean
import time
from lib.utils import block_LDL
from lib.algo.ldlq import LDLQ_combt

def pack_trellis(Qidxs, td_x, td_y, cb, m, n, KV, V):
    Qidxs = Qidxs.cpu()
    packed = cb.pack_trellis(
        Qidxs.reshape(m // td_x, td_x, n // td_y,
                        td_y // V).transpose(1, 2).reshape(
                            -1, td_x * td_y // V))
    
    packed_8 = packed.view(torch.uint8).view(-1, 2)
    packed_4 = torch.cat([packed_8.unsqueeze(-1) & (2 ** 4 - 1), (packed_8.unsqueeze(-1) & (2 ** 8 - 2 ** 4)) >> 4], dim=-1).view(-1, 4).flip(
                (-1, ))
    
    packed_4 = packed_4.reshape(m // 16 // 2, 2, n // 16 // 2, 2, 16 * 16 // 8,
                                KV).permute(0, 2, 4, 3, 1, 5).flip(
                                    (-1, )).contiguous().flatten()
    packed_8 = torch.sum(packed_4.view(-1, 2) * torch.Tensor([[1, 2 ** 4]]).to(torch.uint8), dim=-1).to(torch.uint8).contiguous()
    packed = packed_8.view(torch.int16).reshape(packed.shape).cuda()
    return packed

def combt_quantize_mat(Wr, HRr, Wscale, cb1, cb2, td_x=16, td_y=16, KV=(4,5), V=2, use_hess=True):
    (m, n) = Wr.shape
    Wr = Wr.to(torch.float64)
    HRr_orig = HRr.clone()
    gs = HRr.shape[-1]
    LRrs = []
    diag = torch.arange(n, device=HRr.device)
    if not use_hess:
        eye = torch.eye(n, device=Wr.device, dtype=torch.float64)
        LRr, D = block_LDL(eye, td_y)
        LRr[diag, diag] = 0
        LRrs.append(LRr)
    else:
        for i in range(gs):
            LRr, D = block_LDL(HRr[:,:,i], td_y)
            LRr[diag, diag] = 0
            LRrs.append(LRr)
    
    args = Args(td_x, td_y, V)

    Qidxs_list = []
    hatWr_list = []
    for i in range(gs):
        cur_Wr = Wr[m // gs * i:m // gs * (i+1)]
        hatWr, Qidxs = LDLQ_combt(cur_Wr, LRrs[i], cb1.cuda(), cb2.cuda(), args, for_kernel=True)
        hatWr_list.append(hatWr)
        Qidxs_list.append(Qidxs)
        torch._dynamo.reset()
        
    hatWr = torch.cat(hatWr_list, dim=0)
    Qidxs = torch.cat(Qidxs_list, dim=0)
    assert hatWr.shape == Wr.shape, f"hatWr.shape {hatWr.shape} != Wr.shape {Wr.shape}"

    packed1 = pack_trellis(Qidxs[:, :n//2//V].contiguous(), td_x, td_y, cb1, m, n//2, KV[0], V)
    packed2 = pack_trellis(Qidxs[:, n//2//V:].contiguous(), td_x, td_y, cb2, m, n//2, KV[1], V)

    Wr *= Wscale.reshape(-1, 1)
    hatWr *= Wscale.reshape(-1, 1)

    orig_err = (Wr - hatWr).pow(2).mean()
    err = orig_err / Wr.pow(2).mean()
    print(
        f'err {err.item()} orig_err {orig_err.item()}'
    )
    quant_info = {
        "quantizer": "combt_ldlq",
        "td_x": td_x,
        "td_y": td_y,
        "KV": KV,
        "V": V,
        "tlut_bits": cb1.tlut_bits,
        "use_hess": use_hess,
        "orig_err": orig_err.item(),
        "err": err.item(),
    }
    return packed1, packed2, hatWr, quant_info
    
def inc_linear_to_inc_combt_linear(inc_linear, HRr, cb1, cb2, td_x=16, td_y=16, in_part=(2048, 2048), KV=(3, 4), V=2, scale_override=0.9, use_hess=True):
    Wr = (inc_linear.linear.weight.data * scale_override).to(HRr.dtype)
    Wscale = inc_linear.Wscale.data / scale_override    
    inc_linear.Wscale.data.copy_(Wscale)
    assert in_part[0] + in_part[1] == Wr.shape[1], "in_part is not correct"
    assert torch.allclose(cb1.tlut, cb2.tlut), "cb1 and cb2 must have the same tlut"
   
    packed1, packed2, hatWr, quant_info = combt_quantize_mat(Wr, HRr, Wscale, cb1, cb2, td_x=td_x, td_y=td_y, KV=KV, V=V, use_hess=use_hess)
    torch._dynamo.reset()
    out_features, in_features = Wr.shape
    comb_linear = CombtLinearTCQ(
        in_features,
        out_features,
        td_x=td_x,
        td_y=td_y,
        in_part=in_part,
        L=16,
        KV=KV,
        V=V,
        tlut_bits=cb1.tlut_bits,
        bias=inc_linear.bias is not None,
        dtype=inc_linear.dtype,
    )

    comb_linear.trellis1.data.copy_(packed1)
    comb_linear.trellis2.data.copy_(packed2)
    comb_linear.tlut.data.copy_(cb1.tlut)
    inc_linear.linear = comb_linear
    return inc_linear, quant_info


def inc_linear_to_inc_comb_linear(inc_linear, HRr, cb1, cb2, td_x=16, td_y=16, out_part=(2048, 2048), KV=(3, 4), V=2, scale_override=0.9, use_hess=True):
    Wr = (inc_linear.linear.weight.data * scale_override).to(HRr.dtype)
    Wscale = inc_linear.Wscale.data / scale_override    
    inc_linear.Wscale.data.copy_(Wscale)
    assert out_part[0] + out_part[1] == Wr.shape[0], "out_part is not correct"
    assert len(HRr.shape) == 3 and HRr.shape[0] == HRr.shape[1] and HRr.shape[-1] == 1, f"support only none-grouped hessian but shape: {HRr.shape}"

    packed1, hatWr1, quant_info1 = qtip_quantize_mat(Wr[:out_part[0]], HRr, Wscale[:out_part[0]], cb1, td_x=td_x, td_y=td_y, KV=KV[0], V=V, use_hess=use_hess)
    torch._dynamo.reset()
    packed2, hatWr2, quant_info2 = qtip_quantize_mat(Wr[out_part[0]:], HRr, Wscale[out_part[0]:], cb2, td_x=td_x, td_y=td_y, KV=KV[1], V=V, use_hess=use_hess)
    torch._dynamo.reset()
    out_features, in_features = Wr.shape
    comb_linear = CombLinearTCQ(
        in_features,
        out_features,
        td_x=td_x,
        td_y=td_y,
        out_part=out_part,
        L=16,
        KV=KV,
        V=V,
        tlut_bits=cb1.tlut_bits,
        bias=inc_linear.bias is not None,
        dtype=inc_linear.dtype,
    )

    comb_linear.trellis1.data.copy_(packed1)
    comb_linear.trellis2.data.copy_(packed2)
    comb_linear.tlut.data.copy_(cb1.tlut)

    hatWr = torch.cat([hatWr1, hatWr2], dim=0).to(HRr.dtype)
    orig_err = (Wr - hatWr).pow(2).mean()
    err = orig_err / Wr.pow(2).mean()

    quant_info = {
        "quantizer": "comb_ldlq",
        "td_x": td_x,
        "td_y": td_y,
        "KV": KV,
        "V": V,
        "use_hess": use_hess,
        "orig_err": orig_err.item(),
        "err": err.item(),
        "quant_info1": quant_info1,
        "quant_info2": quant_info2,
    }

    inc_linear.linear = comb_linear
    return inc_linear, quant_info

def linear_to_comb_linear(target_layer, hess_path, cb1, cb2, scale_override=0.9, out_part=(2048, 2048), KV=[3, 4], V=2, use_hess=True, SU=None, SV=None, lnorm=None, hadU=None, hadV=None, rot_info="all", left_only=False, ghess_key=""):
    assert torch.allclose(cb1.tlut, cb2.tlut), "cb1 and cb2 must have the same tlut"
    t0 = time.time()
    out_features, in_features = target_layer.weight.shape
    if ghess_key == "":
        HR = load_hessian(hess_path).cuda() if hess_path is not None else torch.eye(in_features, device="cuda", dtype=torch.float64).unsqueeze(-1)
    else:
        HR = load_group_hessian(hess_path, layer_key=ghess_key).cuda()
    layer, HRr = linear_to_incoherent_for_tcq(target_layer, cb1, HR, scale_override, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    HRr = HRr.cuda()
    layer = layer.cuda()
    layer, quant_info = inc_linear_to_inc_comb_linear(layer, HRr, cb1, cb2, scale_override=1.0, td_x=16, td_y=16, out_part=out_part, KV=KV, V=V, use_hess=use_hess)
    quant_info["scale_override"] = scale_override
    quant_info["hess_path"] = hess_path
    quant_info["time"] = time.time() - t0
    
    return layer.to(torch.float16), quant_info

def linear_to_combt_linear(target_layer, hess_path, cb1, cb2, scale_override=0.9, in_part=(2048, 2048), KV=[3, 4], V=2, use_hess=True, SU=None, SV=None, lnorm=None, hadU=None, hadV=None, rot_info="all", left_only=False, ghess_key=""):
    assert torch.allclose(cb1.tlut, cb2.tlut), "cb1 and cb2 must have the same tlut"
    t0 = time.time()
    out_features, in_features = target_layer.weight.shape
    if ghess_key == "":
        HR = load_hessian(hess_path).cuda() if hess_path is not None else torch.eye(in_features, device="cuda", dtype=torch.float64).unsqueeze(-1)
    else:
        HR = load_group_hessian(hess_path, layer_key=ghess_key).cuda()
    layer, HRr = linear_to_incoherent_for_tcq(target_layer, cb1, HR, scale_override, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    HRr = HRr.cuda()
    layer = layer.cuda()
    layer, quant_info = inc_linear_to_inc_combt_linear(layer, HRr, cb1, cb2, scale_override=1.0, td_x=16, td_y=16, in_part=in_part, KV=KV, V=V, use_hess=use_hess)
    quant_info["scale_override"] = scale_override
    quant_info["hess_path"] = hess_path
    quant_info["time"] = time.time() - t0
    
    return layer.to(torch.float16), quant_info
