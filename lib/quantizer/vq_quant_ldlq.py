import torch
from lib.quantizer.quant_op import load_hessian, load_group_hessian
from lib.utils import clean, block_LDL
from lib.algo.ldlq import LDLQ_VQ
from lib.quantizer.quant_op import pack_qweight, pack_qweight_vq_simt, pack_qweight_sq_simt
from lib.linear import VQLinearPackTensorCore
from lib.quantizer.vq_quant import linear_to_incoherent_for_vq

import time

def vq_quantize_mat_ldlq(Wr, HRr, Wscale, cb, use_hess=True, pack_simt=False):
    HRr_orig = HRr.clone()
    Wr = Wr.to(torch.float64)
    (m, n) = Wr.shape
    gs = HRr.shape[-1]
    LRrs = []
    diag = torch.arange(n, device=HRr.device)
    if not use_hess:
        eye = torch.eye(n, device=Wr.device, dtype=torch.float64)
        LRr, D = block_LDL(eye, cb.vec_sz)
        LRr[diag, diag] = 0
        LRrs.append(LRr)
    else:
        for i in range(gs):
            LRr, D = block_LDL(HRr[:,:,i], cb.vec_sz)
            LRr[diag, diag] = 0
            LRrs.append(LRr)
    
    Qidxs_list = []
    hatWr_list = []
    for i in range(gs):
        cur_Wr = Wr[m // gs * i:m // gs * (i+1)] # (m // gs, n)
        hatWr, Qidxs = LDLQ_VQ(cur_Wr, LRrs[i], cb.cuda())
        hatWr_list.append(hatWr)
        Qidxs_list.append(Qidxs)
    hatWr = torch.cat(hatWr_list, dim=0)
    Qidxs = torch.cat(Qidxs_list, dim=0)
    assert hatWr.shape == Wr.shape, f"hatWr.shape {hatWr.shape} != Wr.shape {Wr.shape}"

    Wr *= Wscale.view(-1, 1)
    hatWr *= Wscale.view(-1, 1)

    orig_err = (Wr - hatWr).pow(2).mean()
    err = (Wr - hatWr).pow(2).mean() / (Wr.pow(2).mean())
    print(
        f'err {err.item()} orig_err {orig_err.item()}'
    )
    quant_info = {
        "quantizer": "vq_ldlq",
        "vec_sz": cb.vec_sz,
        "lut_bits": cb.lut_bits,
        "use_hess": use_hess,
        "orig_err": orig_err.item(),
        "err": err.item(),
    }

    # pack P appropriately to kernel
    if pack_simt or cb.vec_sz > 2:
        if cb.vec_sz == 1:
            packed = pack_qweight_sq_simt(Qidxs, cb.lut_bits)
        else:
            packed = pack_qweight_vq_simt(Qidxs, cb.lut_bits, cb.vec_sz, cb.lut_bits)
    else:
        packed = pack_qweight(Qidxs, cb.vec_sz, cb.lut_bits)
    return packed, hatWr, quant_info

def inc_linear_to_inc_vq_linear_ldlq(inc_linear, HRr, cb, scale_override=0.9, use_hess=True):
    Wr = inc_linear.linear.weight.data * scale_override
    Wscale = inc_linear.Wscale.data / scale_override
    inc_linear.Wscale.data.copy_(Wscale)

    packed, hatWr, quant_info = vq_quantize_mat_ldlq(Wr, HRr, Wscale, cb, use_hess=use_hess)
    out_features, in_features = Wr.shape
    sq_linear = VQLinearPackTensorCore(
        in_features,
        out_features,
        lut_bits=cb.lut_bits,
        vec_sz=cb.vec_sz,
        bias=inc_linear.bias is not None,
        dtype=inc_linear.dtype,
    )
    sq_linear.qweight.data.copy_(packed)
    sq_linear.lut.data.copy_(cb.tlut)

    inc_linear.linear = sq_linear
    return inc_linear, quant_info

def linear_to_vq_linear_ldlq(target_layer, hess_path, cb, scale_override=0.9, use_hess=True, SU=None, SV=None, lnorm=None, hadU=None, hadV=None, rot_info="all", left_only=False, ghess_key=""):
    t0 = time.time()
    out_features, in_features = target_layer.weight.shape
    if ghess_key == "":
        HR = load_hessian(hess_path).cuda() if hess_path is not None else torch.eye(in_features, device="cuda", dtype=torch.float64).unsqueeze(-1)
    else:
        HR = load_group_hessian(hess_path, layer_key=ghess_key).cuda()
    layer, HRr = linear_to_incoherent_for_vq(target_layer, HR, scale_override, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    HRr = HRr.cuda()
    layer = layer.cuda()
    layer, quant_info = inc_linear_to_inc_vq_linear_ldlq(layer, HRr, cb, scale_override=1.0, use_hess=use_hess)
    
    quant_info["scale_override"] = scale_override
    quant_info["hess_path"] = hess_path
    quant_info["time"] = time.time() - t0
    print("elapsed time", time.time() - t0)
    return layer.to(torch.float16), quant_info
