import torch
from lib.quantizer.quant_op import _INV_PERMUTE, load_hessian, load_group_hessian
from lib.linear import IncoherentLinear
from lib.utils import matmul_hadUt, matmul_hadUt_head, clean
from lib.utils.kmeans import kmeans_sklearn, kmeans_flash1d
from lib.quantizer.nuq_op import train_least_squares
from lib.quantizer.quant_op import pack_qweight
from lib.linear import VQLinearPackTensorCore

import random
import time
def simple_vq(Wr, vec_sz, lut_bits, batch_size=256):
    batch_size = 64 if lut_bits >= 12 else batch_size
    # kmeans
    Wr_flatten = Wr.reshape(-1, vec_sz)

    if vec_sz == 1:
        init_centroids = kmeans_flash1d(Wr_flatten, 2 ** lut_bits)
    else:
        init_centroids = kmeans_sklearn(Wr_flatten, 2 ** lut_bits)
    Wr_vec = Wr.reshape(Wr.shape[0], -1, vec_sz) # (W_row, W_col // vec_sz, vec_sz)

    min_indices = torch.zeros(Wr.shape[0], Wr.shape[1] // vec_sz).to(Wr.device).long()
    for s_idx in range(0, Wr.shape[0], batch_size):
        e_idx = min(s_idx + batch_size, Wr.shape[0])
        dist_sq = ((Wr_vec[s_idx:e_idx].unsqueeze(2) - init_centroids.unsqueeze(0).unsqueeze(0)) ** 2).sum(dim=-1)
        idx = dist_sq.argmin(dim=-1) # batch_size, W_col // vec_sz
        min_indices[s_idx:e_idx] = idx

    init_P = torch.zeros(Wr.shape[0], Wr.shape[1] // vec_sz, 2 ** lut_bits, dtype=torch.uint8).to(Wr.device)
    init_P.scatter_(2, min_indices.unsqueeze(-1), 1)

    return init_P, init_centroids.to(torch.float32)

def vq_quantize_mat(Wr, HRr, Wscale, vec_sz, lut_bits, iterations=6, use_hess=True):
    Wr = Wr.to(torch.float64)
    if use_hess:
        assert len(HRr.shape) == 3, "HRr must be a 3D tensor"
        assert HRr.shape[0] == HRr.shape[1], "HRr must be a square matrix"
        init_P, init_centroids = simple_vq(Wr, vec_sz, lut_bits)
        P, C, log_dict = train_least_squares(
            W=Wr.detach().cpu().numpy(),
            init_P=init_P.detach().cpu().to(torch.float32).numpy(),
            init_centroids=init_centroids.detach().cpu().numpy(),
            H=HRr.permute(2, 0, 1).detach().cpu().numpy(),
            num_iterations=iterations,
        )
    else:
        init_P, init_centroids = simple_vq(Wr, vec_sz, lut_bits)
        P, C = init_P, init_centroids
    P = P.to(Wr.device)
    C = C.to(Wr.device)

    P_ind = torch.argmax(P, dim=-1)
    hatWr = C[P_ind]
    hatWr = hatWr.view(hatWr.shape[0], -1)

    Wr *= Wscale.view(-1, 1)
    hatWr *= Wscale.view(-1, 1)

    orig_err = (Wr - hatWr).pow(2).mean()
    err = (Wr - hatWr).pow(2).mean() / (Wr.pow(2).mean())
    print(
        f'err {err.item()} orig_err {orig_err.item()}'
    )
    quant_info = {
        "quantizer": "vq_lnq",
        "vec_sz": vec_sz,
        "lut_bits": lut_bits,
        "use_hess": use_hess,
        "iterations": iterations,
        "orig_err": orig_err.item(),
        "err": err.item(),
    }

    # pack P appropriately to kernel
    packed = pack_qweight(P, vec_sz, lut_bits)
    return packed, C, hatWr, quant_info

def inc_linear_to_inc_vq_linear(inc_linear, HRr, lut_bits=4, vec_sz=2, scale_override=0.9, use_hess=True):
    Wr = inc_linear.linear.weight.data * scale_override
    Wscale = inc_linear.Wscale.data / scale_override
    inc_linear.Wscale.data.copy_(Wscale)

    packed, C, hatWr, quant_info = vq_quantize_mat(Wr, HRr, Wscale, vec_sz, lut_bits, use_hess=use_hess)
    out_features, in_features = Wr.shape
    sq_linear = VQLinearPackTensorCore(
        in_features,
        out_features,
        lut_bits=lut_bits,
        vec_sz=vec_sz,
        bias=inc_linear.bias is not None,
        dtype=inc_linear.dtype,
    )
    sq_linear.qweight.data.copy_(packed)
    sq_linear.lut.data.copy_(C.view(2 ** lut_bits, vec_sz))

    inc_linear.linear = sq_linear
    return inc_linear, quant_info

def linear_to_incoherent_for_vq(linear, HR, scale_override=0.9, SU=None, SV=None, lnorm=None, hadU=None, hadV=None, rot_info="all", left_only=False):
    dtype_ = torch.float32
    device = linear.weight.device
    inc_linear = IncoherentLinear(linear.in_features, linear.out_features, hadU, hadV, linear.bias is not None, dtype_)
    if SU is None:
        SU = ((torch.randn(linear.in_features, dtype=dtype_) > 0.0) * 2.0 - 1.0).to(device).to(dtype_)
    if SV is None:
        SV = ((torch.randn(linear.out_features, dtype=dtype_) > 0.0) * 2.0 - 1.0).to(device).to(dtype_)

    if left_only:
        SV = torch.ones_like(SV)

    if linear.bias is not None:
        inc_linear.bias.data.copy_(linear.bias)

    W = linear.weight.data.to(dtype_)
    Wr = matmul_hadUt_head(matmul_hadUt_head(W.T.to(device) * SV, hadV).T * SU, hadU) if not left_only else matmul_hadUt_head(W * SU, hadU)
    Wscale = Wr.to(torch.float64).square().mean(-1).sqrt().view(-1, 1).to(dtype_) / scale_override

    Wr = Wr / Wscale
    HRr = torch.zeros_like(HR)
    for i in range(HR.shape[-1]):
        HRr[:,:,i] = matmul_hadUt_head(matmul_hadUt_head(HR[:,:,i].to(device).contiguous() * (1./ SU), hadU).T * (1./ SU), hadU)

    inc_linear.SU.data.copy_(1./SU.to(dtype_))
    inc_linear.SV.data.copy_((1./SV).to(dtype_))
    inc_linear.Wscale.data.copy_(Wscale.view(-1))
    inc_linear.linear.weight.data.copy_(Wr.to(dtype_))
    inc_linear.rot_info = rot_info
    inc_linear.apply_rot_info()
    return inc_linear, HRr

def linear_to_vq_linear(target_layer, hess_path, scale_override=0.9, lut_bits=4, vec_sz=1, use_hess=True, SU=None, SV=None, lnorm=None, hadU=None, hadV=None, rot_info="all", left_only=False, ghess_key=""):
    t0 = time.time()
    out_features, in_features = target_layer.weight.shape
    if ghess_key == "":
        HR = load_hessian(hess_path).cuda() if hess_path is not None else torch.eye(in_features, device="cuda", dtype=torch.float64).unsqueeze(-1)
    else:
        HR = load_group_hessian(hess_path, layer_key=ghess_key).cuda()
    layer, HRr = linear_to_incoherent_for_vq(target_layer, HR, scale_override, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    HRr = HRr.cuda()
    layer = layer.cuda()
    layer, quant_info = inc_linear_to_inc_vq_linear(layer, HRr, scale_override=1.0, lut_bits=lut_bits, vec_sz=vec_sz, use_hess=use_hess)
    
    quant_info["scale_override"] = scale_override
    quant_info["hess_path"] = hess_path
    quant_info["time"] = time.time() - t0
    print("elapsed time", time.time() - t0)
    return layer.to(torch.float16), quant_info
