import torch
from lib.utils import block_LDL, matmul_hadUt, matmul_hadUt_head
from lib.algo.ldlq import LDLQ
from lib.quantizer.quant_op import load_hessian, load_group_hessian
from lib.linear import QTIPLinearTCQ, IncoherentLinear
from lib.codebook.bitshift import bitshift_codebook
import torch._dynamo
import time
class Args:
    def __init__(self, td_x, td_y, V):
        self.td_x = td_x
        self.td_y = td_y
        self.V = V

def qtip_quantize_mat(Wr, HRr, Wscale, cb, td_x=16, td_y=16, KV=4, V=2, use_hess=True):
    HRr_orig = HRr.clone()
    Wr = Wr.to(torch.float64)
    (m, n) = Wr.shape
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
        hatWr, Qidxs = LDLQ(cur_Wr, LRrs[i], cb.cuda(), args, for_kernel=True)
        hatWr_list.append(hatWr)
        Qidxs_list.append(Qidxs)
    hatWr = torch.cat(hatWr_list, dim=0)
    Qidxs = torch.cat(Qidxs_list, dim=0)
    assert hatWr.shape == Wr.shape, f"hatWr.shape {hatWr.shape} != Wr.shape {Wr.shape}"

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

    Wr *= Wscale.reshape(-1, 1)
    hatWr *= Wscale.reshape(-1, 1)

    orig_err = (Wr - hatWr).pow(2).mean()
    err = orig_err / Wr.pow(2).mean()
    print(
        f'err {err.item()} orig_err {orig_err.item()}'
    )
    quant_info = {
        "quantizer": "tcq_ldlq",
        "td_x": td_x,
        "td_y": td_y,
        "KV": KV,
        "V": V,
        "use_hess": use_hess,
        "orig_err": orig_err.item(),
        "err": err.item(),
    }
    return packed, hatWr, quant_info
    
def inc_linear_to_inc_tcq_linear(inc_linear, HRr, cb, td_x=16, td_y=16, KV=4, V=2, scale_override=0.9, use_hess=True):
    Wr = inc_linear.linear.weight.data * scale_override
    Wscale = inc_linear.Wscale.data / scale_override    
    inc_linear.Wscale.data.copy_(Wscale)

    packed, hatWr, quant_info = qtip_quantize_mat(Wr, HRr, Wscale, cb, td_x=td_x, td_y=td_y, KV=KV, V=V, use_hess=use_hess)
    out_features, in_features = Wr.shape
    tcq_linear = QTIPLinearTCQ(
        in_features,
        out_features,
        td_x=16,
        td_y=16,
        L=16,
        KV=KV,
        V=V,
        tlut_bits=cb.tlut_bits,
        bias=inc_linear.bias is not None,
        dtype=inc_linear.dtype,
    )

    tcq_linear.trellis.data.copy_(packed)
    tcq_linear.tlut.data.copy_(cb.tlut)

    inc_linear.linear = tcq_linear
    return inc_linear, quant_info

def linear_to_incoherent_for_tcq(linear, cb, HR, scale_override=0.9, SU=None, SV=None, lnorm=None, hadU=None, hadV=None, rot_info="all", left_only=False):
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

    W = linear.weight.data.clone().to(dtype_)
    Wr = matmul_hadUt_head(matmul_hadUt_head(W.T.to(device) * SV, hadV).T * SU, hadU) if not left_only else matmul_hadUt_head(W * SU, hadU)

    if left_only:
        Wscale = Wr.to(torch.float64).square().mean(-1).sqrt().view(-1, 1).to(dtype_) / (cb.lut.to(torch.float64).square().mean().sqrt().float() * scale_override) # (out_features, 1)
    else:
        Wscale = Wr.to(torch.float64).square().mean().sqrt().view(-1, 1).to(dtype_) / (cb.lut.to(torch.float64).square().mean().sqrt().float() * scale_override) # (1, 1)
        Wscale = Wscale.repeat(Wr.shape[0], 1) # (out_features, 1)

    Wr = Wr / Wscale
    HRr = torch.zeros_like(HR)
    for i in range(HR.shape[-1]):
        HRr[:,:,i] = matmul_hadUt_head(matmul_hadUt_head(HR[:,:,i].to(device).contiguous() * (1./ SU), hadU).T * (1./ SU), hadU)

    inc_linear.SU.data.copy_(1./SU.to(dtype_))
    inc_linear.SV.data.copy_(1./SV.to(dtype_))
    inc_linear.Wscale.data.copy_(Wscale.view(-1))
    inc_linear.linear.weight.data.copy_(Wr.to(dtype_))
    inc_linear.rot_info = rot_info
    inc_linear.apply_rot_info()
    return inc_linear, HRr

def linear_to_tcq_linear(target_layer, hess_path, cb, scale_override=0.9, KV=4, V=2, use_hess=True, SU=None, SV=None, lnorm=None, hadU=None, hadV=None, rot_info="all", left_only=False, ghess_key=""):
    t0 = time.time()
    out_features, in_features = target_layer.weight.shape
    if ghess_key == "":
        HR = load_hessian(hess_path).cuda() if hess_path is not None else torch.eye(in_features, device="cuda", dtype=torch.float64).unsqueeze(-1)
    else:
        HR = load_group_hessian(hess_path, layer_key=ghess_key).cuda()
    layer, HRr = linear_to_incoherent_for_tcq(target_layer, cb, HR, scale_override, SU=SU, SV=SV, lnorm=lnorm, hadU=hadU, hadV=hadV, rot_info=rot_info, left_only=left_only)
    HRr = HRr.cuda()
    layer = layer.cuda()
    layer, quant_info = inc_linear_to_inc_tcq_linear(layer, HRr, cb, scale_override=1.0, td_x=16, td_y=16, KV=KV, V=V, use_hess=use_hess)
    quant_info["scale_override"] = scale_override
    quant_info["hess_path"] = hess_path
    quant_info["time"] = time.time() - t0
    
    return layer.to(torch.float16), quant_info
