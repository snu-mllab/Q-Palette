import copy
import os

import glog
import torch
from tqdm import tqdm
import time
from lib import utils

_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0,
                                                            4).flatten()

_PERMUTE_HALF = torch.arange(128).reshape(2, 8, 2, 4, 1).permute(1, 3, 2, 0,
                                                            4).flatten()
_INV_PERMUTE = torch.zeros(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)
_INV_PERMUTE_HALF = torch.zeros(128, dtype=torch.int64)
_INV_PERMUTE_HALF[_PERMUTE_HALF] = torch.arange(128)

def LDLQ(Wr, L, cb, args, D=None, buf_cols=128, for_kernel=True, use_beam_search=False, use_diag=False):
    if for_kernel:
        assert args.td_x == 16 and args.td_y == 16
    buf_cols = max(buf_cols, args.td_y)
    trellissz = args.td_x * args.td_y
    (m, n) = Wr.shape
    assert buf_cols % args.td_y == 0
    assert n % buf_cols == 0
    assert args.td_y % args.V == 0
    buf_size = buf_cols // args.td_y

    hatWr_T = torch.zeros(n, m, dtype=L.dtype, device=L.device)
    Qidxs_T = torch.zeros(n // args.V, m, dtype=cb.idx_dtype, device=L.device)

    device = Wr.device
    Wr = Wr.cpu()
    utils.clean()
    Wr_T = Wr.T.contiguous().to(device)

    # quip
    prod_cache = torch.zeros(n, m, dtype=Wr_T.dtype, device=Wr_T.device)
    for cur_col in tqdm(range(n // args.td_y, 0, -buf_size)):
        b_Wr_T = Wr_T[args.td_y * (cur_col - buf_size):args.td_y * cur_col]
        b_hatWr_T = hatWr_T[args.td_y * (cur_col - buf_size):args.td_y *
                            cur_col]
        b_L = L[args.td_y * (cur_col - buf_size):args.td_y *
                cur_col].contiguous()
        b_prod = prod_cache[args.td_y * (cur_col - buf_size):args.td_y *
                            cur_col]
        b_Qidxs_T = Qidxs_T[args.td_y * (cur_col - buf_size) //
                            args.V:args.td_y * cur_col // args.V]
        L_offset = args.td_y * (cur_col - buf_size)
        for i in reversed(range(buf_size)):
            WXWX = b_Wr_T[args.td_y * i : args.td_y * (i + 1)] + \
                b_L[args.td_y * (i + 1):, L_offset + args.td_y * i : L_offset + args.td_y * (i + 1)].T @ \
                (b_Wr_T[args.td_y * (i + 1):] - b_hatWr_T[args.td_y * (i + 1):]) + \
                b_prod[args.td_y * i : args.td_y * (i + 1)]
            if trellissz > -1:
                WXWXshape = WXWX.shape
                thing = WXWX.T.reshape(-1, trellissz)
                if for_kernel:
                    thing = thing[..., _PERMUTE]
                if use_beam_search:
                    # D: (n // td_y, td_y, td_y)
                    D_cur = D[cur_col - buf_size + i] # (td_y, td_y)
                    D_tiled = torch.kron(torch.eye(args.td_y, device=D_cur.device, dtype=D_cur.dtype), D_cur)
                    if for_kernel:
                        D_tiled = D_tiled[:, _PERMUTE][_PERMUTE, :]
                    q_out = cb.quantize_beam_search_with_hessian(thing, D_tiled, beam_sz=1024)
                else:
                    if use_diag:
                        D_cur = D[cur_col - buf_size + i] # (td_y, td_y)
                        weight = torch.diag(D_cur).repeat(trellissz // args.td_y)[_PERMUTE]
                        q_out = cb.quantize(thing, w2=weight)
                    else:
                        q_out = cb.quantize(thing)
                if for_kernel:
                    thing = q_out[0][..., _INV_PERMUTE].reshape(
                        WXWXshape[1], WXWXshape[0])
                else:
                    thing = q_out[0].reshape(WXWXshape[1], WXWXshape[0])
                idxs = q_out[1].reshape(WXWXshape[1], WXWXshape[0] // args.V)
                b_hatWr_T[args.td_y * i:args.td_y * (i + 1)] = thing.T
                b_Qidxs_T[args.td_y // args.V * i:args.td_y // args.V *
                          (i + 1)] = idxs.T
            else:
                raise NotImplementedError
                # q_out = cb.quantize(WXWX.T)
                # b_hatWr_T[args.td_y * i:args.td_y * (i + 1)] = q_out[0].T
                # b_Qidxs_T[args.td_y // args.V * i:args.td_y // args.V *
                #           (i + 1)] = q_out[1].T

        prod_cache += b_L.T @ (b_Wr_T - b_hatWr_T)
        hatWr_T[args.td_y * (cur_col - buf_size):args.td_y *
                cur_col] = b_hatWr_T

    del b_Wr_T, b_hatWr_T, b_L, b_prod, L_offset, prod_cache
    utils.clean()
    return hatWr_T.T.contiguous(), Qidxs_T.T.contiguous()

def calc_obj(hatWr_T, Wr_T, HRr):
    diff_T = hatWr_T.cuda() - Wr_T.cuda()
    obj = torch.trace(diff_T.T @ HRr @ diff_T)
    return obj.cpu().item()

def CD(Wr, HRr, Qidxs, hatWr, cb, args, buf_cols=128, for_kernel=True, use_beam_search=False):
    if for_kernel:
        assert args.td_x == 16 and args.td_y == 16
    buf_cols = max(buf_cols, args.td_y)
    trellissz = args.td_x * args.td_y
    (m, n) = Wr.shape
    assert buf_cols % args.td_y == 0
    assert n % buf_cols == 0
    assert args.td_y % args.V == 0
    buf_size = buf_cols // args.td_y

    hatWr_T = hatWr.T.contiguous()
    Qidxs_T = Qidxs.T.contiguous()
    device = hatWr.device
    hatWr = hatWr.cpu()
    utils.clean()
    Wr_T = Wr.T.contiguous().to(device)

    # obj = calc_obj(hatWr_T, Wr_T, HRr)
    # print("init obj", obj)
    for cur_col in tqdm(range(n // args.td_y, 0, -buf_size)):
        b_Wr_T = Wr_T[args.td_y * (cur_col - buf_size):args.td_y * cur_col]
        b_hatWr_T = hatWr_T[args.td_y * (cur_col - buf_size):args.td_y *
                            cur_col]
        b_Qidxs_T = Qidxs_T[args.td_y * (cur_col - buf_size) //
                            args.V:args.td_y * cur_col // args.V]
        b_HRr = HRr[args.td_y * (cur_col - buf_size):args.td_y * cur_col, args.td_y * (cur_col - buf_size):args.td_y * cur_col] # (buf_size * td_y, buf_size * td_y)

        # update global hessian
        res_inds = torch.cat([
            torch.arange(0, (cur_col - buf_size) * args.td_y, device=device),
            torch.arange(cur_col * args.td_y, n, device=device)
        ])
        Wr_diff_T = hatWr_T - Wr_T # (n, m)
        b_global_hess = torch.matmul(Wr_diff_T[res_inds].T, HRr[res_inds, args.td_y * (cur_col - buf_size):args.td_y * cur_col]) # (m, buf_size * td_y)
        for i in reversed(range(buf_size)):
            start_col, end_col = args.td_y * i, args.td_y * (i + 1)
            WXWX = b_Wr_T[start_col:end_col] # (td_y, m) 
            b_Wr_diff_T = b_hatWr_T - b_Wr_T # (td_y * buf_size, m)
            if trellissz > -1:
                WXWXshape = WXWX.shape 
                thing = WXWX.T.reshape(-1, trellissz) # (-1, trellissz)
                if for_kernel:
                    thing = thing[..., _PERMUTE] 
                
                # local hessian
                HRr_cur = b_HRr[start_col:end_col, start_col:end_col].contiguous() # (td_y, td_y) 
                HRr_tiled = torch.kron(torch.eye(args.td_y, device=HRr_cur.device, dtype=HRr_cur.dtype), HRr_cur)
                
                # global hessian
                cur_global_hess = b_global_hess[:, start_col:end_col] # (m, td_y)
                cur_res_ind =  torch.cat([
                                        torch.arange(0, start_col, device=device),
                                        torch.arange(end_col, buf_size * args.td_y, device=device)
                                    ]) # 나머지 indices for args.td_y * i : args.td_y * (i + 1)
                
                cur_global_hess_res = torch.matmul(b_Wr_diff_T[cur_res_ind].T, b_HRr[cur_res_ind, start_col:end_col]) # (m, td_y)
                cur_weight = cur_global_hess + cur_global_hess_res # (m, td_y)
                cur_weight = cur_weight.reshape(-1, trellissz)
                if for_kernel:
                    cur_weight = cur_weight[..., _PERMUTE] # (-1, trellissz)
                    HRr_tiled = HRr_tiled[:, _PERMUTE][_PERMUTE, :]
                
                cur_hatWr_T = b_hatWr_T[start_col:end_col].T.reshape(-1, trellissz)[..., _PERMUTE].contiguous() # (-1, trellissz)
                cur_qidx = b_Qidxs_T[args.td_y // args.V * i:args.td_y // args.V * (i + 1)].T.reshape(-1, trellissz // args.V) # (-1, trellissz)
                diff = cur_hatWr_T - thing
                obj_before = torch.diag(diff @ HRr_tiled @ diff.T) + torch.sum(cur_weight * diff, dim=-1) * 2 # (-1)

                if use_beam_search:
                    q_out = cb.quantize_beam_search_with_hessian(thing, HRr_tiled, U=cur_weight, beam_sz=1024)
                else:
                    q_out = cb.quantize(thing, w1=cur_weight * 2, w2=torch.diag(HRr_tiled))
                diff = q_out[0] - thing
                obj_after = torch.diag(diff @ HRr_tiled @ diff.T) + torch.sum(cur_weight * diff, dim=-1) * 2 # (-1)

                # select only improved 
                improved = obj_before > obj_after
                # out[i] = q_out[0][i] if improved[i] else cur_hatWr_T[i]
                new_hatWr_T = torch.where(improved.unsqueeze(-1), q_out[0], cur_hatWr_T)
                new_qidx = torch.where(improved.unsqueeze(-1), q_out[1], cur_qidx)

                if for_kernel:
                    thing = new_hatWr_T[..., _INV_PERMUTE].reshape(
                        WXWXshape[1], WXWXshape[0])
                else:
                    thing = new_hatWr_T.reshape(WXWXshape[1], WXWXshape[0])
                idxs = new_qidx.reshape(WXWXshape[1], WXWXshape[0] // args.V)
                b_hatWr_T[args.td_y * i:args.td_y * (i + 1)] = thing.T
                b_Qidxs_T[args.td_y // args.V * i:args.td_y // args.V *
                          (i + 1)] = idxs.T
                

                hatWr_T[args.td_y * (cur_col - buf_size):args.td_y *
                cur_col] = b_hatWr_T
            else:
                raise NotImplementedError
        hatWr_T[args.td_y * (cur_col - buf_size):args.td_y *
                cur_col] = b_hatWr_T
        
        # obj = calc_obj(hatWr_T, Wr_T, HRr)
        # print("cur_col", cur_col, "obj", obj)

    del b_Wr_T, b_hatWr_T
    utils.clean()
    return hatWr_T.T.contiguous(), Qidxs_T.T.contiguous()