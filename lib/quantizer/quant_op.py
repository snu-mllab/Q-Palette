import lib.utils as utils
from lib.quantizer.pack_op import pack_codes, pack_for_sq_pack_kernel
import torch
_PERMUTE = torch.arange(256).reshape(2, 8, 2, 4, 2).permute(1, 3, 2, 0,
                                                            4).flatten()
_INV_PERMUTE = torch.zeros(256, dtype=torch.int64)
_INV_PERMUTE[_PERMUTE] = torch.arange(256)

def random_mat(N, K):
    return torch.randn(N, K, dtype=torch.float16).cuda()

def random_lut(nbits, vec_sz):
    return torch.randn(1 << (nbits), vec_sz, dtype=torch.float16).cuda()

def vq_pack_reshape_pack_routine(packed_codes, warp_size, code_n, N):
    '''
        packed_codes: torch.Tensor, shape = (N, -1, code_n)
    '''
    packed_codes = packed_codes.reshape(N, -1, code_n)
    if packed_codes.shape[1] % warp_size == 0:
        packed_codes = packed_codes.reshape(N, -1, warp_size, code_n).permute(0, 1, 3, 2).reshape(N, -1)
    else:
        full_warp_part = packed_codes.shape[1] - packed_codes.shape[1] % warp_size
        packed_codes_full = packed_codes[:, :full_warp_part]
        packed_codes_partial = packed_codes[:, full_warp_part:]
        effective_warp_size = packed_codes_partial.shape[1] 
        packed_codes_full = packed_codes_full.reshape(N, -1, warp_size, code_n).permute(0, 1, 3, 2).reshape(N, -1)
        packed_codes_partial = packed_codes_partial.reshape(N, 1, effective_warp_size, code_n).permute(0, 1, 3, 2).reshape(N, -1)
        packed_codes = torch.cat([packed_codes_full, packed_codes_partial], dim=1)

    return packed_codes

def reshape_mat(mat, chunk_size, warp_size=32):
    '''
        mat: torch.Tensor, shape = (N, K)
        chunk_size: int

        return: reshaped_mat (N, K//chunk_size, chunk_size)
    '''
    N, K = mat.shape
    assert K % 8 == 0 and chunk_size % 8 == 0, "K and chunk_size must be divisible by 8"
    assert K % (chunk_size * warp_size) == 0, "K must be divisible by chunk_size * warp_size"
    K_iter = K // (chunk_size * warp_size)
    new_mat = mat.reshape(N, K_iter, chunk_size // 8, warp_size, 8).permute(0, 1, 3, 2, 4).reshape(N, K_iter, warp_size, chunk_size)
    return new_mat

def vq_pack_reshape_mat_routine(mat, chunk_size, vec_sz, warp_size=32):
    '''
        mat: torch.Tensor, shape = (N, K)
        chunk_size: int

        return: vecs
    '''
    N, K = mat.shape
    assert K % chunk_size == 0, "K must be divisible by chunk_size"
    if K % (chunk_size * warp_size) == 0:
        mat = reshape_mat(mat, chunk_size, warp_size)
        vecs = mat.reshape(-1, vec_sz)
    else:
        full_warp_part = K - K % (chunk_size * warp_size)
        mat_full = reshape_mat(mat[:, :full_warp_part], chunk_size, warp_size)
        effective_warp_size = (K % (chunk_size * warp_size)) // chunk_size
        mat_partial = reshape_mat(mat[:, full_warp_part:], chunk_size, effective_warp_size)

        vecs = torch.cat([mat_full.reshape(N, -1, vec_sz), mat_partial.reshape(N, -1, vec_sz)], dim=1).reshape(-1, vec_sz)

    return vecs

def pack_qweight_vq_simt(P, lut_bits, vec_sz, code_n, codeT_sz=32):
    '''
        P: (N, K // vec_sz) 
    '''
    N = P.shape[0]
    expanded_P = P.unsqueeze(-1).expand(-1, -1, vec_sz).reshape(N, -1)
    reshaped_P = vq_pack_reshape_mat_routine(expanded_P, chunk_size=int(32*vec_sz), vec_sz=vec_sz)[:, 0].contiguous()
    packed_codes = torch.from_numpy(pack_codes(reshaped_P.view(-1).cpu().numpy(), lut_bits, code_n, codeT_sz)).reshape(N, -1).cuda()
    packed_codes = vq_pack_reshape_pack_routine(packed_codes, 32, code_n, N)
    return packed_codes

def pack_qweight_sq_simt(P, lut_bits):
    '''
        P: (N, K) 
    '''
    N = P.shape[0]
    packed_codes = torch.from_numpy(pack_for_sq_pack_kernel(P.cpu().numpy(), lut_bits)).reshape(N, -1).cuda()
    return packed_codes

# for tensor core
def pack_qweight(P, vec_sz, lut_bits, td_x=16, td_y=16, batch_size=1024):
    '''
        P: (N, K // vec_sz, 2 ** lut_bits) 0, 1
    '''
    N = P.shape[0]
    mat_packed = []
    for i in range(0, N, batch_size):
        sidx, eidx = i, min(i + batch_size, N)
        cur_size = eidx - sidx
        mat_packed.append(pack_qweight_routine(P[sidx:eidx], vec_sz, lut_bits, td_x, td_y))
    return torch.cat(mat_packed, dim=0)

def pack_qweight_routine(P, vec_sz, lut_bits, td_x=16, td_y=16):
    '''
        P: (N, K // vec_sz, 2 ** lut_bits) 0, 1
    '''
    if vec_sz == 1:        
        if len(P.shape) == 3:
            P_ind = P.argmax(dim=-1) # (N, K)
        elif len(P.shape) == 2:
            P_ind = P
        N, K = P_ind.shape
        P_tiled = P_ind.reshape(N // td_x, td_x, K // td_y, td_y) \
                    .permute(0, 2, 1, 3) \
                    .reshape(-1, td_x * td_y)  
        P_tiled_permuted = P_tiled[..., _PERMUTE]
    elif vec_sz == 2:
        if len(P.shape) == 3:
            P_ind = P.argmax(dim=-1) # (N, K // vec_sz)
        else:
            P_ind = P
        P_ind = P_ind.unsqueeze(-1).expand(-1, -1, vec_sz)
        P_ind = P_ind.reshape(P_ind.shape[0], -1).contiguous()
        N, K = P_ind.shape
        P_tiled = P_ind.reshape(N // td_x, td_x, K // td_y, td_y) \
                    .permute(0, 2, 1, 3) \
                    .reshape(-1, td_x * td_y)  
        # permute and flatten
        P_tiled_permuted = P_tiled[..., _PERMUTE].contiguous().view(-1, vec_sz)
        assert torch.allclose(P_tiled_permuted[:, 0], P_tiled_permuted[:, 1]), \
            "P_tiled_permuted[:, 0] and P_tiled_permuted[:, 1] are not the same"
        P_tiled_permuted = P_tiled_permuted[:, 0].contiguous() 
        P_tiled_permuted = P_tiled_permuted.reshape((N * K) // (td_x * td_y), td_x * td_y // vec_sz)

    m = P_tiled_permuted.shape[0]  
    c = (td_x * td_y) // vec_sz    

    K_mask = 2 ** torch.arange(lut_bits, device=P.device).view(1, 1, -1)  # => [1,1,lut_bits]
    bits_bool = (P_tiled_permuted.unsqueeze(-1) & K_mask) > 0   # => [m, c, lut_bits]
    if vec_sz == 1:

        # group 4 bytes => 1 uint32
        # group 8 bits => 1 byte
        bits_bool_8 = bits_bool.reshape(m, (c * lut_bits) // 8, 8)  # => [m, c*lut_bits/8, 8]
        uint_mask = (2 ** torch.arange(8, device=P.device, dtype=torch.int16)).view(1, 1, 8)
        packed_8 = (bits_bool_8.to(torch.int16) * uint_mask).sum(dim=-1).to(torch.uint8)  # => [m, (c*lut_bits)//8]

        mat_packed = packed_8.reshape(N // td_x // 2, 2, K // td_y // 2, 2, td_x * td_y // 8, lut_bits) \
                    .permute(0, 2, 4, 3, 1, 5).contiguous().flatten().view(torch.uint32)\
                    .reshape((N * K) // (td_x * td_y), (td_x * td_y * lut_bits) // (32 * vec_sz))
    elif vec_sz == 2:
        # group 8 bits => 1 byte
        bits_bool_4 = bits_bool.reshape(m, (c * lut_bits) // 4, 4)  # => [m, c*nbits/8, 8]
        uint_mask = (2 ** torch.arange(4, device=bits_bool_4.device, dtype=torch.int16)).view(1, 1, 4)
        packed_4 = (bits_bool_4.to(torch.int16) * uint_mask).sum(dim=-1).to(torch.uint8)  # => [m, (c*nbits)//8]

        mat_packed_48 = packed_4.reshape(N // td_x // 2, 2, K // td_y // 2, 2, td_x * td_y // 8, lut_bits) \
                    .permute(0, 2, 4, 3, 1, 5).contiguous().flatten()
        # uint 4 packed in uint 8 to uint 32
        packing_mask = torch.Tensor([1, 2 ** 4]).to(torch.int8).view(1,2).cuda()
        mat_packed8 = (mat_packed_48.reshape(-1, 2) * packing_mask).sum(dim=-1).to(torch.uint8).contiguous().flatten()
        
        mat_packed = mat_packed8.view(torch.uint32).reshape((N * K) // (td_x * td_y), (td_x * td_y * lut_bits) // (32 * vec_sz))
    return mat_packed.view(N, -1)

def load_hessian(in_hess_path, sigma_reg=0.01):
    H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
    H = utils.flat_to_sym(H_data['flatH'], H_data['n'])
    if 'mu' in H_data:
        mu = H_data['mu']
        H += mu[None, :] * mu[:, None]
        del mu
    del H_data
    H = utils.regularize_H(H, sigma_reg)
    assert len(H.shape) == 2 and H.shape[0] == H.shape[1], "H must be a square matrix"
    return H.to(torch.float64).unsqueeze(-1)

def load_group_hessian(in_hess_path, sigma_reg=0.01, layer_key=None):
    H_data = torch.load(in_hess_path, map_location=torch.device('cpu'))
    H = H_data[layer_key]
    for i in range(H.shape[-1]):
        H[:, :, i] = utils.regularize_H(H[:, :, i], sigma_reg)
    assert len(H.shape) == 3 and H.shape[0] == H.shape[1], "H must be a square matrix"
    return H.to(torch.float64)

# deprecated func for dequantization
@torch.compile
def dequantize_mat_sq(mat_packed, lut, N, K, nbits, td_x=16, td_y=16):
    packed = mat_packed.flatten().view(torch.uint8).reshape(N // td_x // 2, 
                                                         K // td_y // 2,
                                                         td_x * td_y // 8,
                                                         2, 2, nbits)
    packed_8 = packed.permute(0, 4, 1, 3, 2, 5).contiguous().reshape(N * K // (td_x * td_y), (td_x * td_y) * nbits // 8)
    bits_mask = (2 ** torch.arange(8, device=mat_packed.device, dtype=torch.int16)).view(1, 1, 8)
    bits_bool_8 = (packed_8.unsqueeze(-1) & bits_mask) > 0
    bits_bool = bits_bool_8.reshape(N * K // (td_x * td_y), (td_x * td_y), nbits)
    K_mask = 2 ** torch.arange(nbits, device=mat_packed.device).view(1, 1, -1)
    indices = (bits_bool * K_mask).sum(dim=-1)
    recon = lut[indices.long()].reshape(N * K // (td_x * td_y), td_x * td_y)
    recon = recon.index_select(dim=1, index=_INV_PERMUTE.to(mat_packed.device))
    recon = recon.reshape(N // td_x, K // td_y, td_x, td_y)
    recon = recon.permute(0, 2, 1, 3).reshape(N, K)
    return recon

@torch.compile
def dequantize_mat_sq_inds(mat_packed, N, K, nbits, td_x=16, td_y=16):
    packed = mat_packed.flatten().view(torch.uint8).reshape(N // td_x // 2, 
                                                         K // td_y // 2,
                                                         td_x * td_y // 8,
                                                         2, 2, nbits)
    packed_8 = packed.permute(0, 4, 1, 3, 2, 5).contiguous().reshape(N * K // (td_x * td_y), (td_x * td_y) * nbits // 8)
    bits_mask = (2 ** torch.arange(8, device=mat_packed.device, dtype=torch.int16)).view(1, 1, 8)
    bits_bool_8 = (packed_8.unsqueeze(-1) & bits_mask) > 0
    bits_bool = bits_bool_8.reshape(N * K // (td_x * td_y), (td_x * td_y), nbits)
    K_mask = 2 ** torch.arange(nbits, device=mat_packed.device).view(1, 1, -1)
    indices = (bits_bool * K_mask).sum(dim=-1)
    indices = indices.reshape(N * K // (td_x * td_y), td_x * td_y).index_select(dim=1, index=_INV_PERMUTE.to(mat_packed.device))
    indices = indices.reshape(N // td_x, K // td_y, td_x, td_y)
    indices = indices.permute(0, 2, 1, 3).reshape(N, K).contiguous()
    return indices

@torch.compile
def dequantize_mat_sq_inds_vec2(mat_packed: torch.Tensor,
                                N: int,
                                K: int,
                                lut_bits: int,
                                td_x: int = 16,
                                td_y: int = 16) -> torch.Tensor:
    mat_packed8 = mat_packed.view(torch.uint8).flatten().unsqueeze(-1).expand(-1, 2).contiguous()

    mat_packed48 = torch.zeros_like(mat_packed8)
    mat_packed48[:, 0] = mat_packed8[:, 0] & 0b1111
    mat_packed48[:, 1] = mat_packed8[:, 1] >> 4

    packed_4 = mat_packed48.reshape(N // td_x // 2, K // td_y // 2, td_x * td_y // 8, 2, 2, lut_bits).permute(0,4,1,3,2,5).reshape(N * K // (td_x * td_y), -1).contiguous()
    bits_mask = (2 ** torch.arange(4, device=mat_packed.device, dtype=torch.int16)).view(1, 1, 4)
    bits_bool_4 = (packed_4.unsqueeze(-1) & bits_mask) > 0
    bits_bool = bits_bool_4.reshape(N * K // (td_x * td_y), td_x * td_y // 2, lut_bits)
    K_mask = 2 ** torch.arange(lut_bits, device=mat_packed.device).view(1, 1, -1)
    indices = (bits_bool * K_mask).sum(dim=-1)
    indices = indices.reshape(N * K // (td_x * td_y), td_x * td_y // 2, 1).expand(-1, -1, 2).reshape(N * K // (td_x * td_y), td_x * td_y).contiguous()
    indices = indices.index_select(dim=1, index=_INV_PERMUTE.to(mat_packed.device))
    indices = indices.reshape(N // td_x, K // td_y, td_x, td_y)
    indices = indices.permute(0, 2, 1, 3).reshape(N, K // 2, 2)
    indices = indices[:, :, 0].contiguous()
    return indices

def convert_tensor_core_to_simt(mat_packed, N, K, vec_sz, lut_bit, code_n, codeT_sz=32, td_x=16, td_y=16):
    device = mat_packed.device
    mat_packed = mat_packed.cuda()
    if vec_sz == 2:
        indices = dequantize_mat_sq_inds_vec2(mat_packed, N, K, lut_bit, td_x, td_y)
        packed_codes = pack_qweight_vq_simt(indices, lut_bit, vec_sz, code_n, codeT_sz)
        packed_codes = packed_codes.to(device)
    else:
        indices = dequantize_mat_sq_inds(mat_packed, N, K, lut_bit, td_x, td_y)
        packed_codes = pack_qweight_sq_simt(indices, lut_bit)
        packed_codes = packed_codes.to(device)
    return packed_codes.contiguous()



if __name__ == "__main__":
    nbits = 5
    Qidxs = torch.randint(0, 2**nbits, (11008, 4096)).cuda()
    packed = pack_qweight(Qidxs, 1, nbits)

    indices = dequantize_mat_sq_inds(packed, 11008, 4096, nbits)

    converted = convert_tensor_core_to_simt(packed, 11008, 4096, 1, nbits, code_n=nbits)

    Qidxs2 = torch.randint(0, 2**nbits, (11008, 2048)).cuda()
    packed2 = pack_qweight(Qidxs2, 2, nbits)

    indices2 = dequantize_mat_sq_inds_vec2(packed2, 11008, 4096, nbits)

    converted2 = convert_tensor_core_to_simt(packed2, 11008, 4096, 2, nbits, code_n=nbits)

    import ipdb; ipdb.set_trace()