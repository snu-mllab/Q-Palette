import numba
import numpy as np
@numba.njit(cache=True)
def general_pack(unpacked, nbits, codeT, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            unpacked: np.int (n_unpacked,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype

        return: out_code (code_n,) dtype: codeT
    '''
    n_unpacked = unpacked.shape[0]
    codeT_sz = codeT.itemsize * 8
    assert n_unpacked * nbits / codeT_sz == code_n, "code_n must be equal to n_unpacked * nbits / codeT_sz"
    out_code = np.zeros(code_n, dtype=codeT)
    for i in range(n_unpacked):
        val = codeT(unpacked[i])
        offset = i * nbits
        wIndex = offset // codeT_sz
        bIndex = offset % codeT_sz
        out_code[wIndex] |= (val << bIndex) & np.iinfo(codeT).max

        bits_in_word = codeT_sz - bIndex
        if bits_in_word < nbits:
            upper = val >> bits_in_word
            out_code[wIndex + 1] |= upper & np.iinfo(codeT).max

    return out_code

@numba.njit(cache=True)
def general_pack_8(unpacked, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            unpacked: np.int (n_unpacked,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype

        return: out_code (code_n,) dtype: codeT
    '''
    n_unpacked = unpacked.shape[0]
    assert n_unpacked * nbits / 8 == code_n, "code_n must be equal to n_unpacked * nbits / 8"
    out_code = np.zeros(code_n, dtype=np.uint8)
    for i in range(n_unpacked):
        val = unpacked[i]
        offset = i * nbits
        wIndex = offset // 8
        bIndex = offset % 8
        out_code[wIndex] |= (val << bIndex) & np.iinfo(np.uint8).max

        bits_in_word = 8 - bIndex
        if bits_in_word < nbits:
            upper = val >> bits_in_word
            out_code[wIndex + 1] |= upper & np.iinfo(np.uint8).max
    return out_code

@numba.njit(cache=True)
def general_pack_16(unpacked, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            unpacked: np.int (n_unpacked,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype

        return: out_code (code_n,) dtype: codeT
    '''
    n_unpacked = unpacked.shape[0]
    assert n_unpacked * nbits / 16 == code_n, "code_n must be equal to n_unpacked * nbits / 16"
    out_code = np.zeros(code_n, dtype=np.uint16)
    for i in range(n_unpacked):
        val = unpacked[i]
        offset = i * nbits
        wIndex = offset // 16
        bIndex = offset % 16
        out_code[wIndex] |= (val << bIndex) & np.iinfo(np.uint16).max

        bits_in_word = 16 - bIndex
        if bits_in_word < nbits:
            upper = val >> bits_in_word
            out_code[wIndex + 1] |= upper & np.iinfo(np.uint16).max

    return out_code


@numba.njit(cache=True)
def general_pack_32(unpacked, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            unpacked: np.int (n_unpacked,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype

        return: out_code (code_n,) dtype: codeT
    '''
    n_unpacked = unpacked.shape[0]
    assert n_unpacked * nbits / 32 == code_n, "code_n must be equal to n_unpacked * nbits / 32"
    out_code = np.zeros(code_n, dtype=np.uint32)
    for i in range(n_unpacked):
        val = unpacked[i]
        offset = i * nbits
        wIndex = offset // 32
        bIndex = offset % 32
        out_code[wIndex] |= (val << bIndex) & np.iinfo(np.uint32).max

        bits_in_word = 32 - bIndex
        if bits_in_word < nbits:
            upper = val >> bits_in_word
            out_code[wIndex + 1] |= upper & np.iinfo(np.uint32).max

    return out_code

@numba.njit(cache=True)
def general_pack_64(unpacked, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            unpacked: np.int (n_unpacked,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype

        return: out_code (code_n,) dtype: codeT
    '''
    n_unpacked = unpacked.shape[0]
    assert n_unpacked * nbits / 64 == code_n, "code_n must be equal to n_unpacked * nbits / 64"
    out_code = np.zeros(code_n, dtype=np.uint64)
    for i in range(n_unpacked):
        val = unpacked[i]
        offset = i * nbits
        wIndex = offset // 64
        bIndex = offset % 64
        out_code[wIndex] |= (val << bIndex) & np.iinfo(np.uint64).max

        bits_in_word = 64 - bIndex
        if bits_in_word < nbits:
            upper = val >> bits_in_word
            out_code[wIndex + 1] |= upper & np.iinfo(np.uint64).max

    return out_code

@numba.njit(cache=True)
def pack_codes_8(codes, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            codes: np.int (n_samples,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype (uint64 or uint32 or uint16 or uint8)
            code_n: int

        return: 
            packed_codes (-1, code_n) dtype: codeT
    '''
    n_samples = codes.shape[0]
    n_unpacked = code_n * 8 // nbits
    packed_codes = np.zeros((n_samples // n_unpacked, code_n), dtype=np.uint8)
    for i in range(n_samples // n_unpacked):
        unpacked = codes[i * n_unpacked: (i + 1) * n_unpacked]
        packed_codes[i] = general_pack_8(unpacked, nbits, code_n)
    return packed_codes
    
@numba.njit(cache=True)
def pack_codes_16(codes, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            codes: np.int (n_samples,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype (uint64 or uint32 or uint16 or uint8)
            code_n: int

        return: 
            packed_codes (-1, code_n) dtype: codeT
    '''
    n_samples = codes.shape[0]
    n_unpacked = code_n * 16 // nbits
    packed_codes = np.zeros((n_samples // n_unpacked, code_n), dtype=np.uint16)
    for i in range(n_samples // n_unpacked):
        unpacked = codes[i * n_unpacked: (i + 1) * n_unpacked]
        packed_codes[i] = general_pack_16(unpacked, nbits, code_n)
    return packed_codes

@numba.njit(cache=True)
def pack_codes_32(codes, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            codes: np.int (n_samples,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype (uint64 or uint32 or uint16 or uint8)
            code_n: int

        return: 
            packed_codes (-1, code_n) dtype: codeT
    '''
    n_samples = codes.shape[0]
    n_unpacked = code_n * 32 // nbits
    packed_codes = np.zeros((n_samples // n_unpacked, code_n), dtype=np.uint32)
    for i in range(n_samples // n_unpacked):
        unpacked = codes[i * n_unpacked: (i + 1) * n_unpacked]
        packed_codes[i] = general_pack_32(unpacked, nbits, code_n)
    return packed_codes

@numba.njit(cache=True)
def pack_codes_64(codes, nbits, code_n):
    '''
        sequentially packing codes into codeT type array.
        args:
            codes: np.int (n_samples,) each entry is 0 .. 2 ** nbits - 1
            nbits: int
            codeT: np.dtype (uint64 or uint32 or uint16 or uint8)
            code_n: int

        return: 
            packed_codes (-1, code_n) dtype: codeT
    '''
    n_samples = codes.shape[0]
    n_unpacked = code_n * 64 // nbits
    packed_codes = np.zeros((int(n_samples // n_unpacked), code_n), dtype=np.uint64)
    for i in range(n_samples // n_unpacked):
        unpacked = codes[i * n_unpacked: (i + 1) * n_unpacked]
        packed_codes[i] = general_pack_64(unpacked, nbits, code_n)
    return packed_codes

def pack_codes(codes, nbits, code_n, codeT_sz):
    if codeT_sz == 8:
        return pack_codes_8(codes, nbits, code_n)
    elif codeT_sz == 16:
        return pack_codes_16(codes, nbits, code_n)
    elif codeT_sz == 32:
        return pack_codes_32(codes, nbits, code_n)
    elif codeT_sz == 64:
        return pack_codes_64(codes, nbits, code_n)
    else:
        raise ValueError(f"Unsupported codeT_sz: {codeT_sz}")
    


@numba.njit(cache=True)
def pack_32(cluster_idx: np.ndarray, nbits: int) -> np.ndarray:
    """
    NumPy 버전의 pack_32 함수.

    Parameters
    ----------
    cluster_idx : np.ndarray of shape (32,), dtype=int
        길이 32의 정수 배열. (C 코드에서 const int* cluster_idx와 동일 역할)
    nbits : int
        각 정수를 몇 비트로 저장할지.

    Returns
    -------
    out_code : np.ndarray of shape (out_size,), dtype=np.uint32
        32개의 값(각각 nbits 비트)으로 구성된 연속 비트열을
        32비트 워드(uint32) 단위로 나눈 결과.
    """

    # 32개의 값을 nbits비트씩 사용하면 총 32*nbits 비트가 필요.
    # 이를 32비트 단위로 나누면 아래처럼 워드 수가 결정됨.
    out_size = (32 * nbits + 31) // 32  # 올림
    
    # 결과 버퍼 (np.uint32로)
    out_code = np.zeros(out_size, dtype=np.uint32)

    for i in range(32):
        # cluster_idx[i]를 unsigned 처리
        val = np.uint32(cluster_idx[i])

        offset = i * nbits
        wIndex = offset // 32  # 몇 번째 워드인지
        bIndex = offset % 32   # 그 워드 내에서 몇 번째 비트부터 시작?

        # 첫 번째 워드에 bIndex부터 nbits비트 중 일부 혹은 전부를 저장
        out_code[wIndex] |= (val << bIndex) & np.uint32(0xFFFFFFFF)

        # 현재 워드에 다 못 들어가는 나머지 비트가 있으면, 다음 워드에 저장
        bits_in_word = 32 - bIndex
        if bits_in_word < nbits:
            upper = val >> bits_in_word
            out_code[wIndex + 1] |= upper & np.uint32(0xFFFFFFFF)

    return out_code

@numba.njit(cache=True)
def pack_for_sq_pack_kernel(unpacked_code: np.ndarray,
                            nbits: int,
                            blockDimX: int=32) -> np.ndarray:
    """
    Python 버전의 pack_for_sq_pack_kernel.
    unpacked_code: shape = (N, K), dtype=uint32
    nbits: int
    blockDimX: int
    return: Bcode (1D np.uint32 배열, 길이 = N * (K*nbits//32))
    """

    N, K = unpacked_code.shape
    out_size = N * (K * nbits // 32)
    Bcode = np.zeros(out_size, dtype=np.uint32)

    K_iter = int(np.ceil(K / (32 * blockDimX)))

    # 임시 버퍼
    unpacked_Bcode_row = np.zeros(32, dtype=np.uint32)

    for n_ in range(N):
        eff_warp_size = blockDimX
        for k_ in range(K_iter):
            for thx in range(blockDimX):
                if k_ == K // (32 * blockDimX):
                    eff_warp_size = (K % (32 * blockDimX)) // 32
                    if thx >= eff_warp_size:
                        break
                k_val  = k_ * 32 * blockDimX + 8 * thx
                k_code = k_ * nbits * blockDimX + thx

                # unpacked_Bcode_row에 32개 로드
                idx_out = 0
                for j in range(4):
                    k_val_idx = k_val + 8 * j * eff_warp_size
                    for i in range(8):
                        unpacked_Bcode_row[idx_out] = unpacked_code[n_, k_val_idx + i]
                        idx_out += 1

                # pack_32 호출
                Bcode_row = pack_32(unpacked_Bcode_row, nbits)

                # Bcode에 저장
                for j in range(nbits):
                    k_code_idx = k_code + j * eff_warp_size
                    Bcode[n_ * (K * nbits // 32) + k_code_idx] = Bcode_row[j]

    return Bcode
