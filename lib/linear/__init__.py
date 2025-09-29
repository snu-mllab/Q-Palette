from .quantized_linear import QuantizedLinear
from .incoherent_linear import IncoherentLinear
from .vq_linear import VQLinearPackSIMT, VQLinearPackTensorCore
from .tcq_linear import QTIPLinearTCQ
from .comb_linear import CombLinearTCQ, CombtLinearTCQ
import vq_tensor_kernels
import torch

kernels = [
  (53248, 16384),
  (16384, 53248),
  (1024, 16384),
  (16384, 16384),
  (4096, 14336),
  (14336, 4096),
  (28672, 4096),
  (2048, 4096),
  (5120, 4096),
  (6144, 4096),
  (1024, 4096),
  (4096, 4096),
  (4096, 11008),
  (11008, 4096),
  (22016, 4096),
  (12288, 4096),
  (8192, 4096),
  (8192, 8192),
  (10240, 8192),
  (57344, 8192),
  (8192, 1024),
  (8192, 28672),
  (28672, 8192),
  (1024, 8192),
  (5120, 5120),
  (10240, 5120),
  (15360, 5120),
  (13568, 5120),
  (27136, 5120),
  (5120, 13568),
]

kdict = {}
for vtype, max_bits, bit_stride in [
    ('sq_dup', 4, 1),
    ('sq', 8, 1),
    ('vq2', 12, 1),
]:
    for m, k in kernels:
        for n in [1,2,4,8]:
            for bitrate in range(2,max_bits+1, bit_stride):

                torch.library.define(
                    f"ours_lib::decompress_gemm_{m}_{n}_{k}_{bitrate}_{vtype}",
                    "(Tensor compressed, Tensor x, Tensor codebook) -> Tensor")

                name = f"decompress_gemm_{m}_{n}_{k}_{bitrate}_{vtype}"
                kernel_name = f"vq_tensor_kernels.decompress_gemm_{bitrate}_{m}_{n}_{k}_{vtype}"
                exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    return torch.zeros({n}, {m}, dtype=torch.float32, device=x.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({n}, {m}), dtype=torch.float32, device=x.device)
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), x.to(torch.float16), codebook.reshape(-1))
    return out
            """)

        for bitrate in range(2,max_bits+1, bit_stride):
            name = f"decompress_gemv_{m}_{k}_{bitrate}_{vtype}"
            kernel_name = f"vq_tensor_kernels.decompress_gemm_{bitrate}_{m}_{1}_{k}_{vtype}"
            exec(f"""\
@torch.library.custom_op("ours_lib::{name}", mutates_args={{"out"}})
def {name}(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor,
        out: torch.Tensor) -> torch.Tensor:
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), x.to(torch.float16), codebook.reshape(-1))
@{name}.register_fake
def {name}_fake(compressed, x, codebook, out):
    return None
            """)

    for bitrate in range(2,max_bits+1, bit_stride):
        torch.library.define(
                f"ours_lib::decompress_{bitrate}_{vtype}",
                "(Tensor compressed, Tensor codebook, int m, int k) -> Tensor")

        name = f"decompress_{bitrate}_{vtype}"
        kernel_name = f"vq_tensor_kernels.decompress_{bitrate}_{vtype}"
        exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        codebook: torch.Tensor,
        m: int,
        k: int) -> torch.Tensor:
    return torch.zeros(m, k, dtype=torch.float16, device=compresed.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        codebook: torch.Tensor,
        m: int,
        k: int) -> torch.Tensor:
    out = torch.zeros((m, k), dtype=torch.float16, device=compressed.device)
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), codebook.reshape(-1))
    return out
            """)
        
import tcq_kernels
import torch
MKSHAPE = [
  (53248, 16384),
  (16384, 53248),
  (1024, 16384),
  (16384, 16384),
  (4096, 14336),
  (14336, 4096),
  (28672, 4096),
  (5120, 4096),
  (6144, 4096),
  (512, 4096),
  (1024, 4096),
  (2048, 4096),
  (4096, 4096),
  (2048, 11008),
  (4096, 11008),
  (5504, 4096),
  (11008, 4096),
  (22016, 4096),
  (12288, 4096),
  (8192, 4096),
  (8192, 8192),
  (10240, 8192),
  (57344, 8192),
  (8192, 1024),
  (8192, 28672),
  (28672, 8192),
  (1024, 8192),
  (5120, 5120),
  (10240, 5120),
  (15360, 5120),
  (13568, 5120),
  (27136, 5120),
  (5120, 13568),
  (3072, 3072),
  (1024, 3072),
  (4096, 3072),
  (2048, 3072),
  (5120, 3072),
  (8192, 3072),
  (16384, 3072),
  (3072, 8192),
]

kdict = {}
for S in [9, 10, 11]:
    if S == 9:
        bitrate_list = [2,3,4,5,6,7,8,9,10]
    elif S == 10:
        bitrate_list = [8, 9, 10]
    elif S == 11:
        bitrate_list = [9, 10]
    for m, k in MKSHAPE:
        for n in [1,2,4,8]:
            for bitrate in bitrate_list:
                torch.library.define(
            f"ours_lib::decompress_gemm_tcq_{m}_{n}_{k}_{S}_{bitrate}",
            "(Tensor compressed, Tensor x, Tensor codebook) -> Tensor")

                name = f"decompress_gemm_tcq_{m}_{n}_{k}_{S}_{bitrate}"
                kernel_name = f"tcq_kernels.decompress_gemm_16_{S}_{bitrate}_1_{m}_{n}_{k}"
                exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    return torch.zeros({n}, {m}, dtype=torch.float32, device=x.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({n}, {m}), dtype=torch.float32, device=x.device)
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), x.to(torch.float16), codebook.reshape(-1))
    return out
    """)
                if bitrate == bitrate_list[-1]:
                    continue
                torch.library.define(
            f"ours_lib::decompress_gemm_tcq_comb_{m}_{n}_{k}_{S}_{bitrate}_{int(bitrate+1)}",
            "(Tensor compressed1, Tensor compressed2, Tensor x, Tensor codebook) -> Tensor")

                name = f"decompress_gemm_tcq_comb_{m}_{n}_{k}_{S}_{bitrate}_{int(bitrate+1)}"
                kernel_name = f"tcq_kernels.decompress_gemm_comb_16_{S}_{bitrate}_{int(bitrate+1)}_1_{m}_{n}_{k}"
                exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
        compressed1: torch.Tensor,
        compressed2: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    return torch.zeros({n}, {m}, dtype=torch.float32, device=x.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
        compressed1: torch.Tensor,
        compressed2: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({n}, {m}), dtype=torch.float32, device=x.device)
    {kernel_name}(out, compressed1.reshape(-1).view(torch.int32), compressed2.reshape(-1).view(torch.int32), x.to(torch.float16), codebook.reshape(-1))
    return out
    """)
                torch.library.define(
            f"ours_lib::decompress_gemm_tcq_combt_{m}_{n}_{k}_{S}_{bitrate}_{int(bitrate+1)}",
            "(Tensor compressed1, Tensor compressed2, Tensor x, Tensor codebook) -> Tensor")

                name = f"decompress_gemm_tcq_combt_{m}_{n}_{k}_{S}_{bitrate}_{int(bitrate+1)}"
                kernel_name = f"tcq_kernels.decompress_gemm_combt_16_{S}_{bitrate}_{int(bitrate+1)}_1_{m}_{n}_{k}"
                exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
        compressed1: torch.Tensor,
        compressed2: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    return torch.zeros({n}, {m}, dtype=torch.float32, device=x.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
        compressed1: torch.Tensor,
        compressed2: torch.Tensor,
        x: torch.Tensor,
        codebook: torch.Tensor) -> torch.Tensor:
    out = torch.zeros(({n}, {m}), dtype=torch.float32, device=x.device)
    {kernel_name}(out, compressed1.reshape(-1).view(torch.int32), compressed2.reshape(-1).view(torch.int32), x.to(torch.float16), codebook.reshape(-1))
    return out
    """)
    
for S in [9, 10, 11]:
    if S == 9:
        bitrate_list = [2,3,4,5,6,7,8,9,10]
    elif S == 10:
        bitrate_list = [8, 9, 10]
    elif S == 11:
        bitrate_list = [9, 10]
    for bitrate in bitrate_list:
        torch.library.define(
        f"ours_lib::decompress_tcq_{S}_{bitrate}",
        "(Tensor compressed, Tensor codebook, int m, int k) -> Tensor")

        name = f"decompress_tcq_{S}_{bitrate}"
        kernel_name = f"tcq_kernels.decompress_16_{S}_{bitrate}"
        exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
        compressed: torch.Tensor,
        codebook: torch.Tensor,
        m: int,
        k: int) -> torch.Tensor:
    return torch.zeros(m, k, dtype=torch.float16, device=compresed.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
        compressed: torch.Tensor,
        codebook: torch.Tensor,
        m: int,
        k: int) -> torch.Tensor:
    out = torch.zeros((m, k), dtype=torch.float16, device=compressed.device)
    {kernel_name}(out, compressed.reshape(-1).view(torch.int32), codebook.reshape(-1))
    return out
        """)
        if bitrate == bitrate_list[-1]:
            break

        torch.library.define(
        f"ours_lib::decompress_tcq_comb_{S}_{bitrate}_{int(bitrate+1)}",
        "(Tensor compressed1, Tensor compressed2, Tensor codebook, int m, int k) -> Tensor")

        name = f"decompress_tcq_comb_{S}_{bitrate}_{int(bitrate+1)}"
        kernel_name = f"tcq_kernels.decompress_comb_16_{S}_{bitrate}_{int(bitrate+1)}"
        exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
    compressed1: torch.Tensor,
    compressed2: torch.Tensor,
    codebook: torch.Tensor,
    m: int, k: int) -> torch.Tensor:
    return torch.zeros(m, k, dtype=torch.float16, device=compressed1.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
    compressed1: torch.Tensor,
    compressed2: torch.Tensor,
    codebook: torch.Tensor,
    m: int, k: int) -> torch.Tensor:
    out = torch.zeros((m, k), dtype=torch.float16, device=compressed1.device)
    {kernel_name}(out, compressed1.reshape(-1).view(torch.int32), compressed2.reshape(-1).view(torch.int32), codebook.reshape(-1))
    return out
    """)
        torch.library.define(
        f"ours_lib::decompress_tcq_combt_{S}_{bitrate}_{int(bitrate+1)}",
        "(Tensor compressed1, Tensor compressed2, Tensor codebook, int m, int k) -> Tensor")

        name = f"decompress_tcq_combt_{S}_{bitrate}_{int(bitrate+1)}"
        kernel_name = f"tcq_kernels.decompress_combt_16_{S}_{bitrate}_{int(bitrate+1)}"
        exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(
    compressed1: torch.Tensor,
    compressed2: torch.Tensor,
    codebook: torch.Tensor,
    m: int, k: int) -> torch.Tensor:
    return torch.zeros(m, k, dtype=torch.float16, device=compressed1.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(
    compressed1: torch.Tensor,
    compressed2: torch.Tensor,
    codebook: torch.Tensor,
    m: int, k: int) -> torch.Tensor:
    out = torch.zeros((m, k), dtype=torch.float16, device=compressed1.device)
    {kernel_name}(out, compressed1.reshape(-1).view(torch.int32), compressed2.reshape(-1).view(torch.int32), codebook.reshape(-1))
    return out
    """)






import sq_pack_gemm
import vq_pack_gemm
"""
SQ Pack SIMT
"""
torch.library.define("ours_lib::sq_pack_gemm_simt", "(Tensor x, Tensor q_weight, Tensor lut, int bitwidth) -> Tensor")
@torch.library.register_fake("ours_lib::sq_pack_gemm_simt")
def sq_pack_gemm_simt_abstract(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, bitwidth:int) -> torch.Tensor:
    return torch.zeros(x.shape[0], 1, q_weight.shape[0], dtype=torch.float16, device=x.device)

@torch.library.impl("ours_lib::sq_pack_gemm_simt", "cuda")
def sq_pack_gemm_simt_cuda(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, bitwidth:int) -> torch.Tensor:
    output = torch.zeros(x.shape[0], 1, q_weight.shape[0], dtype=torch.float16, device=x.device)
    sq_pack_gemm.pack_gemm(x, output, q_weight, lut.view(-1), bitwidth)
    return output

torch.library.define("ours_lib::sq_pack_dequant_simt", "(Tensor q_weight, Tensor lut, int bitwidth, int m, int k) -> Tensor")
@torch.library.register_fake("ours_lib::sq_pack_dequant_simt")
def sq_pack_dequant_simt_abstract(q_weight: torch.Tensor, lut: torch.Tensor, bitwidth:int, m:int, k:int) -> torch.Tensor:
    return torch.zeros(m, k, dtype=torch.float16, device=q_weight.device)

@torch.library.impl("ours_lib::sq_pack_dequant_simt", "cuda")
def sq_pack_dequant_simt_cuda(q_weight: torch.Tensor, lut: torch.Tensor, bitwidth:int, m:int, k:int) -> torch.Tensor:
    output = torch.zeros(m, k, dtype=torch.float16, device=q_weight.device)
    sq_pack_gemm.pack_dequant(output, q_weight, lut.view(-1), bitwidth)
    return output


@torch.library.custom_op("ours_lib::sq_pack_gemm_inplace_simt", mutates_args={"output"})
def sq_pack_gemm_inplace_simt(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor, output:torch.Tensor, bitwidth:int) -> None:
    sq_pack_gemm.pack_gemm(x, output, q_weight, lut, bitwidth)

@sq_pack_gemm_inplace_simt.register_fake
def _(x, q_weight, lut, output, bitwidth):
    return None

"""
VQ Pack SIMT
"""
codeT_sz = 32
for vec_sz in [2,4]:
    if vec_sz == 2:
        lut_bits_list = [3,4,5,6,7,8,9,10,11,12]
    elif vec_sz == 4:
        lut_bits_list = [6,7,8,9,10,11,12]
    for lut_bits in lut_bits_list:
        code_n = lut_bits
        recons_n = int(vec_sz * 16)
        for maxm in [1,2,4,8]:
            name = f"vq_pack_gemm_simt_{maxm}_{vec_sz}_{lut_bits}"
            kernel_name = f"vq_pack_gemm.vq_pack_gemm_{maxm}_{lut_bits}_{vec_sz}_{code_n}_{codeT_sz}_{recons_n}"
            torch.library.define(f"ours_lib::{name}", "(Tensor x, Tensor q_weight, Tensor lut) -> Tensor")
            exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    return torch.zeros(x.shape[0], 1, q_weight.shape[0], dtype=torch.float16, device=x.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(x: torch.Tensor, q_weight: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    output = torch.zeros(x.shape[0], 1, q_weight.shape[0], dtype=torch.float16, device=x.device)
    {kernel_name}(x, output, q_weight.view(torch.uint32), lut)
    return output
    """)
        name = f"vq_pack_dequant_simt_{vec_sz}_{lut_bits}"
        kernel_name = f"vq_pack_gemm.vq_pack_dequant_{lut_bits}_{vec_sz}_{code_n}_{codeT_sz}_{recons_n}"
        torch.library.define(f"ours_lib::{name}", "(Tensor q_weight, Tensor lut, int m, int k) -> Tensor")
        exec(f"""\
@torch.library.register_fake("ours_lib::{name}")
def {name}_abstract(q_weight: torch.Tensor, lut: torch.Tensor, m: int, k: int) -> torch.Tensor:
    return torch.zeros(m, k, dtype=torch.float16, device=q_weight.device)

@torch.library.impl("ours_lib::{name}", "cuda")
def {name}_cuda(q_weight: torch.Tensor, lut: torch.Tensor, m: int, k: int) -> torch.Tensor:
    output = torch.zeros(m, k, dtype=torch.float16, device=q_weight.device)
    {kernel_name}(output, q_weight.view(torch.uint32), lut)
    return output
    """)


if __name__ == "__main__":
    # layer = QTIPLinearTCQ(4096, 4096, 16, 16, 16, 8, 1, 9, False, torch.float16)
    # print(layer._info())
    # x = torch.randn(4, 4096)
    # print(layer(x).shape)
    layer = CombLinearTCQ(4096, 4096, 16, 16, (2048, 2048), 16, (3, 4), 2, 9, False)
    print(layer._info())
    layer.forward(torch.randn(1, 4096).cuda())
