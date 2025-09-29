MKSHAPE = [
#   (256, 256),
#   (53248, 16384),
#   (16384, 53248),
#   (1024, 16384),
#   (16384, 16384),
#   (4096, 14336),
#   (14336, 4096),
#   (1024, 4096),
#   (4096, 4096),
#   (4096, 11008),
#   (11008, 4096),
#   (12288, 4096),
#   (22016, 4096),
#   (8192, 8192),
#   (256, 256),
  (256, 256),
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
  (5120, 13824),
  (13824, 5120),
  (27648, 5120),
  (10240, 5120),
  (15360, 5120),
  (13568, 5120),
  (27136, 5120),
  (5120, 13568),
]

max_batch = 8
# qmode PER_TENSOR, PER_COLUMN, PER_GROUP
# luttype SQ_LUT, VQ_LUT_2, VQ_LUT_4, VQ_LUT_8
qmode_lut_ld_maxbits_list = [
    (0, 0, 0, 4),
    (0, 0, 1, 8),
    # (1, 0, 0, 7),
    # (2, 0, 0, 7),
    # (0, 0, 1, 8),
    # (0, 1, 1, 14),
    # (0, 2, 1, 12),
    # (0, 3, 1, 12),
    (0, 1, 1, 12),
    # (0, 2, 1, 12),
    # (2, 0, 1, 8),
    # (2, 1, 1, 14),
    # (2, 2, 1, 12),
    # (2, 3, 1, 12),
]
Q_MODES = ["Q_MODE::PER_TENSOR", "Q_MODE::PER_COLUMN", "Q_MODE::PER_GROUP"]
LUT_TYPES = ["LUT_TYPE::SQ_LUT", "LUT_TYPE::VQ_LUT_2", "LUT_TYPE::VQ_LUT_4", "LUT_TYPE::VQ_LUT_8"]
LOAD_TYPES = ["LOAD_TYPE::DUP", "LOAD_TYPE::ONCE"]

def mdef_gen(bitrate, m, n, k, q, l, ld):
    qmode = f"{"_pc" if q == 1 else "" if q==0 else "_g"}"
    lut_type = f"{"_sq" if l == 0 else "_vq2" if l==1 else "_vq4" if l==2 else "_vq8"}"
    load_type = f"{"_dup" if ld == 0 else ""}"
    return f'm.def("decompress_gemm_{bitrate}_{m}_{n}_{k}{qmode}{lut_type}{load_type}", &decompress_gemm_{bitrate}_{m}_{n}_{k}{qmode}{lut_type}{load_type}, "decompress_gemm_{bitrate}_{m}_{n}_{k}{qmode}{lut_type}{load_type}");\n'

def mdef_gen_dq(bitrate, q, l, ld):
    qmode = f"{"_pc" if q == 1 else "" if q==0 else "_g"}"
    lut_type = f"{"_sq" if l == 0 else "_vq2" if l==1 else "_vq4" if l==2 else "_vq8"}"
    load_type = f"{"_dup" if ld == 0 else ""}"
    return f'm.def("decompress_{bitrate}{qmode}{lut_type}{load_type}", &decompress_{bitrate}{qmode}{lut_type}{load_type}, "decompress_{bitrate}{qmode}{lut_type}{load_type}");\n'

def func_gen(bitrate, m, n, k, q, l, ld):
    qmode = f"{"_pc" if q == 1 else "" if q==0 else "_g"}"
    lut_type = f"{"_sq" if l == 0 else "_vq2" if l==1 else "_vq4" if l==2 else "_vq8"}"
    load_type = f"{"_dup" if ld == 0 else ""}"
    str_ = f"""
void decompress_gemm_{bitrate}_{m}_{n}_{k}{qmode}{lut_type}{load_type}(
                    torch::Tensor &out, 
                    torch::Tensor &compressed,
                    torch::Tensor &x,
                    torch::Tensor &codebook
                    );
"""
    return str_

def func_gen_dq(bitrate, q, l, ld):
    qmode = f"{"_pc" if q == 1 else "" if q==0 else "_g"}"
    lut_type = f"{"_sq" if l == 0 else "_vq2" if l==1 else "_vq4" if l==2 else "_vq8"}"
    load_type = f"{"_dup" if ld == 0 else ""}"
    str_ = f"""
void decompress_{bitrate}{qmode}{lut_type}{load_type}(
                    torch::Tensor &out, 
                    torch::Tensor &compressed,
                    torch::Tensor &codebook
                    );
"""
    return str_

def header_gen():
    return "#include <torch/extension.h>\n"

str_ = header_gen()
for q, l, ld, max_bits in qmode_lut_ld_maxbits_list:
    stride = 2 if l == 2 else 1
    for bitrate in range(2, max_bits+1, stride):
        str_ += func_gen_dq(bitrate, q, l, ld)
        for m, k in MKSHAPE:
            for n in [1,2,4,8]:
                str_ += func_gen(bitrate, m, n, k, q, l, ld)


str_ += "PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n"
for q, l, ld, max_bits in qmode_lut_ld_maxbits_list:
    stride = 2 if l == 2 else 1
    for bitrate in range(2, max_bits+1, stride):
        str_ += mdef_gen_dq(bitrate, q, l, ld)
        for m, k in MKSHAPE:
            for n in [1,2,4,8]:
                str_ += mdef_gen(bitrate, m, n, k, q, l, ld)

str_ += "}\n"

print(str_)

with open("wrapper.cpp", "w") as f:
    f.write(str_)


def qtip_torch_header_gen():
    str_ = f"""
#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/cuda/Atomic.cuh>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>

#include <torch/types.h>
#include <torch/extension.h>
#include "inference.cu"

using namespace torch::indexing;

template <uint32_t R, uint32_t M, uint32_t N, uint32_t K, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__host__ static void decompress_gemm(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {{
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.scalar_type() == torch::kFloat32);

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 1);
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    CHECK_INPUT(x);
    TORCH_CHECK(x.dim() == 2);
    TORCH_CHECK(x.scalar_type() == torch::kFloat16);

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    size_t m = out.size(1);
    size_t n = out.size(0);
    size_t k = x.size(1);

    TORCH_CHECK(m == M);
    TORCH_CHECK(n == N);
    TORCH_CHECK(k == K);
    TORCH_CHECK(compressed.numel() * 32 * (1 << ((int)L)) == R * m * k);
    TORCH_CHECK(x.size(0) == n);
    if constexpr((Q == Q_MODE::PER_TENSOR)) {{
        TORCH_CHECK(codebook.size(0) == (1<<R) * (1 << ((int)L)));
    }} else if constexpr(Q == Q_MODE::PER_COLUMN) {{
        TORCH_CHECK(codebook.size(0) == (1<<R) * m * (1 << ((int)L)));
    }}

    at::DeviceGuard guard(x.device());
    
    decompress_gemm_ptr<R, M, N, K, Q, L, LD>(
            reinterpret_cast<float *>(out.data_ptr<float>()),
            reinterpret_cast<const uint32_t *>(compressed.data_ptr<int32_t>()),
            reinterpret_cast<const half2 *>(x.data_ptr<c10::Half>()),
            reinterpret_cast<const half *>(codebook.data_ptr<c10::Half>()),
            at::cuda::getCurrentCUDAStream()
    );
}}

template <uint32_t R, Q_MODE Q, LUT_TYPE L, LOAD_TYPE LD>
__host__ static void decompress(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
){{
    CHECK_INPUT(out);
    TORCH_CHECK(out.dim() == 2);
    TORCH_CHECK(out.scalar_type() == torch::kFloat16);

    CHECK_INPUT(compressed);
    TORCH_CHECK(compressed.dim() == 1);
    TORCH_CHECK(compressed.scalar_type() == torch::kInt32);

    CHECK_INPUT(codebook);
    TORCH_CHECK(codebook.dim() == 1);
    TORCH_CHECK(codebook.scalar_type() == torch::kFloat16);

    size_t m = out.size(0);
    size_t k = out.size(1);

    TORCH_CHECK(compressed.numel() * 32 * (1 << ((int)L)) == R * m * k);
    if constexpr((Q == Q_MODE::PER_TENSOR)) {{
        TORCH_CHECK(codebook.size(0) == (1<<R) * (1 << ((int)L)));
    }} else if constexpr(Q == Q_MODE::PER_COLUMN) {{
        TORCH_CHECK(codebook.size(0) == (1<<R) * m * (1 << ((int)L)));
    }}

    decompress_ptr<R, Q, L, LD>(
            reinterpret_cast<half2 *>(out.data_ptr<c10::Half>()),
            reinterpret_cast<const uint32_t *>(compressed.data_ptr<int32_t>()),
            reinterpret_cast<const half *>(codebook.data_ptr<c10::Half>()),
            static_cast<uint32_t>(m),
            static_cast<uint32_t>(k),
            at::cuda::getCurrentCUDAStream()
    );
}}
"""
    return str_

def qtip_torch_func_gen(bitrate, m, n, k, q, l, ld):  
    qmode = f"{"_pc" if q == 1 else "" if q==0 else "_g"}"
    lut_type = f"{"_sq" if l == 0 else "_vq2" if l==1 else "_vq4" if l==2 else "_vq8"}"
    load_type = f"{"_dup" if ld == 0 else ""}"
    str_ = f"""
__host__ extern void decompress_gemm_{bitrate}_{m}_{n}_{k}{qmode}{lut_type}{load_type}(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &x,
        torch::Tensor &codebook
) {{
    decompress_gemm<{bitrate}U, {m}U, {n}U, {k}U, {Q_MODES[q]}, {LUT_TYPES[l]}, {LOAD_TYPES[ld]}>(out, compressed, x, codebook);
}}
""" 
    return str_

def qtip_torch_func_gen_dq(bitrate, q, l, ld):  
    qmode = f"{"_pc" if q == 1 else "" if q==0 else "_g"}"
    lut_type = f"{"_sq" if l == 0 else "_vq2" if l==1 else "_vq4" if l==2 else "_vq8"}"
    load_type = f"{"_dup" if ld == 0 else ""}"
    str_ = f"""
__host__ extern void decompress_{bitrate}{qmode}{lut_type}{load_type}(
        torch::Tensor &out,
        torch::Tensor &compressed,
        torch::Tensor &codebook
) {{
    decompress<{bitrate}U, {Q_MODES[q]}, {LUT_TYPES[l]}, {LOAD_TYPES[ld]}>(out, compressed, codebook);
}}
""" 
    return str_

str_ = qtip_torch_header_gen()
for q, l, ld, max_bits in qmode_lut_ld_maxbits_list:
    stride = 2 if l == 2 else 1
    for bitrate in range(2, max_bits+1, stride):
        str_ += qtip_torch_func_gen_dq(bitrate, q, l, ld)
        for m, k in MKSHAPE:
            for n in [1,2,4,8]:
                str_ += qtip_torch_func_gen(bitrate, m, n, k, q, l, ld)

print(str_)

with open("qtip_torch.cu", "w") as f:
    f.write(str_)
