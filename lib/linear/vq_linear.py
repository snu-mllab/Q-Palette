import torch
import torch
import torch.nn as nn

class VQLinearPackTensorCore(nn.Module):
    def __init__(self, in_features, out_features, lut_bits, vec_sz=2, bias=False, dtype=torch.half):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.lut_bits = lut_bits
        self.dtype = dtype
        self.vec_sz = vec_sz

        self.register_buffer(
            'qweight',
            torch.randint(0, 4, (out_features, lut_bits*in_features // 32 // vec_sz), dtype=torch.int32, device='cuda')
        )

        self.register_buffer(
            'lut',
            torch.randn((2 ** lut_bits, vec_sz), dtype=self.dtype, device='cuda')
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.randn((out_features,), dtype=self.dtype, device='cuda')
            )
        else:
            self.bias = None

        self.vq_type = f"vq{self.vec_sz}" if self.vec_sz > 1 else "sq_dup" if lut_bits <= 4 else "sq"

    def _info(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "lut_bits": self.lut_bits,
            "dtype": self.dtype,
            "vec_sz": self.vec_sz,
            "qweight": self.qweight.detach().cpu(),
            "lut": self.lut.detach().cpu().half(),
            "bias": self.bias.detach().cpu() if self.bias is not None else None,
        }
        return info
    
    def forward(self, inp, **kwargs):
        x = inp.view(-1, self.in_features)
        bs = x.shape[0]
        m, k = self.out_features, self.in_features
        if bs <= 8:
            wrapper = getattr(
                torch.ops.ours_lib,
                f"decompress_gemm_{m}_{bs}_{k}_{self.lut_bits}_{self.vq_type}"
            )

            x = wrapper(self.qweight, x, self.lut)
        else:
            wrapper = getattr(
                torch.ops.ours_lib,
                f"decompress_{self.lut_bits}_{self.vq_type}"
            )
            with torch.no_grad():
                dq = wrapper(self.qweight, self.lut, m, k)
            x = (x.to(dq.dtype) @ dq.T)

        return x.view(*inp.shape[:-1], m).to(inp.dtype)
    
    @staticmethod
    def gen_layer_from_info(info):
        layer = VQLinearPackTensorCore(info["in_features"], info["out_features"], info["lut_bits"], info["vec_sz"], info["bias"] is not None, info["dtype"])
        layer.qweight.data.copy_(info["qweight"])
        layer.lut.data.copy_(info["lut"])
        if info["bias"] is not None:
            layer.bias.data.copy_(info["bias"])
        return layer
    
    @staticmethod
    def merge_infos(info1, info2):
        assert info1["in_features"] == info2["in_features"]
        assert info1["lut_bits"] == info2["lut_bits"]
        assert info1["vec_sz"] == info2["vec_sz"]
        assert info1["bias"] is None and info2["bias"] is None
        assert info1["dtype"] == info2["dtype"]
        if not torch.allclose(info1["lut"], info2["lut"], atol=1e-4):
            print("warning: lut is not close. it is unexpected behavior if you do not use dummy quantizers.")
        info = {}
        info["in_features"] = info1["in_features"] 
        info["out_features"] = info1["out_features"] + info2["out_features"]
        info["lut_bits"] = info1["lut_bits"]
        info["vec_sz"] = info1["vec_sz"]
        info["bias"] = None
        info["dtype"] = info1["dtype"]
        info["qweight"] = torch.cat([info1["qweight"], info2["qweight"]], dim=0)
        info["lut"] = info1["lut"]
        return info

class VQLinearPackSIMT(nn.Module):
    def __init__(self, in_features, out_features, lut_bits, vec_sz=1, bias=False, dtype=torch.half):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.lut_bits = lut_bits
        self.dtype = dtype
        self.vec_sz = vec_sz

        self.register_buffer(
            'qweight',
            torch.randint(0, 4, (out_features, lut_bits*in_features // 32 // vec_sz), dtype=torch.int32, device='cuda')
        )

        self.register_buffer(
            'lut',
            torch.randn((2 ** lut_bits, vec_sz), dtype=self.dtype, device='cuda')
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.randn((out_features,), dtype=self.dtype, device='cuda')
            )
        else:
            self.bias = None

    def _info(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "lut_bits": self.lut_bits,
            "dtype": self.dtype,
            "vec_sz": self.vec_sz,
            "qweight": self.qweight.detach().cpu(),
            "lut": self.lut.detach().cpu().half(),
            "bias": self.bias.detach().cpu() if self.bias is not None else None,
        }
        return info
    
    def forward(self, inp, **kwargs):
        x = inp.view(-1, 1, self.in_features)
        bs = x.shape[0]
        m, k = self.out_features, self.in_features
        if bs <= 8:
            if self.vec_sz == 1:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"sq_pack_gemm_simt"
                )
                x = wrapper(x, self.qweight, self.lut, self.lut_bits)
            else:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"vq_pack_gemm_simt_{bs}_{self.vec_sz}_{self.lut_bits}"
                )
                x = wrapper(x, self.qweight, self.lut)
        else:
            if self.vec_sz == 1:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"sq_pack_dequant_simt"
                )
                with torch.no_grad():
                    dq = wrapper(self.qweight, self.lut, self.lut_bits, m, k) 
            else:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"vq_pack_dequant_simt_{self.vec_sz}_{self.lut_bits}"
                )
                with torch.no_grad():
                    dq = wrapper(self.qweight, self.lut, m, k) 
            x = (x.to(dq.dtype) @ dq.T)
        return x.view(*inp.shape[:-1], m).to(inp.dtype)
    
    @staticmethod
    def gen_layer_from_info(info):
        layer = VQLinearPackSIMT(info["in_features"], info["out_features"], info["lut_bits"], info["vec_sz"], info["bias"] is not None, info["dtype"])
        if info["vec_sz"] <= 2:
            from lib.quantizer.quant_op import convert_tensor_core_to_simt
            # qweight is stored in tensor core format in default.
            # we should convert it to simt format.
            converted_qweight = convert_tensor_core_to_simt(info["qweight"], info["out_features"], info["in_features"], info["vec_sz"], info["lut_bits"], code_n=info["lut_bits"])
            layer.qweight.data.copy_(converted_qweight)
        else:
            layer.qweight.data.copy_(info["qweight"])
        layer.lut.data.copy_(info["lut"])
        if info["bias"] is not None:
            layer.bias.data.copy_(info["bias"])
        return layer
    
    @staticmethod
    def merge_infos(info1, info2):
        assert info1["in_features"] == info2["in_features"]
        assert info1["lut_bits"] == info2["lut_bits"]
        assert info1["vec_sz"] == info2["vec_sz"]
        assert info1["bias"] is None and info2["bias"] is None
        assert info1["dtype"] == info2["dtype"]
        if not torch.allclose(info1["lut"], info2["lut"], atol=1e-4):
            print("warning: lut is not close. it is unexpected behavior if you do not use dummy quantizers.")
        info = {}
        info["in_features"] = info1["in_features"] 
        info["out_features"] = info1["out_features"] + info2["out_features"]
        info["lut_bits"] = info1["lut_bits"]
        info["vec_sz"] = info1["vec_sz"]
        info["bias"] = None
        info["dtype"] = info1["dtype"]
        info["qweight"] = torch.cat([info1["qweight"], info2["qweight"]], dim=0)
        info["lut"] = info1["lut"]
        return info