import torch
import torch.nn as nn
import math

class CombLinearTCQ(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        td_x,
        td_y,
        out_part,
        L,  # trellis window
        KV,  # bpw
        V,  # vq dim
        tlut_bits, 
        bias=False,
        dtype=torch.float16,
    ):
        super().__init__()
        assert len(out_part) == 2 and len(KV) == 2
        assert out_part[0] + out_part[1] == out_features

        self.in_features = in_features
        self.out_features = out_features
        self.out_part = out_part
        self.td_x = td_x
        self.td_y = td_y
        self.L = L
        self.KV = KV
        self.V = V
        self.tlut_bits = tlut_bits
        self.dtype = dtype
        # packed into int16
        self.register_buffer(
            'trellis1',
            torch.zeros((out_part[0] // td_x) * (in_features // td_y),
                        math.ceil((td_x * td_y) * KV[0] / 16 / V),
                        dtype=torch.int16))
        self.register_buffer(
            'trellis2',
            torch.zeros((out_part[1] // td_x) * (in_features // td_y),
                        math.ceil((td_x * td_y) * KV[1] / 16 / V),
                        dtype=torch.int16))
        self.tlut = nn.Parameter(torch.zeros(2**tlut_bits,
                                                V,
                                                dtype=torch.float16),
                                    requires_grad=False)

        if bias:
            self.register_buffer('bias', torch.ones(out_features))
        else:
            self.bias = None
        
        if out_part[0] == out_part[1]:
            self.use_comb_kernel = True
        else:
            self.use_comb_kernel = False


    def _info(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "td_x": self.td_x,
            "td_y": self.td_y,
            "out_part": self.out_part,
            "L": self.L,
            "KV": self.KV,
            "V": self.V,
            'tlut_bits': self.tlut_bits,
            "dtype": self.dtype,    
            "trellis1": self.trellis1.detach().cpu(),
            "trellis2": self.trellis2.detach().cpu(),
            "tlut": self.tlut.detach().cpu().half(),
            "bias": self.bias.detach().cpu() if self.bias is not None else None,
        }
        return info

    def forward(self, inp, **kwargs):
        x = inp.view(-1, self.in_features)
        bs = x.shape[0]
        m, k = self.out_features, self.in_features
        if bs <= 8:
            if self.use_comb_kernel:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"decompress_gemm_tcq_comb_{self.out_features}_{bs}_{k}_{self.tlut_bits}_{self.KV[0]}_{self.KV[1]}"
                )
                x = wrapper(self.trellis1, self.trellis2, x, self.tlut)
            else:
                wrapper1 = getattr(
                    torch.ops.ours_lib,
                    f"decompress_gemm_tcq_{self.out_part[0]}_{bs}_{k}_{self.tlut_bits}_{self.KV[0]}"
                )
                wrapper2 = getattr( 
                    torch.ops.ours_lib,
                    f"decompress_gemm_tcq_{self.out_part[1]}_{bs}_{k}_{self.tlut_bits}_{self.KV[1]}"
                )
                x1 = wrapper1(self.trellis1, x, self.tlut)
                x2 = wrapper2(self.trellis2, x, self.tlut)
                x = torch.cat([x1, x2], dim=1)
        else:
            if self.use_comb_kernel:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"decompress_tcq_comb_{self.tlut_bits}_{self.KV[0]}_{self.KV[1]}"
                )
                with torch.no_grad():
                    dq = wrapper(self.trellis1, self.trellis2, self.tlut, self.out_features, k)
                x = x.to(dq.dtype) @ dq.T
            else:
                wrapper1 = getattr(
                torch.ops.ours_lib,
                f"decompress_tcq_{self.tlut_bits}_{self.KV[0]}"
                )
                wrapper2 = getattr(
                    torch.ops.ours_lib,
                    f"decompress_tcq_{self.tlut_bits}_{self.KV[1]}"
                )
                with torch.no_grad():
                    dq1 = wrapper1(self.trellis1, self.tlut, self.out_part[0], k)
                    dq2 = wrapper2(self.trellis2, self.tlut, self.out_part[1], k)
                x1 = x.to(dq1.dtype) @ dq1.T
                x2 = x.to(dq2.dtype) @ dq2.T
                x = torch.cat([x1, x2], dim=1)
        return x.view(*inp.shape[:-1], m).to(inp.dtype)

    @staticmethod
    def gen_layer_from_info(info):
        layer = CombLinearTCQ(info["in_features"], info["out_features"], info["td_x"], info["td_y"], info["out_part"], info["L"], info["KV"], info["V"], info["tlut_bits"], info["bias"] is not None, info["dtype"])
        layer.trellis1.data.copy_(info["trellis1"])
        layer.trellis2.data.copy_(info["trellis2"])
        layer.tlut.data.copy_(info["tlut"])
        if info["bias"] is not None:
            layer.bias.data.copy_(info["bias"])
        return layer

    def get_weight(self):
        wrapper = getattr(
                    torch.ops.ours_lib,
                    f"decompress_tcq_comb_{self.tlut_bits}_{self.KV[0]}_{self.KV[1]}"
                )
        dq = wrapper(self.trellis1, self.trellis2, self.tlut, self.out_features, self.in_features)
        return dq


class CombtLinearTCQ(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        td_x,
        td_y,
        in_part,
        L,  # trellis window
        KV,  # bpw
        V,  # vq dim
        tlut_bits, 
        bias=False,
        dtype=torch.float16,
    ):
        super().__init__()
        assert len(in_part) == 2 and len(KV) == 2
        assert in_part[0] + in_part[1] == in_features

        self.in_features = in_features
        self.out_features = out_features
        self.in_part = in_part
        self.td_x = td_x
        self.td_y = td_y
        self.L = L
        self.KV = KV
        self.V = V
        self.tlut_bits = tlut_bits
        self.dtype = dtype
        # packed into int16
        self.register_buffer(
            'trellis1',
            torch.zeros((out_features // td_x) * (in_part[0] // td_y),
                        math.ceil((td_x * td_y) * KV[0] / 16 / V),
                        dtype=torch.int16))
        self.register_buffer(
            'trellis2',
            torch.zeros((out_features // td_x) * (in_part[1] // td_y),
                        math.ceil((td_x * td_y) * KV[1] / 16 / V),
                        dtype=torch.int16))
        self.tlut = nn.Parameter(torch.zeros(2**tlut_bits,
                                                V,
                                                dtype=torch.float16),
                                    requires_grad=False)

        if bias:
            self.register_buffer('bias', torch.ones(out_features))
        else:
            self.bias = None
        
        if in_part[0] == in_part[1]:
            self.use_comb_kernel = True
        else:
            self.use_comb_kernel = False


    def _info(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "td_x": self.td_x,
            "td_y": self.td_y,
            "in_part": self.in_part,
            "L": self.L,
            "KV": self.KV,
            "V": self.V,
            'tlut_bits': self.tlut_bits,
            "dtype": self.dtype,    
            "trellis1": self.trellis1.detach().cpu(),
            "trellis2": self.trellis2.detach().cpu(),
            "tlut": self.tlut.detach().cpu().half(),
            "bias": self.bias.detach().cpu() if self.bias is not None else None,
        }
        return info

    def forward(self, inp, **kwargs):
        x = inp.view(-1, self.in_features)
        bs = x.shape[0]
        m, k = self.out_features, self.in_features
        if bs <= 8:
            if self.use_comb_kernel:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"decompress_gemm_tcq_combt_{self.out_features}_{bs}_{k}_{self.tlut_bits}_{self.KV[0]}_{self.KV[1]}"
                )
                x = wrapper(self.trellis1, self.trellis2, x, self.tlut)
            else:
                wrapper1 = getattr(
                    torch.ops.ours_lib,
                    f"decompress_gemm_tcq_{m}_{bs}_{self.in_part[0]}_{self.tlut_bits}_{self.KV[0]}"
                )
                wrapper2 = getattr( 
                    torch.ops.ours_lib,
                    f"decompress_gemm_tcq_{m}_{bs}_{self.in_part[1]}_{self.tlut_bits}_{self.KV[1]}"
                )
                x1 = wrapper1(self.trellis1, x[:, :self.in_part[0]], self.tlut)
                x2 = wrapper2(self.trellis2, x[:, self.in_part[0]:], self.tlut)
                x = x1 + x2
        else:
            if self.use_comb_kernel:
                wrapper = getattr(
                    torch.ops.ours_lib,
                    f"decompress_tcq_combt_{self.tlut_bits}_{self.KV[0]}_{self.KV[1]}"
                )
                with torch.no_grad():
                    dq = wrapper(self.trellis1, self.trellis2, self.tlut, self.out_features, k)
                x = x.to(dq.dtype) @ dq.T
            else:
                wrapper1 = getattr(
                torch.ops.ours_lib,
                f"decompress_tcq_{self.tlut_bits}_{self.KV[0]}"
                )
                wrapper2 = getattr(
                    torch.ops.ours_lib,
                    f"decompress_tcq_{self.tlut_bits}_{self.KV[1]}"
                )
                with torch.no_grad():
                    dq1 = wrapper1(self.trellis1, self.tlut, m, self.in_part[0])
                    dq2 = wrapper2(self.trellis2, self.tlut, m, self.in_part[1])
                x1 = x[:, :self.in_part[0]].to(dq1.dtype) @ dq1.T
                x2 = x[:, self.in_part[0]:].to(dq2.dtype) @ dq2.T
                x = x1 + x2
        return x.view(*inp.shape[:-1], m).to(inp.dtype)

    @staticmethod
    def gen_layer_from_info(info):
        layer = CombtLinearTCQ(info["in_features"], info["out_features"], info["td_x"], info["td_y"], info["in_part"], info["L"], info["KV"], info["V"], info["tlut_bits"], info["bias"] is not None, info["dtype"])
        layer.trellis1.data.copy_(info["trellis1"])
        layer.trellis2.data.copy_(info["trellis2"])
        layer.tlut.data.copy_(info["tlut"])
        if info["bias"] is not None:
            layer.bias.data.copy_(info["bias"])
        return layer

    def get_weight(self):
        wrapper = getattr(
                    torch.ops.ours_lib,
                    f"decompress_tcq_combt_{self.tlut_bits}_{self.KV[0]}_{self.KV[1]}"
                )
        dq = wrapper(self.trellis1, self.trellis2, self.tlut, self.out_features, self.in_features)
        return dq
    
    @staticmethod
    def merge_infos(info1, info2):
        assert info1["in_features"] == info2["in_features"]
        assert info1["td_x"] == info2["td_x"]
        assert info1["td_y"] == info2["td_y"]
        assert info1["L"] == info2["L"]
        assert info1["KV"] == info2["KV"]
        assert info1["V"] == info2["V"]
        assert info1["tlut_bits"] == info2["tlut_bits"]
        if not torch.allclose(info1["tlut"], info2["tlut"], atol=1e-4):
            print("warning: tlut is not close. it is unexpected behavior if you do not use dummy quantizers.")
        assert info1["bias"] is None and info2["bias"] is None
        assert info1["dtype"] == info2["dtype"]
        info = {}
        info["in_features"] = info1["in_features"] 
        info["out_features"] = info1["out_features"] + info2["out_features"]
        info["td_x"] = info1["td_x"]
        info["td_y"] = info1["td_y"]
        info["L"] = info1["L"]
        info["KV"] = info1["KV"]
        info["V"] = info1["V"]
        info["tlut_bits"] = info1["tlut_bits"]
        info["bias"] = None
        info["dtype"] = info1["dtype"]
        info["trellis1"] = torch.cat([info1["trellis1"], info2["trellis1"]], dim=0)
        info["trellis2"] = torch.cat([info1["trellis2"], info2["trellis2"]], dim=0)
        info["tlut"] = info1["tlut"]
        info["in_part"] = info1["in_part"]


        return info
    
if __name__ == "__main__":
    layer = CombLinearTCQ(4096, 4096, 16, 16, (2048, 2048), 16, (3, 4), 2, 9, False)
    print(layer._info())
    layer.forward(torch.randn(1, 4096).cuda())
