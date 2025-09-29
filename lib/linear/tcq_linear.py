import torch
import torch.nn as nn
import math

class QTIPLinearTCQ(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        td_x,
        td_y,
        L,  # trellis window
        KV,  # bpw
        V,  # vq dim
        tlut_bits, 
        bias=False,
        dtype=torch.float16,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.td_x = td_x
        self.td_y = td_y
        self.L = L
        self.KV = KV
        self.V = V
        self.tlut_bits = tlut_bits
        self.dtype = dtype
        # packed into int16
        self.register_buffer(
            'trellis',
            torch.zeros((out_features // td_x) * (in_features // td_y),
                        math.ceil((td_x * td_y) * KV / 16 / V),
                        dtype=torch.int16))

        self.tlut = nn.Parameter(torch.zeros(2**tlut_bits,
                                                V,
                                                dtype=torch.float16),
                                    requires_grad=False)

        if bias:
            self.register_buffer('bias', torch.ones(out_features))
        else:
            self.bias = None
    
    def _info(self):
        info = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "td_x": self.td_x,
            "td_y": self.td_y,
            "L": self.L,
            "KV": self.KV,
            "V": self.V,
            'tlut_bits': self.tlut_bits,
            "dtype": self.dtype,    
            "trellis": self.trellis.detach().cpu(),
            "tlut": self.tlut.detach().cpu().half(),
            "bias": self.bias.detach().cpu() if self.bias is not None else None,
        }
        return info

    def forward(self, inp, **kwargs):
        x = inp.view(-1, self.in_features)#.to(torch.float32)
        bs = x.shape[0]
        m, k = self.out_features, self.in_features
        if bs <= 8:
            wrapper = getattr(
                torch.ops.ours_lib,
                f"decompress_gemm_tcq_{m}_{bs}_{k}_{self.tlut_bits}_{self.KV}")

            x = wrapper(self.trellis, x, self.tlut)

        else:
            wrapper = getattr(
                torch.ops.ours_lib,
                f"decompress_tcq_{self.tlut_bits}_{self.KV}"
            )
            # dq = wrapper(self.trellis, self.tlut).to(x.dtype)
            # x = x @ dq.T 
            with torch.no_grad():
                dq = wrapper(self.trellis, self.tlut, m, k) #.to(x.dtype)
            x = (x.to(dq.dtype) @ dq.T)#.to(x.dtype)
        return x.view(*inp.shape[:-1], m).to(inp.dtype)

    @staticmethod
    def gen_layer_from_info(info):
        layer = QTIPLinearTCQ(info["in_features"], info["out_features"], info["td_x"], info["td_y"], info["L"], info["KV"], info["V"], info["tlut_bits"], info["bias"] is not None, info["dtype"])
        layer.trellis.data.copy_(info["trellis"])
        layer.tlut.data.copy_(info["tlut"])
        if info["bias"] is not None:
            layer.bias.data.copy_(info["bias"])
        return layer
    
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
        info["trellis"] = torch.cat([info1["trellis"], info2["trellis"]], dim=0)
        info["tlut"] = info1["tlut"]
        return info