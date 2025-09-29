import os

import torch
from torch import nn

from lib.utils.kmeans import kmeans_flash1d, kmeans_sklearn

class vq_codebook(nn.Module):
    def __init__(self,
                 vec_sz=2,
                 lut_bits=8):
        super(vq_codebook, self).__init__()
        self.idx_dtype = torch.int32
        self.vec_sz = vec_sz
        self.lut_bits = lut_bits

        fname = f'assets/lut_cache/vq_kmeans_{lut_bits}_{vec_sz}.pt'
        if not os.path.exists(fname):
            if vec_sz == 1:
                data = torch.randn(int(1e8), vec_sz)
                tlut = kmeans_flash1d(data, 2**lut_bits)
            elif vec_sz in [2,4]:
                data = torch.randn(int(1e8), vec_sz)
                if lut_bits <= 5:
                    tlut = kmeans_sklearn(data, 2**lut_bits, max_data=int(1e8))
                else:
                    tlut = kmeans_sklearn(data, 2**lut_bits, max_data=int(1e7))
            torch.save(tlut, fname)
        else:
            tlut = torch.load(fname)
        self.register_buffer("tlut", tlut)
        self.register_buffer("lut", tlut.T.contiguous())
   
    def recons(self, encoded, **kwargs):
        return self.tlut[encoded].contiguous()
     
    def quantize(self, X, **kwargs):
        """
            X : [B, vec_sz]
        """
        dist = torch.cdist(X, self.tlut.to(X.device, dtype=X.dtype)) # [B, 2**lut_bits]
        state = torch.argmin(dist, dim=-1) # [B,] each entry is in [0, 2**lut_bits)
        hatX = self.recons(state)
        return hatX.to(X.device), state.to(X.device)
    

if __name__ == "__main__":
    for vec_sz in [4]:
        for lut_bits in [6,7,8,9,10,11,12]:
        # for lut_bits in [1,2,3,4,5,6,7,8,9,10,11,12]:
            if vec_sz == 1 and lut_bits > 8:
                continue
            vq = vq_codebook(vec_sz=vec_sz, lut_bits=lut_bits)
            X = torch.randn(int(1e5), vec_sz)
            hatX, state = vq.quantize(X)
            print(f"vec_sz: {vec_sz}, lut_bits: {lut_bits}, mse: {(hatX-X).pow(2).mean()}")
