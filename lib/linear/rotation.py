from lib.utils import matmul_hadU_cuda, matmul_hadUt_cuda, matmul_hadUt_head, matmul_hadU_head, get_hadK
import torch.nn as nn

class RotateWeights(nn.Module):
    def __init__(self, had_dim_U, had_dim_V, SU=None, SV=None):
        super().__init__()
        self.had_dim_U = had_dim_U
        self.had_dim_V = had_dim_V
        self.SU = SU
        self.SV = SV

        self.had_left_U, self.K_left_U = get_hadK(had_dim_U)
        self.had_left_V, self.K_left_V = get_hadK(had_dim_V)

    def apply_weights(self, weights):
        return matmul_hadUt_head(matmul_hadUt_head(weights.T, self.had_left_U, self.K_left_U), self.had_left_V, self.K_left_V)