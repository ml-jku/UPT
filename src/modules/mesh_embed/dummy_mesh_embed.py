import math
import einops
import torch.nn.functional as F
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_geometric.nn.pool import SAGPooling
import torch

class DummyMeshEmbed(nn.Module):
    def __init__(self, in_features, hidden_features, pool_ratio):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.pool_ratio = pool_ratio
        self.proj = nn.Linear(in_features, hidden_features)
        self.pool = SAGPooling(hidden_features, ratio=pool_ratio)
        self.reset_parameters()

    def reset_parameters(self):
        init_xavier_uniform_zero_bias(self.proj)

    def forward(self, x, pos, edge_index, batch_idx):
        pool_result = self.pool(self.proj(x), edge_index.T, batch=batch_idx)
        x_pool, _, _, batch_pool, _, _ = pool_result
        return x_pool, batch_pool