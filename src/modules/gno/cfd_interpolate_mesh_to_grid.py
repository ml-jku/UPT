import einops
import numpy as np
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection, Residual
from torch import nn
from torch_scatter import segment_csr
from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from torch_geometric.nn.unpool.knn_interpolate import knn_interpolate

class CfdInterpolateMeshToGrid(nn.Module):
    @staticmethod
    def forward(x, mesh_pos, grid_pos, batch_idx):
        assert x.ndim == 2
        assert mesh_pos.ndim == 2
        assert grid_pos.ndim == 2
        batch_size = batch_idx.max() + 1
        assert len(grid_pos) % batch_size == 0
        num_grid_points = len(grid_pos) // batch_size
        batch_y = torch.arange(batch_size, device=x.device).repeat_interleave(num_grid_points)
        x = knn_interpolate(
            x=x,
            pos_x=mesh_pos,
            pos_y=grid_pos,
            batch_x=batch_idx,
            batch_y=batch_y,
        )

        return x
