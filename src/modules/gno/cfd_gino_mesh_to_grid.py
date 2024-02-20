import einops
import numpy as np
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection, Residual
from torch import nn
from torch_scatter import segment_csr
from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias


class CfdGinoMeshToGrid(nn.Module):
    def __init__(self, input_dim, hidden_dim, resolution, init_weights="xavier_uniform"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.resolution = resolution
        self.init_weights = init_weights
        self.num_grid_points = int(np.prod(resolution))

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=len(resolution))
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, x, mesh_pos, grid_pos, mesh_to_grid_edges):
        assert x.ndim == 2
        assert mesh_pos.ndim == 2
        assert grid_pos.ndim == 2
        assert mesh_to_grid_edges.ndim == 2
        assert len(grid_pos) % self.num_grid_points == 0
        # embed mesh
        x = self.input_proj(x) + self.pos_embed(mesh_pos)

        # embed grid
        grid_pos = self.pos_embed(grid_pos)

        # create message input
        grid_idx, mesh_idx = mesh_to_grid_edges.unbind(1)
        x = torch.concat([x[mesh_idx], grid_pos[grid_idx]], dim=1)
        x = self.message(x)
        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = grid_idx.unique(return_counts=True)
        # first index has to be 0 + add padding for target indices that dont occour
        padded_counts = torch.zeros(len(grid_pos) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[dst_indices + 1] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")

        # convert to dense tensor (dim last)
        x = x.reshape(-1, *self.resolution, self.output_dim)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size (...) dim")

        return x
