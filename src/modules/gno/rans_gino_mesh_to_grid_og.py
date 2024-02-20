import einops
import numpy as np
import torch
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_scatter import segment_csr


class RansGinoMeshToGridOg(nn.Module):
    def __init__(self, dim, resolution):
        super().__init__()
        self.dim = dim
        self.resolution = resolution
        self.num_grid_points = int(np.prod(resolution))

        if isinstance(dim, int):
            # rectangular shape
            self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=len(resolution))
            self.message = nn.Sequential(
                nn.Linear(dim * 2, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
            )
            self.output_dim = dim
        else:
            # custom shape (original is 640 -> 512 -> 256 -> 86)
            assert dim[0] % 2 == 0
            self.pos_embed = ContinuousSincosEmbed(dim=dim[0] // 2, ndim=len(resolution))
            layers = []
            for i in range(len(dim) - 1):
                layers.append(nn.Linear(dim[i], dim[i + 1]))
                if i < len(dim) - 2:
                    layers.append(nn.GELU())
            self.message = nn.Sequential(*layers)
            self.output_dim = dim[-1]

    def forward(self, mesh_pos, grid_pos, mesh_to_grid_edges):
        assert mesh_pos.ndim == 2
        assert grid_pos.ndim == 2
        assert mesh_to_grid_edges.ndim == 2
        assert len(grid_pos) % self.num_grid_points == 0

        # embed mesh
        mesh_pos = self.pos_embed(mesh_pos)

        # embed grid
        grid_pos = self.pos_embed(grid_pos)

        # create message input
        grid_idx, mesh_idx = mesh_to_grid_edges.unbind(1)
        x = torch.concat([mesh_pos[mesh_idx], grid_pos[grid_idx]], dim=1)
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
