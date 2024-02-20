import einops
import numpy as np
import torch
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_scatter import segment_csr


class RansGinoMeshToGridSdfOg(nn.Module):
    def __init__(self, hidden_dim, output_dim, resolution):
        super().__init__()
        # original parameters:
        # dim=64
        # resolution=64
        # output_dim=86

        self.hidden_dim = hidden_dim
        # GINO concats the raw SDF and the raw grid position before the FNO
        self.output_dim = output_dim + 4
        self.resolution = resolution
        self.num_grid_points = int(np.prod(resolution))

        # "df_embed"
        self.sdf_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 3),
        )
        # GINO concats a constant 1 as 4th dimension for some reason
        self.mesh_pos_embed = ContinuousSincosEmbed(dim=hidden_dim * 4, ndim=len(resolution) + 1)
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim * 3, ndim=len(resolution))
        # "gno1"
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 10, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, mesh_pos, sdf, grid_pos, mesh_to_grid_edges):
        assert mesh_pos.ndim == 2
        assert grid_pos.ndim == 2
        assert mesh_to_grid_edges.ndim == 2
        assert len(grid_pos) % self.num_grid_points == 0
        # NOTE: we rescale all positions to [0, 200] instead of [-1, 1] -> revert
        mesh_pos = mesh_pos / 100 - 1
        grid_pos = grid_pos / 100 - 1

        # embed mesh
        # original implementation adds a 4th dimension with constant 1 during training
        ones = torch.ones(size=(len(mesh_pos),), dtype=mesh_pos.dtype, device=mesh_pos.device).unsqueeze(1)
        mesh_pos = torch.concat([mesh_pos, ones], dim=1)
        mesh_pos = self.mesh_pos_embed(mesh_pos)

        # embed grid
        grid_pos_embed = self.pos_embed(grid_pos)

        # flatten sdf -> embed SDF
        sdf = sdf.view(-1, 1)
        sdf_embed = self.sdf_embed(sdf)

        # create grid embedding (positional embedding of grid posisionts + SDF embedding)
        grid_embed = torch.concat([grid_pos_embed, sdf_embed], dim=1)

        # create message input
        grid_idx, mesh_idx = mesh_to_grid_edges.unbind(1)
        x = torch.concat([mesh_pos[mesh_idx], grid_embed[grid_idx]], dim=1)
        x = self.message(x)
        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = grid_idx.unique(return_counts=True)
        # first index has to be 0 + add padding for target indices that dont occour
        padded_counts = torch.zeros(len(grid_embed) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[dst_indices + 1] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")

        # GINO concats grid positions (without sincos embedding) and the raw sdf (i.e. without sdf_embed) before FNO
        x = torch.concat([grid_pos, sdf, x], dim=1)

        # convert to dense tensor (dim last)
        x = x.reshape(-1, *self.resolution, self.output_dim)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size (...) dim")

        return x
