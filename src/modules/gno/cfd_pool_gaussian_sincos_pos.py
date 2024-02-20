import einops
import numpy as np
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection, Residual
from kappamodules.init import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from torch import nn
from torch_scatter import segment_csr


class CfdPoolGaussianSincosPos(nn.Module):
    def __init__(self, input_dim, hidden_dim, positional_std, ndim=2, init_weights="xavier_uniform"):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Try with other method from https://arxiv.org/pdf/2006.10739.pdf
        generator = torch.Generator().manual_seed(42)
        self.register_buffer(
            "b",
            torch.normal(mean=torch.zeros(hidden_dim // 2, ndim), std=positional_std, generator=generator)
        )
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

    def forward(self, x, mesh_pos, mesh_edges, batch_idx):
        assert x.ndim == 2
        assert mesh_pos.ndim == 2
        assert mesh_edges.ndim == 2

        # embed mesh
        x = self.input_proj(x) + self.pos_embed(mesh_pos)

        # create message input
        dst_idx, src_idx = mesh_edges.unbind(1)
        x = torch.concat([x[src_idx], x[dst_idx]], dim=1)
        x = self.message(x)
        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = dst_idx.unique(return_counts=True)
        # first index has to be 0
        # NOTE: padding for target indices that dont occour is not needed as self-loop is always present
        padded_counts = torch.zeros(len(counts) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")


        # sanity check: dst_indices has len of batch_size * num_supernodes and has to be divisible by batch_size
        # if num_supernodes is not set in dataset this assertion fails
        batch_size = batch_idx.max() + 1
        assert dst_indices.numel() % batch_size == 0

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )


        return x

    def pos_embed(self, pos):
        return torch.concat([torch.cos(2.0 * torch.pi * pos @ self.b.T), 
                             torch.sin(2.0 * torch.pi * pos @ self.b.T)], dim=-1)
