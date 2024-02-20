import einops
import torch
import torch.nn.functional as F
from torch import nn

from models.base.single_model_base import SingleModelBase


class RansInterpolated(SingleModelBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        # ouptut_shape is (None, output_dim)
        _, output_dim = self.output_shape

        # pred (input_dim is the hidden dimension of the latent model so no seperate hidden dim is needed)
        self.pred = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x, query_pos):
        assert torch.all(-1 <= query_pos)
        assert torch.all(query_pos <= 1)

        # dim last without spatial -> dim first with spatial
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")

        # grid_sample requires 5d dense tensor
        # TODO should be implemented via padding in collator
        query_pos = einops.rearrange(
            query_pos,
            "(batch_size num_query_pos) ndim -> batch_size num_query_pos 1 1 ndim",
            num_query_pos=3586,
        )

        # interpolate to mesh
        # x.shape: (batch_size, dim, height, width, depth)
        # mesh_pos.shape: (batch_size, num_mesh_pos, 3)
        x_hat = F.grid_sample(input=x, grid=query_pos, align_corners=False)
        # to sparse tensor
        x_hat = einops.rearrange(x_hat, "batch_size dim num_query_pos 1 1 -> (batch_size num_query_pos) dim ")

        # predict
        x_hat = self.pred(x_hat)
        return x_hat
