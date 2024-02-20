import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_interpolate_grid_to_mesh import CfdInterpolateGridToMesh

class CfdInterpolate(SingleModelBase):
    def __init__(self, dim=None, clamp=None, clamp_mode="log", **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.clamp = clamp
        self.clamp_mode = clamp_mode
        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        # ouptut_shape is (None, output_dim)
        _, output_dim = self.output_shape
        self.grid_to_mesh = CfdInterpolateGridToMesh()
        hidden_dim = dim or input_dim
        self.pred = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.resolution = self.static_ctx["grid_resolution"]

    def forward(self, x, grid_pos, query_pos, grid_to_query_edges):
        # TODO variable query pos not supported
        assert len(query_pos) % len(x) == 0
        query_pos = einops.rearrange(
            query_pos,
            "(batch_size num_query_pos) ndim -> batch_size num_query_pos ndim",
            batch_size=len(x),
        )

        # dim-last without spatial -> dim-first with spatial
        x = x.reshape(len(x), *self.resolution, -1)
        x = einops.rearrange(x, "batch_size height width dim -> batch_size dim width height")

        x = self.grid_to_mesh(x, query_pos=query_pos)
        # predict
        x = self.pred(x)
        if self.clamp is not None:
            assert self.clamp_mode == "log"
            # TODO torch.log1p should be equivalent
            x = torch.sign(x) * (self.clamp + torch.log(1 + x.abs()) - np.log(1 + self.clamp))

        return x
