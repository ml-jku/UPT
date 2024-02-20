import einops
import numpy as np
from torch import nn

from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_interpolate_mesh_to_grid import CfdInterpolateMeshToGrid


class CfdInterpolate(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.resolution = self.data_container.get_dataset().grid_resolution
        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        self.embed = CfdInterpolateMeshToGrid()
        self.proj = nn.Linear(input_dim, dim)
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = len(self.resolution)
        self.output_shape = (int(np.prod(self.resolution)), dim)

    def forward(self, x, mesh_pos, grid_pos, mesh_to_grid_edges, batch_idx):
        x = self.embed(x=x, mesh_pos=mesh_pos, grid_pos=grid_pos, batch_idx=batch_idx)

        # convert to dense tensor (dim last)
        x = x.reshape(-1, *self.resolution, x.size(1))
        x = einops.rearrange(x, "batch_size ... dim -> batch_size (...) dim")

        x = self.proj(x)
        return x
