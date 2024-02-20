import numpy as np

from models.base.single_model_base import SingleModelBase
from modules.gno.rans_gino_mesh_to_grid_sdf_og import RansGinoMeshToGridSdfOg
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn

class RansGinoSdfOg(SingleModelBase):
    def __init__(self, hidden_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.resolution = self.data_container.get_dataset().grid_resolution
        self.mesh_to_grid = RansGinoMeshToGridSdfOg(
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            resolution=self.resolution,
        )
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = len(self.resolution)
        self.output_shape = (int(np.prod(self.resolution)), self.mesh_to_grid.output_dim)

    def forward(self, mesh_pos, sdf, grid_pos, mesh_to_grid_edges):
        return self.mesh_to_grid(mesh_pos=mesh_pos, sdf=sdf, grid_pos=grid_pos, mesh_to_grid_edges=mesh_to_grid_edges)
