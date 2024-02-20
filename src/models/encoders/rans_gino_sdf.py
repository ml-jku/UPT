import numpy as np

from models.base.single_model_base import SingleModelBase
from modules.gno.rans_gino_mesh_to_grid_sdf import RansGinoMeshToGridSdf
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn

class RansGinoSdf(SingleModelBase):
    def __init__(self, dim, resolution=None, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.resolution = resolution or self.data_container.get_dataset().grid_resolution
        self.mesh_to_grid = RansGinoMeshToGridSdf(dim=dim, resolution=self.resolution)
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = len(self.resolution)
        self.output_shape = (int(np.prod(self.resolution)), dim)

    def forward(self, mesh_pos, sdf, grid_pos, mesh_to_grid_edges):
        return self.mesh_to_grid(mesh_pos=mesh_pos, sdf=sdf, grid_pos=grid_pos, mesh_to_grid_edges=mesh_to_grid_edges)
