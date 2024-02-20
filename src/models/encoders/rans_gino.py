import numpy as np

from models.base.single_model_base import SingleModelBase
from modules.gno.rans_gino_mesh_to_grid import RansGinoMeshToGrid


class RansGino(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.resolution = self.data_container.get_dataset().grid_resolution
        self.embed = RansGinoMeshToGrid(dim=dim, resolution=self.resolution)
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = len(self.resolution)
        self.output_shape = (int(np.prod(self.resolution)), self.embed.output_dim)

    def forward(self, mesh_pos, grid_pos, mesh_to_grid_edges):
        return self.embed(mesh_pos=mesh_pos, grid_pos=grid_pos, mesh_to_grid_edges=mesh_to_grid_edges)
