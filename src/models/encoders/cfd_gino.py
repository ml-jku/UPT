import numpy as np

from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_gino_mesh_to_grid import CfdGinoMeshToGrid


class CfdGino(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.resolution = self.data_container.get_dataset().grid_resolution
        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        self.embed = CfdGinoMeshToGrid(
            input_dim=input_dim,
            hidden_dim=dim,
            resolution=self.resolution,
        )
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = len(self.resolution)
        self.output_shape = (int(np.prod(self.resolution)), self.embed.output_dim)

    def forward(self, x, mesh_pos, grid_pos, mesh_to_grid_edges, batch_idx):
        return self.embed(x=x, mesh_pos=mesh_pos, grid_pos=grid_pos, mesh_to_grid_edges=mesh_to_grid_edges)
