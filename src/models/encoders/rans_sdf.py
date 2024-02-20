import numpy as np

from models.base.single_model_base import SingleModelBase
from modules.gno.rans_gino_mesh_to_grid_sdf import RansGinoMeshToGridSdf
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from kappamodules.layers.continuous_sincos_embed import ContinuousSincosEmbed

class RansSdf(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.resolution = self.data_container.get_dataset().grid_resolution
        self.sdf_embed = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=len(self.resolution))
        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = len(self.resolution)
        self.output_shape = (int(np.prod(self.resolution)), dim)

    def forward(self, sdf, grid_pos):
        # convert sdf to sparse tensor
        assert sdf.size(-1) == 1

        # embed
        sdf_embed = self.sdf_embed(sdf.view(-1, 1))
        grid_pos_embed = self.pos_embed(grid_pos)
        embed = sdf_embed + grid_pos_embed

        # convert to dim-last without spatial resolution
        embed = embed.view(len(sdf), *self.resolution, -1)
        return embed
