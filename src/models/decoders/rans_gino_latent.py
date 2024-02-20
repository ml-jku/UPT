import einops

from models.base.single_model_base import SingleModelBase
from modules.gno.rans_gino_latent_to_mesh import RansGinoLatentToMesh


class RansGinoLatent(SingleModelBase):
    def __init__(self, hidden_dim, bottleneck_dim=None, pred_hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        # ouptut_shape is (None, output_dim)
        _, output_dim = self.output_shape
        self.grid_to_mesh = RansGinoLatentToMesh(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            ndim=self.static_ctx["ndim"],
            bottleneck_dim=bottleneck_dim,
            pred_hidden_dim=pred_hidden_dim,
        )

    def forward(self, x, query_pos, unbatch_idx, unbatch_select):
        return self.grid_to_mesh(x, query_pos=query_pos)
