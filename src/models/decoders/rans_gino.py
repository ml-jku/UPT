from models.base.single_model_base import SingleModelBase
from modules.gno.rans_gino_grid_to_mesh import RansGinoGridToMesh


class RansGino(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        # input_shape is (None, input_dim)
        _, input_dim = self.input_shape
        # ouptut_shape is (None, output_dim)
        _, output_dim = self.output_shape
        self.grid_to_mesh = RansGinoGridToMesh(
            input_dim=input_dim,
            hidden_dim=dim,
            output_dim=output_dim,
            ndim=self.static_ctx["ndim"],
        )

    def forward(self, x, query_pos, grid_to_query_edges):
        return self.grid_to_mesh(
            x,
            query_pos=query_pos,
            grid_to_query_edges=grid_to_query_edges,
        )
