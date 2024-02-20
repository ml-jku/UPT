import torch
from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class RansSimformerNognnSdfModel(CompositeModelBase):
    def __init__(
            self,
            grid_encoder,
            mesh_encoder,
            latent,
            decoder,
            **kwargs,
    ):
        super().__init__(**kwargs)
        common_kwargs = dict(
            update_counter=self.update_counter,
            path_provider=self.path_provider,
            dynamic_ctx=self.dynamic_ctx,
            static_ctx=self.static_ctx,
            data_container=self.data_container,
        )
        # grid_encoder
        self.grid_encoder = create(
            grid_encoder,
            model_from_kwargs,
            **common_kwargs,
        )
        # mesh_encoder
        self.mesh_encoder = create(
            mesh_encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
        )
        # latent
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.mesh_encoder.output_shape,
            **common_kwargs,
        )
        # decoder
        self.decoder = create(
            decoder,
            model_from_kwargs,
            **common_kwargs,
            input_shape=self.latent.output_shape,
            output_shape=self.output_shape,
        )

    @property
    def submodels(self):
        return dict(
            grid_encoder=self.grid_encoder,
            mesh_encoder=self.mesh_encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(self, mesh_pos, sdf, query_pos, batch_idx, unbatch_idx, unbatch_select):
        outputs = {}

        # encode data
        grid_embed = self.grid_encoder(sdf)
        mesh_embed = self.mesh_encoder(mesh_pos=mesh_pos, batch_idx=batch_idx)
        embed = torch.concat([grid_embed, mesh_embed], dim=1)

        # propagate
        propagated = self.latent(embed)

        # decode
        x_hat = self.decoder(
            propagated,
            query_pos=query_pos,
            unbatch_idx=unbatch_idx,
            unbatch_select=unbatch_select,
        )
        outputs["x_hat"] = x_hat

        return outputs
