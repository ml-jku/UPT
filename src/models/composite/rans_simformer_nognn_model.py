from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class RansSimformerNognnModel(CompositeModelBase):
    def __init__(
            self,
            encoder,
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
        # encoder
        self.encoder = create(
            encoder,
            model_from_kwargs,
            input_shape=self.input_shape,
            **common_kwargs,
        )
        # latent
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.encoder.output_shape,
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
            encoder=self.encoder,
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(self, mesh_pos, query_pos, batch_idx, unbatch_idx, unbatch_select):
        outputs = {}

        # encode data
        encoded = self.encoder(mesh_pos=mesh_pos, batch_idx=batch_idx)

        # propagate
        propagated = self.latent(encoded)

        # decode
        x_hat = self.decoder(propagated, query_pos=query_pos, unbatch_idx=unbatch_idx, unbatch_select=unbatch_select)
        outputs["x_hat"] = x_hat

        return outputs
