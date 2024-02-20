from models import model_from_kwargs
from models.base.composite_model_base import CompositeModelBase
from utils.factory import create


class RansInterpolatedModel(CompositeModelBase):
    def __init__(
            self,
            latent,
            decoder,
            grid_resolution=None,
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
        grid_resolution = grid_resolution or self.data_container.get_dataset().grid_resolution
        self.static_ctx["grid_resolution"] = grid_resolution
        self.static_ctx["ndim"] = len(grid_resolution)
        # latent
        self.latent = create(
            latent,
            model_from_kwargs,
            input_shape=self.input_shape,
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
            latent=self.latent,
            decoder=self.decoder,
        )

    # noinspection PyMethodOverriding
    def forward(self, x, query_pos):
        outputs = {}

        # propagate
        propagated = self.latent(x)

        # decode
        x_hat = self.decoder(propagated, query_pos=query_pos)
        outputs["x_hat"] = x_hat

        return outputs
