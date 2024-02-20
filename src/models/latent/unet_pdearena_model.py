import einops
from kappamodules.unet import UnetPdearena

from models.base.single_model_base import SingleModelBase


class UnetPdearenaModel(SingleModelBase):
    def __init__(self, dim, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim

        # propagate output shape
        seqlen, input_dim = self.input_shape
        self.output_shape = (seqlen, dim)

        assert self.static_ctx["ndim"] == 2
        # Unetmod-64
        self.model = UnetPdearena(
            hidden_channels=dim,
            input_dim=input_dim,
            output_dim=dim,
            norm=False,  # dont use norm because it uses a hardcoded num_groups=8
            cond_dim=self.static_ctx.get("condition_dim", None),
        )

    def forward(self, x, condition=None):
        # dim last without spatial -> dim first with spatial
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        # unet
        x = self.model(x, emb=condition)
        # dim first with spatial -> dim last without spatial
        x = einops.rearrange(x, "batch_size dim ... -> batch_size (...) dim")
        return x
