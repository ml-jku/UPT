import einops
from kappamodules.layers import LinearProjection
from kappautils.param_checking import to_ntuple
from kappamodules.unet import UnetGino

from models.base.single_model_base import SingleModelBase


class UnetGinoModel(SingleModelBase):
    """ Unet model from GINO """

    def __init__(self, dim, depth=4, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth

        # propagate output_shape
        seqlen, input_dim = self.input_shape
        self.output_shape = (seqlen, dim)

        self.unet = UnetGino(
            input_dim=input_dim,
            hidden_dim=dim,
            depth=depth,
        )

    def forward(self, x, condition=None):
        assert condition is None
        # dim last without spatial -> dim first with spatial
        x = x.reshape(len(x), *self.static_ctx["grid_resolution"], -1)
        x = einops.rearrange(x, "batch_size ... dim -> batch_size dim ...")
        # unet
        x = self.unet(x)
        # dim first with spatial -> dim last without spatial
        x = einops.rearrange(x, "batch_size dim ... -> batch_size (...) dim")
        return x
