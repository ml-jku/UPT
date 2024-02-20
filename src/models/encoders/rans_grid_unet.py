import einops
import torch
from torch import nn

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from kappamodules.vit import VitPatchEmbed, VitPosEmbed
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock
from kappamodules.unet import UnetGino
from kappamodules.layers import LinearProjection


class RansGridUnet(SingleModelBase):
    def __init__(
            self,
            dim,
            num_attn_heads,
            num_output_tokens,
            depth=4,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens
        self.resolution = self.data_container.get_dataset().grid_resolution
        self.ndim = len(self.resolution)
        # sdf + grid_pos
        if self.data_container.get_dataset().concat_pos_to_sdf:
            input_dim = 4
        else:
            input_dim = 1

        assert dim % depth == 0
        self.unet = UnetGino(
            input_dim=input_dim,
            hidden_dim=dim // depth,
            output_dim=dim,
            depth=depth,
        )
        self.perceiver = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
            perceiver_kwargs=dict(init_weights="truncnormal"),
        )

        self.type_token = nn.Parameter(torch.empty(size=(1, 1, dim,)))

        self.static_ctx["grid_resolution"] = self.resolution
        self.static_ctx["ndim"] = self.ndim
        self.output_shape = (num_output_tokens, dim)

    def model_specific_initialization(self):
        nn.init.trunc_normal_(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="type_token")]

    def forward(self, x):
        # sdf is passed as dim-last with spatial -> convert to dim-first with spatial
        x = einops.rearrange(x, "batch_size height width depth dim -> batch_size dim height width depth")
        # embed
        x = self.unet(x)
        # perceiver
        x = einops.rearrange(x, "batch_size dim height width depth -> batch_size (height width depth) dim")
        x = self.perceiver(x)
        x = x + self.type_token
        return x
