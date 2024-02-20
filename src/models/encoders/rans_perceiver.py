import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed
from kappamodules.transformer import PerceiverPoolingBlock, Mlp
from torch import nn
from torch_geometric.utils import to_dense_batch

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class RansPerceiver(SingleModelBase):
    def __init__(
            self,
            dim,
            num_attn_heads,
            num_output_tokens,
            add_type_token=False,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens
        self.add_type_token = add_type_token

        # set ndim
        _, ndim = self.input_shape
        self.static_ctx["ndim"] = ndim

        # pos_embed
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)

        # perceiver
        self.mlp = Mlp(in_dim=dim, hidden_dim=dim * 4, init_weights=init_weights)
        self.block = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
            perceiver_kwargs=dict(
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            ),
        )

        if add_type_token:
            self.type_token = nn.Parameter(torch.empty(size=(1, 1, dim,)))
        else:
            self.type_token = None

        # output shape
        self.output_shape = (num_output_tokens, dim)

    def model_specific_initialization(self):
        if self.add_type_token:
            nn.init.trunc_normal_(self.type_token)

    def get_model_specific_param_group_modifiers(self):
        modifiers = [ExcludeFromWdByNameModifier(name="block.query")]
        if self.add_type_token:
            modifiers += [ExcludeFromWdByNameModifier(name="type_token")]
        return modifiers

    def forward(self, mesh_pos, batch_idx, mesh_edges=None):
        x = self.pos_embed(mesh_pos)
        x, mask = to_dense_batch(x, batch_idx)
        if torch.all(mask):
            mask = None
        else:
            # add dimensions for num_heads and query (keys are masked)
            mask = einops.rearrange(mask, "batchsize num_nodes -> batchsize 1 1 num_nodes")

        # perceiver
        x = self.mlp(x)
        x = self.block(kv=x, attn_mask=mask)

        if self.add_type_token:
            x = x + self.type_token

        return x
