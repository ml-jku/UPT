import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed
from kappamodules.transformer import PerceiverPoolingBlock, Mlp
from torch_geometric.utils import to_dense_batch

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class LagrangianPerceiver(SingleModelBase):
    def __init__(self, dim, num_attn_heads, num_output_tokens, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.num_output_tokens = num_output_tokens

        ndim = self.data_container.get_dataset().metadata["dim"]
        self.static_ctx["ndim"] = ndim

        # input_embed
        self.embed = Mlp(in_dim=self.input_shape[0], hidden_dim=dim, out_dim=dim)

        # pos_embed
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)

        # perceiver
        self.mlp = Mlp(in_dim=dim, hidden_dim=dim * 4)
        self.block = PerceiverPoolingBlock(
            dim=dim,
            num_heads=num_attn_heads,
            num_query_tokens=num_output_tokens,
        )

        # output shape
        self.output_shape = (num_output_tokens, dim)

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="block.query")]

    def forward(self, x, pos, batch_idx):
        x = self.embed(x) + self.pos_embed(pos)
        x, mask = to_dense_batch(x, batch_idx)
        if torch.all(mask):
            mask = None
        else:
            # add dimensions for num_heads and query (keys are masked)
            mask = einops.rearrange(mask, "batchsize num_nodes -> batchsize 1 1 num_nodes")

        # perceiver
        x = self.mlp(x)
        x = self.block(kv=x, attn_mask=mask)

        return x