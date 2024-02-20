import torch
from functools import partial

import einops
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock
from kappamodules.vit import DitBlock
from torch import nn

from models.base.single_model_base import SingleModelBase
from modules.gno.rans_posembed_message import RansPosembedMessage
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from torch_geometric.utils import to_dense_batch


class RansGnnTransformerPerceiver(SingleModelBase):
    def __init__(
            self,
            gnn_dim,
            enc_dim,
            perc_dim,
            enc_depth,
            enc_num_attn_heads,
            perc_num_attn_heads,
            num_output_tokens,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.perc_dim = perc_dim
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.perc_num_attn_heads = perc_num_attn_heads
        self.num_output_tokens = num_output_tokens

        # set ndim
        _, ndim = self.input_shape
        self.static_ctx["ndim"] = ndim

        # gnn
        self.gnn = RansPosembedMessage(dim=gnn_dim, ndim=ndim)

        # transformer
        self.transformer_proj = LinearProjection(gnn_dim, enc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        self.transformer_blocks = nn.ModuleList([
            block_ctor(dim=enc_dim, num_heads=enc_num_attn_heads)
            for _ in range(enc_depth)
        ])

        # perceiver
        self.perceiver_proj = LinearProjection(enc_dim, perc_dim)
        self.perceiver_pooling = PerceiverPoolingBlock(
            dim=perc_dim,
            num_heads=perc_num_attn_heads,
            num_query_tokens=num_output_tokens,
        )

        # output shape
        self.output_shape = (num_output_tokens, perc_dim)

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="perceiver_pooling.query")]

    def forward(self, mesh_pos, mesh_edges, batch_idx):
        # gnn
        x = self.gnn(mesh_pos=mesh_pos, mesh_edges=mesh_edges)
        x, mask = to_dense_batch(x, batch_idx)
        if torch.all(mask):
            mask = None
        else:
            # add dimensions for num_heads and query (keys are masked)
            mask = einops.rearrange(mask, "batchsize num_nodes -> batchsize 1 1 num_nodes")

        # transformer
        x = self.transformer_proj(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, attn_mask=mask)

        # perceiver
        x = self.perceiver_proj(x)
        x = self.perceiver_pooling(kv=x, attn_mask=mask)

        return x
