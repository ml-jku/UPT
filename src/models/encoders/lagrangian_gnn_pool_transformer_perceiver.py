from functools import partial

import einops
import torch
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock
from kappamodules.layers import LinearProjection
from kappamodules.vit import DitBlock
from torch import nn
from torch_geometric.utils import to_dense_batch

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier
from modules.gno.cfd_gnn_pool import CfdGnnPool


class LagrangianGnnPoolTransformerPerceiver(SingleModelBase):
    def __init__(
            self,
            gnn_dim,
            enc_dim,
            perc_dim,
            gnn_depth,
            enc_depth,
            enc_num_attn_heads,
            perc_num_attn_heads,
            num_supernodes,
            num_latent_tokens=None,
            gnn_norm="none",
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.perc_dim = perc_dim
        self.gnn_depth = gnn_depth
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.perc_num_attn_heads = perc_num_attn_heads
        self.num_supernodes = num_supernodes
        self.num_latent_tokens = num_latent_tokens

        # input_shape is (input_dim, None)
        input_dim, _ = self.input_shape
        ndim = self.data_container.get_dataset().metadata["dim"]
        self.mesh_embed = CfdGnnPool(
            input_dim=input_dim,
            hidden_dim=gnn_dim,
            depth=gnn_depth,
            num_output_nodes=num_supernodes,
            norm=gnn_norm,
            ndim=ndim
        )

        # blocks
        self.enc_proj = LinearProjection(gnn_dim, enc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        self.blocks = nn.ModuleList([
            block_ctor(dim=enc_dim, num_heads=enc_num_attn_heads)
            for _ in range(enc_depth)
        ])

        # perceiver pooling
        self.perc_proj = LinearProjection(enc_dim, perc_dim)
        self.perceiver = PerceiverPoolingBlock(
            dim=perc_dim,
            num_heads=perc_num_attn_heads,
            num_query_tokens=num_latent_tokens,
        )

        # output shape
        self.output_shape = (num_latent_tokens, perc_dim)

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name="perceiver.query")]

    def forward(self, x, mesh_pos, mesh_edges, batch_idx, condition=None, static_tokens=None):
        # embed mesh
        x, batch_idx_pooled = self.mesh_embed(x, mesh_pos=mesh_pos, mesh_edges=mesh_edges, batch_idx=batch_idx)
        x, mask = to_dense_batch(x, batch_idx_pooled)
        assert torch.all(mask)

        # project static_tokens to encoder dim
        # static_tokens = self.static_token_proj(static_tokens)
        # concat static tokens
        # x = torch.cat([static_tokens, x], dim=1)

        # apply blocks
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x, **block_kwargs)

        # perceiver
        x = self.perc_proj(x)
        x = self.perceiver(kv=x)

        return x