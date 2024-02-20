from functools import partial

import torch
from kappamodules.layers import LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock, DitPerceiverPoolingBlock
from kappamodules.vit import DitBlock
from torch import nn
from torch_geometric.utils import to_dense_batch

from models.base.single_model_base import SingleModelBase
from modules.gno.cfd_pool import CfdPool
from modules.gno.cfd_pool_gaussian_sincos_pos import CfdPoolGaussianSincosPos
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class LagrangianPoolTransformerPerceiver(SingleModelBase):
    def __init__(
            self,
            gnn_dim,
            enc_dim,
            perc_dim,
            enc_depth,
            enc_num_attn_heads,
            perc_num_attn_heads,
            num_latent_tokens=None,
            use_enc_norm=False,
            init_weights="xavier_uniform",
            gnn_init_weights=None,
            positional_std=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.gnn_dim = gnn_dim
        self.enc_dim = enc_dim
        self.perc_dim = perc_dim
        self.enc_depth = enc_depth
        self.enc_num_attn_heads = enc_num_attn_heads
        self.perc_num_attn_heads = perc_num_attn_heads
        self.num_latent_tokens = num_latent_tokens
        self.use_enc_norm = use_enc_norm
        self.init_weights = init_weights
        gnn_init_weights = gnn_init_weights or init_weights
        self.gnn_init_weights = gnn_init_weights

        # input_shape is (None, input_dim)
        input_dim, _ = self.input_shape
        ndim = self.data_container.get_dataset().metadata["dim"]
        if positional_std is not None:
            self.mesh_embed = CfdPoolGaussianSincosPos(
                input_dim=input_dim,
                hidden_dim=gnn_dim,
                init_weights=gnn_init_weights,
                ndim=ndim,
                positional_std=positional_std
            )
        else:
            self.mesh_embed = CfdPool(
                input_dim=input_dim,
                hidden_dim=gnn_dim,
                init_weights=gnn_init_weights,
                ndim=ndim
            )

        # blocks
        self.enc_norm = nn.LayerNorm(gnn_dim, eps=1e-6) if use_enc_norm else nn.Identity()
        self.enc_proj = LinearProjection(gnn_dim, enc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        self.blocks = nn.ModuleList([
            block_ctor(dim=enc_dim, num_heads=enc_num_attn_heads, init_weights=init_weights)
            for _ in range(enc_depth)
        ])

        # perceiver pooling
        self.perc_proj = LinearProjection(enc_dim, perc_dim)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(
                DitPerceiverPoolingBlock,
                perceiver_kwargs=dict(
                    cond_dim=self.static_ctx["condition_dim"],
                    init_weights=init_weights,
                ),
            )
        else:
            block_ctor = partial(
                PerceiverPoolingBlock,
                perceiver_kwargs=dict(init_weights=init_weights),
            )
        self.perceiver = block_ctor(
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
        x = self.mesh_embed(x, mesh_pos=mesh_pos, mesh_edges=mesh_edges, batch_idx=batch_idx)

        # project static_tokens to encoder dim
        # static_tokens = self.static_token_proj(static_tokens)
        # concat static tokens
        # x = torch.cat([static_tokens, x], dim=1)

        # apply blocks
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        x = self.enc_norm(x)
        x = self.enc_proj(x)
        for blk in self.blocks:
            x = blk(x, **block_kwargs)

        # perceiver
        x = self.perc_proj(x)
        x = self.perceiver(kv=x, **block_kwargs)

        return x
