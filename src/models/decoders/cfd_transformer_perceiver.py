from functools import partial

import einops
import numpy as np
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock, DitBlock
from kappamodules.vit import VitBlock
from torch import nn
from torch_geometric.utils import unbatch

from models.base.single_model_base import SingleModelBase


class CfdTransformerPerceiver(SingleModelBase):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            use_last_norm=False,
            perc_dim=None,
            perc_num_attn_heads=None,
            drop_path_rate=0.0,
            clamp=None,
            clamp_mode="log",
            init_weights="xavier_uniform",
            **kwargs,
    ):
        super().__init__(**kwargs)
        perc_dim = perc_dim or dim
        perc_num_attn_heads = perc_num_attn_heads or num_attn_heads
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.perc_dim = perc_dim
        self.perc_num_attn_heads = perc_num_attn_heads
        self.use_last_norm = use_last_norm
        self.drop_path_rate = drop_path_rate
        self.clamp = clamp
        self.clamp_mode = clamp_mode
        self.init_weights = init_weights

        # input/output shape
        _, num_channels = self.output_shape
        seqlen, input_dim = self.input_shape

        # input projection
        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # blocks
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = VitBlock
        self.blocks = nn.ModuleList([
            block_ctor(dim=dim, num_heads=num_attn_heads, init_weights=init_weights, drop_path=drop_path_rate)
            for _ in range(self.depth)
        ])

        # query tokens (create them from a positional embedding)
        self.pos_embed = ContinuousSincosEmbed(dim=perc_dim, ndim=2)
        self.query_mlp = nn.Sequential(
            LinearProjection(perc_dim, perc_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim * 4, perc_dim * 4, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(perc_dim * 4, perc_dim, init_weights=init_weights),
        )

        # latent to pixels
        self.perc_proj = LinearProjection(dim, perc_dim, init_weights=init_weights)
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitPerceiverBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PerceiverBlock
        self.perceiver = block_ctor(dim=perc_dim, num_heads=perc_num_attn_heads, init_weights=init_weights)
        self.norm = nn.LayerNorm(perc_dim, eps=1e-6) if use_last_norm else nn.Identity()
        self.pred = LinearProjection(perc_dim, num_channels, init_weights=init_weights)

    def forward(self, x, query_pos, unbatch_idx, unbatch_select, static_tokens=None, condition=None):
        assert x.ndim == 3

        # input projection
        x = self.input_proj(x)

        # apply blocks
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        for blk in self.blocks:
            x = blk(x, **block_kwargs)

        # create query
        pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(pos_embed)

        # latent to pixels
        x = self.perc_proj(x)
        x = self.perceiver(q=query, kv=x, **block_kwargs)
        x = self.norm(x)
        x = self.pred(x)

        if self.clamp is not None:
            assert self.clamp_mode == "log"
            # TODO torch.log1p should be equivalent
            x = torch.sign(x) * (self.clamp + torch.log(1 + x.abs()) - np.log(1 + self.clamp))

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        unbatched = unbatch(x, batch=unbatch_idx)
        x = torch.concat([unbatched[i] for i in unbatch_select])

        return x
