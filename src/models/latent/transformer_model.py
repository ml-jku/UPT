from functools import partial

import torch
from kappamodules.layers import LinearProjection
from kappamodules.transformer import DitBlock, PrenormBlock
from torch import nn

from models.base.single_model_base import SingleModelBase


class TransformerModel(SingleModelBase):
    def __init__(
            self,
            dim,
            depth,
            num_attn_heads,
            drop_path_rate=0.0,
            drop_path_decay=True,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.init_weights = init_weights
        self.init_last_proj_zero = init_last_proj_zero

        # input/output shape
        assert len(self.input_shape) == 2
        seqlen, input_dim = self.input_shape
        self.output_shape = (seqlen, dim)

        self.input_proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # blocks
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock
        if drop_path_decay:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth
        self.blocks = nn.ModuleList([
            block_ctor(
                dim=dim,
                num_heads=num_attn_heads,
                drop_path=dpr[i],
                init_weights=init_weights,
                init_last_proj_zero=init_last_proj_zero,
            )
            for i in range(self.depth)
        ])

    def forward(self, x, condition=None, static_tokens=None):
        assert x.ndim == 3

        # concat static tokens
        if static_tokens is not None:
            x = torch.cat([static_tokens, x], dim=1)

        # input projection
        x = self.input_proj(x)

        # apply blocks
        blk_kwargs = dict(cond=condition) if condition is not None else dict()
        for blk in self.blocks:
            x = blk(x, **blk_kwargs)

        # remove static tokens
        if static_tokens is not None:
            num_static_tokens = static_tokens.size(1)
            x = x[:, num_static_tokens:]

        return x
