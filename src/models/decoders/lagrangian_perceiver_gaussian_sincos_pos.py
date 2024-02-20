from functools import partial

import einops
import torch
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock, DitPerceiverBlock
from torch import nn
from torch_geometric.utils import unbatch

from models.base.single_model_base import SingleModelBase


class LagrangianPerceiverGaussianSincosPos(SingleModelBase):
    def __init__(self, dim, num_attn_heads, positional_std,  **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads

        # input/output shape
        num_channels, _ = self.output_shape
        _, input_dim = self.input_shape
        ndim = self.data_container.get_dataset().metadata["dim"]

        # input projection
        self.proj = LinearProjection(input_dim, dim)

        # query tokens (create them from a positional embedding)
        # Try with other method from https://arxiv.org/pdf/2006.10739.pdf
        generator = torch.Generator().manual_seed(42)
        self.register_buffer(
            "b",
            torch.normal(mean=torch.zeros(dim // 2, ndim), std=positional_std, generator=generator)
        )
        self.query_mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # perceiver
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitPerceiverBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PerceiverBlock
        self.perceiver = block_ctor(dim=dim, num_heads=num_attn_heads)
        self.pred = LinearProjection(dim, num_channels)

    def model_specific_initialization(self):
        self.query_mlp.apply(init_xavier_uniform_zero_bias)

    def forward(self, x, query_pos, unbatch_idx, unbatch_select, static_tokens=None, condition=None):
        assert x.ndim == 3

        # input projection
        x = self.proj(x)

        # create query
        pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(pos_embed)

        # perceiver
        block_kwargs = {}
        if condition is not None:
            block_kwargs["cond"] = condition
        x = self.perceiver(q=query, kv=x, **block_kwargs)
        x = self.pred(x)

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        unbatched = unbatch(x, batch=unbatch_idx)
        x = torch.concat([unbatched[i] for i in unbatch_select])

        return x

    def pos_embed(self, pos):
        return torch.concat([torch.cos(2.0 * torch.pi * pos @ self.b.T), 
                             torch.sin(2.0 * torch.pi * pos @ self.b.T)], dim=-1)
