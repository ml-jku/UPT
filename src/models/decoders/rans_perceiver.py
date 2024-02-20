import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverBlock, Mlp
from torch_geometric.utils import unbatch
from torch import nn
from models.base.single_model_base import SingleModelBase


class RansPerceiver(SingleModelBase):
    def __init__(
            self,
            dim,
            num_attn_heads,
            init_weights="xavier_uniform",
            init_last_proj_zero=False,
            use_last_norm=False,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_attn_heads = num_attn_heads
        self.use_last_norm = use_last_norm

        # input projection
        _, input_dim = self.input_shape
        self.proj = LinearProjection(input_dim, dim, init_weights=init_weights)

        # query tokens (create them from a positional embedding)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=self.static_ctx["ndim"])
        self.query_mlp = Mlp(in_dim=dim, hidden_dim=dim, init_weights=init_weights)

        # latent to pixels
        self.perceiver = PerceiverBlock(
            dim=dim,
            num_heads=num_attn_heads,
            init_last_proj_zero=init_last_proj_zero,
            init_weights=init_weights,
        )
        _, output_dim = self.output_shape
        self.norm = nn.LayerNorm(dim, eps=1e-6) if use_last_norm else nn.Identity()
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

    def forward(self, x, query_pos, unbatch_idx, unbatch_select):
        # input projection
        x = self.proj(x)

        # create query
        query_pos_embed = self.pos_embed(query_pos)
        query = self.query_mlp(query_pos_embed)

        # decode
        x = self.perceiver(q=query, kv=x)
        x = self.norm(x)
        x = self.pred(x)

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor (batch_size * num_points, dim)
        x = einops.rearrange(x, "batch_size max_num_points dim -> (batch_size max_num_points) dim")
        unbatched = unbatch(x, batch=unbatch_idx)
        x = torch.concat([unbatched[i] for i in unbatch_select])

        return x
