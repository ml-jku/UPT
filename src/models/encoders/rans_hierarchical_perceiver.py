from functools import partial

import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed, LinearProjection
from kappamodules.transformer import PerceiverPoolingBlock, PrenormBlock
from kappamodules.vit import DitBlock
from kappautils.param_checking import to_ntuple
from torch import nn
from torch_geometric.utils import to_dense_batch

from models.base.single_model_base import SingleModelBase
from optimizers.param_group_modifiers.exclude_from_wd_by_name_modifier import ExcludeFromWdByNameModifier


class RansHierarchicalPerceiver(SingleModelBase):
    def __init__(
            self,
            num_stages,
            dim,
            depth,
            num_attn_heads,
            num_query_tokens,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_stages = num_stages
        self.dim = to_ntuple(dim, n=num_stages * 2)
        self.depth = to_ntuple(depth, n=num_stages)
        self.num_attn_heads = to_ntuple(num_attn_heads, n=num_stages * 2)
        self.num_query_tokens = to_ntuple(num_query_tokens, n=num_stages)

        # set ndim
        _, ndim = self.input_shape
        self.static_ctx["ndim"] = ndim

        # pos_embed
        self.pos_embed = ContinuousSincosEmbed(dim=self.dim[0], ndim=ndim)

        # ctors
        if "condition_dim" in self.static_ctx:
            block_ctor = partial(DitBlock, cond_dim=self.static_ctx["condition_dim"])
        else:
            block_ctor = PrenormBlock

        self.models = nn.ModuleList()
        for i in range(num_stages):
            stage_models = nn.ModuleList()
            # projection
            if i == 0:
                stage_models.append(nn.Identity())
            else:
                stage_models.append(LinearProjection(self.dim[(i - 1) * 2 + 1], self.dim[i * 2]))
            # transformer
            stage_models.append(
                nn.ModuleList([
                    block_ctor(dim=self.dim[i * 2], num_heads=self.num_attn_heads[i * 2])
                    for _ in range(self.depth[i])
                ]),
            )
            # projection
            stage_models.append(LinearProjection(self.dim[i * 2], self.dim[i * 2 + 1]))
            # perceiver
            stage_models.append(
                PerceiverPoolingBlock(
                    dim=self.dim[i * 2 + 1],
                    num_heads=self.num_attn_heads[i * 2 + 1],
                    num_query_tokens=self.num_query_tokens[i],
                ),
            )
            self.models.append(stage_models)

        # output shape
        self.output_shape = (self.num_query_tokens[-1], self.dim[-1])

    def get_model_specific_param_group_modifiers(self):
        return [ExcludeFromWdByNameModifier(name=f"models.{i}.3.query") for i in range(self.num_stages)]

    def forward(self, mesh_pos, mesh_edges, batch_idx):
        x = self.pos_embed(mesh_pos)
        x, mask = to_dense_batch(x, batch_idx)
        if torch.all(mask):
            mask = None
        else:
            # add dimensions for num_heads and query (keys are masked)
            mask = einops.rearrange(mask, "batchsize num_nodes -> batchsize 1 1 num_nodes")

        for i, (proj1, blocks, proj2, pooling) in enumerate(self.models):
            block_kwargs = dict(attn_mask=mask) if i == 0 else dict()
            x = proj1(x)
            for block in blocks:
                x = block(x, **block_kwargs)
            x = proj2(x)
            x = pooling(x, **block_kwargs)

        return x
