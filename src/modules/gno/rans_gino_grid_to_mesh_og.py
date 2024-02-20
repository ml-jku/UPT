import einops
import torch
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_scatter import segment_csr


class RansGinoGridToMeshOg(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            bottleneck_dim,
            output_dim,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim


        # GINO concats a constant 1 as 4th dimension for some reason
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim * 4, ndim=4)
        # "gno1"
        self.message = nn.Sequential(
            nn.Linear(input_dim + 4 * hidden_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, bottleneck_dim),
        )

        self.pred = nn.Sequential(
            nn.Linear(bottleneck_dim, 256),
            nn.GELU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x, query_pos, grid_to_query_edges):
        assert query_pos.ndim == 2
        assert grid_to_query_edges.ndim == 2
        # NOTE: we rescale all positions to [0, 200] instead of [-1, 1] -> revert
        query_pos = query_pos / 100 - 1

        # convert to sparse tensor
        x = einops.rearrange(x, "batch_size seqlen dim -> (batch_size seqlen) dim")

        # embed mesh
        # original implementation adds a 4th dimension with constant 1 during training
        ones = torch.ones(size=(len(query_pos),), dtype=query_pos.dtype, device=query_pos.device).unsqueeze(1)
        query_pos = torch.concat([query_pos, ones], dim=1)
        query_pos = self.pos_embed(query_pos)

        # create message input
        query_idx, grid_idx = grid_to_query_edges.unbind(1)
        x = torch.concat([x[grid_idx], query_pos[query_idx]], dim=1)
        x = self.message(x)
        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = query_idx.unique(return_counts=True)
        # first index has to be 0 + add padding for target indices that dont occour
        padded_counts = torch.zeros(len(query_pos) + 1, device=counts.device, dtype=counts.dtype)
        padded_counts[dst_indices + 1] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = segment_csr(src=x, indptr=indptr, reduce="mean")

        #
        x = self.pred(x)

        return x
