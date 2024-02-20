import einops
import torch
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_scatter import segment_csr


class RansGinoGridToMesh(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, ndim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ndim = ndim

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        self.message = nn.Sequential(
            nn.Linear(2 * hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pred = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x, query_pos, grid_to_query_edges):
        assert query_pos.ndim == 2
        assert grid_to_query_edges.ndim == 2

        # convert to sparse tensor
        x = einops.rearrange(x, "batch_size seqlen dim -> (batch_size seqlen) dim")
        x = self.proj(x)

        # embed mesh
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
