import einops
import torch
from torch_scatter import segment_csr

from .gino_grid_to_mesh import GinoGridToMesh


class RansGinoLatentToMesh(GinoGridToMesh):
    # noinspection PyMethodOverriding
    def forward(self, x, query_pos):
        assert query_pos.ndim == 3

        # convert to sparse tensor
        _, seqlen, _ = x.shape
        x = einops.rearrange(x, "batch_size seqlen dim -> (batch_size seqlen) dim")
        x = self.proj(x)

        # convert to sparse tensor
        query_pos = einops.rearrange(
            query_pos,
            "batch_size num_query_points ndim -> (batch_size num_query_points) ndim",
        )
        query_pos = self.pos_embed(query_pos)

        # create message input
        query_idx = torch.arange(len(query_pos), device=x.device).repeat_interleave(seqlen)
        latent_idx = torch.arange(seqlen, device=x.device).repeat(len(query_pos))
        x = torch.concat([x[latent_idx], query_pos[query_idx]], dim=1)
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
