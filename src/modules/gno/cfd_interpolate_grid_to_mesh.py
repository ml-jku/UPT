import einops
import torch
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_scatter import segment_csr
import torch.nn.functional as F

class CfdInterpolateGridToMesh(nn.Module):
    @staticmethod
    def forward(x, query_pos):
        assert torch.all(query_pos.abs() <= 1)
        if query_pos.ndim == 3:
            # query_pos.shape: (batch_size, num_query_pos, ndim)
            # grid_sample requires 4d dense tensor
            query_pos = einops.rearrange(
                query_pos,
                "batch_size num_query_pos ndim -> batch_size num_query_pos 1 ndim",
            )
        else:
            raise NotImplementedError

        # interpolate to mesh
        # x.shape: (batch_size, dim, height, width, depth)
        query_pos = torch.stack(list(reversed(query_pos.unbind(-1))), dim=-1)
        x = F.grid_sample(input=x, grid=query_pos, align_corners=False)
        # to sparse tensor
        x = einops.rearrange(x, "batch_size dim num_query_pos 1 -> (batch_size num_query_pos) dim ")

        return x
