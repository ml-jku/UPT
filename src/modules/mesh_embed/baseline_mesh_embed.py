import einops
import torch
from kappamodules.init import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn

from modules.graph.gnn_layer import GNNLayer


class BaselineMeshEmbed(nn.Module):
    def __init__(self, dim, depth, resolution, input_dim):
        super().__init__()
        self.dim = dim
        assert depth >= 1
        self.depth = depth
        assert len(resolution) == 2
        self.resolution = resolution
        self.num_grid_points = self.resolution[0] * self.resolution[1]
        self.register_buffer("grid_points_arange", torch.arange(self.num_grid_points), persistent=False)

        self.proj = nn.Linear(input_dim, dim)
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=len(resolution))
        self.pos_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim=dim, hidden_dim=dim)
            for _ in range(depth)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        init_xavier_uniform_zero_bias(self.proj)
        self.pos_mlp.apply(init_xavier_uniform_zero_bias)

    def forward(self, x, pos, batch_idx, edge_index):
        # get indices of grid nodes
        _, counts = batch_idx.unique(return_counts=True)
        start = (counts.cumsum(dim=0) - counts[0]).repeat_interleave(self.num_grid_points)
        grid_pos_idx = self.grid_points_arange.repeat(len(counts)) + start

        # project input to dim
        x = self.proj(x)

        # add pos embedding
        pos_embed = self.pos_embed(pos)
        # initialze grid nodes with MLP(pos_embed)
        x[grid_pos_idx] = self.pos_mlp(pos_embed[grid_pos_idx])
        # add pos_embed
        x = x + pos_embed

        # message passing
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(x, pos, edge_index.T)

        # select grid nodes
        x = x[grid_pos_idx]

        # convert to dense tensor
        x = einops.rearrange(
            x,
            "(batch_size num_grid_points) dim -> batch_size num_grid_points dim",
            num_grid_points=self.num_grid_points,
        )
        return x
