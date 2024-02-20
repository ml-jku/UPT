import torch
from kappamodules.init.functional import init_xavier_uniform_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_geometric.nn.conv import MessagePassing


class RansPosembedMessage(MessagePassing):
    def __init__(self, dim, ndim):
        super().__init__(aggr="mean")
        self.dim = dim

        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.message_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, mesh_pos, mesh_edges):
        x = self.pos_embed(mesh_pos)
        x = self.propagate(x=x, pos=mesh_pos, edge_index=mesh_edges.T)
        return x

    # noinspection PyMethodOverriding
    def message(self, x_i, x_j, pos_i, pos_j):
        return self.message_net(torch.cat([x_i, x_j], dim=-1))

    # noinspection PyMethodOverriding
    def update(self, message, x):
        return x + self.update_net(torch.cat([x, message], dim=-1))

    def message_and_aggregate(self, adj_t):
        raise NotImplementedError

    def edge_update(self):
        raise NotImplementedError
