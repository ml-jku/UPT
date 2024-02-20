from functools import partial

import torch
from kappamodules.init.functional import init_xavier_uniform_zero_bias, init_truncnormal_zero_bias
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_geometric.nn.conv import MessagePassing

from modules.graph.sag_pool import SAGPoolingFixedNumNodes


class CfdGnnPool(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            num_output_nodes,
            depth=1,
            ndim=2,
            norm="none",
            init_weights="xavier_uniform",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_output_nodes = num_output_nodes
        self.depth = depth
        self.norm = norm
        self.ndim = ndim
        self.init_weights = init_weights

        self.proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        if norm == "none":
            norm_ctor = nn.Identity
        else:
            raise NotImplementedError
        self.gnn_layers = nn.ModuleList([
            self.CfdGnnPoolMessagePassing(
                message_net=nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    norm_ctor(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
                update_net=nn.Sequential(
                    nn.Linear(2 * hidden_dim, hidden_dim),
                    norm_ctor(hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                ),
            )
            for _ in range(depth)
        ])
        self.pool = SAGPoolingFixedNumNodes(hidden_dim, num_output_nodes=num_output_nodes, aggr="mean")
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "xavier_uniform":
            self.apply(init_xavier_uniform_zero_bias)
        elif self.init_weights == "truncnormal":
            self.apply(init_truncnormal_zero_bias)
        else:
            raise NotImplementedError

    def forward(self, x, mesh_pos, mesh_edges, batch_idx):
        # embed + GNN
        x = self.proj(x)
        x = x + self.pos_embed(mesh_pos)
        for gnn_layer in self.gnn_layers:
            x = gnn_layer(mesh_edges=mesh_edges.T, x=x, pos=mesh_pos)

        # pool
        pool_result = self.pool(x, mesh_edges.T, batch=batch_idx)
        # x_pool, edge_index_pool, edge_attr_pool, batch_pool, perm, score = pool_result
        x_pool, _, _, batch_pool, _, _ = pool_result

        return x_pool, batch_pool

    class CfdGnnPoolMessagePassing(MessagePassing):
        def __init__(self, message_net, update_net):
            super().__init__(aggr="mean")
            self.message_net = message_net
            self.update_net = update_net

        def forward(self, mesh_edges, x, pos):
            return self.propagate(edge_index=mesh_edges, x=x, pos=pos)

        # noinspection PyMethodOverriding
        def message(self, x_i, x_j, pos_i, pos_j):
            msg_input = torch.cat([x_i, x_j], dim=1)
            message = self.message_net(msg_input)
            return message

        # noinspection PyMethodOverriding
        def update(self, message, x, pos):
            x = x + self.update_net(torch.cat([x, message], dim=1))
            return x

        def message_and_aggregate(self, adj_t):
            raise NotImplementedError

        def edge_update(self):
            raise NotImplementedError
