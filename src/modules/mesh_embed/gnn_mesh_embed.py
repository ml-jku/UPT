import torch
from kappamodules.layers import ContinuousSincosEmbed
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.pool import SAGPooling
from modules.graph.sag_pool import SAGPoolingFixedNumNodes


class GNNLayer(MessagePassing):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(2 * in_features + 1, hidden_features),
            nn.SiLU(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(in_features + hidden_features, hidden_features),
            nn.SiLU(),
        )

    def forward(self, x, pos, edge_index):
        """ Propagate messages along edges """
        x = self.propagate(edge_index, x=x, pos=pos)
        return x

    # noinspection PyMethodOverriding
    def message(self, x_i, x_j, pos_i, pos_j):
        """ Message update """
        msg_input = torch.cat((x_i, x_j, torch.sqrt(torch.sum((pos_i - pos_j) ** 2, dim=1)).unsqueeze(dim=1)), dim=-1)
        message = self.message_net(msg_input)
        return message

    # noinspection PyMethodOverriding
    def update(self, message, x, pos):
        """ Node update """
        x = x + self.update_net(torch.cat((x, message), dim=-1))
        return x

    def message_and_aggregate(self, adj_t):
        raise NotImplementedError

    def edge_update(self):
        raise NotImplementedError


class GNNMeshEmbed(torch.nn.Module):
    def __init__(
            self,
            in_features=3,
            out_features=None,
            hidden_features=32,
            use_gnn=True,
            pool_ratio=None,
            num_output_nodes=None,
    ):
        super().__init__()
        assert (pool_ratio is None) ^ (num_output_nodes is None),\
            "GnnMeshEmbed requires pool_ratio or num_output_nodes"
        self.in_features = in_features
        self.out_features = out_features or hidden_features
        self.hidden_features = hidden_features
        self.use_gnn = use_gnn
        self.pool_ratio = pool_ratio
        self.num_output_nodes = num_output_nodes

        if use_gnn:
            self.gnn_layer = GNNLayer(in_features=self.hidden_features, hidden_features=self.hidden_features)
        else:
            self.gnn_layer = None

        self.pos_embed = ContinuousSincosEmbed(dim=self.hidden_features, ndim=2)
        self.embedding_proj = nn.Linear(self.in_features, self.hidden_features)
        self.output_proj = nn.Linear(self.hidden_features, self.out_features)

        if num_output_nodes is not None:
            self.pool = SAGPoolingFixedNumNodes(self.hidden_features, num_output_nodes=self.num_output_nodes)
        else:
            self.pool = SAGPooling(self.hidden_features, ratio=pool_ratio)

    def forward(self, x, pos, edge_index, batch_idx):
        # First map node features (v_x, v_y, p) to hidden_space (same size as latent space of the transformer?)
        x = self.embedding_proj(x)
        x = x + self.pos_embed(pos)
        if self.gnn_layer is not None:
            x = self.gnn_layer(x, pos, edge_index.T)

        # pool + project
        pool_result = self.pool(x, edge_index.T, batch=batch_idx)
        # x_pool, edge_index_pool, edge_attr_pool, batch_pool, perm, score = pool_result
        x_pool, _, _, batch_pool, _, _ = pool_result
        x_pool = self.output_proj(x_pool)

        return x_pool, batch_pool
