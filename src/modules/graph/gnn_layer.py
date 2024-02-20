import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing


class GNNLayer(MessagePassing):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.Linear(2 * input_dim + 1, hidden_dim),
            nn.SiLU(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
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
