from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.select import Select, SelectOutput
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import cumsum, scatter


def topk(
        x: Tensor,
        num_output_nodes: int,
        batch: Tensor,
) -> Tensor:
    num_nodes = scatter(batch.new_ones(x.size(0)), batch, reduce='sum')
    k = torch.full(size=(len(num_nodes),), fill_value=num_output_nodes, device=x.device, dtype=torch.long)
    assert torch.all(num_nodes >= k), "num_nodes has to be at >= num_outputs_nodes"

    x, x_perm = torch.sort(x.view(-1), descending=True)
    batch = batch[x_perm]
    batch, batch_perm = torch.sort(batch, descending=False, stable=True)

    arange = torch.arange(x.size(0), dtype=torch.long, device=x.device)
    ptr = cumsum(num_nodes)
    batched_arange = arange - ptr[batch]
    mask = batched_arange < k[batch]

    return x_perm[batch_perm[mask]]


class SelectTopK(Select):
    """
    torch_geometrics.nn.pool.select.topk with a dynamic ratio such that the number of output nodes is constant
    also removed the parameter "weight" that allowed to learn a weighted sum
    """

    def __init__(
            self,
            in_channels: int,
            num_output_nodes: int = 5,
            act: Union[str, Callable] = "tanh",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_output_nodes = num_output_nodes
        self.act = activation_resolver(act)

    def forward(
            self,
            x: Tensor,
            batch: Optional[Tensor] = None,
    ) -> SelectOutput:
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = x.view(-1, 1) if x.dim() == 1 else x
        score = self.act(x.sum(dim=-1))
        node_index = topk(score, self.num_output_nodes, batch)

        return SelectOutput(
            node_index=node_index,
            num_nodes=x.size(0),
            cluster_index=torch.arange(node_index.size(0), device=x.device),
            num_clusters=node_index.size(0),
            weight=score[node_index],
        )

    def __repr__(self) -> str:
        arg = f"num_output_nodes={self.num_output_nodes}"
        return f"{self.__class__.__name__}({self.in_channels}, {arg})"
