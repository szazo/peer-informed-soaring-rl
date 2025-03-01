import torch
from torch.types import Device


def initialize_weights(p, embedding_dimension: int, layer_count: int):
    torch.nn.init.normal_(p, 0., embedding_dimension**-0.5)

    p.data = p * (9 * layer_count)**-0.25


class Embedding(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int, layer_count: int,
                 device: Device):

        super().__init__()

        self._device = device

        self._linear = torch.nn.Linear(in_features=in_features,
                                       out_features=out_features,
                                       device=device)
        initialize_weights(self._linear.weight,
                           embedding_dimension=out_features,
                           layer_count=layer_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        output = self._linear(x)
        return output
