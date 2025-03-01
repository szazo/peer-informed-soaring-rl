import math
import torch
from torch.types import Device


class PositionalEncoding(torch.nn.Module):

    def __init__(self,
                 dimension: int,
                 max_sequence_length: int,
                 device: Device,
                 n=10000):

        if dimension % 2 != 0:
            raise Exception("Dimension should be even")

        super().__init__()

        position = torch.arange(max_sequence_length).unsqueeze(1)
        denominator_term = torch.exp(
            torch.arange(0, dimension, 2) * (-math.log(n) / dimension))

        pe = torch.zeros(max_sequence_length, dimension, device=device)
        pe[:, 0::2] = torch.sin(position * denominator_term)
        pe[:, 1::2] = torch.cos(position * denominator_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor, shape [..., seq_len, embedding_dim]
        """
        seq_len = x.size(-2)
        x = x + self.pe[:seq_len]
        return x
