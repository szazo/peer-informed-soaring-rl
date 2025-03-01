import torch
from torch.types import Device


def create_causal_attention_mask(sequence_length: int,
                                 reverse: bool = False,
                                 device: Device | None = None):

    # source is the row, target is the column
    i = torch.arange(0, sequence_length, device=device).reshape(-1, 1)
    j = torch.arange(0, sequence_length, device=device)

    # if the sequence is reversed, use the transpose
    mask = i < j if not reverse else j < i

    out = torch.zeros((sequence_length, sequence_length), device=device)
    out[mask] = -torch.inf

    return out
