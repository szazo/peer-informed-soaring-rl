import numpy as np
import torch
from tianshou.data import to_torch


def convert_input_to_torch(input: torch.Tensor | np.ndarray,
                           device: torch.device | None):

    assert isinstance(input, np.ndarray) or isinstance(input, torch.Tensor)

    torch_input = to_torch(input,
                           dtype=torch.get_default_dtype(),
                           device=device)
    assert isinstance(input, np.ndarray) or torch_input is input
    assert isinstance(torch_input, torch.Tensor)

    return torch_input
