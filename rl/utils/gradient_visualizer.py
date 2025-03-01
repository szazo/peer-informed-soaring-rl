from typing import Any
import torch
from torchviz import make_dot


def render_gradient(output_tensor: torch.Tensor, params: dict[str, Any],
                    filename: str):
    make_dot(output_tensor, params=params).render("graph", format="png")
