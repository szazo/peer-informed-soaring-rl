import torch


# https://www.geeksforgeeks.org/initialize-weights-in-pytorch/
def initialize_t_fixup_weights(p: torch.Tensor, layer_count: int):

    with torch.no_grad():
        torch.nn.init.xavier_normal_(p)
        p.data = p * (0.67 * layer_count**-0.25)
