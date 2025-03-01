import torch


def collect_nonzero_indices_along_axis(input: torch.Tensor, zero_value: float,
                                       dim: int):
    """
    Collect indices along the specified `dim` dimension where values in `input` are not
    entirely equal to `target_value`.
    The input should be in ... x sequence x dim shape.
    """

    sequence_size = input.shape[dim - 1]

    zero_rows_mask = create_zero_rows_mask(input,
                                           zero_value=zero_value,
                                           dim=dim)
    nonzero_rows_mask = (~zero_rows_mask).to(torch.int64)

    # find the index of the first 1 along the axis
    first_indices = torch.argmax(nonzero_rows_mask, dim)

    # for finding the last index, need to flip
    flipped = torch.flip(nonzero_rows_mask, dims=(dim, ))
    last_indices = torch.argmax(flipped, dim)
    last_indices = sequence_size - last_indices - 1

    return first_indices, last_indices


def create_zero_rows_mask(input: torch.Tensor, zero_value: float, dim: int):
    if torch.isnan(torch.tensor(zero_value)):
        mask = torch.isnan(input)
    else:
        mask = input == zero_value

    zero_rows_mask = torch.all(mask, dim=dim)

    return zero_rows_mask


def gather_based_on_indices_along_axis(input: torch.Tensor,
                                       indices: torch.Tensor):
    """Gather items along an axis based on the provided index,
       The input should be in the B0 x ... x Bn x x sequence x dim shape,
       while the index should select items from the sequence dimension,
       so it is in B0 x ... x Bn shape and selects the item from the sequence axis.
    """

    # add the sequence x dim shapes at the end
    target_shape = list(indices.shape) + [1] * 2

    assert input.ndim == len(
        target_shape), f'indices shape is invalid: {indices.shape}'
    indices = indices.view(target_shape)

    # expand to the size of the last dimension (dim)
    indices = indices.expand(*indices.shape[:-1], input.size(-1))

    # gather along the 'sequence' dim
    result = torch.gather(input, dim=-2, index=indices)

    return result
