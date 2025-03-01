import torch
from ..index_and_gather_nonzero_items_along_axis import (
    collect_nonzero_indices_along_axis, create_zero_rows_mask,
    gather_based_on_indices_along_axis)


def test_fill_zero_rows_should_fill():

    # given
    v = torch.nan

    input = torch.tensor([[v, v, v], [v, 1., v], [1., 2., 3.], [v, v, v],
                          [2., 3., v], [v, v, v]])

    zero_value = v
    dim = -1
    fill = -torch.inf

    # when
    mask = create_zero_rows_mask(input, zero_value=zero_value, dim=dim)
    input[mask] = fill
    # fill_zero_rows(input, zero_value=zero_value, dim=dim, fill_value=fill)

    # then
    expected = torch.tensor([[fill, fill, fill], [v, 1., v], [1., 2., 3.],
                             [fill, fill, fill], [2., 3., v],
                             [fill, fill, fill]])

    assert torch.allclose(expected, input, equal_nan=True)


def test_collect_nonzero_indices_along_axis_should_work_for_nan():

    # given
    v = torch.nan

    input = torch.tensor([[v, v, v], [v, 1., v], [1., 2., 3.], [v, v, v],
                          [2., 3., v], [v, v, v]])

    # when
    first_indices, last_indices = collect_nonzero_indices_along_axis(
        input, zero_value=v, dim=-1)

    # then
    expected_first_indices = torch.tensor([1])
    expected_last_indices = torch.tensor([4])

    assert torch.allclose(first_indices, expected_first_indices)
    assert torch.allclose(last_indices, expected_last_indices)


def test_collect_nonzero_indices_along_axis():

    # given
    input = torch.tensor([
        [0., 0., 0.],
        [0., 1e-10, 0.],  # it is not zero
        [1., 2., 3.],
        [0., 0., 0.],
        [2., 3., 0.],
        [0., 0., 0.]
    ]  # should work with very close value too
                         )

    # when
    first_indices, last_indices = collect_nonzero_indices_along_axis(
        input, zero_value=0., dim=-1)

    # then
    expected_first_indices = torch.tensor([1])
    expected_last_indices = torch.tensor([4])

    assert torch.allclose(first_indices, expected_first_indices)
    assert torch.allclose(last_indices, expected_last_indices)

    # when gather
    gathered_first_items = gather_based_on_indices_along_axis(
        input, first_indices)
    gathered_last_items = gather_based_on_indices_along_axis(
        input, last_indices)

    # then
    assert gathered_first_items.shape == (1, 3)
    assert gathered_last_items.shape == (1, 3)
    assert torch.allclose(gathered_first_items, torch.tensor([[0., 1e-10,
                                                               0.]]))
    assert torch.allclose(gathered_last_items, torch.tensor([[2., 3., 0.]]))


def test_collect_nonzero_indices_and_gather_along_axis_when_multiple_batched():

    # given
    batch0_input0 = torch.tensor([[
        0.,
        0.,
    ], [
        1.,
        2.,
    ]])
    batch0_input1 = torch.tensor([[
        2.,
        3.,
    ], [
        0.,
        0.,
    ]])
    batch1_input0 = torch.tensor([[
        4.,
        5.,
    ], [
        6.,
        7.,
    ]])
    batch1_input1 = torch.tensor([[
        0.,
        0.,
    ], [
        8.,
        9.,
    ]])
    batch0 = torch.stack((batch0_input0, batch0_input1))
    batch1 = torch.stack((batch1_input0, batch1_input1))
    batch = torch.stack((batch0, batch1))

    # when
    first_indices, last_indices = collect_nonzero_indices_along_axis(
        batch, zero_value=0., dim=-1)

    # then
    assert first_indices.shape == (2, 2)
    assert last_indices.shape == (2, 2)

    assert torch.allclose(first_indices, torch.tensor([[1, 0], [0, 1]]))
    assert torch.allclose(last_indices, torch.tensor([[1, 0], [1, 1]]))

    # when gather
    gathered_first_items = gather_based_on_indices_along_axis(
        batch, first_indices)
    gathered_last_items = gather_based_on_indices_along_axis(
        batch, last_indices)

    # then
    expected_shape = (batch.shape[0], batch.shape[1], 1, batch.shape[3])
    expected_first_items = torch.tensor([[[[1., 2.]], [[2., 3.]]],
                                         [[[4., 5.]], [[8., 9.]]]])
    expected_last_items = torch.tensor([[[[1., 2.]], [[2., 3.]]],
                                        [[[6., 7.]], [[8., 9.]]]])

    assert gathered_first_items.shape == expected_shape
    assert gathered_last_items.shape == expected_shape
    assert torch.allclose(expected_first_items, gathered_first_items)
    assert torch.allclose(expected_last_items, gathered_last_items)


def test_collect_nonzero_indices_along_axis_when_batched():

    # given
    input0 = torch.tensor([[
        0.,
        0.,
    ], [
        0.,
        1.,
    ], [0., 0.], [0., 0.]])
    input1 = torch.tensor([[
        1.,
        1.,
    ], [
        0.,
        0.,
    ], [2., 3.], [4., 5.]])
    batch = torch.stack((input0, input1))

    # when
    first_indices, last_indices = collect_nonzero_indices_along_axis(
        batch, zero_value=0., dim=-1)

    # then
    expected_first_indices = torch.tensor([1, 0])
    expected_last_indices = torch.tensor([1, 3])

    assert torch.allclose(first_indices, expected_first_indices)
    assert torch.allclose(last_indices, expected_last_indices)

    # when gather
    gathered_first_items = gather_based_on_indices_along_axis(
        batch, first_indices)
    gathered_last_items = gather_based_on_indices_along_axis(
        batch, last_indices)

    # then
    expected_shape = (batch.shape[0], 1, batch.shape[-1])
    expected_first_items = torch.tensor([[[[0., 1.]], [[1., 1.]]]])
    expected_last_items = torch.tensor([[[[0., 1.]], [[4., 5.]]]])

    assert gathered_first_items.shape == expected_shape
    assert gathered_last_items.shape == expected_shape
    assert torch.allclose(expected_first_items, gathered_first_items)
    assert torch.allclose(expected_last_items, gathered_last_items)
