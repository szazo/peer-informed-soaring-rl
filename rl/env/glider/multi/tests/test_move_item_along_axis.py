import numpy as np
from ..multi_agent_observation_share_wrapper import move_item_along_axis
from trainer.multi_agent.tests.create_sample_items import create_sample_items


def test_move_item_along_axis_should_work_with_single_dimension():

    # given
    items = create_sample_items(shape=(2, ), count=3)
    axis = 0

    # stack them along the first axis
    input = np.stack(items, axis=axis)
    output = move_item_along_axis(input,
                                  item_axis=axis,
                                  source_index=1,
                                  target_index=0)

    # then
    expected = np.stack([items[1], items[0], items[2]], axis=axis)
    assert output.shape == expected.shape
    assert np.allclose(expected, output)


def test_move_item_along_axis_should_work_with_middle_axis():

    # given
    items = create_sample_items(shape=(2, 2), count=3)
    axis = 1

    # stack them along the first axis
    input = np.stack(items, axis=axis)
    output = move_item_along_axis(input,
                                  item_axis=axis,
                                  source_index=2,
                                  target_index=0)

    # then
    expected = np.stack([items[2], items[0], items[1]], axis=axis)
    assert output.shape == expected.shape
    assert np.allclose(expected, output)


def test_move_item_along_axis_should_work_with_first_axis():

    # given
    items = create_sample_items(shape=(2, 2), count=3)
    axis = 1

    # stack them along the first axis
    input = np.stack(items, axis=axis)
    output = move_item_along_axis(input,
                                  item_axis=axis,
                                  source_index=2,
                                  target_index=0)

    # then
    expected = np.stack([items[2], items[0], items[1]], axis=axis)

    assert output.shape == expected.shape
    assert np.allclose(expected, output)


def test_move_item_along_axis_should_work_with_last_axis():

    # given
    items = create_sample_items(shape=(2, 2), count=3)
    axis = 2

    # stack them along the first axis
    input = np.stack(items, axis=axis)
    output = move_item_along_axis(input,
                                  item_axis=axis,
                                  source_index=2,
                                  target_index=0)

    # then
    expected = np.stack([items[2], items[0], items[1]], axis=axis)

    assert output.shape == expected.shape
    assert np.allclose(expected, output)
