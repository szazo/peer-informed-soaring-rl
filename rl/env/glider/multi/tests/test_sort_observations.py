import numpy as np
from ..multi_agent_observation_share_wrapper import calculate_smallest_distance_for_items_along_axis, calculate_smallest_distance_between_trajectories, clear_empty_items_along_axis, sort_items_along_axis
from trainer.multi_agent.tests.mock_trajectory import calculate_offset
from utils import VectorNxN
from utils.vector import VectorN

from trainer.multi_agent.tests.create_sample_items import create_and_stack_sample_items, create_sample_items


def test_create_metric():

    return

    # given
    positions1 = np.array([[1, 1], [1, 1], [2, 2], [1, 1]])

    positions2 = np.array([[4, 4], [4, 4], [3, 3], [-1, 1]])

    # when
    min = calculate_smallest_distance(positions1, positions2)

    # then
    assert np.isclose(min, np.sqrt(2))


def test_calculate_distance_along_axis():

    # given
    items = create_sample_items(shape=(2, 3), count=3)

    # fill different rows with nans for the first two items
    item0 = items[0]
    item1 = items[1]
    item0[1] = np.nan
    item1[0] = np.nan

    item_axis = 1
    input = np.stack(items, axis=item_axis)

    print('input', input, input.shape)
    print('item0', input[:, 0, ...])
    print('item1', input[:, 1, ...])

    # when
    output = calculate_smallest_distance_for_items_along_axis(
        input, item_axis=item_axis, self_item_index=1)
    print('output', output)

    # then
    # expected = [
    #     calculate_smallest_distance(items[2], items[0]),
    #     calculate_smallest_distance(items[2], items[1]),
    #     calculate_smallest_distance(items[2], items[2])
    # ]

    # remove items which has no common points with the self_index
    mask = np.logical_not(np.isclose(output, np.inf))
    print('todelete', mask)
    # clear_empty_items_along_axis

    input = np.compress(mask, input, axis=item_axis)
    print('mask', mask, type(mask), output)

    output: VectorN = output[mask]

    print('items after delete', input, input.shape)
    print('metric after delete', output)

    #assert expected == output

    sorted = sort_items_along_axis(input, item_axis=1, metric=output)

    print('sorted', sorted)

    # print(expected)

    # print(output)

    # # output = move_item_along_axis(input,
    # #                               item_axis=axis,
    # #                               source_index=1,
    # #                               target_index=0)

    # pass
