import numpy as np
from ..multi_agent_observation_share_wrapper import (
    calculate_smallest_distance_for_items_along_axis,
    calculate_smallest_distance_between_trajectories,
    clear_empty_items_along_axis, sort_items_along_axis)
from utils.vector import VectorN

from trainer.multi_agent.tests.create_sample_items import create_sample_items


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

    # when
    output = calculate_smallest_distance_for_items_along_axis(
        input, item_axis=item_axis, self_item_index=1)

    # remove items which has no common points with the self_index
    mask = np.logical_not(np.isclose(output, np.inf))
    print('todelete', mask)
    # clear_empty_items_along_axis

    input = np.compress(mask, input, axis=item_axis)

    output: VectorN = output[mask]

    sorted = sort_items_along_axis(input, item_axis=1, metric=output)
