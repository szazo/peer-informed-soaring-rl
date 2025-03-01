import numpy as np
from ..multi_agent_observation_share_wrapper import append_item_index_to_the_target_axis
from trainer.multi_agent.tests.create_sample_items import create_and_stack_sample_items


def test_append_item_index_should_work_for_single_dimension():

    # given
    count = 3
    item_axis = 0
    target_axis = 1

    input = create_and_stack_sample_items(shape=(1, ),
                                          item_axis=item_axis,
                                          count=count)

    # when
    output = append_item_index_to_the_target_axis(input,
                                                  item_axis=item_axis,
                                                  target_axis=target_axis)

    # then
    indices = np.array([[0], [1], [2]])
    expected = np.append(input, indices, axis=target_axis)

    assert output.shape == expected.shape
    assert np.allclose(expected, output)


def test_append_item_index_should_add_when_middle_item_axis():

    # given
    count = 3
    item_axis = 1
    target_axis = 2

    input = create_and_stack_sample_items(shape=(2, 2),
                                          item_axis=item_axis,
                                          count=count)

    # when
    output = append_item_index_to_the_target_axis(input,
                                                  item_axis=item_axis,
                                                  target_axis=target_axis)

    # then
    indices = np.array([[[0], [1], [2]], [[0], [1], [2]]])
    expected = np.append(input, indices, axis=target_axis)

    assert output.shape == expected.shape
    assert np.allclose(expected, output)


def test_append_item_index_should_add_when_first_item_axis():

    # given
    count = 3
    item_axis = 0
    target_axis = 2

    input = create_and_stack_sample_items(shape=(2, 2),
                                          item_axis=item_axis,
                                          count=count)

    # when
    output = append_item_index_to_the_target_axis(input,
                                                  item_axis=item_axis,
                                                  target_axis=target_axis)

    # then
    indices = np.array([[[0], [0]], [[1], [1]], [[2], [2]]])
    expected = np.append(input, indices, axis=target_axis)

    assert output.shape == expected.shape
    assert np.allclose(expected, output)


def test_append_item_index_should_add_when_last_item_axis():

    # given
    count = 3
    item_axis = 2
    target_axis = 1

    input = create_and_stack_sample_items(shape=(2, 2),
                                          item_axis=item_axis,
                                          count=count)

    # when
    output = append_item_index_to_the_target_axis(input,
                                                  item_axis=item_axis,
                                                  target_axis=target_axis)

    # then
    indices = np.array([[[0, 1, 2]], [[0, 1, 2]]])
    expected = np.append(input, indices, axis=target_axis)

    assert output.shape == expected.shape
    assert np.allclose(expected, output)
