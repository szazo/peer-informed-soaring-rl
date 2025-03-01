import numpy as np
import pytest
from env.glider.base.wrappers.sequence_window import SequenceWindow


def test_add_given_empty_then_should_add():

    # given
    c = SequenceWindow(max_sequence_length=3)
    item = np.array([1, 2, 3])

    # when
    window = c.add(item)

    # then
    assert c.window is not None
    assert c.window.shape == (1, 3)
    assert np.allclose(np.array([item]), c.window)
    assert not c.window is window
    assert np.allclose(c.window, window)


def test_add_given_non_empty_then_should_add_to_end():

    # given
    c = SequenceWindow(max_sequence_length=3)
    item1 = np.array([1, 2, 3])
    c.add(item1)

    # when
    item2 = np.array([4, 5, 6])
    window = c.add(item2)

    # then
    assert c.window is not None
    assert c.window.shape == (2, 3)
    assert np.allclose(np.array([item1, item2]), c.window)
    assert not c.window is window
    assert np.allclose(c.window, window)


def test_add_given_max_sequence_length_arrived_then_should_roll_window():

    # given
    c = SequenceWindow(max_sequence_length=2)
    item1 = np.array([1, 2, 3])
    item2 = np.array([4, 5, 6])
    c.add(item1)
    c.add(item2)

    # when
    item3 = np.array([7, 8, 9])
    window = c.add(item3)

    # then
    assert c.window is not None
    assert c.window.shape == (2, 3)
    assert np.allclose(np.array([item2, item3]), c.window)
    assert not c.window is window
    assert np.allclose(c.window, window)


def test_add_given_scalar_should_add_as_row_to_the_end():

    # given
    c = SequenceWindow(max_sequence_length=3)
    item1 = np.array(1)
    c.add(item1)

    # when
    item2 = np.array(2)
    window = c.add(item2)

    # then
    assert c.window is not None
    assert c.window.shape == (2, 1)
    assert np.allclose(np.array([[item1], [item2]]), c.window)
    assert not c.window is window
    assert np.allclose(c.window, window)


def test_add_given_multidimensional_should_add_to_end():

    # given
    c = SequenceWindow(max_sequence_length=3)
    item1 = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    c.add(item1)

    # when
    item2 = item1 + item1.size
    window = c.add(item2)

    # then
    assert c.window is not None
    assert c.window.shape == (2, 2, 2, 3)
    assert np.allclose(np.array([item1, item2]), c.window)
    assert not c.window is window
    assert np.allclose(c.window, window)


def test_add_given_multidimensional_max_sequence_length_arrived_then_should_roll_window(
):

    # given
    c = SequenceWindow(max_sequence_length=2)
    item1 = np.array([[1, 2], [3, 4]])
    item2 = np.array([[4, 5], [6, 7]])
    c.add(item1)
    c.add(item2)

    # when
    item3 = np.array([[7, 8], [9, 10]])
    window = c.add(item3)

    # then
    assert c.window is not None
    assert c.window.shape == (2, 2, 2)
    assert np.allclose(np.array([item2, item3]), c.window)
    assert not c.window is window
    assert np.allclose(c.window, window)


def test_add_given_zero_sequence_length_then_should_raise():

    with pytest.raises(Exception):

        # given
        SequenceWindow(max_sequence_length=0)
