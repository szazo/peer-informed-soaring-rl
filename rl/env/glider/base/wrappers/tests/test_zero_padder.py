from env.glider.base.wrappers.zero_pad_sequence import zero_pad_sequence
import numpy as np


def test_pad_should_pad_at_the_beginning_when_start():

    # given
    input = np.array([[1., 2., 3.], [4., 5., 6.]])
    max_sequence_length = 4

    # when
    padded = zero_pad_sequence(input,
                               max_sequence_length=max_sequence_length,
                               pad_at='start')

    # then
    expected = np.array([[0., 0., 0.], [0., 0., 0.], [1., 2., 3.],
                         [4., 5., 6.]])
    assert np.allclose(expected, padded)


def test_pad_should_pad_at_the_end_when_end():

    # given
    input = np.array([[1., 2., 3.], [4., 5., 6.]])
    max_sequence_length = 4

    # when
    padded = zero_pad_sequence(input,
                               max_sequence_length=max_sequence_length,
                               pad_at='end')

    # then
    expected = np.array([[1., 2., 3.], [4., 5., 6.], [0., 0., 0.],
                         [0., 0., 0.]])
    assert np.allclose(expected, padded)


def test_pad_should_not_pad_when_same_as_sequence_length():

    # given
    input = np.array([[1., 2., 3.], [4., 5., 6.]])
    max_sequence_length = 2

    # when
    padded = zero_pad_sequence(input,
                               max_sequence_length=max_sequence_length,
                               pad_at='end')

    # then
    expected = input
    assert np.allclose(expected, padded)


def test_pad_should_not_pad_when_greater_than_sequence_length():

    # given
    input = np.array([[1., 2., 3.], [4., 5., 6.]])
    max_sequence_length = 1

    # when
    padded = zero_pad_sequence(input,
                               max_sequence_length=max_sequence_length,
                               pad_at='end')

    # then
    expected = input
    assert np.allclose(expected, padded)


def test_pad_should_use_constant_value_when_set():

    # given
    input = np.array([[1., 2., 3.], [4., 5., 6.]])
    max_sequence_length = 4

    # when
    padded = zero_pad_sequence(input,
                               max_sequence_length=max_sequence_length,
                               pad_at='start',
                               constant_value=np.nan)

    # then
    expected = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan],
                         [1., 2., 3.], [4., 5., 6.]])
    assert np.allclose(expected, padded, equal_nan=True)


def test_pad_should_pad_when_multi_dimensional():

    # given
    input = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]]])
    max_sequence_length = 4

    # when
    padded = zero_pad_sequence(input,
                               max_sequence_length=max_sequence_length,
                               pad_at='end')

    # then
    assert padded.shape == (4, 2, 2)
    expected = np.array([[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]],
                         [[0., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
    assert np.allclose(expected, padded)
