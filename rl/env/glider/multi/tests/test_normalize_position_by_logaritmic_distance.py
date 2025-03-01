import numpy as np

from utils.vector import Vector3

from ..multi_agent_observation_share_wrapper import (
    normalize_position_by_logaritmic_distance)


def test_normalize_position_by_logaritmic_distance_should_handle_zero():

    # given
    input = np.array([[[1., 0., 0., 0., 2.], [2., 3., 4., 5., 6.]],
                      [[3., 1., 0., 0., 0.], [4., 1., 1., 1., 0.]]])
    input_copy = input.copy()

    # when
    result = normalize_position_by_logaritmic_distance(
        input, position_3d_start_column_index=1)

    # then
    expected = input.copy()
    expected[0, 1, 1:4] = _normalize_vector(np.array([3., 4., 5.]))
    expected[1, 0, 1:4] = _normalize_vector(np.array([1., 0., 0.]))
    expected[1, 1, 1:4] = _normalize_vector(np.array([1., 1., 1.]))

    assert np.allclose(input, input_copy)  # do not change the input
    assert np.allclose(expected, result)
    assert input.shape == result.shape


def _normalize_vector(input: Vector3):
    norm = np.linalg.norm(input)
    normalizer = norm / np.log(1 + norm)
    normalized = input / normalizer

    return normalized
