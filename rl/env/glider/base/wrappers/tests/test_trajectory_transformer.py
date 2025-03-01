import numpy as np
from env.glider.base.wrappers.trajectory_transformer import TrajectoryTransformer, TrajectoryTransformerParams


def test_trajectory_transformer_should_rotate_and_translate():

    # given
    y_axis = [0., 1., 0.]
    params = TrajectoryTransformerParams(rotate_around='first',
                                         rotate_to=y_axis,
                                         translate_relative_to='first')
    transformer = TrajectoryTransformer(params=params)

    position = np.array([[1., 0., 0.], [2., 0., 0.]])
    velocity = np.array(([1., 0., 0.], [1., 0., 0.]))

    # when
    position_rotated, velocity_rotated = transformer.transform(
        position, velocity)

    # then
    expected_position = np.array([[0., 0., 0], [0., 1., 0.]])
    expected_velocity = np.array([[0., 1., 0], [0., 1., 0.]])

    assert position_rotated.shape == (2, 3)
    assert velocity_rotated.shape == (2, 3)
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_transformer_should_not_rotate_when_none():

    # given
    y_axis = [0., 1., 0.]
    params = TrajectoryTransformerParams(rotate_around=None,
                                         rotate_to=y_axis,
                                         translate_relative_to='first')
    transformer = TrajectoryTransformer(params=params)

    position = np.array([[1., 0., 0.], [2., 0., 0.]])
    velocity = np.array(([1., 0., 0.], [1., 0., 0.]))

    # when
    position_rotated, velocity_rotated = transformer.transform(
        position, velocity)

    # then
    expected_position = np.array([[0., 0., 0], [1., 0., 0.]])
    expected_velocity = np.array([[1., 0., 0], [1., 0., 0.]])

    assert position_rotated.shape == (2, 3)
    assert velocity_rotated.shape == (2, 3)
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_transformer_should_not_project_when_project_to_none():

    # given
    y_axis = [0., 1., 0.]
    params = TrajectoryTransformerParams(rotate_around='first',
                                         rotate_to=y_axis,
                                         translate_relative_to='first',
                                         project_to=None)
    transformer = TrajectoryTransformer(params=params)

    position = np.array([[1., 0., 0.], [2., 0., 1.]])
    velocity = np.array(([1., 0., 1.], [1., 0., 1.]))

    # when
    position_rotated, velocity_rotated = transformer.transform(
        position, velocity)

    # then
    expected_position = np.array([[0., 0., 0], [0., np.sqrt(2), 0.]])
    expected_velocity = np.array([[0., np.sqrt(2), 0], [0., np.sqrt(2), 0.]])

    assert position_rotated.shape == (2, 3)
    assert velocity_rotated.shape == (2, 3)
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_transformer_should_project_when_project_to_xy():

    # given
    y_axis = [0., 1., 0.]
    params = TrajectoryTransformerParams(rotate_around='first',
                                         rotate_to=y_axis,
                                         translate_relative_to='first',
                                         project_to='xy_plane')
    transformer = TrajectoryTransformer(params=params)

    position = np.array([[1., 0., 0.], [2., 0., 1.]])
    velocity = np.array(([1., 0., 1.], [1., 0., 1.]))

    # when
    position_rotated, velocity_rotated = transformer.transform(
        position, velocity)

    # then
    expected_position = np.array([[0., 0., 0], [0., 1., 1.]])
    expected_velocity = np.array([[0., 1, 1], [0., 1, 1.]])

    assert position_rotated.shape == (2, 3)
    assert velocity_rotated.shape == (2, 3)
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)
