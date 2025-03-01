from env.glider.base.wrappers.trajectory_rotator import TrajectoryRotator, construct_quaternion_from_vectors, rotate_by_quaternion, translate_trajectory_to_origo
import numpy as np
import quaternion


def test_construct_quaternion_should_construct_when_orthogonal_vectors():

    # given
    source = np.array([1, 0, 0])
    target = np.array([0, 1, 0])

    # when
    q = construct_quaternion_from_vectors(source, target)

    # then
    assert np.allclose(quaternion.as_rotation_matrix(q),
                       np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
    assert np.isclose(np.linalg.norm(quaternion.as_float_array(q)), 1)


def test_construct_quaternion_should_return_unit_quaternion_when_parallel_vectors(
):

    # given
    source = np.array([1, 0, 0])
    target = np.array([1, 0, 0])

    # when
    q = construct_quaternion_from_vectors(source, target)

    # then
    assert np.allclose(quaternion.as_float_array(q), [1., 0., 0., 0.])


def test_rotate_should_rotate_when_single_item():

    # given
    input = np.array([[0, 2, 0]])
    q = quaternion.from_rotation_vector([[0., 0., np.deg2rad(90)]])

    # when
    rotated = rotate_by_quaternion(input, q)

    # then
    assert rotated.shape == (1, 3)
    assert np.linalg.norm(rotated, axis=1) == np.linalg.norm(input, axis=1)
    assert np.allclose(rotated, np.array([[-2, 0, 0]]))


def test_rotate_should_rotate_all_when_multiple_items():

    # given
    input = np.array([[0, 2, 0], [3, 0, 0]])
    q = quaternion.from_rotation_vector([[0., 0., np.deg2rad(-90)]])

    # when
    rotated = rotate_by_quaternion(input, q)

    # then
    assert rotated.shape == (2, 3)
    assert np.allclose(np.linalg.norm(rotated, axis=1),
                       np.linalg.norm(input, axis=1))

    assert np.allclose(rotated, np.array([[2, 0, 0], [0, -3, 0]]))


def test_translate_trajectory_to_origo_should_raise_error_when_empty():

    # when
    trajectory_translated, translate_vector = translate_trajectory_to_origo(
        trajectory=np.array([]), relative_to='first')

    # then
    assert np.allclose(np.array([]), trajectory_translated)
    assert np.allclose(np.array([0., 0., 0.]), translate_vector)


def test_translate_trajectory_to_origo_should_translate_first():

    # given
    trajectory = np.array([[1., 2., 3.], [2., 4., 6.], [4., 5., 6.]])

    # when
    translated, translate_vector = translate_trajectory_to_origo(
        trajectory, relative_to='first')

    # then
    expected_trajectory = np.array([[0., 0., 0.], [1., 2., 3.], [3., 3., 3.]])
    expected_translate_vector = np.array([-1., -2., -3.])

    assert np.allclose(expected_trajectory, translated)
    assert np.allclose(expected_translate_vector, translate_vector)


def test_translate_trajectory_to_origo_should_translate_last():

    # given
    trajectory = np.array([[1., 2., 3.], [2., 4., 6.], [4., 5., 6.]])

    # when
    translated, translate_vector = translate_trajectory_to_origo(
        trajectory, relative_to='last')

    # then
    expected_trajectory = np.array([[-3., -3., -3.], [-2., -1., 0.],
                                    [0., 0., 0.]])
    expected_translate_vector = np.array([-4., -5., -6.])

    assert np.allclose(expected_trajectory, translated)
    assert np.allclose(expected_translate_vector, translate_vector)


def test_trajectory_rotator_should_do_nothing_when_empty_and_using_position():

    # given
    y_axis = np.array([0., 1., 0.])
    rotator = TrajectoryRotator(rotate_around='first',
                                anchor_vector='position_diff',
                                target_axis=y_axis)

    position = np.empty((0, 3))
    velocity = np.empty((0, 3))

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_position = position
    expected_velocity = velocity
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_rotator_should_do_nothing_when_single_and_using_position():

    # given
    y_axis = np.array([0., 1., 0.])
    rotator = TrajectoryRotator(rotate_around='first',
                                anchor_vector='position_diff',
                                target_axis=y_axis)

    position = np.arange(3.).reshape(1, 3)
    velocity = np.arange(start=3., stop=6.).reshape(1, 3)

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_position = position
    expected_velocity = velocity
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_rotator_should_rotate_around_first_when_2D_using_position(
):

    # given
    y_axis = np.array([0., 1., 0.])
    rotator = TrajectoryRotator(rotate_around='first',
                                anchor_vector='position_diff',
                                target_axis=y_axis)

    position = np.array([[1., 1., 0.], [2., 1., 0.], [2., 2., 0]])
    velocity = np.zeros((position.shape[0], 3))

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_position = np.array([[1., 1., 0], [1., 2., 0], [0., 2., 0.]])
    expected_velocity = velocity
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_rotator_should_rotate_around_last_when_2D_using_position():

    # given
    y_axis = np.array([0., 1., 0.])
    rotator = TrajectoryRotator(rotate_around='last',
                                anchor_vector='position_diff',
                                target_axis=y_axis)

    position = np.array([[1., 1., 0.], [1., 2., 0.], [2., 2., 0]])
    velocity = np.zeros((position.shape[0], 3))

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_position = np.array([[3., 1., 0], [2., 1., 0], [2., 2., 0.]])
    expected_velocity = velocity
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_rotator_should_rotate_around_first_when_3D_using_position(
):

    # given
    y_axis = np.array([0., 1., 0.])
    rotator = TrajectoryRotator(rotate_around='first',
                                anchor_vector='position_diff',
                                target_axis=y_axis)

    position = np.array([[1., 1., 0.], [2., 1., 1.]])
    velocity = np.zeros((position.shape[0], 3))

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_position = np.array([[1., 1., 0], [1., 1. + np.sqrt(2), 0]])
    expected_velocity = velocity
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_rotator_should_use_anchor_transform_when_set_and_using_position(
):

    # given
    y_axis = np.array([0., 1., 0.])
    xy_plane_project = lambda v: np.array([v[0], v[1], 0.])
    rotator = TrajectoryRotator(rotate_around='first',
                                target_axis=y_axis,
                                anchor_vector='position_diff',
                                anchor_section_transform=xy_plane_project)

    position = np.array([[1., 1., 0.], [2., 1., 1.]])
    velocity = np.zeros((position.shape[0], 3))

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_position = np.array([[1., 1., 0], [1., 2., 1.]])
    expected_velocity = velocity
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(expected_velocity, velocity_rotated)


def test_trajectory_rotator_should_rotate_velocity_vector_when_using_position(
):

    # given
    y_axis = np.array([0., 1., 0.])

    rotator = TrajectoryRotator(rotate_around='first',
                                anchor_vector='position_diff',
                                target_axis=y_axis)
    position = np.array([[1., 1., 0.], [2., 1., 1.]])
    velocity = np.array([[0.5, 0.6, 0.7], [-0.7, 0.6, -0.5]])
    dot_before = calculate_dot_products(position, velocity)

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_position = np.array([[1., 1., 0], [1., 1. + np.sqrt(2), 0]])
    dot_after = calculate_dot_products(position_rotated, velocity_rotated)

    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(dot_before, dot_after)


def test_trajectory_rotator_should_rotate_position_when_using_first_velocity_as_anchor(
):

    # given
    y_axis = np.array([0., 1., 0.])

    rotator = TrajectoryRotator(rotate_around='first',
                                anchor_vector='velocity',
                                target_axis=y_axis)
    velocity = np.array([[1., 0., 1.], [0., 1., 0.]])
    position = np.array([[1., -1., 0.], [2., -1., 1.]])
    dot_before = calculate_dot_products(position, velocity)

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_velocity = np.array([[0., np.sqrt(2), 0],
                                  [-np.sqrt(2) / 2., 0., -np.sqrt(2) / 2.]])
    expected_position = np.array([[1., -1., 0], [1, np.sqrt(2) - 1, 0]])

    dot_after = calculate_dot_products(position_rotated, velocity_rotated)

    assert np.allclose(expected_velocity, velocity_rotated)
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(dot_before, dot_after)


def test_trajectory_rotator_should_rotate_position_when_using_last_velocity_as_anchor(
):

    # given
    y_axis = np.array([0., 1., 0.])

    rotator = TrajectoryRotator(rotate_around='last',
                                anchor_vector='velocity',
                                target_axis=y_axis)
    velocity = np.array([[0., 1., 0.], [1., 0., 1.]])
    position = np.array([[2., -1., 1.], [1., -1., 0.]])
    dot_before = calculate_dot_products(position, velocity)

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_velocity = np.array([[-np.sqrt(2) / 2., 0., -np.sqrt(2) / 2.],
                                  [0., np.sqrt(2), 0]])
    expected_position = np.array([[1, np.sqrt(2) - 1, 0], [1., -1., 0]])

    dot_after = calculate_dot_products(position_rotated, velocity_rotated)

    assert np.allclose(expected_velocity, velocity_rotated)
    assert np.allclose(expected_position, position_rotated)
    assert np.allclose(dot_before, dot_after)


def test_trajectory_rotator_should_rotate_when_using_velocity_as_anchor_and_has_single_point(
):

    # given
    y_axis = np.array([0., 1., 0.])

    rotator = TrajectoryRotator(rotate_around='first',
                                anchor_vector='velocity',
                                target_axis=y_axis)
    velocity = np.array([[1., 0., 1.]])
    position = np.array([[1., -1., 0.]])

    # when
    position_rotated, velocity_rotated = rotator.rotate(position, velocity)

    # then
    expected_velocity = np.array([[0., np.sqrt(2), 0]])
    expected_position = np.array([[1., -1., 0]])

    assert np.allclose(expected_velocity, velocity_rotated)
    assert np.allclose(expected_position, position_rotated)


def calculate_dot_products(position, velocity):

    position_vectors = np.diff(position, axis=0)
    # add same orientation as the last point
    position_vectors = np.vstack((position_vectors, position_vectors[-1, :]))

    dot = np.sum(position_vectors * velocity, axis=1)

    return dot
