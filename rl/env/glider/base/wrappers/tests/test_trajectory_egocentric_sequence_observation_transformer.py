import numpy as np
from env.glider.base.wrappers.trajectory_egocentric_sequence_observation_transformer import (
    TrajectoryEgocentricSequenceObservationTransformer,
    TrajectoryTransformerParams,
    TrajectoryEgocentricSequenceObservationTransformerParams)


def test_should_transform_in_two_step_when_no_reverse_no_zero_pad():

    # given
    y_axis = [0., 1., 0.]
    transform_params = TrajectoryTransformerParams(
        rotate_around='first', rotate_to=y_axis, translate_relative_to='last')
    params = TrajectoryEgocentricSequenceObservationTransformerParams(
        trajectory_transform_params=transform_params,
        max_sequence_length=4,
        position_3d_start_column_index=2,
        velocity_3d_start_column_index=6,
        reverse=False,
        zero_pad_at=None)
    transformer = TrajectoryEgocentricSequenceObservationTransformer(
        params=params)

    position = np.array([[1., 0., 0.], [2., 0., 0.]])
    length = position.shape[0]
    velocity = np.array([[1., 0., 0.], [1., 0., 0.]])

    additional1 = np.arange(start=1., stop=length * 2 + 1).reshape(length, 2)
    additional2 = np.arange(start=length * 2 + 1.,
                            stop=length * 3 + 1.).reshape(length, 1)
    additional3 = np.arange(start=length * 3 + 1.,
                            stop=length * 6 + 1.).reshape(length, 3)

    input = np.hstack(
        (additional1, position, additional2, velocity, additional3))

    # when
    step1_output = transformer.modify_obs(input[0, :])
    step2_output = transformer.modify_obs(input[1, :])

    # then

    # step1
    step1_expected_position = np.array([[0., 0., 0.]])
    step1_expected_velocity = np.array([[0., 1., 0.]])
    step1_expected = np.hstack(
        (additional1[0, :], step1_expected_position[0, :], additional2[0, :],
         step1_expected_velocity[0, :], additional3[0, :]))

    # step2
    step2_expected_position = np.array([[0., -1., 0.], [0., 0., 0.]])
    step2_expected_velocity = np.array([[0., 1., 0.], [0., 1., 0.]])
    step2_expected = np.hstack(
        (additional1, step2_expected_position, additional2,
         step2_expected_velocity, additional3))

    assert step1_output.shape == (1, 12)
    assert np.allclose(step1_output, step1_expected)

    assert step2_output.shape == (2, 12)
    assert np.allclose(step2_output, step2_expected)


def test_should_transform_in_two_step_when_no_reverse_zero_pad_and_reverse():

    # given
    y_axis = [0., 1., 0.]
    transform_params = TrajectoryTransformerParams(
        rotate_around='first', rotate_to=y_axis, translate_relative_to='first')
    params = TrajectoryEgocentricSequenceObservationTransformerParams(
        trajectory_transform_params=transform_params,
        max_sequence_length=2,
        position_3d_start_column_index=2,
        velocity_3d_start_column_index=6,
        reverse=True,
        zero_pad_at='start')
    transformer = TrajectoryEgocentricSequenceObservationTransformer(
        params=params)

    position = np.array([[1., 0., 0.], [2., 0., 0.]])
    length = position.shape[0]
    velocity = np.array([[1., 0., 0.], [1., 0., 0.]])

    additional1 = np.arange(start=1., stop=length * 2 + 1).reshape(length, 2)
    additional2 = np.arange(start=length * 2 + 1.,
                            stop=length * 3 + 1.).reshape(length, 1)
    additional3 = np.arange(start=length * 3 + 1.,
                            stop=length * 6 + 1.).reshape(length, 3)

    input = np.hstack(
        (additional1, position, additional2, velocity, additional3))

    # when
    step1_output = transformer.modify_obs(input[0, :])
    step2_output = transformer.modify_obs(input[1, :])

    # then

    # step1
    step1_expected_position = np.array([[0., 0., 0.]])
    step1_expected_velocity = np.array([[0., 1., 0.]])
    step1_expected = np.hstack(
        (additional1[0, :], step1_expected_position[0, :], additional2[0, :],
         step1_expected_velocity[0, :], additional3[0, :]))
    step1_expected = np.expand_dims(step1_expected, axis=0)
    zero_row = np.zeros((1, step1_expected.shape[1]))
    step1_expected = np.vstack((zero_row, step1_expected))

    assert step1_output.shape == (2, 12)
    assert np.allclose(step1_output, step1_expected)

    # step2
    step2_expected_position = np.array([[0., 0., 0.], [0., 1., 0.]])
    step2_expected_velocity = np.array([[0., 1., 0.], [0., 1., 0.]])
    step2_expected = np.hstack(
        (additional1, step2_expected_position, additional2,
         step2_expected_velocity, additional3))
    step2_expected = np.flip(step2_expected, axis=0)

    assert step2_output.shape == (2, 12)
    assert np.allclose(step2_output, step2_expected)


def test_should_transform_use_windowing():

    # given
    y_axis = [0., 1., 0.]
    transform_params = TrajectoryTransformerParams(
        rotate_around='first', rotate_to=y_axis, translate_relative_to='first')
    params = TrajectoryEgocentricSequenceObservationTransformerParams(
        trajectory_transform_params=transform_params,
        max_sequence_length=1,
        position_3d_start_column_index=0,
        velocity_3d_start_column_index=3,
        reverse=False,
        zero_pad_at=None)
    transformer = TrajectoryEgocentricSequenceObservationTransformer(
        params=params)

    position = np.array([[1., 0., 0.], [2., 0., 0.]])
    velocity = np.array([[1., 0., 0.], [1., 0., 0.]])

    input = np.hstack((position, velocity))

    # when
    step1_output = transformer.modify_obs(input[0, :])
    step2_output = transformer.modify_obs(input[1, :])

    # then

    # step1
    step1_expected_position = np.array([[0., 0., 0.]])
    step1_expected_velocity = np.array([[0., 1., 0.]])
    step1_expected = np.hstack(
        (step1_expected_position[0, :], step1_expected_velocity[0, :]))
    step1_expected = np.expand_dims(step1_expected, axis=0)

    assert step1_output.shape == (1, 6)
    assert np.allclose(step1_output, step1_expected)

    # # step2
    step2_expected_position = np.array([[0., 0., 0.]])
    step2_expected_velocity = np.array([[0., 1., 0.]])
    step2_expected = np.hstack(
        (step2_expected_position, step2_expected_velocity))

    assert step2_output.shape == (1, 6)
    assert np.allclose(step2_output, step2_expected)


def test_reset_should_reset_history():

    # given
    y_axis = [0., 1., 0.]
    transform_params = TrajectoryTransformerParams(
        rotate_around='first', rotate_to=y_axis, translate_relative_to='first')
    params = TrajectoryEgocentricSequenceObservationTransformerParams(
        trajectory_transform_params=transform_params,
        max_sequence_length=10,
        position_3d_start_column_index=0,
        velocity_3d_start_column_index=3,
        reverse=False,
        zero_pad_at=None)
    transformer = TrajectoryEgocentricSequenceObservationTransformer(
        params=params)

    position = np.array([[1., 0., 0.], [2., 0., 0.]])
    velocity = np.array([[1., 0., 0.], [1., 0., 0.]])

    input = np.hstack((position, velocity))

    # when
    step1_output = transformer.modify_obs(input[0, :])
    transformer.reset()
    step2_output = transformer.modify_obs(input[1, :])

    # then
    assert step1_output.shape == (1, 6)
    assert step2_output.shape == (1, 6)
