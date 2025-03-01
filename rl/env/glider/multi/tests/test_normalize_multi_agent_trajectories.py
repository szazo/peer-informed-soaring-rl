import numpy as np
from env.glider.base.wrappers import TrajectoryTransformerParams
from ..multi_agent_observation_share_wrapper import (
    NormalizeMultiAgentTrajectoriesObservationParams, Trajectory,
    normalize_multi_agent_trajectories,
    normalize_multi_agent_trajectories_observation)


def test_normalize_multi_agent_trajectories_should_use_first_agent():

    # given
    y_axis = [0., 1., 0.]
    params = TrajectoryTransformerParams(rotate_around='first',
                                         rotate_to=y_axis,
                                         translate_relative_to='first',
                                         project_to='xy_plane')

    agent0_position = np.array([[1., 0., 0.], [2., 0., 1.]])
    agent0_velocity = np.array(([1., 0., 1.], [1., 0., 1.]))

    agent1_position = np.array([[2., 0., 0.], [3., 0., 1.]])
    agent1_velocity = np.array(([0., -1., 1.], [-1., 0., 1.]))

    trajectories = [
        Trajectory(agent0_position, agent0_velocity),
        Trajectory(agent1_position, agent1_velocity)
    ]

    # when
    result = normalize_multi_agent_trajectories(trajectories, params=params)

    # then
    expected_agent0_position = np.array([[0., 0., 0], [0., 1., 1.]])
    expected_agent0_velocity = np.array([[0., 1, 1], [0., 1, 1.]])

    expected_agent1_position = np.array([[0., 1., 0], [0., 2., 1.]])
    expected_agent1_velocity = np.array([[1., 0, 1], [0., -1, 1.]])

    expected_trajectories = [
        Trajectory(expected_agent0_position, expected_agent0_velocity),
        Trajectory(expected_agent1_position, expected_agent1_velocity)
    ]

    for i, expected_trajectory in enumerate(expected_trajectories):

        result_trajectory = result[i]
        assert result_trajectory.position.shape == (2, 3)
        assert result_trajectory.velocity.shape == (2, 3)
        assert np.allclose(expected_trajectory.position,
                           result_trajectory.position)
        assert np.allclose(expected_trajectory.velocity,
                           result_trajectory.velocity)


def test_normalize_multi_agent_trajectories_observation_should_integrated():

    # given
    agent0_input = np.array([[1., 0., 0., 1., 0., 1.],
                             [2., 0., 1., 1., 0., 1.]])
    agent1_input = np.array([[2., 0., 0., 0., -1., 1.],
                             [3., 0., 1., -1., 0., 1.]])

    # place them shifted, and add an empty one too
    seq_length = 4
    dim = 6
    observation = np.full((3, seq_length, dim), np.nan)

    observation[0, 0:2, ...] = agent0_input
    observation[1, 1:3, ...] = agent1_input

    y_axis = [0., 1., 0.]
    transformer_params = TrajectoryTransformerParams(
        rotate_around='first',
        rotate_to=y_axis,
        translate_relative_to='first',
        project_to='xy_plane')

    params = NormalizeMultiAgentTrajectoriesObservationParams(
        trajectory_transform_params=transformer_params,
        position_3d_start_column_index=0,
        velocity_3d_start_column_index=3,
        pad_value=np.nan)

    # when
    output = normalize_multi_agent_trajectories_observation(observation,
                                                            params=params)

    # then
    expected_agent0_output = np.array([[0., 0., 0., 0., 1, 1.],
                                       [0., 1., 1., 0., 1, 1.]])
    expected_agent1_output = np.array([[0., 1., 0., 1., 0, 1.],
                                       [0., 2., 1., 0., -1, 1.]])

    expected = np.full((3, seq_length, dim), np.nan)

    expected[0, 0:2, ...] = expected_agent0_output
    expected[1, 1:3, ...] = expected_agent1_output

    assert np.allclose(output, expected, equal_nan=True)
    assert expected.shape == output.shape
