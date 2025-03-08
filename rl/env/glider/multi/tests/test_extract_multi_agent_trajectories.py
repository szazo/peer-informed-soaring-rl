from env.glider.multi.multi_agent_observation_share_wrapper import (
    Trajectory, extract_multi_agent_trajectories, write_trajectories_back)
import numpy as np

from trainer.multi_agent.tests.create_sample_items import create_sample_item


def test_extract_and_rewrite_multi_agent_trajectories():
    """Extract non padded position and velocity and return the good mask for writing back"""

    # given
    dim = 6
    item0 = create_sample_item((2, dim))
    item1 = create_sample_item((3, dim), offset=item0.size)
    item2 = create_sample_item((4, dim), offset=item0.size + item1.size)
    item3 = create_sample_item((2, dim), offset=item0.size + item1.size)

    input = np.full((4, 4, dim), np.nan)
    input[0, 0:2, ...] = item0
    input[1, 1:4, ...] = item1
    input[2, 0:4, ...] = item2
    input[3, 1:3, ...] = item3

    # when
    agent_trajectories, agent_masks = extract_multi_agent_trajectories(
        input,
        position_3d_start_column_index=0,
        velocity_3d_start_column_index=3,
        pad_value=np.nan)

    # then
    expected0 = Trajectory(position=input[0, 0:2, 0:3],
                           velocity=input[0, 0:2, 3:])
    expected1 = Trajectory(position=input[1, 1:4, 0:3],
                           velocity=input[1, 1:4, 3:])
    expected2 = Trajectory(position=input[2, 0:4, 0:3],
                           velocity=input[2, 0:4, 3:])
    expected3 = Trajectory(position=input[3, 1:3, 0:3],
                           velocity=input[3, 1:3, 3:])

    expected_trajectories = [expected0, expected1, expected2, expected3]
    for i, expected in enumerate(expected_trajectories):
        agent_trajectory = agent_trajectories[i]
        assert agent_trajectory.position.shape == expected.position.shape
        assert agent_trajectory.velocity.shape == expected.velocity.shape

        assert np.allclose(agent_trajectory.position, expected.position)
        assert np.allclose(agent_trajectory.velocity, expected.velocity)

    # write back based on the mask
    changed_trajectories = [
        Trajectory(trajectory.position + 1, trajectory.velocity + 1)
        for trajectory in agent_trajectories
    ]

    changed = write_trajectories_back(input,
                                      agent_trajectories=changed_trajectories,
                                      agent_masks=agent_masks,
                                      position_3d_start_column_index=0,
                                      velocity_3d_start_column_index=3,
                                      copy=True)

    assert np.allclose(changed, input + 1, equal_nan=True)
    assert changed.shape == input.shape
