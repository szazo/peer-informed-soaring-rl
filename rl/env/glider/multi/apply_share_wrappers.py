import numpy as np
import pettingzoo
from .multi_agent_observation_share_wrapper import MoveAxisParams, MultiAgentObservationShareWrapper, NormalizeMultiAgentTrajectoriesObservationParams, RemoveNonExistentAgentObservationWrapper, SortAgentObservationWrapper, SortAgentObservationParams, ReverseSequenceObservationWrapper, move_axis_obs_wrapper, nan_to_zero_obs_wrapper, normalize_multi_agent_trajectories_obs_wrapper, normalize_position_by_logaritmic_distance_obs_wrapper
from env.glider.base.wrappers import (SequenceWindowObsParams,
                                      TrajectoryTransformerParams,
                                      pad_sequence_obs_wrapper,
                                      sequence_window_obs_wrapper,
                                      PadSequenceObsParams)


def apply_share_wrappers(
        env: pettingzoo.ParallelEnv,
        max_sequence_length: int,
        max_closest_agent_count: int,
        normalize_trajectories: bool = True) -> pettingzoo.ParallelEnv:

    env0 = MultiAgentObservationShareWrapper(env)
    env0 = sequence_window_obs_wrapper(
        env0,
        params=SequenceWindowObsParams(
            max_sequence_length=max_sequence_length))

    # do the padding
    # env0 = pad_sequence_obs_wrapper(env0,
    #                                 params=PadSequenceObsParams(
    #                                     max_sequence_length=4,
    #                                     pad_at='end',
    #                                     value=np.nan))
    assert isinstance(env0, pettingzoo.ParallelEnv)

    # removes non existent agents from the observation (after the observation sharing)
    env0 = RemoveNonExistentAgentObservationWrapper(env0)

    # # append agent index to the last dimension
    # env0 = append_multi_agent_item_index_obs_wrapper(env0,
    #                                                  item_axis=1, target_axis=-1)

    # set that the self agent should be at the first index along the agent axis

    # clear empty observations from each agents' observation
    input_mask = input_mask = (slice(None), slice(None), slice(
        None, -1))  # [:, :,-1] we dont use the index for the mask
    # clear_empty_items_params = ClearEmptyItemsObsParams(axis=1,
    #                                                     empty_value=np.nan,
    #                                                     input_mask=input_mask)
    # env0 = clear_empty_items_obs_wrapper(env0, clear_empty_items_params)
    assert isinstance(env0, pettingzoo.ParallelEnv)

    # TODO: order the remaining observations
    # the first item is always the self, others are ordered relative to that
    # e.g. minimum distance, mean distance, ...
    env0 = SortAgentObservationWrapper(
        env0,
        params=SortAgentObservationParams(
            max_closest_agent_count=max_closest_agent_count,
            remove_non_meeting_agents=True,
            item_axis=1))

    # reverse each remaining agents' observation
    env0 = ReverseSequenceObservationWrapper(env0)

    # rotate and translate each item based on the self_index

    # pad along sequence axis
    env0 = pad_sequence_obs_wrapper(
        env0,
        params=PadSequenceObsParams(max_sequence_length=max_sequence_length,
                                    pad_at='end',
                                    value=np.nan,
                                    target_axis=0))
    assert isinstance(env0, pettingzoo.ParallelEnv)

    # pad along item axis
    env0 = pad_sequence_obs_wrapper(
        env0,
        params=PadSequenceObsParams(
            max_sequence_length=max_closest_agent_count,
            pad_at='end',
            value=np.nan,
            target_axis=1))
    assert isinstance(env0, pettingzoo.ParallelEnv)

    # move the item axis before the sequence axis
    env0 = move_axis_obs_wrapper(env0,
                                 params=MoveAxisParams(source_axis=1,
                                                       destination_axis=0))
    assert isinstance(env0, pettingzoo.ParallelEnv)

    if normalize_trajectories:
        # normalize multi agent trajectories for each agent relative to itself (first trajectory)
        y_axis = [0., 1., 0.]
        normalize_params = NormalizeMultiAgentTrajectoriesObservationParams(
            trajectory_transform_params=TrajectoryTransformerParams(
                rotate_around='first',
                rotate_to=y_axis,
                translate_relative_to='first',
                project_to='xy_plane'),
            position_3d_start_column_index=0,
            velocity_3d_start_column_index=3,
            pad_value=np.nan)

        env0 = normalize_multi_agent_trajectories_obs_wrapper(
            env0, params=normalize_params)
        assert isinstance(env0, pettingzoo.ParallelEnv)

    # replace nans to zeros
    env0 = nan_to_zero_obs_wrapper(env0)
    assert isinstance(env0, pettingzoo.ParallelEnv)

    # if normalize_trajectories:
    #     # normalize distance
    #     env0 = normalize_position_by_logaritmic_distance_obs_wrapper(env0,
    #                                                           position_3d_start_column_index=0)
    #     assert isinstance(env0, pettingzoo.ParallelEnv)

    return env0
