from typing import Any
import numpy as np
import gymnasium
import pettingzoo
from supersuit import frame_skip_v0

from .time_params import TimeParameters, calculate_frame_skip_number
from .glider_env_params import EgocentricSpatialTransformationParameters

from .wrappers import (
    DiscreteToContinuousParams, DiscreteToContinuousDegreesParams,
    GliderDictToArrayObservationParams, glider_dict_to_array_obs_wrapper,
    TrajectorySequenceParams, trajectory_sequence_obs_wrapper,
    trajectory_egocentric_sequence_obs_wrapper,
    TrajectoryEgocentricSequenceObservationTransformerParams,
    TrajectoryTransformerParams, ContextWindowRelativePositionParams,
    context_window_relative_position_obs_wrapper,
    discrete_to_continuous_action_wrapper)


# create np.array observation from dictionary
def apply_observation_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv
) -> gymnasium.Env | pettingzoo.ParallelEnv:
    observation_params = GliderDictToArrayObservationParams(
        include_absolute_position=True,
        include_velocity_vector=True,
        include_vertical_velocity=False,
        include_roll=True,
        include_yaw=False,
        include_earth_relative_velocity_norm=False,
        include_airmass_relative_velocity_norm=False,
    )

    wrapped_env = glider_dict_to_array_obs_wrapper(env, observation_params)
    return wrapped_env


def apply_frame_skip_wrapper(
        env: gymnasium.Env | pettingzoo.ParallelEnv,
        time_params: TimeParameters,
        default_action: Any) -> gymnasium.Env | pettingzoo.ParallelEnv:

    frame_skip_num = calculate_frame_skip_number(time_params)

    if isinstance(env, gymnasium.Env):
        # no need to pass default action
        wrapped_env = frame_skip_v0(
            env,
            num_frames=frame_skip_num,
        )
    else:
        wrapped_env = frame_skip_v0(
            env,
            num_frames=frame_skip_num,
            default_action=default_action,
        )

    assert isinstance(wrapped_env, gymnasium.Env) or isinstance(
        wrapped_env, pettingzoo.ParallelEnv)
    return wrapped_env


def apply_spatial_transformation_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv, spatial_transformation: str,
    max_sequence_length: int, egocentric_spatial_transformation: None
    | EgocentricSpatialTransformationParameters
) -> gymnasium.Env | pettingzoo.ParallelEnv:

    wrapped_env = None
    if spatial_transformation == 'obsolete':
        # trajectory sequence with zero padding
        is_trajectory_reversed = False

        trajectory_sequence_params = TrajectorySequenceParams(
            max_sequence_length=max_sequence_length,
            fill_before_with_zeros=False,
            fill_after_with_zeros=True,
            reverse=is_trajectory_reversed,
        )
        wrapped_env = trajectory_sequence_obs_wrapper(
            env, trajectory_sequence_params)

        # absolute to relative position relative to the first position in the context window
        relative_position_params = ContextWindowRelativePositionParams(
            is_reversed=is_trajectory_reversed,
            absolute_position_xyz_start_column=0,
        )

        wrapped_env = context_window_relative_position_obs_wrapper(
            wrapped_env, relative_position_params)

    elif spatial_transformation == 'obsolete_new_implementation':

        egocentric_params = TrajectoryEgocentricSequenceObservationTransformerParams(
            max_sequence_length=max_sequence_length,
            trajectory_transform_params=TrajectoryTransformerParams(
                rotate_around=None,
                rotate_to=None,
                translate_relative_to='first'),
            position_3d_start_column_index=0,
            velocity_3d_start_column_index=3,
            reverse=False,
            zero_pad_at='end')

        wrapped_env = trajectory_egocentric_sequence_obs_wrapper(
            env, egocentric_params)

    elif spatial_transformation == 'egocentric':

        assert egocentric_spatial_transformation is not None, 'egocentric_spatial_transformation parameters missing'

        egocentric_params = egocentric_spatial_transformation
        assert egocentric_params.relative_to == 'first' or egocentric_params.relative_to == 'last'

        y_axis = [0., 1., 0.]
        egocentric_wrapper_params = TrajectoryEgocentricSequenceObservationTransformerParams(
            max_sequence_length=max_sequence_length,
            trajectory_transform_params=TrajectoryTransformerParams(
                rotate_around='first'
                if egocentric_params.relative_to == 'first' else 'last',
                rotate_to=y_axis,
                translate_relative_to='first'
                if egocentric_params.relative_to == 'first' else 'last',
                project_to='xy_plane'),
            position_3d_start_column_index=0,
            velocity_3d_start_column_index=3,
            reverse=egocentric_params.reverse,
            zero_pad_at='end')

        wrapped_env = trajectory_egocentric_sequence_obs_wrapper(
            env, egocentric_wrapper_params)
    else:
        assert False, f'invalid parameters: {spatial_transformation}'

    assert wrapped_env is not None
    assert isinstance(wrapped_env, gymnasium.Env) or isinstance(
        wrapped_env, pettingzoo.ParallelEnv)
    return wrapped_env


def apply_discrete_to_continous_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv,
    params: DiscreteToContinuousDegreesParams
) -> gymnasium.Env | pettingzoo.ParallelEnv:

    discrete_params = DiscreteToContinuousParams(
        discrete_count=params.discrete_count,
        low=np.deg2rad(params.low_deg),
        high=np.deg2rad(params.high_deg))

    wrapped_env = discrete_to_continuous_action_wrapper(env, discrete_params)
    assert isinstance(wrapped_env, gymnasium.Env) or isinstance(
        wrapped_env, pettingzoo.ParallelEnv)

    return wrapped_env
