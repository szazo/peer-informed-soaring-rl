import gymnasium

from env.glider.base import (apply_discrete_to_continous_wrapper,
                             apply_frame_skip_wrapper,
                             apply_observation_wrapper,
                             apply_spatial_transformation_wrapper)
from env.glider.single.singleglider_env_base import SingleGliderEnvBase
from .singleglider_env_params import SingleGliderEnvParameters
from ..air_velocity_field import AirVelocityFieldInterface
from ..aerodynamics import AerodynamicsInterface

from ..base.agent import GliderInitialConditionsCalculator


def make_singleglider_env(params: SingleGliderEnvParameters,
                          air_velocity_field: AirVelocityFieldInterface,
                          aerodynamics: AerodynamicsInterface):

    glider_initial_conditions_calculator = GliderInitialConditionsCalculator(
        initial_conditions_params=params.initial_conditions_params,
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
    )

    env = SingleGliderEnvBase(
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
        glider_agent_params=params.glider_agent_params,
        simulation_box_params=params.simulation_box_params,
        time_params=params.time_params,
        cutoff_params=params.cutoff_params,
        reward_params=params.reward_params,
        render_params=params.render_params,
        initial_conditions_calculator=glider_initial_conditions_calculator)

    # frame_skip
    env = apply_frame_skip_wrapper(
        env,
        time_params=params.time_params,
        default_action=params.glider_agent_params.default_action)

    # observation wrapper
    env = apply_observation_wrapper(env)

    # spatial transformation
    env = apply_spatial_transformation_wrapper(
        env,
        spatial_transformation=params.spatial_transformation,
        max_sequence_length=params.max_sequence_length,
        egocentric_spatial_transformation=params.
        egocentric_spatial_transformation)

    # discrete to continous action wrapper
    env = apply_discrete_to_continous_wrapper(
        env, params=params.discrete_continuous_mapping)

    assert isinstance(env, gymnasium.Env)

    return env
