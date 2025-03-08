from env.glider.multi.apply_share_wrappers import apply_share_wrappers
import pettingzoo
from ..base.agent import (GliderInitialConditionsCalculator)
from .agent_spawner import AgentSpawner2

from ..aerodynamics import AerodynamicsInterface
from ..air_velocity_field import AirVelocityFieldInterface
from .multiglider_env_params import MultiGliderEnvParameters
from .multiglider_env_base import MultiGliderEnvBase, AgentGroup
from ..base.visualization import make_visualization
from ..base.agent import (AgentID, GliderAgentObsType, GliderAgentActType,
                          GliderInitialConditionsParameters)
from ..base import (
    AgentID,
    apply_discrete_to_continous_wrapper,
    apply_frame_skip_wrapper,
    apply_observation_wrapper,
)
from trainer.multi_agent.parallel_pettingzoo_env import ParallelPettingZooEnv


def make_multiglider_env(
        env_name: str, params: MultiGliderEnvParameters,
        air_velocity_field: AirVelocityFieldInterface,
        aerodynamics: AerodynamicsInterface) -> ParallelPettingZooEnv:

    # sort the agent groups by order
    sorted_agent_groups = sorted(params.agent_groups.items(),
                                 key=lambda item: item[1].order)
    agent_groups = [
        AgentGroup(
            name=name,
            terminate_if_finished=agent_group.terminate_if_finished,
            spawner=AgentSpawner2[AgentID](name, agent_group.spawner),
            initial_conditions_calculator=_create_initial_conditions_calculator(
                agent_group.initial_conditions_params,
                air_velocity_field=air_velocity_field,
                aerodynamics=aerodynamics))
        for name, agent_group in sorted_agent_groups
    ]

    visualization = make_visualization(
        simulation_box_params=params.simulation_box_params,
        render_params=params.render_params,
        layout_params=params.layout_params,
        air_velocity_field=air_velocity_field,
        thermal_core_3d_plot_params=params.thermal_core_3d_plot_params,
    )

    env = MultiGliderEnvBase(
        time_params=params.time_params,
        simulation_box_params=params.simulation_box_params,
        glider_agent_params=params.glider_agent_params,
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
        agent_groups=agent_groups,
        cutoff_params=params.cutoff_params,
        reward_params=params.reward_params,
        render_params=params.render_params,
        visualization=visualization,
    )

    # frame_skip
    env = apply_frame_skip_wrapper(
        env,
        time_params=params.time_params,
        default_action=params.glider_agent_params.default_action)

    # observation wrapper
    env = apply_observation_wrapper(env)
    assert isinstance(env, pettingzoo.ParallelEnv)

    env = apply_share_wrappers(
        env,
        max_sequence_length=params.max_sequence_length,
        max_closest_agent_count=params.max_closest_agent_count)

    assert isinstance(env, pettingzoo.ParallelEnv)

    # discrete to continous action wrapper
    env = apply_discrete_to_continous_wrapper(
        env, params=params.discrete_continuous_mapping)

    assert isinstance(env, pettingzoo.ParallelEnv)

    env = ParallelPettingZooEnv[AgentID, GliderAgentObsType,
                                GliderAgentActType](name=env_name, env=env)

    return env


def _create_initial_conditions_calculator(
        params: GliderInitialConditionsParameters,
        air_velocity_field: AirVelocityFieldInterface,
        aerodynamics: AerodynamicsInterface):

    glider_initial_conditions_calculator = GliderInitialConditionsCalculator(
        initial_conditions_params=params,
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
    )

    return glider_initial_conditions_calculator
