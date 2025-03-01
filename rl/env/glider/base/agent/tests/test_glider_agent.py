from dataclasses import asdict
from env.glider.base.agent.air_velocity_post_processor import (
    AirVelocityGaussianNoiseParameters, AirVelocityPostProcessorParams,
    GaussianNoiseParameters)
import numpy as np
from pytest_mock import MockerFixture
from env.glider.aerodynamics import SimpleAerodynamics, SimpleAerodynamicsParameters
from env.glider.aerodynamics.api import AerodynamicsInterface
from env.glider.base.agent.air_velocity_filter import create_exponential_filter_kernel, create_mean_kernel
from env.glider.base.agent.control_dynamics import ControlDynamicsParams, SystemParams, ControlParams
from gymnasium.utils import seeding
from thermal.api import AirVelocityFieldInterface
from thermal.zero import ZeroAirVelocityField
from env.glider.base import AgentID, SimulationBoxParameters
from env.glider.base.agent import GliderAgent
from env.glider.base.agent.glider_agent import GliderAgentParameters
from env.glider.base.agent.initial_conditions.glider_initial_conditions_calculator import GliderInitialConditionsCalculator, GliderInitialConditionsParameters
from env.glider.base.agent.reward_cutoff import GliderCutoffCalculator, GliderRewardCalculator, GliderRewardParameters, GliderCutoffParameters
from deepdiff import DeepDiff
from utils import Vector3


def test_state_clone_should_deep_copy_state():
    # given
    agent = _create_agent(
        air_velocity_post_process_params=AirVelocityPostProcessorParams(
            filter_kernel=create_exponential_filter_kernel(10, 4),
            velocity_noise=_create_noise_params()))
    agent.reset(time_s=0.0)
    agent.step(action=0.0, current_time_s=0.0, next_time_s=0.4, dt_s=0.4)

    # when
    agent_clone = agent.state_clone()

    # then
    deep_diff = DeepDiff(agent, agent_clone)
    assert deep_diff == {}


def test_air_velocity_post_processor_integration(mocker: MockerFixture):

    # given
    air_velocity_field = ZeroAirVelocityField()
    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))

    noise_mean = np.array([1., 2., 3.])  # we use zero sigma
    air_velocity_t_0: Vector3 = np.array([1., 2., 3.])
    air_velocity_t_1: Vector3 = np.array([2., 3., 4.])
    air_velocity_t_2: Vector3 = np.array([3., 4., 5.])

    mocker.patch.object(air_velocity_field,
                        'get_velocity',
                        side_effect=[(air_velocity_t_0, {}),
                                     (air_velocity_t_1, {}),
                                     (air_velocity_t_2, {})])
    aerodynamics_step_spy = mocker.spy(aerodynamics, 'step')

    air_velocity_post_process_params = AirVelocityPostProcessorParams(
        filter_kernel=create_mean_kernel(kernel_size=4),
        velocity_noise=_create_noise_params())

    agent = _create_agent(
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
        air_velocity_post_process_params=air_velocity_post_process_params)

    # when
    agent.reset(time_s=0.0)
    agent.step(action=0.0, current_time_s=0.0, next_time_s=0.4, dt_s=0.4)
    agent.step(action=0.0, current_time_s=0.4, next_time_s=0.8, dt_s=0.4)

    # then
    assert np.allclose(
        aerodynamics_step_spy.call_args_list[0].
        kwargs['wind_velocity_earth_m_per_s'], air_velocity_t_0 + noise_mean)
    assert np.allclose(aerodynamics_step_spy.call_args_list[1].
                       kwargs['wind_velocity_earth_m_per_s'],
                       ((air_velocity_t_0 + noise_mean) +
                        (air_velocity_t_1 + noise_mean)) / 2)  # mean filter


def _create_agent(
    aerodynamics: AerodynamicsInterface | None = None,
    air_velocity_field: AirVelocityFieldInterface | None = None,
    air_velocity_post_process_params:
    AirVelocityPostProcessorParams = AirVelocityPostProcessorParams()):

    if aerodynamics is None:
        aerodynamics_params = SimpleAerodynamicsParameters()
        aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))

    if air_velocity_field is None:
        air_velocity_field = ZeroAirVelocityField()

    control_dynamics_params = ControlDynamicsParams(
        system=SystemParams(omega_natural_frequency=0.2,
                            zeta_damping_ratio=0.5,
                            k_process_gain=1.),
        control=ControlParams(proportional_gain=15.,
                              integral_gain=0.66,
                              derivative_gain=40.))

    params = GliderAgentParameters(
        roll_control_dynamics_params=control_dynamics_params,
        air_velocity_post_process=air_velocity_post_process_params)

    initial_conditions_calculator = GliderInitialConditionsCalculator(
        initial_conditions_params=GliderInitialConditionsParameters(),
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field)

    cutoff_calculator = GliderCutoffCalculator(
        params=GliderCutoffParameters(),
        simulation_box=SimulationBoxParameters(box_size=(100, 100, 100)))

    reward_calculator = GliderRewardCalculator(params=GliderRewardParameters())

    np_random, _ = seeding.np_random()

    agent_id = 'test'
    agent = GliderAgent(
        agent_id=AgentID(agent_id),
        parameters=params,
        initial_conditions_calculator=initial_conditions_calculator,
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
        cutoff_calculator=cutoff_calculator,
        reward_calculator=reward_calculator,
        np_random=np_random)

    return agent


def _create_noise_params(noise_mean: Vector3 = np.array([1., 2., 3.])):

    return AirVelocityGaussianNoiseParameters(
        x=GaussianNoiseParameters(mean=noise_mean[0], sigma=0.),
        y=GaussianNoiseParameters(mean=noise_mean[1], sigma=0.),
        z=GaussianNoiseParameters(mean=noise_mean[2], sigma=0.))
