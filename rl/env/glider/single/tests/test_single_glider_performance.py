import time
import pytest
from dataclasses import asdict, dataclass

from thermal.api import AirVelocityFieldInterface
from thermal.gaussian import GaussianAirVelocityFieldParameters, make_gaussian_air_velocity_field
from env.glider.aerodynamics import SimpleAerodynamicsParameters, SimpleAerodynamics

from env.glider.base import SimulationBoxParameters, TimeParameters
from env.glider.base.agent import (
    GliderCutoffParameters, GliderCutoffCalculator,
    GliderInitialConditionsParameters, GliderRewardParameters,
    GliderRewardCalculator, GliderInitialConditionsCalculator,
    GliderAgentParameters)
from env.glider.base.visualization import RenderParameters

from env.glider.single import SingleGliderEnvBase
from utils.create_params_from_yaml_string import create_params_from_yaml_string


@dataclass
class Instances:
    env: SingleGliderEnvBase
    aerodynamics: SimpleAerodynamics
    air_velocity_field: AirVelocityFieldInterface
    initial_conditions_calculator: GliderInitialConditionsCalculator
    cutoff_calculator: GliderCutoffCalculator
    reward_calculator: GliderRewardCalculator


@pytest.mark.skip(reason='this is a performance test')
def test_perfomance():
    """
    Simple test for measuring performance (remove skip to run)
    """

    # given
    instances = _create_env()
    env = instances.env
    env.reset()

    episode_count = 100

    # when
    step_count = 0

    start = time.time()

    for _ in range(episode_count):

        ep_step_count = 0
        env.reset()
        while True:
            action = env.action_space.sample()
            _, _, terminated, truncated, info = env.step(action)
            step_count += 1
            ep_step_count += 1

            if terminated or truncated:
                print('termination/truncation reason:', info['cutoff_reason'])
                break
    end = time.time()
    print('step count', step_count)
    print(f'elapsed time {end - start:.2f}s')
    print(f'speed: {step_count / float(end-start):.2f} step/s')


air_velocity_config_yaml = """
box_size: [6050, 6050, 2050]
turbulence_episode_regenerate_probability: 0.1
thermal:
  # maximum radius
  max_r_m_normal_mean: 60.0
  max_r_m_normal_sigma: 25.0

  # altitude of the maximum radius
  max_r_altitude_m_normal_mean: 1000.0
  max_r_altitude_m_normal_sigma: 100.0

  # spread of the maximum radius around the maximum radius altitude
  max_r_m_sigma_normal_mean: 1500.0
  max_r_m_sigma_normal_sigma: 100.0

  # vertical velocity at the core
  w_max_m_per_s_normal_mean: 3.0
  w_max_m_per_s_normal_sigma: 1.5

  # used for control that the specified sigma should contain the most of the bell volume: sigma'=sigma/k
  sigma_k: 2.5
  # because the thermal with the radius is gaussian, we use
  # k for the radius too to limit distribution into the range of specified thermal radius
  radius_k: 1.5

turbulence:
  noise_multiplier_normal_mean: 1.2
  noise_multiplier_normal_sigma: 0.2
  noise_gaussian_filter_sigma_normal_mean_m: 20.0
  noise_gaussian_filter_sigma_normal_sigma_m: 5.0
  noise_grid_spacing_m: 50.0
  sigma_k: 2.5


wind:
  horizontal_wind_speed_at_2m_m_per_s_normal_mean: 1.2
  horizontal_wind_speed_at_2m_m_per_s_normal_sigma: 0.2
  horizontal_wind_profile_vertical_spacing_m: 100.0

  noise:
    noise_multiplier_normal_mean: 3.0
    noise_multiplier_normal_sigma: 2.5
    noise_gaussian_filter_sigma_normal_mean_m: 30.0
    noise_gaussian_filter_sigma_normal_sigma_m: 10.0
    noise_grid_spacing_m: 50.0
    sigma_k: 2.5
"""


def _create_env() -> Instances:
    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))

    air_velocity_field_params = create_params_from_yaml_string(
        air_velocity_config_yaml, node_type=GaussianAirVelocityFieldParameters)
    air_velocity_field = make_gaussian_air_velocity_field(
        **asdict(air_velocity_field_params))

    initial_conditions_calculator = GliderInitialConditionsCalculator(
        initial_conditions_params=GliderInitialConditionsParameters(),
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field)

    simulation_box_params = SimulationBoxParameters(box_size=(5000, 5000,
                                                              2000))
    time_params = TimeParameters(dt_s=0.1, decision_dt_s=0.4)

    reward_cutoff_calculator = GliderRewardAndCutoffCalculator(
        reward_params=GliderRewardParameters(),
        cutoff_params=GliderCutoffParameters(),
        simulation_box=simulation_box_params)

    render_params = RenderParameters(mode='rgb_array')

    agent_params = GliderAgentParameters()

    env = SingleGliderEnvBase(
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
        initial_conditions_calculator=initial_conditions_calculator,
        reward_and_cutoff_calculator=reward_cutoff_calculator,
        glider_agent_params=agent_params,
        simulation_box_params=simulation_box_params,
        time_params=time_params,
        render_params=render_params,
    )

    return Instances(env, aerodynamics, air_velocity_field,
                     initial_conditions_calculator, reward_cutoff_calculator)
