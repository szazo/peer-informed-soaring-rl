from dataclasses import asdict, dataclass
from pytest_mock import MockerFixture
import math
import numpy as np
from gymnasium.utils import seeding

from thermal.zero import ZeroAirVelocityField
from env.glider.aerodynamics import SimpleAerodynamicsParameters, SimpleAerodynamics

from env.glider.base import SimulationBoxParameters, TimeParameters
from env.glider.base.agent import (GliderCutoffParameters,
                                   GliderInitialConditionsParameters,
                                   GliderRewardParameters,
                                   GliderInitialConditionsCalculator,
                                   GliderAgentParameters, GliderAgentObsType)
from env.glider.base.visualization import RenderParameters
from env.glider.single import SingleGliderEnvBase


@dataclass
class Instances:
    env: SingleGliderEnvBase
    aerodynamics: SimpleAerodynamics
    air_velocity_field: ZeroAirVelocityField
    initial_conditions_calculator: GliderInitialConditionsCalculator


def _create_env() -> Instances:
    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))
    air_velocity_field = ZeroAirVelocityField()

    initial_conditions_calculator = GliderInitialConditionsCalculator(
        initial_conditions_params=GliderInitialConditionsParameters(),
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field)

    simulation_box_params = SimulationBoxParameters(box_size=(5000, 5000,
                                                              2000))
    time_params = TimeParameters(dt_s=0.1, decision_dt_s=0.4)

    render_params = RenderParameters(mode='rgb_array')

    agent_params = GliderAgentParameters()

    env = SingleGliderEnvBase(
        aerodynamics=aerodynamics,
        air_velocity_field=air_velocity_field,
        initial_conditions_calculator=initial_conditions_calculator,
        cutoff_params=GliderCutoffParameters(),
        reward_params=GliderRewardParameters(),
        glider_agent_params=agent_params,
        simulation_box_params=simulation_box_params,
        time_params=time_params,
        render_params=render_params,
    )

    return Instances(env, aerodynamics, air_velocity_field,
                     initial_conditions_calculator)


def test_should_use_seed_and_return_initial_conditions_when_reset_called(
        mocker: MockerFixture):

    # given
    instances = _create_env()
    env = instances.env

    seed = 42

    initial_conditions_seed_spy = mocker.spy(
        instances.initial_conditions_calculator, 'seed')
    air_velocity_field_seed_spy = mocker.spy(instances.air_velocity_field,
                                             'seed')

    # when
    obs, info = env.reset(seed=seed)

    # then
    initial_conditions_seed_spy.assert_called_with(seed)
    air_velocity_field_seed_spy.assert_called_with(seed)

    reference_np_random, _ = seeding.np_random(seed)
    assert reference_np_random.uniform() == env.np_random.uniform()

    _check_info(info, obs)


def test_step_should_step():

    # given
    instances = _create_env()
    env = instances.env
    env.reset()

    # when
    obs, reward, terminated, truncated, info = env.step(0.)

    # then
    assert not math.isclose(reward, 0)
    assert not terminated
    assert not truncated
    _check_info(info, obs)

    assert math.isclose(obs['t_s'], 0.1)


def _check_info(info: dict, obs: GliderAgentObsType):
    # check initial conditions merged
    initial_conditions = info['initial_conditions']
    assert initial_conditions['glider'] is not None
    assert initial_conditions['air_velocity_field'] is not None

    # should also contain the observation
    for key, value in obs.items():
        assert info[key] == value if np.isscalar(value) else np.allclose(
            info[key], np.array(value))
