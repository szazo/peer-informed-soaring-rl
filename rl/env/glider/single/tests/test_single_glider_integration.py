from dataclasses import asdict
from env.glider.single.make_singleglider_env import make_singleglider_env
import numpy as np
from utils import VectorNxN
from env.glider.aerodynamics import SimpleAerodynamics, SimpleAerodynamicsParameters
from env.glider.single.singleglider_env_params import SingleGliderEnvParameters
from thermal.zero import ZeroAirVelocityField
from env.glider.base.glider_env_params import EgocentricSpatialTransformationParameters, SimulationBoxParameters


def test_should_position_integrate_the_velocity():

    # given
    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))
    air_velocity_field = ZeroAirVelocityField()

    params = SingleGliderEnvParameters(
        max_sequence_length=10,
        spatial_transformation='egocentric',
        egocentric_spatial_transformation=
        EgocentricSpatialTransformationParameters(relative_to='first',
                                                  reverse=False),
        simulation_box_params=SimulationBoxParameters(box_size=(5000, 5000,
                                                                2000)))
    env = make_singleglider_env(params,
                                aerodynamics=aerodynamics,
                                air_velocity_field=air_velocity_field)

    env.reset()

    action = -np.deg2rad(-10)  # little left turn

    # when
    obs: VectorNxN = np.array([[]])
    for _ in range(9):
        obs, *_ = env.step(action)

    # then
    position = obs[:, 0:3]
    velocity = obs[:, 3:6]

    integrated = (position + velocity)[:-1, :]

    assert np.allclose(position[1:, :], integrated)


def test_should_reverse_and_relative_to_last_when_set():

    # given
    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))
    air_velocity_field = ZeroAirVelocityField()

    params = SingleGliderEnvParameters(
        max_sequence_length=10,
        spatial_transformation='egocentric',
        egocentric_spatial_transformation=
        EgocentricSpatialTransformationParameters(relative_to='last',
                                                  reverse=True),
        simulation_box_params=SimulationBoxParameters(box_size=(5000, 5000,
                                                                2000)))
    env = make_singleglider_env(params,
                                aerodynamics=aerodynamics,
                                air_velocity_field=air_velocity_field)

    env.reset()

    action = -np.deg2rad(-10)  # little left turn

    # when
    obs: VectorNxN = np.array([[]])
    for _ in range(9):
        obs, *_ = env.step(action)

    # then
    position = obs[:, 0:3]
    velocity = obs[:, 3:6]

    # first is zero (because reversed)
    assert np.allclose(position[0], np.array([0., 0., 0.]))

    # z is increasing (because relative to the last)
    assert np.all(np.diff(position, axis=0)[:, 2] > 0)

    # position should integrate velocity
    integrated = (position + velocity)[1:, :]
    assert np.allclose(position[:-1, :], integrated)


def test_should_new_implementation_give_same():

    # given
    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))
    air_velocity_field = ZeroAirVelocityField()

    params1 = SingleGliderEnvParameters(
        max_sequence_length=10,
        spatial_transformation='obsolete',
        simulation_box_params=SimulationBoxParameters(box_size=(5000, 5000,
                                                                2000)))
    params2 = SingleGliderEnvParameters(
        max_sequence_length=10,
        spatial_transformation='obsolete_new_implementation',
        simulation_box_params=SimulationBoxParameters(box_size=(5000, 5000,
                                                                2000)))
    env1 = make_singleglider_env(params1,
                                 aerodynamics=aerodynamics,
                                 air_velocity_field=air_velocity_field)

    env1.reset(seed=42)
    env2 = make_singleglider_env(params2,
                                 aerodynamics=aerodynamics,
                                 air_velocity_field=air_velocity_field)
    env2.reset(seed=42)

    action = -np.deg2rad(-10)  # little left turn

    # when
    obs1: VectorNxN = np.array([[]])
    obs2: VectorNxN = np.array([[]])
    for _ in range(9):
        obs1, *_ = env1.step(action)
        obs2, *_ = env2.step(action)

    # then
    assert obs1.shape == obs2.shape
    assert np.allclose(obs1, obs2)
