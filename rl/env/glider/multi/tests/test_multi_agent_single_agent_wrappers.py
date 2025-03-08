from env.glider.base import EgocentricSpatialTransformationParameters, apply_spatial_transformation_wrapper
from env.glider.multi.make_multiglider_env import apply_share_wrappers
import numpy as np
import gymnasium
import pettingzoo
from pytest_mock import MockerFixture


def test_sanity_should_observation_wrappers_should_return_same_result(
        mocker: MockerFixture):

    # given
    positions = np.array(([200., 300., 500.], [210., 320.,
                                               530.], [220., 340., 560.]))

    velocities = np.array(([10., 20., 30.], [15., 25., 35.], [20., 30., 40.]))

    rolls = np.array([[0.1], [0.2], [0.3]])

    input = np.hstack((positions, velocities, rolls))

    multi_env = _create_multi_agent_wrapped_env(input, mocker)
    single_env = _create_single_agent_wrapped_env(input, mocker)

    single_env_iterator = _single_agent_iterator(single_env)
    multi_env_iterator = _multi_agent_iterator(multi_env)

    obs_zip = zip(single_env_iterator, multi_env_iterator)
    for (single_obs, multi_obs) in obs_zip:
        assert np.allclose(single_obs, multi_obs)
        assert single_obs.shape == (2, 7)
        assert multi_obs.shape == (1, 2, 7)


def _multi_agent_iterator(env: pettingzoo.ParallelEnv):

    obs, info = env.reset()
    yield obs['a0']

    obs, *_ = env.step({'a0': 1.})
    yield obs['a0']

    obs, *_ = env.step({'a0': 2.})
    yield obs['a0']


def _single_agent_iterator(env: gymnasium.Env):

    obs, info = env.reset()
    yield obs

    obs, *_ = env.step(1.)
    yield obs

    obs, *_ = env.step(2.)
    yield obs


def _create_single_agent_wrapped_env(input: np.ndarray,
                                     mocker: MockerFixture) -> gymnasium.Env:
    env = mocker.create_autospec(gymnasium.Env)
    env.observation_space = gymnasium.spaces.Box(0., 10., shape=(7, ))
    env.action_space = gymnasium.spaces.Box(0., 5., shape=(1, ))

    env.reset.return_value = (input[0], {})
    step_results = [(input[1], 1., False, False, {}),
                    (input[2], 2., False, False, {})]
    env.step.side_effect = step_results

    env = apply_spatial_transformation_wrapper(
        env,
        spatial_transformation='egocentric',
        max_sequence_length=2,
        egocentric_spatial_transformation=
        EgocentricSpatialTransformationParameters(relative_to='last',
                                                  reverse=True))

    assert isinstance(env, gymnasium.Env)
    return env


def _create_multi_agent_wrapped_env(input: np.ndarray, mocker: MockerFixture):
    env = mocker.create_autospec(pettingzoo.ParallelEnv)
    env.observation_space.return_value = gymnasium.spaces.Box(0.,
                                                              10.,
                                                              shape=(7, ))
    env.possible_agents = ['a0']
    env.agents = env.possible_agents

    def agent_result(agent_id: str, obs: np.ndarray, reward: float):
        return ({
            agent_id: obs
        }, {agent_id, reward}, {
            agent_id: False
        }, {
            agent_id: False
        }, {
            agent_id: {}
        })

    env.reset.return_value = ({'a0': input[0]}, {'a0': {}})
    step_results = [
        agent_result('a0', input[1], 1.),
        agent_result('a0', input[2], 2.)
    ]
    env.step.side_effect = step_results

    env = apply_share_wrappers(env,
                               max_sequence_length=2,
                               max_closest_agent_count=1,
                               normalize_trajectories=True)

    return env
