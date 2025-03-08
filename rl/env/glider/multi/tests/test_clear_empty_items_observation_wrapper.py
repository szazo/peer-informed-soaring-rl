import gymnasium
from functools import reduce
import numpy as np
from pytest_mock import MockerFixture

from ..multi_agent_observation_share_wrapper import (
    ClearEmptyItemsObsParams, clear_empty_items_along_axis,
    clear_empty_items_obs_wrapper)


def test_clear_empty_items_obs_wrapper(mocker: MockerFixture):

    # given
    shape = (3, )
    length = 3
    item0 = np.arange(length).reshape(shape)
    item1 = np.full(shape, 0)

    obs0 = np.vstack((item0, item1))

    item2 = np.full(shape, np.nan)
    item3 = item0 + length
    obs1 = np.vstack((item2, item3))

    env = mocker.create_autospec(gymnasium.Env)
    env.observation_space = gymnasium.spaces.Box(0., 10., shape=(2, 2))
    env.action_space = gymnasium.spaces.Box(0., 5., shape=(1, ))

    env.reset.return_value = (obs0, {})
    step_results = [
        (obs1, 1, False, False, {}),
    ]
    env.step.side_effect = step_results

    params = ClearEmptyItemsObsParams(axis=0, empty_value=np.nan)
    env = clear_empty_items_obs_wrapper(env, params)
    assert isinstance(env, gymnasium.Env)

    # when
    obs_reset, _ = env.reset()

    obs_step1, reward, terminated, truncated, info = env.step(1)

    # then
    assert np.allclose(obs0, obs_reset)
    assert np.allclose(item3, obs_step1)
    assert obs_step1.shape == (1, 3)


def test_clear_empty_items_should_clear_when_non_first_axis():

    # given
    shape = (2, 2, 3)
    length = reduce(lambda x, y: x * y, shape)

    item1 = np.arange(length).reshape(shape)
    item2 = np.full(shape, np.nan)
    item3 = item1 + length
    item4 = np.full(shape, np.nan)

    # stack along the second axis
    axis = 1
    stack = np.stack((item1, item2, item3, item4), axis=axis)

    # when
    clean = clear_empty_items_along_axis(stack, axis=axis, empty_value=np.nan)

    # then
    expected = np.stack((item1, item3), axis=1)

    assert expected.shape == clean.shape
    assert np.allclose(expected, clean)
