import numpy as np
import gymnasium
from pytest_mock import MockerFixture
from utils.vector import VectorNx3
from ..sequence_window_observation_wrapper import sequence_window_obs_wrapper, SequenceWindowObsParams


def test_should_stack(mocker: MockerFixture):

    # given
    observations: VectorNx3 = np.array([[[1., 2.], [3., 4.]],
                                        [[5., 6.], [7., 8.]],
                                        [[9., 10.], [11., 12.]]])

    env = mocker.create_autospec(gymnasium.Env)
    env.observation_space = gymnasium.spaces.Box(0., 10., shape=(2, 2))
    env.action_space = gymnasium.spaces.Box(0., 5., shape=(1, ))

    env.reset.return_value = (observations[0], {})
    step_results = [
        (observations[1], 1, False, False, {}),
        (observations[2], 2, False, False, {}),
    ]

    env.step.side_effect = step_results

    env = sequence_window_obs_wrapper(
        env, params=SequenceWindowObsParams(max_sequence_length=2))
    assert isinstance(env, gymnasium.Env)

    # when
    obs_reset, _ = env.reset()

    obs_step1, reward, terminated, truncated, info = env.step(1)
    obs_step2, reward, terminated, truncated, info = env.step(2)

    # then
    assert env.observation_space == gymnasium.spaces.Box(0, 10, (2, 2, 2))
    assert obs_reset.shape == (1, 2, 2)
    assert np.allclose(obs_reset, [observations[0, :]])

    assert obs_step1.shape == (2, 2, 2)
    assert np.allclose(obs_step1, observations[:2, :])

    assert obs_step2.shape == (2, 2, 2)
    assert np.allclose(obs_step2, observations[1:3, :])
