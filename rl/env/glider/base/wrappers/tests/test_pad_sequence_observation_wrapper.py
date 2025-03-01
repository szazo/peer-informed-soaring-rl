import numpy as np
import gymnasium
from pytest_mock import MockerFixture
from utils.vector import VectorNx3
from ..sequence_window_observation_wrapper import sequence_window_obs_wrapper, SequenceWindowObsParams
from ..pad_sequence_observation_wrapper import pad_sequence_obs_wrapper, PadSequenceObsParams


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

    # first, wrap with sequence window to get 2 dimensional observation
    env = sequence_window_obs_wrapper(
        env, params=SequenceWindowObsParams(max_sequence_length=2))

    # then with the padder wrapper
    params = PadSequenceObsParams(max_sequence_length=4,
                                  pad_at='start',
                                  value=np.nan)
    env = pad_sequence_obs_wrapper(env, params=params)
    assert isinstance(env, gymnasium.Env)

    # when
    obs_reset, _ = env.reset()

    obs_step1, reward, terminated, truncated, info = env.step(1)
    obs_step2, reward, terminated, truncated, info = env.step(2)

    # then
    assert env.observation_space == gymnasium.spaces.Box(0, 10, (4, 2, 2))
    assert obs_reset.shape == (4, 2, 2)

    assert np.allclose(obs_reset,
                       np.vstack((np.full((3, 2, 2), np.nan),
                                  np.array([observations[0, :, :]]))),
                       equal_nan=True)

    assert obs_step1.shape == (4, 2, 2)
    assert np.allclose(obs_step1,
                       np.vstack((np.full((2, 2, 2),
                                          np.nan), observations[:2, :])),
                       equal_nan=True)

    assert obs_step2.shape == (4, 2, 2)
    assert np.allclose(obs_step2,
                       np.vstack((np.full((2, 2, 2),
                                          np.nan), observations[1:3, :])),
                       equal_nan=True)
