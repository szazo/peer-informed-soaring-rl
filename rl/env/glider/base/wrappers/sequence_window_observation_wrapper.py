from typing import Any
import logging
from dataclasses import dataclass
import numpy as np
import gymnasium
import pettingzoo
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper

from utils import VectorN, VectorNxN
from .sequence_window import SequenceWindow


@dataclass
class SequenceWindowObsParams:
    max_sequence_length: int


def sequence_window_obs_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv,
    params: SequenceWindowObsParams
) -> gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv:

    class SequenceWindowObservationModifier(BaseModifier):

        _sequence_window: SequenceWindow

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

            self._sequence_window = SequenceWindow(params.max_sequence_length)

        def reset(self, seed: int | None = None, options: dict | None = None):
            self._log.debug("reset")
            self._sequence_window.reset()

        def modify_obs_space(
                self,
                obs_space: gymnasium.spaces.Box) -> gymnasium.spaces.Space:

            self._log.debug("modify_obs_space; obs_space=%s", obs_space)

            low = np.repeat(obs_space.low[np.newaxis, ...],
                            params.max_sequence_length,
                            axis=0)
            high = np.repeat(obs_space.high[np.newaxis, ...],
                             params.max_sequence_length,
                             axis=0)
            dtype: Any = obs_space.dtype
            new_obs_space = gymnasium.spaces.Box(low=low,
                                                 high=high,
                                                 dtype=dtype)

            self._log.debug("modify_obs_space; new_obs_space=%s",
                            new_obs_space)

            return new_obs_space

        def modify_obs(self, obs: VectorN) -> VectorNxN:
            self._log.debug("modify_obs; obs=%s", obs)

            # add it at the end of the current sequence with windowing
            result_obs = self._sequence_window.add(obs)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(env, SequenceWindowObservationModifier)
