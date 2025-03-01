from typing import Any, Literal
import logging
from dataclasses import dataclass
import numpy as np
import gymnasium
import pettingzoo
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper

from utils import VectorNxN
from .zero_pad_sequence import zero_pad_sequence


@dataclass
class PadSequenceObsParams:
    max_sequence_length: int
    pad_at: Literal['start', 'end']
    value: Any
    target_axis: int = 0


def pad_sequence_obs_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv,
    params: PadSequenceObsParams
) -> gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv:

    class PadSequenceObservationModifier(BaseModifier):

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

        def modify_obs_space(
                self,
                obs_space: gymnasium.spaces.Box) -> gymnasium.spaces.Space:

            # recreate max_sequence_length observation space based on the first row
            low = obs_space.low[0]
            high = obs_space.high[0]

            self._log.debug("modify_obs_space; obs_space=%s", obs_space)

            low = np.repeat(low[np.newaxis, ...],
                            params.max_sequence_length,
                            axis=0)
            high = np.repeat(high[np.newaxis, ...],
                             params.max_sequence_length,
                             axis=0)
            dtype: Any = obs_space.dtype
            new_obs_space = gymnasium.spaces.Box(low=low,
                                                 high=high,
                                                 dtype=dtype)

            self._log.debug("modify_obs_space; new_obs_space=%s",
                            new_obs_space)

            return new_obs_space

        def modify_obs(self, obs: VectorNxN) -> VectorNxN:
            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = zero_pad_sequence(
                obs,
                max_sequence_length=params.max_sequence_length,
                pad_at=params.pad_at,
                constant_value=params.value,
                target_axis=params.target_axis)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(env, PadSequenceObservationModifier)
