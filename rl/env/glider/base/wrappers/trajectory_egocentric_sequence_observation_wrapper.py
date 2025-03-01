from typing import Any
import logging

import numpy as np
import gymnasium
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper

from .trajectory_egocentric_sequence_observation_transformer import (
    TrajectoryEgocentricSequenceObservationTransformerParams,
    TrajectoryEgocentricSequenceObservationTransformer)


def trajectory_egocentric_sequence_obs_wrapper(
        env, params: TrajectoryEgocentricSequenceObservationTransformerParams):

    class TrajectoryEgocentricSequenceObservationModifier(BaseModifier):

        _transformer: TrajectoryEgocentricSequenceObservationTransformer

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

            self._transformer = TrajectoryEgocentricSequenceObservationTransformer(
                params)

        def reset(self,
                  seed: int | None = None,
                  options: dict | None = None) -> None:
            self._log.debug("reset; seed=%s,options=%s", seed, options)

            self._transformer.reset()

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

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:

            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = self._transformer.modify_obs(obs)
            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(env, TrajectoryEgocentricSequenceObservationModifier)
