from dataclasses import dataclass
import logging

import numpy as np
import gymnasium
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper


# REVIEW: DELETE
@dataclass
class TrajectorySequenceParams:
    max_sequence_length: int
    fill_before_with_zeros: bool
    fill_after_with_zeros: bool
    reverse: bool


def trajectory_sequence_obs_wrapper(env, params: TrajectorySequenceParams):

    class TrajectorySequenceModifier(BaseModifier):

        _params: TrajectorySequenceParams

        _stack: np.ndarray | None

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

            assert not (params.fill_before_with_zeros
                        and params.fill_after_with_zeros)
            assert params.max_sequence_length > 0, 'please set max_sequence_length'

            self._log.debug("__init__; max_sequence_length==%s",
                            params.max_sequence_length)

            self._params = params

        def reset(self,
                  seed: int | None = None,
                  options: dict | None = None) -> None:
            self._log.debug("reset; seed=%s,options=%s", seed, options)

            self._stack = None

        def modify_obs_space(
                self,
                obs_space: gymnasium.spaces.Box) -> gymnasium.spaces.Space:

            self._log.debug("modify_obs_space; obs_space=%s", obs_space)

            low = np.repeat(obs_space.low[np.newaxis, ...],
                            self._params.max_sequence_length,
                            axis=0)
            high = np.repeat(obs_space.high[np.newaxis, ...],
                             self._params.max_sequence_length,
                             axis=0)
            new_obs_space = gymnasium.spaces.Box(low=low,
                                                 high=high,
                                                 dtype=obs_space.dtype)

            self._log.debug("modify_obs_space; new_obs_space=%s",
                            new_obs_space)

            return new_obs_space

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:

            self._log.debug("modify_obs; obs=%s", obs)

            item = np.expand_dims(obs, axis=0)
            stack = self._stack
            self._log.debug("stack before; stack=%s", stack)

            if stack is None:
                stack = item
            else:
                # insert at the end
                stack = np.vstack((stack, item))

            # use only the last max_sequence_length items
            stack = stack[-self._params.max_sequence_length:]

            # save the stack before the post process
            self._stack = stack

            # zero padding
            stack = self._zero_pad(stack)

            if self._params.reverse:
                # reverse the time
                stack = np.flip(stack, axis=0)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            stack.shape, stack)

            return stack

        def _zero_pad(self, stack: np.ndarray):
            new_length = stack.shape[0]
            if new_length < self._params.max_sequence_length:

                # zero padding
                zeros = np.zeros((self._params.max_sequence_length -
                                  new_length, ) + stack.shape[1:])

                if self._params.fill_before_with_zeros:
                    stack = np.vstack((zeros, stack))
                elif self._params.fill_after_with_zeros:
                    stack = np.vstack((stack, zeros))
            return stack

    return shared_wrapper(env, TrajectorySequenceModifier)
