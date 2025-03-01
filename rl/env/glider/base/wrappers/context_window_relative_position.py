from dataclasses import dataclass
import logging

import numpy as np
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper

@dataclass
class ContextWindowRelativePositionParams:
    is_reversed: bool
    absolute_position_xyz_start_column: int

def context_window_relative_position_obs_wrapper(env, params: ContextWindowRelativePositionParams):

    class ContextWindowRelativePositionModifier(BaseModifier):

        _params: ContextWindowRelativePositionParams

        _stack: np.ndarray | None

        def __init__(self):

            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

            self._params = params

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:

            self._log.debug("modify_obs; obs=%s", obs)

            assert obs.ndim == 2, 'only two dimensional observation is supported'

            # check the array
            non_zero_row_indices = np.nonzero(obs.any(axis=-1))[-1]
            assert non_zero_row_indices.size > 0, 'full zero observation not supported'

            first_non_zero_row_index = non_zero_row_indices[0]
            last_non_zero_row_index = non_zero_row_indices[-1]

            # find the reference position
            position_column = self._params.absolute_position_xyz_start_column
            reference_absolute_position = \
                obs[first_non_zero_row_index, position_column:position_column + 3] \
                if not self._params.is_reversed \
                else obs[last_non_zero_row_index, position_column:position_column + 3]

            self._log.debug('reference_absolute_position=%s', reference_absolute_position)

            # calculate the relative positions
            relative_positions = obs[first_non_zero_row_index:last_non_zero_row_index + 1,
                                     position_column:position_column + 3] - reference_absolute_position
            self._log.debug("relative positions: %s", relative_positions)

            # replace relative positions in the observation
            obs = obs.copy()
            obs[first_non_zero_row_index:last_non_zero_row_index + 1,
                position_column:position_column + 3] = relative_positions

            self._log.debug("modify_obs; new_obs=%s", obs)

            return obs

    return shared_wrapper(env, ContextWindowRelativePositionModifier)
