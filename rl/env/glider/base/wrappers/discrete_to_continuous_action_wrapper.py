from dataclasses import dataclass
import logging
import numpy as np

import gymnasium
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper


@dataclass
class DiscreteToContinuousDegreesParams:
    discrete_count: int = 13
    low_deg: float = -45.
    high_deg: float = 45.


@dataclass
class DiscreteToContinuousParams:
    discrete_count: int
    low: float
    high: float


def discrete_to_continuous_action_wrapper(env,
                                          params: DiscreteToContinuousParams):

    class DiscreteToContinuousModifier(BaseModifier):

        _calculator: DiscreteToContinuousCalculator
        _params: DiscreteToContinuousParams

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

            self._params = params
            self._calculator = DiscreteToContinuousCalculator(
                discrete_count=params.discrete_count,
                low=params.low,
                high=params.high)

        def modify_action(self, act: int) -> float:
            self._log.debug('modify_action; act=%s', act)

            continuous_act = self._calculator.discrete_to_continuous(act)
            self._log.debug('continuous_act=%s', continuous_act)

            return continuous_act

        def modify_action_space(
                self,
                act_space: gymnasium.spaces.Space) -> gymnasium.spaces.Space:
            new_space = gymnasium.spaces.Discrete(self._params.discrete_count)

            self._log.debug('modify_action_space; new_space=%s', new_space)
            return new_space

    return shared_wrapper(env, DiscreteToContinuousModifier)


class DiscreteToContinuousCalculator():

    def __init__(self, discrete_count: int, low: float, high: float):

        self._discrete_count = discrete_count
        self._low = low
        self._high = high

    def discrete_to_continuous(self, discrete: int):
        if discrete < 0 or discrete > self._discrete_count - 1:
            raise Exception(f"Invalid discrete action: {discrete}")

        cont = self._low + ((self._high - self._low) /
                            (self._discrete_count - 1)) * discrete
        return cont

    def continuous_to_discrete(self, cont: float):

        discrete = (cont - self._low) / ((self._high - self._low) /
                                         (self._discrete_count - 1))

        return np.round(discrete).astype(int)
