from typing import Literal, Protocol
from dataclasses import dataclass, asdict, replace
import logging

from utils import Vector3
from .api import RewardResult
from .parameters import GliderRewardParameters


@dataclass
class RewardState:
    maximum_altitude: float


class GliderState(Protocol):
    position_earth_xyz_m: Vector3
    velocity_earth_xyz_m_per_s: Vector3


class AdditionalInfo(Protocol):
    dt_s: float
    cutoff_result: Literal['none', 'success', 'fail'] = 'none'


@dataclass
class RewardAdditionalInfo(AdditionalInfo):
    dt_s: float
    cutoff_result: Literal['none', 'success', 'fail'] = 'none'


class GliderRewardCalculator:

    _params: GliderRewardParameters
    _reward_state: RewardState | None

    def __init__(self, params: GliderRewardParameters):

        self._log = logging.getLogger(__class__.__name__)

        self._params = params
        self._reward_state = None

    def reset(self, state: GliderState):
        self._reward_state = RewardState(
            maximum_altitude=state.position_earth_xyz_m[2])

    def clone(self):
        clone = self.__class__(params=self._params)

        if self._reward_state is not None:
            # clone the state
            clone._reward_state = replace(self._reward_state)

        return clone


    def calculate(self,
                  state: GliderState,
                  info: AdditionalInfo) \
                  -> RewardResult:

        assert self._reward_state is not None, 'reset not called'

        result = RewardResult()

        if self._params.vertical_velocity_reward_enabled:
            # reward vertical velocity
            current_result = self._reward_vertical_velocity(state=state,
                                                            dt_s=info.dt_s)
            result = result.add(current_result)

            self._log.debug(
                'after reward verical velocity; result=%s,reward_state=%s',
                result, self._reward_state)

        if self._params.new_maximum_altitude_reward_enabled:

            # reward new maximum altitude
            current_result, reward_state = \
                self._reward_new_altitude_maximum(state=state,
                                                  reward_state=self._reward_state)
            result = result.add(current_result)

            self._log.debug(
                'after altitude maximum reward; result=%s,reward_state=%s',
                result, reward_state)

            self._reward_state = reward_state

        # success/fail reward
        current_result = self._success_fail_reward(info.cutoff_result)
        result = result.add(current_result)

        self._log.debug('after success/fail reward; result=%s', result)

        # finalize
        if not self._params.negative_reward_enabled and result.reward < 0.0:
            result = result.zero()

        self._log.debug('after finalization; result=%s', result)

        return result

    def _reward_vertical_velocity(self, state: GliderState,
                                  dt_s: float) -> RewardResult:
        reward = state.velocity_earth_xyz_m_per_s[2]
        return RewardResult(reward=reward * dt_s)

    def _reward_new_altitude_maximum(self, state: GliderState, reward_state: RewardState) \
            -> tuple[RewardResult, RewardState]:

        current_max_altitude_m = reward_state.maximum_altitude
        altitude_m = state.position_earth_xyz_m[2]
        if altitude_m > current_max_altitude_m:
            next_reward_state = RewardState(**asdict(reward_state))
            next_reward_state.maximum_altitude = altitude_m
            return RewardResult(
                reward=(altitude_m -
                        current_max_altitude_m)), next_reward_state

        return RewardResult(), reward_state

    def _success_fail_reward(self, cutoff_result: Literal['none', 'success',
                                                          'fail']):
        if cutoff_result == 'success':
            return RewardResult(reward=self._params.success_reward)
        elif cutoff_result == 'fail':
            return RewardResult(reward=self._params.fail_reward)
        else:
            return RewardResult()

    def _finalize_reward(self, result: RewardResult) -> RewardResult:

        if not self._params.negative_reward_enabled and result.reward < 0.0:
            return result.zero()

        return result
