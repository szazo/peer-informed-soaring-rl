from typing import Literal, Self
from dataclasses import dataclass, replace


@dataclass
class CutoffInfo:
    time_s_without_lift: float = 0.


CutoffReason = Literal['none', 'time_without_lift', 'distance_from_core',
                       'simulation_box_leave', 'fail_altitude',
                       'success_altitude', 'duration']


@dataclass
class RewardResult:
    reward: float = 0.

    def add(self, other: Self):
        return replace(self, reward=self.reward + other.reward)

    def zero(self):
        return replace(self, reward=0.)


@dataclass
class CutoffResult:
    terminated: bool = False
    truncated: bool = False
    reason: CutoffReason = 'none'
    result: Literal['none', 'success', 'fail'] = 'none'

    def add(self, other: Self):

        if self.terminated or self.truncated:

            # already terminated or truncated, we do not set anything
            return self

        return other


class RewardAndCutoffResult:

    _reward_result: RewardResult
    _cutoff_result: CutoffResult

    def __init__(self, reward_result: RewardResult,
                 cutoff_result: CutoffResult):
        self._reward_result = reward_result
        self._cutoff_result = cutoff_result

    @property
    def reward(self):
        return self._reward_result.reward

    @property
    def terminated(self):
        return self._cutoff_result.terminated

    @property
    def truncated(self):
        return self._cutoff_result.truncated

    @property
    def reason(self):
        return self._cutoff_result.reason

    @property
    def result(self):
        return self._cutoff_result.result
