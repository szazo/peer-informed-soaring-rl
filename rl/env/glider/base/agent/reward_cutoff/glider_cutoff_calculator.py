from typing import Protocol
from dataclasses import dataclass, replace
import logging
import numpy as np

from utils import Vector3
from .api import CutoffResult, CutoffInfo
from ...simulation_box_params import SimulationBoxParameters
from .parameters import GliderCutoffParameters


@dataclass
class CutoffState:
    last_time_with_lift: float
    initial_time_s: float


class GliderState(Protocol):
    position_earth_xyz_m: Vector3
    velocity_earth_xyz_m_per_s: Vector3


class AdditionalInfo(Protocol):
    distance_from_core_m: float


@dataclass
class CutoffAdditionalInfo(AdditionalInfo):
    distance_from_core_m: float


class GliderCutoffCalculator:

    _params: GliderCutoffParameters
    _cutoff_state: CutoffState | None

    def __init__(self, params: GliderCutoffParameters,
                 simulation_box: SimulationBoxParameters):

        self._log = logging.getLogger(__class__.__name__)

        self._params = params
        self._simulation_box = simulation_box

        self._cutoff_state = None

    def reset(self, time_s: float):
        self._cutoff_state = CutoffState(last_time_with_lift=time_s,
                                         initial_time_s=time_s)

    def clone(self):
        clone = self.__class__(params=self._params,
                               simulation_box=self._simulation_box)

        if self._cutoff_state is not None:
            # clone the state
            clone._cutoff_state = replace(self._cutoff_state)

        return clone

    def calculate(self,
                  glider_state: GliderState,
                  info: AdditionalInfo,
                  time_s: float) \
                  -> tuple[CutoffResult, CutoffInfo]:

        assert self._cutoff_state is not None, 'reset has not been called, please call'

        result: CutoffResult = CutoffResult()
        result_info: CutoffInfo = CutoffInfo()

        next_state = self._cutoff_state

        # altitude cutoff
        current_result = self._check_altitude_cutoff(glider_state)
        result = result.add(current_result)

        self._log.debug('after altitude check; result=%s,state=%s', result,
                        next_state)

        # calculate time without lift
        current_result, next_state, result_info = self._check_maximum_time_without_lift(
            glider_state, next_state, result_info, time_s)
        result = result.add(current_result)

        self._log.debug('after lift timeout check; result=%s,state=%s', result,
                        next_state)

        # duration
        current_result = self._check_duration(time_s, next_state)
        result = result.add(current_result)

        self._log.debug('after duration check; result=%s,state=%s', result,
                        next_state)

        # check distance from core
        current_result = self._check_maximum_distance_from_core(info=info)
        result = result.add(current_result)

        self._log.debug('after maximum distance check; result=%s,state=%s',
                        result, next_state)

        # simulation box leaving check
        current_result = self._check_simulation_box_leaving(state=glider_state)
        result = result.add(current_result)

        self._log.debug('after simulation box check; result=%s,state=%s',
                        result, next_state)

        self._cutoff_state = next_state

        return result, result_info

    def _check_altitude_cutoff(
        self,
        state: GliderState,
    ) -> CutoffResult:

        altitude_m = state.position_earth_xyz_m[2]
        if altitude_m <= self._params.fail_altitude_m:
            return CutoffResult(terminated=True,
                                result='fail',
                                reason='fail_altitude')
        elif altitude_m >= self._params.success_altitude_m:
            return CutoffResult(terminated=True,
                                result='success',
                                reason='success_altitude')

        return CutoffResult()

    def _check_duration(self, time_s: float,
                        cutoff_state: CutoffState) -> CutoffResult:
        if (time_s -
                cutoff_state.initial_time_s) > self._params.maximum_duration_s:
            return CutoffResult(truncated=True, reason='duration')

        return CutoffResult()

    def _check_maximum_time_without_lift(
            self, state: GliderState, cutoff_state: CutoffState,
            info: CutoffInfo,
            time_s: float) -> tuple[CutoffResult, CutoffState, CutoffInfo]:

        next_cutoff_state = replace(cutoff_state)
        vertical_velocity_m_per_s = state.velocity_earth_xyz_m_per_s[2]
        has_lift = vertical_velocity_m_per_s > 0
        if has_lift:
            next_cutoff_state.last_time_with_lift = time_s

        time_s_without_lift = (time_s - next_cutoff_state.last_time_with_lift)
        self._log.debug('time without lift: %s', time_s_without_lift)

        is_maximum_time_without_lift_reached = (
            time_s_without_lift > self._params.maximum_time_without_lift_s)

        self._log.debug('is_maximum_time_without_lift_reached=%s',
                        is_maximum_time_without_lift_reached)

        info = replace(info, time_s_without_lift=time_s_without_lift)

        return CutoffResult(
            truncated=is_maximum_time_without_lift_reached,
            reason='time_without_lift'), next_cutoff_state, info

    def _check_maximum_distance_from_core(
            self, info: AdditionalInfo) -> CutoffResult:

        if info.distance_from_core_m > self._params.maximum_distance_from_core_m:
            return CutoffResult(truncated=True,
                                result='fail',
                                reason='distance_from_core')

        return CutoffResult()

    def _check_simulation_box_leaving(self,
                                      state: GliderState) -> CutoffResult:

        box = self._simulation_box
        assert box.limit_earth_xyz_low_m is not None
        assert box.limit_earth_xyz_high_m is not None

        if np.any(state.position_earth_xyz_m < np.array(box.limit_earth_xyz_low_m)) or \
           np.any(state.position_earth_xyz_m > np.array(box.limit_earth_xyz_high_m)):
            return CutoffResult(truncated=True,
                                result='fail',
                                reason='simulation_box_leave')

        return CutoffResult()
