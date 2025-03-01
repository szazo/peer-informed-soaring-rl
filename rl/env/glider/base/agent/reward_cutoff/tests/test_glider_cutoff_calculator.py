from dataclasses import dataclass, field
from env.glider.base.agent.reward_cutoff.api import CutoffInfo, CutoffResult
import numpy as np

from utils import Vector3
from env.glider.base import SimulationBoxParameters
from env.glider.base.agent.reward_cutoff import (GliderCutoffParameters,
                                                 CutoffAdditionalInfo,
                                                 GliderCutoffCalculator)


def empty_vector3() -> Vector3:
    return np.array([0., 0., 0.])


@dataclass
class MockGliderState:
    position_earth_xyz_m: Vector3 = field(default_factory=empty_vector3)
    velocity_earth_xyz_m_per_s: Vector3 = field(default_factory=empty_vector3)


def disabled_params() -> GliderCutoffParameters:
    return GliderCutoffParameters(maximum_distance_from_core_m=10.**4,
                                  success_altitude_m=10.**4,
                                  fail_altitude_m=-10.**4,
                                  maximum_time_without_lift_s=10.**4,
                                  maximum_duration_s=10.**4)


simulation_box = SimulationBoxParameters.create_from_box_size(
    (2000., 2000., 2000.))


def test_should_cutoff_at_success_altitude():

    # given
    params = disabled_params()
    params.fail_altitude_m = 50.
    params.success_altitude_m = 400.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=0.)

    # when
    state = MockGliderState(position_earth_xyz_m=np.array([0., 0., 401.]))
    info = CutoffAdditionalInfo(distance_from_core_m=0.)
    result, cutoff_info = calculator.calculate(state, info, time_s=1.)

    # then
    assert result == CutoffResult(True, False, 'success_altitude', 'success')
    assert cutoff_info == CutoffInfo(time_s_without_lift=1.)


def test_should_cutoff_at_fail_altitude():

    # given
    params = disabled_params()
    params.fail_altitude_m = 50.
    params.success_altitude_m = 400.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=0.)

    # when
    state = MockGliderState(position_earth_xyz_m=np.array([0., 0., 49.]))
    info = CutoffAdditionalInfo(distance_from_core_m=0.)
    result, cutoff_info = calculator.calculate(state, info, time_s=1.)

    # then
    assert result == CutoffResult(True, False, 'fail_altitude', 'fail')
    assert cutoff_info == CutoffInfo(time_s_without_lift=1.)


def test_should_cutoff_when_time_without_lift():

    # given
    params = disabled_params()
    params.maximum_time_without_lift_s = 10.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)

    # when
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([0., 0., 0.]))
    info = CutoffAdditionalInfo(distance_from_core_m=0.)
    result, cutoff_info = calculator.calculate(state, info, time_s=21.)

    # then
    assert result == CutoffResult(False, True, 'time_without_lift', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=11.)


def test_should_not_cutoff_when_boundary():

    # given
    params = disabled_params()
    params.maximum_time_without_lift_s = 10.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)

    # when
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([0., 0., 0.]))
    info = CutoffAdditionalInfo(distance_from_core_m=0.)
    result, cutoff_info = calculator.calculate(state, info, time_s=20.)

    # then
    assert result == CutoffResult(False, False, 'none', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=10.)


def test_should_not_cutoff_when_timer_reset_because_of_lift():

    # given
    params = disabled_params()
    params.maximum_time_without_lift_s = 10.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)
    info = CutoffAdditionalInfo(distance_from_core_m=0.)

    # when
    state = MockGliderState(
        velocity_earth_xyz_m_per_s=np.array([0., 0., 0.001]))
    result, cutoff_info = calculator.calculate(state, info, time_s=20.)

    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([0., 0., 0.0]))
    result, cutoff_info = calculator.calculate(state, info, time_s=30.)

    # then
    assert result == CutoffResult(False, False, 'none', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=10.)


def test_should_cutoff_when_maximum_duration_reached():

    # given
    params = disabled_params()
    params.maximum_duration_s = 20.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)
    info = CutoffAdditionalInfo(distance_from_core_m=0.)

    # when
    state = MockGliderState()
    result, cutoff_info = calculator.calculate(state, info, time_s=31.)

    # then
    assert result == CutoffResult(False, True, 'duration', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=21.)


def test_should_cutoff_when_distance_from_core_reached():

    # given
    params = disabled_params()
    params.maximum_distance_from_core_m = 100.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)

    # when
    state = MockGliderState()
    info = CutoffAdditionalInfo(distance_from_core_m=101.)
    result, cutoff_info = calculator.calculate(state, info, time_s=20.)

    # then
    assert result == CutoffResult(False, True, 'distance_from_core', 'fail')
    assert cutoff_info == CutoffInfo(time_s_without_lift=10.)


def test_should_cutoff_when_simulation_box_leaving():

    # given
    params = disabled_params()
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)

    # when
    state = MockGliderState(position_earth_xyz_m=np.array([-1001., 0., 500.]))
    info = CutoffAdditionalInfo(distance_from_core_m=0.)
    result, cutoff_info = calculator.calculate(state, info, time_s=20.)

    # then
    assert result == CutoffResult(False, True, 'simulation_box_leave', 'fail')
    assert cutoff_info == CutoffInfo(time_s_without_lift=10.)


def test_should_not_cutoff_when_inside_limits():

    # given
    params = disabled_params()
    params.fail_altitude_m = 50.
    params.success_altitude_m = 400.
    params.maximum_time_without_lift_s = 10.
    params.maximum_distance_from_core_m = 400.
    params.maximum_duration_s = 10.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)

    # when
    state = MockGliderState(
        position_earth_xyz_m=np.array([-1000., 1000., 399.99]))
    info = CutoffAdditionalInfo(distance_from_core_m=400.)
    result, cutoff_info = calculator.calculate(state, info, time_s=20.)

    # then
    assert result == CutoffResult(False, False, 'none', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=10.)


def test_should_reset():

    # given
    params = disabled_params()
    params.maximum_time_without_lift_s = 10.
    params.maximum_duration_s = 10.
    calculator = GliderCutoffCalculator(params=params,
                                        simulation_box=simulation_box)
    calculator.reset(time_s=10.)
    info = CutoffAdditionalInfo(distance_from_core_m=0.)
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([0., 0., 0.]))

    result, cutoff_info = calculator.calculate(state, info, time_s=20.)
    assert result == CutoffResult(False, False, 'none', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=10.)

    # when
    calculator.reset(time_s=100.)
    result, cutoff_info = calculator.calculate(state, info, time_s=109.)

    # then
    assert result == CutoffResult(False, False, 'none', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=9.)


def test_clone_should_clone_state():

    # given
    params = disabled_params()
    params.maximum_time_without_lift_s = 10.
    params.maximum_duration_s = 10.
    original = GliderCutoffCalculator(params=params,
                                      simulation_box=simulation_box)
    original.reset(time_s=10.)
    info = CutoffAdditionalInfo(distance_from_core_m=0.)

    # when
    clone = original.clone()

    original_state = MockGliderState(
        velocity_earth_xyz_m_per_s=np.array([0., 0., 1.]))
    result, cutoff_info = original.calculate(original_state, info, time_s=20.)

    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([0., 0., 0.]))
    clone_result, clone_cutoff_info = clone.calculate(state, info, time_s=21.)

    # then
    assert result == CutoffResult(False, False, 'none', 'none')
    assert cutoff_info == CutoffInfo(time_s_without_lift=0.)

    assert clone_result == CutoffResult(False, True, 'time_without_lift',
                                        'none')
    assert clone_cutoff_info == CutoffInfo(time_s_without_lift=11.)
