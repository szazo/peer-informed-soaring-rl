from dataclasses import dataclass, field
from env.glider.base.agent.reward_cutoff.api import RewardResult
import numpy as np

from utils import Vector3
from env.glider.base.agent.reward_cutoff import GliderRewardCalculator, GliderRewardParameters, RewardAdditionalInfo


def empty_vector3() -> Vector3:
    return np.array([0., 0., 0.])


@dataclass
class MockGliderState:
    position_earth_xyz_m: Vector3 = field(default_factory=empty_vector3)
    velocity_earth_xyz_m_per_s: Vector3 = field(default_factory=empty_vector3)


def disabled_params() -> GliderRewardParameters:
    return GliderRewardParameters(success_reward=False,
                                  fail_reward=False,
                                  negative_reward_enabled=True,
                                  vertical_velocity_reward_enabled=False,
                                  new_maximum_altitude_reward_enabled=False)


def test_calculate_should_calculate_current_vertical_velocity_reward_based_on_dt_s(
):

    # given
    params = disabled_params()
    params.vertical_velocity_reward_enabled = True
    calculator = GliderRewardCalculator(params)

    state = MockGliderState(position_earth_xyz_m=np.array([0., 0., 100.]))
    calculator.reset(state)
    info = RewardAdditionalInfo(dt_s=0.19)

    # when
    state = MockGliderState(
        position_earth_xyz_m=np.array([0., 0., 105.]),
        velocity_earth_xyz_m_per_s=np.array([1., 2., 10.]),
    )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=1.9)

    # when
    state = MockGliderState(
        position_earth_xyz_m=np.array([0., 0., 105.]),
        velocity_earth_xyz_m_per_s=np.array([1., 2., 20.]),
    )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=3.8)


def test_calculate_should_add_altitude_diff_when_new_altitude_maximum():

    # given
    params = disabled_params()
    params.new_maximum_altitude_reward_enabled = True
    calculator = GliderRewardCalculator(params)

    state = MockGliderState(position_earth_xyz_m=np.array([0., 0., 100.]))
    calculator.reset(state)
    info = RewardAdditionalInfo(dt_s=0.19)

    # when
    state = MockGliderState(
        position_earth_xyz_m=np.array([0., 0., 105.]),
        velocity_earth_xyz_m_per_s=np.array([1., 2., 10.]),
    )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=5.0)

    # when (altitude decreased)
    state = MockGliderState(
        position_earth_xyz_m=np.array([0., 0., 104.]),
        velocity_earth_xyz_m_per_s=np.array([1., 2., 10.]),
    )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=0.0)


def test_calculate_should_reset_max_altitude_when_reset():

    # given
    params = disabled_params()
    params.new_maximum_altitude_reward_enabled = True
    calculator = GliderRewardCalculator(params)

    reset_state = MockGliderState(
        position_earth_xyz_m=np.array([0., 0., 100.]))
    calculator.reset(reset_state)
    info = RewardAdditionalInfo(dt_s=0.19)

    state = MockGliderState(position_earth_xyz_m=np.array([0., 0., 105.]), )

    result = calculator.calculate(state=state, info=info)

    # when
    calculator.reset(reset_state)

    state = MockGliderState(position_earth_xyz_m=np.array([0., 0., 105.]), )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=5.0)


def test_calculate_should_return_zero_when_negative_reward_and_negative_disabled(
):

    # given
    params = disabled_params()
    params.vertical_velocity_reward_enabled = True
    params.negative_reward_enabled = False
    calculator = GliderRewardCalculator(params)

    state = MockGliderState()
    calculator.reset(state)
    info = RewardAdditionalInfo(dt_s=0.1)

    # when
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([1., 2.,
                                                                 -10.]), )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=0.0)


def test_calculate_should_return_negative_when_negative_reward_and_negative_enabled(
):

    # given
    params = disabled_params()
    params.vertical_velocity_reward_enabled = True
    params.negative_reward_enabled = True
    calculator = GliderRewardCalculator(params)

    state = MockGliderState()
    calculator.reset(state)
    info = RewardAdditionalInfo(dt_s=0.1)

    # when
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([1., 2.,
                                                                 -10.]), )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=-1.0)


def test_calculate_should_add_success_reward_when_enabled_and_success():

    # given
    params = disabled_params()
    params.vertical_velocity_reward_enabled = True
    params.negative_reward_enabled = True
    params.success_reward = 42.
    calculator = GliderRewardCalculator(params)

    state = MockGliderState()
    calculator.reset(state)
    info = RewardAdditionalInfo(dt_s=0.1, cutoff_result='success')

    # when
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([1., 2.,
                                                                 -10.]), )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=41.0)


def test_calculate_should_add_fail_reward_when_enabled_and_fail():

    # given
    params = disabled_params()
    params.vertical_velocity_reward_enabled = True
    params.negative_reward_enabled = True
    params.success_reward = 42.
    params.fail_reward = -500.
    calculator = GliderRewardCalculator(params)

    state = MockGliderState()
    calculator.reset(state)
    info = RewardAdditionalInfo(dt_s=0.1, cutoff_result='fail')

    # when
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([1., 2.,
                                                                 -10.]), )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=-501.)


def test_calculate_should_not_add_cutoff_reward_when_enabled_and_none():

    # given
    params = disabled_params()
    params.vertical_velocity_reward_enabled = True
    params.negative_reward_enabled = True
    params.success_reward = 42.
    params.fail_reward = -500.
    calculator = GliderRewardCalculator(params)

    state = MockGliderState()
    calculator.reset(state)
    info = RewardAdditionalInfo(dt_s=0.1, cutoff_result='none')

    # when
    state = MockGliderState(velocity_earth_xyz_m_per_s=np.array([1., 2.,
                                                                 -10.]), )

    result = calculator.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=-1.)


def test_clone_should_clone_state():

    # given
    params = disabled_params()
    params.new_maximum_altitude_reward_enabled = True
    original = GliderRewardCalculator(params)

    reset_state = MockGliderState(position_earth_xyz_m=np.array([0., 0.,
                                                                 100.]), )
    original.reset(reset_state)
    info = RewardAdditionalInfo(dt_s=0.19)

    # when
    clone = original.clone()

    state = MockGliderState(position_earth_xyz_m=np.array([0., 0., 105.]), )
    result = original.calculate(state=state, info=info)

    clone_result = clone.calculate(state=state, info=info)

    # then
    assert result == RewardResult(reward=5.0)
    assert clone_result == RewardResult(reward=5.0)
