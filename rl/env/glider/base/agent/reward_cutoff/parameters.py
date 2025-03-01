from dataclasses import dataclass


@dataclass
class GliderCutoffParameters:
    maximum_distance_from_core_m: float = 700.
    success_altitude_m: float = 1500.
    fail_altitude_m: float = 10.
    maximum_time_without_lift_s: float = 180.
    maximum_duration_s: float = 100.


@dataclass
class GliderRewardParameters:
    success_reward: float = 0.0
    fail_reward: float = 0.0
    negative_reward_enabled: bool = True
    vertical_velocity_reward_enabled: bool = True
    new_maximum_altitude_reward_enabled: bool = True
