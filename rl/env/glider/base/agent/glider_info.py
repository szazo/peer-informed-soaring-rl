from dataclasses import dataclass

from ...aerodynamics.api import AerodynamicsInfo
from utils.vector import VectorNx3
from .reward_cutoff.api import CutoffReason

from utils import Vector2, Vector3, VectorNx3, VectorN


@dataclass
class GliderInfo:
    core_position_earth_xy_m: Vector2
    distance_from_core_m: float
    air_velocity_earth_xyz_m_s: Vector3
    time_s_without_lift: float
    success: bool
    cutoff_reason: CutoffReason
    aerodynamics_info: AerodynamicsInfo


@dataclass
class GliderTrajectory:
    time_s: VectorN
    position_earth_xyz_m: VectorNx3
    velocity_earth_xyz_m_per_s: VectorNx3
    yaw_pitch_roll_earth_to_body_rad: VectorNx3
    air_velocity_earth_xyz_m_per_s: VectorNx3
    indicated_airspeed_m_per_s: VectorN
