from dataclasses import dataclass

from utils import Vector3


@dataclass
class GliderState:
    position_earth_xyz_m: Vector3
    velocity_earth_xyz_m_per_s: Vector3
    yaw_pitch_roll_earth_to_body_rad: Vector3
    yaw_pitch_roll_earth_to_body_dot_rad_per_s: Vector3
    velocity_airmass_relative_xyz_m_per_s: Vector3
