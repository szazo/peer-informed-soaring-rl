from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from utils import Vector3D


@dataclass
class AerodynamicsInfo:
    mass_kg: float
    wing_area_m2: float
    CD: float
    CL: float
    rho_kg_per_m3: float
    g_m_per_s2: float


class AerodynamicsInterface(ABC):

    @abstractmethod
    def reset(self) -> AerodynamicsInfo:
        pass

    @abstractmethod
    def get_initial_velocity_earth(
            self, heading_earth_to_body_rad: float,
            wind_velocity_earth_m_per_s: Vector3D
    ) -> tuple[Vector3D, Vector3D]:
        pass

    @abstractmethod
    def step(
        self, position_earth_m: Vector3D, velocity_earth_m_per_s: Vector3D,
        yaw_pitch_roll_earth_to_body_rad: Vector3D,
        wind_velocity_earth_m_per_s: Vector3D, dt_s: float,
        velocity_airmass_relative_m_per_s: Vector3D
    ) -> tuple[Vector3D, Vector3D, Vector3D, np.float_, Vector3D]:
        """Calculate dynamics

        :param position_earth_m: current position (in earth frame)
        :param velocity_earth_m_per_s: current velocity (in earth frame)
        :param yaw_pitch_roll_earth_to_body_rad: current body attitude (in earth frame)
        :param wind_velocity_earth_m_per_s: wind velocity vector (in earth frame)
        :param dt_s: time diff
        :returns: (next_position_earth_m, next_velocity_earth_m_per_s, next_yaw_pitch_roll_earth_to_body, indicated_airspeed_m_per_s)
        """
        pass
