import numpy as np
from ..api import AirVelocityFieldInterface
from utils import Vector2, Vector3, VectorNx3, VectorNx2, VectorN


class ZeroAirVelocityField(AirVelocityFieldInterface):

    def reset(self):
        return {}

    def seed(self, seed: int):
        pass

    def get_velocity(
            self, x_earth_m: float | np.ndarray, y_earth_m: float | np.ndarray,
            z_earth_m: float | np.ndarray,
            t_s: float | np.ndarray) -> tuple[Vector3 | VectorNx3, dict]:

        return np.array([0., 0., 0.]), {}

    def get_thermal_core(self, z_earth_m: float | VectorN,
                         t_s: float | np.ndarray) -> Vector2 | VectorNx2:

        return np.array([0., 0.])

    @property
    def name(self) -> str:
        return 'zero'
