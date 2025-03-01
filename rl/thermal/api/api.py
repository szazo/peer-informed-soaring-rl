from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt
from utils import Vector2, Vector3, VectorNx2, VectorNx3


class AirVelocityFieldInterface(ABC):

    @abstractmethod
    def reset(self) -> dict:
        """Resets the thermal using implementation dependent parameters
        """
        pass

    @abstractmethod
    def seed(self, seed: int):
        """Seeds the thermal's random generator
        """
        pass

    @abstractmethod
    def get_velocity(
            self, x_earth_m: float | np.ndarray, y_earth_m: float | np.ndarray,
            z_earth_m: float | np.ndarray,
            t_s: float | np.ndarray) -> tuple[Vector3 | VectorNx3, dict]:
        """Returns the current wind velocity relative to earth at the specified position and time
        """
        pass

    @abstractmethod
    def get_thermal_core(self, z_earth_m: float | npt.NDArray[np.float_],
                         t_s: float | np.ndarray) -> Vector2 | VectorNx2:
        """Returns the thermal core's (x,y) position at the specified z coordinate

        :param z_earth_m: we get thermal core at this altitude
        :param t_s: time of the query
        """
        pass

    @property
    def name(self) -> str:
        return 'unknown'
