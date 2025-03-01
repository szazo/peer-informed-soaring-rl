from dataclasses import dataclass
from omegaconf import MISSING
from deprecated import deprecated
import numpy as np


# unsafe_hash is needed for @functools.cache
@dataclass(unsafe_hash=True)
class SimulationBoxParameters:

    box_size: tuple[float, float, float] | None = MISSING
    limit_earth_xyz_low_m: tuple[float, float, float] | None = None
    limit_earth_xyz_high_m: tuple[float, float, float] | None = None

    _post_init_executed: bool = False

    @staticmethod
    @deprecated()
    def create_from_box_size(box_size: tuple[float, float, float]):
        simulation_box_params = SimulationBoxParameters(box_size=box_size)
        return simulation_box_params

    def _calculate_limits(self, box_size: tuple[float, float, float]):
        half_x = box_size[0] / 2
        half_y = box_size[1] / 2

        limit_earth_xyz_low_m = (-half_x, -half_y, 0.0)
        limit_earth_xyz_high_m = (half_x, half_y, box_size[2])

        return limit_earth_xyz_low_m, limit_earth_xyz_high_m

    # configure using box size or limits
    def __post_init__(self):

        if self.box_size == MISSING:
            return

        if not self._post_init_executed:
            # it can be executed more than once, so we check only once

            if self.box_size is not None:

                assert self.limit_earth_xyz_low_m is None, "cannot set both 'box_size' and 'limit_earth_xyz_low_m'"
                assert self.limit_earth_xyz_high_m is None, "cannot set both 'box_size' and 'limit_earth_xyz_high_m'"

                self.limit_earth_xyz_low_m, self.limit_earth_xyz_high_m = self._calculate_limits(
                    self.box_size)

            else:
                assert self.limit_earth_xyz_low_m is not None, "either 'box_size' or 'limit_earth_xyz_low_m' must be set"
                assert self.limit_earth_xyz_high_m is not None, "either 'box_size' or 'limit_earth_xyz_high_m' must be set"

                self.box_size = tuple(
                    np.abs(
                        np.array(self.limit_earth_xyz_high_m) -
                        np.array(self.limit_earth_xyz_low_m)))

        # workaround because hydra does not restore tuples now
        # https://github.com/omry/omegaconf/issues/392
        self.box_size = tuple(self.box_size)
        self.limit_earth_xyz_low_m = tuple(self.limit_earth_xyz_low_m)
        self.limit_earth_xyz_high_m = tuple(self.limit_earth_xyz_high_m)

        self._post_init_executed = True
