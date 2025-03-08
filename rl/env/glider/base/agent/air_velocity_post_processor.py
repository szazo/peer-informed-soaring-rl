from typing import Any, cast
import logging
from dataclasses import dataclass
import numpy as np

from utils import Vector3, VectorN, RollingWindowFilter


@dataclass
class GaussianNoiseParameters:
    mean: float
    sigma: float


@dataclass
class AirVelocityGaussianNoiseParameters:
    x: GaussianNoiseParameters | None = None
    y: GaussianNoiseParameters | None = None
    z: GaussianNoiseParameters | None = None


class AirVelocityNoise:

    _log: logging.Logger

    _params: AirVelocityGaussianNoiseParameters
    _np_random: np.random.Generator

    def __init__(self, params: AirVelocityGaussianNoiseParameters,
                 np_random: np.random.Generator):
        self._log = logging.getLogger(__class__.__name__)

        self._params = params
        self._np_random = np_random

    def add_noise(self, air_velocity_earth_xyz_m_s: VectorN):
        before_noise_air_velocity_earth_xyz_m_s = air_velocity_earth_xyz_m_s
        noise_xyz = self._create_air_velocity_noise()
        air_velocity_earth_xyz_m_s = air_velocity_earth_xyz_m_s + noise_xyz
        self._log.debug('noise; before=%s,after=%s,noise=%s',
                        before_noise_air_velocity_earth_xyz_m_s,
                        air_velocity_earth_xyz_m_s, noise_xyz)

        return air_velocity_earth_xyz_m_s

    def _create_air_velocity_noise(self) -> Vector3:

        result = np.array([0., 0., 0.])

        if self._params.x is not None:
            result[0] = self._gaussian_noise(self._params.x)

        if self._params.y is not None:
            result[1] = self._gaussian_noise(self._params.y)

        if self._params.z is not None:
            result[2] = self._gaussian_noise(self._params.z)

        return result

    def _gaussian_noise(self, params: GaussianNoiseParameters) -> float:
        value = self._np_random.normal(params.mean, params.sigma)
        return value


class AirVelocityFilter:

    _kernel: VectorN
    _air_velocity_filters: tuple[RollingWindowFilter, RollingWindowFilter,
                                 RollingWindowFilter]

    def __init__(self, kernel: VectorN):

        self._kernel = kernel

        # 3 because of the three dimension
        filters = cast(
            tuple[RollingWindowFilter, RollingWindowFilter,
                  RollingWindowFilter],
            tuple([RollingWindowFilter(kernel) for _ in range(3)]))

        self._air_velocity_filters = filters

    def reset(self):
        for filter in self._air_velocity_filters:
            filter.reset()

    def filter(self, air_velocity_earth_xyz_m_s: Vector3):
        # feed the filters and return the filtered
        filtered_air_velocity_earth_xyz_m_s = np.zeros((3))
        for i, filter in enumerate(self._air_velocity_filters):
            filtered_air_velocity_earth_xyz_m_s[i] = filter.feed(
                air_velocity_earth_xyz_m_s[i])

        return filtered_air_velocity_earth_xyz_m_s

    def state_clone(self):
        clone = self.__class__(self._kernel)

        for i in range(3):
            clone._air_velocity_filters[i].clone_state_from(
                self._air_velocity_filters[i])

        return clone


@dataclass
class AirVelocityPostProcessorParams:
    filter_kernel: Any | None = None
    velocity_noise: AirVelocityGaussianNoiseParameters | None = None


class AirVelocityPostProcessor:

    _params: AirVelocityPostProcessorParams
    _np_random: np.random.Generator

    _noise: AirVelocityNoise | None
    _filter: AirVelocityFilter | None

    def __init__(self, params: AirVelocityPostProcessorParams,
                 np_random: np.random.Generator):
        self._params = params
        self._np_random = np_random

        self._noise = None
        if params.velocity_noise is not None:
            self._noise = AirVelocityNoise(params=params.velocity_noise,
                                           np_random=np_random)

        self._filter = None
        if params.filter_kernel is not None:
            self._filter = AirVelocityFilter(kernel=params.filter_kernel)

    def clone(self):
        clone = self.__class__(params=self._params, np_random=self._np_random)

        if self._filter is not None:
            # clone the filter's state
            clone._filter = self._filter.state_clone()

        return clone

    def reset(self):
        if self._filter is not None:
            self._filter.reset()

    def process(self, air_velocity_earth_xyz_m_s: Vector3):

        result = air_velocity_earth_xyz_m_s
        if self._noise is not None:
            result = self._noise.add_noise(result)

        if self._filter is not None:
            result = self._filter.filter(result)

        return result
