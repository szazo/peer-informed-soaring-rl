import logging
from typing import Any
from dataclasses import dataclass
import numpy as np
import scipy
from typing import Union
from ..api import AirVelocityFieldInterface
from .turbulent_gaussian_thermal import TurbulentGaussianThermal
from .horizontal_wind import HorizontalWind
from gymnasium.utils import seeding
from utils import serialize_random_state, deserialize_random_state, RandomGeneratorState
from .simple_air_velocity_field_params import (GaussianThermalParameters,
                                               HorizontalWindParameters)
from .noise_field_generator import NoiseFieldGenerator
from .noise_field import NoiseField


@dataclass
class TurbulencePoolEntry:
    noise_field: NoiseField
    initial_conditions: dict[str, Any]


class SingleStochasticTurbulencePool:

    _noise_field_generator: NoiseFieldGenerator
    _regenerate_probability: float

    _entry: TurbulencePoolEntry | None
    _np_random: np.random.Generator | None

    def __init__(self, noise_field_generator: NoiseFieldGenerator,
                 regenerate_probability: float):

        self._noise_field_generator = noise_field_generator
        self._regenerate_probability = regenerate_probability

        self._entry = None
        self._np_random = None

    def initialize_random_generator(self, generator: np.random.Generator):
        self._noise_field_generator.initialize_random_generator(generator)
        self._np_random = generator

    def get(self) -> tuple[NoiseField, dict[str, Any]]:

        assert self._np_random is not None, 'random generator not initialized'

        should_generate = self._entry is None or self._np_random.uniform(
            0., 1.) < self._regenerate_probability

        if should_generate:
            noise_field, initial_conditions = self._noise_field_generator.generate(
            )
            self._entry = TurbulencePoolEntry(
                noise_field=noise_field, initial_conditions=initial_conditions)

        assert self._entry is not None, 'should have an entry here'
        return self._entry.noise_field, self._entry.initial_conditions


class SimpleAirVelocityField(AirVelocityFieldInterface):

    _initial_random_state: RandomGeneratorState | None

    _random_state: RandomGeneratorState

    _turbulence_pool: SingleStochasticTurbulencePool | None
    _is_turbulence_noise_time_dependent: bool | None
    _is_horizontal_wind_noise_time_dependent: bool | None

    def __init__(
            self,
            thermal_params: GaussianThermalParameters,
            horizontal_wind_params: HorizontalWindParameters | None,
            horizontal_wind_noise_generator: NoiseFieldGenerator | None,
            is_horizontal_wind_noise_time_dependent: bool | None,
            turbulence_noise_generator: NoiseFieldGenerator | None,
            is_turbulence_noise_time_dependent: bool | None,
            turbulence_episode_regenerate_probability: float,
            max_altitude_m: float,
            # horizontal_wind_profile_vertical_resolution: int = 10,
            seed: int | None = None,
            random_state: RandomGeneratorState | None = None,
            name: str | None = None):

        self._log = logging.getLogger(__class__.__name__)

        self._max_r_m_normal_mean = thermal_params.max_r_m_normal_mean / thermal_params.radius_k
        self._max_r_m_normal_sigma = thermal_params.max_r_m_normal_sigma / thermal_params.sigma_k
        self._max_r_m_sigma_normal_mean = thermal_params.max_r_m_sigma_normal_mean
        self._max_r_m_sigma_normal_sigma = thermal_params.max_r_m_sigma_normal_sigma / thermal_params.sigma_k
        self._max_r_altitude_m_normal_mean = thermal_params.max_r_altitude_m_normal_mean
        self._max_r_altitude_m_normal_sigma = thermal_params.max_r_altitude_m_normal_sigma / thermal_params.sigma_k
        self._w_max_m_per_s_normal_mean = thermal_params.w_max_m_per_s_normal_mean
        self._w_max_m_per_s_normal_sigma = thermal_params.w_max_m_per_s_normal_sigma / thermal_params.sigma_k
        self._max_altitude_m = max_altitude_m

        self._turbulence_pool = SingleStochasticTurbulencePool(
            turbulence_noise_generator,
            regenerate_probability=turbulence_episode_regenerate_probability
        ) if turbulence_noise_generator is not None else None
        self._is_turbulence_noise_time_dependent = is_turbulence_noise_time_dependent

        # wind parameters
        if horizontal_wind_params is not None:
            self._horizontal_wind_speed_at_2m_m_per_s_normal_mean = horizontal_wind_params.horizontal_wind_speed_at_2m_m_per_s_normal_mean
            self._horizontal_wind_speed_at_2m_m_per_s_normal_sigma = horizontal_wind_params.horizontal_wind_speed_at_2m_m_per_s_normal_sigma / horizontal_wind_params.sigma_k
            self._horizontal_wind_noise_generator = horizontal_wind_noise_generator
            self._is_horizontal_wind_noise_time_dependent = is_horizontal_wind_noise_time_dependent
            self._horizontal_wind_profile_vertical_spacing_m = horizontal_wind_params.horizontal_wind_profile_vertical_spacing_m
        else:
            self._horizontal_wind_speed_at_2m_m_per_s_normal_mean = 0
            self._horizontal_wind_speed_at_2m_m_per_s_normal_sigma = 0

        self._name = name

        self.seed(seed)

        self._initial_random_state = None
        if random_state is not None:
            self._initial_random_state = random_state
            self._restore_random_state(random_state)

    def _restore_random_state(self, random_state: RandomGeneratorState):
        self._log.debug('restoring random state %s', random_state)
        self._random_state = random_state

        generator = deserialize_random_state(random_state)

        self._initialize_random_generator(generator)

        self._log.debug('random restored')

    def seed(self, seed: Union[int, None]):
        np_random, seed = seeding.np_random(seed)
        self._log.debug("random generator initialized with seed %s", seed)

        self._initialize_random_generator(np_random)

    def _initialize_random_generator(self, generator: np.random.Generator):
        self._np_random = generator
        if self._turbulence_pool is not None:
            self._turbulence_pool.initialize_random_generator(generator)

        if self._horizontal_wind_noise_generator is not None:
            self._horizontal_wind_noise_generator.initialize_random_generator(
                generator)

    @property
    def name(self) -> str:
        if self._name is not None:
            return self._name

        if self._random_state is not None:
            return f'{self._random_state.generator_name}:{self._random_state.encoded_state_values}'

        return 'unknown'

    def reset(self):

        if self._initial_random_state is not None:
            self._restore_random_state(self._initial_random_state)

        self._random_state = serialize_random_state(self._np_random)

        # the maximum radius of the thermal
        max_r_m = self._np_random.normal(self._max_r_m_normal_mean,
                                         self._max_r_m_normal_sigma)

        # vertical spread of the maximum radius
        max_r_m_sigma = self._np_random.normal(
            self._max_r_m_sigma_normal_mean, self._max_r_m_sigma_normal_sigma)

        # the altitude where the thermal has the maximum radius
        max_r_altitude_m = self._np_random.normal(
            self._max_r_altitude_m_normal_mean,
            self._max_r_altitude_m_normal_sigma)

        # the maximum vertical velocity
        w_max_m_per_s = self._np_random.normal(
            self._w_max_m_per_s_normal_mean, self._w_max_m_per_s_normal_sigma)
        self._w_max_m_per_s = w_max_m_per_s

        # wind speed at 2m
        horizontal_wind_speed_m_per_s = self._np_random.normal(
            self._horizontal_wind_speed_at_2m_m_per_s_normal_mean,
            self._horizontal_wind_speed_at_2m_m_per_s_normal_sigma)

        # horizontal wind direction
        horizontal_wind_direction_rad = self._np_random.uniform(low=0,
                                                                high=2 * np.pi)

        # create horizontal wind
        horizontal_wind_speed_reference_altitude_m = 2.
        horizontal_wind_noise_field = None
        horizontal_wind_noise_field_initial_conditions = None
        if self._horizontal_wind_noise_generator is not None:
            horizontal_wind_noise_field, horizontal_wind_noise_field_initial_conditions = self._horizontal_wind_noise_generator.generate(
            )
        self._horizontal_wind = HorizontalWind(
            direction_rad=horizontal_wind_direction_rad,
            reference_altitude_m=horizontal_wind_speed_reference_altitude_m,
            wind_speed_at_reference_altitude_m_per_s=
            horizontal_wind_speed_m_per_s,
            noise_field=horizontal_wind_noise_field,
            is_time_dependent_noise=self.
            _is_horizontal_wind_noise_time_dependent)

        # center of the thermal
        x0_m = 0.0
        y0_m = 0.0

        self._log.debug(
            "thermal initial conditions: max_r_m=%s,max_r_altitude_m=%s,max_r_m_sigma=%s,w_max_m_per_s=%s,x0_m=%s,y0_m=%s,horizontal_wind_speed_m_per_s=%s,horizontal_wind_direction_rad=%s",
            max_r_m, max_r_altitude_m, max_r_m_sigma, w_max_m_per_s, x0_m,
            y0_m, horizontal_wind_speed_m_per_s, horizontal_wind_direction_rad)

        noise_field = None
        turbulence_noise_field_initial_conditions = None
        if self._turbulence_pool is not None:
            noise_field, turbulence_noise_field_initial_conditions = self._turbulence_pool.get(
            )

        self._thermal = TurbulentGaussianThermal(
            x0=x0_m,
            y0=y0_m,
            max_r=max_r_m,
            max_r_altitude=max_r_altitude_m,
            max_r_sigma=max_r_m_sigma,
            w_max=w_max_m_per_s,
            noise_field=noise_field,
            is_time_dependent_noise=self._is_turbulence_noise_time_dependent)

        # else:
        #     self._thermal = GaussianThermal(
        #         x0_m,
        #         y0_m,
        #         max_r_m,
        #         max_r_altitude_m,
        #         max_r_m_sigma,
        #         w_max_m_per_s,
        #         horizontal_circulation_w_factor,
        #     )

        initial_conditions = dict(
            max_r_m=max_r_m,
            max_r_altitude_m=max_r_altitude_m,
            max_r_m_sigma=max_r_m_sigma,
            w_max_m_per_s=w_max_m_per_s,
            x0_m=x0_m,
            y0_m=y0_m,
            turbulence_noise_field=turbulence_noise_field_initial_conditions,
            horizontal_wind_noise_field=
            horizontal_wind_noise_field_initial_conditions,
            horizontal_wind_speed_reference_altitude_m=
            horizontal_wind_speed_reference_altitude_m,
            horizontal_wind_speed_m_per_s=horizontal_wind_speed_m_per_s,
            horizontal_wind_direction_rad=horizontal_wind_direction_rad,
            horizontal_wind_noise_field_sum=(
                horizontal_wind_noise_field._noise.sum()
                if horizontal_wind_noise_field is not None else 0),
            thermal_noise_field_sum=(noise_field._noise.sum()
                                     if noise_field is not None else 0),
            rng_name=(self._random_state.generator_name
                      if self._random_state is not None else None),
            rng_state=(self._random_state.encoded_state_values
                       if self._random_state is not None else None))

        return initial_conditions

        # self._visualizer = SimpleAirVelocityFieldVisualization(self._thermal)

    # @property
    # def visualizer(self):
    #     return self._visualizer

    # def _create_thermal(self):

    #     # center of the thermal
    #     x0_m = 0
    #     y0_m = 0

    #     # parameters of the thermal
    #     max_r_m = 120
    #     max_r_altitude_m = 1200
    #     max_r_sigma = 900

    #     w_max_ms = 3.0
    #     horizontal_circulation_w_factor = 0.1

    #     thermal = GaussianThermal(
    #         x0_m,
    #         y0_m,
    #         max_r_m,
    #         max_r_altitude_m,
    #         max_r_sigma,
    #         w_max_ms,
    #         horizontal_circulation_w_factor,
    #     )

    #     return thermal

    @property
    def thermal(self):
        return self._thermal

    def get_velocity(self, x_earth_m: Union[float, np.ndarray],
                     y_earth_m: Union[float, np.ndarray],
                     z_earth_m: Union[float,
                                      np.ndarray], t_s: Union[float,
                                                              np.ndarray]):

        self._log.debug('get_velocity; x=%s,y=%s,z=%s,t_s=%s', x_earth_m,
                        y_earth_m, z_earth_m, t_s)

        #         # TODO: use time
        #         t_s = 0.

        # #        print('Z', z_earth_m)
        #         # generate core horizontal displacement profile
        #         z_tmp = np.linspace(0., self._max_altitude_m, 10)

        #         horizontal_displacement_m = self._horizontal_wind.integrate_horizontal_displacement(
        #             time_s=t_s, altitude_m=z_tmp, vertical_velocity_m_per_s=self._w_max_m_per_s)

        #print('DISPLACEMENT', horizontal_displacement_m)
        #       print('z_tmp', z_tmp.shape)
        #       print('horizontal_displacement_m', horizontal_displacement_m.shape)
        # print('z_earth_m', z_earth_m)
        # displacements_xy_m = scipy.interpolate.interpn(
        #     points = [z_tmp], values=horizontal_displacement_m, xi=np.array([z_earth_m]) if np.isscalar(z_earth_m) else z_earth_m[..., np.newaxis]
        # )
        #print('DISPLACEMENT', displacements_xy_m.shape, displacements_xy_m[0])

        thermal_core = self.get_thermal_core(z_earth_m=z_earth_m, t_s=t_s)

        x0 = thermal_core[..., 0]
        y0 = thermal_core[..., 1]

        # print('CORE', x0, y0)
        #print('x0.shape' ,x0.shape)
        #print('x0' ,x0[0])
        #print('x0' ,x0[50])

        # print('z_earth_m', z_earth_m)
        u, v, w = self._thermal.get_velocity(time_s=t_s,
                                             x=x_earth_m,
                                             y=y_earth_m,
                                             z=z_earth_m,
                                             x0=x0,
                                             y0=y0)

        # print('U before wind', u.shape if not np.isscalar(u) else u, np.max(u))
        # print('V before wind', v.shape if not np.isscalar(v) else v, np.max(v))
        # print('Z', z_earth_m.shape if not np.isscalar(z_earth_m) else z_earth_m, np.min(z_earth_m), np.max(z_earth_m))
        # TODO: zero now
        wind_time_s = 0.
        wind_velocity_vector = self._horizontal_wind.get_wind(
            time_s=wind_time_s, altitude_m=z_earth_m)
        # print('WIND', wind_velocity_vector.shape)

        u_wind = wind_velocity_vector[..., 0]
        v_wind = wind_velocity_vector[..., 1]

        # print('U wind', u_wind.shape)
        # print('V wind', v_wind.shape)
        # print('U wind', u_wind)
        # print('V wind', v_wind)

        # add horizontal wind

        # print('U before wind', u.shape if not np.isscalar(u) else u, np.max(u))
        # print('V before wind', v.shape if not np.isscalar(v) else v, np.max(v))
        u += u_wind
        v += v_wind
        # print('U AFTER wind', u.shape if not np.isscalar(u) else u, np.max(u))
        # print('V AFTER wind', v.shape if not np.isscalar(v) else v, np.max(v))

        # u = u if u.size > 1 else u.item()
        # v = v if v.size > 1 else v.item()
        # w = w if w.size > 1 else w.item()

        #        print('u v w', u.shape, v.shape, w.shape)
        result = np.array([u, v, w]), {}
        self._log.debug('get_velocity; result=%s', result)

        return result

    def get_thermal_core(self, z_earth_m: Union[float, np.ndarray],
                         t_s: Union[float, np.ndarray]):

        if self._horizontal_wind_speed_at_2m_m_per_s_normal_mean == 0 and self._horizontal_wind_speed_at_2m_m_per_s_normal_sigma == 0:
            # there is no wind, return zero displacement
            return np.zeros((1 if np.isscalar(z_earth_m) else len(z_earth_m),
                             2)).squeeze()

        # TODO: use time
        t_s = 0.

        # generate core horizontal displacement profile
        resolution = int(self._max_altitude_m /
                         self._horizontal_wind_profile_vertical_spacing_m + 1)
        z_space = np.linspace(0., self._max_altitude_m, resolution)

        horizontal_displacement_m = self._horizontal_wind.integrate_horizontal_displacement(
            time_s=t_s,
            altitude_m=z_space,
            vertical_velocity_m_per_s=self._w_max_m_per_s)

        displacements_xy_m = scipy.interpolate.interpn(
            points=[z_space],
            values=horizontal_displacement_m,
            xi=np.array([z_earth_m])
            if np.isscalar(z_earth_m) else z_earth_m[..., np.newaxis])

        result = np.squeeze(displacements_xy_m) if np.isscalar(
            z_earth_m) else displacements_xy_m

        if np.isscalar(z_earth_m):
            assert result.shape == (
                2, ), f'result shape {result.shape} is invalid'
        else:
            assert isinstance(z_earth_m, np.ndarray)
            assert result.shape == (
                *z_earth_m.shape, 2), f'result shape {result.shape} is invalid'

        return result
