import logging
import numpy as np
import dacite
from .simple_air_velocity_field_params import GaussianAirVelocityFieldParameters, NoiseParameters
from .simple_air_velocity_field import SimpleAirVelocityField
from .noise_field_generator import NoiseFieldGenerator


def _make_noise_field_generator(
        noise_count: int,
        noise_field_grid_range_low: np.ndarray | tuple[float],
        noise_field_grid_range_high: np.ndarray | tuple[float],
        params: NoiseParameters) -> tuple[NoiseFieldGenerator, bool]:

    log = logging.getLogger(__name__)

    # first dimension will be the time
    time_size = params.time_high_s - params.time_low_s
    time_resolution = int((time_size) / params.time_space_s + 1)
    log.debug('_make_noise_field_generator')
    log.debug('time_size=%s, time_spacing=%s, time_resolution=%s', time_size,
              params.time_space_s, time_resolution)

    # space
    space_size = np.abs(
        np.array(noise_field_grid_range_high) -
        np.array(noise_field_grid_range_low))
    space_resolution = (space_size / params.noise_grid_spacing_m +
                        1).astype(int)

    log.debug('space_size=%s, space_spacing=%s, space_resolution=%s',
              space_size, params.noise_grid_spacing_m, space_resolution)

    # merge time+space
    is_time_dependent_noise = time_resolution > 1
    resolution = space_resolution
    range_low = noise_field_grid_range_low
    range_high = noise_field_grid_range_high

    if is_time_dependent_noise:
        # we have time, add the time dimension
        resolution = (time_resolution, *space_resolution)
        range_low = (params.time_low_s, *noise_field_grid_range_low)
        range_high = (params.time_high_s, *noise_field_grid_range_high)

    log.debug('noise_count=%s,resolution=%s', noise_count, resolution)

    filter_sigma_normal_mean = params.noise_gaussian_filter_sigma_normal_mean_m / params.noise_grid_spacing_m
    filter_sigma_normal_sigma = params.noise_gaussian_filter_sigma_normal_sigma_m / params.noise_grid_spacing_m / params.sigma_k

    noise_field_generator = NoiseFieldGenerator(
        noise_count=noise_count,
        noise_multiplier_normal_mean=params.noise_multiplier_normal_mean,
        noise_multiplier_normal_sigma=params.noise_multiplier_normal_sigma /
        params.sigma_k,
        noise_gaussian_filter_sigma_normal_mean=filter_sigma_normal_mean,
        noise_gaussian_filter_sigma_normal_sigma=filter_sigma_normal_sigma,
        noise_field_grid_range_low=range_low,
        noise_field_grid_range_high=range_high,
        noise_field_grid_resolution=resolution,
        seed=params.seed)

    return noise_field_generator, is_time_dependent_noise


def make_gaussian_air_velocity_field(**kwargs) -> SimpleAirVelocityField:

    params = dacite.from_dict(data_class=GaussianAirVelocityFieldParameters,
                              data=kwargs)

    assert len(params.box_size) == 3, 'box size should be 3d'

    half_x = params.box_size[0] / 2
    half_y = params.box_size[1] / 2

    low_xyz_m = (-half_x, -half_y, 0.0)
    high_xyz_m = (half_x, half_y, params.box_size[2])

    turbulence_noise = None
    is_turbulence_noise_time_dependent: bool | None = None
    if params.turbulence is not None:
        turbulence_noise, is_turbulence_noise_time_dependent = _make_noise_field_generator(
            noise_count=3,
            noise_field_grid_range_low=low_xyz_m,
            noise_field_grid_range_high=high_xyz_m,
            params=params.turbulence)

    wind_noise = None
    is_wind_noise_time_dependent: bool | None = None
    if params.wind is not None and params.wind.noise is not None:
        wind_noise, is_wind_noise_time_dependent = _make_noise_field_generator(
            noise_count=2,  # for u,v components
            noise_field_grid_range_low=(low_xyz_m[2], ),
            noise_field_grid_range_high=(high_xyz_m[2], ),
            params=params.wind.noise)

    assert params.thermal is not None, 'params.thermal is missing'
    field = SimpleAirVelocityField(
        thermal_params=params.thermal,
        horizontal_wind_params=params.wind,
        horizontal_wind_noise_generator=wind_noise,
        is_horizontal_wind_noise_time_dependent=is_wind_noise_time_dependent,
        turbulence_noise_generator=turbulence_noise,
        is_turbulence_noise_time_dependent=is_turbulence_noise_time_dependent,
        turbulence_episode_regenerate_probability=params.
        turbulence_episode_regenerate_probability,
        max_altitude_m=high_xyz_m[2],
        seed=params.seed,
        random_state=params.random_state)
    return field
