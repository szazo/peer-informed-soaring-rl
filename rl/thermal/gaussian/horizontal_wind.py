from typing import Union
import numpy as np
from scipy import integrate
from .noise_field import NoiseField


class HorizontalWind:

    _direction_rad: float
    _reference_altitude_m: float
    _wind_speed_at_reference_altitude_m_per_s: float
    _noise_field: NoiseField | None
    _is_time_dependent_noise: bool | None

    def __init__(self, direction_rad: float, reference_altitude_m: float,
                 wind_speed_at_reference_altitude_m_per_s: float,
                 noise_field: NoiseField | None,
                 is_time_dependent_noise: bool | None):

        self._direction_rad = direction_rad
        self._reference_altitude_m = reference_altitude_m
        self._wind_speed_at_reference_altitude_m_per_s = (
            wind_speed_at_reference_altitude_m_per_s)
        self._noise_field = noise_field
        self._is_time_dependent_noise = is_time_dependent_noise

        # noise_field = oiseField2(dimension_resolution=[t_res, resolution], spaces=[t_space, z], gaussian_filter_sigma=[50 / 2.5, 50 / 2.5], noise_multiplier=3., np_random=np_random()[0])
        # pass

    def get_wind(self, time_s: Union[float, np.ndarray],
                 altitude_m: Union[float, np.ndarray]):

        if np.isscalar(time_s) and not np.isscalar(altitude_m):
            time_s = np.full_like(altitude_m, time_s)

        # print('time_s', time_s)

        # get the velocity in m_per_s
        wind_velocity_m_per_s = self._calculate_vertical_wind_profile(
            altitude_m)
        # print('wind_velocity_m_per_s', wind_velocity_m_per_s)

        # create the vector
        wind_velocity_x_m_per_s = wind_velocity_m_per_s * np.cos(
            self._direction_rad)
        wind_velocity_y_m_per_s = wind_velocity_m_per_s * np.sin(
            self._direction_rad)

        # stack them to create x,y components for the different altitudes
        #        wind_velocity_vector_m_per_s = np.array([np.cos(self._direction_rad), ])

        # get the noise for each component for the specified time
        # print('time_s', time_s.shape if not np.isscalar(time_s) else time_s, time_s)
        # print('altitude_m', altitude_m.shape if not np.isscalar(altitude_m) else altitude_m, altitude_m)
        # print('wind_velocity_x_m_per_s', wind_velocity_x_m_per_s, wind_velocity_x_m_per_s.shape)
        #        coordinates = np.vstack((time_s, altitude_m)).T
        if self._is_time_dependent_noise:
            coordinates = np.stack((time_s, altitude_m), axis=-1)
        else:
            coordinates: np.ndarray = np.asarray(altitude_m)
            # we have no time, only a single dimension, add a new axis for the interpolator to be (...,ndim)
            coordinates = coordinates[..., np.newaxis]

        noise_x = 0.
        noise_y = 0.

        if self._noise_field is not None:
            # print('COORD XI', np.min(coordinates), np.max(coordinates), coordinates.shape)
            noise_x = self._noise_field.get_interpolated_noise(
                noise_index=0, coordinates=coordinates)
            # print('NOISE X shape', noise_x.shape)
            noise_y = self._noise_field.get_interpolated_noise(
                noise_index=1, coordinates=coordinates)
            # print('NOISE Y shape', noise_y.shape)
        # print('noise_x', noise_x.shape, noise_x)
        # print('noise_y', noise_y.shape, noise_y)

        # additive noise
        wind_velocity_x_with_noise_m_per_s = wind_velocity_x_m_per_s + noise_x
        wind_velocity_y_with_noise_m_per_s = wind_velocity_y_m_per_s + noise_y
        # print('wind_velocity_x_with_noise_m_per_s', wind_velocity_x_with_noise_m_per_s.shape)
        # print('wind_velocity_y_with_noise_m_per_s', wind_velocity_y_with_noise_m_per_s.shape)

        # print('wind_velocity_x_with_noise_m_per_s', wind_velocity_x_with_noise_m_per_s.shape)
        # print('wind_velocity_y_with_noise_m_per_s', wind_velocity_y_with_noise_m_per_s.shape)

        # wind_velocity_vector_m_per_s = np.vstack(
        #     (wind_velocity_x_with_noise_m_per_s, wind_velocity_y_with_noise_m_per_s)
        # ).T

        wind_velocity_vector_m_per_s = np.stack(
            (wind_velocity_x_with_noise_m_per_s,
             wind_velocity_y_with_noise_m_per_s),
            axis=-1)
        # print('wind_velocity_vector_m_per_s', wind_velocity_vector_m_per_s)

        # print('wind_x', wind_velocity_x_m_per_s)
        # print('wind_y', wind_velocity_y_m_per_s)
        # print('wind vector', wind_velocity_vector_m_per_s)
        # print('wind vector norm', np.linalg.norm(wind_velocity_vector_m_per_s, axis=1))

        # print('wind norm', np.linalg.norm(wind_velocity_vector_m_per_s, axis=1))
        #        alpha = 0.143
        #        f = lambda z: wind_speed_at_reference_height_m_per_s * (z / reference_height_m) ** alpha

        #        cumt = integrate.cumulative_trapezoid(y=f(z / average_vertical_lift_m_per_s) + noise[0], x=(z / average_vertical_lift_m_per_s), initial=0)

        if np.isscalar(altitude_m):
            wind_velocity_vector_m_per_s = wind_velocity_vector_m_per_s.squeeze(
            )

        return wind_velocity_vector_m_per_s

    def integrate_horizontal_displacement(
        self,
        time_s: Union[float, np.ndarray],
        altitude_m: np.ndarray,
        vertical_velocity_m_per_s: Union[float, np.ndarray],
    ):

        horizontal_wind_m_per_s = self.get_wind(time_s=time_s,
                                                altitude_m=altitude_m)
        # print('horizontal_wind', horizontal_wind_m_per_s)

        # stack vertical velocity with the two wind components
        # stacked_vertical_velocity_m_per_s = np.vstack((vertical_velocity_m_per_s, vertical_velocity_m_per_s)).T
        # print('stacked_vertical_velocity_m_per_s', stacked_vertical_velocity_m_per_s)

        #        horizontal_wind_x_m_per_s = horizontal_wind_m_per_s[:, 0]
        #        horizontal_wind_y_m_per_s = horizontal_wind_m_per_s[:, 0]

        #        horizontal_vertical_factor_x = horizontal_wind_x_m_per_s / vertical_velocity_m_per_s
        #        horizontal_vertical_factor_y = horizontal_wind_y_m_per_s / vertical_velocity_m_per_s
        horizontal_vertical_factor_x = (horizontal_wind_m_per_s[:, 0] /
                                        vertical_velocity_m_per_s)
        horizontal_vertical_factor_y = (horizontal_wind_m_per_s[:, 1] /
                                        vertical_velocity_m_per_s)

        # print('horizontal_vertical_factor', horizontal_vertical_factor_x, horizontal_vertical_factor_y)

        # print('stacked altitude', np.vstack((altitude_m, altitude_m)).T)

        #        displacements_m = integrate.cumulative_trapezoid(y=horizontal_vertical_factor[:,0], x=np.vstack((altitude_m, altitude_m)).T[:,0])
        displacements_x_m = integrate.cumulative_trapezoid(
            y=horizontal_vertical_factor_x, x=altitude_m, initial=0)
        displacements_y_m = integrate.cumulative_trapezoid(
            y=horizontal_vertical_factor_y, x=altitude_m, initial=0)
        # cumt = integrate.cumulative_trapezoid(y=horizontal_vertical_factor_f(z, t=0), x=(z), initial=0)

        # print('displacements_x_m', displacements_x_m)
        # print('displacements_y_m', displacements_y_m)

        displacement_vector_m = np.vstack(
            (displacements_x_m, displacements_y_m)).T
        # print('displacement_vector_m', displacement_vector_m)

        #        horizontal_vertical_
        return displacement_vector_m

    def _calculate_vertical_wind_profile(self, altitude_m: Union[float,
                                                                 np.ndarray]):
        # calculate the vertical wind profile using wind profile power law (https://en.wikipedia.org/wiki/Wind_profile_power_law)
        alpha = 0.143

        return (self._wind_speed_at_reference_altitude_m_per_s *
                (altitude_m / self._reference_altitude_m)**alpha)
