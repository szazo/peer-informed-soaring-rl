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

    def get_wind(self, time_s: Union[float, np.ndarray],
                 altitude_m: Union[float, np.ndarray]):

        if np.isscalar(time_s) and not np.isscalar(altitude_m):
            time_s = np.full_like(altitude_m, time_s)

        # get the velocity in m_per_s
        wind_velocity_m_per_s = self._calculate_vertical_wind_profile(
            altitude_m)

        # create the vector
        wind_velocity_x_m_per_s = wind_velocity_m_per_s * np.cos(
            self._direction_rad)
        wind_velocity_y_m_per_s = wind_velocity_m_per_s * np.sin(
            self._direction_rad)

        if self._is_time_dependent_noise:
            coordinates = np.stack((time_s, altitude_m), axis=-1)
        else:
            coordinates: np.ndarray = np.asarray(altitude_m)
            # we have no time, only a single dimension, add a new axis for the interpolator to be (...,ndim)
            coordinates = coordinates[..., np.newaxis]

        noise_x = 0.
        noise_y = 0.

        if self._noise_field is not None:
            noise_x = self._noise_field.get_interpolated_noise(
                noise_index=0, coordinates=coordinates)
            noise_y = self._noise_field.get_interpolated_noise(
                noise_index=1, coordinates=coordinates)

        # additive noise
        wind_velocity_x_with_noise_m_per_s = wind_velocity_x_m_per_s + noise_x
        wind_velocity_y_with_noise_m_per_s = wind_velocity_y_m_per_s + noise_y

        wind_velocity_vector_m_per_s = np.stack(
            (wind_velocity_x_with_noise_m_per_s,
             wind_velocity_y_with_noise_m_per_s),
            axis=-1)

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
        horizontal_vertical_factor_x = (horizontal_wind_m_per_s[:, 0] /
                                        vertical_velocity_m_per_s)
        horizontal_vertical_factor_y = (horizontal_wind_m_per_s[:, 1] /
                                        vertical_velocity_m_per_s)

        displacements_x_m = integrate.cumulative_trapezoid(
            y=horizontal_vertical_factor_x, x=altitude_m, initial=0)
        displacements_y_m = integrate.cumulative_trapezoid(
            y=horizontal_vertical_factor_y, x=altitude_m, initial=0)

        displacement_vector_m = np.vstack(
            (displacements_x_m, displacements_y_m)).T
        return displacement_vector_m

    def _calculate_vertical_wind_profile(self, altitude_m: Union[float,
                                                                 np.ndarray]):
        # calculate the vertical wind profile using wind profile power law (https://en.wikipedia.org/wiki/Wind_profile_power_law)
        alpha = 0.143

        return (self._wind_speed_at_reference_altitude_m_per_s *
                (altitude_m / self._reference_altitude_m)**alpha)
