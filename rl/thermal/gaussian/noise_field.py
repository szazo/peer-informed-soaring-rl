from typing import Tuple, Union, List
import logging
import numpy as np
import scipy


class NoiseField:

    _log: logging.Logger

    # the interpolators for each noise
    _interpolators: List[scipy.interpolate.RegularGridInterpolator] = []

    def __init__(
        self,
        noise_count: int,
        dimension_resolution: Union[np.ndarray, Tuple[float]],
        spaces: List[np.ndarray],
        gaussian_filter_sigma: Union[float, np.ndarray, Tuple[float]],
        noise_multiplier: float,
        np_random: np.random.Generator,
    ):

        self._log = logging.getLogger(__class__.__name__)

        self._log.debug(
            '__init__; noise_count=%s,dimension_resolution=%s,gaussian_filter_sigma=%s,noise_multiplier=%s',
            noise_count, dimension_resolution, gaussian_filter_sigma,
            noise_multiplier)

        self._noise_count = noise_count
        self._resolution = dimension_resolution
        self._spaces = spaces
        self._gaussian_filter_sigma = gaussian_filter_sigma
        self._noise_multiplier = noise_multiplier
        self._np_random = np_random

        noise = self._generate_filtered_noise()

        # normalize between -1 and 1
        noise = 2 * (noise - np.min(noise)) / (np.max(noise) -
                                               np.min(noise)) - 1

        self._noise = noise * noise_multiplier

        # create the interpolators
        self._interpolators = [
            scipy.interpolate.RegularGridInterpolator(
                points=self._spaces, values=self._noise[noise_index])
            for noise_index in range(self._noise_count)
        ]

    def _generate_filtered_noise(self):

        noise_size = [self._noise_count, *self._resolution]
        self._log.debug("generating noise size=%s", noise_size)

        noise = self._np_random.uniform(low=-1.0, high=1.0,
                                        size=noise_size).astype(np.float32)

        self._log.debug("filtering using gaussian filter; sigma=%s",
                        self._gaussian_filter_sigma)
        # there is no smoothing between the coordinates now
        filtered_noise = scipy.ndimage.gaussian_filter(
            input=noise, sigma=self._gaussian_filter_sigma, mode="wrap")
        self._log.debug("filtering done")

        return filtered_noise

    def get_interpolated_noise(self, noise_index: int,
                               coordinates: np.ndarray | tuple):

        try:
            interpolated_value = self._interpolators[noise_index](coordinates)

        except Exception:
            print(
                'out of bound coordinates passed; using default 0.; coordinates=',
                coordinates, 'noise_index=', noise_index, 'spaces(min,max)=',
                [(np.min(space), np.max(space)) for space in self._spaces])

            self._interpolators[noise_index].bounds_error = False
            self._interpolators[noise_index].fill_value = 0.
            try:
                interpolated_value = self._interpolators[noise_index](
                    coordinates)
            finally:
                self._interpolators[noise_index].bounds_error = True
                self._interpolators[noise_index].fill_value = np.nan

        self._log.debug(
            "returning interpolated noise for coordinates: %s; result: %s",
            coordinates,
            interpolated_value,
        )

        return interpolated_value

    def get_noise(self, noise_index: int):
        return self._noise[noise_index]
