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

        #        print('NOISE_SHAPE', noise.shape, np.min(noise), np.max(noise))
        #        print('NOISE_SIGMA', self._gaussian_filter_sigma)

        self._log.debug("filtering using gaussian filter; sigma=%s",
                        self._gaussian_filter_sigma)
        # there is no smoothing between the coordinates now
        # sigma = (0., 0., self._gaussian_filter_sigma, self._gaussian_filter_sigma, self._gaussian_filter_sigma)
        # print('SIGMA', sigma)
        filtered_noise = scipy.ndimage.gaussian_filter(
            input=noise, sigma=self._gaussian_filter_sigma, mode="wrap")
        self._log.debug("filtering done")

        #        print('FILTERED NOISE_SHAPE', filtered_noise.shape, np.min(filtered_noise), np.max(filtered_noise))

        return filtered_noise

    def get_interpolated_noise(self, noise_index: int,
                               coordinates: np.ndarray | tuple):

        # grid_points: linear spaces with same dimension as the noise
        # coordinates: coordinate list with the same dimension as noise

        # Interpolate the data at the specified points
        # print("self._noise", self._noise.shape)
        # print(
        #     "grid_points",
        #     grid_points[0].shape,
        #     grid_points[1].shape,
        #     grid_points[2].shape,
        # )
        # print('coords', coordinates[0].shape, coordinates[1].shape, coordinates[2].shape)
        # if type(coordinates[0]) != np.ndarray:
        #     print('COORD', coordinates)

        # print('SPACE XI', self._spaces)
        # print('NOISE', self._noise[noise_index].shape)
        try:
            # interpolated_value = scipy.interpolate.interpn(
            #     points=self._spaces, values=self._noise[noise_index], xi=coordinates
            # )

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


# class NoiseField:
#     def __init__(
#         self,
#         dimension_resolution: Union[np.ndarray, Tuple[float]],
#         spaces: List[np.ndarray],
#         gaussian_filter_sigma: float,
#         noise_multiplier: float,
#         np_random: np.random.Generator,
#     ):

#         self._log = logging.getLogger(__class__.__name__)

#         self._resolution = dimension_resolution
#         self._spaces = spaces
#         self._gaussian_filter_sigma = gaussian_filter_sigma
#         self._noise_multiplier = noise_multiplier
#         self._np_random = np_random

#         self._dim = len(dimension_resolution)
#         self._noise = self._generate_filtered_noise()

#     def _generate_filtered_noise(self):

#         self._log.debug(
#             "generating noise resolution=%s,gaussian_filter_sigma=%s,noise_multiplier=%s",
#             self._resolution,
#             self._gaussian_filter_sigma,
#             self._noise_multiplier
#         )

#         noise = self._np_random.uniform(
#             low=-1.0, high=1.0, size=(self._dim, *self._resolution)
#         )
#         filtered_noise = scipy.ndimage.gaussian_filter(
#             input=noise, sigma=self._gaussian_filter_sigma, mode="wrap"
#         )
#         return filtered_noise * self._noise_multiplier

#     def get_interpolated_noise(
#         self, dim: int, coordinates: np.ndarray):

#         # grid_points: linear spaces with same dimension as the noise
#         # coordinates: coordinate list with the same dimension as noise

#         # Interpolate the data at the specified points
#         # print("self._noise", self._noise.shape)
#         # print(
#         #     "grid_points",
#         #     grid_points[0].shape,
#         #     grid_points[1].shape,
#         #     grid_points[2].shape,
#         # )
#         # print('coords', coordinates[0].shape, coordinates[1].shape, coordinates[2].shape)
#         # if type(coordinates[0]) != np.ndarray:
#         #     print('COORD', coordinates)

#         # print('XI', coordinates)
#         try:
#             interpolated_value = scipy.interpolate.interpn(
#                 points=self._spaces, values=self._noise[dim], xi=coordinates
#             )
#         except Exception:
#             print('out of bound coordinates passed', coordinates)
#             raise

#         self._log.debug(
#             "returning interpolated noise for dimension %s; coordinates: %s; result: %s",
#             dim,
#             coordinates,
#             interpolated_value,
#         )

#         return interpolated_value

#     def get_noise(self, dim: int):
#         return self._noise[dim]
