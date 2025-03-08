import logging
import numpy as np
from typing import Optional, Union, Tuple, List
from .noise_field import NoiseField
from gymnasium.utils import seeding


class NoiseFieldGenerator():

    def __init__(self,
                 noise_count: int,
                 noise_multiplier_normal_mean: float,
                 noise_multiplier_normal_sigma: float,
                 noise_gaussian_filter_sigma_normal_mean: List[float],
                 noise_gaussian_filter_sigma_normal_sigma: List[float],
                 noise_field_grid_range_low: Union[np.ndarray, Tuple[float]],
                 noise_field_grid_range_high: Union[np.ndarray, Tuple[float]],
                 noise_field_grid_resolution: Union[np.ndarray, Tuple[int]],
                 seed: Optional[int] = None):

        self._log = logging.getLogger(__class__.__name__)

        self.seed(seed)

        self._noise_count = noise_count
        self._noise_multiplier_normal_mean = noise_multiplier_normal_mean
        self._noise_multiplier_normal_sigma = noise_multiplier_normal_sigma
        self._noise_gaussian_filter_sigma_normal_mean = noise_gaussian_filter_sigma_normal_mean
        self._noise_gaussian_filter_sigma_normal_sigma = noise_gaussian_filter_sigma_normal_sigma
        self._noise_field_grid_range_low = noise_field_grid_range_low
        self._noise_field_grid_range_high = noise_field_grid_range_high
        self._noise_field_grid_resolution = noise_field_grid_resolution

        self._spaces = [
            np.linspace(x[0], x[1], x[2]) for x in zip(
                noise_field_grid_range_low, noise_field_grid_range_high,
                noise_field_grid_resolution)
        ]

    def seed(self, seed: int):
        self._np_random, seed = seeding.np_random(seed)
        self._log.debug("random generator initialized with seed %s", seed)

    def initialize_random_generator(self, generator: np.random.Generator):
        self._np_random = generator
        self._log.debug("custom random generator initialized")

    def generate(self):

        noise_multiplier = self._np_random.normal(
            self._noise_multiplier_normal_mean,
            self._noise_multiplier_normal_sigma)

        gaussian_filter_sigma = self._np_random.normal(
            self._noise_gaussian_filter_sigma_normal_mean,
            self._noise_gaussian_filter_sigma_normal_sigma)

        noise_field = NoiseField(
            noise_count=self._noise_count,
            dimension_resolution=self._noise_field_grid_resolution,
            spaces=self._spaces,
            gaussian_filter_sigma=gaussian_filter_sigma,
            noise_multiplier=noise_multiplier,
            np_random=self._np_random)

        self._log.debug(
            "noise field: noise_multiplier=%s,gaussian_filter_sigma=%s",
            noise_multiplier,
            gaussian_filter_sigma,
        )

        initial_conditions = dict(
            noise_multiplier=noise_multiplier,
            gaussian_filter_sigma=gaussian_filter_sigma,
            grid_range=(np.array(self._noise_field_grid_range_high) -
                        np.array(self._noise_field_grid_range_low)),
            grid_resolution=np.array(self._noise_field_grid_resolution))

        return noise_field, initial_conditions
