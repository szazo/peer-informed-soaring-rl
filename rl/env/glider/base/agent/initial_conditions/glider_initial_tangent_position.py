from dataclasses import dataclass
import numpy as np
import logging
from gymnasium.utils import seeding


@dataclass
class GliderInitialTangentPositionParameters:

    starting_distance_from_tangent_position_m_normal_mean: float = 400.
    starting_distance_from_tangent_position_m_normal_sigma: float = 100.
    # used for control that the specified sigma should contain the most of the bell volume: sigma'=sigma/k
    starting_distance_from_tangent_position_m_normal_sigma_k: float = 2.5

    tangent_distance_from_core_m_normal_mean: float = 50.0
    tangent_distance_from_core_m_normal_sigma: float = 40.0
    # used for control that the specified sigma should contain the most of the bell volume: sigma'=sigma/k
    tangent_distance_from_core_m_normal_sigma_k: float = 2.5

    increase_distance_with_every_generation_by: float | None = None


@dataclass
class GliderInitialTangentPositionInfo:
    distance_from_core_m: float
    starting_distance_from_tangent_position_m: float


class GliderInitialTangentPosition:

    _params: GliderInitialTangentPositionParameters

    _current_dynamic_distance_from_tangent_position_m: float

    def __init__(
        self,
        params: GliderInitialTangentPositionParameters,
        seed: int | None = None,
    ):

        self._log = logging.getLogger(__class__.__name__)
        self._log.debug('__init__: params=%s, seed=%s', params, seed)

        self._params = params
        self.seed(seed)

        self._current_dynamic_distance_from_tangent_position_m = 0.

    def seed(self, seed: int | None):
        self._log.debug('seed: %s', seed)

        self._np_random, seed = seeding.np_random(seed)
        self._log.debug("random generator initialized with seed %s", seed)

    def generate(self, core_x_m: float, core_y_m: float):

        self._log.debug('generate, core_x_m=%s,core_y_m=%s', core_x_m,
                        core_y_m)

        # calculate the starting distance from the core
        radius_m = np.abs(
            self._np_random.normal(
                self._params.tangent_distance_from_core_m_normal_mean,
                self._params.tangent_distance_from_core_m_normal_sigma /
                self._params.tangent_distance_from_core_m_normal_sigma_k,
            ))

        # orientation of the radius
        theta_rad = self._np_random.random() * 2 * np.pi

        # sign for the tangent theta orientation
        tangent_theta_sign = 1.0 if self._np_random.random() < 0.5 else -1.0

        # tangent orientation
        tangent_theta_rad = (theta_rad + tangent_theta_sign *
                             (np.pi / 2)) % (np.pi * 2)

        # the position of the tangent vector
        tangent_position_xy_m = (
            np.array([np.cos(theta_rad), np.sin(theta_rad)]) * radius_m +
            np.array([core_x_m, core_y_m]))

        if self._params.increase_distance_with_every_generation_by is not None:
            if self._current_dynamic_distance_from_tangent_position_m < self._params.starting_distance_from_tangent_position_m_normal_mean:
                self._current_dynamic_distance_from_tangent_position_m += self._params.increase_distance_with_every_generation_by
            start_distance_from_tangent_m = self._current_dynamic_distance_from_tangent_position_m
            self._log.debug('current starting distance=%s',
                            start_distance_from_tangent_m)
        else:
            start_distance_from_tangent_m = self._np_random.normal(
                self._params.
                starting_distance_from_tangent_position_m_normal_mean,
                self._params.
                starting_distance_from_tangent_position_m_normal_sigma /
                self._params.
                starting_distance_from_tangent_position_m_normal_sigma_k)

        if start_distance_from_tangent_m < 0.:
            start_distance_from_tangent_m = 0.

        # unit tangent vector
        unit_tangent_vector = np.array(
            [np.cos(tangent_theta_rad),
             np.sin(tangent_theta_rad)])

        starting_position_xy_m = tangent_position_xy_m - \
            unit_tangent_vector * start_distance_from_tangent_m

        self._log.debug(
            "tangential position radius_m=%s,tangent_position_xy_m=%s,tangent_theta_rad=%s,tangent_theta_deg=%s,unit_tangent_vector=%s,starting_position_xy_m=%s",
            radius_m, tangent_position_xy_m, tangent_theta_rad,
            np.rad2deg(tangent_theta_rad), unit_tangent_vector,
            starting_position_xy_m)

        info = GliderInitialTangentPositionInfo(
            distance_from_core_m=radius_m,
            starting_distance_from_tangent_position_m=
            start_distance_from_tangent_m)

        return starting_position_xy_m, tangent_theta_rad, unit_tangent_vector, info
