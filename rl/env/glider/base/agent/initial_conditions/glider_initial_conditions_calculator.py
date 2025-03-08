import logging
from dataclasses import dataclass, field
import numpy as np
from gymnasium.utils import seeding

from utils import Vector3
from ....aerodynamics import AerodynamicsInterface
from ....air_velocity_field import AirVelocityFieldInterface
from ..glider_state import GliderState
from ..air_velocity_post_processor import AirVelocityPostProcessor

from .glider_initial_tangent_position import \
    GliderInitialTangentPosition, GliderInitialTangentPositionParameters, GliderInitialTangentPositionInfo


@dataclass
class GliderInitialConditionsParameters:
    tangent_position_parameters: GliderInitialTangentPositionParameters = field(
        default_factory=GliderInitialTangentPositionParameters)
    altitude_earth_m_mean: float = 400.
    altitude_earth_m_sigma: float = 50.
    # used for control that the specified sigma should contain the most of the bell volume: sigma'=sigma/k
    altitude_earth_m_sigma_k: float = 2.5


@dataclass
class GliderInitialConditionsInfo:
    tangent_position: GliderInitialTangentPositionInfo
    air_velocity_earth_xyz_m_s: Vector3


class GliderInitialConditionsCalculator():

    _initial_conditions_params: GliderInitialConditionsParameters
    _aerodynamics: AerodynamicsInterface
    _air_velocity_field: AirVelocityFieldInterface
    _np_random: np.random.Generator

    def __init__(self,
                 initial_conditions_params: GliderInitialConditionsParameters,
                 aerodynamics: AerodynamicsInterface,
                 air_velocity_field: AirVelocityFieldInterface):

        # create the logger
        self._log = logging.getLogger(__class__.__name__)

        # set instance fields
        self._initial_conditions_params = initial_conditions_params
        self._aerodynamics = aerodynamics
        self._air_velocity_field = air_velocity_field

        # create the calculator which will be used to calculate the tangential
        # orientation (to the core) and position
        self._initial_tangent_position_calculator = GliderInitialTangentPosition(
            params=self._initial_conditions_params.tangent_position_parameters)
        # initialize the random number generator without seed
        self._np_random, _ = seeding.np_random()

    def seed(self, seed: int | None = None) -> None:
        self._log.debug('seed: %s', seed)
        self._initial_tangent_position_calculator.seed(seed)
        self._np_random, seed = seeding.np_random(seed)

    def generate(
        self, t_s: float, air_velocity_post_processor: AirVelocityPostProcessor
    ) -> tuple[GliderState, GliderInitialConditionsInfo]:

        self._log.debug('generate; t_s=%s', t_s)

        # calculate the position and orientation
        position_earth_xyz_m, \
            yaw_pitch_roll_earth_to_body_rad, \
            yaw_pitch_roll_earth_to_body_dot_rad_per_s, \
            initial_position_info = self._calculate_random_position_and_orientation(
                t_s=t_s)

        # get the air velocity vector
        air_velocity_earth_xyz_m_s, _ = self._air_velocity_field.get_velocity(
            position_earth_xyz_m[0], position_earth_xyz_m[1],
            position_earth_xyz_m[2], t_s)

        # postprocess
        air_velocity_earth_xyz_m_s = air_velocity_post_processor.process(
            air_velocity_earth_xyz_m_s)

        # get initial velocity of the glider
        velocity_earth_xyz_m_per_s, velocity_airmass_relative_xyz_m_per_s = \
            self._aerodynamics.get_initial_velocity_earth(
                heading_earth_to_body_rad=yaw_pitch_roll_earth_to_body_rad[0],
                wind_velocity_earth_m_per_s=air_velocity_earth_xyz_m_s)

        state = GliderState(
            position_earth_xyz_m=position_earth_xyz_m,
            velocity_earth_xyz_m_per_s=velocity_earth_xyz_m_per_s,
            yaw_pitch_roll_earth_to_body_rad=yaw_pitch_roll_earth_to_body_rad,
            yaw_pitch_roll_earth_to_body_dot_rad_per_s=
            yaw_pitch_roll_earth_to_body_dot_rad_per_s,
            velocity_airmass_relative_xyz_m_per_s=
            velocity_airmass_relative_xyz_m_per_s)

        self._log.debug('generated state=%s', state)
        self._log.debug('initial_conditions_info=%s', initial_position_info)

        info = GliderInitialConditionsInfo(
            tangent_position=initial_position_info,
            air_velocity_earth_xyz_m_s=air_velocity_earth_xyz_m_s)

        return state, info

    def _calculate_random_position_and_orientation(
        self, t_s: float
    ) -> tuple[Vector3, Vector3, Vector3, GliderInitialTangentPositionInfo]:

        self._log.debug('_calculate_random_position_and_orientation; t_s=%s',
                        t_s)

        params = self._initial_conditions_params

        # calculate the random altitude
        altitude_m = self._np_random.normal(
            params.altitude_earth_m_mean,
            params.altitude_earth_m_sigma / params.altitude_earth_m_sigma_k)

        # get the core at that altitude
        core_position_earth_xy_m = self._air_velocity_field.get_thermal_core(
            altitude_m, t_s)

        # calculate tangential position
        tangent_position_xy_m, \
            tangent_theta_rad, \
            unit_tangent_vector, \
            position_info = self._initial_tangent_position_calculator.generate(
                core_position_earth_xy_m[0], core_position_earth_xy_m[1])

        # the position vector
        position_earth_xyz_m = np.array(
            [tangent_position_xy_m[0], tangent_position_xy_m[1], altitude_m])

        # use the orientation as yaw
        yaw_pitch_roll_earth_to_body_rad = np.array(
            [tangent_theta_rad, 0.0, 0.0])

        # zero dot
        yaw_pitch_roll_earth_to_body_dot_rad_per_s = np.array([0.0, 0.0, 0.0])

        return position_earth_xyz_m, \
            yaw_pitch_roll_earth_to_body_rad, \
            yaw_pitch_roll_earth_to_body_dot_rad_per_s, \
            position_info
