from typing import TypedDict, cast, OrderedDict
from copy import deepcopy
from dataclasses import dataclass, asdict, field
import logging
from collections import OrderedDict
import functools
from env.glider.aerodynamics.api import AerodynamicsInfo
import numpy as np
import gymnasium

from utils import Vector3D, Vector2D
from ...aerodynamics import AerodynamicsInterface
from ...air_velocity_field import AirVelocityFieldInterface
from .types import AgentID
from ..simulation_box_params import SimulationBoxParameters
from .glider_state import GliderState
from .glider_info import GliderInfo, GliderTrajectory
from .initial_conditions import GliderInitialConditionsCalculator
from .reward_cutoff import RewardAndCutoffResult, CutoffReason, GliderRewardCalculator, RewardAdditionalInfo, GliderCutoffCalculator, CutoffAdditionalInfo
from .control_dynamics import ControlDynamics, ControlDynamicsParams, ControlState
from .air_velocity_post_processor import AirVelocityPostProcessor, AirVelocityPostProcessorParams


class GliderAgentObsType(TypedDict):
    position_earth_xyz_m: Vector3D
    velocity_earth_xyz_m_per_s: Vector3D
    yaw_pitch_roll_earth_to_body_rad: Vector3D
    velocity_airmass_relative_xyz_m_per_s: Vector3D
    t_s: float


GliderAgentActType = float


@dataclass
class GliderAgentParameters:
    roll_control_dynamics_params: ControlDynamicsParams | None = None
    default_action: GliderAgentActType = 0.0
    air_velocity_post_process: AirVelocityPostProcessorParams = field(
        default_factory=AirVelocityPostProcessorParams)


class GliderAgent:

    _agent_id: AgentID
    _state: GliderState | None = None
    _info: GliderInfo | None = None
    _trajectory: np.ndarray

    _initial_conditions_info: dict | None
    _aerodynamics_info: AerodynamicsInfo | None = None
    _air_velocity_post_processor: AirVelocityPostProcessor

    _log: logging.Logger
    _parameters: GliderAgentParameters
    _aerodynamics: AerodynamicsInterface
    _roll_control_dynamics: ControlDynamics | None
    _air_velocity_field: AirVelocityFieldInterface
    _initial_conditions_calculator: GliderInitialConditionsCalculator
    _cutoff_calculator: GliderCutoffCalculator
    _reward_calculator: GliderRewardCalculator
    _np_random: np.random.Generator

    def state_clone(self):
        clone = self.__class__(
            agent_id=self._agent_id,
            parameters=self._parameters,
            initial_conditions_calculator=self._initial_conditions_calculator,
            cutoff_calculator=self._cutoff_calculator.clone(),
            reward_calculator=self._reward_calculator.clone(),
            aerodynamics=self._aerodynamics,
            air_velocity_field=self._air_velocity_field,
            np_random=self._np_random)
        clone._state = deepcopy(self._state)
        clone._aerodynamics_info = deepcopy(self._aerodynamics_info)
        clone._initial_conditions_info = deepcopy(
            self._initial_conditions_info)
        clone._info = deepcopy(self._info)
        clone._trajectory = deepcopy(self._trajectory)
        clone._air_velocity_post_processor = self._air_velocity_post_processor.clone(
        )

        if self._roll_control_dynamics is not None:
            assert clone._roll_control_dynamics is not None
            clone._roll_control_dynamics.clone_state_from(
                self._roll_control_dynamics)

        return clone

    # type for the numpy array which stores trajectory info for rendering
    _trajectory_dtype = np.dtype([
        ("time_s", "f4"), ("position_earth_xyz_m", "f4", (3, )),
        ("velocity_earth_xyz_m_per_s", "f4", (3, )),
        ("yaw_pitch_roll_earth_to_body_rad", "f4", (3, )),
        ("air_velocity_earth_xyz_m_per_s", "f4", (3, )),
        ("indicated_airspeed_m_per_s", "f4")
    ])

    def get_trajectory(self) -> GliderTrajectory:

        return GliderTrajectory(
            time_s=self._trajectory['time_s'],
            position_earth_xyz_m=self._trajectory['position_earth_xyz_m'],
            velocity_earth_xyz_m_per_s=self.
            _trajectory['velocity_earth_xyz_m_per_s'],
            yaw_pitch_roll_earth_to_body_rad=self.
            _trajectory['yaw_pitch_roll_earth_to_body_rad'],
            air_velocity_earth_xyz_m_per_s=self.
            _trajectory['air_velocity_earth_xyz_m_per_s'],
            indicated_airspeed_m_per_s=self.
            _trajectory['indicated_airspeed_m_per_s'])

    def __init__(
        self,
        agent_id: AgentID,
        parameters: GliderAgentParameters,
        initial_conditions_calculator: GliderInitialConditionsCalculator,
        cutoff_calculator: GliderCutoffCalculator,
        reward_calculator: GliderRewardCalculator,
        aerodynamics: AerodynamicsInterface,
        air_velocity_field: AirVelocityFieldInterface,
        # we can't use per agent seed, because otherwise we should somehow handle multi agent seeding in reproducible manner
        np_random: np.random.Generator):

        self._log = logging.getLogger(__class__.__name__)

        self._np_random = np_random

        self._parameters = parameters
        self._aerodynamics = aerodynamics
        self._roll_control_dynamics = None
        if parameters.roll_control_dynamics_params is not None:
            self._roll_control_dynamics = ControlDynamics(
                params=parameters.roll_control_dynamics_params)

        self._air_velocity_field = air_velocity_field

        self._agent_id = agent_id
        self._initial_conditions_calculator = initial_conditions_calculator
        self._cutoff_calculator = cutoff_calculator
        self._reward_calculator = reward_calculator

        # agent dependent air velocity filter and noise
        self._air_velocity_post_processor = AirVelocityPostProcessor(
            params=self._parameters.air_velocity_post_process,
            np_random=np_random)

    def reset(self, time_s: float):

        self._air_velocity_post_processor.reset()
        self._aerodynamics_info = self._aerodynamics.reset()

        # generate initial position info
        state, initial_conditions_info = self._initial_conditions_calculator.generate(
            time_s,
            air_velocity_post_processor=self._air_velocity_post_processor)

        distance_from_core_m, core_position_earth_xy_m = self._calculate_distance_from_core_for_position(
            position_earth_xyz_m=state.position_earth_xyz_m, time_s=time_s)

        self._state = state
        self._info = self._create_info(
            air_velocity_earth_xyz_m_s=initial_conditions_info.
            air_velocity_earth_xyz_m_s,
            distance_from_core_m=distance_from_core_m,
            core_position_earth_xy_m=core_position_earth_xy_m,
            time_s_without_lift=0.,
            success=False,
            cutoff_reason='none',
            aerodynamics_info=self._aerodynamics_info)

        self._initial_conditions_info = asdict(initial_conditions_info) | dict(
            position_earth_xyz_m=state.position_earth_xyz_m)

        self._cutoff_calculator.reset(time_s=time_s)
        self._reward_calculator.reset(state)

        self._trajectory = np.array([], dtype=self._trajectory_dtype)

    def step(self, action: GliderAgentActType, current_time_s: float,
             next_time_s: float, dt_s: float) -> RewardAndCutoffResult:

        assert np.isclose(
            current_time_s + dt_s,
            next_time_s), 'dt_s should match current and next time'

        # calculate the new state vector
        self._log.debug('step; id=%s,action=%s', self._agent_id, action)

        if np.isnan(action):
            self._log.debug('nan action; using defaul_actiont=%s',
                            self._parameters.default_action)
            action = self._parameters.default_action

        current_state = self._state
        current_info = self._info
        assert current_state is not None, 'missing current state'
        assert current_info is not None, 'missing current state'

        # get air velocity at the current position and time
        current_air_velocity_earth_xyz_m_s = current_info.air_velocity_earth_xyz_m_s
        assert current_air_velocity_earth_xyz_m_s.shape == (3, )

        # the new action will be the current setpoint of roll control
        current_roll_state: ControlState = np.array([
            current_state.yaw_pitch_roll_earth_to_body_rad[2],
            current_state.yaw_pitch_roll_earth_to_body_dot_rad_per_s[2]
        ])

        roll_earth_to_body_rad = action
        roll_earth_to_body_dot_rad_per_s = (
            roll_earth_to_body_rad -
            current_state.yaw_pitch_roll_earth_to_body_rad[2]) / dt_s
        if self._roll_control_dynamics is not None:
            # use the control dynamics to filter the movement
            roll_earth_to_body_rad_setpoint = np.array(
                [action, 0.])  # second is the theta_dot
            self._roll_control_dynamics.update_target_setpoint(
                setpoint=roll_earth_to_body_rad_setpoint)

            # update the control dynamics
            next_roll_state = self._roll_control_dynamics.calculate(
                current_roll_state, np.array([0.0, dt_s]))[1]
            # use the new roll from the current control dynamics state
            roll_earth_to_body_rad = next_roll_state[0]
            roll_earth_to_body_dot_rad_per_s = next_roll_state[1]

        # update the roll in the state
        yaw_pitch_roll_earth_to_body_rad = np.copy(
            current_state.yaw_pitch_roll_earth_to_body_rad)
        yaw_pitch_roll_earth_to_body_rad[2] = roll_earth_to_body_rad

        # step the aerodynamics using the new roll and queried air velocity
        (next_position_earth_xyz_m, next_velocity_earth_xyz_m_per_s,
         next_yaw_pitch_roll_earth_to_body_rad,
         next_indicated_airspeed_m_per_s,
         next_velocity_airmass_relative_xyz_m_per_s) = self._aerodynamics.step(
             position_earth_m=current_state.position_earth_xyz_m,
             velocity_earth_m_per_s=current_state.velocity_earth_xyz_m_per_s,
             yaw_pitch_roll_earth_to_body_rad=
             yaw_pitch_roll_earth_to_body_rad,  # use the new control
             wind_velocity_earth_m_per_s=current_air_velocity_earth_xyz_m_s,
             velocity_airmass_relative_m_per_s=current_state.
             velocity_airmass_relative_xyz_m_per_s,
             dt_s=dt_s)

        # check that it is not overridden by the aerodynamics
        assert next_yaw_pitch_roll_earth_to_body_rad[
            2] == roll_earth_to_body_rad, 'roll updated by aerodynamics'
        next_yaw_pitch_roll_earth_to_body_dot_rad_per_s = np.array(
            [0., 0., roll_earth_to_body_dot_rad_per_s])

        self._log.debug(
            'next_position_earth_xyz_m=%s,'
            'next_velocity_earth_xyz_m_per_s=%s,'
            'next_yaw_pitch_roll_earth_to_body_rad=%s,'
            'next_yaw_pitch_roll_earth_to_body_dot_rad_per_s=%s,'
            'next_indicated_airspeed_m_per_s=%s,'
            'next_velocity_airmass_relative_xyz_m_per_s=%s',
            next_position_earth_xyz_m, next_velocity_earth_xyz_m_per_s,
            next_yaw_pitch_roll_earth_to_body_rad,
            next_yaw_pitch_roll_earth_to_body_dot_rad_per_s,
            next_indicated_airspeed_m_per_s,
            next_velocity_airmass_relative_xyz_m_per_s)

        # create the new state
        next_state = GliderState(
            position_earth_xyz_m=next_position_earth_xyz_m,
            velocity_earth_xyz_m_per_s=next_velocity_earth_xyz_m_per_s,
            yaw_pitch_roll_earth_to_body_rad=
            next_yaw_pitch_roll_earth_to_body_rad,
            yaw_pitch_roll_earth_to_body_dot_rad_per_s=
            next_yaw_pitch_roll_earth_to_body_dot_rad_per_s,
            velocity_airmass_relative_xyz_m_per_s=
            next_velocity_airmass_relative_xyz_m_per_s)

        # query the air velocity at the new position
        (
            next_air_velocity_earth_xyz_m_s,
            _,
        ) = self._air_velocity_field.get_velocity(next_position_earth_xyz_m[0],
                                                  next_position_earth_xyz_m[1],
                                                  next_position_earth_xyz_m[2],
                                                  next_time_s)
        # post process the air velocity
        next_air_velocity_earth_xyz_m_s = self._air_velocity_post_processor.process(
            next_air_velocity_earth_xyz_m_s)
        assert next_air_velocity_earth_xyz_m_s.shape == (3, )

        # create trajectory item for rendering
        trajectory_item = np.array(
            [(next_time_s, next_state.position_earth_xyz_m,
              next_state.velocity_earth_xyz_m_per_s,
              next_state.yaw_pitch_roll_earth_to_body_rad,
              next_air_velocity_earth_xyz_m_s, next_indicated_airspeed_m_per_s)
             ],
            dtype=self._trajectory_dtype,
        )
        self._trajectory = np.append(self._trajectory, trajectory_item)

        distance_from_core_m, core_position_earth_xy_m = self._calculate_distance_from_core_for_position(
            next_state.position_earth_xyz_m, time_s=next_time_s)

        # check cutoff
        cutoff_additional_info = CutoffAdditionalInfo(
            distance_from_core_m=distance_from_core_m)
        cutoff_result, cutoff_info = self._cutoff_calculator.calculate(
            glider_state=next_state,
            info=cutoff_additional_info,
            time_s=next_time_s)

        # calculate the reward
        reward_additional_info = RewardAdditionalInfo(
            dt_s=dt_s, cutoff_result=cutoff_result.result)
        reward_result = self._reward_calculator.calculate(
            state=next_state, info=reward_additional_info)

        assert self._aerodynamics_info is not None
        next_info = self._create_info(
            air_velocity_earth_xyz_m_s=next_air_velocity_earth_xyz_m_s,
            distance_from_core_m=distance_from_core_m,
            core_position_earth_xy_m=core_position_earth_xy_m,
            time_s_without_lift=cutoff_info.time_s_without_lift,
            success=cutoff_result.result == 'success',
            cutoff_reason=cutoff_result.reason,
            aerodynamics_info=self._aerodynamics_info)

        self._state = next_state
        self._info = next_info

        return RewardAndCutoffResult(reward_result, cutoff_result)

    def _create_info(self, air_velocity_earth_xyz_m_s: Vector3D,
                     distance_from_core_m: float,
                     core_position_earth_xy_m: Vector2D,
                     time_s_without_lift: float, success: bool,
                     cutoff_reason: CutoffReason,
                     aerodynamics_info: AerodynamicsInfo) -> GliderInfo:

        return GliderInfo(
            distance_from_core_m=distance_from_core_m,
            core_position_earth_xy_m=core_position_earth_xy_m,
            air_velocity_earth_xyz_m_s=air_velocity_earth_xyz_m_s,
            success=success,
            time_s_without_lift=time_s_without_lift,
            cutoff_reason=cutoff_reason,
            aerodynamics_info=aerodynamics_info)

    def _calculate_distance_from_core_for_position(
            self, position_earth_xyz_m: Vector3D, time_s: float):
        # query core position
        altitude_m = position_earth_xyz_m[2]
        core_position_earth_xy_m = self._air_velocity_field.get_thermal_core(
            z_earth_m=altitude_m, t_s=time_s)

        distance_from_core_m = self._calculate_distance_from_core(
            position_earth_xyz_m, core_position_earth_xy_m)

        return distance_from_core_m, core_position_earth_xy_m

    def _calculate_distance_from_core(
            self, position_earth_xyz_m: Vector3D,
            core_position_earth_xy_m: Vector2D) -> float:

        distance_from_core_m = np.linalg.norm(position_earth_xyz_m[:2] -
                                              core_position_earth_xy_m)
        return float(distance_from_core_m)

    @property
    def agent_id(self) -> AgentID:
        return self._agent_id

    def get_observation(self, t_s: float) -> GliderAgentObsType:

        assert self._state is not None, 'missing state; reset called?'

        obs = OrderedDict(
            position_earth_xyz_m=self._state.position_earth_xyz_m,
            velocity_earth_xyz_m_per_s=self._state.velocity_earth_xyz_m_per_s,
            yaw_pitch_roll_earth_to_body_rad=self._state.
            yaw_pitch_roll_earth_to_body_rad,
            velocity_airmass_relative_xyz_m_per_s=self._state.
            velocity_airmass_relative_xyz_m_per_s,
            t_s=t_s)

        return cast(GliderAgentObsType, obs)

    def get_info_as_dict(self):

        info = self._info
        assert info is not None

        info = dict(initial_conditions=self._initial_conditions_info,
                    time_s_without_lift=info.time_s_without_lift,
                    distance_from_core_m=info.distance_from_core_m,
                    core_position_earth_m_xy=info.core_position_earth_xy_m,
                    air_velocity_earth_m_s=info.air_velocity_earth_xyz_m_s,
                    success=info.success,
                    cutoff_reason=info.cutoff_reason,
                    aerodynamics=asdict(info.aerodynamics_info))

        return info

    @functools.cache
    @staticmethod
    def observation_space(
        simulation_box_params: SimulationBoxParameters
    ) -> gymnasium.spaces.Space:

        observation_space = gymnasium.spaces.Dict({
            "position_earth_xyz_m":
            gymnasium.spaces.Box(
                low=np.array(simulation_box_params.limit_earth_xyz_low_m,
                             dtype=np.float32),
                high=np.array(simulation_box_params.limit_earth_xyz_high_m,
                              dtype=np.float32),
                dtype=np.float32,
            ),
            "velocity_earth_xyz_m_per_s":
            gymnasium.spaces.Box(low=-50.0,
                                 high=50.0,
                                 shape=(3, ),
                                 dtype=np.float32),
            "yaw_pitch_roll_earth_to_body_rad":
            gymnasium.spaces.Box(low=-np.pi,
                                 high=np.pi,
                                 shape=(3, ),
                                 dtype=np.float32)
        })
        return observation_space

    @staticmethod
    def action_space() -> gymnasium.spaces.Space:
        action_space = gymnasium.spaces.Box(
            low=-np.pi / 2, high=np.pi / 2,
            shape=())  # only roll_earth_to_body_rad

        return action_space
