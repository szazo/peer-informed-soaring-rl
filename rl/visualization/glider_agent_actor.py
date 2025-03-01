from typing import Literal
from dataclasses import dataclass
import logging
import numpy as np
import vedo

from utils import Vector3, VectorN, VectorNx3
from utils.differential_rotator import DifferentialRotator

ActorState = Literal['not_started'] | Literal['running'] | Literal['finished']


@dataclass
class AgentParams:
    agent_type: str
    agent_name: str
    agent_training: str


@dataclass
class Trajectory:
    time_s: VectorN
    position_earth_xyz_m: VectorNx3
    velocity_earth_xyz_m_per_s: VectorNx3
    yaw_pitch_roll_earth_to_body_rad: VectorNx3
    air_velocity_earth_xyz_m_per_s: VectorNx3


@dataclass
class AgentEpisode:
    agent: AgentParams
    trajectory: Trajectory


class GliderAgentActor:

    _log: logging.Logger

    _episode: AgentEpisode
    _agent_color: np.ndarray | None

    _current_state: ActorState
    _current_frame_index: int | None

    _plane: vedo.Mesh | None
    _flagpost: vedo.Flagpost | None
    _velocity_arrow: vedo.Arrow | None

    _plane_rotator: DifferentialRotator

    def __init__(self, episode: AgentEpisode, agent_color: np.ndarray | None):

        self._log = logging.getLogger(__class__.__name__)

        self._episode = episode
        self._agent_color = agent_color

        self._current_state = 'not_started'
        self._current_frame_index = None

        self._plane_rotator = DifferentialRotator()

        self._plane = None
        self._flagpost = None
        self._velocity_arrow = None

    def step(self, current_time_s: float, plotter: vedo.Plotter) -> ActorState:

        self._log.debug('step; current_time_s=%f', current_time_s)

        # find the smallest frame index for the current time
        frame_times = self._episode.trajectory.time_s

        if current_time_s > frame_times[-1]:
            # finished, remove from the scene
            if self._current_state == 'running':

                self._log.debug('agent finished, removing; current_time_s=%f',
                                current_time_s)

                self._destroy(plotter)

            self._current_state = 'finished'
            return self._current_state

        mask = frame_times <= current_time_s
        indices = np.flatnonzero(mask)
        if indices.size == 0:
            # not yet started
            assert self._current_state == 'not_started'
            return self._current_state

        if self._current_state == 'not_started':

            self._log.debug('agent started; current_time_s=%f', current_time_s)
            self._create(plotter)
            self._current_state = 'running'

        frame_index = indices[-1]
        if self._current_frame_index == frame_index:
            # nothing to do
            return self._current_state

        # update
        self._update(frame_index=frame_index, plotter=plotter)
        self._current_frame_index = frame_index

        return self._current_state

    def _update(self, frame_index: int, plotter: vedo.Plotter):

        self._log.debug('_update; frame_index=%d', frame_index)

        trajectory = self._episode.trajectory

        current_position: Vector3 = trajectory.position_earth_xyz_m[
            frame_index]
        current_velocity: Vector3 = trajectory.velocity_earth_xyz_m_per_s[
            frame_index]
        current_yaw_pitch_roll_rad: Vector3 = trajectory.yaw_pitch_roll_earth_to_body_rad[
            frame_index]

        self._update_plane(position=current_position,
                           yaw_pitch_roll_rad=current_yaw_pitch_roll_rad,
                           plotter=plotter)
        self._update_flagpost(position=current_position,
                              velocity=current_velocity)
        self._update_velocity_arrow(position=current_position,
                                    velocity=current_velocity,
                                    plotter=plotter)

    def _update_plane(self, position: Vector3, yaw_pitch_roll_rad: Vector3,
                      plotter: vedo.Plotter):
        assert self._plane is not None

        yaw_rad = yaw_pitch_roll_rad[0]
        roll_rad = yaw_pitch_roll_rad[2]

        # calculate the differential rotation based on the target orientation
        axis, angle_rad = self._plane_rotator.rotate_to(
            np.array([yaw_rad, 0., -roll_rad]))

        around = (position[0], position[1], position[2])

        self._plane.pos(position[0], position[1], position[2])
        self._plane.rotate(angle=angle_rad, axis=axis, point=around, rad=True)

        if self._plane.trail is None:
            self._plane.add_trail(n=100)
            plotter.add(self._plane.trail)

        self._plane.update_trail()
        self._plane.update_shadows()

    def _update_flagpost(self, position: Vector3, velocity: Vector3):

        assert self._flagpost is not None
        self._flagpost.pos((position[0], position[1], position[2]))

        txt = f'{self._episode.agent.agent_name}\n' + \
            f'alt: {position[2]:.1f} m\n' + \
            f'v_z: {velocity[2]:.1f} m/s'

        self._flagpost.text(txt)

    def _update_velocity_arrow(self, position: Vector3, velocity: Vector3,
                               plotter: vedo.Plotter):
        velocity_arrow_start = position
        velocity_arrow_end = position + velocity * 5
        if self._velocity_arrow is not None:
            plotter.remove(self._velocity_arrow)
        color = 'yellow'
        if self._agent_color is not None:
            color = self._agent_color
        self._velocity_arrow = vedo.Arrow(start_pt=(velocity_arrow_start[0],
                                                    velocity_arrow_start[1],
                                                    velocity_arrow_start[2]),
                                          end_pt=(velocity_arrow_end[0],
                                                  velocity_arrow_end[1],
                                                  velocity_arrow_end[2]),
                                          s=0.1,
                                          c=color)
        plotter.add(self._velocity_arrow)

    def _create(self, plotter: vedo.Plotter):

        self._plane_rotator.reset()

        self._plane = self._create_plane()
        self._flagpost = self._create_flagpost(self._plane)

        plotter.add(self._plane, self._flagpost)

    def _create_plane(self):

        plane = vedo.Mesh(vedo.dataurl + 'cessna.vtk')

        plane_color = 'yellow'
        if self._agent_color is not None:
            plane_color = self._agent_color

        plane.color(plane_color)
        plane.scale(5.)

        plane.add_shadow('z', 0)

        return plane

    def _create_flagpost(self, plane: vedo.Mesh):

        # text_size = 0.8
        text_size = 0.6

        flagpost = plane.flagpost(
            f"Heigth:\nz={42}m",
            offset=(0., 0., 30.),
            alpha=0.5,
            c=(0, 0, 255),
            bc=(255, 0, 255),
            lw=1,
            vspacing=1.2,
            s=text_size,
            font='VictorMono',
        )
        assert flagpost is not None

        return flagpost

    def _destroy(self, plotter: vedo.Plotter):

        plotter.remove(self._plane, self._flagpost)
        self._plane = None
        self._flagpost = None

        if self._velocity_arrow is not None:
            plotter.remove(self._velocity_arrow)
            self._velocity_arrow = None
