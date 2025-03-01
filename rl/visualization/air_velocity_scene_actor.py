from typing import Literal
import logging
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from dataclasses import dataclass, asdict
import numpy as np
import vedo
from vedo.colors import colors
import matplotlib as mpl
from omegaconf import MISSING
import hydra

from utils import Vector3, RandomGeneratorState
from thermal.api import AirVelocityFieldConfigBase, AirVelocityFieldInterface
from env.glider.base import SimulationBoxParameters

from .glider_agent_actor import AgentEpisode, GliderAgentActor


@dataclass(kw_only=True)
class SceneParams:
    air_velocity_field: AirVelocityFieldConfigBase = MISSING
    simulation_box: SimulationBoxParameters
    air_velocity_grid_spacing_m: float
    minimum_vertical_velocity_m_per_s: float | None
    # max_agent_color_num: int
    colormap_name: str
    agent_color_index_map: dict[str, int]


@dataclass
class ThermalParams:
    thermal_name: str
    thermal_params: str
    thermal_random_state: RandomGeneratorState


@dataclass
class Episode:
    agents: list[AgentEpisode]


@dataclass
class Scene:
    thermal: ThermalParams
    episodes: list[Episode]


SceneState = Literal['not_started'] | Literal['running'] | Literal['finished']


class AirVelocitySceneActor:

    _log: logging.Logger

    _scene: Scene
    _params: SceneParams

    _air_velocity_field_volume: vedo.Volume | None
    _timer_text: vedo.CornerAnnotation | None

    _current_state: SceneState
    _last_time_s: float | None

    _current_episode_index: int

    _current_agents: list[GliderAgentActor]
    _agent_color_map: dict[str, np.ndarray]

    def __init__(self, scene: Scene, params: SceneParams, start_episode: int,
                 start_time_s: float):

        self._log = logging.getLogger(__class__.__name__)

        self._scene = scene
        self._params = params

        self._air_velocity_field_volume = None

        self._current_state = 'not_started'
        if start_time_s > 0:
            self._last_time_s = start_time_s
        else:
            self._last_time_s = None

        if start_episode >= len(scene.episodes):
            raise ValueError(
                f'invalid start_episode index {start_episode}; episode count: {len(scene.episodes)}'
            )

        self._current_episode_index = start_episode - 1
        self._current_agents = []

        self._agent_color_map = self._create_agent_name_color_mapping(
            self._params.colormap_name, self._params.agent_color_index_map)

    def create(self, plotter: vedo.Plotter):

        air_velocity_field = self._create_air_velocity_field()
        self._air_velocity_field_volume = self._create_air_velocity_field_volume(
            air_velocity_field)

        self._timer_text = vedo.CornerAnnotation()

        plotter.add(self._air_velocity_field_volume, self._timer_text)

        state, _ = self.step(dt_s=0., plotter=plotter)
        return state

    def destroy(self, plotter: vedo.Plotter):
        if self._air_velocity_field_volume is None:
            return

        plotter.remove(self._air_velocity_field_volume)
        self._air_velocity_field_volume = None

    def step(self, dt_s: float,
             plotter: vedo.Plotter) -> tuple[SceneState, str]:

        if self._current_state == 'not_started':
            # initialize the episode
            self._select_next_episode()

        if self._current_state == 'finished':
            self._log.debug('scene is finished')
            return self._current_state, 'finished'

        if self._last_time_s is None:
            current_time_s = 0.
        else:
            current_time_s = self._last_time_s + dt_s

        # assert self._timer_text is not None
        # self._timer_text.text(f'time: {current_time_s:.2f}s')

        self._log.debug('step; current_time_s=%f;dt_s=%f', current_time_s,
                        dt_s)

        assert len(self._current_agents) > 0, 'there is no current agent'

        all_agents_finished = True
        for agent in self._current_agents:
            agent_state = agent.step(current_time_s=current_time_s,
                                     plotter=plotter)
            if agent_state != 'finished':
                all_agents_finished = False

        self._last_time_s = current_time_s

        if all_agents_finished:
            # create the next agent
            self._select_next_episode()

        if self._current_state == 'finished':
            self._log.debug('scene is finished')
            return self._current_state, 'finished'

        annotation = f'episode: {self._current_episode_index}; time: {current_time_s:.2f}s'

        return self._current_state, annotation

    def _select_next_episode(self):
        self._current_episode_index += 1
        if len(self._scene.episodes) > self._current_episode_index:
            self._log.debug('playing episode %d', self._current_episode_index)

            episode = self._scene.episodes[self._current_episode_index]
            self._current_agents = self._create_agent_actors(episode)
            self._current_state = 'running'
            self._last_time_s = None
        else:
            # no more episodes
            self._log.debug('no more episodes, finish the scene')
            self._current_state = 'finished'

    def _create_agent_actors(self, episode: Episode) -> list[GliderAgentActor]:

        agent_actors = []

        for i, agent_episode in enumerate(episode.agents):

            agent_color = None
            agent_name = agent_episode.agent.agent_name
            if agent_name in self._agent_color_map:
                agent_color = self._agent_color_map[agent_name]

            agent_actor = GliderAgentActor(agent_episode,
                                           agent_color=agent_color)
            agent_actors.append(agent_actor)

        return agent_actors

    def _create_agent_name_color_mapping(
            self, colormap_name: str,
            agent_color_index_map: dict[str, int]) -> dict[str, np.ndarray]:

        colormap = mpl.colormaps[colormap_name]

        if isinstance(colormap, LinearSegmentedColormap):
            max_color_index = np.max(list(agent_color_index_map.values()))
            index_color_map = self._generate_discrete_colormap_from_linear(
                colormap, max_color_index + 1)
        elif isinstance(colormap, ListedColormap):
            index_color_map = [np.array(c) for c in colormap.colors]
        else:
            assert False, f'unsupported color map: {colormap}'

        result = {
            agent_id: index_color_map[color_index % len(index_color_map)]
            for agent_id, color_index in agent_color_index_map.items()
        }

        return result

    def _generate_discrete_colormap_from_linear(
            self, colormap: LinearSegmentedColormap,
            max_colors: int) -> list[np.ndarray]:

        colors = []
        for i in range(max_colors):
            color = vedo.color_map(i, colormap, vmin=0, vmax=max_colors - 1)
            colors.append(color)

        return colors

    def _create_air_velocity_field_volume(
            self,
            air_velocity_field: AirVelocityFieldInterface) -> vedo.Volume:

        self._log.debug('creating air velocity field volume...')

        box = self._params.simulation_box
        assert box.box_size is not None
        assert box.limit_earth_xyz_low_m is not None and box.limit_earth_xyz_high_m is not None

        resolution = self._calculate_resolution(
            np.array(box.box_size),
            spacing_m=self._params.air_velocity_grid_spacing_m)

        self._log.debug('resolution=%s', resolution)

        # create the grid coordinates
        X, Y, Z = np.meshgrid(
            np.linspace(box.limit_earth_xyz_low_m[0],
                        box.limit_earth_xyz_high_m[0],
                        num=resolution[0]),
            np.linspace(box.limit_earth_xyz_low_m[1],
                        box.limit_earth_xyz_high_m[1],
                        num=resolution[1]),
            np.linspace(box.limit_earth_xyz_low_m[2],
                        box.limit_earth_xyz_high_m[2],
                        num=resolution[2]))
        # coords = [item.ravel() for item in (X,Y,Z)]
        x = X.ravel()
        y = Y.ravel()
        z = Z.ravel()

        # get the air velocities
        uvw, _ = air_velocity_field.get_velocity(x_earth_m=x,
                                                 y_earth_m=y,
                                                 z_earth_m=z,
                                                 t_s=0)
        uvw = uvw.T

        # create the volume points
        xyz = np.stack((x, y, z), axis=1)

        if self._params.minimum_vertical_velocity_m_per_s is not None:
            mask = uvw[:, 2] > self._params.minimum_vertical_velocity_m_per_s

            xyz = xyz[mask, :]
            uvw = uvw[mask, :]

        volume_points = vedo.Points(
            xyz, r=int(self._params.air_velocity_grid_spacing_m))
        volume_points.pointdata['vertical_velocity'] = uvw[:, 2]

        # create the volume
        volume = volume_points.tovolume(kernel='shepard', n=4, dims=resolution)

        color_map = self._create_thermal_colormap(cmap_min=-2.0, cmap_max=5.0)
        cols = vedo.color_map(range(256), color_map)
        volume.color(cols, vmin=-2.0, vmax=5.0)
        volume.alpha([(-2., 0), (0., 0.01), (0.5, 0.05), (1.0, 0.8),
                      (2.0, 0.97)])
        volume.alpha_unit(300)

        volume.add_scalarbar(title='thermal vertical velocity (m/s)')

        self._log.debug('air velocity field volume created')

        return volume

    def _create_thermal_colormap(self, cmap_min: float, cmap_max: float):
        colors = ['#a34d93', '#c4deff', '#ffff3d', '#ff3333', '#9c0000']

        cmap_whole = cmap_max - cmap_min
        zero_pos = np.abs(cmap_min) / cmap_whole

        nodes = [0.0, zero_pos, zero_pos + (1 - zero_pos) / 3, 0.8, 1.0]
        cmap = list(zip(nodes, colors))

        #return cmap
        #return vedo.build_lut(cmap, vmin=0, vmax=1, interpolate=True)

        return LinearSegmentedColormap.from_list('thermal_cmap', cmap)

    def _calculate_resolution(self, box_size: Vector3,
                              spacing_m: float) -> Vector3:
        return (box_size / spacing_m + 1).astype(int)

    def _create_air_velocity_field(self) -> AirVelocityFieldInterface:

        self._log.debug('creating air velocity field...')

        # REVIEW: some method to pass random state in other ways
        params = {
            **asdict(self._params.air_velocity_field), "random_state":
            self._scene.thermal.thermal_random_state
        }

        air_velocity_field: AirVelocityFieldInterface = hydra.utils.instantiate(
            params, _convert_='object')

        self._log.debug('air velocity field created; air_velocity_field=%s',
                        air_velocity_field)

        air_velocity_field.reset()

        return air_velocity_field

        return air_velocity_field
