from dataclasses import dataclass
import logging
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d.art3d import Line3D
import pygame
from utils import Vector3D
from ...air_velocity_field import AirVelocityFieldInterface
from ..agent import AgentID, GliderTrajectory
from ..simulation_box_params import SimulationBoxParameters


@dataclass
class RenderParameters:
    # currently omegaconf does not support literal:
    # https://github.com/omry/omegaconf/issues/422
    mode: str = 'rgb_array'  # human | rgb_array
    human_render_fps: int = 60


@dataclass
class LayoutParameters:
    width_height_px: tuple[float, float] | None = (1280, 720)
    title: str = 'Multi-agent simulation #42'


@dataclass
class TrajectoryPlotParameters:
    simulation_box: SimulationBoxParameters


class TrajectoryArtists:
    _trajectory_line: Line3D
    _trajectory_shadow_line: Line3D

    def __init__(self, label: str, ax: plt.Axes):

        self._trajectory_line = ax.plot([], [], [], label=label)[0]
        self._trajectory_shadow_line = ax.plot([], [], [],
                                               zdir="z",
                                               color="k",
                                               alpha=0.1)[0]

    def render(self, position_earth_xyz_m: npt.NDArray[Vector3D]):

        xy = position_earth_xyz_m[:, :2]
        z = position_earth_xyz_m[:, 2]

        self._trajectory_line.set_data(xy.T)
        self._trajectory_line.set_3d_properties(z)

        self._trajectory_shadow_line.set_data(xy.T)
        self._trajectory_shadow_line.set_3d_properties(0)

    def clear(self):
        self._trajectory_line.remove()
        self._trajectory_shadow_line.remove()


@dataclass
class TimeSeriesParameters:
    # if set, the plot will contain only y values for the last 'x_recent_range' x values
    x_recent_range: float | None = 30.


@dataclass
class Line2DPlotParameters:
    x_label: str
    y_label: str


@dataclass
class Line2DPlotData:
    x_data: npt.ArrayLike
    y_data: npt.ArrayLike


class Line2DPlotBase:

    _params: Line2DPlotParameters
    _x_recent_range: float | None

    _ax: plt.Axes

    _line_artists: dict[AgentID, plt.Line2D]

    def __init__(self,
                 params: Line2DPlotParameters,
                 x_recent_range: float | None = None):
        self._params = params
        self._x_recent_range = x_recent_range
        self._line_artists = {}

    def create(self, ax: plt.Axes):

        ax.set_xlabel(self._params.x_label)
        ax.set_ylabel(self._params.y_label)

        self._ax = ax

    def render(self, data: dict[AgentID, Line2DPlotData]):

        for agent_id, data_item in data.items():
            if agent_id not in self._line_artists:
                self._line_artists[agent_id] = self._ax.plot([], [],
                                                             label=agent_id)[0]

            line = self._line_artists[agent_id]

            x_data = data_item.x_data
            y_data = data_item.y_data
            if self._x_recent_range is not None and len(x_data) > 0:
                last_x_value = x_data[-1]
                indices = x_data > (last_x_value - self._x_recent_range)
                x_data = x_data[indices]
                y_data = y_data[indices]

            line.set_data(x_data, y_data)

        self._ax.relim()
        self._ax.autoscale_view(True, True, True)

        # remove dead agents
        current_trajectory_agent_ids = list(self._line_artists.keys())
        for agent_id in current_trajectory_agent_ids:
            if agent_id not in data:
                line = self._line_artists[agent_id]
                line.remove()
                del self._line_artists[agent_id]


class VerticalVelocityPlot(Line2DPlotBase):

    def __init__(self, params: TimeSeriesParameters):
        super().__init__(params=Line2DPlotParameters(
            x_label='time (s)', y_label='vertical velocity (m/s)'),
                         x_recent_range=params.x_recent_range)

    def render(self, trajectories: dict[AgentID, GliderTrajectory]):

        data: dict[AgentID, Line2DPlotData] = {
            agent_id:
            Line2DPlotData(
                x_data=agent_trajectory.time_s,
                y_data=agent_trajectory.velocity_earth_xyz_m_per_s[:, 2])
            for agent_id, agent_trajectory in trajectories.items()
        }

        super().render(data=data)


class AltitudePlot(Line2DPlotBase):

    def __init__(self, params: TimeSeriesParameters):
        super().__init__(params=Line2DPlotParameters(x_label='time (s)',
                                                     y_label='altitude (m)'),
                         x_recent_range=params.x_recent_range)

    def render(self, trajectories: dict[AgentID, GliderTrajectory]):

        data: dict[AgentID, Line2DPlotData] = {
            agent_id:
            Line2DPlotData(x_data=agent_trajectory.time_s,
                           y_data=agent_trajectory.position_earth_xyz_m[:, 2])
            for agent_id, agent_trajectory in trajectories.items()
        }

        super().render(data=data)


class RollPlot(Line2DPlotBase):

    def __init__(self, params: TimeSeriesParameters):
        super().__init__(params=Line2DPlotParameters(
            x_label='time (s)', y_label='roll (bank) angle (degrees)'),
                         x_recent_range=params.x_recent_range)

    def render(self, trajectories: dict[AgentID, GliderTrajectory]):

        data: dict[AgentID, Line2DPlotData] = {
            agent_id:
            Line2DPlotData(
                x_data=agent_trajectory.time_s,
                y_data=np.rad2deg(
                    agent_trajectory.yaw_pitch_roll_earth_to_body_rad[:, 2]))
            for agent_id, agent_trajectory in trajectories.items()
        }

        super().render(data=data)


@dataclass
class ThermalCore3DPlotParameters:
    resolution_m: int = 30


class ThermalCore3DPlot:

    _params: ThermalCore3DPlotParameters
    _simulation_box_params: SimulationBoxParameters
    _air_velocity_field: AirVelocityFieldInterface

    _thermal_core_line: Line3D | None
    _thermal_core_shadow_line: Line3D | None

    def __init__(self, params: ThermalCore3DPlotParameters,
                 simulation_box_params: SimulationBoxParameters,
                 air_velocity_field: AirVelocityFieldInterface):

        self._params = params
        self._simulation_box_params = simulation_box_params
        self._air_velocity_field = air_velocity_field

    def create(self, ax: plt.Axes):
        self._thermal_core_line = ax.plot([], [], [], color='goldenrod')[0]
        self._thermal_core_shadow_line = ax.plot([], [], [],
                                                 zdir="z",
                                                 color="k",
                                                 alpha=0.1)[0]

    def render(self, time_s: float):

        box = self._simulation_box_params

        z_min = box.limit_earth_xyz_low_m[2]
        z_max = box.limit_earth_xyz_high_m[2]

        z_space = np.linspace(z_min, z_max, self._params.resolution_m)
        thermal_core_xy_m = self._air_velocity_field.get_thermal_core(
            z_earth_m=z_space, t_s=time_s)

        self._thermal_core_line.set_data(thermal_core_xy_m.T)
        self._thermal_core_line.set_3d_properties(z_space)

        self._thermal_core_shadow_line.set_data(thermal_core_xy_m.T)
        self._thermal_core_shadow_line.set_3d_properties(0)


class TrajectoryPlot:

    _params: TrajectoryPlotParameters
    _rc_style_sheet: dict
    _trajectory_ax: plt.Axes

    _trajectory_artists: dict[AgentID, TrajectoryArtists]

    def __init__(self, params: TrajectoryPlotParameters):
        self._params = params
        self._trajectory_artists = {}

    def _initialize_style_sheet(self, rc_style_sheet: dict | None):
        default_stylesheet = {
            'figure.titlesize': 24,
            'axes.labelsize': 19,
            'axes.labelpad': 18,
            'axes.xtick.labelsize': 10,
            'axes.ytick.labelsize': 10,
            'axes.ztick.labelsize': 10
        }

        self._rc_style_sheet = rc_style_sheet if not None else default_stylesheet

    def create(self, ax: plt.Axes):

        box = self._params.simulation_box
        ax.set(xlim3d=(box.limit_earth_xyz_low_m[0],
                       box.limit_earth_xyz_high_m[0]))
        ax.set(ylim3d=(box.limit_earth_xyz_low_m[1],
                       box.limit_earth_xyz_high_m[1]))
        ax.set(zlim3d=(box.limit_earth_xyz_low_m[2],
                       box.limit_earth_xyz_high_m[2]))

        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")

        self._trajectory_ax = ax

    def render(self, trajectories: dict[AgentID, GliderTrajectory]):

        trajectory_count = 0
        for agent_id, trajectory in trajectories.items():

            # find the line for the agent id
            if agent_id not in self._trajectory_artists:
                # create the line plot for the agent
                self._trajectory_artists[agent_id] = TrajectoryArtists(
                    label=agent_id, ax=self._trajectory_ax)

            trajectory_artists = self._trajectory_artists[agent_id]

            trajectory_artists.render(trajectory.position_earth_xyz_m[-100 *
                                                                      10:, :])

            trajectory_count += 1

        # remove dead trajectories
        current_trajectory_agent_ids = list(self._trajectory_artists.keys())
        for agent_id in current_trajectory_agent_ids:
            if agent_id not in trajectories:
                self._trajectory_artists[agent_id].clear()
                del self._trajectory_artists[agent_id]

        if trajectory_count > 0:
            self._trajectory_ax.legend()


class PlotBase:
    pass


class LayoutPlotBase:
    pass


class GridLayoutPlot(LayoutPlotBase):

    _trajectory_plot: TrajectoryPlot
    _thermal_core_plot: ThermalCore3DPlot
    _vertical_velocity_plot: VerticalVelocityPlot
    _altitude_plot: AltitudePlot
    _roll_plot: RollPlot

    def __init__(
        self,
        trajectory_plot: TrajectoryPlot,
        thermal_core_plot: ThermalCore3DPlot,
        time_series_params: TimeSeriesParameters = TimeSeriesParameters()):
        self._trajectory_plot = trajectory_plot
        self._thermal_core_plot = thermal_core_plot
        self._vertical_velocity_plot = VerticalVelocityPlot(
            params=time_series_params)
        self._altitude_plot = AltitudePlot(params=time_series_params)
        self._roll_plot = RollPlot(params=time_series_params)

    def create(self, figure: plt.Figure):

        grid_shape = (3, 3)
        # 3d view
        threed_ax = plt.subplot2grid(shape=grid_shape,
                                     loc=(0, 0),
                                     rowspan=3,
                                     colspan=2,
                                     projection='3d',
                                     fig=figure)

        # trajectory
        self._trajectory_plot.create(threed_ax)

        # thermal core
        self._thermal_core_plot.create(threed_ax)

        # vertical velocity
        vertical_velocity_ax = plt.subplot2grid(grid_shape, (0, 2), fig=figure)
        self._vertical_velocity_plot.create(vertical_velocity_ax)

        # altitude
        altitude_ax = plt.subplot2grid(grid_shape, (1, 2), fig=figure)
        self._altitude_plot.create(altitude_ax)

        # roll
        roll_ax = plt.subplot2grid(grid_shape, (2, 2), fig=figure)
        self._roll_plot.create(roll_ax)

    def render(self, trajectories: dict[AgentID, GliderTrajectory],
               time_s: float):
        self._trajectory_plot.render(trajectories=trajectories)
        self._thermal_core_plot.render(time_s=time_s)
        self._vertical_velocity_plot.render(trajectories=trajectories)
        self._altitude_plot.render(trajectories=trajectories)
        self._roll_plot.render(trajectories=trajectories)


class MultigliderVisualization:

    _surface: pygame.Surface | None
    _render_params: RenderParameters
    _layout_params: LayoutParameters
    _rc_style_sheet: dict

    _plot: GridLayoutPlot

    _figure: plt.Figure | None
    _canvas: plt.FigureCanvasBase | None
    _clock: pygame.time.Clock | None

    def __init__(self,
                 render_params: RenderParameters,
                 layout_params: LayoutParameters,
                 plot: GridLayoutPlot,
                 rc_style_sheet: dict | None = None):

        self._log = logging.getLogger(__class__.__name__)

        self._surface = None
        self._render_params = render_params
        self._layout_params = layout_params
        self._plot = plot

        assert (render_params.mode == 'human' and layout_params.width_height_px is not None) or \
            render_params.mode == 'rgb_array'

        if render_params.mode == 'human':
            self._clock = pygame.time.Clock()

        self._initialize_style_sheet(rc_style_sheet)

        self._create()

    def _initialize_style_sheet(self, rc_style_sheet: dict | None):
        default_stylesheet = {
            'figure.titlesize': 24,
            'axes.labelsize': 19,
            'axes.labelpad': 18,
            'xtick.labelsize': 16,
            'ytick.labelsize': 16,
        }

        self._rc_style_sheet = rc_style_sheet if rc_style_sheet is not None else default_stylesheet

    def _create(self):

        fig = self._create_figure()

        # attach the canvas to the figure
        canvas = FigureCanvasAgg(fig)

        # create the plot
        self._plot.create(figure=fig)

        self._figure = fig
        self._canvas = canvas

        # prevent displaying plot in jupyter notebook
        plt.close()

    def _create_figure(self):

        px = 1 / plt.rcParams['figure.dpi']
        width_px, height_px = self._layout_params.width_height_px
        fig = plt.figure(figsize=(width_px * px, height_px * px),
                         layout='constrained')
        fig.suptitle(t=self._layout_params.title)

        return fig

    def render(self, trajectories: dict[AgentID, GliderTrajectory],
               time_s: float) -> None | np.ndarray:

        self._log.debug('render; trajectories=%s', trajectories)

        render_mode = self._render_params.mode
        if self._surface is None:

            pygame.init()

            if render_mode == 'human':
                self._surface = pygame.display.set_mode(
                    self._layout_params.width_height_px)
                pygame.display.set_caption(self._layout_params.title)
            else:
                self._surface = pygame.Surface(
                    self._layout_params.width_height_px)

        # update the plot
        self._plot.render(trajectories=trajectories, time_s=time_s)

        # draw the plot
        self._canvas.draw()

        # create image
        rgba_buffer = self._canvas.buffer_rgba()
        canvas_size = self._canvas.get_width_height()
        image = pygame.image.frombuffer(rgba_buffer, canvas_size, 'RGBA')

        # draw the image to the surface
        self._surface.fill((255, 255, 255))
        self._surface.blit(image, (0, 0))

        if render_mode == 'human':
            pygame.display.update()
            self._clock.tick(self._render_params.human_render_fps)
        elif render_mode == 'rgb_array':
            return np.transpose(np.array(
                pygame.surfarray.pixels3d(self._surface)),
                                axes=(1, 0, 2))

    def close(self):
        if self._surface is not None:
            self._surface = None
        plt.close(self._figure)
