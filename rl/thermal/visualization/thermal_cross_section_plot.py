from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.contour import ContourSet
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable, axes_size
from utils import Vector3D, VectorN
from ..api import AirVelocityFieldInterface
from .thermal_plot_data_serializer import (Box, Box2D, VerticalThermalPlotData,
                                           HorizontalThermalPlotData)


@dataclass
class Box2DParams:
    sample_altitude_m: float
    box_size_xy_m: tuple[float, float] = (250, 250)
    center_xy_m: tuple[float, float] | None = None
    spacing_m: float = 10


@dataclass
class HorizontalPlotParameters:
    major_locator_gap_m: float = 25
    minor_locator_gap_count: int | None = 2


@dataclass
class HorizontalCrossSectionParameters:
    box: Box2DParams
    plot: HorizontalPlotParameters


@dataclass
class Box3DParams:
    altitude_range_m: tuple[float, float] | None = None
    box_size_xyz_m: tuple[float, float, float] | None = (260, 260, 1500)
    center_xyz_m: tuple[float, float, float] | None = (0, 0, 750)
    spacing_m: float = 10


@dataclass
class VerticalPlotParameters:
    vertical_major_locator_gap_m: float = 100
    vertical_minor_locator_gap_count: int | None = 2
    horizontal_major_locator_gap_m: float = 100
    horizontal_minor_locator_gap_count: int | None = 2
    # hydra does not support Literal enums now
    #projection: Literal['XZ', 'YZ'] = 'XZ'
    projection: str = 'XZ'
    #projection_type: Literal['max', 'follow_core'] = 'max'
    projection_type: str = 'follow_core'


@dataclass
class VerticalCrossSectionParameters:
    box: Box3DParams
    plot: VerticalPlotParameters


def create_thermal_colormap(cmap_min: float, cmap_max: float):
    colors = ['#a34d93', '#c4deff', '#ffff3d', '#ff3333', '#9c0000']

    cmap_whole = cmap_max - cmap_min
    zero_pos = np.abs(cmap_min) / cmap_whole

    nodes = [0.0, zero_pos, zero_pos + (1 - zero_pos) / 3, 0.8, 1.0]
    cmap = LinearSegmentedColormap.from_list('thermal_cmap',
                                             list(zip(nodes, colors)))
    return cmap


cmap_min = -2.0
cmap_max = 5.0
thermal_cmap = create_thermal_colormap(cmap_min=cmap_min, cmap_max=cmap_max)


@dataclass
class ColorStyleParams:
    color_levels: VectorN = field(
        default_factory=lambda: np.linspace(cmap_min, cmap_max, 60))
    colormap_vmin: float | None = cmap_min
    colormap_vmax: float | None = cmap_max
    colormap = thermal_cmap  #cmocean.cm.thermal # mpl.cm.coolwarm


@dataclass
class ThermalCrossSectionPlotParameters:
    horizontal: HorizontalCrossSectionParameters
    vertical: VerticalCrossSectionParameters


class ThermalCrossSectionPlot:

    _color_style: ColorStyleParams

    def __init__(self):

        self._color_style = ColorStyleParams()

    def plot(self,
             params: ThermalCrossSectionPlotParameters,
             field: AirVelocityFieldInterface,
             t_s: float,
             show_horizontal: bool,
             show_vertical: bool,
             figure: Figure,
             axes: list[Axes],
             show_w_min_max: bool,
             show_title: bool = True):

        ax_index = 0
        plots = []

        horizontal_data = None
        if show_horizontal:
            # horizontal
            horizontal_params = params.horizontal
            horizontal_data = self._calculate_horizontal_thermal_plot_data(
                params=horizontal_params.box, field=field, t_s=t_s)
            plots.append(
                self._plot_horizontal_cross_section(
                    ax=axes[ax_index],
                    data=horizontal_data,
                    params_plot=horizontal_params.plot,
                    show_title=show_title))
            ax_index += 1

        vertical_data = None
        if show_vertical:
            # vertical
            vertical_params = params.vertical
            vertical_data = self._calculate_vertical_thermal_plot_data(
                params=vertical_params, field=field, t_s=t_s)

            plots.append(
                self.plot_vertical_cross_section(ax=axes[ax_index],
                                                 data=vertical_data,
                                                 params=vertical_params.plot,
                                                 show_w_min_max=show_w_min_max,
                                                 show_title=show_title))
            ax_index += 1

        self.create_colorbar(fig=figure, ax=axes[-1], plot=plots[-1])

        return horizontal_data, vertical_data

    def plot_vertical_cross_section(self,
                                    ax: Axes,
                                    data: VerticalThermalPlotData,
                                    params: VerticalPlotParameters,
                                    x_lim: tuple[float, float] | None = None,
                                    y_lim: tuple[float, float] | None = None,
                                    show_w_min_max: bool = True,
                                    show_title: bool = True) -> ContourSet:

        w = data.w
        box = data.box

        w_min = np.min(w)
        w_max = np.max(w)

        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)

        if show_w_min_max:
            ax.text(
                0.83,
                0.94,
                '$w_{min}: %.1f \\frac{m}{s}$\n$w_{max}: %.1f \\frac{m}{s}$' %
                (w_min, w_max),
                transform=ax.transAxes,
                bbox={
                    'pad': 4,
                    'facecolor': 'white',
                    'edgecolor': 'black',
                    'alpha': 0.5
                },
                color='black',
                ha='left',
                fontfamily='monospace')

        x_space, y_space, z_space = self._3d_box_to_spaces(box)

        x_label = ''
        projection_label = ''

        if params.projection == 'XZ':

            X, Z = np.meshgrid(x_space, z_space, indexing='ij')

            w = np.amax(w, axis=1)
            x_label = 'x (m)'
            projection_label = 'Y'

        elif params.projection == 'YZ':
            X, Z = np.meshgrid(y_space, z_space, indexing='ij')

            w = np.amax(w, axis=0)
            x_label = 'y (m)'
            projection_label = 'X'
        else:
            assert False, f'invalid projection: {params.projection}'

        cp = ax.contourf(X,
                         Z,
                         w,
                         vmin=self._color_style.colormap_vmin,
                         vmax=self._color_style.colormap_vmax,
                         cmap=self._color_style.colormap,
                         levels=self._color_style.color_levels)

        self._svg_compatible_contour(cp)

        if params.projection_type == 'max':
            title = f'Vertical cross section (max projection of {projection_label} axis)'
        elif params.projection_type == 'follow_core':
            title = f'Vertical cross section (follow core along the {projection_label} axis)'
        else:
            assert False

        if show_title:
            ax.set_title(title)

        ax.xaxis.set_major_locator(
            MultipleLocator(params.horizontal_major_locator_gap_m))
        ax.xaxis.set_minor_locator(
            AutoMinorLocator(params.horizontal_minor_locator_gap_count))
        ax.set_xlabel(x_label)

        ax.yaxis.set_major_locator(
            MultipleLocator(params.vertical_major_locator_gap_m))
        ax.yaxis.set_minor_locator(
            AutoMinorLocator(params.vertical_minor_locator_gap_count))
        ax.set_ylabel('z (m)')

        return cp

    def _calculate_vertical_thermal_plot_data(
            self, params: VerticalCrossSectionParameters,
            field: AirVelocityFieldInterface,
            t_s: float) -> VerticalThermalPlotData:

        box = self._determine_3d_roi_box(params=params.box,
                                         field=field,
                                         t_s=t_s)
        plot_params = params.plot

        assert plot_params.projection_type == 'follow_core' or plot_params.projection_type == 'max', f'invalid projection_type: {plot_params.projection_type}'
        if plot_params.projection_type == 'follow_core':
            x_space, y_space, z_space = self._3d_box_to_spaces(box)

            # query the core position along the z space
            core_xy = field.get_thermal_core(z_space, t_s=t_s)
            if plot_params.projection == 'XZ':
                # create the meshgrid in a way that the y dimension size is 1
                X, Y, Z = np.meshgrid(x_space,
                                      np.array([0.]),
                                      z_space,
                                      indexing='ij')

                # set the core's Y position for the meshgrid's Y position
                Y[:, 0, :] = core_xy[:, 1]
            elif plot_params.projection == 'YZ':
                # create the meshgrid in a way that the x dimension size is 1
                X, Y, Z = np.meshgrid(np.array([0.]),
                                      y_space,
                                      z_space,
                                      indexing='ij')

                # set the core's X position for the meshgrid's X position
                X[0, :, :] = core_xy[:, 0]
            else:
                assert False

        else:
            x_space, y_space, z_space = self._3d_box_to_spaces(box)
            X, Y, Z = np.meshgrid(x_space, y_space, z_space, indexing='ij')

        u_v_w_m_per_s, _ = field.get_velocity(t_s=t_s,
                                              x_earth_m=X,
                                              y_earth_m=Y,
                                              z_earth_m=Z)

        u, v, w = \
            u_v_w_m_per_s[0], \
            u_v_w_m_per_s[1], \
            u_v_w_m_per_s[2]

        return VerticalThermalPlotData(u=u, v=v, w=w, box=box)

    def _3d_box_to_spaces(self, box: Box) -> tuple[VectorN, VectorN, VectorN]:
        x_space = np.linspace(box.range_low_xyz_m[0], box.range_high_xyz_m[0],
                              box.resolution[0])
        y_space = np.linspace(box.range_low_xyz_m[1], box.range_high_xyz_m[1],
                              box.resolution[1])
        z_space = np.linspace(box.range_low_xyz_m[2], box.range_high_xyz_m[2],
                              box.resolution[2])

        return x_space, y_space, z_space

    def _determine_3d_roi_box(self, params: Box3DParams,
                              field: AirVelocityFieldInterface,
                              t_s: float) -> Box:

        def calculate_resolution(box_size: Vector3D,
                                 spacing_m: float) -> Vector3D:
            return (box_size / spacing_m + 1).astype(int)

        if params.altitude_range_m is not None:

            assert params.box_size_xyz_m is None, "both 'altitude_range_m' and 'box_size_xyz_m' is not supported"
            assert params.center_xyz_m is None, "both 'altitude_range_m' and 'center_xyz_m' is not supported"

            # determine box size based on altitude range
            core_xy = field.get_thermal_core(np.array(params.altitude_range_m),
                                             t_s=t_s)
            core_x_minmax = (np.min(core_xy[:, 0]), np.max(core_xy[:, 0]))
            core_y_minmax = (np.min(core_xy[:, 1]), np.max(core_xy[:, 1]))

            range_low_xyz_m = np.array([
                core_x_minmax[0], core_y_minmax[0], params.altitude_range_m[0]
            ])
            range_high_xyz_m = np.array([
                core_x_minmax[1], core_y_minmax[1], params.altitude_range_m[1]
            ])

            box_size = np.abs(range_high_xyz_m - range_low_xyz_m)
            resolution = calculate_resolution(box_size, params.spacing_m)
        else:
            # use the provided box parameters
            assert params.box_size_xyz_m is not None, 'box_size_xyz_m is required'
            assert params.center_xyz_m is not None, 'center_xyz_m is required'

            half_x = params.box_size_xyz_m[0] / 2
            half_y = params.box_size_xyz_m[1] / 2
            half_z = params.box_size_xyz_m[2] / 2

            center_x = params.center_xyz_m[0]
            center_y = params.center_xyz_m[1]
            center_z = params.center_xyz_m[2]

            resolution = calculate_resolution(np.array(params.box_size_xyz_m),
                                              params.spacing_m)

            range_low_xyz_m = np.array(
                [center_x - half_x, center_y - half_y, center_z - half_z])
            range_high_xyz_m = np.array(
                [center_x + half_x, center_y + half_y, center_z + half_z])

        return Box(range_low_xyz_m=range_low_xyz_m,
                   range_high_xyz_m=range_high_xyz_m,
                   resolution=resolution)

    def _calculate_horizontal_thermal_plot_data(
            self, params: Box2DParams, field: AirVelocityFieldInterface,
            t_s: float) -> HorizontalThermalPlotData:

        box = self._determine_2d_roi_box(params=params, field=field, t_s=t_s)

        x_space, y_space = self._2d_box_to_spaces(box)

        X, Y = np.meshgrid(x_space, y_space)
        z = params.sample_altitude_m

        u_v_w_m_per_s, _ = field.get_velocity(t_s=t_s,
                                              x_earth_m=X,
                                              y_earth_m=Y,
                                              z_earth_m=z)

        u, v, w = \
            u_v_w_m_per_s[0], \
            u_v_w_m_per_s[1], \
            u_v_w_m_per_s[2]

        return HorizontalThermalPlotData(u=u, v=v, w=w, box=box)

    def _determine_2d_roi_box(self, params: Box2DParams,
                              field: AirVelocityFieldInterface,
                              t_s: float) -> Box2D:
        half_x = params.box_size_xy_m[0] / 2
        half_y = params.box_size_xy_m[1] / 2
        resolution = (np.array(params.box_size_xy_m) / params.spacing_m +
                      1).astype(int)

        if params.center_xy_m is not None:
            center_x = params.center_xy_m[0]
            center_y = params.center_xy_m[1]
        else:
            # query the center based on the thermal core
            core_xy = field.get_thermal_core(params.sample_altitude_m, t_s=t_s)
            center_x = core_xy[0]
            center_y = core_xy[1]

        range_low_xy_m = np.array([center_x - half_x, center_y - half_y])
        range_high_xy_m = np.array([center_x + half_x, center_y + half_y])

        return Box2D(range_low_xy_m=range_low_xy_m,
                     range_high_xy_m=range_high_xy_m,
                     resolution_xy_m=resolution,
                     z_m=params.sample_altitude_m)

    def _2d_box_to_spaces(self, box: Box2D) -> tuple[VectorN, VectorN]:

        x_space = np.linspace(box.range_low_xy_m[0], box.range_high_xy_m[0],
                              box.resolution_xy_m[0])
        y_space = np.linspace(box.range_low_xy_m[1], box.range_high_xy_m[1],
                              box.resolution_xy_m[1])
        return x_space, y_space

    def _plot_horizontal_cross_section(self,
                                       ax: Axes,
                                       data: HorizontalThermalPlotData,
                                       params_plot: HorizontalPlotParameters,
                                       show_title: bool = True):

        w = data.w
        box = data.box

        x_space, y_space = self._2d_box_to_spaces(box)
        X, Y = np.meshgrid(x_space, y_space)

        cp = ax.contourf(X,
                         Y,
                         w,
                         vmin=self._color_style.colormap_vmin,
                         vmax=self._color_style.colormap_vmax,
                         cmap=self._color_style.colormap,
                         levels=self._color_style.color_levels)
        self._svg_compatible_contour(cp)

        if show_title:
            ax.set_title(
                f'Horizontal cross section at $z={data.box.z_m:.1f} m$')

        ax.xaxis.set_major_locator(
            MultipleLocator(params_plot.major_locator_gap_m))
        ax.xaxis.set_minor_locator(
            AutoMinorLocator(params_plot.minor_locator_gap_count))
        ax.set_xlabel('x (m)')

        ax.yaxis.set_major_locator(
            MultipleLocator(params_plot.major_locator_gap_m))
        ax.yaxis.set_minor_locator(
            AutoMinorLocator(params_plot.minor_locator_gap_count))
        ax.set_ylabel('y (m)')
        ax.tick_params(pad=1)

        # ax1.quiver(X_horizontal, Y_horizontal, u_horizontal, v_horizontal)
        # ax.set_aspect('equal', adjustable='box')
        # ax.relim()
        # ax.autoscale_view()

        return cp

    def create_colorbar(self,
                        fig: Figure,
                        ax: Axes,
                        plot: plt.cm.ScalarMappable,
                        aspect: float = 20.,
                        pad_fraction: float = 0.5,
                        ticks: float = 0.5):
        # https://stackoverflow.com/a/33505522
        divider = make_axes_locatable(ax)
        width = axes_size.AxesY(ax, aspect=1. / aspect)
        pad = axes_size.Fraction(pad_fraction, width)
        cax = divider.append_axes('right', size=width, pad=pad)

        colorbar = fig.colorbar(mappable=plot,
                                ticks=MultipleLocator(ticks),
                                ax=ax,
                                cax=cax)
        colorbar.set_label(
            'Vertical speed (m/s)')  # fontsize=self._style.axis_fontsize)

        # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.colorbar.html
        self._svg_compatible_contour(colorbar.solids)

    def _svg_compatible_contour(self, p: ContourSet):
        # # https://github.com/matplotlib/matplotlib/issues/4419
        p.set_edgecolor('face')
