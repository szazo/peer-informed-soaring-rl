from dataclasses import dataclass, field
from typing import Any
import warnings
import os
import shutil
from pathlib import Path
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, MISSING
import ray
import time
from IPython.core import getipython
from slugify import slugify
import matplotlib as mpl
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from thermal.gaussian import register_gaussian_air_velocity_field_config_groups
from thermal.visualization import (ThermalCrossSectionPlotParameters,
                                   ThermalCrossSectionPlot,
                                   ThermalPlotDataSerializer)
from utils import plot_style

cs = ConfigStore.instance()
register_gaussian_air_velocity_field_config_groups(
    group='env/glider/air_velocity_field', config_store=cs)


@dataclass
class ExperimentConfig:
    show_plots_in_cli: bool = True
    target_dir: str | None = None
    plot_config: ThermalCrossSectionPlotParameters = MISSING
    air_velocity_field: Any = MISSING
    fig_size: list[float] = field(default_factory=lambda: [17., 9.])
    horizontal_cross_section: bool = True
    vertical_cross_section: bool = True
    show_title: bool = True
    show_axes_title: bool = True
    show_w_min_max: bool = True
    rc_params: dict[str, Any] = field(default_factory=lambda: {})


cs = hydra.core.config_store.ConfigStore.instance()
cs.store(name='experiment_config', node=ExperimentConfig)


def create_axes(fig: Figure, grid_shape: tuple[int, int], loc: tuple[int,
                                                                     int]):
    ax = plt.subplot2grid(fig=fig, shape=grid_shape, loc=loc)
    ax.set_aspect('equal', adjustable='box', anchor='NW')
    return ax


def create_figure_and_axes(cfg: ExperimentConfig):

    fig = plt.figure(figsize=(cfg.fig_size[0], cfg.fig_size[1]),
                     layout='constrained')

    # grid parameters
    ax_count: int = 0
    if cfg.horizontal_cross_section:
        ax_count += 1
    if cfg.vertical_cross_section:
        ax_count += 1
    grid_shape = (1, ax_count)

    # create axes
    axes = []
    current_col_index = 0
    if cfg.horizontal_cross_section:
        axes.append(
            create_axes(fig=fig,
                        grid_shape=grid_shape,
                        loc=(0, current_col_index)))
        current_col_index += 1
    if cfg.vertical_cross_section:
        axes.append(
            create_axes(fig=fig,
                        grid_shape=grid_shape,
                        loc=(0, current_col_index)))
        current_col_index += 1

    return fig, axes


def create_plot(cfg: ExperimentConfig):

    plot_config = cfg.plot_config
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print('output directory is ', output_dir)

    # ignore ray warnings https://github.com/ray-project/ray/issues/10279
    warnings.filterwarnings(
        'ignore',
        message=".*unclosed file <_io\\.TextIOWrapper name='/tmp/ray/session.*"
    )

    # ray.init()
    try:
        with plot_style(cfg.rc_params):
            t_start = time.perf_counter()

            air_velocity_field = hydra.utils.instantiate(
                cfg.air_velocity_field, _convert_='object')

            initial_conditions = air_velocity_field.reset()
            name = air_velocity_field.name
            print('air velocity field info: ', initial_conditions)

            cross_section_plot = ThermalCrossSectionPlot()

            fig, axes = create_figure_and_axes(cfg)

            if cfg.show_title:
                fig.suptitle(name)

            horizontal_data, vertical_data = cross_section_plot.plot(
                params=plot_config,
                figure=fig,
                field=air_velocity_field,
                t_s=0.,
                show_horizontal=cfg.horizontal_cross_section,
                show_vertical=cfg.vertical_cross_section,
                axes=axes,
                show_w_min_max=cfg.show_w_min_max,
                show_title=cfg.show_axes_title)

            t_end = time.perf_counter()
            print(
                f'plot for "{name}" finished; elapsed time (s): {t_end - t_start}'
            )

            slugified_name = slugify(name, lowercase=False)

            fig_path = os.path.join(output_dir, f'{slugified_name}.png')
            print(f'saving plot to "{fig_path}"...')
            fig.savefig(fig_path, bbox_inches='tight', pad_inches=0)

            # SVG save
            fig_svg_path = os.path.join(output_dir, f'{slugified_name}.svg')
            fig.savefig(fig_svg_path, bbox_inches='tight', pad_inches=0)
            # with mpl.rc_context({'svg.fonttype': 'path'}):
            #     # print('SVG font', plt.rcParams['svg.fonttype'])

            # save 3d data to hdf5
            hdf_path = os.path.join(output_dir, f'{slugified_name}.hdf5')
            serializer = ThermalPlotDataSerializer()
            serializer.save(horizontal_data=horizontal_data,
                            vertical_data=vertical_data,
                            out_filepath=hdf_path)

            if cfg.target_dir is not None:
                # copy the files to the output dir
                target_dir = Path(cfg.target_dir)
                target_dir.mkdir(parents=True, exist_ok=True)
                assert target_dir.is_dir, f"{target_dir} should be a directory"

                shutil.copy(fig_path, target_dir)
                shutil.copy(fig_svg_path, target_dir)
                shutil.copy(hdf_path, target_dir)

            if getipython.get_ipython() is not None or cfg.show_plots_in_cli:
                print('displaying plot...')
                plt.show(block=True)
    finally:
        if ray.is_initialized:
            print('shutting down...')
            # ignore ray warning
            warnings.filterwarnings(
                'ignore', message='.*subprocess .* is still running.*')
            ray.shutdown()


@hydra.main(version_base=None,
            config_name='plot_thermal_config',
            config_path='config')
def exp_main(cfg: ExperimentConfig):
    create_plot(OmegaConf.to_object(cfg))


if __name__ == '__main__':
    exp_main()
