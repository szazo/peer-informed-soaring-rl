from ..simulation_box_params import SimulationBoxParameters
from ...air_velocity_field import AirVelocityFieldInterface
from .multiglider_visualization import (
    MultigliderVisualization,
    RenderParameters,
    GridLayoutPlot,
    LayoutParameters,
    TrajectoryPlot,
    TrajectoryPlotParameters,
    ThermalCore3DPlot,
    ThermalCore3DPlotParameters,
)


def make_visualization(
        simulation_box_params: SimulationBoxParameters,
        render_params: RenderParameters,
        layout_params: LayoutParameters,
        air_velocity_field: AirVelocityFieldInterface,
        thermal_core_3d_plot_params: ThermalCore3DPlotParameters,
        title="Multi-agent simulation"):

    trajectory_plot = TrajectoryPlot(params=TrajectoryPlotParameters(
        simulation_box=simulation_box_params))
    thermal_core_plot = ThermalCore3DPlot(
        simulation_box_params=simulation_box_params,
        air_velocity_field=air_velocity_field,
        params=thermal_core_3d_plot_params,
    )
    plot = GridLayoutPlot(trajectory_plot=trajectory_plot,
                          thermal_core_plot=thermal_core_plot)

    visualization = MultigliderVisualization(render_params=render_params,
                                             layout_params=layout_params,
                                             plot=plot)

    return visualization
