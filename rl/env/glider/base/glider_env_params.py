from dataclasses import dataclass, field
from ..base.wrappers.discrete_to_continuous_action_wrapper import DiscreteToContinuousDegreesParams
from omegaconf import MISSING
from .simulation_box_params import SimulationBoxParameters
from .agent import (GliderRewardParameters, GliderCutoffParameters)
from .time_params import TimeParameters
from .visualization import (
    RenderParameters,
    LayoutParameters,
    ThermalCore3DPlotParameters,
)
from .agent import GliderAgentParameters


@dataclass
class EgocentricSpatialTransformationParameters:
    relative_to: str = 'first'  # first | last
    reverse: bool = False


@dataclass(kw_only=True)
class GliderSimulationParameters:
    simulation_box_params: SimulationBoxParameters = field(
        default_factory=SimulationBoxParameters)
    time_params: TimeParameters = field(default_factory=TimeParameters)
    cutoff_params: GliderCutoffParameters = field(
        default_factory=GliderCutoffParameters)
    reward_params: GliderRewardParameters = field(
        default_factory=GliderRewardParameters)
    glider_agent_params: GliderAgentParameters = field(
        default_factory=GliderAgentParameters)


@dataclass(kw_only=True)
class GliderEnvParameters(GliderSimulationParameters):
    render_params: RenderParameters = field(default_factory=RenderParameters)
    layout_params: LayoutParameters = field(default_factory=LayoutParameters)
    thermal_core_3d_plot_params: ThermalCore3DPlotParameters = field(
        default_factory=ThermalCore3DPlotParameters)
    max_sequence_length: int = MISSING
    discrete_continuous_mapping: DiscreteToContinuousDegreesParams = field(
        default_factory=DiscreteToContinuousDegreesParams)
