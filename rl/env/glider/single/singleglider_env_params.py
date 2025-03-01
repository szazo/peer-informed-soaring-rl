from dataclasses import dataclass, field

from ..base import EgocentricSpatialTransformationParameters, GliderEnvParameters
from ..base.agent import (GliderInitialConditionsParameters)


@dataclass
class SingleGliderEnvParameters(GliderEnvParameters):
    # obsolete or egocentric
    spatial_transformation: str = 'obsolete'
    egocentric_spatial_transformation: None | EgocentricSpatialTransformationParameters = None
    initial_conditions_params: GliderInitialConditionsParameters = field(
        default_factory=GliderInitialConditionsParameters)
