from hydra.core.config_store import ConfigStore
from dataclasses import dataclass

from ..api import AirVelocityFieldConfigBase
from .simple_air_velocity_field_params import GaussianAirVelocityFieldParameters


@dataclass(kw_only=True)
class GaussianAirVelocityFieldConfig(GaussianAirVelocityFieldParameters,
                                     AirVelocityFieldConfigBase):
    _target_: str = 'thermal.gaussian.make_gaussian_air_velocity_field'


def register_gaussian_air_velocity_field_config_groups(
        group: str, config_store: ConfigStore):

    config_store.store(group=group,
                       name='base_gaussian',
                       node=GaussianAirVelocityFieldConfig)
