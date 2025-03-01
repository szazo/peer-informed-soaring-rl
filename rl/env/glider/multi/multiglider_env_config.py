from typing import Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from .multiglider_env_params import MultiGliderEnvParameters


@dataclass
class MultiGliderEnvConfig:
    air_velocity_field: Any
    aerodynamics: Any
    env_name: str = 'multiglider'
    params: MultiGliderEnvParameters = field(
        default_factory=MultiGliderEnvParameters)
    _target_: str = 'env.glider.multi.make_multiglider_env.make_multiglider_env'


def register_multiglider_env_config_groups(config_store: ConfigStore):

    config_store.store(group='env/glider',
                       name='base_multiglider',
                       node=MultiGliderEnvConfig)
