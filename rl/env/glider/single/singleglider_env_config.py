from typing import Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore

from .singleglider_env_params import SingleGliderEnvParameters
from ..base import (register_glider_env_aerodynamics_config_groups,
                    register_glider_env_air_velocity_field_config_groups)
from ..base.agent.config import register_glider_env_agent_config_groups

from .statistics import register_singleglider_statistics_config_groups


@dataclass
class SingleGliderEnvConfig:
    air_velocity_field: Any
    aerodynamics: Any
    params: SingleGliderEnvParameters = field(
        default_factory=SingleGliderEnvParameters)
    _target_: str = 'env.glider.single.make_singleglider_env.make_singleglider_env'


def register_singleglider_env_config_groups(config_store: ConfigStore):

    config_store.store(group='env/glider',
                       name='base_singleglider',
                       node=SingleGliderEnvConfig)

    register_singleglider_statistics_config_groups(config_store=config_store)
    register_glider_env_aerodynamics_config_groups(config_store=config_store)
    register_glider_env_air_velocity_field_config_groups(
        config_store=config_store)
    register_glider_env_agent_config_groups(config_store=config_store)
