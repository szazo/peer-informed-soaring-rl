from hydra.core.config_store import ConfigStore
from thermal.gaussian import (
    register_gaussian_air_velocity_field_config_groups)
from ..aerodynamics import SimpleAerodynamicsConfig


def register_glider_env_aerodynamics_config_groups(config_store: ConfigStore):

    config_store.store(group='env/glider/aerodynamics',
                       name='base_simple_aerodynamics',
                       node=SimpleAerodynamicsConfig)


def register_glider_env_air_velocity_field_config_groups(
        config_store: ConfigStore):
    register_gaussian_air_velocity_field_config_groups(
        group='env/glider/air_velocity_field', config_store=config_store)
