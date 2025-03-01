from hydra.core.config_store import ConfigStore
from .air_velocity_filter import (GaussianFilterParams,
                                  ExponentialFilterParams)


def register_glider_env_agent_config_groups(config_store: ConfigStore):

    config_store.store(group='env/glider/air_velocity_filter',
                       name='gaussian_kernel',
                       node=GaussianFilterParams)
    config_store.store(group='env/glider/air_velocity_filter',
                       name='exponential_kernel',
                       node=ExponentialFilterParams)
