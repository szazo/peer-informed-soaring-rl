from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from ..api import StatisticsConfigBase
from .observation_statistics import ObservationStatisticsParameters


@dataclass
class ObservationLogStatisticsConfig(StatisticsConfigBase,
                                     ObservationStatisticsParameters):
    _target_: str = 'trainer.statistics.observation.ObservationStatistics'


def register_observation_statistics_config_groups(config_store: ConfigStore):

    config_store.store(group='statistics',
                       name='observation_statistics',
                       node=ObservationLogStatisticsConfig)
