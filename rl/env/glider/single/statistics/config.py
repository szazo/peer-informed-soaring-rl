from typing import Any
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore

from thermal.api import AirVelocityFieldConfigBase
from trainer.statistics.observation import ObservationStatisticsPluginConfigBase
from trainer.tianshou_evaluator import EnvironmentObservationLoggerConfigBase
from .singleglider_statistics import SingleGliderStatisticsParameters
from .singleglider_trajectory_comparison_statistics import SingleGliderTrajectoryComparisonStatisticsParameters
from .singleglider_trajectory_merge_statistics import SingleGliderTrajectoryMergeStatisticsParameters


@dataclass
class SingleGliderStatisticsConfig(ObservationStatisticsPluginConfigBase,
                                   SingleGliderStatisticsParameters):
    _target_: str = 'env.glider.single.statistics.SingleGliderStatistics'


@dataclass(kw_only=True)
class SingleGliderTrajectoryComparisonStatisticsConfig(
        ObservationStatisticsPluginConfigBase,
        SingleGliderTrajectoryComparisonStatisticsParameters):
    trajectory_air_velocity_field: AirVelocityFieldConfigBase
    _target_: str = 'env.glider.single.statistics.SingleGliderTrajectoryComparisonStatistics'


@dataclass(kw_only=True)
class SingleGliderTrajectoryMergeStatisticsConfig(
        ObservationStatisticsPluginConfigBase,
        SingleGliderTrajectoryMergeStatisticsParameters):
    trajectory_air_velocity_field: AirVelocityFieldConfigBase
    _target_: str = 'env.glider.single.statistics.SingleGliderTrajectoryMergeStatistics'


@dataclass
class SingleGliderObservationLoggerConfig(
        EnvironmentObservationLoggerConfigBase):
    _target_: str = 'env.glider.single.statistics.GliderObservationLogger'


def register_singleglider_statistics_config_groups(config_store: ConfigStore):

    config_store.store(group='env/observation_logger',
                       name='singleglider_observation_logger',
                       node=SingleGliderObservationLoggerConfig)

    config_store.store(group='env/statistics',
                       name='singleglider',
                       node=SingleGliderStatisticsConfig)

    config_store.store(
        group='env/statistics',
        name='singleglider_trajectory_comparison_observation_statistics_plugin',
        node=SingleGliderTrajectoryComparisonStatisticsConfig)

    config_store.store(
        group='env/statistics',
        name='singleglider_trajectory_merge_observation_statistics_plugin',
        node=SingleGliderTrajectoryMergeStatisticsConfig)
