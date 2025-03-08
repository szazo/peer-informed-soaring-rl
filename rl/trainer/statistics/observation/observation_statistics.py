import logging
import os
from dataclasses import dataclass
import pandas as pd
import hydra
from ...experiment_logger import ExperimentLoggerInterface
from ..api import StatisticsBase
from .api import ObservationStatisticsPlugin, ObservationStatisticsPluginConfigBase


@dataclass
class ObservationStatisticsParameters:
    observation_log_key: str
    target_stats_log_key: str
    plugin: ObservationStatisticsPluginConfigBase


@dataclass
class ObservationStatistics(StatisticsBase):

    _log: logging.Logger
    _params: ObservationStatisticsParameters

    def __init__(self, **params):

        self._log = logging.getLogger(__class__.__name__)
        self._params = ObservationStatisticsParameters(**params)

        self._log.debug('__init__; params=%s', self._params)

    def run(self, output_dir: str, source_logger: ExperimentLoggerInterface,
            target_logger: ExperimentLoggerInterface):

        self._log.debug('downloading observation log with key "%s"...',
                        self._params.observation_log_key)

        os.makedirs(output_dir, exist_ok=True)
        observation_log_filepath = os.path.join(
            output_dir, 'downloaded_observation_log.csv')
        source_logger.query_file(self._params.observation_log_key,
                                 observation_log_filepath)

        self._log.debug('creating observation log plugin "%s"...',
                        self._params.plugin._target_)

        # create the statistics plugin
        stat_plugin: ObservationStatisticsPlugin = hydra.utils.instantiate(
            self._params.plugin, _convert_='object')

        self._log.debug('running observation stat plugin "%s"...',
                        self._params.plugin._target_)

        observation_log_df = pd.read_csv(observation_log_filepath)
        stats_df = stat_plugin.run(observation_log=observation_log_df)

        self._log.debug('observation stat plugin finished; result=%s',
                        stats_df)

        stats_filepath = os.path.join(output_dir, 'stats.csv')

        self._log.debug('saving stat file to "%s"', stats_filepath)
        stats_df.to_csv(stats_filepath)

        target_logger.log_file(key=self._params.target_stats_log_key,
                               path=stats_filepath)
