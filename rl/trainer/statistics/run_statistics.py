import os
import logging
import hydra

from .api import StatisticsBase, StatisticsConfigBase
from ..experiment_logger import ExperimentLoggerInterface


def run_statistics(statistics: dict[str, StatisticsConfigBase],
                   source_logger: ExperimentLoggerInterface,
                   target_logger: ExperimentLoggerInterface, output_dir: str):

    log = logging.getLogger(__name__)

    for key, stat_config in statistics.items():

        stat_output_dir = os.path.join(output_dir, key)

        log.debug('creating "%s" statistics with "%s"', key,
                  stat_config._target_)

        stat: StatisticsBase = hydra.utils.instantiate(stat_config,
                                                       _convert_='object',
                                                       _recursive_=False)
        log.debug('running "%s" statistics with "%s"', key,
                  stat_config._target_)

        stat.run(output_dir=stat_output_dir,
                 source_logger=source_logger,
                 target_logger=target_logger)
