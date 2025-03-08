import os
from dataclasses import dataclass
import logging

from ..experiment_logger import (ExperimentLoggerConfigBase,
                                 create_experiment_logger)
from .api import StatisticsConfigBase
from .run_statistics import run_statistics


@dataclass
class StatisticsJobParameters:
    source_logger: ExperimentLoggerConfigBase
    target_logger: ExperimentLoggerConfigBase
    statistics: dict[str, StatisticsConfigBase]
    experiment_name: str | None = None


class StatisticsJob:

    _log: logging.Logger
    _params: StatisticsJobParameters
    _output_dir: str

    def __init__(self, params: StatisticsJobParameters, output_dir: str):
        self._log = logging.getLogger(__class__.__name__)
        self._params = params
        self._output_dir = output_dir

    def run(self):
        self._log.debug('run; params=%s', self._params)

        self._log.debug('creating source experiment logger "%s"...',
                        self._params.source_logger._target_)
        source_logger = create_experiment_logger(self._params.source_logger)

        self._log.debug('creating target experiment logger "%s"...',
                        self._params.source_logger._target_)
        target_logger = create_experiment_logger(
            self._params.target_logger,
            name_override=self._params.experiment_name)
        try:
            stats_output_dir = os.path.join(self._output_dir, 'stats')
            run_statistics(statistics=self._params.statistics,
                           source_logger=source_logger,
                           target_logger=target_logger,
                           output_dir=stats_output_dir)

            source_logger.stop(success=True)
            target_logger.stop(success=True)
            self._log.debug('job completed successfully')

        except Exception as e:
            self._log.error('error occurred during statistics; exception=%s',
                            e)
            source_logger.stop(success=True)
            target_logger.stop(success=False)
            raise
