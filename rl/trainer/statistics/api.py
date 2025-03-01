from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..experiment_logger import ExperimentLoggerInterface


@dataclass
class StatisticsConfigBase:
    _target_: str = 'trainer.statistics.StatisticsBase'


class StatisticsBase(ABC):

    @abstractmethod
    def run(self, output_dir: str, source_logger: ExperimentLoggerInterface,
            target_logger: ExperimentLoggerInterface):
        pass
