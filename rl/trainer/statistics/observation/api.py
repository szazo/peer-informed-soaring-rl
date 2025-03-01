from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd


@dataclass
class ObservationStatisticsPluginConfigBase:
    _target_: str = 'trainer.statistics.StatisticsPlugin'


class ObservationStatisticsPlugin(ABC):

    @abstractmethod
    def run(self, observation_log: pd.DataFrame) -> pd.DataFrame:
        pass
