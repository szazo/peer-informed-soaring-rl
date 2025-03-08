from abc import ABC, abstractmethod
from dataclasses import dataclass
from numbers import Number

import numpy as np


@dataclass
class ExperimentLoggerParametersBase:
    name: str | None = None


class ExperimentLoggerInterface(ABC):

    @abstractmethod
    def log_metrics(self,
                    key: str,
                    value: float | str | Number | np.number,
                    step: int | None = None):

        pass

    @abstractmethod
    def log_dict(self,
                 key: str,
                 dictionary: dict,
                 log_as_str=False,
                 log_as_pickle=False):
        pass

    @abstractmethod
    def log_file(self, key: str, path: str):
        pass

    @abstractmethod
    def query_file(self, key: str, destination_path: str):
        pass

    @abstractmethod
    def log_video(self, key: str, path: str):
        pass

    @abstractmethod
    def stop(self, success: bool):
        pass
