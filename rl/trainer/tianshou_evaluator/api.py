from functools import partial
from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import tianshou
import pandas as pd
from ..experiment_logger import ExperimentLoggerInterface


class TianshouEvaluatorBase(ABC):

    @abstractmethod
    def evaluate(self,
                 policy: partial[tianshou.policy.BasePolicy],
                 output_dir: str,
                 experiment_logger: ExperimentLoggerInterface,
                 weights_state_dict: dict[str, Any] | None = None):
        pass


class ObservationLogger(ABC):

    @abstractmethod
    def transform_buffer_to_dataframe(self, buffer: tianshou.data.ReplayBuffer,
                                      output_df: pd.DataFrame):
        pass


@dataclass
class EvaluatorConfigBase:
    _target_: str = 'trainer.tianshou_evaluator.TianshouEvaluatorBase'


# environment specific observation logger
@dataclass
class EnvironmentObservationLoggerConfigBase:
    _target_: str = 'trainer.tianshou_evaluator.ObservationLogger'


@dataclass
class ObservationLoggerParameters:
    log_buffer_size: int
    env: EnvironmentObservationLoggerConfigBase | None = None
    additional_columns: dict[str, str] = field(default_factory=dict)
