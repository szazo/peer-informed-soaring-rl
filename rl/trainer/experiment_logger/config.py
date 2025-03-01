from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from .api import ExperimentLoggerParametersBase
from .local_tensorboard_experiment_logger import LocalTensorBoardExperimentLoggerParameters
from .neptune_experiment_logger import NeptuneExperimentLoggerParameters


@dataclass
class ExperimentLoggerConfigBase(ExperimentLoggerParametersBase):
    _target_: str = 'trainer.experiment_logger.ExperimentLoggerInterface'


@dataclass(kw_only=True)
class LocalTensorBoardExperimentLoggerConfig(ExperimentLoggerConfigBase,
                                             LocalTensorBoardExperimentLoggerParameters):
    _target_: str = 'trainer.experiment_logger.LocalTensorBoardExperimentLogger'


@dataclass(kw_only=True)
class NeptuneExperimentLoggerConfig(ExperimentLoggerConfigBase,
                                    NeptuneExperimentLoggerParameters):
    _target_: str = 'trainer.experiment_logger.NeptuneExperimentLogger'


def register_experiment_logger_config_groups(base_group: str,
                                             config_store: ConfigStore):
    config_store.store(group=f'{base_group}',
                       name='local_tensorboard',
                       node=LocalTensorBoardExperimentLoggerConfig)
    config_store.store(group=f'{base_group}',
                       name='neptune',
                       node=NeptuneExperimentLoggerConfig)
