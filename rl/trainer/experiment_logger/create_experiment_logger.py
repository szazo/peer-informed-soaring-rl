import hydra
from dataclasses import replace
from .api import ExperimentLoggerInterface
from .config import ExperimentLoggerConfigBase


def create_experiment_logger(
        logger_config: ExperimentLoggerConfigBase,
        name_override: str | None = None) -> ExperimentLoggerInterface:

    if name_override is not None:
        logger_config = replace(logger_config, name=name_override)

    experiment_logger = hydra.utils.instantiate(logger_config,
                                                _convert_='object')

    return experiment_logger
