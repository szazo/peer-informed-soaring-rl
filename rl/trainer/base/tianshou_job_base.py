from functools import partial
from typing import Any
from omegaconf import MISSING
import os
import logging
from dataclasses import dataclass, field
import hydra
import torch
import tianshou
from ..experiment_logger import (ExperimentLoggerConfigBase,
                                 ExperimentLoggerInterface,
                                 create_experiment_logger)
from ..tianshou_evaluator import EvaluatorConfigBase, TianshouEvaluatorBase
from ..statistics import StatisticsConfigBase, run_statistics
from ..common import TianshouModelConfigBase


@dataclass
class TianshouJobParametersBase:
    evaluators: dict[str, EvaluatorConfigBase] = MISSING
    statistics: dict[str, StatisticsConfigBase] = field(
        default_factory=lambda: dict())
    experiment_name: str = MISSING


class TianshouJobBase:

    _log: logging.Logger
    _output_dir: str

    def __init__(self, output_dir: str):

        self._log = logging.getLogger(__class__.__name__)
        self._output_dir = output_dir

    def _create_experiment_logger(self,
                                  logger_config: ExperimentLoggerConfigBase,
                                  name_override: str | None = None):
        self._log.debug('creating experiment logger; logger_config=%s',
                        logger_config)

        return create_experiment_logger(logger_config,
                                        name_override=name_override)

    def _create_policy_model(
        self, device: torch.device, policy_config: TianshouModelConfigBase
    ) -> partial[tianshou.policy.BasePolicy]:

        policy = hydra.utils.instantiate(policy_config, _convert_='object')
        policy = partial(policy, device=device)

        return policy

    def _evaluate(self,
                  policy: partial[tianshou.policy.BasePolicy],
                  evaluators: dict[str, EvaluatorConfigBase],
                  experiment_logger: ExperimentLoggerInterface,
                  weights_state_dict: dict[str, Any] | None = None):

        for name, evaluator_config in evaluators.items():
            self._log.debug('evaluating trained policy using \'%s\' evaluator',
                            name)

            # create the evaluator
            evaluator: TianshouEvaluatorBase = hydra.utils.instantiate(
                evaluator_config, _recursive_=False, _convert_='object')
            output_dir = os.path.join(self._output_dir, f'eval/{name}')
            evaluator.evaluate(policy=policy,
                               output_dir=output_dir,
                               experiment_logger=experiment_logger,
                               weights_state_dict=weights_state_dict)

    def _run_statistics(self, statistics: dict[str, StatisticsConfigBase],
                        source_logger: ExperimentLoggerInterface,
                        target_logger: ExperimentLoggerInterface):

        stats_output_dir = os.path.join(self._output_dir, 'stats')
        run_statistics(statistics=statistics,
                       source_logger=source_logger,
                       target_logger=target_logger,
                       output_dir=stats_output_dir)
