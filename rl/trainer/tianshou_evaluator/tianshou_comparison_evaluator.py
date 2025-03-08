from dataclasses import dataclass
from functools import partial
from typing import Any
import os
import logging
from gymnasium.utils import seeding
import tianshou
import pandas as pd
from ..common import TianshouEnviromentParameters
from .api import TianshouEvaluatorBase, ObservationLoggerParameters
from .tianshou_evaluator import (TianshouEvaluator,
                                 TianshouEvaluatorParameters,
                                 TianshouEvaluatorResult)
from ..experiment_logger import ExperimentLoggerInterface


@dataclass
class TianshouComparisonEvaluatorParameters:
    # results will be logged under this key
    log_parent_key: str
    env: TianshouEnviromentParameters
    episode_count: int
    deterministic_eval: bool  # whether sample from action distribution or use the max
    deterministic_comparison: bool
    compare_with_random_policy: bool
    observation_logger: ObservationLoggerParameters | None
    seed: int | None = None
    create_video: bool = False


@dataclass
class TianshouComparisonEvaluator(TianshouEvaluatorBase):

    _log: logging.Logger
    _params: TianshouComparisonEvaluatorParameters

    def __init__(self, **params):

        self._log = logging.getLogger(__class__.__name__)
        self._params = TianshouComparisonEvaluatorParameters(**params)

    def evaluate(self,
                 policy: partial[tianshou.policy.BasePolicy],
                 output_dir: str,
                 experiment_logger: ExperimentLoggerInterface,
                 weights_state_dict: dict[str, Any] | None = None):

        self._log.debug('starting comparison evaluation')

        params = self._params

        seed = None
        if params.deterministic_comparison:
            _, seed = seeding.np_random(params.seed)
            self._log.debug(
                'deterministic evaluation; all policies will use the environments with the same seed=%s',
                seed)

        self._log.debug('initializing policies...')
        policies = self._create_policies(policy=policy)

        policy_results: dict[str, TianshouEvaluatorResult] = {}

        for current_name, current_policy in policies.items():

            policy_output_dir = self._policy_output_dir(
                output_dir, current_name)
            self._log.debug(
                'evaluating using \'%s\' policy (output_dir=%s)...',
                current_name, policy_output_dir)

            evaluator_params = TianshouEvaluatorParameters(
                log_parent_key=self._params.log_parent_key +
                f'/{current_name}',
                env=self._params.env,
                episode_count=self._params.episode_count,
                deterministic_eval=self._params.deterministic_eval,
                seed=seed,
                create_video=self._params.create_video,
                observation_logger=self._params.observation_logger,
                output_dir=policy_output_dir)
            evaluator = TianshouEvaluator(params=evaluator_params,
                                          experiment_logger=experiment_logger)
            random = current_name == '_random_'
            policy_result = evaluator.evaluate(
                policy=current_policy,
                random=random,
                weights_state_dict=weights_state_dict)
            policy_results[current_name] = policy_result

        # merge and create the resulting observation log
        if self._params.observation_logger is not None:
            self._log.debug('merging observation logs...')
            observation_log_df = self._merge_observation_logs(
                policy_results=policy_results, output_dir=output_dir)

    def _merge_observation_logs(self,
                                policy_results: dict[str,
                                                     TianshouEvaluatorResult],
                                output_dir: str):

        merged_observation_log_df = pd.DataFrame()
        for name, result in policy_results.items():

            if result.observation_log_path is None:
                continue

            self._log.debug('merging observation log for policy "%s"...', name)

            observation_log_df = pd.read_csv(result.observation_log_path)
            observation_log_df.insert(0, 'policy', name, allow_duplicates=True)
            merged_observation_log_df = pd.concat(
                [merged_observation_log_df, observation_log_df])

        merged_log_path = os.path.join(output_dir, 'observation_log.csv')
        self._log.debug('writing merged observation log "%s"...',
                        merged_log_path)

        # save the merged log, skipping newly generated df index
        merged_observation_log_df.to_csv(merged_log_path, index=False)

        return merged_observation_log_df

    def _policy_output_dir(self, output_dir: str, policy_name: str) -> str:
        policy_output_dir = os.path.join(output_dir, policy_name)

        return policy_output_dir

    def _create_policies(
        self, policy: partial[tianshou.policy.BasePolicy]
    ) -> dict[str, partial[tianshou.policy.BasePolicy]]:

        policies = {'policy': policy}
        if self._params.compare_with_random_policy:
            # random will use the same policy, but collector parameter will be set to random
            policies['_random_'] = policy

        return policies
