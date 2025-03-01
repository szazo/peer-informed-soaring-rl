from abc import ABC, abstractmethod
import logging
from pathlib import Path
from typing import Generic, TypeVar, Any
import numpy as np
import tianshou
from tianshou.data import CollectStats, SequenceSummaryStats
from trainer.common import ExperimentLoggerWeightStore
from trainer.experiment_logger import ExperimentLoggerInterface

from trainer.multi_agent.tests.test_multi_agent_policy import ParallelMultiAgentPolicyManager
from trainer.tianshou_trainer_job.tianshou_experiment_log_adapter import collect_multi_agent_returns_and_stats

AgentIDType = TypeVar('AgentIDType', bound=str)


class BestWeightsSaver(ABC):

    _experiment_logger: ExperimentLoggerInterface
    _base_path: Path
    _log: logging.Logger

    def __init__(self, experiment_logger: ExperimentLoggerInterface,
                 base_path: Path):

        self._experiment_logger = experiment_logger
        self._base_path = base_path

        self._log = logging.getLogger(__class__.__name__)

    @abstractmethod
    def save(self, epoch: int, test_stats: CollectStats):
        pass

    def _calculate_metric(self, returns_stat: SequenceSummaryStats):

        log_wmsr = self._calculate_log_wmsr(returns_stat)
        return log_wmsr

    def _log_common_metrics(self,
                            epoch: int,
                            stats: SequenceSummaryStats,
                            prefix: str = ''):

        log_key = self._base_log_key(prefix)

        sharpe_ratio = self._calculate_sharpe_ratio(stats)
        log_wmsr = self._calculate_log_wmsr(stats)

        self._experiment_logger.log_metrics(f'{log_key}sharpe_ratio',
                                            sharpe_ratio,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}log_wmsr',
                                            log_wmsr,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}mean',
                                            stats.mean,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}std',
                                            stats.std,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}mean_minus_std',
                                            stats.mean - stats.std,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}wmsr',
                                            self._calculate_wmsr(stats),
                                            step=epoch)

    def _calculate_sharpe_ratio(self, stats: SequenceSummaryStats):
        sharpe_ratio = stats.mean

        if not np.isclose(stats.std, 0.):
            sharpe_ratio = stats.mean / stats.std

        return sharpe_ratio

    def _calculate_log_wmsr(self, stats: SequenceSummaryStats, p=2.5):

        log_wmsr = -1.
        if not np.isclose(stats.std, 0.) and stats.mean > 0:
            log_wmsr = np.log(stats.mean**p / stats.std)

        return log_wmsr

    def _calculate_wmsr(self, stats: SequenceSummaryStats, p=2.5):

        wmsr = -1.
        if not np.isclose(stats.std, 0.) and stats.mean > 0:
            wmsr = stats.mean**p / stats.std

        return wmsr

    def _log_best_metrics(self,
                          epoch: int,
                          best_metric: float,
                          stats: SequenceSummaryStats,
                          prefix: str = ''):

        log_key = self._base_log_key(prefix)

        self._log.debug(
            'saving best metric...; log_key=%s,best_metric=%s,stats=%s',
            log_key, best_metric, stats)

        self._experiment_logger.log_metrics(f'{log_key}best_log_wmsr',
                                            best_metric,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}best_reward_mean',
                                            stats.mean,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}best_reward_std',
                                            stats.std,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}best_reward_max',
                                            stats.max,
                                            step=epoch)
        self._experiment_logger.log_metrics(f'{log_key}best_reward_min',
                                            stats.max,
                                            step=epoch)
        self._experiment_logger.log_metrics(
            f'{log_key}best_reward_mean_minus_std',
            stats.mean - stats.std,
            step=epoch)

    def _save_weights(self,
                      state_dict: dict[str, Any],
                      filename: str,
                      log_key: str,
                      prefix: str = ''):

        self._base_path.mkdir(parents=True, exist_ok=True)

        file_prefix = f'{prefix}_' if prefix else ''
        file_path = self._base_path / f'{file_prefix}{filename}'

        self._log.debug('saving weight file...; checkpoint_filepath=%s',
                        file_path)

        log_key = self._base_log_key(prefix) + log_key

        ExperimentLoggerWeightStore(self._experiment_logger).save_best_weights(
            state_dict=state_dict,
            log_key=log_key,
            target_filepath=str(file_path))

    def _base_log_key(self, prefix: str = ''):
        prefix_part = f'/{prefix}' if prefix else ''
        log_key = f'checkpoint{prefix_part}/'

        return log_key


class SingleAgentBestWeightsSaver(BestWeightsSaver):

    _policy: tianshou.policy.BasePolicy

    _best_metric: float

    def __init__(self, experiment_logger: ExperimentLoggerInterface,
                 policy: tianshou.policy.BasePolicy, base_path: Path):

        super().__init__(experiment_logger=experiment_logger,
                         base_path=base_path)

        self._policy = policy

        self._best_metric = float(np.finfo(np.float32).min)

    def save(self, epoch: int, test_stats: CollectStats):

        returns_stat = test_stats.returns_stat
        assert returns_stat is not None

        new_metric = self._calculate_metric(returns_stat)

        self._log_common_metrics(epoch, returns_stat)

        if new_metric > self._best_metric:

            self._log.debug(
                'new best weights found (%s), saving as checkpoint',
                new_metric)

            self._log_best_metrics(epoch, new_metric, returns_stat)

            state = self._policy.state_dict()
            self._save_weights(state,
                               filename='best_weights.pth',
                               log_key='best_weights')

            self._best_metric = new_metric


# Review: actually it would be better if we would save for subpolicy ids and not for agents
class MultiAgentBestWeightsSaver(Generic[AgentIDType], BestWeightsSaver):

    _log: logging.Logger
    _possible_agent_ids: list[AgentIDType]
    _learning_agent_ids: list[AgentIDType]

    _agent_bests: dict[AgentIDType, float]

    _policy: ParallelMultiAgentPolicyManager

    def __init__(self, possible_agent_ids: list[AgentIDType],
                 experiment_logger: ExperimentLoggerInterface,
                 policy: tianshou.policy.BasePolicy, base_path: Path):

        super().__init__(experiment_logger=experiment_logger,
                         base_path=base_path)

        self._log = logging.getLogger(__class__.__name__)

        assert isinstance(policy, ParallelMultiAgentPolicyManager)
        self._policy = policy

        self._possible_agent_ids = possible_agent_ids
        self._learning_agent_ids = self._policy.learn_agent_ids

        self._agent_bests = {}
        for agent_id in self._learning_agent_ids:
            self._agent_bests[agent_id] = float(np.finfo(np.float32).min)

    def save(self, epoch: int, test_stats: CollectStats):

        per_agent_stats = collect_multi_agent_returns_and_stats(
            test_stats.returns, possible_agents=self._possible_agent_ids)

        # check only the learning agents
        required_agent_ids = []
        new_bests = self._agent_bests.copy()
        for agent_id in self._learning_agent_ids:
            agent_returns = per_agent_stats[agent_id]
            new_metric = self._calculate_metric(agent_returns.returns_stat)
            current_best = self._agent_bests[agent_id]

            self._log_common_metrics(epoch,
                                     agent_returns.returns_stat,
                                     prefix=agent_id)

            if new_metric > current_best:
                self._log.debug(
                    'new best weights found (%s) for agent \'%s\', saving as checkpoint',
                    new_metric, agent_id)

                new_bests[agent_id] = new_metric
                required_agent_ids.append(agent_id)

        # save the weights for all
        # TODO: save the best weights for each agent separately if required
        policy_state = self._policy.state_dict()
        if len(required_agent_ids) > 0:
            self._save_weights(policy_state,
                               filename='best_weights.pth',
                               log_key='best_weights')

        self._save_weights(policy_state,
                           filename=f'weights{epoch}.pth',
                           log_key=f'weights{epoch}')

        # log metrics for the required ones
        for agent_id in required_agent_ids:
            self._log_best_metrics(epoch,
                                   new_bests[agent_id],
                                   per_agent_stats[agent_id].returns_stat,
                                   prefix=agent_id)

        self._agent_bests = new_bests
