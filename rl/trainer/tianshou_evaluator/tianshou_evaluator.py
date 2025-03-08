from dataclasses import dataclass, asdict
from functools import partial
from typing import Any
import os
import logging
import tianshou
import gymnasium
from tianshou.data import CollectStats
from utils import cleanup_numpy_from_dictionary
from .api import ObservationLoggerParameters
from ..common import TianshouEnviromentParameters, TianshouVectorizedCollectorFactory
from ..experiment_logger import ExperimentLoggerInterface
from .vectorized_observation_logger import VectorizedObservationLogger
from .vectorized_video_recorder import VectorizedVideoRecorder
from .vectorized_env_wrapper import VectorizedEnvWrapper


@dataclass
class TianshouEvaluatorParameters:
    # results will be logged under this key
    log_parent_key: str
    env: TianshouEnviromentParameters
    episode_count: int
    output_dir: str
    deterministic_eval: bool
    observation_logger: ObservationLoggerParameters | None
    create_video: bool
    seed: int | None = None


@dataclass
class TianshouEvaluatorResult:
    observation_log_path: str | None


class TianshouEvaluator:

    _log: logging.Logger
    _params: TianshouEvaluatorParameters
    _experiment_logger: ExperimentLoggerInterface | None

    _vectorized_wrappers: list[VectorizedEnvWrapper]

    def __init__(self,
                 params: TianshouEvaluatorParameters,
                 experiment_logger: ExperimentLoggerInterface | None = None):
        self._params = params
        self._experiment_logger = experiment_logger
        self._log = logging.getLogger(__class__.__name__)

        self._vectorized_wrappers = []
        if params.observation_logger is not None:
            self._vectorized_wrappers.append(
                VectorizedObservationLogger(
                    output_dir=params.output_dir,
                    output_observation_log_filepath=self.
                    _observation_log_filepath(),
                    logger_params=params.observation_logger))

        if params.create_video:
            self._vectorized_wrappers.append(
                VectorizedVideoRecorder(output_dir=params.output_dir))

    def evaluate(self,
                 policy: partial[tianshou.policy.BasePolicy],
                 random: bool = False,
                 weights_state_dict: dict[str, Any] | None = None):

        self._log.debug('evaluate; random=%s', random)

        # used for wrapping the vectorized environment
        def env_wrapper_func(env: gymnasium.Env,
                             vectorized_index: int) -> gymnasium.Env:

            for wrapper in self._vectorized_wrappers:
                env = wrapper.wrap_env(env=env,
                                       vectorized_index=vectorized_index)

            return env

        collector_factory = TianshouVectorizedCollectorFactory(
            env_wrapper_func=env_wrapper_func)

        self._log.debug(
            'creating the vectorized environment; vectorized_params=%s',
            self._params.env.vectorized)

        # create the vectorized environment
        vectorized_env, spaces = collector_factory.create_vectorized_environment(
            vectorized_params=self._params.env.vectorized,
            env_config=self._params.env.env,
            seed=self._params.seed)

        self._log.debug('vectorized environment created')

        self._log.debug(
            'initializing the policy; deterministic_eval=%s, spaces=%s',
            self._params.deterministic_eval, spaces)

        policy_instance = policy(
            observation_space=spaces.observation_space,
            action_space=spaces.action_space,
            deterministic_eval=self._params.deterministic_eval)

        # load the parameters
        if weights_state_dict is not None:
            self._log.debug('loading weights...')
            policy_instance.load_state_dict(weights_state_dict)
            self._log.debug('weights loaded')

        # switch the policy to eval mode
        policy_instance.eval()

        self._log.debug('policy initialized; type=%s', type(policy_instance))

        # create the collector
        self._log.debug('creating collector; collector_params=%s',
                        self._params.env.collector)
        collector = collector_factory.create_collector(
            collector_params=self._params.env.collector,
            vectorized_env=vectorized_env,
            policy=policy_instance)

        self._log.debug('collector created')

        # need to reset the environment before collect, if the environment
        # is already reset with a seed, this new reset won't override the seed
        collector.reset_env()

        collect_stats: CollectStats
        try:

            # collect
            self._log.debug('collecting...')
            collect_stats = collector.collect(
                n_episode=self._params.episode_count, random=random)

            self._log.debug('collect finished; result=%s', collect_stats)

        finally:
            self._log.debug('closing collector environments...')
            collector.env.close()

        # TODO: we use now the episode start indices to map episode_index to vectorized env index
        # episode_start_indices contains episode index -> buffer index mapping
        # But because during evaluation every buffer is 1 length, and VectorizedReplayBuffer
        # uses stacked buffer, the buffer index will be equal to the vectorized env index.
        # It's kind of hack, so need to find other method to map episode to vectorized env index.
        assert collect_stats.episode_start_indices is not None
        episode_env_indices = collect_stats.episode_start_indices

        self._log.debug('episode_env_indices=%s', episode_env_indices)

        # merge the vectorized outputs
        for wrapper in self._vectorized_wrappers:
            wrapper.merge_vectorized_output(
                episode_env_indices=episode_env_indices)

        evaluator_result = TianshouEvaluatorResult(observation_log_path=None)

        if self._params.observation_logger is not None:
            # we have observation log, return the path
            evaluator_result.observation_log_path = self._observation_log_filepath(
            )

        # upload to experiment logger
        if self._experiment_logger is not None:

            self._log.debug('saving results to experiment logger...')

            # log the result
            result = asdict(collect_stats)
            result = cleanup_numpy_from_dictionary(result)
            self._experiment_logger.log_dict(
                f'{self._params.log_parent_key}_result', result)

            # log using the vectorized loggers
            for wrapper in self._vectorized_wrappers:
                self._log.debug(
                    'uploading results for "%s" vectorized wrapper...',
                    wrapper)
                wrapper.log_merged_output(
                    parent_key=self._params.log_parent_key,
                    episode_count=len(episode_env_indices),
                    experiment_logger=self._experiment_logger)

        return evaluator_result

    def _observation_log_filepath(self):
        return os.path.join(self._params.output_dir, 'observation_log.csv')
