import gc
import pathlib
import logging
from functools import partial
from dataclasses import dataclass, asdict
import hydra
import tianshou
import torch
from trainer.common.tianshou_vectorized_collector_factory import EnvSpaces
from utils import find_suitable_torch_device
from trainer.experiment_logger import (ExperimentLoggerInterface,
                                       ExperimentLoggerConfigBase)
from .api import (TianshouTrainerConfigBase)
from ..common import (TianshouVectorizedCollectorFactory, EnvSpaces,
                      TianshouEnviromentParameters, TianshouModelConfigBase,
                      ExperimentLoggerParameterStore,
                      ExperimentLoggerWeightStore)
from ..base.tianshou_job_base import TianshouJobBase, TianshouJobParametersBase
from .tianshou_experiment_log_adapter import TianshouExperimentLogAdapter
from .best_weights_saver import (BestWeightsSaver, MultiAgentBestWeightsSaver,
                                 SingleAgentBestWeightsSaver)
from utils import cleanup_numpy_from_dictionary


@dataclass(kw_only=True)
class TianshouTrainingJobParameters(TianshouJobParametersBase):
    device: str
    model: TianshouModelConfigBase
    train_env: TianshouEnviromentParameters
    test_env: TianshouEnviromentParameters
    trainer: TianshouTrainerConfigBase
    logger: ExperimentLoggerConfigBase


class TianshouTrainingJob(TianshouJobBase):

    _log: logging.Logger
    _params: TianshouTrainingJobParameters

    def __init__(self, params: TianshouTrainingJobParameters, output_dir: str):

        super().__init__(output_dir=output_dir)

        self._log = logging.getLogger(__class__.__name__)
        self._params = params

    def train(self):

        experiment_logger = self._create_experiment_logger(
            self._params.logger, name_override=self._params.experiment_name)

        try:
            # log the parameters
            ExperimentLoggerParameterStore(experiment_logger).save_parameters(
                asdict(self._params))

            # create the policy
            device = find_suitable_torch_device(self._params.device)

            gc.collect()
            self._log.debug('using device=%s', device)
            if device != 'cpu':
                self._log.debug('clearing cuda cache...')
                torch.cuda.empty_cache()

            GC_AFTER_EPOCH = False
            PRINT_MEMORY_INFO = False
            ENABLE_MEMORY_SNAPSHOT = False
            MEMORY_SNAPSHOT_DUMP_EPOCH = 1
            memory_recording_started = False
            if device != 'cpu' and ENABLE_MEMORY_SNAPSHOT:
                self._log.debug('starting memory monitoring')
                torch.cuda.memory._record_memory_history(max_entries=1000000)
                memory_recording_started = True

            policy = self._create_policy_model(
                device=device, policy_config=self._params.model)

            # create the trainer
            trainer, policy_instance, spaces = self._create_trainer_with_collectors(
                policy=policy, experiment_logger=experiment_logger)

            # log parameter counts
            experiment_logger.log_dict(
                'model_param_counts',
                self._count_model_parameters(policy_instance),
                log_as_str=True)

            self._log.debug('starting training...')

            # switch the policy to traing mode
            policy_instance.train()

            best_weight_saver = self._create_best_weights_saver(
                spaces=spaces,
                logger=experiment_logger,
                policy=policy_instance)
            result = {}

            for epoch_stats in trainer:
                self._log.info(
                    'epoch=%s,train_collect_stat=%s,training_stat=%s,test_collect_stat=%s',
                    epoch_stats.epoch, epoch_stats.train_collect_stat,
                    epoch_stats.training_stat, epoch_stats.test_collect_stat)
                self._log.debug('info=%s', epoch_stats.info_stat)

                if epoch_stats.test_collect_stat is not None:
                    best_weight_saver.save(
                        epoch=epoch_stats.epoch,
                        test_stats=epoch_stats.test_collect_stat)

                result = asdict(epoch_stats.info_stat)

                if PRINT_MEMORY_INFO:
                    # print memory info
                    print(
                        f'memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB'
                    )
                    print(
                        f'memory cached: {torch.cuda.memory_reserved() / (1024 ** 2)} MB'
                    )
                    print(
                        f'max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB'
                    )
                    print(
                        f'max memory cached: {torch.cuda.max_memory_reserved() / (1024 ** 2)} MB'
                    )
                    print(
                        f'active: {torch.cuda.memory_stats().get("active_bytes.all.current", 0) / (1024 ** 2)} MB'
                    )
                    print(
                        f'active max: {torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / (1024 ** 2)} MB'
                    )

                if GC_AFTER_EPOCH:
                    gc.collect()
                    torch.cuda.empty_cache()

                    if PRINT_MEMORY_INFO:
                        # print memory info again
                        print(f'-> after cache clear')
                        print(
                            f'memory allocated: {torch.cuda.memory_allocated() / (1024 ** 2)} MB'
                        )
                        print(
                            f'memory cached: {torch.cuda.memory_reserved() / (1024 ** 2)} MB'
                        )
                        print(
                            f'max memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 2)} MB'
                        )
                        print(
                            f'max memory cached: {torch.cuda.max_memory_reserved() / (1024 ** 2)} MB'
                        )
                        print(
                            f'active: {torch.cuda.memory_stats().get("active_bytes.all.current", 0) / (1024 ** 2)} MB'
                        )
                        print(
                            f'active max: {torch.cuda.memory_stats().get("active_bytes.all.peak", 0) / (1024 ** 2)} MB'
                        )

                if memory_recording_started and epoch % MEMORY_SNAPSHOT_DUMP_EPOCH == 0:
                    # memory snapshot
                    try:
                        memory_dump_filename = f'cuda_snapshot_{epoch}.pickle'
                        print(
                            f'creating memory snapshot: {memory_dump_filename}'
                        )
                        torch.cuda.memory._dump_snapshot(memory_dump_filename)
                    except:
                        print(
                            f'failed to create memory snapshot: {memory_dump_filename}'
                        )

            self._log.debug('training finished; result=%s', result)

            if memory_recording_started == True:
                torch.cuda.memory._record_memory_history(enabled=None)

            # convert all numpy specific type to pure python (e.g. np.float64 to float)
            result = cleanup_numpy_from_dictionary(result)
            experiment_logger.log_dict('train_result', result, log_as_str=True)

            # evaluate if necessary
            best_state_dict = ExperimentLoggerWeightStore(
                experiment_logger).load_best_weights(map_device=device)

            self._evaluate(policy=policy,
                           evaluators=self._params.evaluators,
                           experiment_logger=experiment_logger,
                           weights_state_dict=best_state_dict)

            self._finished(success=True, logger=experiment_logger)

            return result
        except Exception as e:
            self._log.error('error occurred during training; exception=%s', e)
            print('error occurred during training', e, flush=True)
            self._finished(success=False, logger=experiment_logger)
            raise

    def _finished(self, success: bool, logger: ExperimentLoggerInterface):
        self._log.debug('job finished; success=%s', success)

        logger.stop(success=success)

    def _create_best_weights_saver(
            self, spaces: EnvSpaces, logger: ExperimentLoggerInterface,
            policy: tianshou.policy.BasePolicy) -> BestWeightsSaver:

        base_path = self._best_weights_dir()
        if spaces.is_multi_agent:

            saver = MultiAgentBestWeightsSaver(
                possible_agent_ids=spaces.possible_agents,
                experiment_logger=logger,
                policy=policy,
                base_path=base_path)
        else:

            saver = SingleAgentBestWeightsSaver(experiment_logger=logger,
                                                policy=policy,
                                                base_path=base_path)

        return saver

    def _best_weights_dir(self):

        checkpoint_dir = pathlib.Path(self._output_dir) / 'checkpoint'
        return checkpoint_dir

    def _create_trainer_with_collectors(
            self, policy: partial[tianshou.policy.BasePolicy],
            experiment_logger: ExperimentLoggerInterface):

        collector_factory = TianshouVectorizedCollectorFactory()

        # create the vectorized environemnts
        self._log.debug('creating train vectorized environments...')
        train_vectorized_env, spaces = collector_factory.create_vectorized_environment(
            vectorized_params=self._params.train_env.vectorized,
            env_config=self._params.train_env.env,
            seed=None)

        self._log.debug('creating test vectorized environments...')
        test_vectorized_env, _ = collector_factory.create_vectorized_environment(
            vectorized_params=self._params.test_env.vectorized,
            env_config=self._params.test_env.env,
            seed=None)

        # initialize the policy with the spaces
        self._log.debug('initializing the policy; spaces=%s', spaces)

        policy_instance = policy(observation_space=spaces.observation_space,
                                 action_space=spaces.action_space)

        self._log.debug('policy initialized; type=%s', type(policy_instance))

        # create the collectors
        self._log.debug('creating train collector...')
        train_collector = collector_factory.create_collector(
            collector_params=self._params.train_env.collector,
            vectorized_env=train_vectorized_env,
            policy=policy_instance)

        self._log.debug('creating test collector...')

        test_collector = collector_factory.create_collector(
            collector_params=self._params.test_env.collector,
            vectorized_env=test_vectorized_env,
            policy=policy_instance)

        # test_collector.collect(n_episode=1, render=0.001)

        self._log.debug('creating trainer...')
        # create tianshou logger on top of experiment logger
        tianshou_log_adapter = TianshouExperimentLogAdapter(
            train_step_interval=1,
            test_step_interval=1,
            update_step_interval=1,
            experiment_logger=experiment_logger,
            is_multi_agent=spaces.is_multi_agent,
            multi_agent_possible_agents=spaces.possible_agents)

        trainer = self._create_trainer(trainer_config=self._params.trainer,
                                       policy=policy_instance,
                                       train_collector=train_collector,
                                       test_collector=test_collector,
                                       tianshou_logger=tianshou_log_adapter)

        self._log.debug('trainer created')

        return trainer, policy_instance, spaces

    def _create_trainer(
        self, trainer_config: TianshouTrainerConfigBase,
        policy: tianshou.policy.BasePolicy,
        train_collector: tianshou.data.Collector,
        test_collector: tianshou.data.Collector,
        tianshou_logger: tianshou.utils.BaseLogger
    ) -> tianshou.trainer.BaseTrainer:

        trainer_partial = hydra.utils.instantiate(trainer_config,
                                                  _convert_='object')
        trainer = trainer_partial(policy=policy,
                                  train_collector=train_collector,
                                  test_collector=test_collector,
                                  logger=tianshou_logger)

        return trainer

    def _count_model_parameters(self, model: torch.nn.Module):
        result = {}

        total_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            param_count = parameter.numel()
            result[name] = param_count
            total_params += param_count

        result['_total_'] = total_params

        return result
