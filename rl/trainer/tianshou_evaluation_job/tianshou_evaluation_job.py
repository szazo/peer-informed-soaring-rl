from dataclasses import dataclass, asdict
from omegaconf import MISSING
import time
import logging
from utils import find_suitable_torch_device
from ..experiment_logger import ExperimentLoggerConfigBase, ExperimentLoggerInterface
from ..base.tianshou_job_base import TianshouJobBase, TianshouJobParametersBase
from ..common import (ExperimentLoggerParameterStore,
                      ExperimentLoggerWeightStore)


@dataclass(kw_only=True)
class TianshouEvaluationJobParameters(TianshouJobParametersBase):
    device: str = 'cpu'
    model_source_logger: ExperimentLoggerConfigBase = MISSING
    checkpoint_name: str = 'best_weights'
    logger: ExperimentLoggerConfigBase = MISSING


class TianshouEvaluationJob(TianshouJobBase):

    _params: TianshouEvaluationJobParameters

    def __init__(self, params: TianshouEvaluationJobParameters,
                 output_dir: str):

        super().__init__(output_dir=output_dir)

        self._log = logging.getLogger(__class__.__name__)
        self._params = params

    def evaluate(self):

        # create the logger for the evaluation
        experiment_logger = self._create_experiment_logger(
            self._params.logger, name_override=self._params.experiment_name)
        try:
            self._log.debug('evaluating (experiment_name=%s)...',
                            self._params.experiment_name)
            t1 = time.perf_counter(), time.process_time()

            model_source_logger = self._create_experiment_logger(
                self._params.model_source_logger)

            self._log.debug('loading parameters from the experiment logger...')
            parameter_store = ExperimentLoggerParameterStore(
                experiment_logger=model_source_logger)
            parameters = parameter_store.load_parameters()

            model_parameters = parameters['model']
            self._log.debug('parameters loaded; parameters.model=%s',
                            model_parameters)

            self._log.debug('creating model based on the parameters...')

            # create the policy
            device = find_suitable_torch_device(self._params.device)
            policy = self._create_policy_model(device=device,
                                               policy_config=model_parameters)

            # create the model based on parameters from the logger
            self._log.debug('model loaded')

            # load the best weights
            self._log.debug('loading best weights...')
            weights_store = ExperimentLoggerWeightStore(
                experiment_logger=model_source_logger)
            weights_state_dict = weights_store.load_weights(
                checkpoint_name=self._params.checkpoint_name,
                map_device=device)

            self._log.debug('weights loaded; running the evaluators...')

            # log the parameters
            ExperimentLoggerParameterStore(experiment_logger).save_parameters(
                asdict(self._params))

            # evaluate
            self._evaluate(policy=policy,
                           evaluators=self._params.evaluators,
                           experiment_logger=experiment_logger,
                           weights_state_dict=weights_state_dict)

            # run statistics
            self._run_statistics(statistics=self._params.statistics,
                                 source_logger=experiment_logger,
                                 target_logger=experiment_logger)

            t2 = time.perf_counter(), time.process_time()

            self._log.debug(
                f'evaluation finished; perf_counter: {t2[0]-t1[0]:.2f}s, process_time: {t2[1]-t1[1]:.2f}s'
            )
            self._finished(success=True, logger=experiment_logger)

        except Exception as e:
            self._log.error('error occurred during evaluation; exception=%s',
                            e)
            self._finished(success=False, logger=experiment_logger)
            raise

    def _finished(self, success: bool, logger: ExperimentLoggerInterface):
        self._log.debug('job finished; success=%s', success)

        logger.stop(success=success)
