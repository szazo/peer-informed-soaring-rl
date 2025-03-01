from typing import cast
from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from trainer.tianshou_evaluation_job import (TianshouEvaluationJob,
                                             TianshouEvaluationJobParameters)

from trainer.tianshou_evaluator import register_tianshou_evaluator_config_groups
from trainer.experiment_logger import register_experiment_logger_config_groups
from trainer.statistics.observation import register_observation_statistics_config_groups
from trainer.base import register_base_job_config_groups

from env.glider.single import register_singleglider_env_config_groups
from env.glider.single.statistics import register_singleglider_statistics_config_groups

from model.multi_agent_model_config import register_multi_agent_model_config_groups
from env.glider.multi.multiglider_env_config import register_multiglider_env_config_groups


@dataclass
class Config(TianshouEvaluationJobParameters):
    pass


cs = ConfigStore.instance()
register_base_job_config_groups(base_group='', config_store=cs)
register_tianshou_evaluator_config_groups(base_group='evaluator',
                                          config_store=cs)
register_experiment_logger_config_groups(base_group='logger', config_store=cs)
register_singleglider_env_config_groups(config_store=cs)

register_observation_statistics_config_groups(config_store=cs)
register_singleglider_statistics_config_groups(config_store=cs)

# multi agent model
register_multi_agent_model_config_groups(base_group='model', config_store=cs)

# multi agent env
register_multiglider_env_config_groups(config_store=cs)

cs.store(name='job_config', node=Config)


def exp_main(cfg: Config):

    output_dir = HydraConfig.get().runtime.output_dir
    cfg_obj = cast(TianshouEvaluationJobParameters, OmegaConf.to_object(cfg))

    # create the job
    training_job = TianshouEvaluationJob(params=cfg_obj, output_dir=output_dir)
    training_job.evaluate()


@hydra.main(version_base=None, config_name='eval_config', config_path='config')
def exp_entrypoint(cfg: Config):
    exp_main(cfg)


if __name__ == '__main__':
    exp_entrypoint()
