from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from trainer.tianshou_trainer_job import (
    TianshouTrainingJob, TianshouTrainingJobParameters,
    register_tianshou_trainer_config_groups)
from trainer.base import register_base_job_config_groups
from trainer.statistics.observation import register_observation_statistics_config_groups
from trainer.tianshou_evaluator import register_tianshou_evaluator_config_groups
from trainer.experiment_logger import register_experiment_logger_config_groups
from model.ppo_custom_transformer_model_config import (
    register_ppo_transformer_model_config_groups)

from model.multi_agent_model_config import register_multi_agent_model_config_groups

from env.glider.single import register_singleglider_env_config_groups
from env.glider.single.statistics import register_singleglider_statistics_config_groups
from env.glider.multi.multiglider_env_config import register_multiglider_env_config_groups


@dataclass
class JobConfig(TianshouTrainingJobParameters):
    pass


cs = ConfigStore.instance()
register_base_job_config_groups(base_group='', config_store=cs)
register_experiment_logger_config_groups(base_group='logger', config_store=cs)

# trainer
register_tianshou_trainer_config_groups(base_group='trainer', config_store=cs)

# model
register_ppo_transformer_model_config_groups(base_group='model',
                                             config_store=cs)

# evaluator
register_tianshou_evaluator_config_groups(base_group='evaluator',
                                          config_store=cs)

# singleglider
register_singleglider_env_config_groups(config_store=cs)

# statistics
register_observation_statistics_config_groups(config_store=cs)
register_singleglider_statistics_config_groups(config_store=cs)

# multi agent model
register_multi_agent_model_config_groups(base_group='model', config_store=cs)

# multi agent env
register_multiglider_env_config_groups(config_store=cs)

cs.store(name='job_config', node=JobConfig)


def exp_main(cfg: JobConfig):

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    cfg_obj = OmegaConf.to_object(cfg)

    # create the job
    training_job = TianshouTrainingJob(params=cfg_obj, output_dir=output_dir)
    training_job.train()


@hydra.main(version_base=None,
            config_name='train_config',
            config_path='config')
def exp_entrypoint(cfg: JobConfig):
    exp_main(cfg)


if __name__ == '__main__':
    exp_entrypoint()
