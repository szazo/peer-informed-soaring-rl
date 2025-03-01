from dataclasses import dataclass
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig
from typing import cast
from omegaconf import OmegaConf

from trainer.statistics import StatisticsJobParameters, StatisticsJob
from trainer.statistics.observation import register_observation_statistics_config_groups
from trainer.experiment_logger import register_experiment_logger_config_groups

from env.glider.single.statistics import register_singleglider_statistics_config_groups
from env.glider.base import register_glider_env_air_velocity_field_config_groups


@dataclass
class Config(StatisticsJobParameters):
    pass


cs = ConfigStore.instance()
register_observation_statistics_config_groups(config_store=cs)
register_experiment_logger_config_groups(base_group='logger', config_store=cs)
register_singleglider_statistics_config_groups(config_store=cs)
register_glider_env_air_velocity_field_config_groups(config_store=cs)

cs.store(name='job_config', node=Config)


def entrypoint(cfg: Config):

    output_dir = HydraConfig.get().runtime.output_dir
    cfg_obj = cast(Config, OmegaConf.to_object(cfg))

    # create the job
    statistics_job = StatisticsJob(params=cfg_obj, output_dir=output_dir)
    statistics_job.run()


@hydra.main(version_base=None,
            config_name='statistics_config',
            config_path='config')
def main(cfg: Config):
    entrypoint(cfg)


if __name__ == '__main__':
    main()
