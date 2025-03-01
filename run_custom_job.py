from typing import cast, Any
from dataclasses import dataclass
from omegaconf import OmegaConf, MISSING
import hydra
from hydra.core.config_store import ConfigStore
from hydra.core.hydra_config import HydraConfig

from utils import CustomJobBase, CustomJobBaseConfig, register_custom_job_config_group
from env.glider.base import register_glider_env_air_velocity_field_config_groups
from env.glider.utils.trajectory_to_observation_log_converter_job import TrajectoryToObservationLogConverterJobConfig
from thermal.visualization.thermal_vedo_sandbox import ThermalVedoSandboxJobConfig
from visualization.observation_log_player_job import ObservationLogPlayerJobConfig


@dataclass
class ExperimentConfig:
    job: CustomJobBaseConfig = MISSING


def initialize_config_store():

    # register config instances
    cs = ConfigStore.instance()
    cs.store(name='experiment_config', node=ExperimentConfig)

    # common config groups
    register_glider_env_air_velocity_field_config_groups(config_store=cs)

    # register custom jobs
    register_custom_job_config_group(
        'trajectory_to_observation_log_converter',
        TrajectoryToObservationLogConverterJobConfig, cs)

    register_custom_job_config_group('thermal_vedo_sandbox',
                                     ThermalVedoSandboxJobConfig, cs)

    register_custom_job_config_group('observation_log_player',
                                     ObservationLogPlayerJobConfig, cs)


initialize_config_store()


@hydra.main(version_base=None,
            config_name='thermal_baseline_mcts_config',
            config_path='config')
def exp_main(cfg: ExperimentConfig):

    experiment_config = cast(ExperimentConfig, OmegaConf.to_object(cfg))

    job: CustomJobBase = hydra.utils.instantiate(experiment_config.job,
                                                 _convert_='object')

    output_dir = HydraConfig.get().runtime.output_dir
    job.run(output_dir=output_dir)


if __name__ == '__main__':
    exp_main()
