from env.glider.base import SimulationBoxParameters
from utils.custom_job_api import CustomJobBase, CustomJobBaseConfig

from typing import Any
from dataclasses import dataclass
import logging
from omegaconf import MISSING
from thermal.api import AirVelocityFieldConfigBase, AirVelocityFieldInterface
from utils.custom_job_api import CustomJobBase, CustomJobBaseConfig
from visualization.scene_player import ScenePlayer, ScenePlayerParams
from .observation_log_scene_loader import ObservationLogSceneLoader


@dataclass(kw_only=True)
class ObservationLogPlayerJobConfig(CustomJobBaseConfig):
    observation_log_filepath: str
    scene: ScenePlayerParams

    _target_: str = 'visualization.observation_log_player_job.ObservationLogPlayerJob'
    _recursive_: bool = False


class ObservationLogPlayerJob(CustomJobBase):

    _log: logging.Logger

    _config: ObservationLogPlayerJobConfig

    def __init__(self, **config_kwargs):

        config = ObservationLogPlayerJobConfig(**config_kwargs)

        self._log = logging.getLogger(__class__.__name__)
        self._log.debug('initialized; air_velocity_field=%s',
                        config.scene.air_velocity_field)

        self._config = config

    def run(self, output_dir: str):
        self._log.debug('run; output_dir=%s', output_dir)

        loader = ObservationLogSceneLoader()
        scenes = loader.load(self._config.observation_log_filepath)

        self._log.debug('%d scene(s) found in the observation log',
                        len(scenes))

        player = ScenePlayer(params=self._config.scene, scenes=scenes)
        player.show()
