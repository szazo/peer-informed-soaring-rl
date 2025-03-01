from typing import Any
import os
from dataclasses import dataclass
import logging
import pandas as pd

from env.glider.single.statistics.trajectory_to_observation_log_converter import (
    TrajectoryToObservationLogConverter,
    TrajectoryToObservationLogConverterParameters)
from omegaconf import MISSING
from thermal.api import AirVelocityFieldInterface
from utils.custom_job_api import CustomJobBase, CustomJobBaseConfig


@dataclass
class TrajectoryToObservationLogConverterJobParameters:
    trajectory_path: str
    trajectory_converter: TrajectoryToObservationLogConverterParameters
    additional_columns: dict[str, str]


@dataclass(kw_only=True)
class TrajectoryToObservationLogConverterJobConfig(CustomJobBaseConfig):
    params: TrajectoryToObservationLogConverterJobParameters
    air_velocity_field: Any = MISSING
    _target_: str = 'env.glider.utils.TrajectoryToObservationLogConverterJob'


class TrajectoryToObservationLogConverterJob(CustomJobBase):

    _log: logging.Logger
    _params: TrajectoryToObservationLogConverterJobParameters
    _trajectory_converter: TrajectoryToObservationLogConverter

    def __init__(self, air_velocity_field: AirVelocityFieldInterface,
                 params: TrajectoryToObservationLogConverterJobParameters):
        self._log = logging.getLogger(__class__.__name__)
        self._params = params

        self._trajectory_converter = TrajectoryToObservationLogConverter(
            air_velocity_field=air_velocity_field,
            params=self._params.trajectory_converter)

        self._log.debug('initialized; trajectory_air_velocity_field=%s',
                        air_velocity_field)

    def run(self, output_dir: str):

        self._log.debug('converting trajectory to observation log format...')
        trajectory_df = self._convert_trajectory()

        # add additional columns
        for i, (column_name, column_value) in enumerate(
                self._params.additional_columns.items()):
            trajectory_df.insert(loc=i, column=column_name, value=column_value)

        trajectory_df.to_csv(os.path.join(output_dir, 'observation_log.csv'))

    def _convert_trajectory(self):
        self._log.debug('converting trajectory; path=%s',
                        self._params.trajectory_path)

        trajectory_df = pd.read_csv(self._params.trajectory_path)

        converted_df = self._trajectory_converter.convert(trajectory_df)
        return converted_df
