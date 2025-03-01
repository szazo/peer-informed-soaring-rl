import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from thermal.api import AirVelocityFieldInterface
from trainer.statistics.observation import ObservationStatisticsPlugin
from .trajectory_to_observation_log_converter import (
    TrajectoryToObservationLogConverterParameters,
    TrajectoryToObservationLogConverter)


@dataclass
class SingleGliderTrajectoryMergeStatisticsParameters:
    trajectory_path: str
    trajectory_converter: TrajectoryToObservationLogConverterParameters
    additional_columns: dict[str, str]


# Merge observation log with prerecorded trajectory for the same air velocity field
class SingleGliderTrajectoryMergeStatistics(ObservationStatisticsPlugin):

    _log: logging.Logger
    _params: SingleGliderTrajectoryMergeStatisticsParameters
    _trajectory_converter: TrajectoryToObservationLogConverter

    def __init__(self,
                 trajectory_air_velocity_field: AirVelocityFieldInterface,
                 **kwargs):
        self._log = logging.getLogger(__class__.__name__)
        self._params = SingleGliderTrajectoryMergeStatisticsParameters(
            **kwargs)

        self._trajectory_converter = TrajectoryToObservationLogConverter(
            air_velocity_field=trajectory_air_velocity_field,
            params=self._params.trajectory_converter)

        self._log.debug('initialized; trajectory_air_velocity_field=%s',
                        trajectory_air_velocity_field)

    def run(self, observation_log: pd.DataFrame):

        self._log.debug('converting trajectory to observation log format...',
                        observation_log.size)
        trajectory_df = self._convert_trajectory()

        # add additional columns
        for i, (column_name, column_value) in enumerate(
                self._params.additional_columns.items()):
            trajectory_df.insert(loc=i, column=column_name, value=column_value)

        # select common columns
        _, idx_1, idx_2 = np.intersect1d(observation_log.columns,
                                         trajectory_df.columns,
                                         return_indices=True)
        common_columns = observation_log.columns[np.sort(idx_1)]
        self._log.debug('merging, only common columns will be used: %s',
                        common_columns)

        # concat
        result_df = pd.concat([trajectory_df, observation_log])
        result_df = result_df[common_columns]
        return result_df

    def _convert_trajectory(self):
        self._log.debug('converting trajectory; path=%s',
                        self._params.trajectory_path)

        trajectory_df = pd.read_csv(self._params.trajectory_path)

        converted_df = self._trajectory_converter.convert(trajectory_df)
        return converted_df
