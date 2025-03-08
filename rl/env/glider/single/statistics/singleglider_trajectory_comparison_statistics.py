import logging
from dataclasses import dataclass
import pandas as pd
import sigfig
from thermal.api import AirVelocityFieldInterface
from trainer.statistics.observation import ObservationStatisticsPlugin
from .trajectory_to_observation_log_converter import (
    TrajectoryToObservationLogConverterParameters,
    TrajectoryToObservationLogConverter)


@dataclass
class StatisticsKeys:
    observation: str
    trajectory: str


@dataclass
class SingleGliderTrajectoryComparisonStatisticsParameters:
    trajectory_path: str
    trajectory_converter: TrajectoryToObservationLogConverterParameters
    keys: StatisticsKeys
    additional_columns: dict[str, str]


class SingleGliderTrajectoryComparisonStatistics(ObservationStatisticsPlugin):

    _log: logging.Logger
    _params: SingleGliderTrajectoryComparisonStatisticsParameters
    _trajectory_converter: TrajectoryToObservationLogConverter

    def __init__(self,
                 trajectory_air_velocity_field: AirVelocityFieldInterface,
                 **kwargs):
        self._log = logging.getLogger(__class__.__name__)
        self._params = SingleGliderTrajectoryComparisonStatisticsParameters(
            **kwargs)

        self._trajectory_converter = TrajectoryToObservationLogConverter(
            air_velocity_field=trajectory_air_velocity_field,
            params=self._params.trajectory_converter)

        self._log.debug('initialized; trajectory_air_velocity_field=%s',
                        trajectory_air_velocity_field)

    def run(self, observation_log: pd.DataFrame):

        additional_columns = self._params.additional_columns
        index_value = additional_columns.get('_index_', 'stat')
        additional_columns = {
            k: v
            for k, v in additional_columns.items() if k != '_index_'
        }

        # observation statistics
        self._log.debug(
            'creating statistics for observation; observation_log size=%s',
            observation_log.size)
        observation_stats_df = self._create_non_aggregated_stats(
            observation_log, index_value=index_value)

        # trajectory statistics
        trajectory_df = self._convert_trajectory()
        trajectory_stats_df = self._create_non_aggregated_stats(
            trajectory_df, index_value=index_value)

        index_keys = [
            self._params.keys.observation, self._params.keys.trajectory
        ]

        # concate based on matching stat columns
        column_names = list(
            observation_stats_df.columns.levels[0]) if isinstance(
                observation_stats_df.columns, pd.MultiIndex) else list(
                    observation_stats_df.columns)

        stat_df = pd.DataFrame()
        column_dfs = []
        for column_name in column_names:
            column_df = pd.concat((observation_stats_df[column_name],
                                   trajectory_stats_df[column_name]),
                                  axis=1,
                                  keys=index_keys)
            column_dfs.append(column_df)

        stat_df = pd.concat(column_dfs, axis=1, keys=column_names)

        # add additional columns
        i = 0
        for column_name, column_value in additional_columns.items():
            stat_df.insert(loc=i, column=column_name, value=column_value)
            i += 1

        return stat_df

    def _convert_trajectory(self):
        self._log.debug('converting trajectory; path=%s',
                        self._params.trajectory_path)

        trajectory_df = pd.read_csv(self._params.trajectory_path)

        converted_df = self._trajectory_converter.convert(trajectory_df)
        return converted_df

    def _create_non_aggregated_stats(self, observation_log_df: pd.DataFrame,
                                     index_value: str | int) -> pd.DataFrame:

        columns = ['distance_from_core_m', 'velocity_earth_m_per_s_z']

        stat_df = observation_log_df[columns]

        mean_df = pd.DataFrame(stat_df.mean(), columns=['mean'])
        std_df = pd.DataFrame(stat_df.std(), columns=['std'])

        mean_std_df = pd.concat([mean_df, std_df], axis=1)

        mean_std_df['sigfig'] = mean_std_df.apply(
            lambda row: self._sigfig_mean_std(row['mean'], row['std']), axis=1)

        feature_dfs = []
        for column in columns:
            column_df = observation_log_df[column]
            mean = column_df.mean()
            std = column_df.std()
            feature_df = pd.DataFrame(
                {
                    'mean': mean,
                    'std': std,
                    'sigfig': self._sigfig_mean_std(mean, std)
                },
                index=[index_value])
            feature_dfs.append(feature_df)

        out_df = pd.concat(feature_dfs, axis=1, keys=columns)
        return out_df

    def _sigfig_mean_std(self, mean: float, std: float):
        return sigfig.round(mean, uncertainty=std, spacing=3, spacer=',')
