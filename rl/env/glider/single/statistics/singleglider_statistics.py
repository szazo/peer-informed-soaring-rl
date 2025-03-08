import logging
from dataclasses import dataclass
import pandas as pd
import sigfig
from trainer.statistics.observation import ObservationStatisticsPlugin


@dataclass
class SingleGliderStatisticsParameters:
    pass


class SingleGliderStatistics(ObservationStatisticsPlugin):

    _log: logging.Logger
    _params: SingleGliderStatisticsParameters

    def __init__(self, **kwargs):
        self._log = logging.getLogger(__class__.__name__)
        self._params = SingleGliderStatisticsParameters(**kwargs)

    def run(self, observation_log: pd.DataFrame):

        self._log.debug('run; observation_log size=%s', observation_log.size)

        non_aggregated_stats = self._create_non_aggregated_stats(
            observation_log)

        return non_aggregated_stats

    def _create_non_aggregated_stats(
            self, observation_log_df: pd.DataFrame) -> pd.DataFrame:

        columns = ['distance_from_core_m', 'velocity_earth_m_per_s_z']

        stat_df = observation_log_df[columns]

        mean_df = pd.DataFrame(stat_df.mean(), columns=['mean'])
        std_df = pd.DataFrame(stat_df.std(), columns=['std'])

        mean_std_df = pd.concat([mean_df, std_df], axis=1)

        mean_std_df['sigfig'] = mean_std_df.apply(
            lambda row: self._sigfig_mean_std(row['mean'], row['std']), axis=1)

        return mean_std_df

    def _sigfig_mean_std(self, mean: float, std: float):
        return sigfig.round(mean, uncertainty=std, spacing=3, spacer=',')
