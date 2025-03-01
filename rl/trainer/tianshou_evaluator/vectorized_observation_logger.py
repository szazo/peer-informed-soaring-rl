import os
import logging
import gymnasium
import hydra
import pandas as pd
from .vectorized_env_wrapper import VectorizedEnvWrapper, ExperimentLoggerInterface
from .api import ObservationLoggerParameters
from .observation_log_wrapper import SingleAgentObservationLogWrapper
from .multi_agent_observation_log_wrapper import MultiAgentObservationLogWrapper


class VectorizedObservationLogger(VectorizedEnvWrapper):

    _log: logging.Logger
    _logger_params: ObservationLoggerParameters

    _output_observation_log_filepath: str

    def __init__(self, output_dir: str, output_observation_log_filepath: str,
                 logger_params: ObservationLoggerParameters):

        super().__init__(output_dir=output_dir)
        self._output_observation_log_filepath = output_observation_log_filepath

        self._log = logging.getLogger(__class__.__name__)

        self._logger_params = logger_params

    def wrap_env(self, env: gymnasium.Env,
                 vectorized_index: int) -> gymnasium.Env:

        # create environment specific observation logger (if we have)
        env_observation_logger = None
        if self._logger_params.env is not None:
            env_observation_logger = hydra.utils.instantiate(
                self._logger_params.env, _convert_='object')

        # wrap the environment
        vectorized_output_dir = self._vectorized_output_dir(vectorized_index)
        output_filepath = self._vectorized_observation_log_filepath(
            vectorized_output_dir)

        if hasattr(env, 'possible_agents'):
            # multi agent
            wrapped = MultiAgentObservationLogWrapper(
                env,
                log_buffer_size=self._logger_params.log_buffer_size,
                observation_logger=env_observation_logger,
                output_filepath=output_filepath)
        else:
            wrapped = SingleAgentObservationLogWrapper(
                env=env,
                log_buffer_size=self._logger_params.log_buffer_size,
                observation_logger=env_observation_logger,
                output_filepath=output_filepath)

        return wrapped

    def merge_vectorized_output(self, episode_env_indices: list[int]):
        env_episode_numbers = self._collect_vectorized_env_episode_numbers(
            episode_env_indices)

        self._log.debug(
            'merging vectorized observation logs; env_episode_numbers=%s',
            env_episode_numbers)

        merged_observation_log = pd.DataFrame()

        for env_index, episode_numbers in enumerate(env_episode_numbers):

            self._log.debug('processing env_index=%s,episode_numbers=%s',
                            env_index, episode_numbers)

            env_observation_log_path = self._vectorized_observation_log_filepath(
                self._vectorized_output_dir(env_index))

            # load the observation log
            df = pd.read_csv(env_observation_log_path)

            for old_episode, new_episode in enumerate(episode_numbers):

                # replace episode numbers (old_episode: episode index inside the vectorized env)
                df.loc[df['episode'] == old_episode,
                       'new_episode'] = new_episode

                # set the vectorized env id
                df.loc[df['episode'] == old_episode, 'venv_id'] = env_index

            # merge
            merged_observation_log = pd.concat([merged_observation_log, df])

        # remove rename and move vectorized columns
        merged_observation_log = self._fix_merged_observation_log_columns(
            merged_observation_log)

        # add additional columns
        for i, (column_name, column_value) in enumerate(
                self._logger_params.additional_columns.items()):
            merged_observation_log.insert(loc=i,
                                          column=column_name,
                                          value=column_value)

        # save the merged observation log
        merged_path = self._output_observation_log_filepath
        self._ensure_dir(os.path.dirname(merged_path))
        self._log.debug('saving merged observation log; path=%s', merged_path)

        # save to csv, skip the new index of the merged df
        merged_observation_log.to_csv(merged_path, index=False)

    def _fix_merged_observation_log_columns(
            self, merged_observation_log: pd.DataFrame):
        # rename old venv specific episode column
        merged_observation_log = merged_observation_log.rename(
            columns={'episode': 'venv_episode'})

        # move venv id
        try:
            venv_id_col = merged_observation_log.pop('venv_id')
            merged_observation_log.insert(0, 'venv_id',
                                          venv_id_col.astype(int))
        except:
            print(merged_observation_log)
            raise

        # move and rename new(merged) episode column
        new_episode_col = merged_observation_log.pop('new_episode')
        merged_observation_log.insert(0, 'episode',
                                      new_episode_col.astype(int))

        # move index field
        merged_observation_log.insert(1, "index",
                                      merged_observation_log.pop("index"))

        # sort
        merged_observation_log = merged_observation_log.sort_values(
            by=['episode', 'index'])

        # reset the index after the merge
        merged_observation_log = merged_observation_log.reset_index(drop=True)

        return merged_observation_log

    def log_merged_output(self, parent_key: str, episode_count: int,
                          experiment_logger: ExperimentLoggerInterface):
        merged_path = self._output_observation_log_filepath
        experiment_logger.log_file(f'{parent_key}_observation_log',
                                   merged_path)

    def _vectorized_observation_log_filepath(self, output_dir: str):
        filepath = os.path.join(output_dir, 'vobservation_log.csv')
        return filepath
