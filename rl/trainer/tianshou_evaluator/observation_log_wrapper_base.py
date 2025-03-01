from typing import TypeVar, Any, SupportsFloat, Generic, cast
import logging
import os
import warnings
import pathlib
from typing import SupportsFloat
import tianshou
import gymnasium
from tianshou.data.types import RolloutBatchProtocol
from trainer.tianshou_evaluator.convert_buffer_to_observation_log import convert_buffer_to_observation_log
from .api import ObservationLogger

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')

LogObsType = TypeVar('LogObsType')
LogActType = TypeVar('LogActType')


class ObservationLogWrapperBase(Generic[ObsType, ActType, LogObsType,
                                        LogActType],
                                gymnasium.Wrapper[ObsType, ActType, ObsType,
                                                  ActType]):

    _log: logging.Logger

    _observation_logger: ObservationLogger | None
    _output_filepath: str

    _log_buffer_size: int
    _log_buffer: tianshou.data.ReplayBuffer | None

    def __init__(self, env: gymnasium.Env[ObsType,
                                          ActType], log_buffer_size: int,
                 observation_logger: ObservationLogger | None,
                 output_filepath: str):

        super().__init__(env)

        self._log = logging.getLogger(__class__.__name__)

        self._log_buffer_size = log_buffer_size
        self._observation_logger = observation_logger
        self._output_filepath = output_filepath
        self._log_buffer = None

    def _log_to_buffer(self, obs_before: LogObsType, info_before: dict[str,
                                                                       Any],
                       action: LogActType, reward: SupportsFloat,
                       terminated: bool, truncated: bool):

        log_buffer = self._ensure_log_buffer()
        batch = tianshou.data.Batch(obs=obs_before,
                                    info=info_before,
                                    act=action,
                                    rew=reward,
                                    terminated=terminated,
                                    truncated=truncated)
        log_buffer.add(cast(RolloutBatchProtocol, batch))

        if len(log_buffer) > self._log_buffer_size:
            raise Exception(
                f'Observation log buffer overflow; max_size={self._buffer_size}'
            )

    def _ensure_log_buffer(self):
        if self._log_buffer is None:
            self._log_buffer = tianshou.data.ReplayBuffer(
                size=self._log_buffer_size, ignore_obs_next=True)

        return self._log_buffer

    def close(self):

        super().close()

        # ignore pandas 1.x numpy deprecation warning
        warnings.filterwarnings("ignore",
                                category=DeprecationWarning,
                                module='^pandas.core.dtypes*')

        if self._log_buffer is not None:
            self._log.debug('close; saving dataframe to file \'%s\'...',
                            self._output_filepath)
            df = self._to_dataframe()

            # create the directory
            parent_dir = os.path.dirname(self._output_filepath)
            pathlib.Path(parent_dir).mkdir(parents=True, exist_ok=True)

            # save to csv, do not save the index
            df.to_csv(self._output_filepath, index=False)
        else:
            self._log.debug('close; there is no observation log; skipping')

    def _to_dataframe(self):

        buffer = self._ensure_log_buffer()

        df = convert_buffer_to_observation_log(
            buffer=buffer, observation_logger=self._observation_logger)

        return df
