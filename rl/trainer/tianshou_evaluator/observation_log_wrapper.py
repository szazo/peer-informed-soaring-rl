from typing import TypeVar, Any, SupportsFloat
from typing import SupportsFloat
import logging
from .api import ObservationLogger
from .observation_log_wrapper_base import ObservationLogWrapperBase

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


class SingleAgentObservationLogWrapper(ObservationLogWrapperBase[ObsType,
                                                                 ActType,
                                                                 ObsType,
                                                                 ActType]):

    _log: logging.Logger
    _logging: bool

    def __init__(self, env, log_buffer_size: int,
                 observation_logger: ObservationLogger | None,
                 output_filepath: str):

        super().__init__(env,
                         log_buffer_size=log_buffer_size,
                         observation_logger=observation_logger,
                         output_filepath=output_filepath)

        self._log = logging.getLogger(__class__.__name__)
        self._logging = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:

        obs, info = self.env.reset(seed=seed, options=options)
        self._prev_obs = obs
        self._prev_info = info
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        obs, reward, terminated, truncated, info = self.env.step(action)

        log_buffer = self._ensure_log_buffer()

        # log using the previous observation/info and with the new action/reward
        self._log_to_buffer(
            obs_before=self._prev_obs,
            info_before=self._prev_info,
            action=action,
            reward=reward,
            terminated=False,
            truncated=False,
        )

        # save the last observation and info
        if terminated or truncated:
            self._log_to_buffer(
                obs_before=obs,
                info_before=info,
                # save the same action (what else can be good?)
                action=action,
                reward=0,  # zero reward here for the last obs
                terminated=terminated,
                truncated=truncated,
            )

        self._prev_obs = obs
        self._prev_info = info
        return obs, reward, terminated, truncated, info
