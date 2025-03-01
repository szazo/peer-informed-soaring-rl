from typing import TypeVar, Any, SupportsFloat, Generic, cast
import numpy as np
import gymnasium
from trainer.multi_agent.create_agent_id_index_mapping import create_agent_id_index_mapping, MultiAgentEnvProtocol
from .api import ObservationLogger
from .observation_log_wrapper_base import ObservationLogWrapperBase

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType', bound=np.ndarray)
AgentIDType = TypeVar('AgentIDType', bound=str)


class MultiAgentObservationLogWrapper(
        Generic[AgentIDType, ObsType,
                ActType], ObservationLogWrapperBase[dict[AgentIDType, ObsType],
                                                    dict[AgentIDType, ActType],
                                                    ObsType, ActType]):

    _index_to_agent_id_map: list[AgentIDType]
    _agent_id_index_map: dict[AgentIDType, int]

    _prev_obs: dict[AgentIDType, ObsType] | None
    _prev_info: dict[AgentIDType, dict[str, Any]] | None

    _current_episode: int
    _current_step_index: int
    _agent_step_indices: dict[str, int]

    def __init__(self, env: gymnasium.Env[dict[AgentIDType, ObsType],
                                          dict[AgentIDType,
                                               ActType]], log_buffer_size: int,
                 observation_logger: ObservationLogger | None,
                 output_filepath: str):

        super().__init__(env,
                         log_buffer_size=log_buffer_size,
                         observation_logger=observation_logger,
                         output_filepath=output_filepath)

        self._prev_obs = None
        self._prev_info = None

        # assert 'possible_agents' in env
        assert isinstance(env, MultiAgentEnvProtocol)
        self._index_to_agent_id_map = env.possible_agents.copy()
        self._agent_id_index_map = create_agent_id_index_mapping(env)

        self._current_episode = 0
        self._current_step_index = 0
        self._agent_step_indices = {
            agent_id: 0
            for agent_id in env.possible_agents
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentIDType, ObsType], dict[str, Any]]:

        obs, info = self.env.reset(seed=seed, options=options)

        casted_info = cast(dict[AgentIDType, dict[str, Any]], info)

        self._prev_obs = obs
        self._prev_info = casted_info

        if self._current_step_index > 0:
            # only count episode if there were steps
            self._current_episode += 1
            self._current_step_index = 0

        return obs, info

    def step(
        self, action: dict[AgentIDType, ActType]
    ) -> tuple[dict[AgentIDType, ObsType], SupportsFloat, bool, bool, dict[
            str, Any]]:

        assert self._prev_obs is not None and self._prev_info is not None

        obs, reward, terminated, truncated, info = self.env.step(action)

        assert isinstance(reward, np.ndarray)
        casted_info = cast(dict[AgentIDType, dict[str, Any]], info)

        self._write_to_buffer(prev_obs=self._prev_obs,
                              prev_info=self._prev_info,
                              actions=action,
                              rewards=reward,
                              obs=obs,
                              info=casted_info)

        self._prev_obs = obs
        self._prev_info = casted_info

        return obs, reward, terminated, truncated, info

    def _write_to_buffer(self, prev_obs: dict[AgentIDType, ObsType],
                         prev_info: dict[AgentIDType, dict[str, Any]],
                         actions: dict[AgentIDType,
                                       ActType], rewards: np.ndarray,
                         obs: dict[AgentIDType,
                                   ObsType], info: dict[AgentIDType,
                                                        dict[str, Any]]):

        for agent_id, agent_prev_obs in prev_obs.items():

            agent_prev_info = prev_info[agent_id]

            if not agent_prev_info['agent_mask']:
                # agent does not exists at the end of previous step
                continue

            agent_obs = obs[agent_id]
            agent_info = info[agent_id]

            agent_index = self._agent_id_index_map[agent_id]
            agent_reward = rewards[agent_index]
            agent_action = actions[agent_id]

            # before the step it is not terminated / truncated
            self._log_agent_to_buffer(
                episode=self._current_episode,
                step_index=self._current_step_index,
                agent_id=agent_id,
                agent_step_index=self._agent_step_indices[agent_id],
                obs_before=agent_prev_obs,
                info_before=agent_prev_info,
                action=agent_action,
                reward=agent_reward,
                terminated=False,
                truncated=False)

            self._current_step_index += 1
            self._agent_step_indices[agent_id] += 1

            # write the final observation/info if it is terminated
            agent_terminated = agent_info['terminated']
            agent_truncated = agent_info['truncated']

            if agent_terminated or agent_truncated:
                self._log_agent_to_buffer(
                    episode=self._current_episode,
                    step_index=self._current_step_index,
                    agent_id=agent_id,
                    agent_step_index=self._agent_step_indices[agent_id],
                    obs_before=agent_obs,
                    info_before=agent_info,
                    action=
                    agent_action,  # use the previous action as placeholder
                    reward=0.,  # use zero reward
                    terminated=agent_terminated,
                    truncated=agent_truncated)

    def _log_agent_to_buffer(self, episode: int, step_index: int,
                             agent_id: str, agent_step_index: int,
                             obs_before: ObsType, info_before: dict[str, Any],
                             action: ActType, reward: SupportsFloat,
                             terminated: bool, truncated: bool):

        info_copy = info_before.copy()
        info_copy['episode'] = episode
        info_copy['step_index'] = step_index
        info_copy['agent_id'] = agent_id
        info_copy['agent_step_index'] = agent_step_index

        self._log_to_buffer(obs_before=obs_before,
                            info_before=info_copy,
                            action=action,
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated)
