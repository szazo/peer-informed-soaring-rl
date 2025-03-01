from typing import Any, TypeVar, Generic, SupportsFloat, Literal
from copy import deepcopy
import logging
import gymnasium
import numpy as np

from pettingzoo import ParallelEnv
from utils.vector import VectorN

from .create_agent_id_index_mapping import create_agent_id_index_mapping

ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType', bound=np.ndarray)
AgentIDType = TypeVar('AgentIDType')


class ParallelPettingZooEnv(Generic[AgentIDType, ObsType, ActType],
                            gymnasium.Env[dict[AgentIDType, ObsType],
                                          dict[AgentIDType, ActType]]):
    """
    Wrap ParallelEnv in gymnasium Env, so it can be used in tianshou for multi agent training.
    """

    _name: str
    _env: ParallelEnv[AgentIDType, ObsType, ActType]

    _agent_id_to_index_map: dict[AgentIDType, int]
    _index_to_agent_id_map: list[AgentIDType]

    # _learning_agent_ids: set[AgentIDType]

    def __init__(
        self,
        name: str,
        env: ParallelEnv[AgentIDType, ObsType, ActType],
        # learning_agent_ids: list[AgentIDType]
    ):

        super().__init__()

        self._log = logging.getLogger(__class__.__name__)

        self._name = name

        self.observation_space = self._create_observation_space(env)
        self.action_space = self._create_action_space(env)

        self._agent_id_to_index_map = create_agent_id_index_mapping(env)
        # print('AGENT_ID', self._agent_id_to_index_map)
        self._index_to_agent_id_map = env.possible_agents.copy()
        # self._learning_agent_ids = {agent_id for agent_id in learning_agent_ids}

        self._env = env

    @property
    def agent_id_to_index_map(self):
        return self._agent_id_to_index_map

    @property
    def agents(self):
        return self._env.agents

    @property
    def possible_agents(self):
        return self._env.possible_agents

    def _create_observation_space(self, env: ParallelEnv[AgentIDType, ObsType,
                                                         ActType]):

        observation_spaces = {}
        for agent_id in env.possible_agents:
            agent_observation_space = env.observation_space(agent_id)
            observation_spaces[agent_id] = agent_observation_space

        return gymnasium.spaces.Dict(observation_spaces)

        # first_possible_agent_id = env.possible_agents[0]
        # observation_space = env.observation_space(
        #     first_possible_agent_id)
        # assert all(env.observation_space(agent) == observation_space
        #            for agent in env.possible_agents), \
        #                    "Observation spaces for all agents must be identical."

        # return observation_space

    def _create_action_space(
        self, env: ParallelEnv[AgentIDType, ObsType,
                               ActType]) -> gymnasium.spaces.Dict:
        action_spaces = {}
        for agent_id in env.possible_agents:
            agent_action_space = env.action_space(agent_id)
            action_spaces[agent_id] = agent_action_space

        return gymnasium.spaces.Dict(action_spaces)

        # first_possible_agent_id = env.possible_agents[0]
        # action_space = env.action_space(first_possible_agent_id)
        # assert all(env.action_space(agent) == action_space
        #            for agent in env.possible_agents), \
        #                    "Action spaces for all agents must be identical."

        # return action_space

    def reset(
        self,
        *args,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[AgentIDType, ObsType], dict[Any, Any]]:

        # print('pettingzoo reset', self._name)
        obs, info = self._env.reset(*args, seed=seed, options=options)

        agents_after_reset = self._env.agents.copy()

        processed_info = deepcopy(info)
        self._include_agent_mask_in_info(processed_info,
                                         key='agent_mask',
                                         agents=agents_after_reset)
        self._include_rew_idx_in_info(agents_after_reset, processed_info)

        self._include_terminated_truncated_in_info(agents_after_reset,
                                                   processed_info,
                                                   multi_termination={},
                                                   multi_truncation={})

        # obs = self._include_all_agents_in_observation(obs)

        return obs, processed_info

    def _include_all_agents_in_observation(
            self, obs: dict[AgentIDType,
                            ObsType]) -> dict[AgentIDType, ObsType]:

        result_obs: dict[AgentIDType, ObsType] = {}

        for agent_id in self._env.possible_agents:
            if agent_id in obs:
                result_obs[agent_id] = obs[agent_id]
            else:
                agent_observation_space = self.observation_space[agent_id]

                assert isinstance(agent_observation_space,
                                  gymnasium.spaces.Box)
                #empty_observation = np.full(agent_observation_space.shape, fill_value=np.nan)
                # TODO: fix the observation space
                empty_observation = np.full((3, 2, 7), fill_value=0.)
                # empty_observation = np.array(0)
                result_obs[agent_id] = empty_observation
                # print('result_obs', result_obs)
                # raise Exception('TEST')
            # result_obs[agent_id] = multi_obs[agent_id] if agent_id in multi_obs else np.array(np.full((2, 75, 7), fill_value=np.nan))

        return result_obs

        # # else { 'obs': gymnasium.vector.utils.create_empty_array(self.observation_space, n=None),                                           'mask': [False] * self._wrapped_env.action_space(agent_id).n}

        # # print('MULTI_STEP PROCESSED', list(result_obs.keys()), self.observation_space)

        # if len(multi_obs) == 0:
        #     print('terminated', terminated, agent_mask)
        #     raise Exception('A')

        # print('result_obs', list(result_obs.keys()), list(processed_info.keys()))
        # for agent_id, agent_info in processed_info.items():
        #     print('agent_mask', agent_id, agent_info['agent_mask'])
        #     print('agent_mask_before', agent_id, agent_info['agent_mask_before'])

        # print('RESULT_OBS', result_obs)
        # for agent_id,

    def _convert_action_list_to_dict(self, action_list: VectorN,
                                     current_agents: list[AgentIDType]):
        action_dict = {}
        for agent_id in current_agents:
            agent_index = self._agent_id_to_index_map[agent_id]
            action_dict[agent_id] = action_list[agent_index]

        return action_dict

    def step(
        self, action: dict[AgentIDType, ActType]
    ) -> tuple[dict[AgentIDType, ObsType], list[SupportsFloat], bool, bool,
               dict[Any, Any]]:

        # print('PETTINgZOO INPUT ACTIONS', self._name, action)

        # action_dict = self._convert_action_list_to_dict(action, self._env.agents)
        # print('ACTION_DICT', action_dict)
        action = dict(action)
        # print('actions', action)

        agents_before_step = self._env.agents.copy()

        # filter the actions, so only actions for the current agents should be included
        filtered_actions = {}
        for agent_id, agent_action in action.items():
            if not agent_id in agents_before_step:
                # it should be nan
                # print('AGENT_ACTION', agent_id, agent_action, 'exp', np.full(np.array(agent_action).shape, np.nan))
                assert np.allclose(
                    agent_action,
                    np.full(np.array(agent_action).shape, np.nan),
                    equal_nan=True), 'missing agent action should be nan'
                continue
            filtered_actions[agent_id] = agent_action

        # print('filtered_actions', filtered_actions)

        # print('STEP ACTIONS', filtered_actions)
        multi_obs, multi_reward, multi_termination, multi_truncation, multi_info = self._env.step(
            filtered_actions)
        # print('multi_obs', multi_obs, len(multi_obs))

        possible_agent_count = len(self._env.possible_agents)

        processed_rewards = self._create_reward_matrix(multi_reward)
        # reward_list = [1.]

        agents_after_step = self._env.agents.copy()
        processed_info = deepcopy(multi_info)
        self._include_agent_mask_in_info(processed_info,
                                         key='agent_mask',
                                         agents=agents_after_step)
        self._include_agent_mask_in_info(processed_info,
                                         key='agent_mask_before',
                                         agents=agents_before_step)

        agents_before_and_after_step = list(
            set(agents_before_step) | set(agents_after_step))

        self._include_rew_idx_in_info(agents_before_and_after_step,
                                      processed_info)

        # self._include_terminated_truncated_in_info(
        #     agents_before_step,
        #     processed_info,
        #     multi_termination={},
        #     multi_truncation={}
        # )

        self._include_terminated_truncated_in_info(
            agents_before_step,
            processed_info,
            multi_termination=multi_termination,
            multi_truncation=multi_truncation)

        # for agent_id in self._env.possible_agents:
        #     if not agent_id in agents_before_step and not agent_id in agents_after_step:
        #         del processed_info[agent_id]

        # # include the agent's index for the reward
        # for agent_id in agents_before_step:
        #     if not agent_id in processed_info:
        #         processed_info[agent_id] = {}

        #     processed_info[agent_id]['rew_idx'] = self._agent_id_to_index_map[
        #         agent_id]

        #processed_info = {}
        # print('MULTI_TERMINATION', multi_termination)
        # print('MULTI_TRUNCATION', multi_truncation)
        # for agent_id in agents_before_step:
        #     if not agent_id in processed_info:
        #         processed_info[agent_id] = {}

        #     processed_info[agent_id]['terminated'] = multi_termination[
        #         agent_id]
        #     processed_info[agent_id]['truncated'] = multi_truncation[agent_id]

        # agent_mask = self._create_agent_mask(agents_after_step)
        # for agent_id, mask in agent_mask.items():
        #     if not agent_id in processed_info:
        #         processed_info[agent_id] = {}
        #     processed_info[agent_id]['agent_mask'] = mask

        terminated = False
        truncated = False
        if len(agents_after_step) == 0:
            # WARNING: it should be truncated (not terminated) because otherwise critic estimation will be invalid
            # see: policy/base.py  value_mask method; (~buffer.terminated[indices])
            truncated = True

        # # check whether non learning agent remained
        # learning_agent_found = False
        # for agent_after_step in agents_after_step:
        #     if agent_after_step in self._learning_agent_ids:
        #         learning_agent_found = True
        #         break

        # if not learning_agent_found:
        #     terminated = True

        #print('processed_info', processed_info)
        # print('MULTI_STEP', list(multi_obs.keys()))

        # create dict from possible agents
        # print('OBS SPACE', self.observation_space)

        # empty_array = gymnasium.vector.utils.create_empty_array(self.observation_space, n=None)

        # print('EMPTY', empty_array, empty_array)

        # raise Exception('TEST')

        # result_obs: dict[AgentIDType, ObsType] = {}
        # for agent_id in self._env.possible_agents:
        #     if agent_id in multi_obs:
        #         result_obs[agent_id] = multi_obs[agent_id]
        #     else:
        #         agent_observation_space = self.observation_space[agent_id]

        #         assert isinstance(agent_observation_space, gymnasium.spaces.Box)
        #         #empty_observation = np.full(agent_observation_space.shape, fill_value=np.nan)
        #         # TODO: fix the observation space
        #         empty_observation = np.full((5, 50, 7), fill_value=0.)
        #         result_obs[agent_id] = empty_observation
        #         # print('result_obs', result_obs)
        #         # raise Exception('TEST')
        #     # result_obs[agent_id] = multi_obs[agent_id] if agent_id in multi_obs else np.array(np.full((2, 75, 7), fill_value=np.nan))

        # # # else { 'obs': gymnasium.vector.utils.create_empty_array(self.observation_space, n=None),                                           'mask': [False] * self._wrapped_env.action_space(agent_id).n}

        # # # print('MULTI_STEP PROCESSED', list(result_obs.keys()), self.observation_space)

        # if len(multi_obs) == 0:
        #     print('terminated', terminated, agent_mask)
        #     raise Exception('A')

        # print('result_obs', list(result_obs.keys()), list(processed_info.keys()))
        # for agent_id, agent_info in processed_info.items():
        #     print('agent_mask', agent_id, agent_info['agent_mask'])
        #     print('agent_mask_before', agent_id, agent_info['agent_mask_before'])

        # print('RESULT_OBS', result_obs)
        # multi_obs = self._include_all_agents_in_observation(multi_obs)

        # print('processed_rewards', processed_rewards)

        return multi_obs, processed_rewards, terminated, truncated, processed_info

    def _include_rew_idx_in_info(self, agents: list[AgentIDType],
                                 processed_info: dict):

        for agent_id in agents:
            if not agent_id in processed_info:
                processed_info[agent_id] = {}

            processed_info[agent_id]['rew_idx'] = self._agent_id_to_index_map[
                agent_id]

    def _include_terminated_truncated_in_info(
            self, agents: list[AgentIDType], processed_info: dict,
            multi_termination: dict[AgentIDType,
                                    bool], multi_truncation: dict[AgentIDType,
                                                                  bool]):

        for agent_id in agents:
            if not agent_id in processed_info:
                processed_info[agent_id] = {}

            if not agent_id in multi_termination:
                terminated = False
                truncated = False
            else:
                terminated = multi_termination[agent_id]
                truncated = multi_truncation[agent_id]

            processed_info[agent_id]['terminated'] = terminated
            processed_info[agent_id]['truncated'] = truncated

    def _include_agent_mask_in_info(self, processed_info: dict,
                                    key: Literal['agent_mask']
                                    | Literal['agent_mask_before'],
                                    agents: list[AgentIDType]):

        agent_mask = self._create_agent_mask(agents)
        for agent_id, mask in agent_mask.items():
            if not agent_id in processed_info:
                processed_info[agent_id] = {}
            processed_info[agent_id][key] = mask

    def _create_agent_mask(self, agents: list[AgentIDType]):
        mask = {
            agent_id: (agent_id in agents)
            for agent_id in self._env.possible_agents
        }
        return mask

    def _create_reward_matrix(self, rewards: dict[AgentIDType, float]):
        """Create an array which represents the rewards for each possible agents.
        Fill the values from the current agent rewards based on possible agent indices."""

        result = np.zeros(len(self._agent_id_to_index_map))
        for agent_id, reward in rewards.items():
            result[self._agent_id_to_index_map[agent_id]] = reward

        return result
