from typing import Sequence
from deepdiff import DeepDiff
import numpy as np
import gymnasium
import pettingzoo

from utils.vector import VectorN, VectorNx3
from .mock_trajectory import ACT_SHAPE, AgentTrajectory, ActionStep, Obs, OBS_SHAPE


class MockParallelEnv(pettingzoo.ParallelEnv[str, VectorNx3, VectorN]):

    _env_name: str
    _agent_trajectories: list[AgentTrajectory]
    _obs_shape: Sequence[int]

    _current_step: int

    def __init__(self,
                 env_name: str,
                 agent_trajectories: list[AgentTrajectory],
                 obs_shape: Sequence[int] = OBS_SHAPE):

        self._env_name = env_name
        self._agent_trajectories = agent_trajectories
        self._obs_shape = obs_shape

        self.possible_agents = self._collect_possible_agents(
            agent_trajectories)
        self.observation_spaces = self._create_observation_spaces(
            self.possible_agents)
        self.action_spaces = self._create_action_spaces(self.possible_agents)

        self.agents = []
        self._current_step = 0

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[dict[str, VectorNx3], dict[str, dict]]:

        obs, info = self._create_reset_result()
        self.agents = list(obs.keys())
        self._current_step = 0

        return obs, info

    def step(
        self, actions: dict[str, VectorN]
    ) -> tuple[
            dict[str, VectorNx3],
            dict[str, float],
            dict[str, bool],
            dict[str, bool],
            dict[str, dict],
    ]:

        action_agent_ids = list(actions.keys())
        action_agent_ids.sort()

        assert action_agent_ids == self.agents, f'actions do not match agents in {self._env_name}; expected: {self.agents}, actual: {action_agent_ids}'

        obs, reward, terminated, truncated, info, expected_actions, agent_ids = self._create_step_result(
            self._current_step)

        deep_diff = DeepDiff(expected_actions, actions)
        assert deep_diff == {}, f'invalid actions received in {self._env_name}; expected: {expected_actions}, actual: {actions}, diff: {deep_diff}'

        self._current_step += 1
        self.agents = agent_ids

        return obs, reward, terminated, truncated, info

    def _create_step_result(self, step: int):

        obs: dict[str, VectorNx3] = {}
        reward: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}
        info: dict[str, dict] = {}
        expected_actions: dict[str, VectorN] = {}

        agent_ids = []

        for agent_trajectory in self._agent_trajectories:

            # the initial offset is +1, because the first observation is the state
            # after the reset
            action_index = step * 2 + 1
            obs_index = step * 2 + 2

            if len(agent_trajectory.trajectory) <= obs_index:
                continue

            agent_id = agent_trajectory.agent_id

            action_item = agent_trajectory.trajectory[action_index]
            obs_item = agent_trajectory.trajectory[obs_index]

            # action
            if action_item is not None:
                assert isinstance(action_item, ActionStep)
                assert obs_item is not None
                reward[agent_id] = action_item.reward
                expected_actions[agent_id] = action_item.action

            # observation
            if obs_item is not None:
                assert isinstance(obs_item, Obs)
                if not obs_item.terminated and not obs_item.truncated:
                    agent_ids.append(agent_id)

                obs[agent_id] = obs_item.obs
                terminated[agent_id] = obs_item.terminated
                truncated[agent_id] = obs_item.truncated
                info[agent_id] = obs_item.info

        return obs, reward, terminated, truncated, info, expected_actions, agent_ids

    def _create_observation_spaces(self, possible_agents: list[str]):

        spaces = {}

        for agent_id in possible_agents:
            spaces[agent_id] = gymnasium.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    shape=self._obs_shape)

        return spaces

    def _create_action_spaces(self, possible_agents: list[str]):

        spaces = {}

        for agent_id in possible_agents:
            spaces[agent_id] = gymnasium.spaces.Box(low=-np.inf,
                                                    high=np.inf,
                                                    shape=ACT_SHAPE)

        return spaces

    def _collect_possible_agents(self,
                                 agent_trajectories: list[AgentTrajectory]):
        agent_ids = []
        for agent in agent_trajectories:
            agent_ids.append(agent.agent_id)

        return agent_ids

    def observation_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.observation_spaces[agent]

    def action_space(self, agent: str) -> gymnasium.spaces.Space:
        return self.action_spaces[agent]

    def _create_reset_result(self):

        obs = {}
        info = {}

        for agent_trajectory in self._agent_trajectories:

            first_item = agent_trajectory.trajectory[0]
            if first_item is None:
                continue

            assert isinstance(first_item, Obs)

            agent_id = agent_trajectory.agent_id
            obs[agent_id] = first_item.obs
            info[agent_id] = first_item.info

        return obs, info
