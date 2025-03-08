from typing import TypeVar, Any
import functools
import logging
from dataclasses import dataclass, replace
import numpy as np
import gymnasium
from gymnasium.utils import seeding
from pettingzoo import ParallelEnv

from ..aerodynamics import AerodynamicsInterface
from ..air_velocity_field import AirVelocityFieldInterface
from ..base import TimeParameters, SimulationBoxParameters
from ..base.visualization import MultigliderVisualization, RenderParameters
from ..base.agent import (AgentID, GliderAgent, GliderAgentParameters,
                          GliderAgentObsType,
                          GliderInitialConditionsCalculator, GliderTrajectory,
                          GliderCutoffParameters, GliderRewardParameters,
                          GliderCutoffCalculator, GliderRewardCalculator)
from .agent_spawner import AgentSpawner2

ObsType = TypeVar('ObsType')
ActionType = float


@dataclass
class State:
    t_s: float


@dataclass
class AgentGroup:
    name: str
    terminate_if_finished: bool
    spawner: AgentSpawner2[AgentID]
    initial_conditions_calculator: GliderInitialConditionsCalculator


class MultiGliderEnvBase(ParallelEnv[AgentID, ObsType, ActionType]):

    metadata: dict[str, Any] = {
        'render_modes': ['human', 'rgb_array'],
        'name': 'multiglider_v0'
    }

    possible_agents: list[AgentID]

    _time_params: TimeParameters
    _simulation_box_params: SimulationBoxParameters
    _glider_agent_params: GliderAgentParameters
    _cutoff_params: GliderCutoffParameters
    _reward_params: GliderRewardParameters
    _aerodynamics: AerodynamicsInterface
    _air_velocity_field: AirVelocityFieldInterface
    _agent_groups: list[AgentGroup]
    _render_params: RenderParameters
    _visualization: MultigliderVisualization

    _air_initial_conditions_info: dict | None
    _agent_instances: dict[AgentID, GliderAgent]
    _deleted_agent_instances: dict[AgentID, GliderAgent]
    _state: State | None
    _np_random: np.random.Generator

    def __init__(self, time_params: TimeParameters,
                 simulation_box_params: SimulationBoxParameters,
                 glider_agent_params: GliderAgentParameters,
                 cutoff_params: GliderCutoffParameters,
                 reward_params: GliderRewardParameters,
                 aerodynamics: AerodynamicsInterface,
                 air_velocity_field: AirVelocityFieldInterface,
                 agent_groups: list[AgentGroup],
                 render_params: RenderParameters,
                 visualization: MultigliderVisualization):

        self._log = logging.getLogger(__class__.__name__)

        self._time_params = time_params
        self._simulation_box_params = simulation_box_params
        self._glider_agent_params = glider_agent_params
        self._cutoff_params = cutoff_params
        self._reward_params = reward_params
        self._aerodynamics = aerodynamics
        self._air_velocity_field = air_velocity_field
        self._agent_groups = agent_groups
        self._render_params = render_params
        self._visualization = visualization

        self.possible_agents = self._create_possible_agents()
        self._agent_instances = {}
        self._deleted_agent_instances = {}

        # initialize the random number generator without seed
        self._np_random, _ = seeding.np_random()

    def _create_possible_agents(self) -> list[AgentID]:

        possible_agents = []
        for group in self._agent_groups:
            possible_agents += group.spawner.possible_agents

        return possible_agents

    @property
    def np_random(self) -> np.random.Generator:
        return self._np_random

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict[AgentID, ObsType], dict[AgentID, dict]]:

        self._log.debug('reset; seed=%s,options=%s', seed, options)

        if seed is not None:
            self._log.debug('reset called with seed %s', seed)
            self._np_random, seed = seeding.np_random(seed)
            self._air_velocity_field.seed(seed)
            for group in self._agent_groups:
                group.spawner.seed(seed)
                group.initial_conditions_calculator.seed(seed)

        # create the new state
        self._state = State(t_s=self._time_params.initial_time_s)

        # reset the air velocity field
        self._air_initial_conditions_info = self._air_velocity_field.reset()

        # reset the spawners
        for group in self._agent_groups:
            group.spawner.reset(initial_time_s=self._state.t_s)

        # clear the instances
        self._agent_instances = {}
        self._deleted_agent_instances = {}

        self._spawn_agents(current_time_s=self._state.t_s,
                           global_agent_count=len(self._agent_instances))

        assert len(self._agent_instances) > 0, 'no agent created at reset'

        # agents created, return the observation and info
        obs = self._get_observation(t_s=self._time_params.initial_time_s)
        info = self._get_info(observations=obs)

        self._log.debug('observations=%s', obs)
        self._log.debug('info=%s', info)

        return obs, info

    def _spawn_agents(self, current_time_s: float, global_agent_count: int):

        created_agent_ids: list[AgentID] = []

        for group in self._agent_groups:

            agent_ids_to_create = group.spawner.spawn(
                current_time_s=current_time_s,
                global_agent_count=global_agent_count)

            for agent_id in agent_ids_to_create:
                self._create_instance(agent_id,
                                      current_time_s=current_time_s,
                                      initial_conditions_calculator=group.
                                      initial_conditions_calculator)
            global_agent_count += len(agent_ids_to_create)

            created_agent_ids += agent_ids_to_create

        if len(created_agent_ids) > 0:
            self._log.debug('created_agent_ids=%s', created_agent_ids)
        return created_agent_ids

    def _create_instance(
            self, agent_id: AgentID, current_time_s: float,
            initial_conditions_calculator: GliderInitialConditionsCalculator):

        self._log.debug('creating agent; id=%s', agent_id)

        # cutoff / reward calculator
        cutoff_calculator = GliderCutoffCalculator(self._cutoff_params,
                                                   self._simulation_box_params)
        reward_calculator = GliderRewardCalculator(self._reward_params)

        # agents use the shared np_random, so if the env is seeded, then their behaviour will be deterministic too
        glider_agent = GliderAgent(
            agent_id=agent_id,
            parameters=self._glider_agent_params,
            aerodynamics=self._aerodynamics,
            air_velocity_field=self._air_velocity_field,
            initial_conditions_calculator=initial_conditions_calculator,
            cutoff_calculator=cutoff_calculator,
            reward_calculator=reward_calculator,
            np_random=self._np_random)
        glider_agent.reset(time_s=current_time_s)

        self._agent_instances[agent_id] = glider_agent

    @property
    def agents(self) -> list[AgentID]:
        return list(self._agent_instances.keys())

    def step(
        self, actions: dict[AgentID, ActionType]
    ) -> tuple[
            dict[AgentID, ObsType],  # observation
            dict[AgentID, float],  # reward
            dict[AgentID, bool],  # terminated
            dict[AgentID, bool],  # truncated
            dict[AgentID, dict]  # info
    ]:

        assert len(self.agents
                   ) > 0, 'Environment has been already finished, call reset'

        self._log.debug('step; actions=%s', actions)

        current_state = self._state
        dt_s = self._time_params.dt_s
        current_time_s = current_state.t_s
        next_time_s = current_time_s + dt_s

        rewards: dict[AgentID, float] = {}
        terminations: dict[AgentID, bool] = {}
        truncations: dict[AgentID, bool] = {}

        agents_to_delete = []
        # iterate over the agents and step them

        for agent_id in self.agents:
            agent_action = actions[agent_id]

            assert agent_action is not None and not np.isnan(
                agent_action), f'missing action for agent {agent_id}'

            agent_instance = self._agent_instances[agent_id]
            reward_cutoff_result = agent_instance.step(
                action=agent_action,
                current_time_s=current_time_s,
                next_time_s=next_time_s,
                dt_s=dt_s)

            terminated = reward_cutoff_result.terminated
            truncated = reward_cutoff_result.truncated
            terminations[agent_id] = terminated
            truncations[agent_id] = truncated

            if reward_cutoff_result.reason != 'none':
                self._log.debug('cutoff reason; %s=%s', agent_id,
                                reward_cutoff_result.reason)

            agent_done = terminated or truncated

            if agent_done:
                # delete the instance
                agents_to_delete.append(agent_id)

            # store the reward even the agent is deleted
            rewards[agent_id] = reward_cutoff_result.reward

        agent_count_after_deletion = len(
            self._agent_instances) - len(agents_to_delete)

        # spawn new glider if necessary
        created_agent_ids = self._spawn_agents(
            current_time_s=next_time_s,
            global_agent_count=agent_count_after_deletion)

        for agent_id in created_agent_ids:
            # set zero reward for the created instance
            rewards[agent_id] = 0.
            terminations[agent_id] = False
            truncations[agent_id] = False

        # update the new state
        self._state = replace(current_state, t_s=next_time_s)

        # REVIEW: it should be true, remove the it
        INCLUDE_DELETED_AGENT_OBSERVATION = True
        if INCLUDE_DELETED_AGENT_OBSERVATION:
            # query the observation before the deletion
            observations = self._get_observation(t_s=self._state.t_s)
            info = self._get_info(observations=observations)

        # delete the agents
        if len(agents_to_delete) > 0:
            self._delete_agents(agents_to_delete)

        # check whether need to terminate, if so, delete all of the remaining agents
        global_termination = False
        for group in self._agent_groups:
            if group.terminate_if_finished and group.spawner.is_finished:
                global_termination = True
                break

        if global_termination:
            for agent_id in self.agents:
                truncations[agent_id] = True

            self._delete_agents(self.agents)

        if not INCLUDE_DELETED_AGENT_OBSERVATION:
            # query the observation after the deletion
            observations = self._get_observation(t_s=self._state.t_s)
            info = self._get_info(observations=observations)

        self._log.debug('observations=%s', observations)
        self._log.debug('rewards=%s', rewards)
        self._log.debug('terminations=%s', terminations)
        self._log.debug('truncations=%s', truncations)
        self._log.debug('infos=%s', info)

        return observations, rewards, terminations, truncations, info

    def _delete_agents(self, agents_to_delete: list[AgentID]):

        self._log.debug('deleting agents=%s', agents_to_delete)
        for agent_id in agents_to_delete:
            self._deleted_agent_instances[agent_id] = self._agent_instances[
                agent_id]
            del self._agent_instances[agent_id]

            for group in self._agent_groups:
                group.spawner.agent_killed(agent_id)

    @functools.cache
    def observation_space(self, agent_id: AgentID) -> gymnasium.spaces.Space:
        return GliderAgent.observation_space(self._simulation_box_params)

    @functools.cache
    def action_space(self, agent_id: AgentID) -> gymnasium.spaces.Space:
        return GliderAgent.action_space()

    def _get_info(
        self, observations: dict[AgentID,
                                 GliderAgentObsType]) -> dict[AgentID, dict]:

        info = dict()

        # create info using the agent instances
        for agent_id, agent_instance in self._agent_instances.items():
            agent_info = agent_instance.get_info_as_dict()
            agent_info = {
                **agent_info,
                **dict(initial_conditions=dict(glider=agent_info['initial_conditions'],
                                               air_velocity_field=self._air_initial_conditions_info)),
                **observations[agent_id]
            }

            info[agent_id] = agent_info

        return info

    def _get_observation(self,
                         t_s: float) -> dict[AgentID, GliderAgentObsType]:

        obs = dict()

        # create observation using the agent instances
        for agent_id, agent_instance in self._agent_instances.items():
            agent_obs = agent_instance.get_observation(t_s=t_s)

            obs[agent_id] = agent_obs

        return obs

    def get_agent_trajectory(self, agent_id: AgentID) -> GliderTrajectory:

        if agent_id in self._agent_instances:
            agent_instance = self._agent_instances[agent_id]
        elif agent_id in self._deleted_agent_instances:
            agent_instance = self._deleted_agent_instances[agent_id]
        else:
            raise Exception(f'Invalid agent id: {agent_id}')

        return agent_instance.get_trajectory()

    def render(self) -> None | np.ndarray:

        # query trajectories for each active agent and render them
        agent_trajectories = {}

        for agent_id, agent_instance in self._agent_instances.items():
            trajectory = agent_instance.get_trajectory()
            agent_trajectories[agent_id] = trajectory

        return self._visualization.render(trajectories=agent_trajectories,
                                          time_s=self._state.t_s)

    def close(self):
        self._visualization.close()
