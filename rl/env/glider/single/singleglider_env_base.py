from typing import Any, SupportsFloat
from dataclasses import dataclass
import logging
import gymnasium
from gymnasium.utils import seeding

from ..aerodynamics import AerodynamicsInterface
from ..air_velocity_field import AirVelocityFieldInterface
from ..base import (
    GliderTrajectory,
    AgentID,
    TimeParameters,
    SimulationBoxParameters,
)
from ..base.visualization import RenderParameters
from ..base.agent import (AgentID, GliderAgent, GliderAgentParameters,
                          GliderAgentObsType,
                          GliderInitialConditionsCalculator,
                          GliderCutoffParameters, GliderRewardParameters,
                          GliderCutoffCalculator, GliderRewardCalculator,
                          GliderTrajectory)

ObsType = GliderAgentObsType
ActionType = float


@dataclass
class State:
    t_s: float


class SingleGliderEnvBase(gymnasium.Env[ObsType, float]):

    _log: logging.Logger

    metadata: dict[str, Any] = {'render_modes': ['human', 'rgb_array']}
    action_space: gymnasium.spaces.Space[ActionType]
    observation_space: gymnasium.spaces.Space[ObsType]

    _glider_agent_params: GliderAgentParameters
    _simulation_box_params: SimulationBoxParameters
    _time_params: TimeParameters
    _cutoff_params: GliderCutoffParameters
    _reward_params: GliderRewardParameters

    _aerodynamics: AerodynamicsInterface
    _air_velocity_field: AirVelocityFieldInterface
    _initial_conditions_calculator: GliderInitialConditionsCalculator

    _current_time_s: float | None
    _agent: GliderAgent | None
    _air_initial_conditions_info: dict | None

    def __init__(
            self, aerodynamics: AerodynamicsInterface,
            air_velocity_field: AirVelocityFieldInterface,
            glider_agent_params: GliderAgentParameters,
            simulation_box_params: SimulationBoxParameters,
            time_params: TimeParameters, cutoff_params: GliderCutoffParameters,
            reward_params: GliderRewardParameters,
            render_params: RenderParameters,
            initial_conditions_calculator: GliderInitialConditionsCalculator):

        self._log = logging.getLogger(__class__.__name__)

        self.render_mode = render_params.mode
        self._log.debug('__init__; render_mode=%s', self.render_mode)

        self.action_space = GliderAgent.action_space()
        self.observation_space = GliderAgent.observation_space(
            simulation_box_params=simulation_box_params)

        self._glider_agent_params = glider_agent_params
        self._simulation_box_params = simulation_box_params
        self._time_params = time_params
        self._cutoff_params = cutoff_params
        self._reward_params = reward_params

        self._aerodynamics = aerodynamics
        self._air_velocity_field = air_velocity_field
        self._initial_conditions_calculator = initial_conditions_calculator

        self._log.debug('action_space=%s,observation_space=%s',
                        self.action_space, self.observation_space)

        # initialize the random number generator without seed by default
        self.np_random, _ = seeding.np_random()

        # set the time and the agent to none
        self._current_time_s = 0.
        self._agent = None

    def reset(self,
              *args,
              seed: int | None = None,
              options: dict[str, Any] | None = None):

        self._log.debug("reset; args=%s,seed=%s,options=%s", args, seed,
                        options)

        # ensure the instance's np_random is initialized in the base class
        super().reset(seed=seed)

        if seed is not None:
            self._log.debug('reset called with seed %s', seed)
            self._initial_conditions_calculator.seed(seed)
            self._air_velocity_field.seed(seed)

        # reset the time
        self._current_time_s = self._time_params.initial_time_s

        # reset the air velocity field
        self._air_initial_conditions_info = self._air_velocity_field.reset()

        # cutoff / reward calculator
        cutoff_calculator = GliderCutoffCalculator(self._cutoff_params,
                                                   self._simulation_box_params)
        reward_calculator = GliderRewardCalculator(self._reward_params)

        # create the agent
        # agents use the shared np_random, so if the env is seeded, then their behaviour will be deterministic too
        self._agent = GliderAgent(
            agent_id=AgentID('glider'),
            parameters=self._glider_agent_params,
            aerodynamics=self._aerodynamics,
            air_velocity_field=self._air_velocity_field,
            initial_conditions_calculator=self._initial_conditions_calculator,
            cutoff_calculator=cutoff_calculator,
            reward_calculator=reward_calculator,
            np_random=self.np_random)
        self._agent.reset(time_s=self._current_time_s)

        observation, info = self._query_current_observation_and_info(
            self._agent, self._current_time_s)
        self._log.debug("observation=%s,info=%s", observation, info)

        if self.render_mode == 'human':
            self.render()

        return observation, info

    def step(
        self, action: ActionType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:

        self._log.debug('step; action=%s', action)

        assert self._current_time_s is not None and self._agent is not None, 'reset is not called'

        next_time_s = self._current_time_s + self._time_params.dt_s

        reward_cutoff_result = self._agent.step(
            action=action,
            current_time_s=self._current_time_s,
            next_time_s=next_time_s,
            dt_s=self._time_params.dt_s)

        terminated = reward_cutoff_result.terminated
        truncated = reward_cutoff_result.truncated
        reward = reward_cutoff_result.reward

        self._current_time_s = next_time_s

        # query the new observation and info
        observation, info = self._query_current_observation_and_info(
            self._agent, self._current_time_s)

        self._log.debug(
            "step; action=%s, observation=%s, reward=%s, termination=%s, truncation=%s, info=%s",
            action,
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

        if self.render_mode == 'human':
            self.render()

        return observation, reward, terminated, truncated, info

    def _query_current_observation_and_info(self, agent: GliderAgent,
                                            current_time_s: float):
        observation = agent.get_observation(current_time_s)
        agent_info = agent.get_info_as_dict()

        assert self._air_initial_conditions_info is not None

        # merge air_velocity initial conditions
        info = self._merge_info(agent_observation=observation,
                                agent_info=agent_info,
                                air_velocity_initial_conditions_info=self.
                                _air_initial_conditions_info)

        return observation, info

    def _merge_info(self, agent_observation: GliderAgentObsType,
                    agent_info: dict,
                    air_velocity_initial_conditions_info: dict):

        info = {
            **agent_info,
            **dict(initial_conditions=dict(glider=agent_info['initial_conditions'],
                                           air_velocity_field=air_velocity_initial_conditions_info)),
            **agent_observation
        }

        return info

    def get_trajectory(self) -> GliderTrajectory:

        assert self._agent is not None, 'reset not called'
        return self._agent.get_trajectory()

    def render(self):
        pass
