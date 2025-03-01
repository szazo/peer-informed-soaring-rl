from typing import cast
import logging
import os
from dataclasses import dataclass
from env.glider.single.statistics.observation_logger import SingleGliderObservationLogger
import numpy as np
import pandas as pd
from gymnasium.utils import seeding
from tianshou.data import Batch, ReplayBuffer

from env.glider.base import GliderEnvParameters, GliderTrajectory, TimeParameters, calculate_frame_skip_number
from env.glider.base.agent import GliderTrajectorySerializer
from env.glider.base.agent import GliderAgent
from mcts.api import EnvAdapter
from trainer.tianshou_evaluator import convert_buffer_to_observation_log


class GliderAgentState:
    _agent: GliderAgent
    _air_initial_conditions_info: dict
    _time_s: float

    # the cumulative reward of this state from the beginning
    _cumulative_reward: float

    def __init__(self, agent: GliderAgent, air_initial_conditions_info: dict,
                 time_s: float, cumulative_reward: float):

        self._agent = agent
        self._air_initial_conditions_info = air_initial_conditions_info
        self._time_s = time_s
        self._cumulative_reward = cumulative_reward

    def get_current_obs(self):
        observation = self._agent.get_observation(self._time_s)
        assert isinstance(observation, dict)

        return observation
        # info = self._agent.get_info_as_dict()

        # info = self._extend_info(info=info, observation=cast(dict, observation))

        # return observation, info

    def get_current_info(self):
        observation = self.get_current_obs()
        info = self._agent.get_info_as_dict()

        info = self._extend_info(info=info,
                                 observation=cast(dict, observation))

        return info

    def _extend_info(self, info: dict, observation: dict):
        info = {
            **info,
            **dict(initial_conditions=dict(glider=info['initial_conditions'],
                                           air_velocity_field=self._air_initial_conditions_info)),
            **observation
        }

        return info

    def step(self, action: float, dt_s: float):
        next_time_s = self._time_s + dt_s
        result = self._agent.step(action,
                                  current_time_s=self._time_s,
                                  next_time_s=next_time_s,
                                  dt_s=dt_s)

        self._cumulative_reward += result.reward
        self._time_s = next_time_s
        # done = result.terminated or result.truncated
        observation = self._agent.get_observation(next_time_s)
        info = self._agent.get_info_as_dict()

        assert isinstance(observation, dict)
        info = self._extend_info(info=info,
                                 observation=cast(dict, observation))

        return observation, result.reward, result.terminated, result.truncated, info

    def clone(self):
        agent_clone = self._agent.state_clone()

        return self.__class__(
            agent_clone,
            air_initial_conditions_info=self._air_initial_conditions_info,
            time_s=self._time_s,
            cumulative_reward=self._cumulative_reward)

    @property
    def cumulative_reward(self):
        return self._cumulative_reward

    @property
    def time_s(self):
        return self._time_s

    @property
    def agent(self):
        return self._agent


@dataclass
class DiscreteToContinuousDegreesParams:
    discrete_count: int
    low_deg: float
    high_deg: float


class GliderEnvLogger:

    buffer: ReplayBuffer
    current_trajectory: GliderTrajectory | None

    def __init__(self, buffer_size=100000):

        self.buffer = ReplayBuffer(buffer_size)
        self.current_trajectory = None


@dataclass
class GliderEnvAdapterParams:
    # fixed additional columns included in the observation log
    additional_columns: dict[str, str] | None


class GliderEnvAdapter(EnvAdapter[GliderAgentState, float, GliderTrajectory,
                                  GliderEnvLogger]):

    _log: logging.Logger
    _params: GliderEnvAdapterParams
    _time_params: TimeParameters
    _frame_skip_num: int
    _snapshot_out_dir: str

    def __init__(
            self, params: GliderEnvAdapterParams,
            discrete_to_continuous_params: DiscreteToContinuousDegreesParams,
            time_params: TimeParameters, snapshot_out_dir: str):

        self._log = logging.getLogger(__class__.__name__)

        self._params = params
        self._time_params = time_params
        self._frame_skip_num = calculate_frame_skip_number(time_params)
        self._snapshot_out_dir = snapshot_out_dir

        # discrete actions
        low = np.deg2rad(discrete_to_continuous_params.low_deg)
        high = np.deg2rad(discrete_to_continuous_params.high_deg)
        discrete_count = discrete_to_continuous_params.discrete_count
        self._allowed_actions = np.linspace(low, high, discrete_count).tolist()

        # initialize the random number generator without seed
        self._np_random, _ = seeding.np_random()

    def seed(self, seed: int | None = None) -> None:
        self._log.debug('seed: %s', seed)
        self._np_random, seed = seeding.np_random(seed)

    def get_allowed_actions(self, state: GliderAgentState) -> list[float]:
        return self._allowed_actions

    def get_render_state(self, state: GliderAgentState) -> GliderTrajectory:
        trajectory = state.agent.get_trajectory()
        return trajectory

    def simulate(self, state: GliderAgentState,
                 max_simulation_step_count: int):

        cur_state = state.clone()
        done = False
        step_count = 0
        while not done and step_count < max_simulation_step_count:

            allowed_actions = self.get_allowed_actions(cur_state)
            index = self._np_random.choice(len(allowed_actions))
            action = allowed_actions[index]

            obs_t_plus_1, r_t, terminated_t, truncated_t, info_t_plus_1 = self._step_state(
                state=cur_state, action=action)
            step_count += 1
            done = truncated_t or terminated_t

        self._log.debug(
            'simulation finished after %s steps; cumulative_reward=%f,time_s=%f',
            step_count, cur_state.cumulative_reward, cur_state.time_s)

        return cur_state.cumulative_reward

    # def clone_step(self, state: GliderAgentState, action: float):
    #     state = state.clone()
    #     done = self._step_state(state, action)
    #     return state, done

    def start(self, state: GliderAgentState, logger: GliderEnvLogger | None):

        if logger is not None:
            obs_t = state.get_current_obs()
            info_t = state.get_current_info()
            logger.current_trajectory = state.agent.get_trajectory()
            logger.buffer.add(
                Batch(obs=obs_t,
                      info=info_t,
                      terminated=False,
                      truncated=False,
                      rew=0.,
                      act={}))

    def step(self, state: GliderAgentState, action: float,
             logger: GliderEnvLogger | None):

        act_t = action
        obs_t = state.get_current_obs()

        obs_t_plus_1, r_t, terminated_t, truncated_t, info_t_plus_1 = self._step_state(
            state, action)

        done = terminated_t or truncated_t

        # if the caller provided a logger, use it
        if logger is not None:
            logger.current_trajectory = state.agent.get_trajectory()
            logger.buffer.add(
                Batch(obs=obs_t,
                      act=act_t,
                      rew=r_t,
                      terminated=terminated_t,
                      truncated=truncated_t,
                      obs_next=obs_t_plus_1,
                      info=info_t_plus_1))

        return done

    def clone_state(self, state: GliderAgentState):
        state = state.clone()
        return state

    def _step_state(self, state: GliderAgentState, action: float):

        observation = None
        info = None
        terminated = False
        truncated = False
        cumulative_reward = 0.
        for _ in range(self._frame_skip_num):
            observation, reward, terminated, truncated, info = state.step(
                action, dt_s=self._time_params.dt_s)
            cumulative_reward += reward
            if terminated or truncated:
                break
        assert observation is not None
        assert info is not None

        return observation, cumulative_reward, terminated, truncated, info

    def save_state(self, state: GliderAgentState,
                   logger: GliderEnvLogger | None, filename_prefix: str):

        assert logger is not None

        trajectory = logger.current_trajectory
        assert trajectory is not None

        # save as .h5
        h5_filepath = os.path.join(self._snapshot_out_dir,
                                   f'{filename_prefix}.h5')
        serializer = GliderTrajectorySerializer()
        serializer.save(trajectory, out_filepath=h5_filepath)

        # save the replay buffer as observation log .csv
        glider_observation_log_converter = SingleGliderObservationLogger()
        df = convert_buffer_to_observation_log(
            logger.buffer, glider_observation_log_converter)

        if self._params.additional_columns is not None:
            # add additional columns
            for i, (column_name, column_value) in enumerate(
                    self._params.additional_columns.items()):
                df.insert(loc=i, column=column_name, value=column_value)

        # save as .csv
        csv_filepath = os.path.join(self._snapshot_out_dir,
                                    f'{filename_prefix}.csv')

        df.to_csv(csv_filepath)
