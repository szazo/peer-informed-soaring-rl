from typing import TypeVar, Generic
from dataclasses import dataclass, replace
import numpy as np
import logging
from gymnasium.utils import seeding

from ..base.agent import AgentID


@dataclass
class AgentSpawnParameters:
    # it will use binomial distribution with p=0.5
    parallel_num_min_max: tuple[int, int] = (3, 8)
    time_between_spawns_min_max_s: tuple[int, int] = (1, 1)


AgentIDType = TypeVar('AgentIDType', bound=str)


@dataclass
class SpawnState2(Generic[AgentIDType]):
    spawned_num: int
    last_spawn_timestamp_s: float | None
    current_agents: set[AgentIDType]


@dataclass
class SpawnState:
    spawned_num: int
    last_spawn_timestamp_s: float | None


@dataclass
class RandomParams:
    random_target_count: int
    random_spawn_time: float


@dataclass
class AgentSpawnParameters2:
    pool_size: int = 3
    # it will use binomial distribution with p=0.5
    initial_time_offset_s_min_max: tuple[int, int] = (0, 0)
    parallel_num_min_max: tuple[int, int] = (3, 8)
    time_between_spawns_min_max_s: tuple[int, int] = (1, 1)
    must_spawn_if_no_global_agent: bool = False


class AgentSpawner2(Generic[AgentIDType]):

    _params: AgentSpawnParameters2
    _possible_agents: list[AgentIDType]
    _state: SpawnState2[AgentIDType] | None
    _current_targets: RandomParams | None

    def __init__(self, prefix: str, params: AgentSpawnParameters2):

        # create the logger
        self._log = logging.getLogger(__class__.__name__)

        # set params
        self._params = params
        self._possible_agents = self._generate_possible_agents(
            prefix, pool_size=params.pool_size)
        self._state = None
        self._current_targets = None

        # initialize the random number generator without seed
        self._np_random, _ = seeding.np_random()

    def reset(self, initial_time_s: float):

        initial_time_offset_s = float(self._calculate_initial_time_offset_s())

        self._state = SpawnState2(spawned_num=0,
                                  last_spawn_timestamp_s=initial_time_s +
                                  initial_time_offset_s,
                                  current_agents=set())

        self._current_targets = RandomParams(random_target_count=0,
                                             random_spawn_time=0)
        self._reset_current_target_count()
        self._reset_current_spawn_time()

    def _calculate_initial_time_offset_s(self):
        min = self._params.initial_time_offset_s_min_max[0]
        max = self._params.initial_time_offset_s_min_max[1]

        return self._random_binomial(min, max)

    @property
    def is_finished(self):

        state = self._state
        if state is None:
            return False

        assert state.spawned_num <= self._params.pool_size
        return len(state.current_agents
                   ) == 0 and state.spawned_num == self._params.pool_size

    def seed(self, seed: int | None = None) -> None:
        self._log.debug('seed: %s', seed)
        self._np_random, seed = seeding.np_random(seed)

    def spawn(self, current_time_s: float,
              global_agent_count: int) -> list[AgentIDType]:

        assert self._state is not None and self._current_targets is not None, 'reset not called'
        state = self._state
        current_targets = self._current_targets

        spawned_agent_ids: list[AgentIDType] = []

        while True:

            if not self._has_more_agents(state):
                break

            # check whether spawn even time not reached
            is_spawn_required = self._is_global_agent_count_spawn_required(
                global_agent_count)

            is_spawn_required = is_spawn_required or \
                (self._is_agent_count_spawn_required(state,
                                                     target=current_targets)
                 and self._is_time_spawn_required(current_time_s=current_time_s,
                                                  state=state,
                                                  target=current_targets)
                 )

            if not is_spawn_required:
                # no more agents required, finish
                break

            # spawn
            self._log.debug(
                'spawn required; current_count=%s, target_count=%s, spawned_count=%s',
                self._current_agent_count(state),
                current_targets.random_target_count, state.spawned_num)

            agent_id, state = self._spawn_agent(state,
                                                current_time_s=current_time_s)
            self._log.debug('spawning %s...', agent_id)
            spawned_agent_ids.append(agent_id)
            global_agent_count += 1

        if len(spawned_agent_ids) > 0:
            # only update the random spawn time
            self._reset_current_spawn_time()

            # update the new state
            self._state = state

        return spawned_agent_ids

    def _spawn_agent(self, state: SpawnState2, current_time_s: float):

        # we need to spawn, generate id
        next_id = self._possible_agents[state.spawned_num]

        # update the state
        new_state = SpawnState2(spawned_num=state.spawned_num + 1,
                                last_spawn_timestamp_s=current_time_s,
                                current_agents=state.current_agents
                                | {next_id})

        return next_id, new_state

    def _has_more_agents(self, state: SpawnState2):
        remaining_count = len(self._possible_agents) - state.spawned_num
        return remaining_count > 0

    def _is_global_agent_count_spawn_required(self, global_agent_count: int):
        return self._params.must_spawn_if_no_global_agent and global_agent_count == 0

    def _is_agent_count_spawn_required(self, state: SpawnState2,
                                       target: RandomParams):

        current_agent_count = self._current_agent_count(state)
        return target.random_target_count - current_agent_count > 0

    def _current_agent_count(self, state: SpawnState2):
        return len(state.current_agents)

    def _is_time_spawn_required(self, current_time_s: float,
                                state: SpawnState2, target: RandomParams):
        last_spawn_diff = current_time_s
        if state.last_spawn_timestamp_s is not None:
            last_spawn_diff = current_time_s - state.last_spawn_timestamp_s

        return last_spawn_diff >= target.random_spawn_time

    def _generate_possible_agents(self, id_prefix: str,
                                  pool_size: int) -> list[AgentIDType]:
        ids = []
        for i in range(pool_size):
            ids.append(id_prefix + str(i))

        return ids

    def _reset_current_target_count(self):

        assert self._current_targets is not None

        min = self._params.parallel_num_min_max[0]
        max = self._params.parallel_num_min_max[1]

        self._current_targets.random_target_count = self._random_binomial(
            min, max)
        self._log.debug('current target count=%s',
                        self._current_targets.random_target_count)

    def _reset_current_spawn_time(self):

        assert self._current_targets is not None

        min = self._params.time_between_spawns_min_max_s[0]
        max = self._params.time_between_spawns_min_max_s[1]

        # now it is binomial, can be e.g. normal
        self._current_targets.random_spawn_time = self._random_binomial(
            min, max)
        self._log.debug('current spawn time=%s',
                        self._current_targets.random_spawn_time)

    def _random_binomial(self, min: int, max: int, p: float = 0.5) -> int:
        return self._np_random.binomial(max - min, p=p) + min

    def agent_killed(self, agent_id: AgentIDType):
        assert self._state is not None

        if not agent_id in self._state.current_agents:
            return

        self._log.debug('agent killed: %s', agent_id)

        self._state = replace(self._state,
                              current_agents=self._state.current_agents -
                              {agent_id})

        self._reset_current_target_count()

    @property
    def possible_agents(self) -> list[AgentIDType]:
        return self._possible_agents


class AgentSpawner:

    _params: AgentSpawnParameters

    _possible_agents: list[AgentID]
    _state: SpawnState | None
    _np_random: np.random.Generator

    _random_params: RandomParams | None

    def __init__(self, params: AgentSpawnParameters):

        # create the logger
        self._log = logging.getLogger(__class__.__name__)

        # set params
        self._params = params

        # initialize the random number generator without seed
        self._np_random, _ = seeding.np_random()

    def seed(self, seed: int | None = None) -> None:
        self._log.debug('seed: %s', seed)
        self._np_random, seed = seeding.np_random(seed)

    def reset(self, initial_time_s: float, possible_agents: list[AgentID]):
        self._possible_agents = possible_agents
        self._state = SpawnState(spawned_num=0,
                                 last_spawn_timestamp_s=initial_time_s)

        self._random_params = RandomParams(random_target_count=0,
                                           random_spawn_time=0)
        self._reset_random_target_count()
        self._reset_random_spawn_time()

    def _reset_random_target_count(self):

        assert self._random_params is not None

        min = self._params.parallel_num_min_max[0]
        max = self._params.parallel_num_min_max[1]

        self._random_params.random_target_count = self._random_binomial(
            min, max)
        self._log.debug('current target count=%s',
                        self._random_params.random_target_count)

    def _reset_random_spawn_time(self):

        assert self._random_params is not None

        min = self._params.time_between_spawns_min_max_s[0]
        max = self._params.time_between_spawns_min_max_s[1]

        # now it is binomial, can be e.g. normal
        self._random_params.random_spawn_time = self._random_binomial(min, max)
        self._log.debug('current spawn time=%s',
                        self._random_params.random_spawn_time)

    def _random_binomial(self, min: int, max: int, p: float = 0.5) -> int:
        return self._np_random.binomial(max - min, p=p) + min

    def agent_killed(self, agent_id):
        self._log.debug('agent killed: %s', agent_id)
        self._reset_random_target_count()

    def spawn(self, current_time_s: float,
              current_count: int) -> AgentID | None:

        assert self._state is not None and self._random_params is not None, 'reset not called'

        state = self._state

        last_spawn_diff = current_time_s
        if state.last_spawn_timestamp_s is not None:
            last_spawn_diff = current_time_s - state.last_spawn_timestamp_s

        target_count = self._random_params.random_target_count
        random_spawn_time = self._random_params.random_spawn_time

        if current_count == 0 or (last_spawn_diff >= random_spawn_time
                                  and target_count - current_count > 0):
            remaining_count = len(self._possible_agents) - state.spawned_num
            if remaining_count <= 0:
                # no more possible agent, skip
                return None

            self._log.debug(
                'spawn required; current_count=%s, target_count=%s, remaining_count=%s, last_spawn_diff=%s, random_spawn_time=%s',
                current_count, target_count, remaining_count, last_spawn_diff,
                random_spawn_time)

            # we need to spawn, generate id
            next_id = self._possible_agents[state.spawned_num]
            self._log.debug('spawning %s...', next_id)

            # update the state
            self._state = replace(state,
                                  spawned_num=state.spawned_num + 1,
                                  last_spawn_timestamp_s=current_time_s)

            # only update the random spawn time
            self._reset_random_spawn_time()

            # return the id of the to be created agent
            return next_id

        # no need to spawn
        return None
