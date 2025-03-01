from typing import TypeVar, Generic
import threading
import logging
import queue
from dataclasses import dataclass
import numpy as np
from gymnasium.utils import seeding

from .api import EnvAdapter
from .mcts import MCTS

TState = TypeVar('TState')
TAct = TypeVar('TAct')
TRenderState = TypeVar('TRenderState')
TLogger = TypeVar('TLogger')


@dataclass
class MCTSParameters:
    # maximum number of MCTS iterations for one step
    n_iter: int

    # maximum number of simulation steps when simulating a node
    # (time horizon for the node estimate)
    max_simulation_step_count: int

    # after how many environment step should we save the current state
    save_snapshot_step_frequency: int

    # the filename used for the snapshot
    snapshot_filename_prefix: str


class MCTSWorker(Generic[TState, TAct, TRenderState, TLogger]):

    _log: logging.Logger
    _env_adapter: EnvAdapter[TState, TAct, TRenderState, TLogger]
    _env_logger: TLogger | None
    _params: MCTSParameters

    _np_random: np.random.Generator

    def __init__(self, env_adapter: EnvAdapter, env_logger: TLogger | None,
                 params: MCTSParameters):

        self._env_adapter = env_adapter
        self._env_logger = env_logger
        self._params = params

        # initialize the random number generator without seed
        self._np_random, _ = seeding.np_random()

        self._log = logging.getLogger(__class__.__name__)

    def seed(self, seed: int | None = None) -> None:
        self._log.debug('seed: %s', seed)
        self._np_random, seed = seeding.np_random(seed)

    def start(self, state: TState, stop_event: threading.Event | None,
              render_queue: queue.Queue[TRenderState] | None):

        done = False
        step_count = 1

        self._env_adapter.start(state, logger=self._env_logger)

        while not (done or (stop_event is not None and stop_event.is_set())):
            state = self._env_adapter.clone_state(state)
            mcts = MCTS(start_state=state,
                        adapter=self._env_adapter,
                        max_simulation_step_count=self._params.
                        max_simulation_step_count)

            # calculate distribution for the current step
            distribution = mcts.run(n_iter=self._params.n_iter)

            # select the action from the distribution
            action_index = self._np_random.choice(len(distribution),
                                                  p=distribution)

            # get the action from the action index
            action = self._env_adapter.get_allowed_actions(state)[action_index]

            # clone the state and step
            state = self._env_adapter.clone_state(state)
            done = self._env_adapter.step(state, action, self._env_logger)

            step_count += 1

            if step_count % self._params.save_snapshot_step_frequency == 0:
                self._env_adapter.save_state(
                    state, self._env_logger,
                    self._params.snapshot_filename_prefix)

            if render_queue is not None:
                render_state = self._env_adapter.get_render_state(state)
                render_queue.put(render_state)

        self._env_adapter.save_state(state, self._env_logger,
                                     self._params.snapshot_filename_prefix)
