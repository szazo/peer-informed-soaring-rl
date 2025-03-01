from typing import TypeVar, Generic
from abc import ABC, abstractmethod

TState = TypeVar('TState')
TAction = TypeVar('TAction')
TRenderState = TypeVar('TRenderState')
TLogger = TypeVar('TLogger')


class EnvAdapter(Generic[TState, TAction, TRenderState, TLogger], ABC):

    @abstractmethod
    def simulate(self, state: TState, max_simulation_step_count: int) -> float:
        """
        Start simulation from the current node.
        """
        pass

    @abstractmethod
    def get_allowed_actions(self, state: TState) -> list[TAction]:
        """
        Returns the allowed actions for the current state.
        """
        pass

    @abstractmethod
    def clone_state(self, state: TState) -> TState:
        """
        Clone the state
        """
        pass

    @abstractmethod
    def start(self, state: TState, logger: TLogger | None):
        """
        Called at the beginning of the mcts search process at the initial state
        """
        pass

    @abstractmethod
    def step(self, state: TState, action: TAction,
             logger: TLogger | None) -> bool:
        """
        Step the state (with side effect), returns done
        """
        pass

    @abstractmethod
    def get_render_state(self, state: TState) -> TRenderState:
        """
        Return the data which can be used for rendering the current state
        """
        pass

    @abstractmethod
    def save_state(self, state: TState, logger: TLogger | None,
                   filename_prefix: str):
        """
        Serializes the current state into the specified file
        """
        pass
