from typing import Any, Self
import math


class MCTSNode:

    _value: float
    _parent: Self | None
    _children: list[Self] | None

    _simulation_count: int
    _state: Any
    _input_action: Any | None
    _done: bool

    def __init__(self, state: Any, input_action: Any, done: bool):
        self._value = 0.
        self._simulation_count = 0
        self._state = state
        self._input_action = input_action
        self._parent = None
        self._children = None
        self._done = done

    def add_simulation(self, value: float):
        self._value += value
        self._simulation_count += 1

    def get_max_ucb1_child(self) -> tuple[Self | None, int]:

        if self._children is None or len(self._children) == 0:
            # not expanded or there is no child
            return None, -1

        max_i = -1
        max_ucb1 = float('-inf')

        for i, child in enumerate(self._children):

            ucb1 = self._get_child_ucb1(child)

            if ucb1 > max_ucb1:
                max_ucb1 = ucb1
                max_i = i

        assert max_i >= 0
        return self._children[max_i], max_i

    # calculate UCB1 score of a child
    def _get_child_ucb1(self, child: Self):
        if child._simulation_count == 0:
            # there was no simulations, maximum exploration incentive
            return float('inf')

        C = 1.  # exploration scaler
        N = self._simulation_count
        n = child._simulation_count

        return (child._value / n) + C * math.sqrt(math.log(N, math.e) / n)

    @property
    def state(self):
        return self._state

    @property
    def parent(self):
        return self._parent

    @property
    def is_expanded(self):
        return self._children is not None

    @property
    def is_simulated(self):
        return self._simulation_count > 0

    @property
    def is_done(self):
        return self._done

    @property
    def value(self):
        return self._value

    @property
    def simulation_count(self):
        return self._simulation_count

    def add_child(self, child: Self):
        assert child._parent is None, 'child already have a parent'

        if self._children is None:
            self._children = []

        self._children.append(child)
        child._parent = self

    @property
    def children(self):
        return self._children

    def __str__(self):
        avgval = self._value / self.simulation_count if self._simulation_count > 0 else 0
        result = f'act={self._input_action},n={self._simulation_count},val={self._value:.1f},avgval={avgval:.1f},done={self._done}'
        return result
