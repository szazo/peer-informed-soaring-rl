from typing import TypeVar, Generic
import logging
import numpy as np
from anytree import Node, RenderTree

from .api import EnvAdapter
from .mcts_node import MCTSNode

TState = TypeVar('TState')


class MCTS(Generic[TState]):

    _log: logging.Logger
    _start_state: TState
    _root_node: MCTSNode
    _adapter: EnvAdapter
    _max_simulation_step_count: int

    def __init__(self, start_state: TState, adapter: EnvAdapter,
                 max_simulation_step_count: int):
        self._log = logging.getLogger(__class__.__name__)
        self._start_state = start_state
        self._root_node = MCTSNode(start_state, input_action=None, done=False)
        self._adapter = adapter
        self._max_simulation_step_count = max_simulation_step_count

    def render_cli(self, skip_zero_value: bool = True):

        out_root = self._convert_to_anytree(self._root_node,
                                            parent_node=None,
                                            skip_zero_value=skip_zero_value)

        for pre, fill, node in RenderTree(out_root):
            mcts_node: MCTSNode = node.name

            print('%s%s' % (pre, mcts_node))

    def _convert_to_anytree(self, mcts_node: MCTSNode,
                            parent_node: Node | None, skip_zero_value: bool):

        out_node = Node(mcts_node, parent=parent_node)

        if mcts_node.children is None:
            return out_node

        for mcts_child in mcts_node.children:
            if skip_zero_value and np.isclose(mcts_child.value, 0.0):
                continue

            self._convert_to_anytree(mcts_child,
                                     parent_node=out_node,
                                     skip_zero_value=skip_zero_value)

        return out_node

    def run(self, n_iter: int):

        # expand the root node
        self._expand_node(self._root_node)

        # REVIEW: additionally we could use horizon to wait while
        # the tree's depth reach some minimum limit
        for i in range(n_iter):
            done = self._iterate()
            if done:
                break

        # the iteration finished, return the probability distribution for the action
        # to be selected in the current step
        allowed_actions = self._adapter.get_allowed_actions(
            self._root_node.state)
        values = [float('-inf')] * len(allowed_actions)

        assert self._root_node.children is not None, 'root children missing'
        for i, child in enumerate(self._root_node.children):
            values[i] = (
                child.value /
                child.simulation_count) if child.simulation_count > 0 else 0

        # prevent overflow
        shifted_values = values - np.max(values)
        # softmax
        distribution = np.exp(shifted_values) / sum(np.exp(shifted_values))

        self._log.info('allowed actions=%s,distribution=%s', allowed_actions,
                       distribution)

        return distribution

    def _iterate(self) -> bool:
        selected_node = self._select_with_expansion()

        if selected_node.is_done:
            # we finished
            return True

        # simulate the node
        cumulative_reward = self._adapter.simulate(
            selected_node.state,
            max_simulation_step_count=self._max_simulation_step_count)

        # backpropagate
        self._backpropagate(selected_node, cumulative_reward=cumulative_reward)

        # not yet finished
        return False

    def _backpropagate(self, node: MCTSNode, cumulative_reward: float):

        cur_node = node
        while cur_node is not None:
            cur_node.add_simulation(cumulative_reward)

            cur_node = cur_node.parent

    def _select_with_expansion(self):
        selected_node = self._select(self._root_node)
        if selected_node is None:
            # no node found
            raise Exception('no node found for selection from root')

        if selected_node.is_done:
            self._log.debug(
                'terminal node found in select_with_expansion, return it')
            return selected_node

        if selected_node.is_simulated:
            self._log.debug('node is already simulated, expand it')
            assert not selected_node.is_expanded, 'the node should not have been expanded yet'

            self._expand_node(selected_node)

            # select the child
            expanded_child = self._select(selected_node)
            if expanded_child is None:
                # no node found
                raise Exception(
                    'no child node found after expanding the selected node')

            # if expanded_child is None:
            #     self._log.debug(
            #         'after expansion, no new child found, we stop here')
            #     return None

            # use the child of the expanded node as selected node
            selected_node = expanded_child

        return selected_node

    def _select(self, start_node: MCTSNode):
        """Start from the root node and select a leaf based on UCB1 score

        :returns: The found leaf node

        """

        cur_node = start_node

        # traverse using ucb1 scores until a leaf node found
        while cur_node.is_expanded:
            child_node, index = cur_node.get_max_ucb1_child()

            if child_node is None:
                # might be expanded but no child found
                break

            cur_node = child_node

        path = self._node_path(cur_node)
        message = [str(node) for node in path]
        self._log.debug('found a leaf node=%s', message)

        if cur_node is start_node:
            # no new node found
            self._log.debug('no new node found for selection')
            return None

        if cur_node.is_done:
            self._log.debug(
                'the selected node represents an end state, we return that')
            return cur_node

        return cur_node

    def _node_path(self, node: MCTSNode):

        path = []

        cur_node = node
        while cur_node is not None:
            path.append(cur_node)
            cur_node = cur_node.parent

        path.reverse()
        return path

    def _expand_node(self, node: MCTSNode):

        self._log.debug('expand_node=%s', node)

        assert not node.is_expanded, 'Node is already expanded'

        allowed_actions = self._adapter.get_allowed_actions(node.state)
        self._log.debug('allowed_actions=%s', allowed_actions)

        for act in allowed_actions:
            # clone and step the state
            new_state = self._adapter.clone_state(node.state)
            done = self._adapter.step(new_state, act, logger=None)

            # create new node and add it as a child
            new_node = MCTSNode(new_state, input_action=act, done=done)
            node.add_child(new_node)
