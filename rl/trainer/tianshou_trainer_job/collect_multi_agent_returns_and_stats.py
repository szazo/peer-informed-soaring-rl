from typing import TypeVar
from dataclasses import dataclass
import numpy as np

from tianshou.data import SequenceSummaryStats
from utils.vector import VectorN


@dataclass
class AgentReturns:
    returns: VectorN
    returns_stat: SequenceSummaryStats


AgentIDType = TypeVar('AgentIDType', bound=str)


def collect_multi_agent_returns_and_stats(returns: np.ndarray,
                                          possible_agents: list[AgentIDType]):

    assert returns.ndim == 2, 'two dimensional returns required'
    assert returns.shape[1] == len(
        possible_agents), 'columns should represent the agents'

    result: dict[AgentIDType, AgentReturns] = {}

    # create per agent return stat
    for i, agent_id in enumerate(possible_agents):
        agent_returns = returns[:, i]
        # TODO: now zero reward is used as placeholder, filter it
        # (it is possible that there are agents which can return zero reward also for existing agent)
        nonzero_agent_returns = agent_returns[agent_returns != 0.]
        agent_returns_stat = SequenceSummaryStats.from_sequence(
            nonzero_agent_returns)

        result[agent_id] = AgentReturns(returns=nonzero_agent_returns,
                                        returns_stat=agent_returns_stat)

    return result
