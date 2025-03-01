from typing import TypeVar, runtime_checkable, Protocol, Generic
import pettingzoo

AgentIDType = TypeVar('AgentIDType')
ObsType = TypeVar('ObsType')
ActType = TypeVar('ActType')


@runtime_checkable
class MultiAgentEnvProtocol(Protocol, Generic[AgentIDType]):
    possible_agents: list[AgentIDType]


def create_agent_id_index_mapping(
    env: MultiAgentEnvProtocol[AgentIDType]
    | pettingzoo.ParallelEnv[AgentIDType, ObsType, ActType]):

    agent_id_to_index: dict[AgentIDType, int] = {}
    for idx, agent_id in enumerate(env.possible_agents):
        agent_id_to_index[agent_id] = idx

    return agent_id_to_index
