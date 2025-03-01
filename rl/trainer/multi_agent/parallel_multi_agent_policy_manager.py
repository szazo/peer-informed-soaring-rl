from abc import ABC
import logging
from typing import TypeVar, Generic, Any, cast, Protocol, Self, Mapping, Iterator
from collections.abc import Hashable
import numpy as np
import gymnasium
import torch
from tianshou.data.batch import Batch, BatchProtocol, TArr
from tianshou.data import ReplayBuffer
from tianshou.policy import BasePolicy, TrainingStats
from tianshou.data.types import (
    ActStateBatchProtocol,
    ObsBatchProtocol,
    RolloutBatchProtocol,
)
from tianshou.policy.multiagent.mapolicy import MapTrainingStats
from utils.vector import VectorNxN, VectorNxNxN

TAgentID = TypeVar('TAgentID', bound=Hashable)


class MultiAgentTrainingStats(TrainingStats):
    pass


class AgentInfoBatchProtocol(BatchProtocol, Protocol):
    agent_mask: list[bool]


# class InfoBatchProtocol(BatchProtocol, Protocol, Generic[TAgentID]):
#     agent_mask: dict[TAgentID, AgentInfoBatchProtocol]


class ParallelMultiAgentBatchProtocol(BatchProtocol, Protocol,
                                      Generic[TAgentID]):
    """Observations of an environment that a policy can turn into actions.

    Typically used inside a policy's forward
    """

    obs: dict[TAgentID, VectorNxNxN]
    info: dict[TAgentID, AgentInfoBatchProtocol]


class ParallelMultiAgentRolloutBatchProtocol(BatchProtocol, Protocol,
                                             Generic[TAgentID]):
    """Typically, the outcome of sampling from a replay buffer."""

    obs: dict[TAgentID, VectorNxNxN]
    obs_next: dict[TAgentID, VectorNxNxN]
    act: dict[TAgentID, np.ndarray]
    rew: VectorNxN
    terminated: TArr
    truncated: TArr
    info: dict[TAgentID, AgentInfoBatchProtocol]


# TAgentObsPreprocessor = Callable[[Any], Any]
TAgentObsPreprocessor = str


# TODO: IMPLEMENT
class PolicyObservationPreprocessor(ABC):
    pass


class GliderNonLearningPolicyObservationPreprocessor(
        PolicyObservationPreprocessor):
    """Removes other agents' observations from the observation, so single agent policy can be used for it."""

    pass


class ParallelMultiAgentPolicyManager(BasePolicy[MapTrainingStats],
                                      Generic[TAgentID]):

    _policies: dict[TAgentID, BasePolicy]
    _learn_agent_ids: list[TAgentID]
    _agent_observation_preprocessors: dict[TAgentID, TAgentObsPreprocessor]
    _initialized: bool

    def __init__(
        self,
        policies: dict[TAgentID, BasePolicy],
        learn_agent_ids: list[TAgentID],
        agent_observation_preprocessors: dict[TAgentID,
                                              TAgentObsPreprocessor] = {}):

        # fill a dummy action space, because the more complex is not used
        super().__init__(action_space=gymnasium.spaces.Discrete(n=1))

        self._log = logging.getLogger(__class__.__name__)

        self._policies = policies
        self._learn_agent_ids = learn_agent_ids
        self._agent_observation_preprocessors = agent_observation_preprocessors
        self._initialized = True

    def named_parameters(
        self,
        prefix: str = '',
        recurse: bool = True,
        remove_duplicate: bool = True
    ) -> Iterator[tuple[str, torch.nn.Parameter]]:

        for agent_id in self._learn_agent_ids:
            agent_policy = self._policies[agent_id]

            yield from agent_policy.named_parameters(
                prefix=prefix,
                recurse=recurse,
                remove_duplicate=remove_duplicate)

    @property
    def learn_agent_ids(self) -> list[TAgentID]:
        return self._learn_agent_ids

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        strict: bool = True,
                        assign: bool = False):

        missing_keys = []
        unexpected_keys = []

        for agent_id in self._learn_agent_ids:
            agent_policy = self._policies[agent_id]

            if agent_id in state_dict:
                agent_state = state_dict[str(agent_id)]
                agent_result = agent_policy.load_state_dict(agent_state,
                                                            strict=strict,
                                                            assign=assign)

                missing_keys += agent_result.missing_keys
                unexpected_keys += agent_result.unexpected_keys

        return torch.nn.modules.module._IncompatibleKeys(
            missing_keys, unexpected_keys)

    def state_dict(self,
                   *args,
                   destination: dict[str, Any] | None = None,
                   prefix: str = '',
                   keep_vars: bool = False) -> dict[str, Any]:

        target: dict[str, Any] = {}
        if destination is not None:
            target = destination

        # save for each learning agents
        for agent_id in self._learn_agent_ids:
            agent_policy = self._policies[agent_id]
            agent_state = agent_policy.state_dict(*args,
                                                  prefix=prefix,
                                                  keep_vars=keep_vars)

            target[str(agent_id)] = agent_state

        return target

    def forward(self,
                batch: ObsBatchProtocol,
                state: dict | BatchProtocol | np.ndarray | None = None,
                **kwargs: Any) -> ActStateBatchProtocol:

        # for each agent currently in the batch execute the associated policy and return the action

        multi_batch = cast(ParallelMultiAgentBatchProtocol[TAgentID], batch)

        obs = batch.obs
        batch_size = obs.shape[0]

        new_agent_actions = {}

        obs = multi_batch.obs
        info = multi_batch.info

        agent_ids = obs.keys()
        for agent_id in agent_ids:

            agent_obs = obs[agent_id]
            agent_info = info[agent_id]

            batch_size = agent_obs.shape[0]

            # get the agent's mask (which env contain this agent)
            # this is at the end of previous step
            agent_env_mask = agent_info.agent_mask

            # filter the observation based on the env mask for the agent
            filtered_agent_obs = agent_obs[agent_env_mask]
            filtered_agent_info = agent_info[agent_env_mask]

            if filtered_agent_obs.shape[0] == 0:
                # neither of the envs contains the agent, skip
                continue

            # preprocess the observation if required for the agent
            filtered_agent_obs = self._preprocess_agent_observation(
                agent_id, filtered_agent_obs)
            # print('PREPROCESS BATCH', filtered_agent_obs.shape)

            # get the agent's batch
            agent_batch = Batch(obs=filtered_agent_obs,
                                info=filtered_agent_info)

            # find the policy for the agent
            agent_policy = self._policies[agent_id]

            # execute the corresponding policy
            # print('CALLING POLICY', agent_id, filtered_agent_obs.shape)
            agent_policy_result = agent_policy(batch=agent_batch, **kwargs)
            # print('POLICY RESULT', agent_id, agent_policy_result)

            # create the action array which match the batch size
            act_shape = (batch_size, *agent_policy_result.act.shape[1:])
            # print('act_shape', act_shape)
            act = np.full(act_shape, np.nan)

            # place into the per-agent action matrix as numpy values
            result_act = agent_policy_result.act
            if isinstance(result_act, torch.Tensor):
                result_act = result_act.detach().cpu().numpy()

            act[agent_env_mask] = result_act
            new_agent_actions[agent_id] = act

        # print('MAP', self._agent_id_to_index_map)
        # action_array = self._convert_agent_actions_to_matrix(
        #     batch_size=batch_size,
        #     actions=new_agent_actions)

        # print('NEW AGENT ACTIONS', new_agent_actions)
        # print('ARRAY', action_array)

        result_batch = Batch({
            'act': new_agent_actions,
        })

        return cast(ActStateBatchProtocol, result_batch)

    def _convert_agent_actions_to_matrix(self, batch_size: int,
                                         actions: dict[TAgentID, Any]):
        """Create an array which represents the actions for each possible agents.
        Fill the values from the current agents' actions based on possible agent indices."""

        result = np.zeros((batch_size, len(self._agent_id_to_index_map)))
        for agent_id, action in actions.items():
            index = self._agent_id_to_index_map[agent_id]
            # print('INDEX', index, action)
            result[:, index] = action

        return result

    def map_action(
        self,
        act: TArr,
    ) -> np.ndarray:

        return act

    def process_fn(
        self,
        batch: ParallelMultiAgentRolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> dict[TAgentID, RolloutBatchProtocol]:

        # for each learning agent select the indices in the batch which they are present at
        # learn_agents: list[TAgentID] = ['a0', 'a1', 'a2']
        # learn_agents: list[TAgentID] = ['a0','a1', 'a2']
        # learn_agents: list[TAgentID] = ['a1', 'a0', 'a2']

        results: dict[TAgentID, RolloutBatchProtocol] = {}

        for agent_id in self._learn_agent_ids:
            # print('extracting agent', agent_id)

            agent_batch, mask = self._extract_agent_from_batch(agent_id, batch)
            if agent_batch is None:
                # no data for the agent, skip
                continue

            # print('agent obs', agent_id, agent_batch.obs)

            assert mask is not None

            # mask the indices
            masked_indices = indices[mask]

            # call the process_fn in the corresponding policy for this agent
            agent_policy = self._policies[agent_id]

            # if the observation of an agent need to preprocessed for a specific agent
            # before sending to the policy that is done here
            agent_batch = self._preprocess_agent_batch(agent_id, agent_batch)

            agent_process_fn_result = agent_policy.process_fn(
                agent_batch, buffer, masked_indices)

            results[agent_id] = agent_process_fn_result

        return results

    def _preprocess_agent_batch(
            self, agent_id: TAgentID,
            batch: RolloutBatchProtocol) -> RolloutBatchProtocol:

        if agent_id not in self._agent_observation_preprocessors:
            return batch

        assert isinstance(batch.obs, np.ndarray)
        assert isinstance(batch.obs_next, np.ndarray)

        new_batch = Batch(batch)
        new_batch.obs = self._preprocess_agent_observation(agent_id, batch.obs)
        new_batch.obs_next = self._preprocess_agent_observation(
            agent_id, batch.obs_next)

        return cast(RolloutBatchProtocol, new_batch)

    def _preprocess_agent_observation(self, agent_id: TAgentID,
                                      obs: np.ndarray):

        if agent_id not in self._agent_observation_preprocessors:
            return obs

        # HACK: need to implement agent specific preprocessor, here we just use the first
        first_agent_obs = obs[..., 0, :, :]

        return first_agent_obs

    def learn(self, batch: dict[TAgentID, RolloutBatchProtocol], *args: Any,
              **kwargs: Any) -> MapTrainingStats | TrainingStats:
        """Perform the back-propagation."""

        agent_stats: dict[Any, TrainingStats] = {}
        for agent_id in self._learn_agent_ids:

            if not agent_id in batch:
                continue

            agent_batch = batch[agent_id]

            # call the policy's learn
            agent_policy = self._policies[agent_id]
            stats = agent_policy.learn(batch=agent_batch, **kwargs)

            agent_stats[agent_id] = stats

        if len(agent_stats) == 0:
            return TrainingStats()
        else:
            return MapTrainingStats(agent_stats)

    def __setattr__(self, name, value):
        """Sets training specific state variables in the learning policies. """

        if hasattr(self, '_initialized') and self._initialized:

            if name == 'is_within_training_step':
                for agent_id in self._learn_agent_ids:
                    agent_policy = self._policies[agent_id]
                    agent_policy.is_within_training_step = value
            elif name == 'updating':
                for agent_id in self._learn_agent_ids:
                    agent_policy = self._policies[agent_id]
                    agent_policy.updating = value
            else:
                raise ValueError(
                    f'not supported property setter {name}={value}')

        super().__setattr__(name, value)

    # updating a train method that set all sub-policies to train mode.
    # No need for a similar eval function, as eval internally uses the train functio.
    def train(self, mode: bool = True) -> Self:

        for agent_id in self._learn_agent_ids:
            agent_policy = self._policies[agent_id]
            agent_policy.train(mode)

        return self

    def _extract_agent_from_batch(
        self, agent_id: TAgentID, batch: ParallelMultiAgentRolloutBatchProtocol
    ) -> tuple[RolloutBatchProtocol, np.ndarray] | tuple[None, None]:

        if not agent_id in batch.obs:
            return None, None

        obs = batch.obs[agent_id]
        act = batch.act[agent_id]
        obs_next = batch.obs_next[agent_id]
        rew = batch.rew
        info_next = batch.info[agent_id]

        terminated = info_next.terminated
        truncated = info_next.truncated

        # print('obs', obs)
        # print('act', act)
        # print('obs_next', obs_next)
        # print('rew', rew)
        # print('terminated', terminated)
        # print('truncated', truncated)
        # print('info_next', info_next)

        # determine the mask (it is based on the s_t-1)
        mask: np.ndarray = info_next['agent_mask_before']

        # slice
        obs = obs[mask]
        act = act[mask]
        obs_next = obs_next[mask]
        rew = rew[mask]
        info_next = info_next[mask]
        terminated = terminated[mask]
        truncated = truncated[mask]
        done = np.logical_or(terminated, truncated)

        if len(obs) == 0:
            # no data for this agent
            return None, None

        # select the rewards from the reward matrix for the agent
        rew_idx = info_next['rew_idx']
        assert np.all(np.isclose(
            rew_idx,
            rew_idx[0])), f'reward index should be the same; rew_idx={rew_idx}'
        rew = rew[:, rew_idx[0]]

        # create the result batch
        result_batch = Batch(obs=obs,
                             act=act,
                             obs_next=obs_next,
                             rew=rew,
                             terminated=terminated,
                             truncated=truncated,
                             done=done,
                             info=info_next)

        return cast(RolloutBatchProtocol, result_batch), mask
