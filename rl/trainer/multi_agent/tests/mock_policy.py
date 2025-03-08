from typing import Any, cast, Mapping
import deepdiff
import gymnasium
import numpy as np
import torch

from tianshou.policy import TrainingStats, BasePolicy
from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ObsBatchProtocol, ActBatchProtocol, RolloutBatchProtocol

from .mock_trajectory import ActionStep, AgentTrajectory, Obs, agent_trajectory_generator, create_action, create_expected_agent_trajectory_rollout_batch


class MockTrainingStats(TrainingStats):
    pass


class MockPolicy(BasePolicy):

    _agent_id: str
    _observation_space: gymnasium.spaces.Box
    _action_space: gymnasium.spaces.Box

    _last_process_fn_result: RolloutBatchProtocol | None

    _loaded_state: Mapping[str, Any] | None

    _forward_inputs_outputs: list[tuple[np.ndarray, np.ndarray]]

    def __init__(self, agent_id: str,
                 observation_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space):

        super().__init__(action_space=action_space)

        assert isinstance(observation_space, gymnasium.spaces.Box)
        assert isinstance(action_space, gymnasium.spaces.Box)
        assert len(
            action_space.shape) == 1, 'one dimensional action space required'

        self._agent_id = agent_id
        self._observation_space = observation_space
        self._action_space = action_space
        self._last_process_fn_result = None
        self._loaded_state = None
        self._forward_inputs_outputs = []

    def load_state_dict(self,
                        state_dict: Mapping[str, Any],
                        strict: bool = True,
                        assign: bool = False):
        self._loaded_state = state_dict
        return torch.nn.modules.module._IncompatibleKeys([], [])

    def assert_load_state_dict_called(self, expected_state: Mapping[str, Any]):
        assert self._loaded_state is not None
        diff = deepdiff.DeepDiff(expected_state, self._loaded_state)
        assert diff == {}

    def state_dict(self,
                   *args,
                   destination: dict[str, Any] | None = None,
                   prefix: str = '',
                   keep_vars: bool = False) -> dict[str, Any]:

        if destination is None:
            destination = {}

        destination['mock_policy_agent'] = self._agent_id

        return destination

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:

        obs = batch.obs
        info = batch.info

        assert isinstance(info, Batch)

        batch_size = obs.shape[0]

        actions = np.zeros((batch_size, self._action_space.shape[0]))
        for i in range(batch_size):
            agent_id: int = info.agent_idx[i]
            step: int = info.step[i]
            env: int = info.env[i]

            action = create_action(step=step, env_id=env, agent_id=agent_id)
            actions[i] = action
        actions = np.float32(actions)

        result = Batch(act=actions)

        assert isinstance(batch.obs, np.ndarray)
        assert isinstance(actions, np.ndarray)
        self._forward_inputs_outputs.append((batch.obs, actions))

        result = Batch(act=actions, state=41, dist=42)
        result = cast(ActBatchProtocol, result)

        return result

    def assert_forward_called_with(self, expected: list[tuple[np.ndarray,
                                                              np.ndarray]]):

        diff = deepdiff.DeepDiff(expected, self._forward_inputs_outputs)
        assert diff == {}

        pass

    def create_expected_forward_calls(
            self, env_agent_trajectories: list[AgentTrajectory],
            step_count: int, with_reset: bool):

        env_agent_steps: list[list[None | tuple[Obs, ActionStep, Obs]]] = []

        # generate obs, act, obs_next pairs for each env and agent
        env_count = len(env_agent_trajectories)
        for env_id in range(env_count):

            agent_trajectory = env_agent_trajectories[env_id]

            generator = agent_trajectory_generator(
                env_id,
                agent_trajectory.agent_idx,
                step_offset=agent_trajectory.step_offset,
                length=agent_trajectory.length)

            steps = list(generator)
            env_agent_steps.append(steps)

        # create the expected forward calls
        forward_inputs_outputs: list[tuple[np.ndarray, np.ndarray]] = []

        env_step_indices = [0] * env_count

        for current_step in range(step_count):

            obs = np.empty([0, *self._observation_space.shape])
            act = np.empty([0, *self._action_space.shape], dtype=np.float32)

            found_env = False

            # iterate every env
            for env_id in range(len(env_agent_steps)):

                agent_steps = env_agent_steps[env_id]

                agent_step = None
                agent_step_index = env_step_indices[env_id]
                if agent_step_index >= 0 and agent_step_index < len(
                        agent_steps):
                    agent_step = agent_steps[agent_step_index]
                if agent_step is not None:

                    found_env = True

                    obs = np.vstack(
                        (obs, np.expand_dims(agent_step[0].obs, axis=0)))
                    act = np.vstack(
                        (act,
                         np.expand_dims(agent_step[1].action.astype(
                             np.float32),
                                        axis=0)))

                # increase the step
                env_step_indices[env_id] += 1

            if found_env:
                forward_inputs_outputs.append((obs, act))
            else:
                # terminated
                if with_reset:
                    env_step_indices = [0] * env_count

        return forward_inputs_outputs

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:

        # split the batch by env indices
        info = batch.info
        assert isinstance(info, Batch)

        env_idxs = info['env']
        agent_idx = info['agent_idx'][0]

        # compute the episodic return for the whole batch
        batch_returns, _ = self.compute_episodic_return(
            batch,
            buffer,
            indices,
            v_s_=None,
            gamma=1.,
            gae_lambda=1.0,
        )

        # assert the input batch for each environment
        for env_idx in np.unique(env_idxs):

            mask = env_idxs == env_idx

            env_batch = batch[mask]

            masked_indices = indices[mask]

            step_offset = int(env_batch.info['step'][0]) - 1
            length = len(env_batch.info['step'])

            # create expected batch
            # REVIEW: refactor to separate assertion method, because
            # the length should be allowed to set from the outside
            expected_batch = create_expected_agent_trajectory_rollout_batch(
                env_id=env_idx,
                agent_id=agent_idx,
                step_offset=step_offset,
                length=length)

            # check whether correct batch arrived
            assert np.allclose(expected_batch.obs, env_batch.obs)
            assert np.allclose(expected_batch.act, env_batch.act)
            assert np.allclose(expected_batch.obs_next, env_batch.obs_next)
            assert np.allclose(expected_batch.terminated, env_batch.terminated)
            assert np.allclose(expected_batch.truncated, env_batch.truncated)
            assert np.allclose(expected_batch.rew, env_batch.rew)
            assert np.allclose(expected_batch.done, env_batch.done)

            # assert that env returns does not leak outside of the env
            env_returns, _ = self.compute_episodic_return(
                env_batch,
                buffer,
                masked_indices,
                v_s_=None,
                gamma=1.,
                gae_lambda=1.0,
            )

            env_batch.returns = env_returns
            assert np.allclose(batch_returns[mask], env_returns)

        # return the returns for the agent for each env
        batch.returns = batch_returns
        batch.act = to_torch(batch.act)

        self._last_process_fn_result = batch

        return batch

    def learn(self, batch: RolloutBatchProtocol, *args: Any,
              **kwargs: Any) -> MockTrainingStats:

        # check whether the result from process_fn arrived here
        assert self._last_process_fn_result == batch

        dist = self(batch)

        self(batch)

        return MockTrainingStats()
