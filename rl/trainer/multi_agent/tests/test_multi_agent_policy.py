from typing import Any, cast, Mapping
import deepdiff
import gymnasium
import pettingzoo
import numpy as np
import torch
from pytest_mock import MockerFixture

from pettingzoo.utils import BaseParallelWrapper

from tianshou.policy import TrainingStats, BasePolicy, PPOPolicy
from tianshou.env import DummyVectorEnv, SubprocVectorEnv
from tianshou.data import VectorReplayBuffer, Batch, Collector, ReplayBuffer
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ObsBatchProtocol, ActBatchProtocol, RolloutBatchProtocol
from tianshou.utils.torch_utils import policy_within_training_step

from env.glider.base.wrappers import (SequenceWindowObsParams,
                                      pad_sequence_obs_wrapper,
                                      sequence_window_obs_wrapper,
                                      pad_sequence_observation_wrapper,
                                      PadSequenceObsParams)

from trainer.multi_agent.parallel_pettingzoo_env import ParallelPettingZooEnv
from trainer.multi_agent.parallel_multi_agent_policy_manager import ParallelMultiAgentPolicyManager

from .mock_trajectory import AgentTrajectory, agent_trajectory_generator, create_action, create_agent_trajectory, create_expected_agent_trajectory_rollout_batch
from .mock_parallel_env import MockParallelEnv
from .mock_policy import MockPolicy


def test_should_call_policy_when_single_env_two_agents_intersected_trajectories(
):

    return

    # given
    agent0 = create_agent_trajectory(env_id=0,
                                     agent_id=0,
                                     step_offset=0,
                                     length=2)
    agent1 = create_agent_trajectory(env_id=0,
                                     agent_id=1,
                                     step_offset=1,
                                     length=2)

    env = MockParallelEnv('env0', [agent0, agent1])
    env = ParallelPettingZooEnv(name='env0', env=env)

    action_space = env.action_space

    agent_policies: dict[str, BasePolicy] = {
        agent_id: MockPolicy(agent_id, action_space=action_space[agent_id])
        for agent_id in env.possible_agents
    }
    # agent_policies: dict[str, BasePolicy] = {
    #     agent_id:MockPolicy(agent_id, action_space=env.action_space) for agent_id in env.possible_agents
    # }
    venv = DummyVectorEnv([lambda: env])

    multi_agent_policy = ParallelMultiAgentPolicyManager(
        policies=agent_policies,
        #        agent_id_to_index_map=env.agent_id_to_index_map,
        action_space=env.action_space)

    env_num = 1
    single_buffer_size = 6
    replay_buffer = VectorReplayBuffer(total_size=single_buffer_size * env_num,
                                       buffer_num=env_num)

    collector = Collector(multi_agent_policy, venv, buffer=replay_buffer)

    collect_stats = collector.collect(n_step=3, reset_before_collect=True)

    # try to learn
    with policy_within_training_step(multi_agent_policy):
        multi_agent_policy.update(sample_size=None,
                                  buffer=collector.buffer,
                                  batch_size=4,
                                  repeat=1)

    # batch, indices = collector.buffer.sample(batch_size=None)

    # print(batch, indices)

    # pass


def test_should_step_sandbox():

    env0_agent0 = create_agent_trajectory(env_id=0,
                                          agent_id=0,
                                          step_offset=0,
                                          length=2)
    env0_agent1 = create_agent_trajectory(env_id=0,
                                          agent_id=1,
                                          step_offset=1,
                                          length=2)

    env = MockParallelEnv('env0', [env0_agent0, env0_agent1])
    env = ParallelPettingZooEnv(name='env0', env=env)

    print('env', env)

    obs0, _ = env.reset()
    obs1, reward1, *_ = env.step(dict(a0=np.array([0., 1.])))
    obs2, reward2, *_ = env.step(
        dict(a0=np.array([1., 2.]), a1=np.array([11., 12.])))
    obs3, reward3, *_ = env.step(dict(a1=np.array([12., 13.])))

    print('obs0', obs0)
    print('obs1', obs1)
    print('obs2', obs2)
    print('obs3', obs3)


def test_sandbox2():
    # given
    env0_agent0 = create_agent_trajectory(env_id=0,
                                          agent_id=0,
                                          step_offset=0,
                                          length=2)
    env0_agent1 = create_agent_trajectory(env_id=0,
                                          agent_id=1,
                                          step_offset=1,
                                          length=3)

    env0 = _create_env_from_trajectories('env0', [env0_agent0, env0_agent1])
    mock_policies = _create_mock_policies(env0)
    multi_agent_policy = _create_multi_agent_policy(mock_policies,
                                                    learn_agent_ids=['a0'])

    collector = _create_collector(multi_agent_policy, [env0],
                                  single_buffer_size=10)
    collector.collect(n_step=4, reset_before_collect=True)

    print(collector.buffer)


def test_collect_should_call_forward_with_the_good_obs():

    # given
    env0_agent0 = create_agent_trajectory(env_id=0,
                                          agent_id=0,
                                          step_offset=0,
                                          length=2)
    env0_agent1 = create_agent_trajectory(env_id=0,
                                          agent_id=1,
                                          step_offset=1,
                                          length=3)

    env1_agent0 = create_agent_trajectory(env_id=1,
                                          agent_id=0,
                                          step_offset=1,
                                          length=3)
    env1_agent1 = create_agent_trajectory(env_id=1,
                                          agent_id=1,
                                          step_offset=0,
                                          length=2)

    env0 = _create_env_from_trajectories('env0', [env0_agent0, env0_agent1])
    env1 = _create_env_from_trajectories('env1', [env1_agent0, env1_agent1])
    mock_policies = _create_mock_policies(env0)
    multi_agent_policy = _create_multi_agent_policy(mock_policies,
                                                    learn_agent_ids=['a0'])

    collector = _create_collector(multi_agent_policy, [env0, env1],
                                  single_buffer_size=10)
    collector.collect(n_step=3, reset_before_collect=True)

    # print(collector.buffer)
    # return

    expected = mock_policies['a0'].create_expected_forward_calls(
        [env0_agent0, env1_agent0], step_count=3, with_reset=True)

    for step, d in enumerate(expected):
        print('expected', step, d)

    # for obs, act, obs_next in agent_trajectory_generator(env_id=0, agent_id=0, step_offset=1, length=3):

    #     print('OBS', obs, 'ACT', act, 'OBS_NEXT', obs_next)

    # print(collector.buffer)
    # print(env0_agent0)
    # print(env0_agent1)
    for step, d in enumerate(mock_policies['a0']._forward_inputs_outputs):
        print('actual', step, d)

    # assert forward calls


#    mock_policies['a0'].assert_forward_called_with(expected)

# diff = deepdiff.DeepDiff(expected, mock_policies['a0']._forward_inputs_outputs)
# assert diff == {}

    sampled_batch, indices = collector.buffer.sample(0)
    print('SAMPLED obs', sampled_batch.obs)
    print('SAMPLED act', sampled_batch.act)
    print('SAMPLED obs_next', sampled_batch.obs_next)

    processed_batch = multi_agent_policy.process_fn(sampled_batch,
                                                    collector.buffer, indices)
    print('PROCESSED BATCH', processed_batch)

    # print('BUFFER obs', collector.buffer.obs)
    # print('BUFFER act', collector.buffer.act)
    # print('BUFFER rew', collector.buffer.rew)
    # print('BUFFER obs_next', collector.buffer.obs_next)

    with policy_within_training_step(multi_agent_policy):
        multi_agent_policy.update(sample_size=None,
                                  buffer=collector.buffer,
                                  batch_size=3,
                                  repeat=1)


def _create_env_from_trajectories(name: str,
                                  trajectories: list[AgentTrajectory]):

    env = MockParallelEnv(name, trajectories)
    env = ParallelPettingZooEnv(name=name, env=env)

    return env


def _create_mock_policies(env: ParallelPettingZooEnv):
    observation_space = env.observation_space
    assert isinstance(observation_space, gymnasium.spaces.Dict)
    action_space = env.action_space
    assert isinstance(action_space, gymnasium.spaces.Dict)

    agent_policies: dict[str, MockPolicy] = {
        agent_id:
        MockPolicy(agent_id,
                   observation_space=observation_space[agent_id],
                   action_space=action_space[agent_id])
        for agent_id in env.possible_agents
    }

    return agent_policies


def _create_multi_agent_policy(agent_policies: dict[str, MockPolicy],
                               learn_agent_ids: list[str]):

    multi_agent_policy = ParallelMultiAgentPolicyManager(
        policies=agent_policies, learn_agent_ids=learn_agent_ids)
    return multi_agent_policy


def _create_collector(multi_agent_policy: ParallelMultiAgentPolicyManager,
                      envs: list[gymnasium.Env],
                      single_buffer_size: int = 10):
    env_num = len(envs)

    env_factories = list([lambda i=i: envs[i] for i in range(env_num)])
    venv = DummyVectorEnv(env_factories)

    replay_buffer = VectorReplayBuffer(total_size=single_buffer_size * env_num,
                                       buffer_num=env_num)

    collector = Collector(multi_agent_policy, venv, buffer=replay_buffer)
    return collector


def test_should_call_policy_when_multi_env_two_agents_intersected_trajectories(
):

    # given
    env0_agent0 = create_agent_trajectory(env_id=0,
                                          agent_id=0,
                                          step_offset=0,
                                          length=3)
    env0_agent1 = create_agent_trajectory(env_id=0,
                                          agent_id=1,
                                          step_offset=5,
                                          length=2)

    env0_agent2 = create_agent_trajectory(env_id=0,
                                          agent_id=2,
                                          step_offset=0,
                                          length=2)

    env1_agent0 = create_agent_trajectory(env_id=1,
                                          agent_id=0,
                                          step_offset=0,
                                          length=2)
    env1_agent1 = create_agent_trajectory(env_id=1,
                                          agent_id=1,
                                          step_offset=1,
                                          length=3)

    env1_agent2 = create_agent_trajectory(env_id=1,
                                          agent_id=2,
                                          step_offset=1,
                                          length=5)

    # env1_agent3 = create_agent_trajectory(env_id=1,
    #                                       agent_id=3,
    #                                       step_offset=1,
    #                                       length=3)

    env0 = MockParallelEnv('env0', [env0_agent0, env0_agent1, env0_agent2])
    env0 = ParallelPettingZooEnv(name='env0', env=env0)

    env1 = MockParallelEnv('env1', [
        env1_agent0,
        env1_agent1,
        env1_agent2,
    ])
    env1 = ParallelPettingZooEnv(name='env1', env=env1)

    action_space = env0.action_space
    observation_space = env0.observation_space
    assert isinstance(observation_space, gymnasium.spaces.Dict)
    assert isinstance(action_space, gymnasium.spaces.Dict)

    agent_policies: dict[str, BasePolicy] = {
        agent_id:
        MockPolicy(agent_id,
                   observation_space=observation_space[agent_id],
                   action_space=action_space[agent_id])
        for agent_id in env0.possible_agents
    }

    #venv = SubprocVectorEnv([lambda: env0, lambda: env1])

    venv = DummyVectorEnv([lambda: env0, lambda: env1])

    multi_agent_policy = ParallelMultiAgentPolicyManager(
        policies=agent_policies, learn_agent_ids=['a0', 'a1'])

    env_num = 2
    single_buffer_size = 7
    replay_buffer = VectorReplayBuffer(total_size=single_buffer_size * env_num,
                                       buffer_num=env_num)

    collector = Collector(multi_agent_policy, venv, buffer=replay_buffer)

    collect_stats = collector.collect(n_episode=2, reset_before_collect=True)

    # try to learn
    if True:
        with policy_within_training_step(multi_agent_policy):
            multi_agent_policy.update(sample_size=None,
                                      buffer=collector.buffer,
                                      batch_size=4,
                                      repeat=1)

    # check policy learn called

    # HACK: CREATE TEST FOR STATE LOADING
    if False:
        # print('COLLECT BUFFER', collector.buffer.obs)
        # print('COLLECT BUFFER', collector.buffer.info)
        # print('COLLECT STATS', collect_stats)

        # save the state
        state = multi_agent_policy.state_dict()

        expected_state = {'a1': {'mock_policy_agent': 'a1'}}

        diff = deepdiff.DeepDiff(expected_state, state)
        assert diff == {}

        # load state
        result = multi_agent_policy.load_state_dict(state)

        state_policy = agent_policies['a1']
        assert isinstance(state_policy, MockPolicy)
        state_policy.assert_load_state_dict_called({'mock_policy_agent': 'a1'})

    # batch, indices = collector.buffer.sample(batch_size=None)

    # print(batch, indices)

    #print(replay_buffer)


def test_sandbox(mocker: MockerFixture):

    return

    env0_agent0 = create_agent_trajectory(env_id=0,
                                          agent_id=0,
                                          step_offset=0,
                                          length=2)
    env0_agent1 = create_agent_trajectory(env_id=0,
                                          agent_id=1,
                                          step_offset=1,
                                          length=2)
    # env1_agent0 = create_agent_trajectory(env_id=1, agent_id=0, step_offset=1, length=2)

    print(env0_agent0)
    print(env0_agent1)

    env0 = MockParallelEnv('env0', [env0_agent0, env0_agent1])
    print('possible agents', env0.possible_agents)

    obs, info = env0.reset()

    # print('obs:', obs, 'info: ', info)
    # print('agents', env0.agents)

    actions = dict(a0=np.array([0., 1.]))
    obs, reward, terminated, truncated, info = env0.step(actions)

    # print(obs, reward, terminated, truncated, info)
    # print('agents', env0.agents)

    actions = dict(a0=np.array([1., 2.]), a1=np.array([11., 12.]))
    obs, reward, terminated, truncated, info = env0.step(actions)

    # print(obs, reward, terminated, truncated, info)
    # print('agents', env0.agents)

    actions = dict(a1=np.array([12., 13.]))
    obs, reward, terminated, truncated, info = env0.step(actions)

    # print(obs, reward, terminated, truncated, info)
    # print('agents', env0.agents)

    #print(create_reset_result([env0_agent0]))

    # create_obs(env_id=2, agent_id=1, time=3)

    # env0_obs = [
    #     dict(agent0=[0.,1.])

    # ]
