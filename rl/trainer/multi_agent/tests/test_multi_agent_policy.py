import deepdiff
import gymnasium

from tianshou.policy import BasePolicy
from tianshou.env import DummyVectorEnv
from tianshou.data import VectorReplayBuffer, Collector
from tianshou.utils.torch_utils import policy_within_training_step

from trainer.multi_agent.parallel_pettingzoo_env import ParallelPettingZooEnv
from trainer.multi_agent.parallel_multi_agent_policy_manager import ParallelMultiAgentPolicyManager

from .mock_trajectory import AgentTrajectory, create_agent_trajectory
from .mock_parallel_env import MockParallelEnv
from .mock_policy import MockPolicy


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
    collector.collect(n_step=8, reset_before_collect=True)

    expected = mock_policies['a0'].create_expected_forward_calls(
        [env0_agent0, env1_agent0], step_count=3, with_reset=True)

    sampled_batch, indices = collector.buffer.sample(0)

    processed_batch = multi_agent_policy.process_fn(sampled_batch,
                                                    collector.buffer, indices)

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
    with policy_within_training_step(multi_agent_policy):
        multi_agent_policy.update(sample_size=None,
                                  buffer=collector.buffer,
                                  batch_size=4,
                                  repeat=1)

    # REVIEW: create separate test from state loading
    # save the state
    state = multi_agent_policy.state_dict()

    expected_state = {
        'a1': {
            'mock_policy_agent': 'a1'
        },
        'a0': {
            'mock_policy_agent': 'a0'
        }
    }

    diff = deepdiff.DeepDiff(expected_state, state)
    assert diff == {}

    # load state
    result = multi_agent_policy.load_state_dict(state)

    state_policy = agent_policies['a1']
    assert isinstance(state_policy, MockPolicy)
    state_policy.assert_load_state_dict_called({'mock_policy_agent': 'a1'})
