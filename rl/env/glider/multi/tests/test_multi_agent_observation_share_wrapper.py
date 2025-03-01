from functools import partial
from env.glider.multi.apply_share_wrappers import apply_share_wrappers
from pytest_mock import MockerFixture

from trainer.multi_agent.tests.mock_trajectory import create_action, create_agent_trajectory
from trainer.multi_agent.tests.mock_parallel_env import MockParallelEnv


def test_multi_agent_observation_sandbox():
    """Include other agents' observation in the observation for each agent"""
    OBS_SHAPE = (3, )
    create_trajectory = partial(create_agent_trajectory, obs_shape=OBS_SHAPE)

    env0_agent0 = create_trajectory(env_id=0,
                                    agent_id=0,
                                    step_offset=0,
                                    length=4)
    env0_agent1 = create_trajectory(env_id=0,
                                    agent_id=1,
                                    step_offset=1,
                                    length=1)
    env0_agent2 = create_trajectory(env_id=0,
                                    agent_id=2,
                                    step_offset=4,
                                    length=2)
    env0_agent3 = create_trajectory(env_id=0,
                                    agent_id=3,
                                    step_offset=5,
                                    length=2)
    env0_agent4 = create_trajectory(env_id=0,
                                    agent_id=4,
                                    step_offset=10,
                                    length=2)
    env0 = MockParallelEnv(
        'env0',
        [env0_agent0, env0_agent1, env0_agent2, env0_agent3, env0_agent4])

    env0 = apply_share_wrappers(env0,
                                max_sequence_length=4,
                                max_closest_agent_count=5,
                                normalize_trajectories=False)

    obs, info = env0.reset()

    print('obs_reset', obs, next(iter(obs.values())).shape)

    for step in range(4):

        actions = {
            agent_id: create_action(step, agent_id=int(agent_id[1]), env_id=0)
            for agent_id in env0.agents
        }
        print('ACT', actions)

        obs, reward, terminated, truncated, info = env0.step(actions)

        # print(f'obs_step{step}', next(iter(obs.values())).shape)
        # print(f'a0_step{step}', obs['a0'])
        # print(f'a1_step{step}', obs['a1'])

        # a0_history = obs['a0'][:,0,...]
        # print('a0_history', a0_history)

        print('STEP RESULT', obs,
              next(iter(obs.values())).shape if len(obs) > 0 else 'NONE')

        for agent_id, agent_obs in obs.items():
            a0_history = agent_obs[0, ...]
            print('a0_history', a0_history)

            a1_history = agent_obs[1, ...]
            print('a1_history', a1_history)

            #print('agent_obs', agent_obs)

            # a2_history = agent_obs[:, 2, ...]
            # print('a2_history', a2_history)

            # a3_history = agent_obs[:, 3, ...]
            # print('a3_history', a3_history)

            # a4_history = agent_obs[:, 4, ...]
            # print('a4_history', a4_history)

            break

        print('agents', obs.keys())

    return

    obs, reward, terminated, truncated, info = env0.step({
        'a0':
        create_action(1, agent_id=0, env_id=0),
        'a1':
        create_action(1, agent_id=1, env_id=0)
    })

    print('obs_step2 shape', next(iter(obs.values())).shape)
    print('a0_step2', obs['a0'])
    print('a1_step2', obs['a1'])

    #   assert obs['a0'].shape == obs['a1'].shape
    #assert np.allclose(obs['a0'], obs['a1'])

    a0_history = obs['a0'][:, 0, ...]
    print('a0_history', a0_history)

    a1_history = obs['a1'][:, 1, ...]
    print('a1_history', a1_history)

    # obs, reward, terminated, truncated, info = env0.step({
    #     'a0':
    #     create_action(2, agent_id=0, env_id=0),
    # })
