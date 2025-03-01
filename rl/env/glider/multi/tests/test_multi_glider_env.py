from functools import partial
from typing import Any, cast
from env.glider.multi.multiglider_env_params import AgentGroupParams
from model.transformer.multi_level_transformer_net import MultiLevelTransformerNet
import torch
from dataclasses import dataclass, asdict
from env.glider.multi.make_multiglider_env import MultiGliderEnvParameters
from env.glider.multi.multiglider_env import make_multiglider_env
from model.actor_critic.discrete_actor_critic import create_discrete_actor_critic_net_with_common_encoder_net
from model.ppo_custom_transformer_model_config import create_ppo_custom_transformer_model
from model.transformer.transformer_net import TransformerNet, TransformerNetParameters
from pytest_mock import MockerFixture
import gymnasium
import tianshou
import numpy as np
from tianshou.utils.torch_utils import policy_within_training_step
from tianshou.data import VectorReplayBuffer, ReplayBuffer, Batch
from tianshou.data.batch import BatchProtocol
from tianshou.data.types import ObsBatchProtocol, ActBatchProtocol, RolloutBatchProtocol
from tianshou.policy import BasePolicy
from tianshou.policy.base import TrainingStats

from thermal.zero import ZeroAirVelocityField
from env.glider.aerodynamics import SimpleAerodynamicsParameters, SimpleAerodynamics

from env.glider.base import AgentID, DiscreteToContinuousDegreesParams, SimulationBoxParameters, TimeParameters
from env.glider.base.agent import (GliderCutoffParameters,
                                   GliderInitialConditionsParameters,
                                   GliderRewardParameters,
                                   GliderAgentParameters)
from env.glider.base.visualization import RenderParameters

from env.glider.multi.agent_spawner import AgentSpawnParameters2
from trainer.common import TianshouModelConfigBase
from trainer.multi_agent.parallel_multi_agent_policy_manager import ParallelMultiAgentPolicyManager

from trainer.multi_agent.parallel_pettingzoo_env import ParallelPettingZooEnv


class RandomTrainingStats(TrainingStats):
    pass


class RandomPolicy(tianshou.policy.BasePolicy):

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol:

        obs = batch.obs

        batch_size = obs.shape[0]
        # print('batch size', obs.shape)
        # print('input', obs)

        actions = np.zeros(batch_size)
        for i in range(batch_size):
            actions[i] = self.action_space.sample()
        #actions = np.expand_dims(actions, axis=0).T
        # print('RANDOM Action', actions)

        result = tianshou.data.Batch(act=actions)

        # print('result shape', actions.shape)

        return cast(ActBatchProtocol, result)

    def process_fn(
        self,
        batch: RolloutBatchProtocol,
        buffer: ReplayBuffer,
        indices: np.ndarray,
    ) -> RolloutBatchProtocol:

        print('RandomPolicy process_fn', batch)

        return batch

    def learn(self, batch: RolloutBatchProtocol, *args: Any,
              **kwargs: Any) -> RandomTrainingStats:
        """Since a random agent learns nothing, it returns an empty dict."""
        return RandomTrainingStats()


def test_sandbox_simple(mocker: MockerFixture):

    return

    agent_pool_size = 9
    instances1 = _create_env(mocker,
                             name='env0',
                             pool_size=agent_pool_size,
                             pool_offset=0,
                             sequence_length=2,
                             max_closest_agent_count=4)
    env1 = instances1.env

    obs, info = env1.reset()
    print('obs', obs)

    step_count = 10
    for i in range(step_count):

        actions = {}
        for agent_id in env1.agents:
            actions[agent_id] = 0.
        print('call actions', actions)

        obs, reward, terminated, truncated, info = env1.step(actions)
        print('obs, reward, terminated, truncated', obs.keys(), reward,
              terminated, truncated)
        print('obs', obs, next(iter(obs.values())).shape)

    # print(env.observation_space(env.possible_agents[0]))

    # print(env.action_space(env.possible_agents[0]))

    print('OBS SPACE', env1.observation_space)


def test_should_obs_info_can_be_inserted_to_buffer(mocker: MockerFixture):

    # given
    agent_pool_size = 10
    instances = _create_env(mocker,
                            name='env',
                            pool_size=agent_pool_size,
                            pool_offset=0)
    env = instances.env

    # when

    buffer_batch = Batch()

    episode = 0
    buffer = ReplayBuffer(size=100000)

    while episode < 1:
        obs, info = env.reset()
        print('env', env.possible_agents)

        step = 0
        done = False
        while not done:

            # print('AGENTS', env.agents)
            actions = {
                agent: 0 if agent in env.agents else np.nan
                for agent in env.possible_agents
            }
            # print('ACTIONS', actions)

            obs_next, rew, terminated, truncated, info_next = env.step(actions)
            # print(info_next['teacher0']['core_position_earth_m_xy'], info_next['teacher0']['core_position_earth_m_xy'].shape,
            #       type(info_next['teacher0']['core_position_earth_m_xy']))
            # del info_next['teacher0']['core_position_earth_m_xy']
            # print(info_next['teacher0']['air_velocity_earth_m_xy'], type(info_next['teacher0']['core_position_earth_m_xy']))
            # break
            # info_next = {}

            if 'student0' in obs_next:
                print('OBS', obs_next['student0'], rew)

            batch = Batch(obs=obs,
                          act=actions,
                          rew=rew,
                          terminated=terminated,
                          truncated=truncated,
                          obs_next=obs_next,
                          info=info_next)
            # print('NEWBATCH', batch)
            buffer.add(cast(RolloutBatchProtocol, batch))

            done = terminated or truncated
            obs = obs_next
            info = info_next

            # if step > 0:
            #     # print('prev_batch', buffer_batch.obs)
            #     print('prev_batch', buffer_batch.info['teacher0']['core_position_earth_m_xy'])
            #     print('new_batch', batch.info['teacher0']['core_position_earth_m_xy'])

            # buffer_batch = Batch.stack((buffer_batch, batch), axis=0)

            # if step > 3:
            #     break
            step += 1
            # print('buffer_batch', buffer_batch)
        episode += 1

    # print('buff', buffer)


def test_sandbox_parallel(mocker: MockerFixture):

    agent_pool_size = 10
    instances1 = _create_env(mocker,
                             name='env0',
                             pool_size=agent_pool_size,
                             pool_offset=0)
    env0 = instances1.env

    instances2 = _create_env(mocker,
                             name='env1',
                             pool_size=agent_pool_size,
                             pool_offset=0)
    env1 = instances2.env

    instances3 = _create_env(mocker,
                             name='test_env',
                             pool_size=agent_pool_size,
                             pool_offset=0)
    env2 = instances3.env

    observation_space = env0.observation_space
    action_space = env0.action_space
    possible_agents = instances1.env.possible_agents
    policy = _create_multi_level_transformer_policy(
        observation_space=observation_space[possible_agents[0]],
        action_space=action_space[possible_agents[0]],
        device='cpu')

    print('possible agents', possible_agents)
    print('action space', env0.action_space)
    print('obs space', env0.observation_space)

    # obs, info = env1.reset()

    # print('obs', obs)

    # obs, reward, terminated, truncated, info = env1.step({'glider_0': 0.})
    # print(obs, reward,terminated, truncated, info)

    # print(env.observation_space(env.possible_agents[0]))
    # print(env.action_space(env.possible_agents[0]))

    assert isinstance(action_space, gymnasium.spaces.Dict)
    # random_policy = RandomPolicy(action_space=action_space)

    # # obs, info = env.reset()
    # # print('action spaces', env.action_spaces)

    env_num = 2
    envs = [lambda: env0, lambda: env1]
    #venv = tianshou.env.DummyVectorEnv([envs[i] for i in range(env_num)])
    train_venv = tianshou.env.SubprocVectorEnv(
        [envs[i] for i in range(env_num)])

    test_envs = [lambda: env2]
    test_venvs = tianshou.env.SubprocVectorEnv(
        [test_envs[i] for i in range(1)])

    agent_policies: dict[AgentID, BasePolicy] = {
        agent_id: RandomPolicy(action_space=action_space[agent_id])
        for agent_id in possible_agents
    }
    agent_policies['glider_0'] = policy

    # print('agent policies', agent_policies)

    multi_agent_policy = ParallelMultiAgentPolicyManager(
        policies=agent_policies, learn_agent_ids=[AgentID('glider_0')])

    TRAINER = False
    if TRAINER:
        single_buffer_size = 10000
        replay_buffer = VectorReplayBuffer(total_size=single_buffer_size *
                                           env_num,
                                           buffer_num=env_num)

        test_collector = tianshou.data.Collector(multi_agent_policy,
                                                 test_venvs)
        train_collector = tianshou.data.Collector(multi_agent_policy,
                                                  train_venv,
                                                  buffer=replay_buffer)

        trainer = tianshou.trainer.OnpolicyTrainer(
            policy=multi_agent_policy,
            train_collector=train_collector,
            test_collector=test_collector,
            max_epoch=1,
            step_per_epoch=10,
            repeat_per_collect=1,
            episode_per_test=1,
            step_per_collect=20,
            batch_size=512,
        )

        for epoch_stats in trainer:

            print('EPOCH STAT', epoch_stats)
    else:
        single_buffer_size = 10000
        replay_buffer = VectorReplayBuffer(total_size=single_buffer_size *
                                           env_num,
                                           buffer_num=env_num)

        collector = tianshou.data.Collector(multi_agent_policy,
                                            train_venv,
                                            buffer=replay_buffer)

        collect_stats = collector.collect(n_episode=2,
                                          reset_before_collect=True)

        print(collect_stats)
        # print('obs:', replay_buffer.obs)
        #print('act:', replay_buffer.act)
        # print('obs_next:', replay_buffer.obs_next)
        #print('rew:', replay_buffer.rew)

        # try to learn
        if True:
            with policy_within_training_step(multi_agent_policy):
                multi_agent_policy.update(sample_size=None,
                                          buffer=collector.buffer,
                                          batch_size=4,
                                          repeat=1)

    # obs, info = venv.reset()

    # print('obs', obs)

    # step_result = venv.step([{ 'glider_0': 0.}, {'glider_0': 1. } ])

    # print('RES', step_result)

    # # pass


# ObsType = TypeVar('ObsType')
# ActType = TypeVar('ActType')
# AgentIDType = TypeVar('AgentIDType')

# class ParallelPettingZooEnv2(Generic[AgentIDType, ObsType, ActType],
#                              gymnasium.Env[dict[AgentIDType, ObsType],
#                                            dict[AgentIDType, ActType]]):

#     _name: str
#     _env: ParallelEnv[AgentIDType, ObsType, ActType]

#     _agent_id_to_index_map: dict[AgentIDType, int]
#     _index_to_agent_id_map: list[AgentIDType]

#     def __init__(self, name: str, env: ParallelEnv[AgentIDType, ObsType,
#                                                    ActType]):

#         super().__init__()

#         self._log = logging.getLogger(__class__.__name__)

#         self._name = name

#         self.observation_space = self._create_observation_space(env)
#         self.action_space = self._create_action_space(env)

#         self._agent_id_to_index_map = self._create_agent_id_index_mapping(env)
#         self._index_to_agent_id_map = env.possible_agents.copy()

#         self._env = env

#     @property
#     def agent_id_to_index_map(self):
#         return self._agent_id_to_index_map

#     @property
#     def agents(self):
#         return self._env.agents

#     def _create_agent_id_index_mapping(self,
#                                        env: ParallelEnv[AgentIDType, ObsType,
#                                                         ActType]):

#         agent_id_to_index: dict[AgentIDType, int] = {}
#         for idx, agent_id in enumerate(env.possible_agents):
#             agent_id_to_index[agent_id] = idx

#         return agent_id_to_index

#     def _create_observation_space(self, env: ParallelEnv[AgentIDType, ObsType,
#                                                          ActType]):
#         first_possible_agent_id = env.possible_agents[0]
#         observation_space = env.observation_space(first_possible_agent_id)
#         assert all(env.observation_space(agent) == observation_space
#                    for agent in env.possible_agents), \
#                            "Observation spaces for all agents must be identical."

#         return observation_space

#     def _create_action_space(self, env: ParallelEnv[AgentIDType, ObsType,
#                                                     ActType]):
#         first_possible_agent_id = env.possible_agents[0]
#         action_space = env.action_space(first_possible_agent_id)
#         assert all(env.action_space(agent) == action_space
#                    for agent in env.possible_agents), \
#                            "Action spaces for all agents must be identical."

#         return action_space

#     def reset(
#         self,
#         *args,
#         seed: int | None = None,
#         options: dict[str, Any] | None = None,
#     ) -> tuple[dict[AgentIDType, ObsType], dict[Any, Any]]:

#         print('pettingzoo reset', self._name)
#         obs, info = self._env.reset(*args, seed=seed, options=options)

#         return obs, info

#     def _convert_action_list_to_dict(self, action_list: VectorN,
#                                      current_agents: list[AgentIDType]):
#         action_dict = {}
#         for agent_id in current_agents:
#             agent_index = self._agent_id_to_index_map[agent_id]
#             action_dict[agent_id] = action_list[agent_index]

#         return action_dict

#     def step(
#         self, action: dict[AgentIDType, ActType]
#     ) -> tuple[ObsType, list[SupportsFloat], bool, bool, dict[Any, Any]]:

#         print('PETTINgZOO INPUT ACTIONS', self._name, action, action.shape)

#         action_dict = self._convert_action_list_to_dict(
#             action, self._env.agents)
#         print('ACTION_DICT', action_dict)

#         agents_before_step = self._env.agents.copy()

#         multi_obs, multi_reward, multi_termination, multi_truncation, multi_info = self._env.step(
#             action_dict)

#         possible_agent_count = len(self._env.possible_agents)

#         processed_rewards = self._process_rewards(multi_reward)
#         # reward_list = [1.]

#         #processed_info = deepcopy(multi_info)
#         processed_info = {}
#         print('MULTI_TERMINATION', multi_termination)
#         print('MULTI_TRUNCATION', multi_truncation)
#         for agent_id in agents_before_step:
#             if not agent_id in processed_info:
#                 processed_info[agent_id] = {}

#             processed_info[agent_id]['terminated'] = multi_termination[
#                 agent_id]
#             processed_info[agent_id]['truncated'] = multi_truncation[agent_id]

#         agents_after_step = self._env.agents.copy()

#         agent_mask = self._create_agent_mask(agents_after_step)
#         for agent_id, mask in agent_mask.items():
#             if not agent_id in processed_info:
#                 processed_info[agent_id] = {}
#             processed_info[agent_id]['agent_mask'] = mask

#         terminated = False
#         truncated = False
#         if len(agents_after_step) == 0:
#             # no more agents, terminate
#             terminated = True

#         print('processed_info', processed_info)
#         return multi_obs, processed_rewards, terminated, truncated, processed_info
#         raise Exception('alma')
#         pass

#     def _create_agent_mask(self, agents: list[AgentIDType]):
#         mask = {
#             agent_id: (agent_id in agents)
#             for agent_id in self._env.possible_agents
#         }
#         return mask

#     def _process_rewards(self, rewards: dict[AgentIDType, float]):
#         """Create an array which represents the rewards for each possible agents.
#         Fill the values from the current agent rewards based on possible agent indices."""

#         result = np.zeros(len(self._agent_id_to_index_map))
#         for agent_id, reward in rewards.items():
#             result[self._agent_id_to_index_map[agent_id]] = reward

#         return result

# multi_level_transformer_config_yaml = """
# defaults:
#   - model/optimizer: adam
# transformer_net:
#   - input_dim: 7
#     output_dim: 64
#     attention_internal_dim: 64 # it is very important to be "large", this is the main communication (not 16)
#     attention_head_num: 4
#     ffnn_hidden_dim: 128
#     ffnn_dropout_rate: 0.0
#     max_sequence_length: ${train_env.env.params.max_sequence_length}
#     embedding_dim: 64
#     encoder_layer_count: 2
#     enable_layer_normalization: false
#     enable_causal_attention_mask: true
#     is_reversed_sequence: true
# actor_critic_net:
#   action_space_n: 13
#   actor_hidden_sizes: [128, 128]
#   critic_hidden_sizes: [128, 128]
#   net_output_dim: ${model.transformer_net.output_dim}
# optimizer:
#   lr: 0.0003 # decreased learning rate to avoid overshooting for larger model (I think)
# ppo_policy:
#   discount_factor: 0.99
#   deterministic_eval: true
#   eps_clip: 0.2
#   vf_coef: 0.5
#   recompute_advantage: true
#   value_clip: false
#   dual_clip: 5.0
#   ent_coef: 0.02
#   gae_lambda: 0.9
#   advantage_normalization: true
#   max_batchsize: 2048
# """


@dataclass(kw_only=True)
class JobParameters():
    device: str
    model: TianshouModelConfigBase


def _create_multi_level_transformer_policy(
        observation_space: gymnasium.spaces.Space,
        action_space: gymnasium.spaces.Space, device: torch.device):

    level1_params = _create_transformer_net_params(input_dim=7,
                                                   encoder_layer_count=1,
                                                   attention_head_num=1,
                                                   attention_internal_dim=2,
                                                   output_dim=3)
    transformer_net1 = partial(_create_transformer_net, level1_params)

    level2_params = _create_transformer_net_params(input_dim=3,
                                                   encoder_layer_count=1,
                                                   attention_head_num=1,
                                                   attention_internal_dim=2,
                                                   output_dim=5)
    transformer_net2 = partial(_create_transformer_net, level2_params)

    multi_level_transformer_net = partial(_create_multi_level_transformer_net,
                                          [transformer_net1, transformer_net2],
                                          device=device)

    actor_critic_net = partial(
        create_discrete_actor_critic_net_with_common_encoder_net,
        action_space_n=13,
        actor_hidden_sizes=[128, 128],
        critic_hidden_sizes=[128, 128],
        net_output_dim=5)

    ppo_policy = partial(tianshou.policy.PPOPolicy,
                         discount_factor=0.99,
                         deterministic_eval=True,
                         eps_clip=0.2,
                         vf_coef=0.5,
                         recompute_advantage=True,
                         value_clip=False,
                         dual_clip=5.0,
                         ent_coef=0.02,
                         gae_lambda=0.9,
                         advantage_normalization=True,
                         max_batchsize=2048,
                         dist_fn=torch.distributions.Categorical,
                         action_scaling=False)
    optimizer = partial(torch.optim.Adam, lr=0.0003)

    model = create_ppo_custom_transformer_model(
        transformer_net=multi_level_transformer_net,
        actor_critic_net=actor_critic_net,
        ppo_policy=ppo_policy,
        optimizer=optimizer,
        lr_scheduler=None,
        device=device,
        observation_space=observation_space,
        action_space=action_space,
        deterministic_eval=True)

    return model

    print('MODEL', model)

    # cs = ConfigStore.instance()
    # register_ppo_transformer_model_config_groups(base_group='model',
    #                                              config_store=cs)

    # params = create_params_from_yaml_string(
    #     multi_level_transformer_config_yaml,
    #     node_type=PPOMultiLevelTransformerModelConfig)

    # print('PARAMS', params)


def _create_transformer_net_params(input_dim: int,
                                   encoder_layer_count=1,
                                   attention_head_num=1,
                                   attention_internal_dim=2,
                                   output_dim=3,
                                   pad_value=0.):
    params = TransformerNetParameters(
        input_dim=input_dim,
        output_dim=output_dim,
        attention_internal_dim=attention_internal_dim,
        attention_head_num=attention_head_num,
        ffnn_hidden_dim=4,
        ffnn_dropout_rate=0.0,
        max_sequence_length=100,
        embedding_dim=4,
        encoder_layer_count=encoder_layer_count,
        enable_layer_normalization=False,
        enable_causal_attention_mask=True,
        is_reversed_sequence=True,
        softmax_output=False,
        pad_value=pad_value)
    return params


def _create_multi_level_transformer_net(transformer_factories: list[partial],
                                        device: torch.device):

    instances = [factory(device=device) for factory in transformer_factories]

    return MultiLevelTransformerNet(instances, device)


def _create_transformer_net(params: TransformerNetParameters,
                            device: torch.device):

    transformer_net = TransformerNet(**asdict(params), device=device)

    return transformer_net


@dataclass
class Instances:
    env: ParallelPettingZooEnv
    aerodynamics: SimpleAerodynamics
    air_velocity_field: ZeroAirVelocityField


def _create_env(mocker: MockerFixture,
                name: str,
                pool_size,
                pool_offset: int,
                sequence_length=2,
                max_closest_agent_count=3) -> Instances:

    aerodynamics_params = SimpleAerodynamicsParameters()
    aerodynamics = SimpleAerodynamics(**asdict(aerodynamics_params))
    air_velocity_field = ZeroAirVelocityField()
    initial_conditions_params = GliderInitialConditionsParameters()

    # initial_conditions_calculator = GliderInitialConditionsCalculator(
    #     initial_conditions_params=GliderInitialConditionsParameters(),
    #     aerodynamics=aerodynamics,
    #     air_velocity_field=air_velocity_field)

    simulation_box_params = SimulationBoxParameters(box_size=(5000, 5000,
                                                              2000))
    time_params = TimeParameters(dt_s=0.5, decision_dt_s=1.0)

    render_params = RenderParameters(mode='rgb_array')

    agent_params = GliderAgentParameters()

    # spawn_params = AgentSpawnParameters(parallel_num_min_max=(2, 2),
    #                                     time_between_spawns_min_max_s=(1, 1))
    # agent_spawner = AgentSpawner(spawn_params)

    agent_group_params = [
        AgentGroupParams(terminate_if_finished=True,
                         spawner=AgentSpawnParameters2(
                             prefix='student',
                             pool_size=1,
                             initial_time_offset_s=5.,
                             parallel_num_min_max=(1, 1),
                             time_between_spawns_min_max_s=(1, 1),
                             must_spawn_if_no_global_agent=False)),
        AgentGroupParams(
            terminate_if_finished=False,
            spawner=AgentSpawnParameters2(prefix='teacher',
                                          pool_size=10,
                                          initial_time_offset_s=0.,
                                          parallel_num_min_max=(3, 8),
                                          time_between_spawns_min_max_s=(1, 5),
                                          must_spawn_if_no_global_agent=True),
        )
    ]

    # visualization = mocker.patch(
    #     'env.glider.base.visualization.MultigliderVisualization').return_value

    cutoff_params = GliderCutoffParameters()
    reward_params = GliderRewardParameters()

    params = MultiGliderEnvParameters(
        simulation_box_params=simulation_box_params,
        initial_conditions_params=initial_conditions_params,
        time_params=time_params,
        cutoff_params=cutoff_params,
        reward_params=reward_params,
        glider_agent_params=agent_params,
        max_closest_agent_count=max_closest_agent_count,
        agent_groups=agent_group_params,
        render_params=render_params,
        max_sequence_length=sequence_length,
        discrete_continuous_mapping=DiscreteToContinuousDegreesParams())

    env = make_multiglider_env(env_name=name,
                               params=params,
                               air_velocity_field=air_velocity_field,
                               aerodynamics=aerodynamics)

    # env = MultiGliderEnvBase(
    #     glider_pool_size=pool_size,
    #     glider_pool_offset=pool_offset,
    #     aerodynamics=aerodynamics,
    #     air_velocity_field=air_velocity_field,
    #     glider_initial_conditions_calculator=initial_conditions_calculator,
    #     glider_agent_params=agent_params,
    #     simulation_box_params=simulation_box_params,
    #     time_params=time_params,
    #     cutoff_params=GliderCutoffParameters(),
    #     reward_params=GliderRewardParameters(),
    #     glider_spawner=agent_spawner,
    #     render_params=render_params,
    #     visualization=visualization)

    # possible_agents = env.possible_agents

    # # frame_skip
    # env = apply_frame_skip_wrapper(
    #     env,
    #     time_params=time_params,
    #     default_action=agent_params.default_action)

    # # observation wrapper
    # env = apply_observation_wrapper(env)
    # assert isinstance(env, ParallelEnv)

    # env = apply_share_wrappers(env,
    #                            max_sequence_length=sequence_length,
    #                            max_closest_agent_count=max_closest_agent_count)

    # sequence window
    # env = sequence_window_obs_wrapper(
    #     env,
    #     params=SequenceWindowObsParams(max_sequence_length=sequence_length))

    # # nan pad
    # env = pad_sequence_obs_wrapper(env,
    #                                params=PadSequenceObsParams(
    #                                    max_sequence_length=sequence_length,
    #                                    pad_at='end',
    #                                    value=np.nan))

    # assert isinstance(env, ParallelEnv)
    # env = ParallelPettingZooEnv[AgentID, GliderAgentObsType,
    #                             GliderAgentActType](name=name, env=env)

    return Instances(env, aerodynamics, air_velocity_field)
