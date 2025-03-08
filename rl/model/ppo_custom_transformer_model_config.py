from dataclasses import dataclass
import functools
import torch
import gymnasium
from torch.types import Device
from model.actor_critic.discrete_actor_critic import DiscreteActorCriticNetworkWithCommonEncoderNetworkParameters
from model.transformer.transformer_net import TransformerNetParameters
from model.transformer.multi_level_transformer_net import TwoLevelTransformerNet
from trainer.common import TianshouModelConfigBase
from hydra.core.config_store import ConfigStore


@dataclass
class LRSchedulerConfig:
    _target_: str = 'torch.optim.lr_scheduler.LRScheduler'
    _partial_: bool = True


@dataclass(kw_only=True)
class ConstantLRSchedulerConfig(LRSchedulerConfig):
    factor: float
    total_iters: int
    _target_: str = 'torch.optim.lr_scheduler.ConstantLR'


@dataclass
class OptimizerConfig:
    _target_: str = 'torch.optim.Optimizer'
    _partial_: bool = True


@dataclass(kw_only=True)
class AdamOptimizerConfig(OptimizerConfig):
    lr: float
    _target_: str = 'torch.optim.Adam'


def resolve_distribution_type():
    return torch.distributions.Distribution


def resolve_categorical_distribution_type():
    return torch.distributions.Categorical


@dataclass
class DistributionConfig:
    _target_: str = __name__ + '.resolve_distribution_type'


@dataclass
class PPOPolicyParameters:
    discount_factor: float
    deterministic_eval: bool
    eps_clip: float
    vf_coef: float
    recompute_advantage: bool
    value_clip: bool
    dual_clip: float | None
    ent_coef: float
    gae_lambda: float
    advantage_normalization: bool
    dist_fn: DistributionConfig
    max_batchsize: int  # the maximum size of the batch when computing GAE
    action_scaling: bool = False  # can only be true if continuous


@dataclass
class CategoricalDistributionConfig(DistributionConfig):
    _target_: str = __name__ + '.resolve_categorical_distribution_type'


def register_ppo_transformer_model_config_groups(base_group: str,
                                                 config_store: ConfigStore):
    config_store.store(group=f'{base_group}/ppo_policy/dist_fn',
                       name='categorical',
                       node=CategoricalDistributionConfig)
    config_store.store(group=f'{base_group}/optimizer',
                       name='adam',
                       node=AdamOptimizerConfig)
    config_store.store(group=f'{base_group}/lr_scheduler',
                       name='constant',
                       node=ConstantLRSchedulerConfig)
    config_store.store(group=f'{base_group}',
                       name='ppo_custom_transformer',
                       node=PPOCustomTransformerModelConfig)
    config_store.store(group=f'{base_group}',
                       name='ppo_multi_level_transformer',
                       node=PPOMultiLevelTransformerModelConfig)
    config_store.store(group=f'{base_group}',
                       name='single_level_transformer',
                       node=TransformerNetConfig)
    config_store.store(group=f'{base_group}',
                       name='two_level_transformer',
                       node=TwoLevelTransformerNetConfig)


@dataclass
class DiscreteActorCriticNetworkWithCommonEncoderNetworkConfig(
        DiscreteActorCriticNetworkWithCommonEncoderNetworkParameters):
    _target_: str = 'model.actor_critic.discrete_actor_critic.create_discrete_actor_critic_net_with_common_encoder_net'
    _partial_: bool = True


@dataclass
class PPOPolicyConfig(PPOPolicyParameters):
    _target_: str = 'tianshou.policy.PPOPolicy'
    _partial_: bool = True


@dataclass
class ModuleConfigBase:
    _target_: str = 'torch.nn.Module'
    _partial_: bool = True


@dataclass(kw_only=True)
class TransformerNetConfig(ModuleConfigBase, TransformerNetParameters):
    _target_: str = 'model.transformer.transformer_net.TransformerNet'
    _partial_: bool = True


@dataclass(kw_only=True)
class TwoLevelTransformerNetConfig(ModuleConfigBase):

    ego_sequence_transformer: TransformerNetConfig
    peer_sequence_transformer: TransformerNetConfig
    agent_transformer: TransformerNetConfig

    _target_: str = 'model.ppo_custom_transformer_model_config.create_two_level_transformer_net'
    _partial_: bool = True


def create_two_level_transformer_net(
        ego_sequence_transformer: functools.partial,
        peer_sequence_transformer: functools.partial,
        agent_transformer: functools.partial, device: torch.device):

    two_level_transformer = TwoLevelTransformerNet(
        ego_sequence_transformer=ego_sequence_transformer(device=device),
        peer_sequence_transformer=peer_sequence_transformer(device=device),
        item_transformer=agent_transformer(device=device),
        device=device)

    return two_level_transformer


@dataclass(kw_only=True)
class PPOCustomTransformerModelConfig(TianshouModelConfigBase):
    transformer_net: TransformerNetConfig
    actor_critic_net: DiscreteActorCriticNetworkWithCommonEncoderNetworkConfig
    ppo_policy: PPOPolicyConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig | None = None
    _target_: str = 'model.ppo_custom_transformer_model_config.create_ppo_custom_transformer_model'
    _partial_: bool = True


@dataclass(kw_only=True)
class PPOMultiLevelTransformerModelConfig(TianshouModelConfigBase):
    transformer_net: ModuleConfigBase
    actor_critic_net: DiscreteActorCriticNetworkWithCommonEncoderNetworkConfig
    ppo_policy: PPOPolicyConfig
    optimizer: OptimizerConfig
    lr_scheduler: LRSchedulerConfig | None = None
    _target_: str = 'model.ppo_custom_transformer_model_config.create_ppo_custom_transformer_model'
    _partial_: bool = True


def create_ppo_custom_transformer_model(transformer_net: functools.partial,
                                        actor_critic_net: functools.partial,
                                        ppo_policy: functools.partial,
                                        optimizer: functools.partial,
                                        lr_scheduler: functools.partial | None,
                                        device: Device,
                                        observation_space: gymnasium.Space,
                                        action_space: gymnasium.Space,
                                        deterministic_eval: bool
                                        | None = None):
    net = transformer_net(device=device)
    actor_critic_net = actor_critic_net(net=net, device=device)

    # create the optimizer
    optimizer = optimizer(params=actor_critic_net.actor_critic.parameters())

    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer=optimizer)

    # ok, create the policy
    if deterministic_eval is not None:
        # override the parameter
        ppo_policy = functools.partial(ppo_policy,
                                       deterministic_eval=deterministic_eval)

    policy = ppo_policy(actor=actor_critic_net.actor,
                        critic=actor_critic_net.critic,
                        optim=optimizer,
                        lr_scheduler=lr_scheduler,
                        observation_space=observation_space,
                        action_space=action_space)

    return policy
