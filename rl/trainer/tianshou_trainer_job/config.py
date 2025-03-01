from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from .api import (TianshouTrainerConfigBase)


def resolve_mean_reward_stop_function(stop_mean_reward: float):

    def stop_fn(mean_reward: float) -> bool:
        return mean_reward >= stop_mean_reward

    return stop_fn


@dataclass
class Instantiatable():
    _target_: str


@dataclass(kw_only=True)
class MeanRewardStopFunctionConfig(Instantiatable):
    stop_mean_reward: float
    _target_: str = __name__ + '.resolve_mean_reward_stop_function'


# https://tianshou.org/en/stable/api/tianshou.trainer.html#tianshou.trainer.OnpolicyTrainer
@dataclass(kw_only=True)
class OnPolicyTianshouTrainerConfig(TianshouTrainerConfigBase):
    # the maximum number of epochs (if stop_fn does not stop earlier)
    max_epoch: int

    # the number of transition collected per epoch, after every epoch, reset the buffer
    step_per_epoch: int

    # after 'step_per_collect' steps, do an update of the policy (learn),
    # onpolicy will reset the buffer after every collect too
    step_per_collect: int | None = None

    # the number of steps that is feeded for policy update
    batch_size: int

    # update the policy with collected steps 'repeat_per_collect' times (with randomized minibatches)
    repeat_per_collect: int

    # the number of episodes for one policy evaluation
    episode_per_test: int

    # mean_test_rewards -> bool function (average undiscounted returns of the test)
    stop_fn: Instantiatable | None = None

    # instead of 'step_per_collect', 'episode_per_collect' can be used to
    # determine the required steps
    episode_per_collect: int | None = None

    _target_: str = 'tianshou.trainer.OnpolicyTrainer'


def register_tianshou_trainer_config_groups(base_group: str,
                                            config_store: ConfigStore):
    config_store.store(group=f'{base_group}',
                       name='onpolicy',
                       node=OnPolicyTianshouTrainerConfig)
    config_store.store(group=f'{base_group}/stop_fn',
                       name='mean_reward_stop',
                       node=MeanRewardStopFunctionConfig)
