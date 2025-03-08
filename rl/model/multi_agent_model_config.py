import logging
from functools import partial
from dataclasses import dataclass
from trainer.multi_agent.parallel_multi_agent_policy_manager import ParallelMultiAgentPolicyManager
from omegaconf import MISSING
import hydra
import torch
import tianshou
import gymnasium
from hydra.core.config_store import ConfigStore
from trainer.experiment_logger import ExperimentLoggerConfigBase
from trainer.common import (ExperimentLoggerParameterStore,
                            ExperimentLoggerWeightStore,
                            TianshouModelConfigBase)
from utils import find_suitable_torch_device


@dataclass
class NonLearningModelConfig(TianshouModelConfigBase):

    device: str = 'cpu'
    model_source_logger: ExperimentLoggerConfigBase = MISSING
    _target_: str = 'model.multi_agent_model_config.load_non_learning_model'


# TODO: refactor
def _create_policy_model(
    device: torch.device, policy_config: TianshouModelConfigBase
) -> partial[tianshou.policy.BasePolicy]:

    policy = hydra.utils.instantiate(policy_config, _convert_='object')
    policy = partial(policy, device=device)

    return policy


# TODO: copied from tianshou job base, refactor
def load_non_learning_model(device: str,
                            model_source_logger: ExperimentLoggerConfigBase,
                            observation_space: gymnasium.Space,
                            action_space: gymnasium.Space,
                            deterministic_eval: bool):

    _log = logging.getLogger('load_non_learning_model')

    model_logger_instance = model_source_logger

    _log.debug('loading parameters from the experiment logger...')
    parameter_store = ExperimentLoggerParameterStore(
        experiment_logger=model_logger_instance)
    parameters = parameter_store.load_parameters()

    model_parameters = parameters['model']
    _log.debug('parameters loaded; parameters.model=%s', model_parameters)

    _log.debug('creating model based on the parameters...')

    # create the policy
    found_device = find_suitable_torch_device(device)
    policy = _create_policy_model(device=found_device,
                                  policy_config=model_parameters)(
                                      observation_space=observation_space,
                                      action_space=action_space,
                                      deterministic_eval=deterministic_eval)

    # load the best weights
    _log.debug('loading best weights...')
    weights_store = ExperimentLoggerWeightStore(
        experiment_logger=model_logger_instance)
    weights_state_dict = weights_store.load_best_weights(map_device=device)

    _log.debug('weights loaded; running the evaluators...')

    policy.load_state_dict(weights_state_dict)

    # create the model based on parameters from the logger
    _log.debug('model loaded')

    # no learning for this agent
    policy.eval()

    return policy


@dataclass(kw_only=True)
class MultiAgentModelConfig(TianshouModelConfigBase):

    non_learning: NonLearningModelConfig
    learning: TianshouModelConfigBase
    learning_agent_ids: list[str]
    agent_observation_preprocessors: dict[str, str]
    deterministic_eval: bool | None = None
    _target_: str = 'model.multi_agent_model_config.create_multi_agent_model'


def create_multi_agent_model(non_learning: partial[tianshou.policy.BasePolicy],
                             learning: partial[tianshou.policy.BasePolicy],
                             learning_agent_ids: list[str],
                             agent_observation_preprocessors: dict[str, str],
                             device: torch.device,
                             observation_space: gymnasium.Space,
                             action_space: gymnasium.Space,
                             deterministic_eval: bool | None = None):

    single_action_space = next(iter(action_space.values()))

    learning_policy = learning(device=device,
                               observation_space=observation_space,
                               action_space=single_action_space,
                               deterministic_eval=deterministic_eval)

    non_learning_policy = non_learning(observation_space=observation_space,
                                       action_space=single_action_space,
                                       deterministic_eval=deterministic_eval)

    possible_agents = list(observation_space.keys())

    policies = {}
    for agent_id in possible_agents:

        if agent_id in learning_agent_ids:
            policy = learning_policy
        else:
            policy = non_learning_policy

        policies[agent_id] = policy

    # create the multi agent policy manager
    multi_agent_policy = ParallelMultiAgentPolicyManager(
        policies=policies,
        learn_agent_ids=learning_agent_ids,
        agent_observation_preprocessors=agent_observation_preprocessors)

    return multi_agent_policy


def register_multi_agent_model_config_groups(base_group: str,
                                             config_store: ConfigStore):
    config_store.store(group=f'{base_group}',
                       name='multi_agent',
                       node=MultiAgentModelConfig)
