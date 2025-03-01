from typing import Any, Callable, TypeVar, Generic
from dataclasses import dataclass
import logging
from omegaconf import MISSING
import gymnasium
import hydra
import tianshou


@dataclass
class VectorizedEnvironmentConfigBase:
    _target_: str = 'tianshou.env.BaseVectorEnv'
    _partial_: bool = True


@dataclass
class SubprocVectorEnvConfig(VectorizedEnvironmentConfigBase):
    _target_: str = 'tianshou.env.SubprocVectorEnv'


@dataclass
class DummyVectorEnvConfig(VectorizedEnvironmentConfigBase):
    _target_: str = 'tianshou.env.DummyVectorEnv'


@dataclass
class EnvConfigBase:
    _target_: str


AgentIDType = TypeVar('AgentIDType', bound=str)


@dataclass
class EnvSpaces(Generic[AgentIDType]):
    observation_space: gymnasium.Space
    action_space: gymnasium.Space
    is_multi_agent: bool
    possible_agents: list[AgentIDType]


@dataclass
class TianshouVectorizedEnvironmentParameters:
    # the number of the enviroments in the vectorized environment
    count: int
    # if true tianshou.env.SubprocVectorEnv will be used; else tianshou.env.DummyVectorEnv
    parallel: bool | None
    # instead of using parallel, you can set the vectorized environment config here
    vector_env_config: VectorizedEnvironmentConfigBase | None = None

    def __post_init__(self):

        if self.vector_env_config is None:
            assert self.parallel is not None, \
                "either 'parallel' or 'vector_env_config' should be defined"
            self.vector_env_config = SubprocVectorEnvConfig() \
                if self.parallel else DummyVectorEnvConfig()


@dataclass
class TianshouCollectorParameters:
    exploration_noise: bool
    # None means, it do not need to store the history for testing
    buffer_size: int | None = None


@dataclass
class TianshouEnviromentParameters:
    vectorized: TianshouVectorizedEnvironmentParameters
    collector: TianshouCollectorParameters
    env: Any = MISSING


class TianshouVectorizedCollectorFactory:

    _log: logging.Logger
    _env_wrapper: Callable[[gymnasium.Env, int], gymnasium.Env] | None

    def __init__(
        self,
        # wrap environment at construction using the provided function (env,index)->env
        env_wrapper_func: Callable[[gymnasium.Env, int], gymnasium.Env]
        | None = None):
        self._log = logging.getLogger(__class__.__name__)
        self._env_wrapper_func = env_wrapper_func

    def create_collector(
            self, collector_params: TianshouCollectorParameters,
            vectorized_env: tianshou.env.BaseVectorEnv,
            policy: tianshou.policy.BasePolicy) -> tianshou.data.Collector:

        # create buffer if necessary
        buffer = None
        if collector_params.buffer_size is not None:
            env_count = vectorized_env.env_num
            buffer = tianshou.data.VectorReplayBuffer(
                total_size=collector_params.buffer_size, buffer_num=env_count)

            self._log.debug('buffer has been created with total_size %s',
                            collector_params.buffer_size)

        collector = tianshou.data.Collector(
            policy=policy,
            env=vectorized_env,
            buffer=buffer,
            exploration_noise=collector_params.exploration_noise)

        return collector

    def create_vectorized_environment(
            self, vectorized_params: TianshouVectorizedEnvironmentParameters,
            env_config: EnvConfigBase,
            seed: int | None) -> tuple[tianshou.env.BaseVectorEnv, EnvSpaces]:

        count = vectorized_params.count
        vector_env_config = vectorized_params.vector_env_config

        self._log.debug('creating %d environments using %s', count,
                        vector_env_config)

        vector_env_partial = hydra.utils.instantiate(vector_env_config,
                                                     _convert_='object')

        env_factories = [
            lambda i=i: self._create_env(env_config=env_config,
                                         vectorized_index=i)
            for i in range(count)
        ]

        vector_env = vector_env_partial(env_factories)

        self._log.debug('%d environments created', count)

        assert vector_env.env_num > 0, 'minimum one environment is required'
        observation_space = vector_env.get_env_attr('observation_space',
                                                    id=0)[0]
        action_space = vector_env.get_env_attr('action_space', id=0)[0]

        is_multi_agent = False
        possible_agents = []
        possible_agent_attribute: list | None = vector_env.get_env_attr(
            'possible_agents', id=0)[0]
        if possible_agent_attribute is not None:
            # it is a multi agent environment
            possible_agents = possible_agent_attribute
            is_multi_agent = True

        spaces = EnvSpaces(observation_space=observation_space,
                           action_space=action_space,
                           is_multi_agent=is_multi_agent,
                           possible_agents=possible_agents)

        self._log.debug('observation_space=%s,action_space=%s',
                        observation_space, action_space)

        if seed is not None:
            self._seed_vectorized_environment(vectorized_env=vector_env,
                                              seed=seed)
            self._log.debug('seed initialized')

        return vector_env, spaces

    def _seed_vectorized_environment(
            self, vectorized_env: tianshou.env.BaseVectorEnv,
            seed: int | None) -> None:
        # seed the vectorized environment, the parallel envs will use incremented seed
        env_num = vectorized_env.env_num
        self._log.debug(
            'setting the seed for the vectorized environment; seed=%s,env_num=%s',
            seed, env_num)

        seed_list = [seed + i for i in range(env_num)
                     ] if seed is not None else [None] * env_num
        self._log.debug('seed list: %s', seed_list)

        for i in range(env_num):
            vectorized_env.reset(i, seed=seed_list[i])

    def _create_env(self, env_config: EnvConfigBase, vectorized_index: int):
        self._log.debug('instantiating environment: %s; vectorized_index=%s',
                        env_config._target_, vectorized_index)

        env = hydra.utils.instantiate(env_config, _convert_='object')
        if self._env_wrapper_func:
            env = self._env_wrapper_func(env, vectorized_index)
        return env
