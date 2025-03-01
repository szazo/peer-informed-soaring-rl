from typing import Sequence, Iterator
from functools import partial, reduce
from dataclasses import dataclass
import numpy as np
from tianshou.data import Batch

from utils.vector import VectorN, VectorNx3


@dataclass
class Obs:
    obs: VectorNx3
    info: dict
    terminated: bool = False
    truncated: bool = False


@dataclass
class ActionStep:
    action: VectorN
    reward: float


@dataclass
class AgentTrajectory:
    env_id: int
    agent_id: str
    agent_idx: int
    trajectory: list[Obs | ActionStep | None]
    step_offset: int
    length: int


OBS_SHAPE = (2, 3)
ACT_SHAPE = (2, )


def calculate_offset(step: int, env_id: int, agent_id: int):
    offset = agent_id * 10 + env_id * 100 + step
    return offset


def create_obs(step: int,
               env_id: int,
               agent_id: int,
               step_offset: int,
               terminated: bool = False,
               truncated=False,
               obs_shape: Sequence[int] = OBS_SHAPE):

    step += step_offset
    offset = calculate_offset(step=step, env_id=env_id, agent_id=agent_id)

    item_size = reduce(lambda x, y: x * y, obs_shape)

    obs = np.arange(offset, offset + item_size).reshape(obs_shape)
    info = dict(env=env_id, agent_idx=agent_id, step=step)
    return Obs(obs=obs.astype(float),
               info=info,
               terminated=terminated,
               truncated=truncated)


def create_action(step: int, env_id: int, agent_id: int) -> VectorN:

    offset = calculate_offset(step=step, env_id=env_id, agent_id=agent_id)
    action = np.arange(offset, offset + ACT_SHAPE[0], dtype=np.float_)
    # np.float32(1. * offset)
    return action


def create_reward(step: int, env_id: int, agent_id: int):

    offset = calculate_offset(step=step, env_id=env_id, agent_id=agent_id)
    reward = 1. * offset * 10

    return reward


def create_action_reward(step: int, env_id: int, agent_id: int,
                         step_offset: int):

    action = create_action(step=step + step_offset,
                           env_id=env_id,
                           agent_id=agent_id)
    reward = create_reward(step=step + step_offset,
                           env_id=env_id,
                           agent_id=agent_id)

    return ActionStep(action=action, reward=reward)


def create_trajectory(trajectory: list[Obs | ActionStep],
                      step_offset: int) -> list[Obs | ActionStep | None]:

    return [None] * 2 * step_offset + trajectory


def create_trajectory_factory(env_id: int, agent_id: int, step_offset: int,
                              obs_shape: Sequence[int]):

    obs_factory = partial(create_obs,
                          env_id=env_id,
                          agent_id=agent_id,
                          step_offset=step_offset,
                          obs_shape=obs_shape)
    action_factory = partial(create_action_reward,
                             env_id=env_id,
                             agent_id=agent_id,
                             step_offset=step_offset)
    trajectory_factory = partial(create_trajectory, step_offset=step_offset)

    return obs_factory, action_factory, trajectory_factory


def create_agent_trajectory(
        env_id: int,
        agent_id: int,
        step_offset: int,
        length: int,
        obs_shape: Sequence[int] = OBS_SHAPE) -> AgentTrajectory:

    obs, act, trajectory = create_trajectory_factory(env_id=env_id,
                                                     agent_id=agent_id,
                                                     step_offset=step_offset,
                                                     obs_shape=obs_shape)

    items = []
    for step in range(length):
        obs_act = [obs(step), act(step)]

        items = items + obs_act

    done_obs_act = [obs(length, terminated=True)]
    items = items + done_obs_act

    t = trajectory(items)

    return AgentTrajectory(env_id=env_id,
                           agent_id=f'a{agent_id}',
                           agent_idx=agent_id,
                           trajectory=t,
                           step_offset=step_offset,
                           length=length)


def agent_trajectory_generator(
        env_id: int, agent_id: int, step_offset: int,
        length: int) -> Iterator[None | tuple[Obs, ActionStep, Obs]]:

    agent_trajectory = create_agent_trajectory(env_id, agent_id, step_offset,
                                               length)

    trajectory = agent_trajectory.trajectory

    for i in range(0, len(trajectory) - 2, 2):

        if trajectory[i] is None:
            # skip empty items
            yield None
            continue

        obs = trajectory[i]
        act = trajectory[i + 1]
        obs_next = trajectory[i + 2]

        assert isinstance(obs, Obs)
        assert isinstance(act, ActionStep)
        assert isinstance(obs_next, Obs)

        yield (obs, act, obs_next)


def create_expected_agent_trajectory_rollout_batch(env_id: int, agent_id: int,
                                                   step_offset: int,
                                                   length: int):

    agent_trajectory = create_agent_trajectory(env_id, agent_id, step_offset,
                                               length)

    batch = Batch()

    trajectory = agent_trajectory.trajectory

    for i in range(0, len(trajectory) - 2, 2):

        if trajectory[i] is None:
            # skip empty items
            continue

        obs = trajectory[i]
        act = trajectory[i + 1]
        obs_next = trajectory[i + 2]

        assert isinstance(obs, Obs)
        assert isinstance(act, ActionStep)
        assert isinstance(obs_next, Obs)

        current_batch = Batch(obs=np.array([obs.obs]),
                              act=np.array([act.action]),
                              obs_next=np.array([obs_next.obs]),
                              terminated=np.array([obs_next.terminated]),
                              truncated=np.array([obs_next.truncated]),
                              rew=np.array([act.reward]),
                              done=np.array(
                                  [obs_next.terminated or obs_next.truncated]))

        batch = Batch.cat([batch, current_batch])

    return batch
