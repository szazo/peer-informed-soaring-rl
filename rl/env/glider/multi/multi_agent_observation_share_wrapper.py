from typing import TypeVar, Generic, cast, Any
from functools import partial, lru_cache
from dataclasses import dataclass
from env.glider.base.wrappers import TrajectoryTransformerParams
from env.glider.base.wrappers.trajectory_rotator import calculate_trajectory_to_origo_translate
from env.glider.base.wrappers.trajectory_transformer import create_trajectory_rotator
from env.glider.base.wrappers.zero_pad_sequence import VectorNxN
import numpy as np
import logging
import gymnasium
import pettingzoo
from pettingzoo.utils import BaseParallelWrapper
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper
from utils import VectorN

from trainer.multi_agent.create_agent_id_index_mapping import create_agent_id_index_mapping
from utils.vector import VectorNx3

AgentIDType = TypeVar('AgentIDType')
ObsType = TypeVar('ObsType', bound=np.ndarray)
ActType = TypeVar('ActType')


class ReverseSequenceObservationWrapper(BaseParallelWrapper[AgentIDType,
                                                            ObsType, ActType],
                                        Generic[AgentIDType, ObsType,
                                                ActType]):

    _agent_id_to_index_map: dict[AgentIDType, int]

    def __init__(self, env: pettingzoo.ParallelEnv[AgentIDType, ObsType,
                                                   ActType]):

        super().__init__(env)

        self._agent_id_to_index_map = create_agent_id_index_mapping(env)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict[AgentIDType, ObsType], dict[AgentIDType, dict]]:

        obs, info = self.env.reset(seed=seed, options=options)

        obs = self._modify_obs(obs)

        return obs, info

    def step(
        self, actions: dict[AgentIDType, ActType]
    ) -> tuple[
            dict[AgentIDType, ObsType],
            dict[AgentIDType, float],
            dict[AgentIDType, bool],
            dict[AgentIDType, bool],
            dict[AgentIDType, dict],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._modify_obs(obs)

        return obs, reward, terminated, truncated, info

    def _modify_obs(self, obs: dict[AgentIDType, ObsType]):

        result: dict[AgentIDType, ObsType] = {}
        for agent_id, agent_obs in obs.items():
            agent_idx = self._agent_id_to_index_map[agent_id]

            # for each agent's each item reverse along the sequence axis
            sequence_axis = 0
            flipped_obs = np.flip(agent_obs, sequence_axis)

            result[agent_id] = cast(ObsType, flipped_obs)

        return result


@dataclass
class SortAgentObservationParams:
    max_closest_agent_count: int
    remove_non_meeting_agents: bool
    item_axis: int


class SortAgentObservationWrapper(BaseParallelWrapper[AgentIDType, ObsType,
                                                      ActType],
                                  Generic[AgentIDType, ObsType, ActType]):

    _params: SortAgentObservationParams
    """Only includes agent observations which are in the current agent list"""

    def __init__(self, env: pettingzoo.ParallelEnv[AgentIDType, ObsType,
                                                   ActType],
                 params: SortAgentObservationParams):

        super().__init__(env)

        self._params = params

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict[AgentIDType, ObsType], dict[AgentIDType, dict]]:

        obs, info = self.env.reset(seed=seed, options=options)

        obs = self._modify_obs(obs)

        return obs, info

    def step(
        self, actions: dict[AgentIDType, ActType]
    ) -> tuple[
            dict[AgentIDType, ObsType],
            dict[AgentIDType, float],
            dict[AgentIDType, bool],
            dict[AgentIDType, bool],
            dict[AgentIDType, dict],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._modify_obs(obs)

        return obs, reward, terminated, truncated, info

    def _modify_obs(self, obs: dict[AgentIDType, ObsType]):

        result: dict[AgentIDType, ObsType] = {}
        for agent_idx, agent_id in enumerate(self.possible_agents):

            if agent_id not in obs:
                continue

            agent_obs = obs[agent_id]

            smallest_distances = calculate_smallest_distance_for_items_along_axis(
                agent_obs,
                item_axis=self._params.item_axis,
                self_item_index=agent_idx)

            if self._params.remove_non_meeting_agents:
                # remove items which has no common points with the self_index
                mask = np.logical_not(np.isclose(smallest_distances, np.inf))

                # clear_empty_items_along_axis
                agent_obs = np.compress(mask,
                                        agent_obs,
                                        axis=self._params.item_axis)

                # also filter the distances
                smallest_distances = smallest_distances[mask]

            sorted_obs = sort_items_along_axis(
                agent_obs,
                item_axis=self._params.item_axis,
                metric=smallest_distances)

            # use only the first maximum number of agents

            assert self._params.item_axis == 1
            sorted_obs = sorted_obs[:, :self._params.max_closest_agent_count,
                                    ...]

            result[agent_id] = cast(ObsType, sorted_obs)

        return result


def calculate_smallest_distance_between_trajectories(positions1: VectorNxN,
                                                     positions2: VectorNxN):

    assert positions1.ndim == 2, 'two dimensional input is required'
    assert positions2.ndim == 2, 'two dimensional input is required'

    # filter out steps when any of them is nan along the whole row axis
    isnan1 = np.all(np.isnan(positions1), axis=1)
    isnan2 = np.all(np.isnan(positions2), axis=1)

    isnan = np.logical_not(np.logical_or(isnan1, isnan2))

    # filter both of them if any of them is nan
    positions1 = positions1[isnan]
    positions2 = positions2[isnan]

    assert positions1.shape == positions2.shape
    if positions1.shape[0] == 0:
        return np.inf

    diff = positions2 - positions1
    norms = np.linalg.norm(diff, axis=1)

    min = np.min(norms)

    return min


def sort_items_along_axis(input: np.ndarray, item_axis: int, metric: VectorN):

    # get the permutation
    permutation = np.argsort(metric)

    # take items along the axis
    result = np.take(input, indices=permutation, axis=item_axis)

    return result


def calculate_smallest_distance_for_items_along_axis(input: np.ndarray,
                                                     item_axis: int,
                                                     self_item_index: int):

    # select self item along the axis
    self_item = np.take(input, indices=self_item_index, axis=item_axis)

    # calculate smallest distance for each other item
    func = partial(calculate_smallest_distance_between_trajectories, self_item)

    item_count = input.shape[item_axis]

    # TODO: eliminate iteration
    result = []
    for i in range(item_count):
        current_item = np.take(input, indices=i, axis=item_axis)
        metric = func(current_item)

        result.append(metric)

    return np.array(result)


def move_item_along_axis(input: np.ndarray, item_axis: int, source_index: int,
                         target_index: int):
    """Move an item from source_index to target_index along the specified axis"""

    # possible indices along the axis
    indices = list(range(input.shape[item_axis]))

    item = indices.pop(source_index)
    indices.insert(target_index, item)
    indices = np.array(indices)

    # expand the dimensions of the indices before and after the item axis to match the input dimension
    new_shape = [1] * item_axis + [indices.size
                                   ] + [1] * (input.ndim - item_axis - 1)
    indices = indices.reshape(new_shape)

    result = np.take_along_axis(input, indices, axis=item_axis)

    return result


def clear_empty_items_along_axis(input: np.ndarray,
                                 axis: int,
                                 empty_value: Any,
                                 input_mask: tuple | None = None):
    # check whether nan expect the axis in interest
    axes = list(range(input.ndim))
    axes.remove(axis)

    # input_mask = [...,:-1]
    if input_mask is None:
        input_mask = (slice(None))

    # create the mask
    equal_nan = np.isnan(empty_value)
    mask = np.logical_not(
        np.all(np.isclose(input[input_mask], empty_value, equal_nan=equal_nan),
               axis=tuple(axes)))

    # apply the mask along the axis
    clean = np.compress(mask, input, axis=axis)

    return clean


@dataclass
class Trajectory:
    position: VectorNx3
    velocity: VectorNx3


def write_trajectories_back(input: np.ndarray,
                            agent_trajectories: list[Trajectory],
                            agent_masks: list[np.ndarray],
                            position_3d_start_column_index: int,
                            velocity_3d_start_column_index: int,
                            copy: bool = True):

    if copy:
        input = input.copy()

    for i, agent_trajectory in enumerate(agent_trajectories):
        agent_mask = agent_masks[i]
        input[i][
            agent_mask,
            position_3d_start_column_index:position_3d_start_column_index +
            3] = agent_trajectory.position
        input[i][
            agent_mask,
            velocity_3d_start_column_index:velocity_3d_start_column_index +
            3] = agent_trajectory.velocity

    return input


def extract_multi_agent_trajectories(input: np.ndarray,
                                     position_3d_start_column_index: int,
                                     velocity_3d_start_column_index: int,
                                     pad_value=0.):
    """extract position and velocities from the input which is in shape Agent x Sequence x Dim """

    equal_nan = np.isnan(pad_value)
    mask = np.logical_not(
        np.all(np.isclose(input, pad_value, equal_nan=equal_nan), axis=-1))

    agent_trajectories: list[Trajectory] = []
    agent_masks: list[VectorNxN] = []

    # create the trajectories
    for i in range(input.shape[0]):

        agent_input = input[i]

        agent_mask = mask[i]
        agent_clean_input = agent_input[agent_mask]

        # get the position and velocity parts
        position = agent_clean_input[:, position_3d_start_column_index:
                                     position_3d_start_column_index + 3]
        velocity = agent_clean_input[:, velocity_3d_start_column_index:
                                     velocity_3d_start_column_index + 3]

        agent_trajectories.append(
            Trajectory(position=position, velocity=velocity))
        agent_masks.append(agent_mask)

    return agent_trajectories, agent_masks


def normalize_multi_agent_trajectories(agent_trajectories: list[Trajectory],
                                       params: TrajectoryTransformerParams):

    if len(agent_trajectories) == 0:
        return agent_trajectories

    output_trajectories = [*agent_trajectories]

    # rotate the trajectories based on the first
    rotator = create_trajectory_rotator(params)
    if rotator is not None:
        # create the rotation transform based on the first agent
        first_trajectory = output_trajectories[0]
        rotation_transform = rotator.create_transform(
            position=first_trajectory.position,
            velocity=first_trajectory.velocity)

        if rotation_transform is not None:
            # rotate all of the trajectories based on the first trajectory
            # so the other trajectories will also be rotated relative to the egocentric trajectory
            for i, agent_trajectory in enumerate(output_trajectories):

                rotated_position, rotated_velocity = rotator.transform(
                    rotation_transform,
                    position=agent_trajectory.position,
                    velocity=agent_trajectory.velocity)

                output_trajectories[i] = Trajectory(position=rotated_position,
                                                    velocity=rotated_velocity)

    # calculate the translate vector of the first trajectory to the origo
    # and use the resulting translation vector to translate the others too
    translate_vector = calculate_trajectory_to_origo_translate(
        output_trajectories[0].position,
        relative_to=params.translate_relative_to)
    for i, agent_trajectory in enumerate(output_trajectories):
        output_trajectories[i] = Trajectory(
            position=agent_trajectory.position + translate_vector,
            velocity=agent_trajectory.velocity)

    return output_trajectories


@dataclass
class NormalizeMultiAgentTrajectoriesObservationParams:
    trajectory_transform_params: TrajectoryTransformerParams
    position_3d_start_column_index: int
    velocity_3d_start_column_index: int
    pad_value: float


def normalize_position_by_logaritmic_distance(
        input: np.ndarray, position_3d_start_column_index: int):

    position = input[
        ..., position_3d_start_column_index:position_3d_start_column_index +
        3].copy()

    norm = np.linalg.norm(position, axis=-1)

    # mask zero norms
    nonzero_mask = np.logical_not(norm == 0)
    masked_norm = norm[nonzero_mask]
    masked_pos = position[nonzero_mask]

    # create the normalizer based on the log norm
    normalizer = masked_norm / np.log(1 + masked_norm)
    normalizer = np.expand_dims(normalizer, -1)

    # normalize
    normalized = masked_pos / normalizer
    position[nonzero_mask] = normalized

    # write back
    result = input.copy()
    result[..., position_3d_start_column_index:position_3d_start_column_index +
           3] = position

    return result


def normalize_position_by_logaritmic_distance_obs_wrapper(
        env: gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv,
        position_3d_start_column_index: int):

    class NormalizePositionByLogaritmicDistanceObservationModifier(
            BaseModifier):

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:
            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = normalize_position_by_logaritmic_distance(
                obs, position_3d_start_column_index)

            assert not np.any(
                np.isnan(result_obs)
            ), 'nan found in the observation after logaritmic normalization'

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(
        env, NormalizePositionByLogaritmicDistanceObservationModifier)


def normalize_multi_agent_trajectories_observation(
        input: np.ndarray,
        params: NormalizeMultiAgentTrajectoriesObservationParams):

    # extract the trajectories
    trajectories, masks = extract_multi_agent_trajectories(
        input,
        position_3d_start_column_index=params.position_3d_start_column_index,
        velocity_3d_start_column_index=params.velocity_3d_start_column_index,
        pad_value=params.pad_value)

    # transform them
    normalized_trajectories = normalize_multi_agent_trajectories(
        agent_trajectories=trajectories,
        params=params.trajectory_transform_params)

    # write back
    output = write_trajectories_back(
        input,
        agent_trajectories=normalized_trajectories,
        agent_masks=masks,
        position_3d_start_column_index=params.position_3d_start_column_index,
        velocity_3d_start_column_index=params.velocity_3d_start_column_index,
        copy=True)

    return output


def normalize_multi_agent_trajectories_obs_wrapper(
        env: gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv,
        params: NormalizeMultiAgentTrajectoriesObservationParams):

    class NormalizeMultiAgentTrajectoriesObservationModifier(BaseModifier):

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:
            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = normalize_multi_agent_trajectories_observation(
                obs, params=params)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(env,
                          NormalizeMultiAgentTrajectoriesObservationModifier)


@dataclass
class MoveAxisParams:
    source_axis: int
    destination_axis: int


def move_axis_obs_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv,
    params: MoveAxisParams
) -> gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv:

    class MoveAxisObservationModifier(BaseModifier):

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:
            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = np.moveaxis(obs,
                                     source=params.source_axis,
                                     destination=params.destination_axis)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(env, MoveAxisObservationModifier)


def nan_to_zero_obs_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv
) -> gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv:

    class NanToZeroObservationModifier(BaseModifier):

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:
            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = np.nan_to_num(obs, nan=0.)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(env, NanToZeroObservationModifier)


@dataclass
class ClearEmptyItemsObsParams:
    axis: int
    empty_value: Any
    input_mask: tuple | None = None


def clear_empty_items_obs_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv,
    params: ClearEmptyItemsObsParams
) -> gymnasium.Env | pettingzoo.ParallelEnv | pettingzoo.AECEnv:

    class ClearEmptyItemsObservationModifier(BaseModifier):

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:
            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = clear_empty_items_along_axis(
                obs,
                axis=params.axis,
                empty_value=params.empty_value,
                input_mask=params.input_mask)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

    return shared_wrapper(env, ClearEmptyItemsObservationModifier)


class RemoveNonExistentAgentObservationWrapper(BaseParallelWrapper[AgentIDType,
                                                                   ObsType,
                                                                   ActType],
                                               Generic[AgentIDType, ObsType,
                                                       ActType]):
    """Only includes agent observations which are in the current agent list of before after step"""

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict[AgentIDType, ObsType], dict[AgentIDType, dict]]:

        obs, info = self.env.reset(seed=seed, options=options)

        obs = self._modify_obs(obs, self.agents)

        return obs, info

    def step(
        self, actions: dict[AgentIDType, ActType]
    ) -> tuple[
            dict[AgentIDType, ObsType],
            dict[AgentIDType, float],
            dict[AgentIDType, bool],
            dict[AgentIDType, bool],
            dict[AgentIDType, dict],
    ]:

        original_agents = set(self.agents)
        obs, reward, terminated, truncated, info = self.env.step(actions)

        final_agents = set(self.agents)
        required_agents = original_agents | final_agents

        obs = self._modify_obs(obs, list(required_agents))

        return obs, reward, terminated, truncated, info

    def _modify_obs(self, obs: dict[AgentIDType, ObsType],
                    required_agents: list[AgentIDType]):

        result_obs: dict[AgentIDType, ObsType] = {}
        for agent_id in required_agents:
            result_obs[agent_id] = obs[agent_id]

        return result_obs


def append_item_index_to_the_target_axis(input: np.ndarray, item_axis: int,
                                         target_axis: int):
    """Create a new column at the target axis
    which contains the item index (determined by item_axis)"""

    # create the shape for the column, indices will be broadcasted to this shape
    column_shape = [*input.shape]
    column_shape[target_axis] = 1

    # create the indices, it will contain the indices at the item axis
    count = input.shape[item_axis]
    indices = np.arange(count)
    indices_shape = [1] * item_axis + [indices.size
                                       ] + [1] * (input.ndim - item_axis - 1)
    indices = indices.reshape(indices_shape)

    # broadcast the indices to match the column's shape
    indices = np.broadcast_to(indices, column_shape)

    # add the column to the input on the target axis
    output = np.append(input, indices, axis=target_axis)

    return output


EnvType = TypeVar('EnvType',
                  bound=gymnasium.Env | pettingzoo.ParallelEnv
                  | pettingzoo.AECEnv)


def append_multi_agent_item_index_obs_wrapper(env: EnvType, item_axis: int,
                                              target_axis: int) -> EnvType:

    class AppendMultiAgentItemIndexObservationModifier(BaseModifier):

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)

        def modify_obs(self, obs: np.ndarray) -> np.ndarray:
            self._log.debug("modify_obs; obs=%s", obs)

            result_obs = append_item_index_to_the_target_axis(
                obs, item_axis=item_axis, target_axis=target_axis)

            self._log.debug("modify_obs; new_shape=%s, new_obs=%s",
                            result_obs.shape, result_obs)

            return result_obs

        def modify_obs_space(
                self,
                obs_space: gymnasium.spaces.Box) -> gymnasium.spaces.Space:

            return obs_space

    return shared_wrapper(env, AppendMultiAgentItemIndexObservationModifier)


class MultiAgentObservationShareWrapper(BaseParallelWrapper[AgentIDType,
                                                            ObsType, ActType],
                                        Generic[AgentIDType, ObsType,
                                                ActType]):

    _agent_id_to_index_map: dict[AgentIDType, int]

    def __init__(self, env: pettingzoo.ParallelEnv[AgentIDType, ObsType,
                                                   ActType]):

        super().__init__(env)

        self._agent_id_to_index_map = create_agent_id_index_mapping(env)

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[dict[AgentIDType, ObsType], dict[AgentIDType, dict]]:

        obs, info = self.env.reset(seed=seed, options=options)

        obs = self._modify_obs(obs)

        return obs, info

    def step(
        self, actions: dict[AgentIDType, ActType]
    ) -> tuple[
            dict[AgentIDType, ObsType],
            dict[AgentIDType, float],
            dict[AgentIDType, bool],
            dict[AgentIDType, bool],
            dict[AgentIDType, dict],
    ]:
        obs, reward, terminated, truncated, info = self.env.step(actions)
        obs = self._modify_obs(obs)

        return obs, reward, terminated, truncated, info

    @lru_cache(maxsize=None)
    def observation_space(self, agent):
        space = self.env.observation_space(agent)
        assert isinstance(space, gymnasium.spaces.Box)
        return self._modify_obs_spaces(space)

    def _modify_obs_spaces(self, obs_space: gymnasium.spaces.Box):

        agent_count = len(self.possible_agents)

        low = np.repeat(obs_space.low[np.newaxis, ...], agent_count, axis=0)
        high = np.repeat(obs_space.high[np.newaxis, ...], agent_count, axis=0)
        dtype: Any = obs_space.dtype
        new_obs_space = gymnasium.spaces.Box(low=low, high=high, dtype=dtype)

        return new_obs_space

    def _modify_obs(self, obs: dict[AgentIDType, ObsType]):

        if len(obs) == 0:
            return obs

        agent_count = len(self.possible_agents)

        # select the first agent's observation to get the shape
        first_obs = next(iter(obs.values()))
        new_shape = (agent_count, *first_obs.shape)

        shared_obs = np.full(new_shape, np.nan)

        # fill the rows for the agents that has observation
        for row_agent_id, row_agent_obs in obs.items():
            row_agent_idx = self._agent_id_to_index_map[row_agent_id]
            shared_obs[row_agent_idx] = row_agent_obs

        shared_obs = cast(ObsType, shared_obs)

        # use the same observation for each possible agent (even if not exists in the current list)
        result_obs: dict[AgentIDType, ObsType] = {}
        for agent_id in self.possible_agents:
            # for agent_id, agent_obs in obs.items():
            result_obs[agent_id] = shared_obs

        return result_obs
