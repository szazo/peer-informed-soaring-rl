import logging
from dataclasses import dataclass
import numpy as np
import gymnasium
import pettingzoo
from supersuit.generic_wrappers.utils.base_modifier import BaseModifier
from supersuit.generic_wrappers.utils.shared_wrapper_util import shared_wrapper

from ..agent import GliderAgentObsType


@dataclass
class GliderDictToArrayObservationParams:
    include_absolute_position: bool
    include_velocity_vector: bool
    include_vertical_velocity: bool
    include_roll: bool
    include_yaw: bool
    include_earth_relative_velocity_norm: bool
    include_airmass_relative_velocity_norm: bool
    position_normalization_factor: float = 1.0


def glider_dict_to_array_obs_wrapper(
    env: gymnasium.Env | pettingzoo.ParallelEnv,
    params: GliderDictToArrayObservationParams
) -> gymnasium.Env | pettingzoo.ParallelEnv:

    class GliderDictToArrayModifier(BaseModifier):

        _params: GliderDictToArrayObservationParams

        def __init__(self):
            super().__init__()

            self._log = logging.getLogger(__class__.__name__)
            self._params = params

        def reset(self,
                  seed: int | None = None,
                  options: dict | None = None) -> None:
            self._log.debug("reset; seed=%s,options=%s", seed, options)

        def modify_obs(self, obs: GliderAgentObsType) -> np.ndarray:

            self._log.debug("modify_obs; obs=%s", obs)

            position_earth_xyz_m = obs["position_earth_xyz_m"]
            velocity_earth_xyz_m_per_s = obs["velocity_earth_xyz_m_per_s"]
            yaw_pitch_roll_earth_to_body_rad = obs[
                "yaw_pitch_roll_earth_to_body_rad"]
            velocity_airmass_relative_xyz_m_per_s = obs[
                "velocity_airmass_relative_xyz_m_per_s"]

            result = np.array([])
            if self._params.include_absolute_position:
                result = np.hstack((
                    result,
                    position_earth_xyz_m *
                    self._params.position_normalization_factor,
                ))

            if self._params.include_vertical_velocity:
                result = np.hstack((result, velocity_earth_xyz_m_per_s[2]))

            if self._params.include_velocity_vector:
                result = np.hstack((result, velocity_earth_xyz_m_per_s))

            if self._params.include_roll:
                result = np.hstack(
                    (result, yaw_pitch_roll_earth_to_body_rad[2]))

            if self._params.include_yaw:
                result = np.hstack(
                    (result, yaw_pitch_roll_earth_to_body_rad[0]))

            if self._params.include_earth_relative_velocity_norm:
                result = np.hstack(
                    (result, np.linalg.norm(velocity_earth_xyz_m_per_s)))

            if self._params.include_airmass_relative_velocity_norm:
                result = np.hstack(
                    (result,
                     np.linalg.norm(velocity_airmass_relative_xyz_m_per_s)))

            if not np.isfinite(result).all():
                print("invalid observation:", result)
                raise ValueError(f"Observation is not finite: {result}")

            self._log.debug("modify_obs; new_obs=%s", result)

            return result

        def modify_obs_space(
                self,
                obs_space: gymnasium.spaces.Space) -> gymnasium.spaces.Space:

            self._log.debug("modify_obs_space; obs_space=%s", obs_space)

            dim = 0
            if self._params.include_absolute_position:
                dim += 3

            assert not (self._params.include_velocity_vector
                        and self._params.include_vertical_velocity
                        ), "cannot set both vector and vertical velocity"

            if self._params.include_vertical_velocity:
                dim += 1

            if self._params.include_velocity_vector:
                dim += 3

            if self._params.include_roll:
                dim += 1

            if self._params.include_yaw:
                dim += 1

            if self._params.include_earth_relative_velocity_norm:
                dim += 1

            if self._params.include_airmass_relative_velocity_norm:
                dim += 1

            new_obs_space = gymnasium.spaces.Box(shape=(dim, ),
                                                 low=-np.inf,
                                                 high=np.inf)

            self._log.debug("modify_obs_space; new_obs_space=%s",
                            new_obs_space)

            return new_obs_space

        # def modify_action_space(
        #     self, act_space: gymnasium.spaces.Space
        # ) -> gymnasium.spaces.Space:
        #     print("modify_action_space")
        #     return act_space

        # def modify_action(self, act: ActionType) -> ActionType:
        #     print("modify_action", act)
        #     return act

    wrapped = shared_wrapper(env, GliderDictToArrayModifier)
    assert isinstance(wrapped, gymnasium.Env) or isinstance(
        wrapped, pettingzoo.ParallelEnv)

    return wrapped
