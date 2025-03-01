from abc import ABC, abstractmethod
import os
import pathlib
import numpy as np
import gymnasium
from ..experiment_logger import ExperimentLoggerInterface


# base class for wrapping vectorized environment with a wrapper and collect/merge results
class VectorizedEnvWrapper(ABC):

    _output_dir: str

    def __init__(self, output_dir: str):

        self._output_dir = output_dir

    @abstractmethod
    def wrap_env(self, env: gymnasium.Env,
                 vectorized_index: int) -> gymnasium.Env:
        pass

    @abstractmethod
    def merge_vectorized_output(self, episode_env_indices: list[int]) -> None:
        pass

    @abstractmethod
    def log_merged_output(
            self, parent_key: str, episode_count: int,
            experiment_logger: ExperimentLoggerInterface) -> None:
        pass

    def _collect_vectorized_env_episode_numbers(
            self, episode_env_indices: list[int]):

        episode_env_indices_np = np.array(episode_env_indices)
        vectorized_env_count = np.max(episode_env_indices_np) + 1

        episode_numbers_per_envs = []

        # process all vectorized envs and replace episode counts
        for env_index in range(vectorized_env_count):
            episode_numbers = np.where(episode_env_indices_np == env_index)[0]
            episode_numbers_per_envs.append(episode_numbers.tolist())

        return episode_numbers_per_envs

    def _vectorized_output_dir(self, vectorized_env_index: int):
        return os.path.join(self._output_dir, str(vectorized_env_index))

    def _ensure_output_dir(self):
        return self._ensure_dir(self._output_dir)

    def _ensure_dir(self, directory: str):
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        return directory
