import logging
import os
import gymnasium
from .vectorized_env_wrapper import VectorizedEnvWrapper, ExperimentLoggerInterface


class VectorizedVideoRecorder(VectorizedEnvWrapper):

    _log: logging.Logger

    def __init__(self, output_dir: str):
        super().__init__(output_dir=output_dir)

        self._log = logging.getLogger(__class__.__name__)

        self._log.debug('__init__; output_dir=%s', output_dir)

    def wrap_env(self, env: gymnasium.Env,
                 vectorized_index: int) -> gymnasium.Env:

        vectorized_output_dir = self._vectorized_output_dir(vectorized_index)
        vectorized_video_dir = self._video_dir(vectorized_output_dir)

        env = gymnasium.wrappers.RecordVideo(env,
                                             video_folder=vectorized_video_dir,
                                             episode_trigger=lambda _: True,
                                             name_prefix='video',
                                             disable_logger=True)

        return env

    def merge_vectorized_output(self, episode_env_indices: list[int]):

        env_episode_numbers = self._collect_vectorized_env_episode_numbers(
            episode_env_indices)

        self._log.debug('collecting videos; env_episode_numbers=%s',
                        env_episode_numbers)

        output_dir = self._ensure_output_dir()
        for env_index, episode_numbers in enumerate(env_episode_numbers):
            vectorized_video_dir = self._vectorized_video_dir(env_index)

            for vectorized_episode, episode in enumerate(episode_numbers):
                vectorized_filepath = os.path.join(
                    vectorized_video_dir,
                    f'video-episode-{vectorized_episode}.mp4')
                target_filepath = self._output_filepath(output_dir,
                                                        episode=episode)

                os.link(vectorized_filepath, target_filepath)

    def log_merged_output(self, parent_key: str, episode_count: int,
                          experiment_logger: ExperimentLoggerInterface):

        for i in range(episode_count):
            video_path = self._output_filepath(self._output_dir, episode=i)

            experiment_logger.log_video(f'{parent_key}_video{i}', video_path)

    def _output_filepath(self, output_dir: str, episode: int):
        return os.path.join(output_dir, f'video-{episode}.mp4')

    def _vectorized_video_dir(self, vectorized_env_index: int):
        return self._video_dir(
            self._vectorized_output_dir(vectorized_env_index))

    def _video_dir(self, output_dir: str):
        video_dir = os.path.join(output_dir, 'video')
        return video_dir
