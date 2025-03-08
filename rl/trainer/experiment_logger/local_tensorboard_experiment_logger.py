import logging
from dataclasses import dataclass
from numbers import Number
import numpy as np

from .api import ExperimentLoggerInterface, ExperimentLoggerParametersBase
from torch.utils.tensorboard.writer import SummaryWriter
from .dictionary_to_yaml import dictionary_to_yaml
from .file_cache import FileCache
from torchvision.transforms import functional
import torch
import imageio


@dataclass(kw_only=True)
class LocalTensorBoardExperimentLoggerParameters(ExperimentLoggerParametersBase
                                                 ):
    log_dir: str
    flush_secs: int = 60


class LocalTensorBoardExperimentLogger(ExperimentLoggerInterface):

    _log: logging.Logger
    _writer: SummaryWriter

    _file_cache: FileCache

    def __init__(self, log_dir: str, flush_secs: int, name: str | None):
        self._log = logging.getLogger(__class__.__name__)
        self._file_cache = FileCache()
        self._writer = SummaryWriter(log_dir=log_dir,
                                     flush_secs=flush_secs,
                                     comment=name or '')

    def log_file(self, key: str, path: str):
        # tensorboard does not support file, we just log the path
        self._writer.add_text(tag=key, text_string=path)
        self._file_cache.set(key=key, path=path)

    def query_file(self, key: str, destination_path: str):

        if self._file_cache.get(key=key, destination_path=destination_path):
            return

        raise NotImplementedError(
            f'query_file not supported in Tensorboard logger after closing it; file with key "{key}" not found'
        )

    def log_video(self, key: str, path: str):
        self._log.debug('log_video; key=%s,path=%s', key, path)

        self._log.debug('loading video...')
        reader = imageio.get_reader(path)
        meta = reader.get_meta_data()
        self._log.debug('video meta=%s', meta)

        frames = [functional.to_tensor(frame) for frame in reader]

        # add_video expects a 5D Tensor (batch, time, color, height, width)
        video_tensor = torch.stack(frames).permute(0, 1, 2, 3)
        video_tensor = video_tensor.unsqueeze(0)

        self._log.debug('video_tensor.shape=%s', video_tensor.shape)
        reader.close()

        self._log.debug('adding video...')
        self._writer.add_video(tag=key, vid_tensor=video_tensor)

    def log_dict(self,
                 key: str,
                 dictionary: dict,
                 log_as_str=False,
                 log_as_pickle=False):

        self._log.debug('log_dict; key=%s', key)
        if log_as_pickle:
            self._log.debug(
                'log_as_pickle is not supported in tensorboard logger')

        # only log yaml as markdown code
        yaml_string = dictionary_to_yaml(dictionary)
        yaml_string = '```\n' + yaml_string + '\n```'

        self._writer.add_text(tag=key, text_string=yaml_string)

    def log_metrics(self,
                    key: str,
                    value: float | str | Number | np.number,
                    step: int | None = None):
        self._log.debug('log_metrics; key=%s,value=%s,step=%s', key, value,
                        step)

        self._writer.add_scalar(tag=key, scalar_value=value, global_step=step)

    def stop(self, success: bool):
        self._log.debug('stop; success=%s', success)
        self._writer.close()
