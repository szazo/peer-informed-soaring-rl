import warnings
from numbers import Number
from dataclasses import dataclass
import logging
import numpy as np

from .api import ExperimentLoggerInterface, ExperimentLoggerParametersBase
from .dictionary_to_yaml import dictionary_to_yaml
from .file_cache import FileCache

# filter neptune warnings before import
warnings.filterwarnings("ignore",
                        category=DeprecationWarning,
                        module='^bravado_core.*')
warnings.filterwarnings("ignore",
                        category=DeprecationWarning,
                        module='^swagger_spec_validator.*')
import neptune
from neptune.utils import stringify_unsupported


@dataclass
class NeptuneExperimentLoggerParameters(ExperimentLoggerParametersBase):
    # existing run id to resolve (or load)
    with_existing_id: str | None = None
    # do not override the run (e.g. for evaluation)
    readonly: bool | None = None
    # if not set, NEPTUNE_PROJECT env variable will be used
    project: str | None = None
    # if not set, NEPTUNE_API_TOKEN env variable will be used
    api_token: str | None = None


class NeptuneExperimentLogger(ExperimentLoggerInterface):

    _log: logging.Logger
    _params: NeptuneExperimentLoggerParameters

    _run: neptune.Run | None = None

    _file_cache: FileCache

    def __init__(
        self,
        # custom name of the run
        name: str | None,
        # existing run id to resolve (or load)
        with_existing_id: str | None = None,
        # do not override the run (e.g. for evaluation)
        readonly: bool | None = None,
        # if not set, NEPTUNE_PROJECT env variable will be used
        project: str | None = None,
        # if not set, NEPTUNE_API_TOKEN env variable will be used
        api_token: str | None = None
    ) -> None:

        self._log = logging.getLogger(__class__.__name__)

        self._params = NeptuneExperimentLoggerParameters(
            name=name,
            with_existing_id=with_existing_id,
            readonly=readonly,
            project=project,
            api_token=api_token)
        self._file_cache = FileCache()
        self._log.debug('initializing; params=%s', self._params)

    def log_video(self, key: str, path: str):
        self.log_file(key, path)

    def log_file(self, key: str, path: str):
        self._file_cache.set(key=key, path=path)
        self._resolve_run()[key].upload(path)

    def query_file(self, key: str, destination_path: str):
        if self._file_cache.get(key=key, destination_path=destination_path):
            return

        self._resolve_run()[key].download(destination=destination_path)

    def is_key_exists(self, key: str):
        value = self._resolve_run().exists(key)

        return value

    def log_dict(self,
                 key: str,
                 dictionary: dict,
                 log_as_str=False,
                 log_as_pickle=False):

        self._log.debug('log_dict; key=%s,log_as_str=%s,log_as_pickle=%s', key,
                        log_as_str, log_as_pickle)

        run = self._resolve_run()
        run[key] = stringify_unsupported(dictionary)

        if log_as_str:
            yaml_string = dictionary_to_yaml(dictionary)
            run[f"{key}_str"] = yaml_string

        if log_as_pickle:
            run[f"{key}_pkl"].upload(neptune.types.File.as_pickle(dictionary))

    # REVIEW: rename to log_series or something similar
    def log_metrics(self,
                    key: str,
                    value: float | str | Number | np.number,
                    step: int | None = None):
        self._log.debug('log_metrics; key=%s,value=%s,step=%s', key, value,
                        step)

        self._resolve_run()[key].append(value=value, step=step)

    def _resolve_run(self) -> neptune.Run:

        if self._run is None:
            mode = 'read-only' if self._params.readonly else None

            self._run = neptune.init_run(project=self._params.project,
                                         api_token=self._params.api_token,
                                         name=self._params.name,
                                         with_id=self._params.with_existing_id,
                                         mode=mode,
                                         capture_stderr=True,
                                         capture_stdout=True,
                                         capture_traceback=True,
                                         capture_hardware_metrics=True)
        return self._run

    def stop(self, success: bool):
        if self._run is None:
            return

        if not self._params.readonly:
            if success:
                self._run['info/state'] = 'Success'
            else:
                self._run['info/state'] = 'Failed'
                self._run['sys/failed'] = True

        self._run.stop()
