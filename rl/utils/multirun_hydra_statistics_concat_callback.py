import shutil
from pathlib import Path
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Iterable, Literal
from pathlib import Path
from omegaconf import DictConfig
from hydra.experimental.callback import Callback
import pandas as pd


@dataclass
class InputParameters:
    filepath: str
    index_column: int | None
    header: int | list[int] | Literal['infer']


@dataclass
class OutputParameters:
    filename_prefix: str
    target_dir: str | None = None
    create_xlsx: bool = False


class MultirunHydraStatisticsConcatCallback(Callback):

    _log: logging.Logger

    _input: InputParameters
    _output: OutputParameters

    def __init__(self, input: InputParameters, output: OutputParameters):

        self._log = logging.getLogger(__class__.__name__)

        self._input = InputParameters(**input)
        self._output = OutputParameters(**output)

    def on_multirun_end(self, config: DictConfig, **kwargs: Any) -> None:

        multirun_dir = Path(config.hydra.sweep.dir).resolve()

        output_df = pd.DataFrame()

        # iterate over the directories
        for job_dir in multirun_dir.glob('*/'):

            job_stat_filepath = job_dir / Path(self._input.filepath)

            if job_stat_filepath.exists():
                self._log.debug('processing stat file; path=%s',
                                job_stat_filepath)

                header_param = self._input.header
                if isinstance(self._input.header,
                              Iterable) and self._input.header != 'infer':
                    header_param = list(self._input.header)

                # ignore pandas 1.x numpy deprecation warning
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                stat_df = pd.read_csv(job_stat_filepath,
                                      index_col=self._input.index_column,
                                      header=header_param)

                output_df = pd.concat([output_df, stat_df], axis=0)

        target_dir = None
        if self._output.target_dir is not None:
            target_dir = Path(self._output.target_dir)
            target_dir.mkdir(parents=True, exist_ok=True)
            assert target_dir.is_dir, f"{target_dir} should be a directory"

        output_prefix = self._output.filename_prefix
        output_path = multirun_dir / f'{output_prefix}.csv'
        output_df.to_csv(output_path)
        if target_dir is not None:
            shutil.copy(output_path, target_dir)

        if self._output.create_xlsx:
            # write as excel too
            output_excel_path = multirun_dir / f'{output_prefix}.xlsx'
            with pd.ExcelWriter(output_excel_path) as writer:
                output_df.to_excel(writer)
            if target_dir is not None:
                shutil.copy(output_excel_path, target_dir)
