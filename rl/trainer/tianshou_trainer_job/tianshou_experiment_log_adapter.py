from typing import Callable, Tuple, Any
from dataclasses import asdict
import numpy as np
import tianshou

from tianshou.utils.logger.base import VALID_LOG_VALS_TYPE, VALID_LOG_VALS
from trainer.experiment_logger import ExperimentLoggerInterface
from .collect_multi_agent_returns_and_stats import collect_multi_agent_returns_and_stats


class TianshouExperimentLogAdapter(tianshou.utils.BaseLogger):

    _is_multi_agent: bool
    _multi_agent_possible_agents: list

    _experiment_logger: ExperimentLoggerInterface

    def __init__(self,
                 train_step_interval: int,
                 test_step_interval: int,
                 update_step_interval: int,
                 experiment_logger: ExperimentLoggerInterface,
                 is_multi_agent: bool = False,
                 multi_agent_possible_agents=[]):
        self._experiment_logger = experiment_logger

        self._is_multi_agent = is_multi_agent
        self._multi_agent_possible_agents = multi_agent_possible_agents

        super().__init__(train_interval=train_step_interval,
                         test_interval=test_step_interval,
                         update_interval=update_step_interval)

    def write(self, step_type: str, step: int,
              data: dict[str, VALID_LOG_VALS_TYPE]) -> None:

        scope, _ = step_type.split("/")
        self._experiment_logger.log_metrics(key=step_type,
                                            value=step,
                                            step=step)

        for k, v in data.items():
            assert not isinstance(
                v,
                np.ndarray), 'numpy array is not supported as item in a series'
            scope_key = f"{scope}/{k}"
            self._experiment_logger.log_metrics(key=scope_key,
                                                value=v,
                                                step=step)

    # copied from tianshou tensorboard logger
    def prepare_dict_for_logging(
        self,
        log_data: dict[str, Any],
        parent_key: str = "",
        delimiter: str = "/",
        exclude_arrays: bool = True,
    ) -> dict[str, VALID_LOG_VALS_TYPE]:
        """Flattens and filters a nested dictionary by recursively traversing all levels and compressing the keys.

        Filtering is performed with respect to valid logging data types.

        :param input_dict: The nested dictionary to be flattened and filtered.
        :param parent_key: The parent key used as a prefix before the input_dict keys.
        :param delimiter: The delimiter used to separate the keys.
        :param exclude_arrays: Whether to exclude numpy arrays from the output.
        :return: A flattened dictionary where the keys are compressed and values are filtered.
        """
        result = {}

        if self._is_multi_agent:
            log_data = self._prepare_multi_agent_log_data(log_data)

        def add_to_result(
            cur_dict: dict,
            prefix: str = "",
        ) -> None:
            for key, value in cur_dict.items():
                if exclude_arrays and isinstance(value, np.ndarray):
                    continue

                new_key = prefix + delimiter + key
                new_key = new_key.lstrip(delimiter)

                if isinstance(value, dict):
                    add_to_result(
                        value,
                        new_key,
                    )
                elif isinstance(value, VALID_LOG_VALS):
                    result[new_key] = value

        input_dict = log_data
        add_to_result(input_dict, prefix=parent_key)

        return result

    def _prepare_multi_agent_log_data(self,
                                      log_data: dict[str, Any],
                                      delimiter: str = "/"):

        if 'returns' in log_data:
            log_data = log_data.copy()

            # delete returns stat and create per agent statistics
            del log_data['returns_stat']

            returns = log_data['returns']
            assert isinstance(returns, np.ndarray)

            multi_agent_returns = collect_multi_agent_returns_and_stats(
                returns=returns,
                possible_agents=self._multi_agent_possible_agents)

            for agent_id, agent_returns in multi_agent_returns.items():
                log_data[agent_id + delimiter + 'returns'] = asdict(
                    agent_returns.returns_stat)

        return log_data

    def save_data(
        self,
        epoch: int,
        env_step: int,
        gradient_step: int,
        save_checkpoint_fn: Callable[[int, int, int], str] | None = None,
    ) -> None:
        # we use our own mechanism to save data
        pass

    def restore_data(self) -> Tuple[int, int, int]:
        # we use our own mechanism to restore data
        return 0, 0, 0

    @staticmethod
    def restore_logged_data(log_path: str) -> dict:
        # we don't use this
        return {}

    def finalize(self) -> None:
        # we don't use this
        pass
