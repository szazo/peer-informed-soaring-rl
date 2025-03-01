from typing import Any
import os
import tempfile
import torch
from ..experiment_logger import ExperimentLoggerInterface


# save/load weights using experiment logger
class ExperimentLoggerWeightStore:

    _experiment_logger: ExperimentLoggerInterface

    def __init__(self, experiment_logger: ExperimentLoggerInterface):
        self._experiment_logger = experiment_logger

    def load_best_weights(self,
                          map_device: torch.device | None = None
                          ) -> dict[str, Any]:

        return self.load_weights(checkpoint_name='best_weights',
                                 map_device=map_device)

    def load_weights(self,
                     checkpoint_name: str,
                     map_device: torch.device | None = None) -> dict[str, Any]:

        with tempfile.TemporaryDirectory() as tmp:
            target_path = os.path.join(tmp, 'weights_to_load.pth')

            print('downloading weights:', checkpoint_name)
            self._experiment_logger.query_file(f'checkpoint/{checkpoint_name}',
                                               destination_path=target_path)

            # load into the policy
            loaded = torch.load(target_path, map_location=map_device)
            return loaded

    def save_best_weights(self,
                          state_dict: dict[str, Any],
                          log_key: str,
                          target_filepath: str,
                          log=True):
        torch.save(state_dict, target_filepath)
        if log:
            self._experiment_logger.log_file(log_key, target_filepath)
