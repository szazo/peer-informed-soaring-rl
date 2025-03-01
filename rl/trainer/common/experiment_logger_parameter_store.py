import os
import tempfile
import pickle
from ..experiment_logger import ExperimentLoggerInterface


# save/load parameters using experiment logger
class ExperimentLoggerParameterStore:

    _experiment_logger: ExperimentLoggerInterface

    def __init__(self, experiment_logger: ExperimentLoggerInterface):
        self._experiment_logger = experiment_logger

    def save_parameters(self, parameters: dict) -> None:
        self._experiment_logger.log_dict('parameters',
                                         parameters,
                                         log_as_str=True,
                                         log_as_pickle=True)

    def load_parameters(self) -> dict:
        params = None
        with tempfile.TemporaryDirectory() as tmp:
            target_path = os.path.join(tmp, 'parameters.pkl')
            self._experiment_logger.query_file('parameters_pkl',
                                               destination_path=target_path)

            # deserialize
            with open(target_path, 'rb') as f:
                params = pickle.load(f)

        return params
