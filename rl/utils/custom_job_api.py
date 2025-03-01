from typing import Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hydra.core.config_store import ConfigStore


class CustomJobBase(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def run(self, output_dir: str):
        raise Exception(
            'Coding Error: Inherit from CustomJobBase and implement the job logic.'
        )


@dataclass(kw_only=True)
class CustomJobBaseConfig:
    _target_: str = 'utils.custom_job_api.CustomJobBase'


def register_custom_job_config_group(name: str, config_node: Any,
                                     config_store: ConfigStore):
    config_store.store(group='custom_job', name=name, node=config_node)
