from hydra.core.config_store import ConfigStore
from .tianshou_job_base import TianshouJobParametersBase


def register_base_job_config_groups(base_group: str,
                                    config_store: ConfigStore):

    config_store.store(name='base_job', node=TianshouJobParametersBase)
