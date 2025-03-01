from dataclasses import dataclass
from hydra.core.config_store import ConfigStore
from .api import EvaluatorConfigBase
from .tianshou_comparison_evaluator import TianshouComparisonEvaluatorParameters


@dataclass(kw_only=True)
class TianshouComparisonEvaluatorConfig(EvaluatorConfigBase,
                                        TianshouComparisonEvaluatorParameters):
    _target_: str = 'trainer.tianshou_evaluator.TianshouComparisonEvaluator'


def register_tianshou_evaluator_config_groups(base_group: str,
                                              config_store: ConfigStore):

    config_store.store(group=f'{base_group}',
                       name='comparison',
                       node=TianshouComparisonEvaluatorConfig)
