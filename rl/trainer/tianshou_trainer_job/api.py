from dataclasses import dataclass


@dataclass
class TianshouTrainerConfigBase:
    _target_: str = 'tianshou.trainer.BaseTrainer'
    _partial_: bool = True
