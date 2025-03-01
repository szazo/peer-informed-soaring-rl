from dataclasses import dataclass


@dataclass
class TianshouModelConfigBase:
    _target_: str = 'tianshou.policy.BasePolicy'
    _partial_: bool = True
