from dataclasses import dataclass


@dataclass
class AirVelocityFieldConfigBase:
    _target_: str = 'thermal.api.AirVelocityFieldInterface'
