from typing import Dict
from dataclasses import dataclass


def type_config(name: str, params: Dict):
    return {"type": name, "params": params}


@dataclass
class TypeConfig:
    type: str
    params: dict | None
