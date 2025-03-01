from typing import Any
from hydra.utils import instantiate
from omegaconf import OmegaConf


def create_params_from_yaml_string(yaml: str, node_type: Any) -> Any:
    cfg_default = OmegaConf.structured(node_type)
    cfg_override = OmegaConf.create(yaml)

    cfg = OmegaConf.merge(cfg_default, cfg_override)
    result = instantiate(cfg)
    result = OmegaConf.to_object(result)
    return result
