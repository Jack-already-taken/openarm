\
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import yaml

def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def none_if_missing(cfg: Dict[str, Any], key: str):
    value = cfg.get(key)
    if value in ["", "null", "None"]:
        return None
    return value
