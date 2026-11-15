"""Charge la configuration YAML depuis config/settings.yaml."""
from pathlib import Path
import yaml

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "settings.yaml"


def load_config() -> dict:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()