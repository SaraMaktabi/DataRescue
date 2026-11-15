"""Résout les chemins selon l'environnement (local Windows ou conteneur Docker)."""
import os
from pathlib import Path


def _base() -> Path:
    if os.environ.get("IN_DOCKER") == "true":
        return Path("/opt/DataRescue")
    return Path(".")


def data_raw(*parts: str) -> Path:
    return _base() / "data" / "raw" / Path(*parts)


def data_delta(*parts: str) -> Path:
    return _base() / "data" / "delta" / Path(*parts)