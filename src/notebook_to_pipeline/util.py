"""Small helpers shared by the commands."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def _home() -> str:
    return str(Path.home())


def redact(value: Any) -> Any:
    """Replace the home directory prefix with ~ in every string, so evidence files can be shared."""
    home = _home()
    if not home or home == "/":
        return value
    if isinstance(value, str):
        return value.replace(home, "~")
    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [redact(v) for v in value]
    return value


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(redact(data), indent=2, default=str) + "\n", encoding="utf-8")


def expand(p: str) -> Path:
    return Path(os.path.expanduser(p))
