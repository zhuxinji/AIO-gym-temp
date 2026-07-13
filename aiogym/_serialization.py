"""JSON conversion and writing shared by benchmark orchestration modules."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def jsonable(value):
    if hasattr(value, "metadata") and callable(value.metadata):
        return jsonable(value.metadata())
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist") and callable(value.tolist):
        return jsonable(value.tolist())
    if hasattr(value, "item") and callable(value.item):
        try:
            return jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path: str | Path, data: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as stream:
        json.dump(jsonable(data), stream, indent=2)
        stream.write("\n")
