"""Append-only event records and a reproducible run configuration snapshot."""

from dataclasses import asdict, is_dataclass
from enum import Enum
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np


def json_value(value: Any) -> Any:
    """Convert known structured values, rejecting non-finite JSON numbers."""
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


class RunRecorder:
    """Keep each run in its own directory; never overwrite a previous run."""

    def __init__(self, directory: str | Path, metadata: dict[str, Any]):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self._events = (self.directory / "events.jsonl").open("x", encoding="utf-8")
        self._sequence = 0
        self.write_json("run.json", {"schema_version": 1, "python": platform.python_version(), **metadata})

    def write_json(self, filename: str, value: Any) -> None:
        """Write a JSON artifact within this run directory."""
        path = self.directory / filename
        if path.parent != self.directory:
            raise ValueError("Artifacts must be direct children of the run directory")
        with path.open("x", encoding="utf-8") as output:
            json.dump(json_value(value), output, indent=2, allow_nan=False)
            output.write("\n")

    def event(self, stage: str, **fields: Any) -> None:
        """Record ordered wall-clock events; sensor time is a separate field."""
        record = {"sequence": self._sequence, "wall_time_ns": time.time_ns(), "stage": stage, **fields}
        self._events.write(json.dumps(json_value(record), allow_nan=False) + "\n")
        self._events.flush()
        self._sequence += 1

    def image(self, image: np.ndarray, name: str) -> str:
        """Save a BGR/gray evidence image as PNG without lossy compression."""
        from PIL import Image
        path = self.directory / name
        if path.parent != self.directory or path.exists():
            raise ValueError("Invalid or existing image artifact")
        rgb = image[:, :, ::-1] if image.ndim == 3 else image
        Image.fromarray(rgb).save(path)
        return name

    def close(self) -> None:
        self._events.close()

    def __enter__(self) -> "RunRecorder":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
