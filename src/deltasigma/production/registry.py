"""Minimal file-based model registry: versioned artifacts plus metadata.

Each ``save`` writes ``<root>/<name>/v<N>/`` containing the serialised model and a
``metadata.json`` with the training configuration, validation metrics, a config
hash and the git commit, so any production decision can be traced back to the
exact model and data that produced it.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import subprocess
import time
from pathlib import Path
from typing import Any

DEFAULT_ROOT = Path("artifacts") / "registry"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                             text=True, timeout=5, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:12]


class ModelRegistry:
    def __init__(self, root: str | Path = DEFAULT_ROOT) -> None:
        self.root = Path(root)

    def versions(self, name: str) -> list[int]:
        d = self.root / name
        if not d.exists():
            return []
        return sorted(int(p.name[1:]) for p in d.iterdir() if p.is_dir() and p.name[1:].isdigit())

    def save(self, name: str, model: Any, config: dict[str, Any],
             metrics: dict[str, float] | None = None) -> Path:
        version = (self.versions(name) or [0])[-1] + 1
        path = self.root / name / f"v{version}"
        path.mkdir(parents=True, exist_ok=False)
        with open(path / "model.pkl", "wb") as f:
            pickle.dump(model, f)
        meta = {
            "name": name,
            "version": version,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "git_commit": _git_commit(),
            "config": config,
            "config_hash": config_hash(config),
            "metrics": metrics or {},
        }
        (path / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
        return path

    def find(self, name: str, config: dict[str, Any]) -> int | None:
        """Latest version trained with exactly this configuration (cache lookup)."""
        h = config_hash(config)
        for v in reversed(self.versions(name)):
            if self.metadata(name, v)["config_hash"] == h:
                return v
        return None

    def metadata(self, name: str, version: int | None = None) -> dict[str, Any]:
        version = version or self.versions(name)[-1]
        return json.loads((self.root / name / f"v{version}" / "metadata.json").read_text())

    def load(self, name: str, version: int | None = None) -> Any:
        """Load a model. Only use with registries you control: pickle executes code."""
        if not self.versions(name):
            raise FileNotFoundError(f"no versions of {name!r} in {self.root}")
        version = version or self.versions(name)[-1]
        with open(self.root / name / f"v{version}" / "model.pkl", "rb") as f:
            return pickle.load(f)
