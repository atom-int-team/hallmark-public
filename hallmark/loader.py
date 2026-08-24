"""Load and validate Task YAML files, system specs, and ground-truth references.

Replaces the old ``conftest.py`` YAML auto-discovery with an explicit, validated
loader (Task schema in §6.1 lives in models.py).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from .config import settings
from .models import SystemSpec, Task

# --------------------------------------------------------------------------- #
# Tasks                                                                        #
# --------------------------------------------------------------------------- #


def load_task(path: str | Path) -> Task:
    """Read and validate a single Task YAML file."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if raw is None:
        raise ValueError(f"empty task file: {path}")
    return Task.model_validate(raw)


def load_tasks(
    root: str | Path | None = None,
    *,
    split: str | None = None,
    domain: str | None = None,
    task_ids: list[str] | None = None,
) -> list[Task]:
    """Discover tasks under ``root`` (default: <repo>/tasks), optionally filtered.

    ``root`` may be a directory (globbed for ``**/*.yaml``) or a single file.
    """
    root = Path(root) if root is not None else settings.tasks_dir
    files = [root] if root.is_file() else sorted(root.glob("**/*.yaml"))

    tasks: list[Task] = []
    seen: set[str] = set()
    for f in files:
        task = load_task(f)
        if task.id in seen:
            raise ValueError(f"duplicate task id {task.id!r} (in {f})")
        seen.add(task.id)
        if split and task.split.value != split:
            continue
        if domain and task.domain.value != domain:
            continue
        if task_ids and task.id not in task_ids:
            continue
        tasks.append(task)
    return tasks


# --------------------------------------------------------------------------- #
# Systems                                                                      #
# --------------------------------------------------------------------------- #


def load_systems(path: str | Path) -> list[SystemSpec]:
    """Read a systems YAML file: a list under the ``systems:`` key (or a bare list)."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    items = raw.get("systems", raw) if isinstance(raw, dict) else raw
    return [SystemSpec.model_validate(item) for item in items]


# --------------------------------------------------------------------------- #
# Ground truth (§6.1 ground_truth_ref / grader *_ref params)                   #
# --------------------------------------------------------------------------- #


class GroundTruth:
    """Resolve ``"<task_dir>/<key>"`` references into the ground_truth/ tree.

    A ref like ``"lit_mxene_supercap/key_dois"`` is looked up as a file
    ``ground_truth/lit_mxene_supercap/key_dois.{json,yaml,yml,txt,md}``.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self.root = Path(root) if root is not None else settings.ground_truth_dir

    def resolve(self, ref: str):
        """Return the parsed content of a ground-truth ref (dict/list/str)."""
        base = self.root / ref
        for ext in (".json", ".yaml", ".yml"):
            p = base.with_suffix(ext)
            if p.exists():
                text = p.read_text(encoding="utf-8")
                return json.loads(text) if ext == ".json" else yaml.safe_load(text)
        for ext in (".txt", ".md"):
            p = base.with_suffix(ext)
            if p.exists():
                return p.read_text(encoding="utf-8")
        raise FileNotFoundError(f"ground-truth ref not found: {ref} (under {self.root})")
