"""Runtime configuration: provider keys and repository paths.

Keys come from the environment / .env; paths are derived from the repo root
(discovered from the working directory, see ``_discover_repo_root`` — so a
non-editable install does not write outputs inside site-packages). The output
dirs (transcripts, results) are plain fields so tests can repoint them to a tmpdir.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]

# Markers that identify the benchmark repo root (any one is enough). `tasks/` +
# `ground_truth/` is the data layout; `pyproject.toml` covers a fresh checkout.
_ROOT_MARKERS = ("tasks", "ground_truth", "pyproject.toml")


def _discover_repo_root() -> Path:
    """Locate the benchmark repo root.

    When hallmark is installed non-editably, the package lives under
    ``site-packages/`` and ``Path(__file__).parents[1]`` points there — so
    ``results/``/``transcripts/`` would be written inside the venv. Instead we
    honor ``HALLMARK_REPO_ROOT`` if set, then walk up from the current working
    directory looking for the repo markers, and only fall back to the package
    location (the editable-install case, where it is correct).
    """
    env = os.environ.get("HALLMARK_REPO_ROOT")
    if env:
        return Path(env).expanduser().resolve()

    for base in (Path.cwd(), *Path.cwd().parents):
        if any((base / marker).exists() for marker in _ROOT_MARKERS):
            return base

    return _PACKAGE_ROOT


_REPO_ROOT = _discover_repo_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider credentials (optional — only needed for the adapters in use).
    openai_api_key: str | None = Field(default=None)
    openrouter_api_key: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)
    google_api_key: str | None = Field(default=None)
    # Config for the free-text extractor's model call (graders/judge.py is the
    # shared transport; no grader in this release scores with an LLM). Declared
    # here so pydantic-settings loads them from .env — undeclared keys are
    # dropped (extra="ignore") and never reach os.environ.
    judge_api_key: str | None = Field(default=None)
    judge_model: str | None = Field(default=None)
    judge_api_base_url: str | None = Field(default=None)
    # Semicolon-separated list of judge models for the "judge board" panel
    # (e.g. "anthropic/claude-opus-4-7;openai/gpt-4o"). Activated per-run by the
    # grade `--judge-board` flag; all models share judge_api_key/base_url.
    judge_list: str | None = Field(default=None)

    # Per-call timeout (seconds) and retry count for the LLM judge client. The
    # openai SDK otherwise defaults to a 600s read timeout + 2 retries, so one
    # stalled provider can freeze grading for ~30 min. (env JUDGE_TIMEOUT / JUDGE_MAX_RETRIES)
    judge_timeout: float = 120.0
    judge_max_retries: int = 1

    # Watchdog: max wall-clock (seconds) for a single grader before the runner
    # cancels it and records a timeout error (excluded from the score, like any
    # grader error). Backstop for hangs the client timeout doesn't catch; keep it
    # above judge_timeout * (judge_max_retries + 1). (env GRADER_TIMEOUT)
    grader_timeout: float = 300.0

    # Default number of seeds per task (can be overridden per task).
    default_runs: int = 3

    repo_root: Path = _REPO_ROOT

    # Output dirs — assignable (None means "derive from repo_root" in post-init).
    transcripts_dir: Path | None = None
    results_dir: Path | None = None

    def model_post_init(self, __context) -> None:
        if self.transcripts_dir is None:
            self.transcripts_dir = self.repo_root / "transcripts"
        if self.results_dir is None:
            self.results_dir = self.repo_root / "results"

    @property
    def tasks_dir(self) -> Path:
        return self.repo_root / "tasks"

    @property
    def ground_truth_dir(self) -> Path:
        return self.repo_root / "ground_truth"

    @property
    def fixtures_dir(self) -> Path:
        return self.repo_root / "fixtures"

    @property
    def cache_dir(self) -> Path:
        """On-disk cache for external lookups (CrossRef, …)."""
        return self.repo_root / ".cache"

    def api_key(self, env_name: str) -> str | None:
        """Resolve an API key by the env-var name referenced in a system config."""
        return os.environ.get(env_name) or getattr(self, env_name.lower(), None)


settings = Settings()
