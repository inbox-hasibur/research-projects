#!/usr/bin/env python3
"""
Copy all .py files from each Project_* folder into github_publish/ (mirror paths).
Skips venv, data, __pycache__, etc. — no .env copy (those live outside or are not .py).

New project = new folder named Project_Something under repo root → next sync picks it up
(no manual list edit).

Run from repo root:  python tools/sync_github_publish.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PUBLISH_ROOT = REPO_ROOT / "github_publish"

# If any path segment matches (case-insensitive), skip that file
SKIP_DIR_NAMES = frozenset(
    {
        "__pycache__",
        ".git",
        ".venv",
        "venv",
        "env",
        ".mypy_cache",
        ".pytest_cache",
        "node_modules",
        "site-packages",
        "data",
        "datasets",
        "dataset",
        "raw",
        "output",
        "outputs",
        "logs",
        "wandb",
        "runs",
    }
)


def _should_skip(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return not SKIP_DIR_NAMES.isdisjoint(parts)


def discover_project_dirs() -> list[Path]:
    out: list[Path] = []
    for p in sorted(REPO_ROOT.iterdir()):
        if not p.is_dir():
            continue
        if not p.name.startswith("Project_"):
            continue
        if p.name == "github_publish" or p.name == "tools":
            continue
        out.append(p)
    return out


def iter_py_files(project_dir: Path) -> list[Path]:
    files: list[Path] = []
    for f in project_dir.rglob("*.py"):
        if _should_skip(f):
            continue
        files.append(f)
    return sorted(files)


def main() -> int:
    projects = discover_project_dirs()
    if not projects:
        print("No Project_* directories found under repo root.", file=sys.stderr)
        return 1

    copied = 0
    for proj in projects:
        rel_proj = proj.relative_to(REPO_ROOT)
        dst_base = PUBLISH_ROOT / rel_proj
        dst_base.mkdir(parents=True, exist_ok=True)
        for src in iter_py_files(proj):
            rel = src.relative_to(proj)
            dst = dst_base / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied += 1
            print(f"OK  {dst.relative_to(REPO_ROOT)}")

    print("-" * 60)
    print(f"Synced {copied} .py file(s) from {len(projects)} project folder(s) -> {PUBLISH_ROOT.name}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
