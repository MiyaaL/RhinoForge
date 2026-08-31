#!/usr/bin/env python3
"""Create a non-overwriting RhinoForge kernel campaign workspace."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PROFILE_FILES = {
    "gemm": "contract.toml",
    "gemm+silu_mul": "gemm-silu-mul.toml",
    "gemm+add": "gemm-add.toml",
    "gemm+rope": "gemm-rope.toml",
    "rmsnorm+quant_mxfp8": "rmsnorm-quant-mxfp8.toml",
}

ITERATIONS = """# Iterations

This log is generated from validated machine-readable receipts. Do not rewrite
past entries or record a measurement from memory.
"""

GITIGNORE = """# Sensitive or high-volume run artifacts stay outside Git.
runs/*
!runs/.gitignore
profiles/raw/*
!profiles/raw/.gitignore
results.jsonl
iterations/
BEST.json
_bench_output.txt
*.trace.json
*.trace.json.gz
*.pt
*.bin
"""


def _safe_workspace(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    forbidden = {Path("/"), Path.home().resolve()}
    if resolved in forbidden:
        raise ValueError(f"refusing broad campaign workspace: {resolved}")
    return resolved


def _write_new(path: Path, text: str) -> None:
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise RuntimeError(f"refusing to overwrite existing file: {path}") from exc


def bootstrap(workspace: Path, profile: str, git_init: bool) -> None:
    workspace = _safe_workspace(workspace)
    skill_root = Path(__file__).resolve().parents[1]
    source = skill_root / "assets" / "campaign" / PROFILE_FILES[profile]
    if not source.is_file():
        raise RuntimeError(f"skill profile asset is missing: {source}")

    if git_init and (workspace / "solution" / ".git").exists():
        raise RuntimeError(
            f"candidate is already a Git repository: {workspace / 'solution'}"
        )

    workspace.mkdir(parents=True, exist_ok=True)
    conflicts = [
        workspace / "contract.toml",
        workspace / "ITERATIONS.md",
        workspace / "results.jsonl",
        workspace / ".gitignore",
        workspace / "solution" / ".gitkeep",
        workspace / "reference" / ".gitkeep",
        workspace / "runs" / ".gitignore",
        workspace / "profiles" / "raw" / ".gitignore",
    ]
    existing = [str(path) for path in conflicts if path.exists()]
    if existing:
        raise RuntimeError("refusing to overwrite campaign files: " + ", ".join(existing))

    for relative in ("solution", "reference", "runs", "profiles/raw"):
        (workspace / relative).mkdir(parents=True, exist_ok=True)

    shutil.copy2(source, workspace / "contract.toml")
    _write_new(workspace / "ITERATIONS.md", ITERATIONS)
    _write_new(workspace / "results.jsonl", "")
    _write_new(workspace / "solution" / ".gitkeep", "")
    _write_new(workspace / "reference" / ".gitkeep", "")
    _write_new(workspace / "runs" / ".gitignore", "*\n!.gitignore\n")
    _write_new(workspace / "profiles" / "raw" / ".gitignore", "*\n!.gitignore\n")

    _write_new(workspace / ".gitignore", GITIGNORE)

    if git_init:
        subprocess.run(
            [
                "git",
                "init",
                "--initial-branch",
                "main",
                str(workspace / "solution"),
            ],
            check=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_FILES), required=True)
    parser.add_argument(
        "--git-init",
        action="store_true",
        help="initialize a new Git repository after creating the workspace",
    )
    args = parser.parse_args()

    try:
        bootstrap(args.workspace, args.profile, args.git_init)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(2, f"bootstrap failed: {exc}\n")
    print(f"campaign initialized: {_safe_workspace(args.workspace)}")
    print("contract state is draft; review every field before preflight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
