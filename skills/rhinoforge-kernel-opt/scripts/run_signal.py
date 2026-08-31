#!/usr/bin/env python3
"""Collect and journal one contract-derived, non-promotable signal receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from board_lease import BoardLease
from campaign_common import (
    FUSION_OPS,
    ValidationError,
    content_tree_sha256,
    contract_sha256,
    load_contract,
    load_json,
    validate_result,
)
from record_iteration import record_iteration


MAX_LOG_TAIL_BYTES = 4096


def _git(root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValidationError("candidate Git provenance is unavailable") from error
    if completed.returncode != 0:
        raise ValidationError("candidate Git provenance check failed")
    return completed.stdout.strip()


def _candidate_identity(
    contract_path: Path, contract: dict[str, Any]
) -> tuple[Path, str, str]:
    configured = Path(contract["campaign"]["candidate_root"])
    if not configured.is_absolute():
        configured = contract_path.parent / configured
    if configured.is_symlink():
        raise ValidationError("candidate_root must not be a symbolic link")
    try:
        root = configured.resolve(strict=True)
    except OSError as error:
        raise ValidationError("candidate_root is unavailable") from error
    if not root.is_dir():
        raise ValidationError("candidate_root must be a directory")
    try:
        top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(
            strict=True
        )
    except OSError as error:
        raise ValidationError("candidate Git top-level is unavailable") from error
    if root != top_level:
        raise ValidationError(
            "candidate_root must be its isolated Git top-level"
        )
    head = _git(root, "rev-parse", "--verify", "HEAD")
    lineage = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(lineage) != 2 or lineage[0] != head:
        raise ValidationError("candidate HEAD must have exactly one parent")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValidationError("candidate Git worktree must be clean before signal")
    return root, head, lineage[1]


def _resolve_adapter(contract_path: Path, contract: dict[str, Any]) -> Path:
    configured = Path(contract["commands"]["benchmark_adapter"])
    if not configured.is_absolute():
        configured = contract_path.parent / configured
    if configured.is_symlink():
        raise ValidationError("benchmark adapter must not be a symbolic link")
    try:
        return configured.resolve(strict=True)
    except OSError as error:
        raise ValidationError("benchmark adapter is unavailable") from error


def _adapter_identity(path: Path, expected_sha256: str) -> tuple[int, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValidationError("benchmark adapter is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        try:
            current = os.lstat(path)
        except OSError as error:
            raise ValidationError(
                "benchmark adapter pathname is unavailable"
            ) from error
        if (
            not stat.S_ISREG(opened.st_mode)
            or not stat.S_ISREG(current.st_mode)
            or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
        ):
            raise ValidationError("benchmark adapter identity is unsafe")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
    finally:
        os.close(descriptor)
    if digest.hexdigest() != expected_sha256:
        raise ValidationError("benchmark adapter SHA-256 mismatches the contract")
    return opened.st_dev, opened.st_ino, opened.st_size


def _outside_candidate(path: Path, candidate_root: Path, label: str) -> None:
    resolved = path.resolve(strict=False)
    if resolved == candidate_root or candidate_root in resolved.parents:
        raise ValidationError(f"{label} must live outside the candidate repository")


def _reference_identity(
    contract_path: Path, contract: dict[str, Any]
) -> tuple[Path, str]:
    configured = Path(contract["campaign"]["reference_root"])
    if not configured.is_absolute():
        configured = contract_path.parent / configured
    if configured.is_symlink():
        raise ValidationError("reference_root must not be a symbolic link")
    root = configured.resolve(strict=True)
    observed = content_tree_sha256(root)
    if observed != contract["campaign"]["reference_tree_sha256"]:
        raise ValidationError("reference tree SHA-256 mismatches the contract")
    return root, observed


def _proc_identity(pid: int) -> dict[str, Any]:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip().lower()
        uuid.UUID(boot_id)
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").strip()
        closing = stat_text.rfind(")")
        suffix = stat_text[closing + 1 :].split()
        start_ticks = int(suffix[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ValidationError("could not observe benchmark child process") from error
    if closing < 0 or start_ticks <= 0:
        raise ValidationError("benchmark child process identity is malformed")
    return {"boot_id": boot_id, "pid": pid, "start_ticks": start_ticks}


def _build_bench_command(
    *,
    contract: dict[str, Any],
    adapter: Path,
    candidate_id: str,
    candidate_commit: str,
    parent_commit: str,
    contract_hash: str,
    result_path: Path,
    device: str,
    seed: int,
) -> list[str]:
    measurement = contract["measurement"]
    gates = contract["gates"]
    # bench.py emits two samples per paired-order block.  This meets both its
    # two-block minimum and the contract's requested per-arm sample count.
    repeats = max(2, math.ceil(int(measurement["samples"]) / 2))
    command = [
        sys.executable,
        str(SCRIPT_DIR / "bench.py"),
        "--adapter",
        str(adapter),
        "--profile",
        str(contract["campaign"]["op"]),
        "--device",
        device,
        "--seed",
        str(seed),
        "--warmup",
        str(measurement["warmup"]),
        "--repeats",
        str(repeats),
        "--atol",
        str(gates["max_abs"]),
        "--rtol",
        str(gates["max_rel"]),
        "--anchor-atol",
        str(gates["anchor_max_abs"]),
        "--anchor-rtol",
        str(gates["anchor_max_rel"]),
        "--max-cv",
        str(gates["max_cv"]),
        "--candidate-id",
        candidate_id,
        "--candidate-commit",
        candidate_commit,
        "--parent-commit",
        parent_commit,
        "--contract-sha256",
        contract_hash,
        "--verdict",
        "signal",
        "--output",
        str(result_path),
    ]
    for workload in contract["workloads"]:
        command.extend(("--workload", str(workload["id"])))
    if contract["campaign"]["op"] in FUSION_OPS:
        command.extend(
            (
                "--max-epilogue-tax-pct",
                str(gates["max_epilogue_tax_pct"]),
                "--fusion-confidence",
                str(gates["fusion_confidence"]),
                "--fusion-bootstrap-trials",
                str(gates["fusion_bootstrap_trials"]),
                "--fusion-bootstrap-seed",
                str(gates["fusion_bootstrap_seed"]),
            )
        )
    return command


def _log_tail(handle: Any) -> str:
    try:
        handle.flush()
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - MAX_LOG_TAIL_BYTES))
        return handle.read().decode("utf-8", errors="replace").strip()
    except (OSError, ValueError):
        return ""


def _terminate_child(child: subprocess.Popen[Any]) -> None:
    if child.poll() is not None:
        return
    child.terminate()
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _run_bench_child(
    command: list[str],
    *,
    cwd: Path,
    board_lease: BoardLease,
    timeout_s: float,
) -> tuple[int, dict[str, Any], str]:
    with tempfile.TemporaryFile(mode="w+b") as stdout_log, tempfile.TemporaryFile(
        mode="w+b"
    ) as stderr_log:
        try:
            child = subprocess.Popen(
                command,
                cwd=cwd,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                stdout=stdout_log,
                stderr=stderr_log,
            )
        except OSError as error:
            raise ValidationError("could not start benchmark child") from error
        try:
            observed = _proc_identity(child.pid)
            child.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired as error:
            _terminate_child(child)
            raise ValidationError("benchmark child timed out") from error
        except BaseException:
            _terminate_child(child)
            raise
        board_lease.check_integrity()
        detail = _log_tail(stderr_log) or _log_tail(stdout_log)
        return int(child.returncode), observed, detail


def _signal_gates_passed(
    record: dict[str, Any], contract: dict[str, Any]
) -> bool:
    expected_ids = [str(item["id"]) for item in contract["workloads"]]
    workloads = record.get("workloads")
    metric = record.get("metric")
    top_gates = record.get("hard_gates")
    return bool(
        record.get("verdict") == "signal"
        and record.get("preflight_sha256") == "0" * 64
        and isinstance(workloads, list)
        and [item.get("id") for item in workloads if isinstance(item, dict)]
        == expected_ids
        and all(
            isinstance(item, dict)
            and item.get("eligible") is True
            and isinstance(item.get("hard_gates"), dict)
            and bool(item["hard_gates"])
            and all(value is True for value in item["hard_gates"].values())
            for item in workloads
        )
        and isinstance(metric, dict)
        and isinstance(metric.get("value"), (int, float))
        and not isinstance(metric.get("value"), bool)
        and math.isfinite(float(metric["value"]))
        and isinstance(top_gates, dict)
        and bool(top_gates)
        and all(value is True for value in top_gates.values())
    )


def _lease_report(lease: BoardLease) -> dict[str, Any]:
    return {
        "lease_id": lease.lease_id,
        "start_sequence": lease.start_sequence,
        "finish_sequence": lease.finish_sequence,
        "previous_journal_head_sha256": lease.previous_head_hash,
        "start_record_sha256": lease.start_record_sha256,
        "finish_record_sha256": lease.finish_record_sha256,
        "final_journal_head_sha256": lease.head_hash,
        "started_monotonic_ns": lease.started_monotonic_ns,
        "finished_monotonic_ns": lease.finished_monotonic_ns,
    }


def collect_signal(
    contract_path: Path,
    *,
    candidate_id: str,
    output_dir: Path,
    device: str = "rpu",
    seed: int = 0,
    lease_timeout_s: float = 0.0,
    benchmark_timeout_s: float = 3600.0,
) -> dict[str, Any]:
    if device != "rpu":
        raise ValidationError("signal device must be exactly 'rpu'")
    if not math.isfinite(lease_timeout_s) or lease_timeout_s < 0:
        raise ValidationError("lease timeout must be finite and non-negative")
    if not math.isfinite(benchmark_timeout_s) or benchmark_timeout_s <= 0:
        raise ValidationError("benchmark timeout must be finite and positive")
    supplied_contract = contract_path.expanduser()
    if supplied_contract.is_symlink():
        raise ValidationError("contract must be a regular non-symlink file")
    contract_path = supplied_contract.resolve(strict=True)
    if not contract_path.is_file():
        raise ValidationError("contract must be a regular non-symlink file")
    initial_contract_hash = contract_sha256(contract_path)
    contract = load_contract(contract_path)
    if contract_sha256(contract_path) != initial_contract_hash:
        raise ValidationError("contract changed while it was being loaded")

    candidate_root, candidate_commit, parent_commit = _candidate_identity(
        contract_path, contract
    )
    reference_root, reference_tree_hash = _reference_identity(
        contract_path, contract
    )
    adapter = _resolve_adapter(contract_path, contract)
    expected_adapter_hash = contract["commands"]["benchmark_adapter_sha256"]
    adapter_identity = _adapter_identity(adapter, expected_adapter_hash)
    _outside_candidate(contract_path, candidate_root, "campaign contract")
    _outside_candidate(adapter, candidate_root, "benchmark adapter")
    _outside_candidate(reference_root, candidate_root, "reference root")
    _outside_candidate(
        output_dir.expanduser().resolve(strict=False), candidate_root, "output"
    )

    with tempfile.TemporaryDirectory(prefix="rhinoforge-signal-") as temporary_name:
        result_path = Path(temporary_name) / "signal-result.json"
        command = _build_bench_command(
            contract=contract,
            adapter=adapter,
            candidate_id=candidate_id,
            candidate_commit=candidate_commit,
            parent_commit=parent_commit,
            contract_hash=initial_contract_hash,
            result_path=result_path,
            device=device,
            seed=seed,
        )
        board_lease = BoardLease(lease_timeout_s, command)
        with board_lease:
            try:
                bench_exit_code, observed, detail = _run_bench_child(
                    command,
                    cwd=contract_path.parent,
                    board_lease=board_lease,
                    timeout_s=benchmark_timeout_s,
                )
                if bench_exit_code not in (0, 2):
                    raise ValidationError(
                        "benchmark child failed"
                        + (f": {detail[-512:]}" if detail else "")
                    )
                if result_path.is_symlink() or not result_path.is_file():
                    raise ValidationError("benchmark child did not create a result")
                raw_result = load_json(result_path)
                if not isinstance(raw_result, dict):
                    raise ValidationError("benchmark child result is not an object")
                process = raw_result.get("process")
                if (
                    not isinstance(process, dict)
                    or process.get("attested") is not False
                ):
                    raise ValidationError(
                        "benchmark child result lacks unattested process identity"
                    )
                observed["run_uuid"] = process.get("run_uuid")

                canonical = validate_result(
                    raw_result, contract, initial_contract_hash
                )
                if (
                    canonical.get("candidate_commit") != candidate_commit
                    or canonical.get("parent_commit") != parent_commit
                ):
                    raise ValidationError(
                        "benchmark result candidate identity mismatches"
                    )
                signal_gates_passed = _signal_gates_passed(canonical, contract)
                expected_exit = 0 if signal_gates_passed else 2
                if bench_exit_code != expected_exit:
                    raise ValidationError(
                        "benchmark exit code contradicts signal gates"
                    )

                if contract_sha256(contract_path) != initial_contract_hash:
                    raise ValidationError("contract changed during signal collection")
                if (
                    _adapter_identity(adapter, expected_adapter_hash)
                    != adapter_identity
                ):
                    raise ValidationError(
                        "benchmark adapter changed during signal collection"
                    )
                final_root, final_head, final_parent = _candidate_identity(
                    contract_path, contract
                )
                if (
                    final_root != candidate_root
                    or final_head != candidate_commit
                    or final_parent != parent_commit
                ):
                    raise ValidationError(
                        "candidate identity changed during signal collection"
                    )
                final_reference_root, final_reference_hash = _reference_identity(
                    contract_path, contract
                )
                if (
                    final_reference_root != reference_root
                    or final_reference_hash != reference_tree_hash
                ):
                    raise ValidationError(
                        "reference identity changed during signal collection"
                    )

                receipt = record_iteration(
                    contract_path,
                    result_path,
                    output_dir,
                    observed_process=observed,
                )
                board_lease.finish(bench_exit_code)
            except Exception:
                if not board_lease.finished:
                    board_lease.finish(2)
                raise

    return {
        "schema_version": 1,
        "ok": signal_gates_passed,
        "kind": "rhinoforge-signal-session",
        "collection_ok": True,
        "signal_gates_passed": signal_gates_passed,
        "bench_exit_code": bench_exit_code,
        "release_eligible": False,
        "release_blocker": "signal receipts are never promotable",
        "contract_sha256": initial_contract_hash,
        "adapter_sha256": expected_adapter_hash,
        "reference_tree_sha256": reference_tree_hash,
        "candidate_commit": candidate_commit,
        "parent_commit": parent_commit,
        "board_lease": _lease_report(board_lease),
        "recorded_receipt": receipt,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path, help="exact approved TOML contract")
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", choices=("rpu",), default="rpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lease-timeout", type=float, default=0.0)
    parser.add_argument("--benchmark-timeout", type=float, default=3600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = collect_signal(
            args.contract,
            candidate_id=args.candidate_id,
            output_dir=args.output_dir,
            device=args.device,
            seed=args.seed,
            lease_timeout_s=args.lease_timeout,
            benchmark_timeout_s=args.benchmark_timeout,
        )
    except (OSError, subprocess.SubprocessError, ValidationError, ValueError) as error:
        report = {
            "schema_version": 1,
            "ok": False,
            "kind": "rhinoforge-signal-session",
            "collection_ok": False,
            "signal_gates_passed": False,
            "release_eligible": False,
            "error": str(error),
        }
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["collection_ok"] and report["signal_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
