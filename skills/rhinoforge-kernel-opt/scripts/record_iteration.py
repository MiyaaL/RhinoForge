#!/usr/bin/env python3
"""Validate and atomically journal one kernel campaign iteration."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from campaign_common import (
    ValidationError,
    canonical_json,
    contract_sha256,
    load_contract,
    load_json,
    record_is_promotable,
    strict_json_loads,
    validate_result,
)


MAX_JOURNAL_BYTES = 64 * 1024 * 1024


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


def verify_candidate_git(
    contract_path: Path, contract: dict[str, Any], record: dict[str, Any]
) -> Path:
    """Bind a promotable receipt to the clean, checked-out candidate commit."""

    configured = Path(contract["campaign"]["candidate_root"])
    if not configured.is_absolute():
        configured = contract_path.parent / configured
    if configured.is_symlink():
        raise ValidationError("campaign candidate_root must not be a symbolic link")
    candidate_root = configured.resolve(strict=True)
    if not candidate_root.is_dir():
        raise ValidationError("campaign candidate_root must be a regular directory")
    top_level_text = _git(candidate_root, "rev-parse", "--show-toplevel")
    top_level = Path(top_level_text).resolve(strict=True)
    if candidate_root != top_level:
        raise ValidationError(
            "candidate_root must be its isolated Git top-level; evaluator and "
            "campaign files must live outside the candidate repository"
        )
    head = _git(top_level, "rev-parse", "--verify", "HEAD")
    if head != record["candidate_commit"]:
        raise ValidationError("candidate_commit is not the checked-out Git HEAD")
    lineage = _git(top_level, "rev-list", "--parents", "-n", "1", head).split()
    if len(lineage) != 2 or lineage[0] != head or lineage[1] != record["parent_commit"]:
        raise ValidationError(
            "candidate parent_commit is not the sole parent of candidate_commit"
        )
    if _git(top_level, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValidationError("candidate Git worktree must be clean before measurement")
    return top_level


def _verify_preflight_receipt(
    path: Path, expected_hash: str, expected_contract_hash: str
) -> None:
    if path.is_symlink() or not path.is_file():
        raise ValidationError("preflight receipt must be a regular non-symlink file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValidationError("preflight receipt SHA-256 does not match the result")
    receipt = load_json(path)
    if not isinstance(receipt, dict) or receipt.get("ok") is not True:
        raise ValidationError("bound preflight receipt did not pass")
    if receipt.get("contract_sha256") != expected_contract_hash:
        raise ValidationError("preflight receipt is bound to a different contract")
    checks = receipt.get("checks")
    if not isinstance(checks, list) or any(
        not isinstance(check, dict) or check.get("status") == "fail" for check in checks
    ):
        raise ValidationError("preflight receipt checks are malformed or failed")
    torch_probe = [
        check for check in checks if check.get("name") == "probe.torch_rpu"
    ]
    if len(torch_probe) != 1 or torch_probe[0].get("status") != "pass":
        raise ValidationError("preflight receipt lacks the mandatory FP16 RPU probe")


def _attest_process(raw: dict[str, Any], observed: dict[str, Any]) -> None:
    process = raw.get("process")
    if not isinstance(process, dict):
        raise ValidationError("result lacks process identity")
    if process.get("attested") is not False:
        raise ValidationError("bench result must enter the verifier unattested")
    for key in ("boot_id", "pid", "start_ticks", "run_uuid"):
        if process.get(key) != observed.get(key):
            raise ValidationError(f"observed process identity mismatches {key}")
    process["attested"] = True


def _validate_opened_file(descriptor: int, path: Path, label: str) -> os.stat_result:
    opened = os.fstat(descriptor)
    try:
        current = os.lstat(path)
    except OSError as error:
        raise ValidationError(f"{label} pathname is unavailable") from error
    if (
        not stat.S_ISREG(opened.st_mode)
        or opened.st_uid != os.geteuid()
        or opened.st_nlink != 1
        or stat.S_IMODE(opened.st_mode) & 0o022
        or (opened.st_dev, opened.st_ino) != (current.st_dev, current.st_ino)
    ):
        raise ValidationError(f"{label} identity or mode is unsafe")
    return opened


def _read_candidate_ids(descriptor: int, path: Path) -> set[str]:
    opened = _validate_opened_file(descriptor, path, "results.jsonl")
    try:
        if opened.st_size > MAX_JOURNAL_BYTES:
            raise ValidationError("results.jsonl exceeds 64 MiB")
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ValidationError("results.jsonl changed while reading")
            chunks.append(block)
            remaining -= len(block)
        text = b"".join(chunks).decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValidationError("results.jsonl is unreadable") from error
    result: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValidationError(f"results.jsonl has a blank line at {line_number}")
        try:
            record = strict_json_loads(line)
        except (ValidationError, json.JSONDecodeError, RecursionError) as error:
            raise ValidationError(
                f"results.jsonl has invalid JSON at line {line_number}"
            ) from error
        if not isinstance(record, dict):
            raise ValidationError(f"results.jsonl line {line_number} is not an object")
        candidate_id = record.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValidationError(
                f"results.jsonl line {line_number} lacks candidate_id"
            )
        if candidate_id in result:
            raise ValidationError("results.jsonl already contains duplicate ids")
        result.add(candidate_id)
    return result


def _markdown(record: dict[str, Any]) -> str:
    metric = record["metric"]
    gates_passed = all(record["hard_gates"].values()) and all(
        workload["eligible"] and all(workload["hard_gates"].values())
        for workload in record["workloads"]
    )
    metric_text = (
        f"{metric['value']:.9g}" if metric["value"] is not None else "unavailable"
    )
    lines = [
        f"# Iteration {record['candidate_id']}",
        "",
        f"- Candidate commit: `{record['candidate_commit']}`",
        f"- Parent commit: `{record['parent_commit']}`",
        f"- Contract SHA-256: `{record['contract_sha256']}`",
        f"- Preflight SHA-256: `{record['preflight_sha256']}`",
        f"- Process run UUID: `{record['process']['run_uuid']}`",
        f"- Process attested: `{str(record['process']['attested']).lower()}`",
        f"- Verdict: `{record['verdict']}`",
        f"- All hard gates: `{'pass' if gates_passed else 'fail'}`",
        f"- {metric['name']}: `{metric_text}` ({metric['direction']})",
        "",
        "| Workload | Candidate p50 (ns) | Reference p50 (ns) | Eligible |",
        "|---|---:|---:|:---:|",
    ]
    for workload in record["workloads"]:
        summary = workload["summary"]
        candidate_p50 = summary["candidate_p50_ns"]
        reference_p50 = summary["reference_p50_ns"]
        candidate_text = (
            f"{candidate_p50:.9g}" if candidate_p50 is not None else "unavailable"
        )
        reference_text = (
            f"{reference_p50:.9g}" if reference_p50 is not None else "unavailable"
        )
        lines.append(
            f"| `{workload['id']}` | {candidate_text} | "
            f"{reference_text} | "
            f"{'yes' if workload['eligible'] else 'no'} |"
        )
    lines.append("")
    return "\n".join(lines)


def _write_exclusive(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def record_iteration(
    contract_path: Path,
    result_path: Path,
    output_dir: Path,
    *,
    observed_process: dict[str, Any] | None = None,
    preflight_receipt: Path | None = None,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    contract_hash = contract_sha256(contract_path)
    raw_result = load_json(result_path)
    if not isinstance(raw_result, dict):
        raise ValidationError("result must be a JSON object")
    if observed_process is not None:
        _attest_process(raw_result, observed_process)
    elif isinstance(raw_result.get("process"), dict) and raw_result["process"].get(
        "attested"
    ) is True:
        raise ValidationError("self-attested process receipt is forbidden")
    declared_preflight = raw_result.get("preflight_sha256")
    if preflight_receipt is not None:
        if not isinstance(declared_preflight, str):
            raise ValidationError("result lacks preflight_sha256")
        _verify_preflight_receipt(
            preflight_receipt, declared_preflight, contract_hash
        )
    elif declared_preflight not in (None, "0" * 64):
        raise ValidationError("nonzero preflight hash requires verifier receipt input")
    record = validate_result(raw_result, contract, contract_hash)
    promotable = record_is_promotable(
        record,
        contract["measurement"]["metric"],
        contract["measurement"]["direction"],
        contract_hash,
    )
    if promotable:
        if observed_process is None or preflight_receipt is None:
            raise ValidationError(
                "promotable full receipt requires verifier process and preflight evidence"
            )
        verify_candidate_git(contract_path, contract, record)

    output_dir.mkdir(parents=True, exist_ok=True)
    if output_dir.is_symlink() or not output_dir.is_dir():
        raise ValidationError("output directory must be a regular directory")
    entries_dir = output_dir / "iterations"
    entries_dir.mkdir(exist_ok=True)
    if entries_dir.is_symlink() or not entries_dir.is_dir():
        raise ValidationError("iterations directory must be a regular directory")

    results_path = output_dir / "results.jsonl"
    markdown_path = entries_dir / f"{record['candidate_id']}.md"
    lock_path = output_dir / ".record.lock"
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    lock_descriptor = os.open(
        lock_path, os.O_RDWR | os.O_CREAT | nofollow, 0o600
    )
    try:
        _validate_opened_file(lock_descriptor, lock_path, "record lock")
        fcntl.flock(lock_descriptor, fcntl.LOCK_EX)
        journal_descriptor = os.open(
            results_path,
            os.O_RDWR | os.O_CREAT | os.O_APPEND | nofollow,
            0o644,
        )
        try:
            _validate_opened_file(journal_descriptor, results_path, "results.jsonl")
            candidate_ids = _read_candidate_ids(journal_descriptor, results_path)
            if record["candidate_id"] in candidate_ids or markdown_path.exists():
                raise ValidationError(
                    f"candidate_id already exists: {record['candidate_id']}"
                )

            markdown_bytes = _markdown(record).encode("utf-8")
            _write_exclusive(markdown_path, markdown_bytes)
            try:
                journal_bytes = (canonical_json(record) + "\n").encode("utf-8")
                # One append write under the campaign lock prevents interleaved
                # records among cooperating writers.
                written = os.write(journal_descriptor, journal_bytes)
                if written != len(journal_bytes):
                    raise OSError("short journal append")
                os.fsync(journal_descriptor)
            except Exception:
                # This file was created exclusively by this invocation and has
                # not been published in the journal.
                try:
                    markdown_path.unlink()
                except OSError:
                    pass
                raise
        finally:
            os.close(journal_descriptor)
    finally:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_UN)
        finally:
            os.close(lock_descriptor)

    return {
        "schema_version": 1,
        "ok": True,
        "candidate_id": record["candidate_id"],
        "candidate_commit": record["candidate_commit"],
        "record_sha256": hashlib.sha256(
            canonical_json(record).encode("utf-8")
        ).hexdigest(),
        "journal": results_path.name,
        "markdown_entry": f"iterations/{markdown_path.name}",
        "local_gate_eligible": promotable,
        "release_eligible": False,
        "release_blocker": "external-authority DSSE release proof required",
        "required_process_count": contract["measurement"]["verdict_processes"],
        "promotion_requires_signed_select_best": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a schema-v1 result and append it to results.jsonl.",
        epilog=(
            "Result root keys: schema_version, candidate_id, candidate_commit, "
            "parent_commit, contract_sha256, verdict, metric, workloads, "
            "hard_gates, and optional artifacts. Each workload contains id, "
            "raw_samples, summary, correctness, lifecycle, hard_gates, eligible."
        ),
    )
    parser.add_argument("contract", type=Path, help="exact approved TOML contract")
    parser.add_argument("result", type=Path, help="untrusted iteration result JSON")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="campaign journal directory",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = record_iteration(args.contract, args.result, args.output_dir)
    except (OSError, ValidationError) as error:
        print(
            json.dumps(
                {"schema_version": 1, "ok": False, "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
