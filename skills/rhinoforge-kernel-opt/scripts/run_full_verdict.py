#!/usr/bin/env python3
"""Collect a fail-closed local RPU diagnostic; never issue release attestation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from campaign_common import (
    FUSION_OPS,
    SHA256_RE,
    ValidationError,
    canonical_json,
    content_tree_sha256,
    contract_sha256,
    load_contract,
    load_json,
)
from board_lease import BoardLease
from preflight import run_preflight
from record_iteration import _git, _verify_preflight_receipt, record_iteration
from summarize_hwperf import _atomic_write as _atomic_write_summary
from summarize_hwperf import summarize_trace


RPU_DEVICE_EVENT_CATEGORY = "rpu_device_program"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_identity(path: Path, expected_sha256: str) -> tuple[int, int, int]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValidationError("benchmark adapter is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        current = os.lstat(path)
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


def _candidate_root(contract_path: Path, contract: dict[str, Any]) -> Path:
    configured = Path(contract["campaign"]["candidate_root"])
    if not configured.is_absolute():
        configured = contract_path.parent / configured
    if configured.is_symlink():
        raise ValidationError("candidate_root must not be a symbolic link")
    root = configured.resolve(strict=True)
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve(
        strict=True
    )
    if root != top_level:
        raise ValidationError(
            "candidate_root must be its isolated Git top-level"
        )
    return root


def _candidate_identity(contract_path: Path, contract: dict[str, Any]) -> tuple[str, str]:
    root = _candidate_root(contract_path, contract)
    head = _git(root, "rev-parse", "--verify", "HEAD")
    lineage = _git(root, "rev-list", "--parents", "-n", "1", head).split()
    if len(lineage) != 2 or lineage[0] != head:
        raise ValidationError("candidate HEAD must have exactly one parent")
    if _git(root, "status", "--porcelain=v1", "--untracked-files=all"):
        raise ValidationError("candidate Git worktree must be clean before full verdict")
    return head, lineage[1]


def _outside_candidate(path: Path, candidate_root: Path, label: str) -> None:
    resolved = path.expanduser().resolve(strict=False)
    if resolved == candidate_root or candidate_root in resolved.parents:
        raise ValidationError(f"{label} must be outside candidate_root")


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


def _communicate_with_timeout(
    child: subprocess.Popen[Any], timeout_s: float, label: str
) -> tuple[str, str]:
    try:
        return child.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired as error:
        child.terminate()
        try:
            stdout, stderr = child.communicate(timeout=2.0)
        except subprocess.TimeoutExpired:
            child.kill()
            stdout, stderr = child.communicate(timeout=2.0)
        detail = (stderr or stdout).strip().splitlines()[-1:] or []
        suffix = f": {detail[0][:256]}" if detail else ""
        raise ValidationError(f"{label} timed out after {timeout_s:g}s{suffix}") from error


def _proc_identity(pid: int) -> dict[str, Any]:
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip().lower()
        stat_text = Path(f"/proc/{pid}/stat").read_text(encoding="ascii").strip()
        closing = stat_text.rfind(")")
        suffix = stat_text[closing + 1 :].split()
        start_ticks = int(suffix[19])
    except (OSError, UnicodeError, ValueError, IndexError) as error:
        raise ValidationError("could not observe benchmark child process") from error
    if closing < 0 or start_ticks <= 0:
        raise ValidationError("benchmark child process identity is malformed")
    return {"boot_id": boot_id, "pid": pid, "start_ticks": start_ticks}


def _profile_runner_path(contract_path: Path, contract: dict[str, Any]) -> Path:
    argv = contract["commands"]["profile_one_call"]
    executable_name = Path(argv[0]).name.lower()
    runner_index = 1 if executable_name.startswith("python") else 0
    if len(argv) <= runner_index or argv[runner_index].startswith("-"):
        raise ValidationError("profile_one_call must identify one frozen runner file")
    runner = Path(argv[runner_index])
    if not runner.is_absolute():
        runner = contract_path.parent / runner
    adapter = Path(contract["commands"]["benchmark_adapter"])
    if not adapter.is_absolute():
        adapter = contract_path.parent / adapter
    if runner.resolve(strict=True) != adapter.resolve(strict=True):
        raise ValidationError(
            "profile_one_call runner must be the frozen benchmark adapter"
        )
    return runner.resolve(strict=True)


def _manifest_names(contract_path: Path, contract: dict[str, Any]) -> set[str]:
    manifest = Path(contract["runtime"]["kernel_manifest"])
    if not manifest.is_absolute():
        manifest = contract_path.parent / manifest
    raw = manifest.resolve(strict=True).read_text(encoding="utf-8").splitlines()
    if len(raw) < 3 or raw[0] != "rhinoforge-kernels-v1":
        raise ValidationError("public kernel manifest is malformed")
    return set(raw[2:])


def _trace_execution(
    raw_trace_path: Path,
    summary_path: Path,
    *,
    contract: dict[str, Any],
    manifest_names: set[str],
    preflight_hash: str,
) -> dict[str, Any]:
    if raw_trace_path.is_symlink() or not raw_trace_path.is_file():
        raise ValidationError("raw hardware trace must be a regular non-symlink file")
    if contract["runtime"].get("trace_schema") != "rhinoforge-rpu-chrome-v1":
        raise ValidationError("unsupported RPU trace schema")
    event_category = contract["runtime"].get("trace_event_category")
    if event_category != RPU_DEVICE_EVENT_CATEGORY:
        raise ValidationError("unsupported RPU device-event category")
    summary = summarize_trace(
        raw_trace_path,
        sorted(manifest_names),
        event_category=event_category,
        required_trace_schema=contract["runtime"]["trace_schema"],
        expected_producer_sha256=contract["commands"][
            "benchmark_adapter_sha256"
        ],
    )
    encoded_summary = canonical_json(summary) + "\n"
    # Never replace the raw trace while materializing its sanitized summary;
    # this also rejects hard-link aliases through the sanitizer's checked
    # destination path.
    _atomic_write_summary(
        summary_path, encoded_summary, forbidden=(raw_trace_path,)
    )
    input_trace_hash = summary.get("trace_sha256")
    if (
        not isinstance(input_trace_hash, str)
        or SHA256_RE.fullmatch(input_trace_hash) is None
        or input_trace_hash == "0" * 64
    ):
        raise ValidationError("trace summary lacks its raw-trace SHA-256 identity")
    if _sha256(raw_trace_path) != input_trace_hash:
        raise ValidationError("sanitizer trace identity does not match raw trace bytes")
    events = summary.get("events")
    if not isinstance(events, list):
        raise ValidationError("trace summary events are missing")
    counts: dict[str, int] = {}
    for event in events:
        if not isinstance(event, dict):
            raise ValidationError("trace summary event is malformed")
        name = event.get("name")
        count = event.get("count")
        if isinstance(name, str) and name in manifest_names:
            if not isinstance(count, int) or isinstance(count, bool) or count < 1:
                raise ValidationError("trace kernel event count is invalid")
            counts[name] = counts.get(name, 0) + count
    operation = contract["campaign"]["op"]
    required = set(contract["runtime"]["required_kernels"])
    admissible = (
        {contract["runtime"]["fused_kernel"]}
        if operation in FUSION_OPS
        else required
    )
    selected = [name for name in admissible if counts.get(name) == 1]
    total_manifest_launches = sum(counts.values())
    if (
        summary.get("complete_event_count") != 1
        or len(selected) != 1
        or total_manifest_launches != 1
    ):
        raise ValidationError(
            "trace does not contain exactly one complete admitted device-program launch"
        )
    return {
        "kernel_names": selected,
        "launch_count": 1,
        "single_device_program": True,
        "trace_summary_sha256": hashlib.sha256(
            encoded_summary.encode("utf-8")
        ).hexdigest(),
        "raw_trace_sha256": input_trace_hash,
        "preflight_sha256": preflight_hash,
    }


def _collect_one_call_trace(
    *,
    contract_path: Path,
    contract: dict[str, Any],
    contract_hash: str,
    workload: str,
    raw_trace_path: Path,
    summary_path: Path,
    manifest_names: set[str],
    preflight_hash: str,
    candidate_commit: str,
    parent_commit: str,
    device: str,
    board_lease: BoardLease,
    timeout_s: float = 600.0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if raw_trace_path.exists() or raw_trace_path.is_symlink():
        raise ValidationError("one-call profiler output path must be new")
    command = list(contract["commands"]["profile_one_call"])
    environment = os.environ.copy()
    environment.update(
        {
            "RHINOFORGE_PROFILE_WORKLOAD": workload,
            "RHINOFORGE_PROFILE_TRACE_OUTPUT": str(raw_trace_path),
            "RHINOFORGE_PROFILE_CONTRACT_SHA256": contract_hash,
            "RHINOFORGE_PROFILE_PREFLIGHT_SHA256": preflight_hash,
            "RHINOFORGE_PROFILE_CANDIDATE_COMMIT": candidate_commit,
            "RHINOFORGE_PROFILE_PARENT_COMMIT": parent_commit,
            "RHINOFORGE_PROFILE_DEVICE": device,
            "RHINOFORGE_PROFILE_EVENT_CATEGORY": contract["runtime"][
                "trace_event_category"
            ],
            "RHINOFORGE_PROFILE_PRODUCER_SHA256": contract["commands"][
                "benchmark_adapter_sha256"
            ],
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    started_monotonic_ns = time.monotonic_ns()
    child = subprocess.Popen(
        command,
        cwd=contract_path.parent,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    observed = _proc_identity(child.pid)
    stdout, stderr = _communicate_with_timeout(
        child, timeout_s, "one-call profiler"
    )
    finished_monotonic_ns = time.monotonic_ns()
    board_lease.check_integrity()
    if child.returncode != 0:
        detail = (stderr or stdout).strip().splitlines()[-1:] or ["unknown"]
        raise ValidationError("one-call profiler failed: " + detail[0][:512])
    if raw_trace_path.is_symlink() or not raw_trace_path.is_file():
        raise ValidationError("one-call profiler did not create a regular trace")
    evidence = _trace_execution(
        raw_trace_path,
        summary_path,
        contract=contract,
        manifest_names=manifest_names,
        preflight_hash=preflight_hash,
    )
    adapter_execution = {
        key: evidence[key]
        for key in (
            "kernel_names",
            "launch_count",
            "single_device_program",
            "trace_summary_sha256",
            "preflight_sha256",
        )
    }
    metadata = {
        "workload_id": workload,
        "raw_trace_sha256": evidence["raw_trace_sha256"],
        "raw_trace_size": raw_trace_path.stat().st_size,
        "summary_sha256": evidence["trace_summary_sha256"],
        "summary_size": summary_path.stat().st_size,
        "complete_event_count": 1,
        "manifest_launch_count": 1,
        "unknown_complete_event_count": 0,
        "admitted_kernel_names": evidence["kernel_names"],
        "collector_process": observed,
        "started_monotonic_ns": started_monotonic_ns,
        "finished_monotonic_ns": finished_monotonic_ns,
        "profile_argv_sha256": hashlib.sha256(
            canonical_json(command).encode("utf-8")
        ).hexdigest(),
    }
    return adapter_execution, metadata


def _trusted_execution_envelope(
    *,
    contract_hash: str,
    preflight_hash: str,
    candidate_commit: str,
    parent_commit: str,
    workloads: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Build the exact local verifier-to-benchmark evidence handoff."""

    return {
        "schema_version": 1,
        "kind": "rhinoforge-local-trusted-execution",
        "contract_sha256": contract_hash,
        "preflight_sha256": preflight_hash,
        "candidate_commit": candidate_commit,
        "parent_commit": parent_commit,
        "workloads": workloads,
    }


def run_full_verdict(
    contract_path: Path,
    adapter: Path,
    preflight_receipt: Path,
    output_dir: Path,
    *,
    rhinoforge_root: Path,
    device_compiler: Path | None,
    candidate_id_prefix: str,
    device: str,
    seed: int,
    profile_timeout_s: float = 600.0,
    benchmark_timeout_s: float = 3600.0,
) -> dict[str, Any]:
    if device != "rpu":
        raise ValidationError("full-verdict device must be exactly 'rpu'")
    for label, value in (
        ("profile timeout", profile_timeout_s),
        ("benchmark timeout", benchmark_timeout_s),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValidationError(f"{label} must be numeric")
        if not math.isfinite(float(value)) or value <= 0:
            raise ValidationError(f"{label} must be finite and positive")
    if contract_path.is_symlink():
        raise ValidationError("contract must not be a symlink")
    contract_path = contract_path.resolve(strict=True)
    contract = load_contract(contract_path)
    contract_hash = contract_sha256(contract_path)
    candidate_root = _candidate_root(contract_path, contract)
    reference_root, reference_tree_hash = _reference_identity(
        contract_path, contract
    )
    _outside_candidate(contract_path, candidate_root, "campaign contract")
    _outside_candidate(reference_root, candidate_root, "reference root")
    configured_adapter = Path(contract["commands"]["benchmark_adapter"])
    if not configured_adapter.is_absolute():
        configured_adapter = contract_path.parent / configured_adapter
    if adapter.is_symlink():
        raise ValidationError("benchmark adapter must not be a symlink")
    supplied_adapter = adapter.resolve(strict=True)
    configured_adapter = configured_adapter.resolve(strict=True)
    if supplied_adapter != configured_adapter:
        raise ValidationError("--adapter does not match the frozen contract adapter")
    _outside_candidate(configured_adapter, candidate_root, "benchmark adapter")
    _outside_candidate(preflight_receipt, candidate_root, "preflight receipt")
    _outside_candidate(output_dir, candidate_root, "verdict output")
    expected_adapter_hash = contract["commands"]["benchmark_adapter_sha256"]
    adapter_identity = _adapter_identity(configured_adapter, expected_adapter_hash)
    _profile_runner_path(contract_path, contract)
    if preflight_receipt.is_symlink():
        raise ValidationError("preflight receipt must not be a symlink")
    preflight_receipt = preflight_receipt.resolve(strict=True)
    preflight_hash = _sha256(preflight_receipt)
    _verify_preflight_receipt(preflight_receipt, preflight_hash, contract_hash)
    preflight = load_json(preflight_receipt)
    if not isinstance(preflight, dict) or preflight.get("ok") is not True:
        raise ValidationError("full verdict requires a successful preflight receipt")
    if preflight.get("contract_sha256") != contract_hash:
        raise ValidationError("preflight receipt belongs to another contract")
    workload_ids = [item["id"] for item in contract["workloads"]]
    manifest_names = _manifest_names(contract_path, contract)
    candidate_commit, parent_commit = _candidate_identity(contract_path, contract)

    def revalidate_campaign_identity() -> None:
        if contract_sha256(contract_path) != contract_hash:
            raise ValidationError("contract changed during full verdict")
        if _adapter_identity(configured_adapter, expected_adapter_hash) != adapter_identity:
            raise ValidationError("benchmark adapter changed during full verdict")
        if _candidate_identity(contract_path, contract) != (
            candidate_commit,
            parent_commit,
        ):
            raise ValidationError("candidate identity changed during full verdict")
        final_reference_root, final_reference_hash = _reference_identity(
            contract_path, contract
        )
        if (
            final_reference_root != reference_root
            or final_reference_hash != reference_tree_hash
        ):
            raise ValidationError("reference identity changed during full verdict")
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_summary_dir = output_dir / "trace-summaries"
    raw_trace_dir = output_dir / "raw-traces"
    trace_summary_dir.mkdir(exist_ok=True)
    raw_trace_dir.mkdir(exist_ok=True)
    if (
        output_dir.is_symlink()
        or trace_summary_dir.is_symlink()
        or raw_trace_dir.is_symlink()
    ):
        raise ValidationError("verdict output directories must not be symlinks")
    process_count = int(contract["measurement"]["verdict_processes"])
    repeats = max(2, math.ceil(int(contract["measurement"]["samples"]) / 2))
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    if output_dir.is_symlink() or raw_dir.is_symlink():
        raise ValidationError("verdict output directories must not be symlinks")

    recorded: list[dict[str, Any]] = []
    profile_metadata: list[dict[str, Any]] = []
    trusted_execution: dict[str, dict[str, Any]] = {}
    trusted_execution_path = output_dir / "trusted-execution.json"
    if trusted_execution_path.exists() or trusted_execution_path.is_symlink():
        raise ValidationError("trusted execution output path must be new")
    with BoardLease(0.0, ["run_full_verdict", contract_hash]) as board_lease:
        try:
            live_preflight = run_preflight(
                contract_path,
                probe_torch_rpu=True,
                device_compiler=device_compiler,
                rhinoforge_root=rhinoforge_root,
            )
            if live_preflight.get("ok") is not True:
                raise ValidationError(
                    "live preflight revalidation failed before full verdict"
                )
            if canonical_json(live_preflight) != canonical_json(preflight):
                raise ValidationError(
                    "bound preflight receipt does not match live revalidation"
                )
            for workload in workload_ids:
                execution, metadata = _collect_one_call_trace(
                    contract_path=contract_path,
                    contract=contract,
                    contract_hash=contract_hash,
                    workload=workload,
                    raw_trace_path=raw_trace_dir / f"{workload}.json",
                    summary_path=trace_summary_dir / f"{workload}.json",
                    manifest_names=manifest_names,
                    preflight_hash=preflight_hash,
                    candidate_commit=candidate_commit,
                    parent_commit=parent_commit,
                    device=device,
                    board_lease=board_lease,
                    timeout_s=float(profile_timeout_s),
                )
                trusted_execution[workload] = execution
                profile_metadata.append(metadata)
                revalidate_campaign_identity()
            trusted_envelope = _trusted_execution_envelope(
                contract_hash=contract_hash,
                preflight_hash=preflight_hash,
                candidate_commit=candidate_commit,
                parent_commit=parent_commit,
                workloads=trusted_execution,
            )
            _atomic_write_summary(
                trusted_execution_path, canonical_json(trusted_envelope) + "\n"
            )
            for index in range(process_count):
                candidate_id = f"{candidate_id_prefix}-p{index + 1:02d}"
                result_path = raw_dir / f"{candidate_id}.json"
                command = [
                    sys.executable,
                    str(SCRIPT_DIR / "bench.py"),
                    "--adapter",
                    str(configured_adapter),
                    "--profile",
                    contract["campaign"]["op"],
                    "--device",
                    device,
                    "--seed",
                    str(seed + index * 1_000_003),
                    "--warmup",
                    str(contract["measurement"]["warmup"]),
                    "--repeats",
                    str(repeats),
                    "--atol",
                    str(contract["gates"]["max_abs"]),
                    "--rtol",
                    str(contract["gates"]["max_rel"]),
                    "--anchor-atol",
                    str(contract["gates"]["anchor_max_abs"]),
                    "--anchor-rtol",
                    str(contract["gates"]["anchor_max_rel"]),
                    "--max-cv",
                    str(contract["gates"]["max_cv"]),
                    "--candidate-id",
                    candidate_id,
                    "--candidate-commit",
                    candidate_commit,
                    "--parent-commit",
                    parent_commit,
                    "--contract-sha256",
                    contract_hash,
                    "--preflight-sha256",
                    preflight_hash,
                    "--trusted-execution",
                    str(trusted_execution_path),
                    "--verdict",
                    "full",
                    "--output",
                    str(result_path),
                ]
                for workload in workload_ids:
                    command.extend(("--workload", workload))
                if contract["campaign"]["op"] in FUSION_OPS:
                    gates = contract["gates"]
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
                child = subprocess.Popen(
                    command,
                    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                observed = _proc_identity(child.pid)
                stdout, stderr = _communicate_with_timeout(
                    child, float(benchmark_timeout_s), "benchmark child"
                )
                board_lease.check_integrity()
                revalidate_campaign_identity()
                if child.returncode not in (0, 2) or not result_path.is_file():
                    detail = (stderr or stdout).strip().splitlines()[-1:] or ["unknown"]
                    raise ValidationError(
                        "benchmark child failed: " + detail[0][:512]
                    )
                raw_result = load_json(result_path)
                if not isinstance(raw_result, dict) or not isinstance(
                    raw_result.get("process"), dict
                ):
                    raise ValidationError(
                        "benchmark child result lacks process identity"
                    )
                observed["run_uuid"] = raw_result["process"].get("run_uuid")
                result_workloads = raw_result.get("workloads")
                if not isinstance(result_workloads, list) or len(
                    result_workloads
                ) != len(workload_ids):
                    raise ValidationError(
                        "benchmark result workloads do not match verifier evidence"
                    )
                seen_workloads: set[str] = set()
                for workload in result_workloads:
                    if not isinstance(workload, dict):
                        raise ValidationError("benchmark workload result is malformed")
                    workload_id = workload.get("id")
                    expected = trusted_execution.get(workload_id)
                    if (
                        not isinstance(workload_id, str)
                        or workload_id in seen_workloads
                        or expected is None
                        or workload.get("execution") != expected
                    ):
                        raise ValidationError(
                            "benchmark execution evidence does not match verifier trace"
                        )
                    seen_workloads.add(workload_id)
                receipt = record_iteration(
                    contract_path,
                    result_path,
                    output_dir,
                    observed_process=observed,
                    preflight_receipt=preflight_receipt,
                )
                recorded.append(receipt)
        except Exception:
            try:
                board_lease.finish(2)
            except Exception:
                pass
            raise
        board_lease.finish(0)

    all_gates_passed = bool(recorded) and all(
        receipt.get("local_gate_eligible") is True for receipt in recorded
    )
    diagnostic = {
        "schema_version": 1,
        "ok": all_gates_passed,
        "kind": "rhinoforge-local-diagnostic-session",
        "collection_ok": True,
        "all_gates_passed": all_gates_passed,
        "failed_receipt_count": sum(
            receipt.get("local_gate_eligible") is not True for receipt in recorded
        ),
        "release_eligible": False,
        "release_blocker": "external-authority DSSE release proof required",
        "contract_sha256": contract_hash,
        "preflight_sha256": preflight_hash,
        "adapter_sha256": contract["commands"]["benchmark_adapter_sha256"],
        "candidate_commit": candidate_commit,
        "parent_commit": parent_commit,
        "board_lease": {
            "lease_id": board_lease.lease_id,
            "start_sequence": board_lease.start_sequence,
            "finish_sequence": board_lease.finish_sequence,
            "previous_journal_head_sha256": board_lease.previous_head_hash,
            "start_record_sha256": board_lease.start_record_sha256,
            "finish_record_sha256": board_lease.finish_record_sha256,
            "final_journal_head_sha256": board_lease.head_hash,
            "started_monotonic_ns": board_lease.started_monotonic_ns,
            "finished_monotonic_ns": board_lease.finished_monotonic_ns,
        },
        "profiles": profile_metadata,
        "reference_tree_sha256": reference_tree_hash,
        "trusted_execution": {
            "path": str(trusted_execution_path),
            "sha256": _sha256(trusted_execution_path),
            "size": trusted_execution_path.stat().st_size,
        },
        "recorded_receipts": recorded,
    }
    diagnostic_path = output_dir / "LOCAL-DIAGNOSTIC.json"
    _atomic_write_summary(diagnostic_path, canonical_json(diagnostic) + "\n")
    diagnostic["diagnostic"] = str(diagnostic_path)
    return diagnostic


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--rhinoforge-root", type=Path, required=True)
    parser.add_argument("--device-compiler", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--candidate-id-prefix", required=True)
    parser.add_argument("--device", choices=("rpu",), default="rpu")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--profile-timeout", type=float, default=600.0)
    parser.add_argument("--benchmark-timeout", type=float, default=3600.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        report = run_full_verdict(
            args.contract,
            args.adapter,
            args.preflight,
            args.output_dir,
            rhinoforge_root=args.rhinoforge_root,
            device_compiler=args.device_compiler,
            candidate_id_prefix=args.candidate_id_prefix,
            device=args.device,
            seed=args.seed,
            profile_timeout_s=args.profile_timeout,
            benchmark_timeout_s=args.benchmark_timeout,
        )
    except (
        OSError,
        ValidationError,
        subprocess.SubprocessError,
        ValueError,
    ) as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "ok": False,
                    "kind": "rhinoforge-local-diagnostic-session",
                    "collection_ok": False,
                    "all_gates_passed": False,
                    "release_eligible": False,
                    "error": str(error),
                },
                sort_keys=True,
            )
        )
        return 2
    print(canonical_json(report))
    return 0 if report["all_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
