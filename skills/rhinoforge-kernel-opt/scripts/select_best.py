#!/usr/bin/env python3
"""Promote the best fully validated kernel candidate to BEST.json."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from campaign_common import (
    COMMIT_RE,
    FUSION_OPS,
    SHA256_RE,
    ValidationError,
    canonical_json,
    contract_sha256,
    load_contract,
    record_is_promotable,
    strict_json_loads,
    validate_result,
)
from attestation import verify_release_proof
from record_iteration import _git, verify_candidate_git


MAX_JOURNAL_BYTES = 64 * 1024 * 1024
UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)


def _read_regular_file(path: Path, maximum: int, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValidationError(f"{label} is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_uid != os.geteuid()
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) & 0o022
            or opened.st_size > maximum
        ):
            raise ValidationError(f"{label} identity or mode is unsafe")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            if not block:
                raise ValidationError(f"{label} changed while reading")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        stable_fields = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_uid", "st_size", "st_mtime_ns", "st_ctime_ns")
        if any(getattr(after, field) != getattr(opened, field) for field in stable_fields):
            raise ValidationError(f"{label} changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _load_records(
    results_path: Path, contract: dict[str, Any], contract_hash: str
) -> tuple[list[dict[str, Any]], list[str], bytes]:
    if results_path.is_symlink() or not results_path.is_file():
        raise ValidationError("results.jsonl must be a regular non-symlink file")
    try:
        raw_journal = _read_regular_file(
            results_path, MAX_JOURNAL_BYTES, "results.jsonl"
        )
        text = raw_journal.decode("utf-8")
    except UnicodeError as error:
        raise ValidationError("results.jsonl is unreadable") from error
    if not text:
        raise ValidationError("results.jsonl is empty")
    records: list[dict[str, Any]] = []
    record_hashes: list[str] = []
    candidate_ids: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            raise ValidationError(f"results.jsonl has a blank line at {line_number}")
        try:
            raw = strict_json_loads(line)
        except (ValidationError, json.JSONDecodeError, RecursionError) as error:
            raise ValidationError(
                f"results.jsonl has invalid JSON at line {line_number}"
            ) from error
        try:
            record = validate_result(raw, contract, contract_hash)
        except ValidationError as error:
            raise ValidationError(
                f"results.jsonl line {line_number} failed validation: {error}"
            ) from error
        if record["candidate_id"] in candidate_ids:
            raise ValidationError("results.jsonl contains a duplicate candidate_id")
        candidate_ids.add(record["candidate_id"])
        records.append(record)
        record_hashes.append(
            hashlib.sha256(canonical_json(record).encode("utf-8")).hexdigest()
        )
    return records, record_hashes, raw_journal


def _atomic_write(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (path.is_symlink() or not path.is_file()):
        raise ValidationError("BEST output must be a regular non-symlink file")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        data = contents.encode("utf-8")
        view = memoryview(data)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short BEST write")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists():
            temporary.unlink()


def _exact_object(value: Any, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValidationError(f"{label} schema is not exact")
    return value


def _sha(value: Any, label: str, *, nonzero: bool = True) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    if nonzero and value == "0" * 64:
        raise ValidationError(f"{label} must be nonzero")
    return value


def _positive_int(value: Any, label: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _short_text(value: Any, label: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValidationError(f"{label} must be non-empty short text")
    return value


def _source_tree_identity(root: Path, commit: str) -> tuple[str, str]:
    """Return Git tree OID and a SHA-256 content-tree identity."""

    tree_oid = _git(root, "rev-parse", f"{commit}^{{tree}}")
    try:
        listing = subprocess.run(
            ["git", "-C", str(root), "ls-tree", "-r", "-z", "--full-tree", commit],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValidationError("candidate content-tree identity is unavailable") from error
    if listing.returncode != 0 or len(listing.stdout) > 64 * 1024 * 1024:
        raise ValidationError("candidate content-tree listing failed")
    digest = hashlib.sha256(b"rhinoforge-source-tree-v1\0")
    total_blob_bytes = 0
    entries = listing.stdout.split(b"\0")
    if entries[-1:] == [b""]:
        entries.pop()
    for entry in entries:
        try:
            metadata, path_bytes = entry.split(b"\t", 1)
            mode, object_type, object_id = metadata.split(b" ", 2)
        except ValueError as error:
            raise ValidationError("candidate content-tree entry is malformed") from error
        if object_type == b"blob":
            blob = subprocess.run(
                ["git", "-C", str(root), "cat-file", "blob", object_id.decode("ascii")],
                check=False,
                capture_output=True,
                timeout=30,
            )
            if blob.returncode != 0:
                raise ValidationError("candidate content-tree blob is unavailable")
            total_blob_bytes += len(blob.stdout)
            if total_blob_bytes > 2 * 1024 * 1024 * 1024:
                raise ValidationError("candidate content tree exceeds 2 GiB")
            content_hash = hashlib.sha256(blob.stdout).digest()
        elif object_type == b"commit":
            content_hash = hashlib.sha256(b"gitlink\0" + object_id).digest()
        else:
            raise ValidationError("candidate content-tree object type is unsupported")
        for component in (mode, path_bytes, content_hash):
            digest.update(len(component).to_bytes(8, "big"))
            digest.update(component)
    return tree_oid, digest.hexdigest()


def _validate_release_statement(
    statement: dict[str, Any],
    *,
    contract: dict[str, Any],
    contract_hash: str,
    records: list[dict[str, Any]],
    record_hashes: list[str],
    raw_journal: bytes,
) -> tuple[set[str], dict[str, dict[str, Any]]]:
    root = _exact_object(
        statement,
        {
            "schema_version",
            "kind",
            "proof_id",
            "signer_id",
            "challenge",
            "sequence",
            "previous_proof_sha256",
            "issued_at",
            "issuer",
            "inputs",
            "source",
            "execution",
            "decision",
        },
        "release proof statement",
    )
    if root["schema_version"] != 1 or root["kind"] != "rhinoforge-kernel-release-proof":
        raise ValidationError("release proof statement version or kind is not admitted")
    if not isinstance(root["proof_id"], str) or UUID_RE.fullmatch(root["proof_id"]) is None:
        raise ValidationError("release proof ID is malformed")
    sequence = _positive_int(root["sequence"], "release proof sequence")
    previous_proof = _sha(
        root["previous_proof_sha256"], "previous release proof", nonzero=False
    )
    if (sequence == 1) != (previous_proof == "0" * 64):
        raise ValidationError("release proof sequence/previous-proof chain is inconsistent")
    _short_text(root["challenge"], "release proof challenge", 512)
    _short_text(root["issued_at"], "release proof issued_at", 64)

    issuer = _exact_object(
        root["issuer"],
        {"policy_sha256", "signer_build_sha256", "collection_mode"},
        "release proof issuer",
    )
    if issuer["collection_mode"] != "authority-controlled-board-session":
        raise ValidationError("release proof was not collected by the external authority")
    if issuer["policy_sha256"] != contract["attestation"]["policy_sha256"]:
        raise ValidationError("release proof policy does not match the contract")
    if issuer["signer_build_sha256"] != contract["attestation"]["signer_build_sha256"]:
        raise ValidationError("release proof signer build does not match the contract")

    inputs = _exact_object(
        root["inputs"],
        {
            "campaign_id",
            "contract_sha256",
            "preflight_sha256",
            "adapter_sha256",
            "benchmark_argv_sha256",
            "profile_argv_sha256",
        },
        "release proof inputs",
    )
    if inputs["campaign_id"] != contract["campaign"]["id"]:
        raise ValidationError("release proof campaign ID does not match")
    if inputs["contract_sha256"] != contract_hash:
        raise ValidationError("release proof contract hash does not match")
    _sha(inputs["preflight_sha256"], "release proof preflight")
    if inputs["adapter_sha256"] != contract["commands"]["benchmark_adapter_sha256"]:
        raise ValidationError("release proof adapter hash does not match")
    for command_name, proof_name in (
        ("benchmark", "benchmark_argv_sha256"),
        ("profile_one_call", "profile_argv_sha256"),
    ):
        expected = hashlib.sha256(
            canonical_json(contract["commands"][command_name]).encode("utf-8")
        ).hexdigest()
        if inputs[proof_name] != expected:
            raise ValidationError(f"release proof {proof_name} does not match")

    source = _exact_object(
        root["source"],
        {
            "repository_id",
            "candidate_commit",
            "parent_commit",
            "git_tree_oid",
            "source_root_sha256",
        },
        "release proof source",
    )
    _short_text(source["repository_id"], "release proof repository_id", 256)
    if COMMIT_RE.fullmatch(str(source["candidate_commit"])) is None or COMMIT_RE.fullmatch(
        str(source["parent_commit"])
    ) is None:
        raise ValidationError("release proof Git commits are malformed")
    if not isinstance(source["git_tree_oid"], str) or len(source["git_tree_oid"]) not in (40, 64):
        raise ValidationError("release proof Git tree OID is malformed")
    _sha(source["source_root_sha256"], "release proof source root")

    execution = _exact_object(
        root["execution"], {"board_id", "lease", "processes", "traces"}, "release proof execution"
    )
    _short_text(execution["board_id"], "release proof board_id", 256)
    lease = _exact_object(
        execution["lease"],
        {
            "lease_id",
            "start_sequence",
            "finish_sequence",
            "previous_journal_head_sha256",
            "start_record_sha256",
            "finish_record_sha256",
            "final_journal_head_sha256",
            "started_monotonic_ns",
            "finished_monotonic_ns",
        },
        "release proof lease",
    )
    if not isinstance(lease["lease_id"], str) or UUID_RE.fullmatch(lease["lease_id"]) is None:
        raise ValidationError("release proof lease ID is malformed")
    start_sequence = _positive_int(lease["start_sequence"], "lease start sequence")
    finish_sequence = _positive_int(lease["finish_sequence"], "lease finish sequence")
    if finish_sequence != start_sequence + 1:
        raise ValidationError("release proof lease sequence is not one complete pair")
    for field in (
        "previous_journal_head_sha256",
        "start_record_sha256",
        "finish_record_sha256",
        "final_journal_head_sha256",
    ):
        _sha(lease[field], f"release proof lease {field}", nonzero=field != "previous_journal_head_sha256")
    lease_start = _positive_int(lease["started_monotonic_ns"], "lease start time")
    lease_finish = _positive_int(lease["finished_monotonic_ns"], "lease finish time")
    if lease_finish <= lease_start:
        raise ValidationError("release proof lease time interval is invalid")

    raw_processes = execution["processes"]
    if not isinstance(raw_processes, list) or not raw_processes:
        raise ValidationError("release proof process list is empty")
    processes: dict[str, dict[str, Any]] = {}
    for item in raw_processes:
        process = _exact_object(
            item,
            {
                "process_id",
                "role",
                "workload_id",
                "receipt_sha256",
                "boot_id",
                "pid",
                "start_ticks",
                "run_uuid",
                "executable_sha256",
                "argv_sha256",
                "environment_sha256",
                "started_monotonic_ns",
                "finished_monotonic_ns",
                "exit_code",
            },
            "release proof process",
        )
        process_id = _short_text(process["process_id"], "release proof process_id", 128)
        if process_id in processes:
            raise ValidationError("release proof process IDs are not unique")
        if process["role"] not in {"preflight", "profile", "benchmark"}:
            raise ValidationError("release proof process role is invalid")
        if process["workload_id"] is not None:
            _short_text(process["workload_id"], "release proof process workload", 128)
        if process["receipt_sha256"] is not None:
            _sha(process["receipt_sha256"], "release proof process receipt")
        if not isinstance(process["boot_id"], str) or UUID_RE.fullmatch(process["boot_id"]) is None:
            raise ValidationError("release proof process boot ID is malformed")
        _positive_int(process["pid"], "release proof process PID")
        _positive_int(process["start_ticks"], "release proof process start ticks")
        if process["run_uuid"] is not None and (
            not isinstance(process["run_uuid"], str) or UUID_RE.fullmatch(process["run_uuid"]) is None
        ):
            raise ValidationError("release proof process run UUID is malformed")
        for field in ("executable_sha256", "argv_sha256", "environment_sha256"):
            _sha(process[field], f"release proof process {field}")
        started = _positive_int(process["started_monotonic_ns"], "process start time")
        finished = _positive_int(process["finished_monotonic_ns"], "process finish time")
        if started < lease_start or finished > lease_finish or finished <= started:
            raise ValidationError("release proof process is outside the lease interval")
        if process["exit_code"] != 0:
            raise ValidationError("release proof process did not exit successfully")
        processes[process_id] = process
    if sum(process["role"] == "preflight" for process in processes.values()) != 1:
        raise ValidationError("release proof must contain one authority preflight process")
    process_boot_ids = {process["boot_id"] for process in processes.values()}
    if len(process_boot_ids) != 1:
        raise ValidationError("release proof processes do not share one boot identity")

    raw_traces = execution["traces"]
    if not isinstance(raw_traces, list):
        raise ValidationError("release proof traces must be a list")
    traces: dict[str, dict[str, Any]] = {}
    for item in raw_traces:
        trace = _exact_object(
            item,
            {
                "workload_id",
                "capture_id",
                "collector_process_id",
                "raw_sha256",
                "raw_size",
                "summary_sha256",
                "summary_size",
                "complete_event_count",
                "manifest_launch_count",
                "unknown_complete_event_count",
                "admitted_kernel_names",
            },
            "release proof trace",
        )
        workload_id = _short_text(trace["workload_id"], "release proof trace workload", 128)
        if workload_id in traces:
            raise ValidationError("release proof trace workloads are not unique")
        _short_text(trace["capture_id"], "release proof capture ID", 128)
        collector = processes.get(trace["collector_process_id"])
        if collector is None or collector["role"] != "profile" or collector["workload_id"] != workload_id:
            raise ValidationError("release proof trace collector binding is invalid")
        if collector["argv_sha256"] != inputs["profile_argv_sha256"]:
            raise ValidationError("release proof profile process argv does not match")
        _sha(trace["raw_sha256"], "release proof raw trace")
        _sha(trace["summary_sha256"], "release proof trace summary")
        _positive_int(trace["raw_size"], "release proof raw trace size")
        _positive_int(trace["summary_size"], "release proof trace summary size")
        if (
            trace["complete_event_count"] != 1
            or trace["manifest_launch_count"] != 1
            or trace["unknown_complete_event_count"] != 0
        ):
            raise ValidationError("release proof trace is not exactly one admitted device launch")
        names = trace["admitted_kernel_names"]
        required_names = set(contract["runtime"]["required_kernels"])
        if contract["campaign"]["op"] in FUSION_OPS:
            required_names = {contract["runtime"]["fused_kernel"]}
        if not isinstance(names, list) or len(names) != 1 or names[0] not in required_names:
            raise ValidationError("release proof admitted kernel is invalid")
        traces[workload_id] = trace
    expected_workloads = {item["id"] for item in contract["workloads"]}
    if set(traces) != expected_workloads:
        raise ValidationError("release proof does not cover every exact workload")

    decision = _exact_object(
        root["decision"],
        {"record_sha256s", "results_snapshot", "selected_candidate_commit"},
        "release proof decision",
    )
    if decision["record_sha256s"] != record_hashes:
        raise ValidationError("release proof record set does not equal the journal")
    snapshot = _exact_object(
        decision["results_snapshot"], {"sha256", "size", "line_count"}, "release proof results snapshot"
    )
    if (
        snapshot["sha256"] != hashlib.sha256(raw_journal).hexdigest()
        or snapshot["size"] != len(raw_journal)
        or snapshot["line_count"] != len(records)
    ):
        raise ValidationError("release proof results snapshot does not match the journal")
    if decision["selected_candidate_commit"] != source["candidate_commit"]:
        raise ValidationError("release proof source and decision candidate differ")

    authorized = set(record_hashes)
    benchmark_by_receipt: dict[str, dict[str, Any]] = {}
    for process in processes.values():
        receipt_hash = process["receipt_sha256"]
        if process["role"] == "benchmark" and receipt_hash is not None:
            if process["argv_sha256"] != inputs["benchmark_argv_sha256"]:
                raise ValidationError("release proof benchmark process argv does not match")
            if receipt_hash in benchmark_by_receipt:
                raise ValidationError("release proof repeats a benchmark receipt binding")
            benchmark_by_receipt[receipt_hash] = process
    for record, record_hash in zip(records, record_hashes):
        if (
            record["candidate_commit"] != source["candidate_commit"]
            or record["parent_commit"] != source["parent_commit"]
            or record["preflight_sha256"] != inputs["preflight_sha256"]
        ):
            continue
        process = benchmark_by_receipt.get(record_hash)
        if process is None:
            raise ValidationError("release proof lacks a benchmark process for a candidate receipt")
        identity = record["process"]
        for field in ("boot_id", "pid", "start_ticks", "run_uuid"):
            if process[field] != identity[field]:
                raise ValidationError("release proof benchmark process identity mismatches receipt")
        for workload in record["workloads"]:
            trace = traces[workload["id"]]
            execution_evidence = workload.get("execution")
            if not isinstance(execution_evidence, dict) or execution_evidence.get(
                "trace_summary_sha256"
            ) != trace["summary_sha256"]:
                raise ValidationError("release proof trace does not bind the receipt execution")
    return authorized, traces


def select_best(
    contract_path: Path,
    results_path: Path,
    output_path: Path,
    *,
    release_proof: Path | None = None,
    expected_challenge: str | None = None,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    contract_hash = contract_sha256(contract_path)
    if contract["attestation"]["mode"] != "external":
        raise ValidationError(
            "promotion is blocked because attestation.mode is unavailable"
        )
    records, record_hashes, raw_journal = _load_records(
        results_path, contract, contract_hash
    )
    if release_proof is None or not expected_challenge:
        raise ValidationError(
            "promotion requires an external-authority DSSE release proof and fresh challenge"
        )
    statement, proof_hash = verify_release_proof(
        release_proof,
        public_key_base64=contract["attestation"]["public_key_base64"],
        verifier_executable=contract["attestation"]["verifier_executable"],
        verifier_sha256=contract["attestation"]["verifier_sha256"],
        expected_signer_id=contract["attestation"]["signer_id"],
        expected_challenge=expected_challenge,
    )
    authorized_hashes, _ = _validate_release_statement(
        statement,
        contract=contract,
        contract_hash=contract_hash,
        records=records,
        record_hashes=record_hashes,
        raw_journal=raw_journal,
    )
    source = statement["source"]
    proof_inputs = statement["inputs"]
    metric_name = contract["measurement"]["metric"]
    direction = contract["measurement"]["direction"]
    eligible_receipts = [
        record
        for record, record_hash in zip(records, record_hashes)
        if record_hash in authorized_hashes
        and record["candidate_commit"] == source["candidate_commit"]
        and record["parent_commit"] == source["parent_commit"]
        and record["preflight_sha256"] == proof_inputs["preflight_sha256"]
        and record_is_promotable(record, metric_name, direction, contract_hash)
    ]
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for record in eligible_receipts:
        group_key = (
            record["candidate_commit"],
            record["parent_commit"],
            record["contract_sha256"],
            record["preflight_sha256"],
        )
        grouped.setdefault(group_key, []).append(record)
    required_processes = int(contract["measurement"]["verdict_processes"])
    def process_identity(receipt: dict[str, Any]) -> tuple[str, int, int]:
        process = receipt["process"]
        return (
            process["boot_id"],
            int(process["pid"]),
            int(process["start_ticks"]),
        )

    eligible_groups = []
    for receipts in grouped.values():
        identities = {process_identity(receipt) for receipt in receipts}
        if len(identities) >= required_processes:
            eligible_groups.append(receipts)
    if not eligible_groups:
        raise ValidationError(
            "no full-verdict candidate commit has enough all-gate process receipts"
        )

    def group_metric(receipts: list[dict[str, Any]]) -> float:
        return float(
            statistics.median(float(receipt["metric"]["value"]) for receipt in receipts)
        )

    def key(receipts: list[dict[str, Any]]) -> tuple[float, str]:
        value = group_metric(receipts)
        directed = value if direction == "min" else -value
        return (directed, receipts[0]["candidate_commit"])

    winning_receipts = min(eligible_groups, key=key)
    winning_receipts = sorted(winning_receipts, key=lambda item: item["candidate_id"])
    winner = winning_receipts[0]
    candidate_root = verify_candidate_git(contract_path, contract, winner)
    tree_oid, source_root_sha256 = _source_tree_identity(
        candidate_root, winner["candidate_commit"]
    )
    if tree_oid != source["git_tree_oid"] or source_root_sha256 != source["source_root_sha256"]:
        raise ValidationError("release proof source-tree identity does not match Git")
    if winner["candidate_commit"] != statement["decision"]["selected_candidate_commit"]:
        raise ValidationError("selector winner differs from the externally attested decision")
    winning_process_identities = {
        process_identity(receipt) for receipt in winning_receipts
    }
    selected_metric = dict(winner["metric"])
    selected_metric["value"] = group_metric(winning_receipts)
    selected_metric["aggregation"] = "median_full_process_receipts"
    workload_evidence: list[dict[str, Any]] = []
    for workload_index, contract_workload in enumerate(contract["workloads"]):
        workload_evidence.append(
            {
                "id": contract_workload["id"],
                "process_receipts": [
                    {
                        "candidate_id": receipt["candidate_id"],
                        "summary": receipt["workloads"][workload_index]["summary"],
                        "hard_gates": receipt["workloads"][workload_index][
                            "hard_gates"
                        ],
                        "eligible": receipt["workloads"][workload_index]["eligible"],
                    }
                    for receipt in winning_receipts
                ],
            }
        )
    best = {
        "schema_version": 1,
        "campaign_id": contract["campaign"]["id"],
        "contract_sha256": contract_hash,
        "preflight_sha256": winner["preflight_sha256"],
        "candidate_id": winner["candidate_id"],
        # Never abbreviate either Git identity: this is the exact promotion
        # boundary consumed by a later campaign or release review.
        "candidate_commit": winner["candidate_commit"],
        "parent_commit": winner["parent_commit"],
        "verdict": winner["verdict"],
        "metric": selected_metric,
        "hard_gates": winner["hard_gates"],
        "workloads": workload_evidence,
        "evidence_candidate_ids": [
            receipt["candidate_id"] for receipt in winning_receipts
        ],
        "process_count": len(winning_process_identities),
        "required_process_count": required_processes,
        "eligible_process_receipt_count": len(eligible_receipts),
        "eligible_candidate_commit_count": len(eligible_groups),
        "journal_candidate_count": len(records),
        "release_proof_sha256": proof_hash,
        "release_proof_id": statement["proof_id"],
        "release_signer_id": statement["signer_id"],
        "release_sequence": statement["sequence"],
    }
    _atomic_write(output_path, canonical_json(best) + "\n")
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select BEST.json only from an exact externally signed receipt "
            "snapshot whose full-verdict hard gates all pass."
        )
    )
    parser.add_argument("contract", type=Path, help="exact approved TOML contract")
    parser.add_argument("results", type=Path, help="validated results.jsonl journal")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="BEST.json output path (atomically replaced)",
    )
    parser.add_argument(
        "--release-proof",
        type=Path,
        required=True,
        help="DSSE/Ed25519 proof issued by the contract-pinned external authority",
    )
    parser.add_argument(
        "--expected-challenge",
        required=True,
        help="fresh challenge supplied by the release pipeline",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        best = select_best(
            args.contract,
            args.results,
            args.output,
            release_proof=args.release_proof,
            expected_challenge=args.expected_challenge,
        )
    except (OSError, ValidationError) as error:
        print(
            json.dumps(
                {"schema_version": 1, "ok": False, "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(
        json.dumps(
            {
                "schema_version": 1,
                "ok": True,
                "candidate_id": best["candidate_id"],
                "candidate_commit": best["candidate_commit"],
                "metric": best["metric"],
            },
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
