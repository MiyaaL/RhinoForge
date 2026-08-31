#!/usr/bin/env python3
"""Compute an exact-profile roofline and paired fusion-tax confidence interval."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import statistics
import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from campaign_common import paired_median_bootstrap_ci


SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+@/-]{0,255}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
UNRESOLVED_SOURCE_MARKERS = (
    "todo",
    "tbd",
    "unknown",
    "unresolved",
    "replace-with",
)
MAX_SOURCE_BYTES = 64 * 1024 * 1024


def _positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _source_identity(
    source_id: str, source_sha256: str, peak_kind: str
) -> dict[str, str]:
    if not isinstance(source_id, str) or SOURCE_ID_RE.fullmatch(source_id) is None:
        raise ValueError(
            "source_id must be an auditable identifier using safe characters"
        )
    normalized_id = source_id.lower()
    if normalized_id == "none" or any(
        marker in normalized_id for marker in UNRESOLVED_SOURCE_MARKERS
    ):
        raise ValueError("source_id must identify reviewed evidence")
    if not isinstance(source_sha256, str) or SHA256_RE.fullmatch(source_sha256) is None:
        raise ValueError("source_sha256 must be exactly 64 lowercase hex characters")
    return {
        "kind": (
            "authoritative_specification"
            if peak_kind == "authoritative"
            else "empirical_calibration"
        ),
        "id": source_id,
        "sha256": source_sha256,
    }


def _source_file_sha256(path: Path) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("source_file is unavailable") from error
    try:
        file_status = os.fstat(descriptor)
        if not stat.S_ISREG(file_status.st_mode):
            raise ValueError("source_file must be a regular file")
        if file_status.st_size > MAX_SOURCE_BYTES:
            raise ValueError("source_file exceeds 64 MiB")
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    finally:
        os.close(descriptor)


def roofline(
    *,
    required_ops: float,
    mandatory_bytes: float,
    peak_ops_per_s: float,
    bandwidth_bytes_per_s: float,
    launch_floor_ns: float,
    latency_ns: float,
    peak_kind: str,
    source_id: str,
    source_sha256: str,
) -> dict[str, object]:
    required_ops = _positive("required_ops", required_ops)
    mandatory_bytes = _positive("mandatory_bytes", mandatory_bytes)
    peak_ops_per_s = _positive("peak_ops_per_s", peak_ops_per_s)
    bandwidth_bytes_per_s = _positive(
        "bandwidth_bytes_per_s", bandwidth_bytes_per_s
    )
    launch_floor_ns = _positive("launch_floor_ns", launch_floor_ns)
    latency_ns = _positive("latency_ns", latency_ns)
    if peak_kind not in {"authoritative", "empirical"}:
        raise ValueError("peak_kind must be authoritative or empirical")
    source = _source_identity(source_id, source_sha256, peak_kind)

    compute_floor_ns = required_ops / peak_ops_per_s * 1e9
    memory_floor_ns = mandatory_bytes / bandwidth_bytes_per_s * 1e9
    lower_bound_ns = max(compute_floor_ns, memory_floor_ns, launch_floor_ns)
    efficiency = lower_bound_ns / latency_ns
    _positive("compute_floor_ns", compute_floor_ns)
    _positive("memory_floor_ns", memory_floor_ns)
    _positive("lower_bound_ns", lower_bound_ns)
    _positive("efficiency", efficiency)
    if efficiency > 1.0:
        raise ValueError(
            "latency_ns is below the registered roofline floor; efficiency exceeds 1"
        )
    return {
        "peak_kind": peak_kind,
        "roofline_label": (
            "theoretical"
            if peak_kind == "authoritative"
            else "empirical roofline"
        ),
        "compute_floor_ns": compute_floor_ns,
        "memory_floor_ns": memory_floor_ns,
        "launch_floor_ns": launch_floor_ns,
        "lower_bound_ns": lower_bound_ns,
        "latency_ns": latency_ns,
        "efficiency": efficiency,
        "source": source,
    }


def _parse_samples(text: str) -> list[float]:
    values = [float(item) for item in text.split(",") if item.strip()]
    if len(values) < 5:
        raise ValueError("paired fusion analysis requires at least five samples")
    for value in values:
        _positive("sample", value)
    return values


def fusion_tax(
    fused_ns: Sequence[float],
    bare_ns: Sequence[float],
    *,
    confidence: float,
    bootstrap_trials: int,
    seed: int,
    max_tax_pct: float | None,
) -> dict[str, object]:
    if len(fused_ns) != len(bare_ns):
        raise ValueError("fused and bare samples must be paired and equal length")
    if len(fused_ns) < 5:
        raise ValueError("paired fusion analysis requires at least five samples")
    fused_ns = [_positive("fused sample", value) for value in fused_ns]
    bare_ns = [_positive("bare sample", value) for value in bare_ns]
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    if bootstrap_trials < 1000:
        raise ValueError("bootstrap_trials must be at least 1000")

    count = len(fused_ns)
    paired_pct = [
        100.0 * (fused - bare) / bare
        for fused, bare in zip(fused_ns, bare_ns)
    ]
    lower, upper = paired_median_bootstrap_ci(
        fused_ns,
        bare_ns,
        confidence=confidence,
        trials=bootstrap_trials,
        seed=seed,
    )
    result: dict[str, object] = {
        "sample_count": count,
        "statistic": "median_of_paired_epilogue_tax_pct",
        "epilogue_tax_p50_pct": float(statistics.median(paired_pct)),
        "epilogue_tax_ci_lower_pct": lower,
        "epilogue_tax_ci_upper_pct": upper,
        "epilogue_tax_confidence": confidence,
        "epilogue_tax_bootstrap_trials": bootstrap_trials,
        "epilogue_tax_bootstrap_seed": seed,
    }
    if max_tax_pct is not None:
        if not math.isfinite(max_tax_pct) or max_tax_pct < 0:
            raise ValueError("max_tax_pct must be finite and non-negative")
        result["registered_max_epilogue_tax_pct"] = max_tax_pct
        result["indistinguishable_at_registered_resolution"] = (
            upper <= max_tax_pct
        )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-ops", type=float, required=True)
    parser.add_argument("--mandatory-bytes", type=float, required=True)
    parser.add_argument("--peak-ops-per-s", type=float, required=True)
    parser.add_argument("--bandwidth-bytes-per-s", type=float, required=True)
    parser.add_argument("--launch-floor-ns", type=float, required=True)
    parser.add_argument("--latency-ns", type=float, required=True)
    parser.add_argument(
        "--peak-kind", choices=("authoritative", "empirical"), required=True
    )
    parser.add_argument(
        "--source-id",
        "--peak-source-id",
        "--calibration-source-id",
        dest="source_id",
        required=True,
        help="reviewable specification or calibration receipt identifier",
    )
    parser.add_argument(
        "--source-sha256",
        "--peak-source-sha256",
        "--calibration-source-sha256",
        dest="source_sha256",
        required=True,
        help="SHA-256 of the exact specification or calibration receipt bytes",
    )
    parser.add_argument(
        "--source-file",
        type=Path,
        required=True,
        help="local reviewed source bytes (hashed only; path is never emitted)",
    )
    parser.add_argument("--fused-samples-ns")
    parser.add_argument("--bare-samples-ns")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-trials", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument(
        "--max-epilogue-tax-pct",
        "--max-epilogue-tax",
        dest="max_epilogue_tax_pct",
        type=float,
        help="registered upper CI gate in percentage points",
    )
    args = parser.parse_args(argv)

    try:
        measured_source_sha256 = _source_file_sha256(args.source_file)
        if measured_source_sha256 != args.source_sha256:
            raise ValueError("source_sha256 does not match source_file bytes")
        output: dict[str, object] = {
            "schema_version": 1,
            "roofline": roofline(
                required_ops=args.required_ops,
                mandatory_bytes=args.mandatory_bytes,
                peak_ops_per_s=args.peak_ops_per_s,
                bandwidth_bytes_per_s=args.bandwidth_bytes_per_s,
                launch_floor_ns=args.launch_floor_ns,
                latency_ns=args.latency_ns,
                peak_kind=args.peak_kind,
                source_id=args.source_id,
                source_sha256=args.source_sha256,
            ),
        }
        if (args.fused_samples_ns is None) != (args.bare_samples_ns is None):
            raise ValueError("provide both fused and bare sample lists")
        if args.fused_samples_ns is not None:
            output["fusion"] = fusion_tax(
                _parse_samples(args.fused_samples_ns),
                _parse_samples(args.bare_samples_ns),
                confidence=args.confidence,
                bootstrap_trials=args.bootstrap_trials,
                seed=args.bootstrap_seed,
                max_tax_pct=args.max_epilogue_tax_pct,
            )
    except ValueError as exc:
        parser.exit(2, f"analysis failed: {exc}\n")

    print(json.dumps(output, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
