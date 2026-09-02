#!/usr/bin/env python3
"""Produce a pointer-free public summary from a Chrome trace JSON file.

The normalized RhinoForge schema uses complete ``ph=X`` events.  The official
Rhino Launch SDK emits ``B``/``E`` duration pairs and ``s``/``f`` barrier-flow
events; a trusted profiler adapter must balance and normalize those events
before calling this sanitizer.  This module deliberately does not infer a
duration from an unpaired native trace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from campaign_common import (
    ValidationError,
    canonical_json,
    read_limited_bytes,
    safe_destination,
    strict_json_loads,
)


MAX_TRACE_BYTES = 128 * 1024 * 1024
MAX_EVENTS = 1_000_000
KERNEL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,254}")
HEX_ADDRESS_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])0x[0-9a-f]+")
RAW_ADDRESS_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9_])(?:[0-9a-f]{12,16}|[0-9]{10,20})(?![A-Za-z0-9_])"
)
UNIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_.-])/(?:[^\s|,;:()\[\]{}]+/)*[^\s|,;:()\[\]{}]*")
WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\(?:[^\s|,;:()\[\]{}]+\\)*[^\s|,;:()\[\]{}]*")
BYTE_KEYS = ("bytes", "byte_count", "bytes_transferred")
BANDWIDTH_KEYS = (
    "bandwidth_bytes_per_s",
    "bandwidth_Bps",
    "bytes_per_second",
)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValidationError(f"{label} must be finite and numeric") from error
    if not math.isfinite(result) or result < 0:
        raise ValidationError(f"{label} must be finite and non-negative")
    return result


def _private_group_label(
    value: Any, label: str, allowed_kernels: frozenset[str]
) -> tuple[str, str]:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValidationError(f"trace {label} must be a non-empty short string")
    if label == "name" and value in allowed_kernels:
        return f"allowed:{value}", value
    # All non-allowlisted values share one constant bucket.  A digest would
    # hide the literal while still leaking equality and group cardinality.
    return f"redacted:{label}", "<redacted>"


def _allowed_kernel_set(values: Any) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (str, bytes)):
        raise ValidationError("allowed_kernels must be a sequence of exact names")
    result: set[str] = set()
    try:
        items = list(values)
    except TypeError as error:
        raise ValidationError("allowed_kernels must be iterable") from error
    for value in items:
        if not isinstance(value, str) or KERNEL_NAME_RE.fullmatch(value) is None:
            raise ValidationError("allowed kernel name is invalid")
        if value in result:
            raise ValidationError("allowed kernel names must be unique")
        result.add(value)
    return frozenset(result)


def _one_public_numeric(
    arguments: dict[str, Any], keys: tuple[str, ...], label: str
) -> float | None:
    present = [key for key in keys if key in arguments]
    if not present:
        return None
    values = [_number(arguments[key], f"trace args.{key}") for key in present]
    if any(not math.isclose(values[0], value, rel_tol=1e-12, abs_tol=0.0) for value in values[1:]):
        raise ValidationError(f"conflicting public {label} fields")
    return values[0]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_trace(
    trace_path: Path,
    allowed_kernels: Any = (),
    *,
    event_category: str | None = None,
    required_trace_schema: str | None = None,
    expected_producer_sha256: str | None = None,
) -> dict[str, Any]:
    allowed_kernel_names = _allowed_kernel_set(allowed_kernels)
    if event_category is not None:
        if (
            not isinstance(event_category, str)
            or not event_category
            or len(event_category) > 128
            or any(ord(char) < 32 or ord(char) == 127 for char in event_category)
        ):
            raise ValidationError("event_category must be a non-empty short string")
    raw = read_limited_bytes(trace_path, MAX_TRACE_BYTES, "trace")
    try:
        parsed = strict_json_loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValidationError("trace is not UTF-8") from error
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValidationError("trace JSON is invalid") from error
    if isinstance(parsed, list):
        events = parsed
    elif isinstance(parsed, dict):
        events = parsed.get("traceEvents")
        if not isinstance(events, list):
            raise ValidationError("Chrome trace object must contain traceEvents list")
    else:
        raise ValidationError("Chrome trace root must be an object or list")
    trace_metadata: dict[str, Any] | None = None
    if required_trace_schema is not None or expected_producer_sha256 is not None:
        if not isinstance(parsed, dict):
            raise ValidationError("RPU trace schema requires an object root")
        if (
            not isinstance(required_trace_schema, str)
            or not required_trace_schema
            or expected_producer_sha256 is None
            or re.fullmatch(r"[0-9a-f]{64}", expected_producer_sha256) is None
            or event_category is None
        ):
            raise ValidationError("RPU trace schema arguments are incomplete")
        raw_metadata = parsed.get("rhinoforgeTrace")
        if not isinstance(raw_metadata, dict) or set(raw_metadata) != {
            "schema",
            "device_event_category",
            "device_program_events_exhaustive",
            "producer_sha256",
        }:
            raise ValidationError("RPU trace producer metadata schema is not exact")
        if raw_metadata.get("schema") != required_trace_schema:
            raise ValidationError("RPU trace schema identity mismatches")
        if raw_metadata.get("device_event_category") != event_category:
            raise ValidationError("RPU trace device-event category mismatches")
        if raw_metadata.get("device_program_events_exhaustive") is not True:
            raise ValidationError("RPU trace producer does not assert exhaustive launches")
        if raw_metadata.get("producer_sha256") != expected_producer_sha256:
            raise ValidationError("RPU trace producer SHA-256 mismatches")
        trace_metadata = dict(raw_metadata)
    if len(events) > MAX_EVENTS:
        raise ValidationError("trace contains too many events")

    groups: dict[tuple[str, str], dict[str, Any]] = {}
    complete_event_count = 0
    ignored_event_count = 0
    for index, event in enumerate(events):
        if not isinstance(event, dict):
            raise ValidationError(f"trace event {index} is not an object")
        phase = event.get("ph")
        if phase in {"B", "b", "E", "e"}:
            # The sanitizer must never turn an unbalanced native duration
            # stream into an apparently empty/valid result.  Native Launch
            # input belongs to normalize_hwperf.py first; normalized traces
            # contain only complete X events (plus optional s/f flows).
            raise ValidationError(
                "native B/E trace must be normalized before sanitization"
            )
        if phase != "X":
            ignored_event_count += 1
            continue
        if event_category is not None and event.get("cat") != event_category:
            if event.get("name") in allowed_kernel_names:
                raise ValidationError(
                    "allowlisted device program appears outside the frozen category"
                )
            ignored_event_count += 1
            continue
        if "dur" not in event:
            raise ValidationError(f"complete trace event {index} lacks duration")
        duration_us = _number(event["dur"], f"trace event {index} duration")
        name_group, public_name = _private_group_label(
            event.get("name"), "name", allowed_kernel_names
        )
        raw_category = event.get("cat", "uncategorized")
        category_group, public_category = _private_group_label(
            raw_category, "category", allowed_kernel_names
        )
        arguments = event.get("args", {})
        if not isinstance(arguments, dict):
            raise ValidationError(f"trace event {index} args must be an object")
        byte_count = _one_public_numeric(arguments, BYTE_KEYS, "bytes")
        bandwidth = _one_public_numeric(
            arguments, BANDWIDTH_KEYS, "bandwidth"
        )
        key = (name_group, category_group)
        group = groups.setdefault(
            key,
            {
                "durations": [],
                "bytes": [],
                "bandwidth": [],
                "public_name": public_name,
                "public_category": public_category,
            },
        )
        group["durations"].append(duration_us)
        derived_bandwidth: float | None = None
        if byte_count is not None:
            group["bytes"].append(byte_count)
            if duration_us > 0:
                derived_bandwidth = byte_count * 1_000_000.0 / duration_us
        if bandwidth is not None:
            if derived_bandwidth is not None and not math.isclose(
                bandwidth, derived_bandwidth, rel_tol=1e-6, abs_tol=1.0
            ):
                raise ValidationError(
                    f"trace event {index} bandwidth contradicts bytes/duration"
                )
            group["bandwidth"].append(bandwidth)
        elif derived_bandwidth is not None:
            group["bandwidth"].append(derived_bandwidth)
        complete_event_count += 1

    summaries: list[dict[str, Any]] = []
    for group_index, (_, group) in enumerate(sorted(groups.items()), 1):
        durations = group["durations"]
        summary: dict[str, Any] = {
            "group_id": f"event-group-{group_index}",
            "name": group["public_name"],
            "category": group["public_category"],
            "count": len(durations),
            "duration_us": {
                "total": math.fsum(durations),
                "mean": statistics.fmean(durations),
                "p50": float(statistics.median(durations)),
                "p95": _percentile(durations, 0.95),
                "min": min(durations),
                "max": max(durations),
            },
        }
        byte_values = group["bytes"]
        if byte_values:
            summary["bytes"] = {
                "event_count": len(byte_values),
                "total": math.fsum(byte_values),
                "mean": statistics.fmean(byte_values),
            }
        bandwidth_values = group["bandwidth"]
        if bandwidth_values:
            summary["bandwidth_bytes_per_s"] = {
                "sample_count": len(bandwidth_values),
                "mean": statistics.fmean(bandwidth_values),
                "p50": float(statistics.median(bandwidth_values)),
                "max": max(bandwidth_values),
            }
        summaries.append(summary)

    result = {
        "schema_version": 1,
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "complete_event_count": complete_event_count,
        "ignored_event_count": ignored_event_count,
        "events": summaries,
    }
    if trace_metadata is not None:
        result["trace_producer"] = trace_metadata
    encoded = canonical_json(result)
    if (
        HEX_ADDRESS_RE.search(encoded)
        or UNIX_PATH_RE.search(encoded)
        or WINDOWS_PATH_RE.search(encoded)
    ):
        raise ValidationError("public summary redaction invariant failed")
    return result


def _file_state(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _optional_lstat(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise ValidationError("summary output cannot be inspected") from error


def _atomic_write(
    path: Path,
    data: str,
    *,
    forbidden: tuple[Path | str | None, ...] = (),
) -> None:
    """Atomically write a summary without following output/input symlinks.

    The destination and its parent are checked before and after creating the
    temporary file.  A parent or existing destination replacement during the
    write is treated as an integrity failure rather than silently redirecting
    the result.
    """

    destination = safe_destination(
        path,
        forbidden=forbidden,
        create_parent=True,
        label="summary output",
    )
    parent = destination.parent
    try:
        parent_before = os.lstat(parent)
    except (OSError, ValueError) as error:
        raise ValidationError("summary output parent is unavailable") from error
    if not stat.S_ISDIR(parent_before.st_mode):
        raise ValidationError("summary output parent must be a regular directory")
    destination_before = _optional_lstat(destination)
    if destination_before is not None and (
        stat.S_ISLNK(destination_before.st_mode)
        or not stat.S_ISREG(destination_before.st_mode)
    ):
        raise ValidationError("summary output must be a regular non-symlink file")

    try:
        encoded = data.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValidationError("summary output is not UTF-8") from error
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    descriptor_state = os.fstat(descriptor)
    try:
        os.fchmod(descriptor, 0o644)
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("short summary write")
            view = view[written:]
        os.fsync(descriptor)

        # Verify the directory and destination identities immediately before
        # replacement.  Directory mtime changes when the temporary is created,
        # so identity is intentionally limited to device/inode for the parent.
        try:
            parent_after = os.lstat(parent)
        except (OSError, ValueError) as error:
            raise ValidationError("summary output parent changed") from error
        if (parent_after.st_dev, parent_after.st_ino) != (
            parent_before.st_dev,
            parent_before.st_ino,
        ) or stat.S_ISLNK(parent_after.st_mode) or not stat.S_ISDIR(parent_after.st_mode):
            raise ValidationError("summary output parent changed")

        destination_after = _optional_lstat(destination)
        if destination_before is None:
            if destination_after is not None:
                raise ValidationError("summary output changed during write")
        elif destination_after is None or _file_state(destination_after) != _file_state(
            destination_before
        ):
            raise ValidationError("summary output changed during write")
        temporary_after = _optional_lstat(temporary)
        if temporary_after is None or not stat.S_ISREG(temporary_after.st_mode) or (
            temporary_after.st_dev,
            temporary_after.st_ino,
        ) != (descriptor_state.st_dev, descriptor_state.st_ino):
            raise ValidationError("summary temporary changed during write")

        os.replace(temporary, destination)
        try:
            directory_descriptor = os.open(
                parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
        except OSError:
            directory_descriptor = -1
        if directory_descriptor >= 0:
            try:
                os.fsync(directory_descriptor)
            finally:
                os.close(directory_descriptor)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Chrome complete events using anonymous groups and only "
            "duration/bytes/bandwidth aggregates. Labels, addresses, paths, "
            "timestamps, process/thread IDs, and all other args are omitted."
        )
    )
    parser.add_argument("trace", type=Path, help="Chrome trace JSON")
    parser.add_argument(
        "--allowed-kernel",
        action="append",
        default=[],
        help=(
            "exact reviewed public kernel name allowed to remain visible; "
            "repeat for additional kernels"
        ),
    )
    parser.add_argument(
        "--event-category",
        help="include only complete events with this exact Chrome category",
    )
    parser.add_argument("--trace-schema")
    parser.add_argument("--producer-sha256")
    parser.add_argument("--output", type=Path, help="optional atomic JSON output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = summarize_trace(
            args.trace,
            args.allowed_kernel,
            event_category=args.event_category,
            required_trace_schema=args.trace_schema,
            expected_producer_sha256=args.producer_sha256,
        )
        encoded = canonical_json(summary) + "\n"
        if args.output:
            _atomic_write(args.output, encoded, forbidden=(args.trace,))
        else:
            sys.stdout.write(encoded)
    except (OSError, ValidationError) as error:
        print(
            json.dumps(
                {"schema_version": 1, "ok": False, "error": str(error)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
