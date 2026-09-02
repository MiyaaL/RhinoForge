#!/usr/bin/env python3
"""Normalize native Rhino Launch ``B``/``E`` traces to skill ``ph=X`` JSON.

The Launch SDK records a START/END pair for each kernel or DMA and optional
barrier flow events.  ``summarize_hwperf.py`` intentionally accepts only the
skill's normalized complete-event form.  This converter performs strict,
pointer-free pairing in a private directory. It does not mark the result
exhaustive unless the caller explicitly passes ``--assert-exhaustive`` after
an independent launch-coverage check, and it never replaces the verifier's
manifest checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from campaign_common import ValidationError, canonical_json, strict_json_loads


MAX_TRACE_BYTES = 128 * 1024 * 1024
MAX_OUTPUT_BYTES = 128 * 1024 * 1024
MAX_EVENTS = 1_000_000
MAX_METADATA_INTEGER = (1 << 64) - 1
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
BYTE_KEYS = ("bytes", "byte_count", "bytes_transferred", "size_bytes")
BANDWIDTH_KEYS = (
    "bandwidth_bytes_per_s",
    "bandwidth_Bps",
    "bytes_per_second",
    "bandwidth_GBps",
)
NATIVE_COMPUTE_CATEGORIES = frozenset(
    {"compute", "kernel", "device_program", "rpu_device_program"}
)
NATIVE_DMA_CATEGORIES = frozenset({"dma", "rpu_dma"})
# Chrome metadata/instant phases do not represent a duration.  Unknown phases
# are rejected so an unrecognized launch cannot disappear under an exhaustive
# claim.
IGNORABLE_PHASES = frozenset({"M", "m", "I", "i", "C", "c", "N", "n", "O", "o", "P", "p", "V", "R"})


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise ValidationError(f"{label} must be finite and non-negative") from error
    if not math.isfinite(result) or result < 0:
        raise ValidationError(f"{label} must be finite and non-negative")
    return result


def _metadata_integer(value: Any, label: str) -> int:
    """Parse the SDK's unsigned integer ``otherData`` fields exactly.

    ``dump_hw_perf_chrome`` formats these fields with ``%u``/``%zu``/``%lld``
    but emits JSON strings.  Converting a large string through ``float`` can
    silently round it (and an integral JSON float such as ``1.0`` is not an
    authentic producer value), so keep the representation integer and bounded
    from the outset.
    """

    if isinstance(value, bool):
        raise ValidationError(f"{label} must be a non-negative integer")
    if isinstance(value, int):
        result = value
    elif isinstance(value, str):
        if not value or re.fullmatch(r"[0-9]+", value) is None:
            raise ValidationError(f"{label} must be a decimal integer")
        # Avoid handing an unbounded digit string to int() (and the Python
        # 3.10/3.11 max-digit parser) before applying the protocol bound.
        if len(value) > 20:
            raise ValidationError(f"{label} is out of range")
        try:
            result = int(value, 10)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValidationError(f"{label} must be a decimal integer") from error
    else:
        raise ValidationError(f"{label} must be a non-negative integer")
    if result < 0 or result > MAX_METADATA_INTEGER:
        raise ValidationError(f"{label} is out of range")
    return result


def _optional_identity(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _short_string(value: Any, label: str, *, required: bool = True) -> str | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value) or len(value) > 512:
        raise ValidationError(f"{label} must be a non-empty short string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValidationError(f"{label} contains a control character")
    return value


def _identity(value: Any, label: str) -> tuple[str, Any]:
    """Return a hashable, typed Chrome identity component."""

    if value is None:
        return ("none", None)
    if isinstance(value, bool):
        raise ValidationError(f"trace {label} must not be boolean")
    if isinstance(value, int):
        if abs(value) > (1 << 63) - 1:
            raise ValidationError(f"trace {label} is out of range")
        return ("int", value)
    if isinstance(value, str):
        checked = _short_string(value, f"trace {label}")
        assert checked is not None
        return ("str", checked)
    raise ValidationError(f"trace {label} must be a string, integer, or null")


def _event_identity(event: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    return tuple(_identity(event.get(key), key) for key in ("pid", "tid", "id"))


def _event_key(event: dict[str, Any], name: str) -> tuple[Any, ...]:
    # Category is deliberately excluded: native E records may omit it or use
    # a different spelling from B, while pid/tid/id/name still identify the
    # duration stack.
    return (*_event_identity(event), name)


def _category_target(
    raw_category: Any,
    event_category: str,
    dma_event_category: str,
    compute_categories: frozenset[str],
    dma_categories: frozenset[str],
) -> str:
    category = _short_string(raw_category, "trace event category")
    assert category is not None
    folded = category.casefold()
    if folded in compute_categories or category == event_category:
        return event_category
    if folded in dma_categories or category == dma_event_category:
        return dma_event_category
    raise ValidationError(
        f"unsupported native duration category {category!r}; configure an explicit mapping"
    )


def _public_args(event: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    raw = event.get("args", {})
    if raw is None:
        return {}, {"bytes": None, "bandwidth": None, "bandwidth_gbps": None}
    if not isinstance(raw, dict):
        raise ValidationError("trace event args must be an object")
    byte_values: list[float] = []
    bandwidth_values: list[float] = []
    native_gbps: list[float] = []
    for key in BYTE_KEYS:
        if key in raw:
            byte_values.append(_number(raw[key], f"trace args.{key}"))
    for key in BANDWIDTH_KEYS:
        if key not in raw:
            continue
        value = _number(raw[key], f"trace args.{key}")
        if key == "bandwidth_GBps":
            native_gbps.append(value)
            value *= 1_000_000_000.0
            if not math.isfinite(value):
                raise ValidationError("trace args.bandwidth_GBps is too large")
        bandwidth_values.append(value)

    def one(values: list[float], label: str) -> float | None:
        if not values:
            return None
        if any(not math.isclose(values[0], item, rel_tol=1e-9, abs_tol=1e-6) for item in values[1:]):
            raise ValidationError(f"conflicting public {label} fields")
        return values[0]

    bytes_value = one(byte_values, "bytes")
    bandwidth_value = one(bandwidth_values, "bandwidth")
    result: dict[str, float] = {}
    if bytes_value is not None:
        result["bytes"] = bytes_value
    if bandwidth_value is not None:
        result["bandwidth_bytes_per_s"] = bandwidth_value
    return result, {
        "bytes": bytes_value,
        "bandwidth": bandwidth_value,
        "bandwidth_gbps": native_gbps[0] if native_gbps else None,
    }


def _timing_fields(event: dict[str, Any], label: str) -> dict[str, float]:
    """Validate and retain timing fields until a B/E duration is known.

    Native Launch records normally put these fields on ``E``.  Accepting them
    on ``B`` is useful for alternate producers, but they must be checked
    against the same timestamp delta once the pair is closed; otherwise a
    forged begin-side duration could be silently ignored.
    """

    raw = event.get("args", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValidationError(f"{label} args must be an object")
    result: dict[str, float] = {}
    for key in ("duration_us", "duration_cycles", "bandwidth_GBps"):
        if key in raw:
            result[key] = _number(raw[key], f"{label}.{key}")
    return result


def _merge_args(
    first: dict[str, float], second: dict[str, float], duration_us: float
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in ("bytes", "bandwidth_bytes_per_s"):
        values = [item[key] for item in (first, second) if key in item]
        if values and any(
            not math.isclose(values[0], value, rel_tol=1e-9, abs_tol=1e-6)
            for value in values[1:]
        ):
            raise ValidationError(f"conflicting begin/end {key} fields")
        if values:
            result[key] = values[0]
    # The SDK prints bandwidth_GBps and duration_us to three decimals.  Those
    # rounded values can disagree very slightly with the timestamp delta, so
    # use exact bytes/dur for the normalized sanitizer and retain the native
    # rounded value in the public normalization report.
    if "bytes" in result and duration_us > 0:
        result["bandwidth_bytes_per_s"] = result["bytes"] * 1_000_000.0 / duration_us
    return result


def _validate_timing_values(
    raw: dict[str, float],
    duration_us: float,
    frequency_hz: float | None,
    label: str,
) -> None:
    declared_us: float | None = None
    if "duration_us" in raw:
        declared_us = raw["duration_us"]
        # Timestamps and SDK duration_us are printed to three decimals.
        if not math.isclose(declared_us, duration_us, rel_tol=0.0, abs_tol=0.0021):
            raise ValidationError(f"{label}.duration_us contradicts B/E timestamps")
    cycles: float | None = None
    if "duration_cycles" in raw:
        cycles = raw["duration_cycles"]
    if cycles is not None and frequency_hz is not None and frequency_hz > 0:
        cycles_us = cycles * 1_000_000.0 / frequency_hz
        reference_us = declared_us if declared_us is not None else duration_us
        if not math.isclose(cycles_us, reference_us, rel_tol=0.0, abs_tol=0.0021):
            raise ValidationError(f"{label}.duration_cycles contradicts frequency/timing")


def _validate_timing_args(
    raw_event: dict[str, Any],
    duration_us: float,
    frequency_hz: float | None,
    label: str,
) -> None:
    _validate_timing_values(
        _timing_fields(raw_event, label), duration_us, frequency_hz, label
    )


def _minimal_flow(event: dict[str, Any]) -> dict[str, Any]:
    phase = event.get("ph")
    if phase not in {"s", "f"}:
        raise ValidationError("trace flow phase is invalid")
    result: dict[str, Any] = {"ph": phase}
    name = _short_string(event.get("name"), "trace flow name", required=False)
    category = _short_string(event.get("cat", "SYNC"), "trace flow category")
    if name is not None:
        result["name"] = name
    result["cat"] = category
    result["ts"] = _number(event.get("ts"), "trace flow timestamp")
    for key in ("pid", "tid", "id"):
        if key == "id" and event.get(key) is None:
            raise ValidationError("trace flow id is required")
        value = event.get(key)
        _identity(value, key)
        if value is not None:
            result[key] = value
    return result


def _other_data_report(trace: dict[str, Any]) -> dict[str, Any]:
    raw = trace.get("otherData")
    if raw is None:
        return {"present": False}
    if not isinstance(raw, dict):
        raise ValidationError("trace otherData must be an object")
    result: dict[str, Any] = {"present": True}
    for key in ("frequency_hz", "base_cycle", "batch_events", "records_captured"):
        if key not in raw:
            continue
        value = _metadata_integer(raw[key], f"otherData.{key}")
        result[key] = value
    if "frequency_hz" in result and result["frequency_hz"] <= 0:
        raise ValidationError("otherData.frequency_hz must be positive")
    # Keep only a fixed, non-sensitive source marker; arbitrary root metadata
    # can contain paths or addresses.
    source = raw.get("source")
    if source is not None:
        if source != "rhino-launch-kernel HW perf trace":
            raise ValidationError("otherData.source is not the official Rhino Launch marker")
        result["source"] = source
    return result


def _lexical_absolute(path: Path) -> Path:
    """Make an absolute path without collapsing ``..`` or following links."""

    value = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(value):
        value = os.path.join(os.getcwd(), value)
    return Path(value)


def _reject_symlink_components(path: Path) -> None:
    """Reject symlinks in an existing lexical path prefix.

    This check intentionally runs before ``abspath``/``realpath``.  Collapsing
    ``link/../file`` first would hide the link and permit a pathname to escape
    the reviewed directory.  Missing components are left for the regular
    unavailable/not-a-file diagnostics below.
    """

    current = Path(path.anchor) if path.anchor else Path(".")
    parts = path.parts[1:] if path.anchor else path.parts
    missing = False
    for component_index, part in enumerate(parts):
        if part in {"", "."}:
            continue
        # ``abspath``/``normpath`` collapses ``missing/..``.  That changes the
        # kernel's pathname semantics (the kernel would fail at ``missing``)
        # and could make a reviewed path resolve to an unintended file.  Keep
        # lexical validation strict instead of silently accepting that rewrite.
        if missing:
            if part == "..":
                raise ValidationError("native trace path contains an unavailable component")
            # Once a component is missing, later components cannot be checked
            # with lstat.  A final regular-file/parent diagnostic will report
            # ordinary unavailable paths; no ``..`` rewrite is permitted.
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            missing = True
            continue
        except OSError as error:
            raise ValidationError("native trace path cannot be inspected") from error
        # Leave the final component for the caller's lstat/open check so it
        # can retain the precise regular-file diagnostic.  Every parent
        # component (including one hidden before ``..``) is rejected here.
        if stat.S_ISLNK(info.st_mode) and component_index != len(parts) - 1:
            raise ValidationError("native trace path must not contain symbolic links")


def _read_native_bytes(path: Path) -> bytes:
    """Read a raw trace through an O_NOFOLLOW descriptor with identity checks."""

    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical)
    absolute = _absolute_path(lexical)
    try:
        initial = os.lstat(absolute)
    except OSError as error:
        raise ValidationError("native trace is unavailable") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise ValidationError("native trace must be a regular non-symlink file")
    if initial.st_size > MAX_TRACE_BYTES:
        raise ValidationError(f"native trace exceeds {MAX_TRACE_BYTES} bytes")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValidationError("native trace cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        initial_identity = (
            initial.st_dev,
            initial.st_ino,
            initial.st_size,
            initial.st_mtime_ns,
            initial.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (initial.st_dev, initial.st_ino)
            or opened.st_size != initial.st_size
            or opened.st_mtime_ns != initial.st_mtime_ns
            or opened.st_ctime_ns != initial.st_ctime_ns
        ):
            raise ValidationError("native trace changed before reading")
        # A pre-check cannot prevent an intermediate directory from being
        # replaced by a symlink between lstat and open.  On Linux, compare the
        # descriptor's procfs target with the reviewed lexical path when that
        # facility is available; O_NOFOLLOW still protects the final entry.
        proc_fd = f"/proc/self/fd/{descriptor}"
        if os.path.lexists(proc_fd):
            try:
                fd_target = os.path.realpath(proc_fd)
            except OSError:
                fd_target = ""
            if fd_target and fd_target != os.path.realpath(os.fspath(absolute)):
                raise ValidationError("native trace path changed before reading")
        chunks: list[bytes] = []
        observed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            if observed > MAX_TRACE_BYTES:
                raise ValidationError(f"native trace exceeds {MAX_TRACE_BYTES} bytes")
            chunks.append(block)
        final = os.fstat(descriptor)
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        if observed != opened.st_size or final_identity != initial_identity:
            raise ValidationError("native trace changed while reading")
        return b"".join(chunks)
    except OSError as error:
        raise ValidationError("native trace could not be read") from error
    finally:
        os.close(descriptor)


def normalize_trace(
    trace: Any,
    *,
    producer_sha256: str,
    schema: str = "rhinoforge-rpu-chrome-v1",
    event_category: str = "rpu_device_program",
    dma_event_category: str = "rpu_dma",
    assert_exhaustive: bool = False,
    require_frequency: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(normalized_trace, public_report)`` after strict B/E pairing."""

    producer_sha256 = _optional_identity(producer_sha256, "producer_sha256")
    schema_value = _short_string(schema, "schema")
    event_category_value = _short_string(event_category, "event_category")
    dma_category_value = _short_string(dma_event_category, "dma_event_category")
    assert schema_value is not None and event_category_value is not None
    assert dma_category_value is not None
    if event_category_value.casefold() == dma_category_value.casefold():
        raise ValidationError("event_category and dma_event_category must differ")
    if isinstance(trace, list):
        events = trace
        other_data = {"present": False}
    elif isinstance(trace, dict):
        events = trace.get("traceEvents")
        if not isinstance(events, list):
            raise ValidationError("trace object must contain traceEvents list")
        other_data = _other_data_report(trace)
    else:
        raise ValidationError("trace root must be an object or list")
    if len(events) > MAX_EVENTS:
        raise ValidationError("trace contains too many events")
    if require_frequency and "frequency_hz" not in other_data:
        raise ValidationError("trace otherData.frequency_hz is required")
    frequency_hz = other_data.get("frequency_hz")

    compute_categories = NATIVE_COMPUTE_CATEGORIES | {event_category_value.casefold()}
    dma_categories = NATIVE_DMA_CATEGORIES | {dma_category_value.casefold()}
    stacks: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    # Keep an inverted index for open duration stacks.  The native producer
    # commonly emits one distinct name per batch entry; scanning ``stacks``
    # for every END event would therefore make normalization quadratic in the
    # number of launches.  END fields remain wildcards when omitted (the
    # historical pairing contract), so index each identity component and
    # intersect the smallest matching bucket below.
    index_positions = {"pid": 0, "tid": 1, "id": 2, "name": 3}
    open_index: dict[str, dict[Any, set[tuple[Any, ...]]]] = {
        field: {} for field in index_positions
    }
    open_keys: set[tuple[Any, ...]] = set()

    def add_open_key(key: tuple[Any, ...]) -> None:
        if key in open_keys:
            return
        open_keys.add(key)
        for field, position in index_positions.items():
            value = key[position]
            open_index[field].setdefault(value, set()).add(key)

    def remove_open_key(key: tuple[Any, ...]) -> None:
        if key not in open_keys:
            return
        open_keys.remove(key)
        for field, position in index_positions.items():
            value = key[position]
            bucket = open_index[field].get(value)
            if bucket is None:
                continue
            bucket.discard(key)
            if not bucket:
                del open_index[field][value]
    normalized_events: list[dict[str, Any]] = []
    boundary_count = 0
    begin_count = 0
    complete_count = 0
    flow_count = 0
    passthrough_count = 0
    phase_seen: set[str] = set()
    ignored_phase_counts: dict[str, int] = {}
    open_flow_ids: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    native_category_counts: dict[str, int] = {}
    normalized_category_counts: dict[str, int] = {}
    stream_ids: set[tuple[str, Any]] = set()
    native_bytes_count = 0
    native_bandwidth_count = 0
    native_bandwidth_sum = 0.0
    native_bandwidth_min: float | None = None
    native_bandwidth_max: float | None = None
    weak_pair_identity_count = 0

    def record_args(evidence: dict[str, Any]) -> None:
        nonlocal native_bytes_count, native_bandwidth_count
        nonlocal native_bandwidth_sum, native_bandwidth_min, native_bandwidth_max
        if evidence.get("bytes") is not None:
            native_bytes_count += 1
        gbps = evidence.get("bandwidth_gbps")
        if gbps is not None:
            native_bandwidth_count += 1
            native_bandwidth_sum += gbps
            native_bandwidth_min = gbps if native_bandwidth_min is None else min(native_bandwidth_min, gbps)
            native_bandwidth_max = gbps if native_bandwidth_max is None else max(native_bandwidth_max, gbps)

    def matching_candidates(
        raw_event: dict[str, Any], name: str | None
    ) -> list[tuple[tuple[Any, ...], list[dict[str, Any]]]]:
        constraints: list[tuple[str, Any]] = []
        if name is not None:
            constraints.append(("name", name))
        identity_values = {
            field: _identity(raw_event.get(field), field)
            for field in ("pid", "tid", "id")
            if field in raw_event and raw_event.get(field) is not None
        }
        constraints.extend(
            (field, value) for field, value in identity_values.items()
        )
        if not constraints:
            candidate_keys = open_keys
        else:
            buckets = [
                open_index[field].get(value, set())
                for field, value in constraints
            ]
            if any(not bucket for bucket in buckets):
                return []
            # Intersecting the smallest bucket bounds work for partial
            # wildcard END records.  Once two candidates are found, the
            # caller will reject the END as ambiguous, so no larger list is
            # needed.
            candidate_keys = min(buckets, key=len)
        result: list[tuple[tuple[Any, ...], list[dict[str, Any]]]] = []
        for key in candidate_keys:
            if any(key[index_positions[field]] != value for field, value in constraints):
                continue
            stack = stacks.get(key)
            if not stack:
                continue
            result.append((key, stack))
            if len(result) >= 2:
                break
        return result

    for index, raw_event in enumerate(events):
        if not isinstance(raw_event, dict):
            raise ValidationError(f"trace event {index} is not an object")
        phase = raw_event.get("ph")
        if not isinstance(phase, str) or not phase:
            raise ValidationError(f"trace event {index} phase must be a string")
        phase_seen.add(phase)
        if phase in {"B", "b"}:
            name = _short_string(raw_event.get("name"), f"trace begin event {index} name")
            assert name is not None
            start = _number(raw_event.get("ts"), f"trace begin event {index} timestamp")
            raw_category = _short_string(raw_event.get("cat"), f"trace begin event {index} category")
            assert raw_category is not None
            target_category = _category_target(
                raw_category,
                event_category_value,
                dma_category_value,
                compute_categories,
                dma_categories,
            )
            event_identity = _event_identity(raw_event)
            if all(component[0] == "none" for component in event_identity):
                weak_pair_identity_count += 1
                if assert_exhaustive:
                    raise ValidationError(
                        "exhaustive traces require pid, tid, or id on every begin event"
                    )
            stream_ids.add(_identity(raw_event.get("tid"), "tid"))
            args, evidence = _public_args(raw_event)
            timing = _timing_fields(raw_event, f"trace begin event {index}")
            record_args(evidence)
            key = _event_key(raw_event, name)
            add_open_key(key)
            stacks.setdefault(key, []).append(
                {
                    "name": name,
                    "raw_category": raw_category,
                    "category": target_category,
                    "ts": start,
                    "args": args,
                    "evidence": evidence,
                    "timing": timing,
                    "identity": event_identity,
                }
            )
            native_category_counts[raw_category] = native_category_counts.get(raw_category, 0) + 1
            begin_count += 1
            boundary_count += 1
            continue
        if phase in {"E", "e"}:
            name = _short_string(
                raw_event.get("name"),
                f"trace end event {index} name",
                required=False,
            )
            candidates = matching_candidates(raw_event, name)
            if len(candidates) != 1:
                raise ValidationError(f"trace end event {index} cannot identify one open begin")
            key, stack = candidates[0]
            begin = stack.pop()
            if not stack:
                del stacks[key]
                remove_open_key(key)
            end = _number(raw_event.get("ts"), f"trace end event {index} timestamp")
            duration = end - float(begin["ts"])
            if not math.isfinite(duration) or duration < 0:
                raise ValidationError(f"trace end event {index} precedes its begin")
            end_category_value = raw_event.get("cat")
            if end_category_value is not None:
                end_category = _short_string(end_category_value, f"trace end event {index} category")
                assert end_category is not None
                if _category_target(end_category, event_category_value, dma_category_value, compute_categories, dma_categories) != begin["category"]:
                    raise ValidationError(f"trace end event {index} category mismatches begin")
            end_args, end_evidence = _public_args(raw_event)
            end_timing = _timing_fields(raw_event, f"trace end event {index}")
            record_args(end_evidence)
            # Validate timing declarations on both sides of the pair.  The
            # official producer places them on E, while alternate producers
            # may place them on B or repeat them on both sides.
            if begin.get("timing"):
                _validate_timing_values(
                    begin["timing"],
                    duration,
                    frequency_hz if isinstance(frequency_hz, int) else None,
                    f"trace begin for event {index}",
                )
            _validate_timing_values(
                end_timing,
                duration,
                frequency_hz if isinstance(frequency_hz, int) else None,
                f"trace end event {index}",
            )
            begin_gbps = begin["evidence"].get("bandwidth_gbps")
            end_gbps = end_evidence.get("bandwidth_gbps")
            if (
                begin_gbps is not None
                and end_gbps is not None
                and not math.isclose(begin_gbps, end_gbps, rel_tol=1e-9, abs_tol=1e-6)
            ):
                raise ValidationError(
                    f"trace end event {index}.bandwidth_GBps conflicts with begin"
                )
            normalized_args = _merge_args(begin["args"], end_args, duration)
            native_gbps = end_gbps if end_gbps is not None else begin_gbps
            if native_gbps is not None and normalized_args.get("bytes") is not None and duration > 0:
                derived_gbps = normalized_args["bytes"] * 1_000_000.0 / duration / 1_000_000_000.0
                if not math.isclose(native_gbps, derived_gbps, rel_tol=0.01, abs_tol=0.01):
                    raise ValidationError(
                        f"trace end event {index}.bandwidth_GBps contradicts bytes/timestamps"
                    )
            normalized: dict[str, Any] = {
                "name": begin["name"],
                "cat": begin["category"],
                "ph": "X",
                "ts": begin["ts"],
                "dur": duration,
            }
            if normalized_args:
                normalized["args"] = normalized_args
            normalized_events.append(normalized)
            normalized_category_counts[begin["category"]] = normalized_category_counts.get(begin["category"], 0) + 1
            stream_ids.add(_identity(raw_event.get("tid"), "tid"))
            boundary_count += 1
            complete_count += 1
            continue
        if phase == "X":
            name = _short_string(raw_event.get("name"), f"complete event {index} name")
            assert name is not None
            ts = _number(raw_event.get("ts", 0), f"complete event {index} timestamp")
            dur = _number(raw_event.get("dur"), f"complete event {index} duration")
            raw_category = _short_string(raw_event.get("cat"), f"complete event {index} category")
            assert raw_category is not None
            target_category = _category_target(raw_category, event_category_value, dma_category_value, compute_categories, dma_categories)
            args, evidence = _public_args(raw_event)
            record_args(evidence)
            normalized = {"name": name, "cat": target_category, "ph": "X", "ts": ts, "dur": dur}
            if args:
                normalized["args"] = args
            normalized_events.append(normalized)
            normalized_category_counts[target_category] = normalized_category_counts.get(target_category, 0) + 1
            passthrough_count += 1
            complete_count += 1
            continue
        if phase in {"s", "f"}:
            flow = _minimal_flow(raw_event)
            stream_ids.add(_identity(raw_event.get("tid"), "tid"))
            # Chrome flow arrows intentionally connect different stream tids
            # (the source stream emits ``s`` and the destination emits ``f``).
            # Pair on the flow id/process, while retaining each endpoint's tid
            # only in the pointer-free diagnostic event.
            flow_key = (
                _identity(raw_event.get("id"), "id"),
                _identity(raw_event.get("pid"), "pid"),
            )
            if phase == "s":
                open_flow_ids.setdefault(flow_key, []).append(
                    {
                        "name": flow.get("name"),
                        "cat": flow.get("cat"),
                        "ts": flow["ts"],
                    }
                )
            elif flow_key in open_flow_ids:
                starts = open_flow_ids[flow_key]
                start = starts.pop()
                if flow.get("name") is not None and start.get("name") is not None and flow["name"] != start["name"]:
                    raise ValidationError(f"trace flow end event {index} name mismatches begin")
                if flow.get("cat") != start.get("cat"):
                    raise ValidationError(f"trace flow end event {index} category mismatches begin")
                if flow["ts"] < start["ts"]:
                    raise ValidationError(f"trace flow end event {index} precedes begin")
                if not starts:
                    del open_flow_ids[flow_key]
            else:
                raise ValidationError(f"trace flow end event {index} is unpaired")
            normalized_events.append(flow)
            flow_count += 1
            continue
        if phase in IGNORABLE_PHASES:
            ignored_phase_counts[phase] = ignored_phase_counts.get(phase, 0) + 1
            continue
        raise ValidationError(f"unsupported trace phase {phase!r} at event {index}")

    unclosed = sum(len(stack) for stack in stacks.values())
    if unclosed:
        raise ValidationError(f"trace has {unclosed} unclosed begin event(s)")
    if open_flow_ids:
        raise ValidationError(
            f"trace has {sum(len(starts) for starts in open_flow_ids.values())} unclosed barrier flow(s)"
        )
    coverage_mismatches: list[str] = []
    # ``batch_events`` includes command-buffer packets, and the SDK appends a
    # final packet to the R1 stream.  They therefore need only satisfy the
    # documented lower bounds, not equal the number of B/E pairs.
    if "batch_events" in other_data and other_data["batch_events"] < begin_count:
        coverage_mismatches.append("batch_events")
    if "records_captured" in other_data and (
        other_data["records_captured"] < begin_count * 2
        or other_data["records_captured"] % 2 != 0
    ):
        coverage_mismatches.append("records_captured")
    coverage_consistent = not coverage_mismatches
    if assert_exhaustive and coverage_mismatches:
        raise ValidationError("trace otherData counts do not match paired device events")
    native_bandwidth_report: dict[str, Any] = {"sample_count": native_bandwidth_count}
    if native_bandwidth_count:
        native_bandwidth_report.update(
            {
                "mean_GBps": native_bandwidth_sum / native_bandwidth_count,
                "min_GBps": native_bandwidth_min,
                "max_GBps": native_bandwidth_max,
            }
        )
    normalized_root: dict[str, Any] = {
        "rhinoforgeTrace": {
            "schema": schema_value,
            "device_event_category": event_category_value,
            "device_program_events_exhaustive": bool(assert_exhaustive),
            "producer_sha256": producer_sha256,
        },
        "traceEvents": normalized_events,
    }
    report = {
        "schema_version": 1,
        "source_format": "rhino-launch-BE-v1" if any(item in phase_seen for item in ("B", "E", "b", "e")) else "rhinoforge-X-v1",
        "normalized_format": "rhinoforge-rpu-chrome-v1",
        "event_count": len(events),
        "complete_event_count": complete_count,
        "boundary_event_count": boundary_count,
        "duration_begin_count": begin_count,
        "flow_event_count": flow_count,
        "passthrough_complete_event_count": passthrough_count,
        "unclosed_event_count": 0,
        "barrier_flow_complete": not open_flow_ids,
        "stream_count": len(stream_ids),
        "weak_pair_identity_count": weak_pair_identity_count,
        "producer_sha256": producer_sha256,
        "event_category": event_category_value,
        "dma_event_category": dma_category_value,
        "schema": schema_value,
        "exhaustive_asserted": bool(assert_exhaustive),
        "coverage_consistent": coverage_consistent,
        "count_lower_bound_ok": coverage_consistent,
        "coverage_rule": (
            "batch_events>=duration_begin_count; records_captured>=2*duration_begin_count "
            "and even (SDK may append final packet/command entries)"
        ),
        "coverage_mismatches": coverage_mismatches,
        "other_data": other_data,
        "native_category_counts": native_category_counts,
        "normalized_category_counts": normalized_category_counts,
        "native_public_bytes_event_count": native_bytes_count,
        "native_bandwidth": native_bandwidth_report,
        "ignored_phase_counts": ignored_phase_counts,
    }
    return normalized_root, report


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _safe_destination(path: Path, forbidden: Iterable[Path | None] = ()) -> Path:
    # Inspect the lexical spelling before normalizing ``..``.  Otherwise a
    # pathname such as ``link/../out.json`` could hide a symlink component.
    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical)
    destination = _absolute_path(lexical)
    try:
        info = os.lstat(destination)
    except FileNotFoundError:
        info = None
    except OSError as error:
        raise ValidationError("output cannot be inspected") from error
    if info is not None and (stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode)):
        raise ValidationError("output must be a regular non-symlink file")
    parent = destination.parent
    try:
        parent_info = os.lstat(parent)
    except FileNotFoundError:
        parent.mkdir(parents=True, exist_ok=True)
        parent_info = os.lstat(parent)
    except OSError as error:
        raise ValidationError("output parent cannot be inspected") from error
    if stat.S_ISLNK(parent_info.st_mode) or not stat.S_ISDIR(parent_info.st_mode):
        raise ValidationError("output parent must be a regular directory")
    for candidate in forbidden:
        if candidate is None:
            continue
        candidate_lexical = _lexical_absolute(candidate)
        _reject_symlink_components(candidate_lexical)
        candidate_abs = _absolute_path(candidate_lexical)
        try:
            same = (
                destination.exists()
                and candidate_abs.exists()
                and os.path.samefile(destination, candidate_abs)
            )
        except OSError:
            same = False
        if same or destination == candidate_abs:
            raise ValidationError("output must differ from input files")
    return destination


def _atomic_write(
    path: Path, payload: bytes, *, forbidden: Iterable[Path | None] = ()
) -> None:
    destination = _safe_destination(path, forbidden)
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValidationError("normalized output exceeds size limit")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass


def convert_file(
    input_path: Path,
    output_path: Path,
    *,
    producer_sha256: str,
    schema: str = "rhinoforge-rpu-chrome-v1",
    event_category: str = "rpu_device_program",
    dma_event_category: str = "rpu_dma",
    assert_exhaustive: bool = False,
    require_frequency: bool = False,
    report_path: Path | None = None,
) -> dict[str, Any]:
    # Preserve the source's lexical path until symlink components have been
    # inspected.  Normalizing first would make ``symlink/../trace.json`` look
    # harmless even though the kernel resolves the symlink component first.
    input_lexical = _lexical_absolute(input_path)
    _reject_symlink_components(input_lexical)
    input_path = _absolute_path(input_lexical)
    # Keep destination spellings lexical until _safe_destination has inspected
    # every existing component.  In particular, do not collapse a destination
    # such as ``link/../normalized.json`` before the symlink check.
    output_path = _lexical_absolute(output_path)
    report_path = _lexical_absolute(report_path) if report_path is not None else None
    # Check the source before creating destination directories, and validate
    # all destinations before parsing or writing.
    try:
        input_info = os.lstat(input_path)
    except OSError as error:
        raise ValidationError("native trace is unavailable") from error
    if stat.S_ISLNK(input_info.st_mode) or not stat.S_ISREG(input_info.st_mode):
        raise ValidationError("native trace must be a regular non-symlink file")
    # Validate all destinations before parsing or writing.  In particular,
    # reject a final/parent symlink and hard-link aliases to the raw trace.
    safe_output = _safe_destination(output_path, (input_path, report_path))
    safe_report = None
    if report_path is not None:
        safe_report = _safe_destination(report_path, (input_path, safe_output))
    raw = _read_native_bytes(input_path)
    try:
        parsed = strict_json_loads(raw.decode("utf-8"))
    except UnicodeDecodeError as error:
        raise ValidationError("native trace is not UTF-8") from error
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValidationError("native trace JSON is invalid") from error
    normalized, report = normalize_trace(
        parsed,
        producer_sha256=producer_sha256,
        schema=schema,
        event_category=event_category,
        dma_event_category=dma_event_category,
        assert_exhaustive=assert_exhaustive,
        require_frequency=require_frequency,
    )
    normalized_bytes = (canonical_json(normalized) + "\n").encode("utf-8")
    _atomic_write(safe_output, normalized_bytes, forbidden=(input_path, report_path))
    report = dict(report)
    report.update(
        {
            "input_sha256": hashlib.sha256(raw).hexdigest(),
            "normalized_sha256": hashlib.sha256(normalized_bytes).hexdigest(),
            "normalized_bytes": len(normalized_bytes),
        }
    )
    if report_path is not None:
        _atomic_write(
            safe_report,
            (canonical_json(report) + "\n").encode("utf-8"),
            forbidden=(input_path, safe_output),
        )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Normalize strict Rhino Launch B/E traces to ph=X without exposing private event fields."
    )
    parser.add_argument("input", type=Path, help="native Chrome trace JSON")
    parser.add_argument("output", type=Path, help="private normalized trace JSON")
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--trace-schema", default="rhinoforge-rpu-chrome-v1")
    parser.add_argument("--event-category", default="rpu_device_program")
    parser.add_argument(
        "--dma-event-category",
        default="rpu_dma",
        help="non-program category for native DMA events (default: rpu_dma)",
    )
    parser.add_argument(
        "--assert-exhaustive",
        action="store_true",
        help="explicitly assert that the trusted producer captured every device launch",
    )
    parser.add_argument(
        "--require-frequency",
        action="store_true",
        help="require positive otherData.frequency_hz metadata",
    )
    parser.add_argument("--report", type=Path, help="optional public normalization report")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = convert_file(
            args.input,
            args.output,
            producer_sha256=args.producer_sha256,
            schema=args.trace_schema,
            event_category=args.event_category,
            dma_event_category=args.dma_event_category,
            assert_exhaustive=args.assert_exhaustive,
            require_frequency=args.require_frequency,
            report_path=args.report,
        )
    except (
        OSError,
        ValidationError,
        ValueError,
        TypeError,
        OverflowError,
        RecursionError,
    ) as error:
        print(json.dumps({"schema_version": 1, "ok": False, "error": str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps({"schema_version": 1, "ok": True, **report}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
