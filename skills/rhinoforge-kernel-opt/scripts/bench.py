#!/usr/bin/env python3
"""Run a correctness-gated, paired kernel benchmark campaign.

The adapter is ordinary Python and may load native code.  This runner is not a
security sandbox for malicious candidates.  A final verifier must execute a
frozen candidate in an administrator-approved, read-only harness and a clean
process.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import random
import re
import stat
import statistics
import sys
import tempfile
import time
import uuid
from collections.abc import Mapping, Sequence
from types import ModuleType
from typing import Any, Callable

try:  # The runner can be imported and --help can be used without PyTorch.
    import torch
except ImportError:  # pragma: no cover - exercised only in minimal harnesses.
    torch = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError:  # pragma: no cover - RhinoForge declares NumPy.
    np = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
_COMMIT_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.:+/-]{0,127}$")
_KERNEL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,254}$")
_REQUIRED_HOOKS = ("make_case", "reference", "candidate", "synchronize")
_RELATIVE_FLOOR = 1.0e-12
_ZERO_SHA256 = "0" * 64
_MAX_TRUSTED_EXECUTION_BYTES = 1024 * 1024
_FUSION_PROFILES = frozenset(
    {
        "gemm+silu_mul",
        "gemm+add",
        "gemm+rope",
        "rmsnorm+quant_mxfp8",
    }
)
_PROFILES = frozenset({"gemm", *_FUSION_PROFILES})
_BARE_OPERATION_IDENTITIES = {
    "gemm+silu_mul": "two_gemm_gate_up",
    "gemm+add": "gemm",
    "gemm+rope": "two_gemm_qk",
    "rmsnorm+quant_mxfp8": "rmsnorm",
}


class BenchError(RuntimeError):
    """A fail-closed adapter or benchmark protocol error."""


def _read_process_identity() -> dict[str, Any]:
    """Bind this receipt to the current Linux process without claiming trust."""

    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="ascii"
        ).strip()
        uuid.UUID(boot_id)
        stat_text = Path("/proc/self/stat").read_text(encoding="ascii").strip()
        closing_parenthesis = stat_text.rfind(")")
        if closing_parenthesis < 0:
            raise ValueError("missing process-name terminator")
        # Fields after the process name begin at proc(5) field 3.  starttime is
        # field 22, hence index 19 in this suffix.
        suffix_fields = stat_text[closing_parenthesis + 1 :].split()
        start_ticks = int(suffix_fields[19])
        if start_ticks <= 0:
            raise ValueError("non-positive process start time")
    except (OSError, UnicodeError, ValueError, IndexError) as exc:
        raise BenchError("cannot establish Linux process identity") from exc
    return {
        "boot_id": boot_id.lower(),
        "pid": os.getpid(),
        "start_ticks": start_ticks,
        "run_uuid": str(uuid.uuid4()),
        # Local /proc facts distinguish runs but are not release attestation.
        # The cooperative diagnostic launcher may mark its direct observation;
        # release still requires the separately signed authority statement.
        "attested": False,
    }


def _normalize_execution_evidence(
    raw: Any, workload: str, preflight_sha256: str
) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise BenchError(f"trusted execution for {workload} must be a mapping")
    expected_keys = {
        "single_device_program",
        "launch_count",
        "kernel_names",
        "trace_summary_sha256",
        "preflight_sha256",
    }
    if set(raw) != expected_keys:
        raise BenchError(
            f"trusted execution for {workload} must contain exactly "
            "single_device_program, launch_count, kernel_names, "
            "trace_summary_sha256, and preflight_sha256"
        )
    single_program = raw["single_device_program"]
    launch_count = raw["launch_count"]
    kernel_names = raw["kernel_names"]
    trace_hash = raw["trace_summary_sha256"]
    evidence_preflight = raw["preflight_sha256"]
    if not isinstance(single_program, bool):
        raise BenchError("execution single_device_program must be bool")
    if (
        not isinstance(launch_count, int)
        or isinstance(launch_count, bool)
        or launch_count < 0
    ):
        raise BenchError("execution launch_count must be a non-negative integer")
    if not isinstance(kernel_names, list) or not kernel_names or any(
        not isinstance(name, str) or _KERNEL_NAME_RE.fullmatch(name) is None
        for name in kernel_names
    ):
        raise BenchError("execution kernel_names must be an array of public names")
    if len(kernel_names) != len(set(kernel_names)):
        raise BenchError("execution kernel_names must be unique")
    if not isinstance(trace_hash, str) or _SHA256_RE.fullmatch(trace_hash) is None:
        raise BenchError("execution trace_summary_sha256 must be SHA-256")
    if (
        not isinstance(evidence_preflight, str)
        or _SHA256_RE.fullmatch(evidence_preflight) is None
        or evidence_preflight.lower() != preflight_sha256
    ):
        raise BenchError("execution preflight_sha256 does not match the benchmark")
    return {
        "single_device_program": single_program,
        "launch_count": launch_count,
        "kernel_names": list(kernel_names),
        "trace_summary_sha256": trace_hash.lower(),
        "preflight_sha256": evidence_preflight.lower(),
    }


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise BenchError(f"duplicate JSON key in trusted execution: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise BenchError(f"non-finite JSON constant in trusted execution: {value}")


def _load_trusted_execution_file(
    path: Path,
    *,
    workloads: list[str],
    contract_sha256: str,
    preflight_sha256: str,
    candidate_commit: str,
    parent_commit: str,
) -> dict[str, dict[str, Any]]:
    """Read a verifier-owned evidence envelope without following symlinks."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise BenchError("trusted execution evidence is unavailable") from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise BenchError("trusted execution evidence must be a regular file")
        if metadata.st_size > _MAX_TRUSTED_EXECUTION_BYTES:
            raise BenchError("trusted execution evidence is too large")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            raw_bytes = handle.read(_MAX_TRUSTED_EXECUTION_BYTES + 1)
        if len(raw_bytes) > _MAX_TRUSTED_EXECUTION_BYTES:
            raise BenchError("trusted execution evidence is too large")
    finally:
        os.close(descriptor)
    try:
        envelope = json.loads(
            raw_bytes.decode("utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise BenchError("trusted execution evidence is invalid JSON") from exc
    if not isinstance(envelope, Mapping) or set(envelope) != {
        "schema_version",
        "kind",
        "contract_sha256",
        "preflight_sha256",
        "candidate_commit",
        "parent_commit",
        "workloads",
    }:
        raise BenchError("trusted execution evidence envelope has invalid fields")
    if (
        envelope["schema_version"] != 1
        or envelope["kind"] != "rhinoforge-local-trusted-execution"
    ):
        raise BenchError("trusted execution evidence envelope has invalid identity")
    expected_identity = {
        "contract_sha256": contract_sha256.lower(),
        "preflight_sha256": preflight_sha256.lower(),
        "candidate_commit": candidate_commit.lower(),
        "parent_commit": parent_commit.lower(),
    }
    for field, expected in expected_identity.items():
        value = envelope[field]
        if not isinstance(value, str) or value.lower() != expected:
            raise BenchError(f"trusted execution {field} does not match the benchmark")
    raw_workloads = envelope["workloads"]
    if not isinstance(raw_workloads, Mapping) or set(raw_workloads) != set(workloads):
        raise BenchError("trusted execution workloads do not match the benchmark")
    return {
        workload: _normalize_execution_evidence(
            raw_workloads[workload], workload, preflight_sha256.lower()
        )
        for workload in workloads
    }


def _is_torch_tensor(value: Any) -> bool:
    return torch is not None and isinstance(value, torch.Tensor)


def _is_numpy_tensor(value: Any) -> bool:
    return np is not None and isinstance(value, np.ndarray)


def _is_tensor(value: Any) -> bool:
    return _is_torch_tensor(value) or _is_numpy_tensor(value)


def _path_key(key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        return "." + key
    return "[" + repr(key) + "]"


def _tensor_leaves(value: Any, path: str = "$") -> list[tuple[str, Any]]:
    leaves: list[tuple[str, Any]] = []

    def visit(node: Any, node_path: str) -> None:
        if _is_tensor(node):
            leaves.append((node_path, node))
        elif isinstance(node, Mapping):
            for key in sorted(node.keys(), key=repr):
                visit(node[key], node_path + _path_key(key))
        elif isinstance(node, (tuple, list)):
            for index, item in enumerate(node):
                visit(item, f"{node_path}[{index}]")

    visit(value, path)
    return leaves


def _normalize_quantized_output_roles(
    adapter: ModuleType, output: Any
) -> dict[str, str]:
    """Bind every encoded output leaf to an explicit semantic role."""

    hook = getattr(adapter, "quantized_output_roles", None)
    if not callable(hook):
        raise BenchError(
            "rmsnorm+quant_mxfp8 requires quantized_output_roles()"
        )
    raw = hook()
    if not isinstance(raw, Mapping):
        raise BenchError("quantized_output_roles() must return a mapping")
    roles: dict[str, str] = {}
    for path, role in raw.items():
        if not isinstance(path, str) or not path.startswith("$") or len(path) > 512:
            raise BenchError("quantized output role paths must be bounded '$' paths")
        if not isinstance(role, str) or role not in {"payload", "scale", "metadata"}:
            raise BenchError(
                "quantized output roles must be payload, scale, or metadata"
            )
        if path in roles:
            raise BenchError("quantized output role paths must be unique")
        roles[path] = role
    output_paths = {path for path, _ in _tensor_leaves(output)}
    if set(roles) != output_paths:
        raise BenchError(
            "quantized output roles must cover every encoded tensor leaf exactly"
        )
    if "payload" not in roles.values() or "scale" not in roles.values():
        raise BenchError("quantized output roles require payload and scale leaves")
    return {path: roles[path] for path in sorted(roles)}


def _clone_tree(value: Any) -> Any:
    if _is_torch_tensor(value):
        return value.detach().clone()
    if _is_numpy_tensor(value):
        return value.copy()
    if isinstance(value, Mapping):
        items = [(key, _clone_tree(item)) for key, item in value.items()]
        try:
            return type(value)(items)
        except (TypeError, ValueError):
            return dict(items)
    if isinstance(value, tuple):
        values = [_clone_tree(item) for item in value]
        if hasattr(value, "_fields"):
            return type(value)(*values)
        return tuple(values)
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return copy.deepcopy(value)


def _tensor_shape(value: Any) -> list[int]:
    return [int(dim) for dim in value.shape]


def _tensor_dtype(value: Any) -> str:
    return str(value.dtype)


def _tensor_device(value: Any) -> str:
    if _is_torch_tensor(value):
        return str(value.device)
    return "cpu"


def _tensor_strides(value: Any) -> list[int]:
    if _is_torch_tensor(value):
        return [int(stride) for stride in value.stride()]
    return [int(stride) for stride in value.strides]


def _tensor_kind(value: Any) -> str:
    if _is_torch_tensor(value):
        if value.dtype == torch.bool or not (
            value.is_floating_point() or value.is_complex()
        ):
            return "integral"
        return "floating"
    if np.issubdtype(value.dtype, np.integer) or np.issubdtype(
        value.dtype, np.bool_
    ):
        return "integral"
    if np.issubdtype(value.dtype, np.floating) or np.issubdtype(
        value.dtype, np.complexfloating
    ):
        return "floating"
    return "other"


def _tensor_numel(value: Any) -> int:
    if _is_torch_tensor(value):
        return int(value.numel())
    return int(value.size)


def _tensor_bytes(value: Any) -> bytes:
    """Materialize exact tensor bytes on the host after adapter synchronization."""
    if _is_torch_tensor(value):
        host = value.detach().to(device="cpu").contiguous().reshape(-1)
        if host.numel() == 0:
            return b""
        return host.view(torch.uint8).numpy().tobytes()
    if _is_numpy_tensor(value):
        return np.ascontiguousarray(value).tobytes()
    raise TypeError(f"not a supported tensor: {type(value).__name__}")


def _storage_descriptor(value: Any) -> tuple[Any, ...]:
    """Return allocation, view address, offset, and metadata identity."""
    if _is_torch_tensor(value):
        try:
            storage = value.untyped_storage()
            base_pointer, allocation_bytes = int(storage.data_ptr()), int(
                storage.nbytes()
            )
        except (AttributeError, RuntimeError):
            base_pointer = int(value.data_ptr())
            allocation_bytes = int(value.numel() * value.element_size())
        return (
            base_pointer,
            allocation_bytes,
            int(value.data_ptr()),
            int(value.storage_offset()),
            tuple(int(dim) for dim in value.shape),
            tuple(int(stride) for stride in value.stride()),
            str(value.dtype),
        )
    if _is_numpy_tensor(value):
        pointer = int(value.__array_interface__["data"][0])
        return (
            pointer,
            int(value.nbytes),
            pointer,
            0,
            tuple(int(dim) for dim in value.shape),
            tuple(int(stride) for stride in value.strides),
            str(value.dtype),
        )
    raise TypeError(f"not a supported tensor: {type(value).__name__}")


def _shares_storage(left: Any, right: Any) -> bool:
    if _is_torch_tensor(left) and _is_torch_tensor(right):
        left_descriptor = _storage_descriptor(left)
        right_descriptor = _storage_descriptor(right)
        lptr, lsize = int(left_descriptor[0]), int(left_descriptor[1])
        rptr, rsize = int(right_descriptor[0]), int(right_descriptor[1])
        if not lsize or not rsize:
            return False
        return lptr < rptr + rsize and rptr < lptr + lsize
    if _is_numpy_tensor(left) and _is_numpy_tensor(right):
        return bool(np.shares_memory(left, right))
    return False


def _as_numeric_array(value: Any) -> Any:
    if np is None:
        raise BenchError("NumPy is required to compute parity metrics")
    if _is_torch_tensor(value):
        host = value.detach().to(device="cpu")
        target_dtype = torch.complex128 if host.is_complex() else torch.float64
        return host.to(dtype=target_dtype).numpy()
    if _is_numpy_tensor(value):
        dtype = np.complex128 if np.iscomplexobj(value) else np.float64
        return np.asarray(value, dtype=dtype)
    raise TypeError(f"not a supported tensor: {type(value).__name__}")


def _nonfinite_count(value: Any) -> int:
    array = _as_numeric_array(value)
    return int(array.size - np.count_nonzero(np.isfinite(array)))


def _finite_metric(value: float) -> float:
    if math.isnan(value):
        return sys.float_info.max
    if value == math.inf:
        return sys.float_info.max
    if value == -math.inf:
        return -sys.float_info.max
    return float(value)


def _compare_tensor(
    candidate: Any,
    reference: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
    require_dtype: bool,
    require_device: bool,
    require_bit_exact: bool,
    require_layout: bool,
) -> dict[str, Any]:
    shape_match = _tensor_shape(candidate) == _tensor_shape(reference)
    dtype_match = _tensor_dtype(candidate) == _tensor_dtype(reference)
    device_match = _tensor_device(candidate) == _tensor_device(reference)
    layout_match = _tensor_strides(candidate) == _tensor_strides(reference)
    byte_exact = bool(
        shape_match
        and dtype_match
        and _tensor_bytes(candidate) == _tensor_bytes(reference)
    )
    candidate_nonfinite = _nonfinite_count(candidate)
    reference_nonfinite = _nonfinite_count(reference)
    nonfinite_count = candidate_nonfinite + reference_nonfinite

    result: dict[str, Any] = {
        "path": path,
        "candidate_shape": _tensor_shape(candidate),
        "reference_shape": _tensor_shape(reference),
        "candidate_dtype": _tensor_dtype(candidate),
        "reference_dtype": _tensor_dtype(reference),
        "candidate_device": _tensor_device(candidate),
        "reference_device": _tensor_device(reference),
        "candidate_strides": _tensor_strides(candidate),
        "reference_strides": _tensor_strides(reference),
        "tensor_kind": _tensor_kind(candidate),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "device_match": device_match,
        "layout_match": layout_match,
        "byte_exact": byte_exact,
        "nonfinite_count": nonfinite_count,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "anchor_max_abs": 0.0,
        "anchor_max_rel": 0.0,
        "passed": False,
    }
    if not shape_match:
        return result

    cand = _as_numeric_array(candidate)
    ref = _as_numeric_array(reference)
    if cand.size:
        absolute = np.abs(cand - ref)
        denominator = np.maximum(np.abs(ref), _RELATIVE_FLOOR)
        relative = absolute / denominator
        max_abs = _finite_metric(float(np.max(absolute)))
        max_rel = _finite_metric(float(np.max(relative)))
        # Campaign contracts freeze max-absolute and max-relative gates as
        # independent evidence.  Do not let a large relative miss near zero be
        # waived by an allclose-style combined tolerance (or vice versa).
        close = bool(max_abs <= atol and max_rel <= rtol)
    else:
        max_abs = 0.0
        max_rel = 0.0
        close = True
    result["max_abs"] = max_abs
    result["max_rel"] = max_rel
    result["passed"] = bool(
        shape_match
        and (dtype_match or not require_dtype)
        and (device_match or not require_device)
        and (byte_exact or not require_bit_exact)
        and (layout_match or not require_layout)
        and nonfinite_count == 0
        and close
    )
    return result


def _compare_trees(
    candidate: Any,
    reference: Any,
    *,
    atol: float,
    rtol: float,
    require_dtype: bool,
    require_device: bool = True,
    require_bit_exact: bool = False,
    require_layout: bool = False,
) -> dict[str, Any]:
    leaves: list[dict[str, Any]] = []
    errors: list[str] = []

    def compare(cand: Any, ref: Any, path: str) -> None:
        if _is_tensor(cand) or _is_tensor(ref):
            if not (_is_tensor(cand) and _is_tensor(ref)):
                errors.append(
                    f"{path}: tensor/non-tensor mismatch "
                    f"({type(cand).__name__} vs {type(ref).__name__})"
                )
                return
            leaves.append(
                _compare_tensor(
                    cand,
                    ref,
                    path=path,
                    atol=atol,
                    rtol=rtol,
                    require_dtype=require_dtype,
                    require_device=require_device,
                    require_bit_exact=require_bit_exact,
                    require_layout=require_layout,
                )
            )
            return
        if isinstance(cand, Mapping) or isinstance(ref, Mapping):
            if not (isinstance(cand, Mapping) and isinstance(ref, Mapping)):
                errors.append(f"{path}: mapping/container mismatch")
                return
            cand_keys = set(cand.keys())
            ref_keys = set(ref.keys())
            if cand_keys != ref_keys:
                errors.append(
                    f"{path}: mapping keys differ: "
                    f"{sorted(cand_keys, key=repr)!r} vs "
                    f"{sorted(ref_keys, key=repr)!r}"
                )
                return
            for key in sorted(cand_keys, key=repr):
                compare(cand[key], ref[key], path + _path_key(key))
            return
        cand_sequence = isinstance(cand, (tuple, list))
        ref_sequence = isinstance(ref, (tuple, list))
        if cand_sequence or ref_sequence:
            if type(cand) is not type(ref):
                errors.append(
                    f"{path}: sequence type differs: "
                    f"{type(cand).__name__} vs {type(ref).__name__}"
                )
                return
            if len(cand) != len(ref):
                errors.append(f"{path}: sequence length differs")
                return
            for index, (cand_item, ref_item) in enumerate(zip(cand, ref)):
                compare(cand_item, ref_item, f"{path}[{index}]")
            return
        errors.append(
            f"{path}: output leaves must be tensors, got "
            f"{type(cand).__name__}/{type(ref).__name__}"
        )

    compare(candidate, reference, "$")
    nonfinite_count = sum(int(leaf["nonfinite_count"]) for leaf in leaves)
    max_abs = max((float(leaf["max_abs"]) for leaf in leaves), default=0.0)
    max_rel = max((float(leaf["max_rel"]) for leaf in leaves), default=0.0)
    dtype_match = bool(leaves) and all(bool(leaf["dtype_match"]) for leaf in leaves)
    device_match = bool(leaves) and all(bool(leaf["device_match"]) for leaf in leaves)
    shape_match = bool(leaves) and all(bool(leaf["shape_match"]) for leaf in leaves)
    byte_exact = bool(leaves) and all(bool(leaf["byte_exact"]) for leaf in leaves)
    layout_match = bool(leaves) and all(
        bool(leaf["layout_match"]) for leaf in leaves
    )
    passed = bool(leaves) and not errors and all(bool(leaf["passed"]) for leaf in leaves)
    return {
        "passed": passed,
        "leaf_count": len(leaves),
        "shape_match": shape_match,
        "dtype_match": dtype_match,
        "device_match": device_match,
        "byte_exact": byte_exact,
        "layout_match": layout_match,
        "nonfinite_count": nonfinite_count,
        "max_abs": _finite_metric(max_abs),
        "max_rel": _finite_metric(max_rel),
        "leaves": leaves,
        "errors": errors,
    }


def _case_fingerprint(case: Any, *, include_storage: bool = True) -> str:
    digest = hashlib.sha256()

    def update(node: Any, path: str) -> None:
        digest.update(path.encode("utf-8", errors="backslashreplace"))
        if _is_tensor(node):
            digest.update(b"tensor\0")
            digest.update(str(_tensor_shape(node)).encode("ascii"))
            digest.update(_tensor_dtype(node).encode("ascii", errors="replace"))
            digest.update(_tensor_device(node).encode("ascii", errors="replace"))
            digest.update(repr(_tensor_strides(node)).encode("ascii"))
            if include_storage:
                digest.update(repr(_storage_descriptor(node)).encode("ascii"))
            digest.update(_tensor_bytes(node))
        elif isinstance(node, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(node.keys(), key=repr):
                update(node[key], path + _path_key(key))
        elif isinstance(node, (tuple, list)):
            digest.update(type(node).__name__.encode("ascii") + b"\0")
            for index, item in enumerate(node):
                update(item, f"{path}[{index}]")
        else:
            digest.update(type(node).__qualname__.encode("utf-8"))
            digest.update(repr(node).encode("utf-8", errors="backslashreplace"))

    update(case, "$")
    return digest.hexdigest()


def _call_mutate(hook: Callable[..., Any], case: Any, seed: int) -> Any:
    try:
        signature = inspect.signature(hook)
    except (TypeError, ValueError):
        signature = None
    if signature is not None:
        try:
            signature.bind(case, seed)
        except TypeError:
            try:
                signature.bind(case)
            except TypeError as exc:
                raise BenchError(
                    "mutate_case must accept (case, seed) or (case)"
                ) from exc
            result = hook(case)
        else:
            result = hook(case, seed)
    else:  # pragma: no cover - only opaque extension callables land here.
        result = hook(case, seed)
    return case if result is None else result


def _generic_mutate_in_place(case: Any) -> Any:
    for _, tensor in _tensor_leaves(case):
        if _tensor_numel(tensor) == 0:
            continue
        before = _tensor_bytes(tensor)
        if _is_torch_tensor(tensor):
            with torch.no_grad():
                if tensor.dtype == torch.bool:
                    tensor.logical_not_()
                elif tensor.is_complex():
                    tensor.add_(1.0 + 0.5j)
                elif tensor.is_floating_point():
                    tensor.add_(1.0)
                else:
                    tensor.add_(1)
        else:
            if np.issubdtype(tensor.dtype, np.bool_):
                np.logical_not(tensor, out=tensor)
            elif np.issubdtype(tensor.dtype, np.complexfloating):
                tensor[...] = tensor + (1.0 + 0.5j)
            else:
                tensor[...] = tensor + 1
        if _tensor_bytes(tensor) != before:
            return case
    raise BenchError(
        "cannot construct the same-pointer/changed-value probe; provide "
        "mutate_case(case, seed)"
    )


def _input_probe_details(before_case: Any, after_case: Any) -> dict[str, Any]:
    before = dict(_tensor_leaves(before_case))
    after = dict(_tensor_leaves(after_case))
    paths_match = before.keys() == after.keys() and bool(before)
    same_storage = paths_match and all(
        _storage_descriptor(before[path]) == _storage_descriptor(after[path])
        for path in before
        if _tensor_numel(before[path])
    )
    values_equal = paths_match and all(
        _tensor_bytes(before[path]) == _tensor_bytes(after[path]) for path in before
    )
    return {
        "paths_match": bool(paths_match),
        "same_storage": bool(same_storage),
        "values_equal": bool(values_equal),
    }


def _retained_output_probe(
    first_output: Any,
    before_descriptors: dict[str, tuple[Any, ...]],
    before_bytes: dict[str, bytes],
    second_output: Any,
) -> dict[str, Any]:
    first = dict(_tensor_leaves(first_output))
    second = dict(_tensor_leaves(second_output))
    first_paths_stable = first.keys() == before_descriptors.keys() and bool(first)
    storage_stable = first_paths_stable and all(
        _storage_descriptor(first[path]) == before_descriptors[path] for path in first
    )
    bytes_stable = first_paths_stable and all(
        _tensor_bytes(first[path]) == before_bytes[path] for path in first
    )
    independent = bool(second) and not any(
        _shares_storage(first_tensor, second_tensor)
        for first_tensor in first.values()
        for second_tensor in second.values()
    )
    return {
        "passed": bool(storage_stable and bytes_stable and independent),
        "storage_stable": bool(storage_stable),
        "bytes_stable": bool(bytes_stable),
        "independent_outputs": bool(independent),
    }


def _sync(adapter: ModuleType) -> None:
    adapter.synchronize()


def _invoke(adapter: ModuleType, hook_name: str, case: Any) -> Any:
    output = getattr(adapter, hook_name)(case)
    _sync(adapter)
    return output


def _evaluate_case(
    adapter: ModuleType,
    case: Any,
    *,
    atol: float,
    rtol: float,
    anchor_atol: float,
    anchor_rtol: float,
    exact_encoded_output: bool,
) -> tuple[Any, dict[str, Any]]:
    input_before = _case_fingerprint(case)
    reference = _invoke(adapter, "reference", case)
    input_after_reference = _case_fingerprint(case)
    candidate = _invoke(adapter, "candidate", case)
    input_after_candidate = _case_fingerprint(case)
    same_dtype = _compare_trees(
        candidate,
        reference,
        atol=atol,
        rtol=rtol,
        require_dtype=True,
        require_bit_exact=exact_encoded_output,
        require_layout=True,
    )
    anchor_metrics: dict[str, Any] | None = None
    input_after_anchor = input_after_candidate
    if callable(getattr(adapter, "fp32_anchor", None)):
        anchor = _invoke(adapter, "fp32_anchor", case)
        input_after_anchor = _case_fingerprint(case)
        if exact_encoded_output:
            dequantize = getattr(adapter, "dequantize", None)
            if not callable(dequantize):
                raise BenchError(
                    "rmsnorm+quant_mxfp8 requires dequantize(output, case)"
                )
            encoded_before = _case_fingerprint(candidate)
            dequantized = dequantize(candidate, case)
            _sync(adapter)
            if _case_fingerprint(candidate) != encoded_before:
                raise BenchError("dequantize() mutated the encoded candidate output")
            input_after_anchor = _case_fingerprint(case)
            anchor_metrics = _compare_trees(
                dequantized,
                anchor,
                atol=anchor_atol,
                rtol=anchor_rtol,
                require_dtype=False,
                require_device=False,
            )
            anchor_metrics["kind"] = "dequantized_vs_fp32"
        else:
            anchor_metrics = _compare_trees(
                candidate,
                anchor,
                atol=anchor_atol,
                rtol=anchor_rtol,
                require_dtype=False,
                require_device=False,
            )
            anchor_metrics["kind"] = "candidate_vs_fp32"
    passed = bool(same_dtype["passed"])
    if anchor_metrics is not None:
        passed = passed and bool(anchor_metrics["passed"])
    return candidate, {
        "passed": passed,
        "same_dtype": same_dtype,
        "fp32_anchor": anchor_metrics,
        "input_immutable": bool(
            input_before
            == input_after_reference
            == input_after_candidate
            == input_after_anchor
        ),
        "reference_input_immutable": input_before == input_after_reference,
        "candidate_input_immutable": input_after_reference == input_after_candidate,
        "anchor_input_immutable": input_after_candidate == input_after_anchor,
    }


def _normalize_graph_state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise BenchError("graph_state() must return a mapping")

    def pick(*names: str) -> Any:
        for name in names:
            if name in raw:
                return raw[name]
        raise BenchError(f"graph_state() is missing {names[0]!r}")

    build = pick("build_count", "builds")
    replay = pick("replay_count", "replays", "replay_total")
    cache_size = pick("cache_size", "size", "entries")
    invariant = pick("invariant_ok", "cache_invariant_ok", "invariant")
    if isinstance(cache_size, Sequence) and not isinstance(cache_size, (str, bytes)):
        cache_size = len(cache_size)
    if isinstance(cache_size, Mapping):
        cache_size = len(cache_size)
    if not isinstance(build, int) or isinstance(build, bool) or build < 0:
        raise BenchError("graph_state build_count must be a non-negative integer")
    if not isinstance(replay, int) or isinstance(replay, bool) or replay < 0:
        raise BenchError("graph_state replay_count must be a non-negative integer")
    if not isinstance(cache_size, int) or isinstance(cache_size, bool) or cache_size < 0:
        raise BenchError("graph_state cache_size must be a non-negative integer")
    if not isinstance(invariant, bool):
        raise BenchError("graph_state invariant_ok must be bool")
    return {
        "build_count": build,
        "replay_count": replay,
        "cache_size": cache_size,
        "invariant_ok": invariant,
    }


def _read_graph_state(adapter: ModuleType) -> dict[str, Any] | None:
    hook = getattr(adapter, "graph_state", None)
    if not callable(hook):
        return None
    return _normalize_graph_state(hook())


def _read_arm_graph_state(adapter: ModuleType, arm: str) -> dict[str, Any]:
    hook = getattr(adapter, "arm_graph_state", None)
    if not callable(hook):
        raise BenchError("fusion profile requires arm_graph_state(arm)")
    return _normalize_graph_state(hook(arm))


def _steady_replay_transition(
    stable: dict[str, Any], final: dict[str, Any]
) -> bool:
    return bool(
        stable["build_count"] == 1
        and final["build_count"] == 1
        and stable["replay_count"] >= 1
        and final["replay_count"] > stable["replay_count"]
        and final["cache_size"] == stable["cache_size"]
        and stable["invariant_ok"]
        and final["invariant_ok"]
    )


def _steady_replay_ready(state: dict[str, Any]) -> bool:
    """Return whether timing may start from an already-warm Graph state."""

    return bool(
        state["build_count"] == 1
        and state["replay_count"] >= 1
        and state["invariant_ok"]
    )


def _measure(adapter: ModuleType, hook_name: str, case: Any) -> int:
    # Both synchronizations are part of the protocol; input generation is not.
    _sync(adapter)
    started = time.perf_counter_ns()
    # Retain the returned Python object through synchronization and timestamp
    # capture so output storage cannot be released before the timed boundary.
    output = getattr(adapter, hook_name)(case)
    _sync(adapter)
    elapsed = time.perf_counter_ns() - started
    del output
    if elapsed <= 0:
        raise BenchError(f"non-positive {hook_name} latency sample")
    return int(elapsed)


def _shuffle_copy(values: list[str], rng: random.Random) -> list[str]:
    result = list(values)
    rng.shuffle(result)
    return result


def _timing_block(rng: random.Random, include_bare: bool) -> tuple[list[str], str]:
    candidate_outside_reference = bool(rng.getrandbits(1))
    reference_pattern = "ABBA" if candidate_outside_reference else "BAAB"
    if not include_bare:
        if candidate_outside_reference:
            return ["candidate", "reference", "reference", "candidate"], reference_pattern
        return ["reference", "candidate", "candidate", "reference"], reference_pattern

    candidate_outside_bare = bool(rng.getrandbits(1))
    bare_pattern = "ABBA" if candidate_outside_bare else "BAAB"
    if candidate_outside_reference and candidate_outside_bare:
        middle = _shuffle_copy(
            ["reference", "reference", "bare_operation", "bare_operation"], rng
        )
        order = ["candidate", *middle, "candidate"]
    elif candidate_outside_reference and not candidate_outside_bare:
        order = [
            "bare_operation",
            "candidate",
            "reference",
            "reference",
            "candidate",
            "bare_operation",
        ]
    elif not candidate_outside_reference and candidate_outside_bare:
        order = [
            "reference",
            "candidate",
            "bare_operation",
            "bare_operation",
            "candidate",
            "reference",
        ]
    else:
        before = _shuffle_copy(["reference", "bare_operation"], rng)
        after = _shuffle_copy(["reference", "bare_operation"], rng)
        order = [*before, "candidate", "candidate", *after]
    return order, reference_pattern + "/" + bare_pattern


def _percentile(samples: list[int] | list[float], percentile: float) -> float:
    if not samples:
        raise BenchError("cannot summarize an empty latency sample set")
    ordered = sorted(float(sample) for sample in samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _paired_epilogue_tax_interval(
    candidate_ns: Sequence[int | float],
    bare_ns: Sequence[int | float],
    *,
    confidence: float,
    bootstrap_trials: int,
    bootstrap_seed: int,
) -> dict[str, float | int]:
    """Bootstrap the median of paired percentage taxes deterministically."""

    if len(candidate_ns) != len(bare_ns) or len(candidate_ns) < 2:
        raise BenchError("epilogue tax requires equal paired sample sets")
    if not 0.5 < confidence < 1.0:
        raise BenchError("fusion_confidence must be between 0.5 and 1")
    if bootstrap_trials < 1000:
        raise BenchError("fusion_bootstrap_trials must be at least 1000")
    paired_pct = [
        100.0 * (float(candidate) - float(bare)) / float(bare)
        for candidate, bare in zip(candidate_ns, bare_ns)
    ]
    if any(not math.isfinite(value) for value in paired_pct):
        raise BenchError("epilogue tax samples must be finite")
    rng = random.Random(bootstrap_seed)
    count = len(paired_pct)
    bootstrap_medians = [
        float(statistics.median(paired_pct[rng.randrange(count)] for _ in range(count)))
        for _ in range(bootstrap_trials)
    ]
    alpha = (1.0 - confidence) / 2.0
    return {
        "epilogue_tax_p50_pct": float(statistics.median(paired_pct)),
        "epilogue_tax_ci_lower_pct": float(
            _percentile(bootstrap_medians, alpha)
        ),
        "epilogue_tax_ci_upper_pct": float(
            _percentile(bootstrap_medians, 1.0 - alpha)
        ),
        "epilogue_tax_confidence": confidence,
        "epilogue_tax_bootstrap_trials": bootstrap_trials,
        "epilogue_tax_bootstrap_seed": bootstrap_seed,
    }


def _sample_stats(samples: list[int]) -> dict[str, Any]:
    if not samples or any(not isinstance(value, int) or value <= 0 for value in samples):
        raise BenchError("latency samples must be non-empty positive integer nanoseconds")
    p50 = float(statistics.median(samples))
    deviations = [abs(float(value) - p50) for value in samples]
    mean = float(statistics.fmean(samples))
    cv = float(statistics.pstdev(samples) / mean) if len(samples) > 1 else 0.0
    return {
        "count": len(samples),
        "p50_ns": p50,
        "p95_ns": float(_percentile(samples, 0.95)),
        "mad_ns": float(statistics.median(deviations)),
        "cv": _finite_metric(cv),
    }


def _exception_text(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    if len(text) > 500:
        text = text[:497] + "..."
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _empty_workload(
    workload: str, error: str, *, execution: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "id": workload,
        "raw_samples": {"candidate_ns": [], "reference_ns": []},
        "summary": {
            "candidate_p50_ns": None,
            "reference_p50_ns": None,
        },
        "correctness": {
            "passed": False,
            "nonfinite_count": 0,
            "max_abs": 0.0,
            "max_rel": 0.0,
            "anchor_max_abs": 0.0,
            "anchor_max_rel": 0.0,
            "same_dtype": False,
            "fp32_anchor": None,
            "fp32_anchor_present": False,
            "input_immutable": False,
            "error": error,
        },
        "lifecycle": {
            "passed": False,
            "graph_build_count": 0,
            "graph_replay_count": 0,
            "cache_size_stable": False,
            "cache_invariant_ok": False,
            "independent_outputs": False,
        },
        "execution": execution,
        "hard_gates": {"candidate_execution": False},
        "eligible": False,
    }


def _aggregate_correctness(
    cases: dict[str, dict[str, Any]],
    retained: dict[str, Any],
    same_pointer: dict[str, Any],
    different_pointer: dict[str, Any],
    second_input_changed: bool,
) -> dict[str, Any]:
    same_results = [entry["same_dtype"] for entry in cases.values()]
    anchor_results = [
        entry["fp32_anchor"]
        for entry in cases.values()
        if entry["fp32_anchor"] is not None
    ]
    same_passed = bool(same_results) and all(bool(item["passed"]) for item in same_results)
    same_dtype = bool(same_results) and all(
        bool(item["dtype_match"]) for item in same_results
    )
    input_immutable = bool(cases) and all(
        bool(item["input_immutable"]) for item in cases.values()
    )
    anchor_present = bool(anchor_results)
    anchor_passed = all(bool(item["passed"]) for item in anchor_results)
    # The core error values are same-dtype metrics.  FP32-anchor distributions
    # stay in the per-probe evidence and have their own boolean gate.
    nonfinite_count = sum(int(item["nonfinite_count"]) for item in same_results)
    max_abs = max(
        (float(item["max_abs"]) for item in same_results), default=0.0
    )
    max_rel = max(
        (float(item["max_rel"]) for item in same_results), default=0.0
    )
    anchor_max_abs = max(
        (float(item["max_abs"]) for item in anchor_results), default=0.0
    )
    anchor_max_rel = max(
        (float(item["max_rel"]) for item in anchor_results), default=0.0
    )
    # Storage ownership and pointer/value construction are separate hard gates;
    # correctness.passed describes numerical/tree evidence only.
    passed = bool(same_passed and anchor_passed and input_immutable)
    return {
        "passed": passed,
        "nonfinite_count": nonfinite_count,
        "max_abs": _finite_metric(max_abs),
        "max_rel": _finite_metric(max_rel),
        "anchor_max_abs": _finite_metric(anchor_max_abs),
        "anchor_max_rel": _finite_metric(anchor_max_rel),
        "same_dtype": same_dtype,
        "fp32_anchor": anchor_passed,
        "fp32_anchor_present": anchor_present,
        "input_immutable": input_immutable,
        "probes": cases,
        "retained_output": retained,
        "same_pointer_changed_value": same_pointer,
        "different_pointer_same_value": different_pointer,
        "second_input_changed": bool(second_input_changed),
    }


def _run_workload(
    adapter: ModuleType,
    workload: str,
    *,
    verdict: str,
    profile: str | None,
    preflight_sha256: str,
    device: str,
    seed: int,
    warmup: int,
    repeats: int,
    atol: float,
    rtol: float,
    anchor_atol: float,
    anchor_rtol: float,
    max_cv: float | None,
    max_epilogue_tax_pct: float | None,
    fusion_confidence: float | None,
    fusion_bootstrap_trials: int | None,
    fusion_bootstrap_seed: int | None,
    trusted_execution: dict[str, Any] | None,
) -> dict[str, Any]:
    full_verdict = verdict == "full"
    fusion_profile = profile in _FUSION_PROFILES
    full_fusion = verdict == "full" and fusion_profile
    exact_encoded_output = profile == "rmsnorm+quant_mxfp8"
    execution = copy.deepcopy(trusted_execution)
    try:
        base_case = adapter.make_case(workload, seed, device)
        base_fingerprint = _case_fingerprint(base_case, include_storage=False)
        first_output, baseline = _evaluate_case(
            adapter,
            base_case,
            atol=atol,
            rtol=rtol,
            anchor_atol=anchor_atol,
            anchor_rtol=anchor_rtol,
            exact_encoded_output=exact_encoded_output,
        )
        first_leaves = dict(_tensor_leaves(first_output))
        if not first_leaves:
            raise BenchError("candidate output tree contains no tensor leaves")
        quantized_output_roles = (
            _normalize_quantized_output_roles(adapter, first_output)
            if exact_encoded_output
            else None
        )
        first_descriptors = {
            path: _storage_descriptor(tensor) for path, tensor in first_leaves.items()
        }
        first_bytes = {path: _tensor_bytes(tensor) for path, tensor in first_leaves.items()}

        second_case = adapter.make_case(workload, seed + 1, device)
        if _case_fingerprint(second_case, include_storage=False) == base_fingerprint:
            mutation_hook = getattr(adapter, "mutate_case", None)
            if callable(mutation_hook):
                second_case = _call_mutate(mutation_hook, second_case, seed + 1)
            else:
                second_case = _generic_mutate_in_place(second_case)
        second_input_changed = (
            _case_fingerprint(second_case, include_storage=False) != base_fingerprint
        )
        second_output, second_input = _evaluate_case(
            adapter,
            second_case,
            atol=atol,
            rtol=rtol,
            anchor_atol=anchor_atol,
            anchor_rtol=anchor_rtol,
            exact_encoded_output=exact_encoded_output,
        )
        retained = _retained_output_probe(
            first_output, first_descriptors, first_bytes, second_output
        )

        same_pointer_case = _clone_tree(base_case)
        # Execute once at this exact address before changing its bytes.  Merely
        # proving that an in-place mutation preserved the pointer would not
        # prove that a retained Graph/DMA path observes the new semantic value.
        _, same_pointer_before_result = _evaluate_case(
            adapter,
            same_pointer_case,
            atol=atol,
            rtol=rtol,
            anchor_atol=anchor_atol,
            anchor_rtol=anchor_rtol,
            exact_encoded_output=exact_encoded_output,
        )
        before_mutation = {
            path: {
                "descriptor": _storage_descriptor(tensor),
                "bytes": _tensor_bytes(tensor),
            }
            for path, tensor in _tensor_leaves(same_pointer_case)
        }
        mutation_hook = getattr(adapter, "mutate_case", None)
        if callable(mutation_hook):
            mutated_case = _call_mutate(mutation_hook, same_pointer_case, seed + 2)
        else:
            mutated_case = _generic_mutate_in_place(same_pointer_case)
        after_mutation = dict(_tensor_leaves(mutated_case))
        mutation_paths_match = before_mutation.keys() == after_mutation.keys() and bool(
            before_mutation
        )
        pointers_unchanged = mutation_paths_match and all(
            before_mutation[path]["descriptor"]
            == _storage_descriptor(after_mutation[path])
            for path in before_mutation
        )
        value_changed = mutation_paths_match and any(
            before_mutation[path]["bytes"] != _tensor_bytes(after_mutation[path])
            for path in before_mutation
        )
        _, same_pointer_result = _evaluate_case(
            adapter,
            mutated_case,
            atol=atol,
            rtol=rtol,
            anchor_atol=anchor_atol,
            anchor_rtol=anchor_rtol,
            exact_encoded_output=exact_encoded_output,
        )
        same_pointer = {
            "passed": bool(
                pointers_unchanged
                and value_changed
                and same_pointer_before_result["passed"]
                and same_pointer_result["passed"]
            ),
            "pointers_unchanged": bool(pointers_unchanged),
            "value_changed": bool(value_changed),
            "before_correctness": same_pointer_before_result,
            "correctness": same_pointer_result,
        }

        different_pointer_case = _clone_tree(mutated_case)
        mutated_leaves = dict(_tensor_leaves(mutated_case))
        different_leaves = dict(_tensor_leaves(different_pointer_case))
        clone_paths_match = mutated_leaves.keys() == different_leaves.keys() and bool(
            mutated_leaves
        )
        different_storage = clone_paths_match and all(
            not _shares_storage(mutated_leaves[path], different_leaves[path])
            for path in mutated_leaves
            if _tensor_numel(mutated_leaves[path])
        )
        equal_values = clone_paths_match and all(
            _tensor_bytes(mutated_leaves[path]) == _tensor_bytes(different_leaves[path])
            for path in mutated_leaves
        )
        _, different_pointer_result = _evaluate_case(
            adapter,
            different_pointer_case,
            atol=atol,
            rtol=rtol,
            anchor_atol=anchor_atol,
            anchor_rtol=anchor_rtol,
            exact_encoded_output=exact_encoded_output,
        )
        different_pointer = {
            "passed": bool(
                different_storage and equal_values and different_pointer_result["passed"]
            ),
            "different_storage": bool(different_storage),
            "equal_values": bool(equal_values),
            "correctness": different_pointer_result,
        }

        correctness = _aggregate_correctness(
            {
                "baseline": baseline,
                "second_input": second_input,
                "same_pointer_original_value": same_pointer_before_result,
                "same_pointer_changed_value": same_pointer_result,
                "different_pointer_same_value": different_pointer_result,
            },
            retained,
            same_pointer,
            different_pointer,
            second_input_changed,
        )
        if quantized_output_roles is not None:
            correctness["quantized_output_roles"] = quantized_output_roles
    except Exception as exc:
        return _empty_workload(
            workload, _exception_text(exc), execution=execution
        )

    hard_gates: dict[str, bool] = {
        "candidate_execution": True,
        "shape": all(
            bool(item["same_dtype"]["shape_match"])
            for item in correctness["probes"].values()
        ),
        "dtype": all(
            bool(item["same_dtype"]["dtype_match"])
            for item in correctness["probes"].values()
        ),
        "residency": all(
            bool(item["same_dtype"]["device_match"])
            for item in correctness["probes"].values()
        ),
        "layout": all(
            bool(item["same_dtype"]["layout_match"])
            for item in correctness["probes"].values()
        ),
        "finite": correctness["nonfinite_count"] == 0,
        "same_dtype_parity": all(
            bool(item["same_dtype"]["passed"])
            for item in correctness["probes"].values()
        ),
        "retained_output": bool(retained["passed"]),
        "same_pointer_changed_value": bool(same_pointer["passed"]),
        "different_pointer_same_value": bool(different_pointer["passed"]),
        "second_input_changed": bool(second_input_changed),
        "input_immutable": bool(correctness["input_immutable"]),
    }
    if profile is not None:
        hard_gates["fp32_anchor"] = bool(
            callable(getattr(adapter, "fp32_anchor", None))
            and correctness["fp32_anchor_present"]
            and correctness["fp32_anchor"]
        )
        hard_gates["graph_lifecycle"] = callable(
            getattr(adapter, "graph_state", None)
        )
    elif callable(getattr(adapter, "fp32_anchor", None)):
        hard_gates["fp32_anchor"] = bool(correctness["fp32_anchor"])
    if exact_encoded_output:
        quantized_probes = [
            probe["same_dtype"] for probe in correctness["probes"].values()
        ]
        role_paths = set(correctness["quantized_output_roles"])
        encoded_paths_match = bool(quantized_probes) and all(
            {leaf["path"] for leaf in probe["leaves"]} == role_paths
            for probe in quantized_probes
        )
        hard_gates.update(
            {
                "quantized_output_structure": encoded_paths_match
                and all(
                    probe["leaf_count"] == len(role_paths)
                    and not probe["errors"]
                    and all(
                        leaf["tensor_kind"] in {"integral", "floating"}
                        for leaf in probe["leaves"]
                    )
                    for probe in quantized_probes
                ),
                "quantized_output_bit_exact": bool(quantized_probes)
                and all(probe["byte_exact"] for probe in quantized_probes),
                "quantized_output_layout": bool(quantized_probes)
                and all(probe["layout_match"] for probe in quantized_probes),
                "quantized_dequantized_anchor": bool(quantized_probes)
                and all(
                    probe["fp32_anchor"] is not None
                    and probe["fp32_anchor"].get("kind")
                    == "dequantized_vs_fp32"
                    and probe["fp32_anchor"]["passed"]
                    for probe in correctness["probes"].values()
                ),
            }
        )

    if full_verdict:
        hard_gates.update(
            {
                "execution_evidence": execution is not None,
                "preflight_identity": preflight_sha256 != _ZERO_SHA256,
                "single_device_program": bool(
                    execution is not None
                    and execution["single_device_program"] is True
                ),
                "single_launch": bool(
                    execution is not None and execution["launch_count"] == 1
                ),
                "kernel_identity": bool(
                    execution is not None and execution["kernel_names"]
                ),
                "trace_summary_identity": bool(
                    execution is not None
                    and execution["trace_summary_sha256"] != _ZERO_SHA256
                ),
            }
        )
    if fusion_profile:
        hard_gates["bare_operation"] = bool(
            callable(getattr(adapter, "bare_operation", None))
            and callable(getattr(adapter, "bare_reference", None))
        )
        hard_gates["bare_operation_correctness"] = hard_gates["bare_operation"]
        hard_gates["paired_steady_replay"] = callable(
            getattr(adapter, "arm_graph_state", None)
        )

    raw_samples: dict[str, list[int]] = {"candidate_ns": [], "reference_ns": []}
    if fusion_profile:
        raw_samples["bare_operation_ns"] = []
    summary: dict[str, Any] = {
        "candidate_p50_ns": None,
        "reference_p50_ns": None,
    }
    stable_graph_state: dict[str, Any] | None = None
    final_graph_state: dict[str, Any] | None = None
    stable_arm_states: dict[str, dict[str, Any]] | None = None
    final_arm_states: dict[str, dict[str, Any]] | None = None
    order_blocks: list[str] = []

    if all(hard_gates.values()):
        try:
            timing_case = adapter.make_case(workload, seed + 10_000, device)
            bare_hook = getattr(adapter, "bare_operation", None)
            bare_reference = getattr(adapter, "bare_reference", None)
            include_bare = bool(
                fusion_profile
                and callable(bare_hook)
                and callable(bare_reference)
            )
            if include_bare:
                bare_case = _clone_tree(timing_case)
                bare_input_before = _case_fingerprint(bare_case)
                expected_bare = _invoke(adapter, "bare_reference", bare_case)
                bare_input_after_reference = _case_fingerprint(bare_case)
                actual_bare = _invoke(adapter, "bare_operation", bare_case)
                bare_input_after_candidate = _case_fingerprint(bare_case)
                bare_comparison = _compare_trees(
                    actual_bare,
                    expected_bare,
                    atol=atol,
                    rtol=rtol,
                    require_dtype=True,
                    require_device=True,
                    require_layout=True,
                )
                bare_input_immutable = bool(
                    bare_input_before
                    == bare_input_after_reference
                    == bare_input_after_candidate
                )
                bare_passed = bool(
                    bare_comparison["passed"] and bare_input_immutable
                )
                correctness["bare_operation"] = {
                    "identity": _BARE_OPERATION_IDENTITIES[profile],
                    "passed": bare_passed,
                    "input_immutable": bare_input_immutable,
                    "comparison": bare_comparison,
                }
                hard_gates["bare_operation_correctness"] = bare_passed
                if not bare_passed:
                    raise BenchError(
                        "bare_operation does not match its frozen semantic reference"
                    )
            for _ in range(warmup):
                _invoke(adapter, "candidate", timing_case)
                _invoke(adapter, "reference", timing_case)
            if include_bare:
                try:
                    _invoke(adapter, "bare_operation", timing_case)
                except NotImplementedError:
                    if fusion_profile:
                        hard_gates["bare_operation"] = False
                        raise BenchError(
                            "fusion profile requires an executable bare_operation()"
                        )
                    include_bare = False
            stable_graph_state = _read_graph_state(adapter)
            if stable_graph_state is not None and not _steady_replay_ready(
                stable_graph_state
            ):
                raise BenchError(
                    "timing requires an already-warm Graph state with one BUILD "
                    "and at least one REPLAY"
                )
            if fusion_profile:
                stable_arm_states = {
                    arm: _read_arm_graph_state(adapter, arm)
                    for arm in ("candidate", "bare_operation")
                }
                if not all(
                    _steady_replay_ready(stable_arm_states[arm])
                    for arm in ("candidate", "bare_operation")
                ):
                    raise BenchError(
                        "fusion timing requires both arms to start from steady REPLAY"
                    )

            order_seed = int.from_bytes(
                hashlib.sha256(f"{workload}\0{seed}".encode()).digest()[:8], "big"
            )
            rng = random.Random(order_seed)
            if include_bare and "bare_operation_ns" not in raw_samples:
                raw_samples["bare_operation_ns"] = []
            for _ in range(repeats):
                order, order_name = _timing_block(rng, include_bare)
                order_blocks.append(order_name)
                for hook_name in order:
                    raw_samples[hook_name + "_ns"].append(
                        _measure(adapter, hook_name, timing_case)
                    )

            final_graph_state = _read_graph_state(adapter)
            if fusion_profile:
                final_arm_states = {
                    arm: _read_arm_graph_state(adapter, arm)
                    for arm in ("candidate", "bare_operation")
                }
                hard_gates["paired_steady_replay"] = all(
                    _steady_replay_transition(
                        stable_arm_states[arm], final_arm_states[arm]
                    )
                    for arm in ("candidate", "bare_operation")
                )
            candidate_stats = _sample_stats(raw_samples["candidate_ns"])
            reference_stats = _sample_stats(raw_samples["reference_ns"])
            summary.update(
                {
                    "candidate_p50_ns": candidate_stats["p50_ns"],
                    "reference_p50_ns": reference_stats["p50_ns"],
                    "candidate_p95_ns": candidate_stats["p95_ns"],
                    "reference_p95_ns": reference_stats["p95_ns"],
                    "candidate_mad_ns": candidate_stats["mad_ns"],
                    "reference_mad_ns": reference_stats["mad_ns"],
                    "candidate_cv": candidate_stats["cv"],
                    "reference_cv": reference_stats["cv"],
                    "candidate_sample_count": candidate_stats["count"],
                    "reference_sample_count": reference_stats["count"],
                    "timing_order_blocks": order_blocks,
                }
            )
            if include_bare:
                bare_stats = _sample_stats(raw_samples["bare_operation_ns"])
                paired_delta = [
                    candidate - bare
                    for candidate, bare in zip(
                        raw_samples["candidate_ns"],
                        raw_samples["bare_operation_ns"],
                    )
                ]
                interval = _paired_epilogue_tax_interval(
                    raw_samples["candidate_ns"],
                    raw_samples["bare_operation_ns"],
                    confidence=(
                        fusion_confidence
                        if fusion_confidence is not None
                        else 0.95
                    ),
                    bootstrap_trials=(
                        fusion_bootstrap_trials
                        if fusion_bootstrap_trials is not None
                        else 1000
                    ),
                    bootstrap_seed=(
                        fusion_bootstrap_seed
                        if fusion_bootstrap_seed is not None
                        else 0
                    ),
                )
                summary.update(
                    {
                        "bare_operation_p50_ns": bare_stats["p50_ns"],
                        "bare_operation_p95_ns": bare_stats["p95_ns"],
                        "bare_operation_mad_ns": bare_stats["mad_ns"],
                        "bare_operation_cv": bare_stats["cv"],
                        "bare_operation_sample_count": bare_stats["count"],
                        "epilogue_tax_p50_ns": float(statistics.median(paired_delta)),
                        "epilogue_tax_p95_ns": float(_percentile(paired_delta, 0.95)),
                        **interval,
                    }
                )
                if max_epilogue_tax_pct is not None:
                    hard_gates["epilogue_tax"] = bool(
                        float(interval["epilogue_tax_ci_upper_pct"])
                        <= max_epilogue_tax_pct
                    )
                if full_fusion:
                    hard_gates["bare_operation"] = bool(
                        len(raw_samples["bare_operation_ns"]) == repeats * 2
                    )
            hard_gates["timing_samples"] = bool(
                len(raw_samples["candidate_ns"]) == repeats * 2
                and len(raw_samples["reference_ns"]) == repeats * 2
                and all(value > 0 for value in raw_samples["candidate_ns"])
                and all(value > 0 for value in raw_samples["reference_ns"])
            )
            if max_cv is not None:
                hard_gates["stability"] = bool(
                    candidate_stats["cv"] <= max_cv
                    and reference_stats["cv"] <= max_cv
                    and (
                        not include_bare
                        or bare_stats["cv"] <= max_cv
                    )
                )
        except Exception as exc:
            for samples in raw_samples.values():
                samples.clear()
            summary = {
                "candidate_p50_ns": None,
                "reference_p50_ns": None,
                "timing_error": _exception_text(exc),
            }
            if fusion_profile:
                hard_gates["bare_operation"] = False
                hard_gates["bare_operation_correctness"] = False
                hard_gates["paired_steady_replay"] = False
            hard_gates["timing_samples"] = False
    else:
        hard_gates["timing_samples"] = False

    graph_present = callable(getattr(adapter, "graph_state", None))
    if graph_present:
        graph_passed = bool(
            stable_graph_state is not None
            and final_graph_state is not None
            and _steady_replay_transition(stable_graph_state, final_graph_state)
        )
        hard_gates["graph_lifecycle"] = graph_passed
        graph_build_count = (
            final_graph_state["build_count"] if final_graph_state is not None else 0
        )
        graph_replay_count = (
            final_graph_state["replay_count"] if final_graph_state is not None else 0
        )
        cache_size_stable = bool(
            stable_graph_state is not None
            and final_graph_state is not None
            and stable_graph_state["cache_size"] == final_graph_state["cache_size"]
        )
        cache_invariant_ok = bool(
            stable_graph_state is not None
            and final_graph_state is not None
            and stable_graph_state["invariant_ok"]
            and final_graph_state["invariant_ok"]
        )
    else:
        graph_passed = True
        graph_build_count = 0
        graph_replay_count = 0
        cache_size_stable = True
        cache_invariant_ok = True

    # Re-read the retained output after every later candidate invocation.
    final_retained = _retained_output_probe(
        first_output, first_descriptors, first_bytes, second_output
    )
    if not final_retained["passed"]:
        hard_gates["retained_output"] = False
    independent_outputs = bool(retained["passed"] and final_retained["passed"])
    paired_steady_replay = bool(
        not fusion_profile
        or (
            stable_arm_states is not None
            and final_arm_states is not None
            and all(
                _steady_replay_transition(
                    stable_arm_states[arm], final_arm_states[arm]
                )
                for arm in ("candidate", "bare_operation")
            )
        )
    )
    lifecycle = {
        "passed": bool(graph_passed and independent_outputs and paired_steady_replay),
        "graph_build_count": graph_build_count,
        "graph_replay_count": graph_replay_count,
        "cache_size_stable": cache_size_stable,
        "cache_invariant_ok": cache_invariant_ok,
        "independent_outputs": independent_outputs,
        "graph_state_present": graph_present,
        "stable_state": stable_graph_state,
        "final_state": final_graph_state,
    }
    if fusion_profile:
        lifecycle["paired_steady_replay"] = paired_steady_replay
        lifecycle["arm_states"] = {
            "stable": stable_arm_states,
            "final": final_arm_states,
        }
    if not raw_samples["candidate_ns"]:
        # A stopped-before-timing receipt is evidence of failure, not a
        # partially promotable correctness/lifecycle result.
        correctness["passed"] = False
        lifecycle["passed"] = False
    eligible = bool(hard_gates) and all(hard_gates.values()) and lifecycle["passed"]
    return {
        "id": workload,
        "raw_samples": raw_samples,
        "summary": summary,
        "correctness": correctness,
        "lifecycle": lifecycle,
        "execution": execution,
        "hard_gates": hard_gates,
        "eligible": bool(eligible),
    }


def run_campaign(
    adapter: ModuleType,
    workloads: list[str],
    *,
    candidate_id: str,
    candidate_commit: str,
    parent_commit: str,
    contract_sha256: str,
    verdict: str,
    profile: str | None = None,
    preflight_sha256: str | None = None,
    device: str = "rpu",
    seed: int = 0,
    warmup: int = 3,
    repeats: int = 10,
    atol: float = 1.0e-5,
    rtol: float = 1.0e-5,
    anchor_atol: float | None = None,
    anchor_rtol: float | None = None,
    max_cv: float | None = None,
    max_epilogue_tax_pct: float | None = None,
    fusion_confidence: float | None = None,
    fusion_bootstrap_trials: int | None = None,
    fusion_bootstrap_seed: int | None = None,
    trusted_execution: Mapping[str, Any] | None = None,
    artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not workloads or any(
        _SAFE_ID_RE.fullmatch(item) is None for item in workloads
    ):
        raise BenchError("at least one safe workload identifier is required")
    if len(set(workloads)) != len(workloads):
        raise BenchError("workload IDs must be unique")
    if warmup < 1:
        raise BenchError("warmup must be at least 1")
    if repeats < 2:
        raise BenchError("repeats must be at least 2 (four samples per implementation)")
    for name, value in (("atol", atol), ("rtol", rtol)):
        if not math.isfinite(value) or value < 0:
            raise BenchError(f"{name} must be finite and non-negative")
    anchor_atol = atol if anchor_atol is None else anchor_atol
    anchor_rtol = rtol if anchor_rtol is None else anchor_rtol
    for name, value in (("anchor_atol", anchor_atol), ("anchor_rtol", anchor_rtol)):
        if not math.isfinite(value) or value < 0:
            raise BenchError(f"{name} must be finite and non-negative")
    if max_cv is not None and (not math.isfinite(max_cv) or max_cv < 0):
        raise BenchError("max_cv must be finite and non-negative")
    if max_epilogue_tax_pct is not None and (
        not math.isfinite(max_epilogue_tax_pct) or max_epilogue_tax_pct < 0
    ):
        raise BenchError("max_epilogue_tax_pct must be finite and non-negative")
    if fusion_confidence is not None and not 0.5 < fusion_confidence < 1.0:
        raise BenchError("fusion_confidence must be between 0.5 and 1")
    if fusion_bootstrap_trials is not None and fusion_bootstrap_trials < 1000:
        raise BenchError("fusion_bootstrap_trials must be at least 1000")
    if fusion_bootstrap_seed is not None and (
        not isinstance(fusion_bootstrap_seed, int)
        or isinstance(fusion_bootstrap_seed, bool)
        or fusion_bootstrap_seed < 0
    ):
        raise BenchError("fusion_bootstrap_seed must be a non-negative integer")
    if _SAFE_ID_RE.fullmatch(candidate_id) is None:
        raise BenchError("candidate_id is not a safe identifier")
    if not _COMMIT_RE.fullmatch(candidate_commit):
        raise BenchError("candidate_commit must be a full 40-hex commit ID")
    if not _COMMIT_RE.fullmatch(parent_commit):
        raise BenchError("parent_commit must be a full 40-hex commit ID")
    if not _SHA256_RE.fullmatch(contract_sha256):
        raise BenchError("contract_sha256 must be exactly 64 hexadecimal characters")
    if verdict not in {"signal", "full"}:
        raise BenchError("verdict must be 'signal' or 'full'")
    if profile is not None and profile not in _PROFILES:
        raise BenchError("profile is not a supported exact operation")
    if verdict == "full" and profile is None:
        raise BenchError("full verdict requires an exact --profile")
    normalized_preflight = preflight_sha256 or _ZERO_SHA256
    if _SHA256_RE.fullmatch(normalized_preflight) is None:
        raise BenchError("preflight_sha256 must be exactly 64 hexadecimal characters")
    normalized_preflight = normalized_preflight.lower()
    if verdict == "full" and normalized_preflight == _ZERO_SHA256:
        raise BenchError("full verdict requires a non-zero preflight_sha256")
    if verdict == "signal" and trusted_execution is not None:
        raise BenchError("signal verdict must not accept trusted execution evidence")
    normalized_execution: dict[str, dict[str, Any]] = {}
    if verdict == "full":
        if not isinstance(trusted_execution, Mapping):
            raise BenchError("full verdict requires verifier-owned execution evidence")
        if set(trusted_execution) != set(workloads):
            raise BenchError("trusted execution workloads do not match the benchmark")
        normalized_execution = {
            workload: _normalize_execution_evidence(
                trusted_execution[workload], workload, normalized_preflight
            )
            for workload in workloads
        }
    if verdict == "full" and profile in _FUSION_PROFILES:
        if max_epilogue_tax_pct is None:
            raise BenchError(
                "full fusion verdict requires contract max_epilogue_tax_pct"
            )
        if fusion_confidence is None:
            raise BenchError("full fusion verdict requires contract fusion_confidence")
        if fusion_bootstrap_trials is None:
            raise BenchError(
                "full fusion verdict requires contract fusion_bootstrap_trials"
            )
        if fusion_bootstrap_seed is None:
            raise BenchError(
                "full fusion verdict requires contract fusion_bootstrap_seed"
            )
    for hook_name in _REQUIRED_HOOKS:
        if not callable(getattr(adapter, hook_name, None)):
            raise BenchError(f"adapter is missing callable {hook_name}()")
    if profile == "rmsnorm+quant_mxfp8":
        for hook_name in ("fp32_anchor", "dequantize", "quantized_output_roles"):
            if not callable(getattr(adapter, hook_name, None)):
                raise BenchError(
                    "rmsnorm+quant_mxfp8 adapter is missing callable "
                    f"{hook_name}()"
                )

    workload_results = [
        _run_workload(
            adapter,
            workload,
            verdict=verdict,
            profile=profile,
            preflight_sha256=normalized_preflight,
            device=device,
            seed=seed + index * 1_000_003,
            warmup=warmup,
            repeats=repeats,
            atol=atol,
            rtol=rtol,
            anchor_atol=anchor_atol,
            anchor_rtol=anchor_rtol,
            max_cv=max_cv,
            max_epilogue_tax_pct=max_epilogue_tax_pct,
            fusion_confidence=fusion_confidence,
            fusion_bootstrap_trials=fusion_bootstrap_trials,
            fusion_bootstrap_seed=fusion_bootstrap_seed,
            trusted_execution=normalized_execution.get(workload),
        )
        for index, workload in enumerate(workloads)
    ]
    all_eligible = bool(workload_results) and all(
        bool(item["eligible"]) for item in workload_results
    )
    p50_values = [
        float(item["summary"]["candidate_p50_ns"])
        for item in workload_results
        if item["summary"]["candidate_p50_ns"] is not None
    ]
    metric_value = (
        float(statistics.median(p50_values)) / 1_000.0
        if all_eligible and len(p50_values) == len(workload_results)
        else None
    )
    top_hard_gates = {
        "workloads_present": bool(workload_results),
        "all_workloads_eligible": all_eligible,
        "metric_finite": bool(metric_value is not None and math.isfinite(metric_value)),
    }
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "candidate_id": candidate_id,
        "candidate_commit": candidate_commit.lower(),
        "parent_commit": parent_commit.lower(),
        "contract_sha256": contract_sha256.lower(),
        "preflight_sha256": normalized_preflight,
        "process": _read_process_identity(),
        "verdict": verdict,
        "metric": {
            "name": "latency_us",
            "direction": "min",
            "value": metric_value,
            "aggregation": "median_workload_candidate_p50",
        },
        "workloads": workload_results,
        "hard_gates": top_hard_gates,
    }
    if artifacts:
        result["artifacts"] = dict(sorted(artifacts.items()))
    # Refuse non-standard NaN/Infinity before a receipt writer sees this result.
    json.dumps(result, allow_nan=False)
    return result


def _load_adapter(path: Path) -> ModuleType:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise BenchError(f"adapter does not exist or is not a file: {resolved}")
    module_name = "rhinoforge_kernel_adapter_" + hashlib.sha256(
        str(resolved).encode()
    ).hexdigest()[:16]
    spec = importlib.util.spec_from_file_location(module_name, resolved)
    if spec is None or spec.loader is None:
        raise BenchError(f"cannot load adapter: {resolved}")
    module = importlib.util.module_from_spec(spec)
    # Importing an adapter executes arbitrary Python/native initialization.  It
    # is a protocol boundary, not a malicious-code security boundary.
    sys.path.insert(0, str(resolved.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="." + destination.name + ".",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _artifact(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be KEY=VALUE")
    key, item = value.split("=", 1)
    if _SAFE_TOKEN_RE.fullmatch(key) is None or not item:
        raise argparse.ArgumentTypeError("artifact must have a non-empty key and value")
    if len(item) > 4096 or any(ord(character) < 32 for character in item):
        raise argparse.ArgumentTypeError("artifact value is invalid")
    return key, item


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run correctness-gated paired kernel measurements through an adapter.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Adapter protocol (module-level callables):
  make_case(workload, seed, device)
  reference(case)
  candidate(case)
  synchronize()
Every exact --profile additionally requires:
  fp32_anchor(case), graph_state()
Fusion profiles additionally require:
  bare_operation(case), bare_reference(case), arm_graph_state(arm)
The rmsnorm+quant_mxfp8 profile additionally requires:
  fp32_anchor(case), dequantize(output, case), quantized_output_roles()
Optional only when --profile is omitted:
  fp32_anchor(case), graph_state()
Optional for every profile:
  mutate_case(case, seed)

graph_state() returns build_count, replay_count, cache_size, invariant_ok.
arm_graph_state('candidate'|'bare_operation') returns the same fields for each
timing arm so the epilogue-tax comparison proves independent steady REPLAY.
mutate_case must preserve tensor storage while changing at least one value.
quantized_output_roles() maps every encoded tensor-leaf path to exactly one of
payload, scale, or metadata and must include payload plus scale.

SECURITY: this executes adapter and native candidate code.  It is not a
malicious-code sandbox.  Run the final verifier in an administrator-approved,
controlled read-only harness and a clean process.  Local /proc identity is
recorded with process.attested=false. A cooperative wrapper may attest its
direct child for local gates, but release requires an external DSSE proof.
""",
    )
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--workload", required=True, action="append")
    parser.add_argument("--profile", choices=sorted(_PROFILES))
    parser.add_argument("--device", default="rpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--rtol", type=float, default=1.0e-5)
    parser.add_argument("--anchor-atol", type=float)
    parser.add_argument("--anchor-rtol", type=float)
    parser.add_argument("--max-cv", type=float)
    parser.add_argument("--max-epilogue-tax-pct", type=float)
    parser.add_argument("--fusion-confidence", type=float)
    parser.add_argument("--fusion-bootstrap-trials", type=int)
    parser.add_argument("--fusion-bootstrap-seed", type=int)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument(
        "--preflight-sha256",
        help="SHA-256 of the exact successful preflight receipt (required for full)",
    )
    parser.add_argument(
        "--trusted-execution",
        type=Path,
        help=(
            "verifier-owned execution evidence envelope; required for full and "
            "forbidden for signal"
        ),
    )
    parser.add_argument("--verdict", choices=("signal", "full"), required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        type=_artifact,
        metavar="KEY=VALUE",
        help="add a reviewed artifact path or hash to the receipt",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        adapter = _load_adapter(args.adapter)
        artifacts = dict(args.artifact)
        if len(artifacts) != len(args.artifact):
            raise BenchError("artifact keys must be unique")
        if args.verdict == "signal" and args.trusted_execution is not None:
            raise BenchError("--trusted-execution is forbidden for signal verdicts")
        trusted_execution = None
        if args.verdict == "full":
            if args.trusted_execution is None:
                raise BenchError("full verdict requires --trusted-execution")
            trusted_execution = _load_trusted_execution_file(
                args.trusted_execution,
                workloads=args.workload,
                contract_sha256=args.contract_sha256,
                preflight_sha256=args.preflight_sha256 or _ZERO_SHA256,
                candidate_commit=args.candidate_commit,
                parent_commit=args.parent_commit,
            )
        result = run_campaign(
            adapter,
            args.workload,
            candidate_id=args.candidate_id,
            candidate_commit=args.candidate_commit,
            parent_commit=args.parent_commit,
            contract_sha256=args.contract_sha256,
            verdict=args.verdict,
            profile=args.profile,
            preflight_sha256=args.preflight_sha256,
            device=args.device,
            seed=args.seed,
            warmup=args.warmup,
            repeats=args.repeats,
            atol=args.atol,
            rtol=args.rtol,
            anchor_atol=args.anchor_atol,
            anchor_rtol=args.anchor_rtol,
            max_cv=args.max_cv,
            max_epilogue_tax_pct=args.max_epilogue_tax_pct,
            fusion_confidence=args.fusion_confidence,
            fusion_bootstrap_trials=args.fusion_bootstrap_trials,
            fusion_bootstrap_seed=args.fusion_bootstrap_seed,
            trusted_execution=trusted_execution,
            artifacts=artifacts or None,
        )
        atomic_write_json(args.output, result)
    except Exception as exc:
        parser.exit(2, f"bench.py: error: {_exception_text(exc)}\n")
    return 0 if all(result["hard_gates"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
