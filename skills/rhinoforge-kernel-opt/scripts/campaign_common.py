#!/usr/bin/env python3
"""Shared, fail-closed validation for RhinoForge kernel campaigns.

This module deliberately depends only on the Python standard library.  Python
3.11+ uses ``tomllib``; the conservative fallback accepts the exact TOML subset
used by campaign contracts on Python 3.10 (tables, array tables, scalars, and
arrays).  It is not a general-purpose TOML implementation.
"""

from __future__ import annotations

import ast
import base64
import binascii
import hashlib
import json
import math
import os
import random
import re
import stat
import statistics
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


MAX_CONTRACT_BYTES = 1024 * 1024
MAX_RESULT_BYTES = 8 * 1024 * 1024
SCHEMA_VERSION = 1
SIGNAL_MINIMUM_SAMPLES = 3
SUPPORTED_OPS = frozenset(
    {
        "gemm",
        "gemm+silu_mul",
        "gemm+add",
        "gemm+rope",
        "rmsnorm+quant_mxfp8",
    }
)
FUSION_OPS = frozenset(
    {
        "gemm+silu_mul",
        "gemm+add",
        "gemm+rope",
        "rmsnorm+quant_mxfp8",
    }
)
BARE_OPERATION_IDENTITIES = {
    "gemm+silu_mul": "two_gemm_gate_up",
    "gemm+add": "gemm",
    "gemm+rope": "two_gemm_qk",
    "rmsnorm+quant_mxfp8": "rmsnorm",
}
UNRESOLVED_MARKERS = (
    "must-be-frozen",
    "replace-with",
    "replace_with",
    "unresolved",
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
ZERO_SHA256 = "0" * 64
SAFE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
SAFE_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.:+/-]{0,127}")
KERNEL_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,254}")
HARD_GATE_NAMES = (
    "finite",
    "correctness",
    "lifecycle",
    "same_dtype",
    "fp32_anchor",
    "graph_replay",
    "independent_outputs",
)


def _canonical_uuid(value: Any, label: str, *, version: int | None = None) -> str:
    text = _string(value, label, maximum=36)
    try:
        parsed = uuid.UUID(text)
    except (ValueError, AttributeError) as error:
        raise ValidationError(f"{label} must be a canonical UUID") from error
    if str(parsed) != text.lower() or (version is not None and parsed.version != version):
        suffix = f" version {version}" if version is not None else ""
        raise ValidationError(f"{label} must be a canonical UUID{suffix}")
    return text.lower()


class ValidationError(ValueError):
    """Input is not admissible for a campaign decision."""


def content_tree_sha256(root: Path) -> str:
    """Hash a bounded, symlink-free reference tree while excluding Git metadata."""

    supplied = root.expanduser()
    if supplied.is_symlink():
        raise ValidationError("reference_root must not be a symbolic link")
    try:
        resolved = supplied.resolve(strict=True)
    except OSError as error:
        raise ValidationError("reference_root is unavailable") from error
    if not resolved.is_dir():
        raise ValidationError("reference_root must be a directory")

    digest = hashlib.sha256(b"rhinoforge-reference-tree-v1\0")
    file_count = 0
    total_bytes = 0
    for directory, raw_directories, raw_files in os.walk(
        resolved, topdown=True, followlinks=False
    ):
        directory_path = Path(directory)
        kept_directories: list[str] = []
        for name in sorted(raw_directories):
            path = directory_path / name
            try:
                mode = os.lstat(path).st_mode
            except OSError as error:
                raise ValidationError("reference tree entry is unavailable") from error
            if stat.S_ISLNK(mode):
                raise ValidationError("reference tree must not contain symbolic links")
            if not stat.S_ISDIR(mode):
                raise ValidationError("reference tree contains an invalid directory entry")
            if name != ".git":
                kept_directories.append(name)
        raw_directories[:] = kept_directories
        for name in sorted(raw_files):
            path = directory_path / name
            relative = path.relative_to(resolved).as_posix()
            if "/.git/" in f"/{relative}/" or relative == ".git":
                continue
            try:
                path_stat = os.lstat(path)
            except OSError as error:
                raise ValidationError("reference tree file is unavailable") from error
            if not stat.S_ISREG(path_stat.st_mode):
                raise ValidationError("reference tree must contain only regular files")
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(path, flags)
            except OSError as error:
                raise ValidationError("reference tree file cannot be opened safely") from error
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino)
                    != (path_stat.st_dev, path_stat.st_ino)
                ):
                    raise ValidationError("reference tree file identity changed")
                file_digest = hashlib.sha256()
                observed_bytes = 0
                while True:
                    block = os.read(descriptor, 1024 * 1024)
                    if not block:
                        break
                    observed_bytes += len(block)
                    total_bytes += len(block)
                    if total_bytes > 2 * 1024 * 1024 * 1024:
                        raise ValidationError("reference tree exceeds 2 GiB")
                    file_digest.update(block)
                final_stat = os.fstat(descriptor)
                if (
                    observed_bytes != opened.st_size
                    or final_stat.st_size != opened.st_size
                    or final_stat.st_mtime_ns != opened.st_mtime_ns
                ):
                    raise ValidationError("reference tree file changed while hashing")
            finally:
                os.close(descriptor)
            file_count += 1
            if file_count > 100_000:
                raise ValidationError("reference tree contains too many files")
            relative_bytes = relative.encode("utf-8", errors="surrogateescape")
            executable = b"1" if opened.st_mode & 0o111 else b"0"
            for component in (
                relative_bytes,
                executable,
                observed_bytes.to_bytes(8, "big"),
                file_digest.digest(),
            ):
                digest.update(len(component).to_bytes(8, "big"))
                digest.update(component)
    if file_count == 0:
        raise ValidationError("reference tree must contain at least one regular file")
    return digest.hexdigest()


def _reject_json_constant(value: str) -> None:
    raise ValidationError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_limited_bytes(path: Path, maximum: int, label: str) -> bytes:
    try:
        stat_result = path.stat()
    except OSError as error:
        raise ValidationError(f"{label} is unavailable") from error
    if not path.is_file():
        raise ValidationError(f"{label} is not a regular file")
    if stat_result.st_size > maximum:
        raise ValidationError(f"{label} exceeds {maximum} bytes")
    try:
        return path.read_bytes()
    except OSError as error:
        raise ValidationError(f"{label} is unreadable") from error


def load_json(path: Path, maximum: int = MAX_RESULT_BYTES) -> Any:
    raw = read_limited_bytes(path, maximum, "JSON input")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError("JSON input is not UTF-8") from error
    try:
        return strict_json_loads(text)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValidationError("JSON input is invalid") from error


def strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_json_constant,
    )


def canonical_json(value: Any) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise ValidationError("value cannot be encoded as strict JSON") from error


def contract_sha256(path: Path) -> str:
    return hashlib.sha256(
        read_limited_bytes(path, MAX_CONTRACT_BYTES, "contract")
    ).hexdigest()


def _strip_toml_comment(line: str) -> str:
    quote = ""
    escaped = False
    for index, char in enumerate(line):
        if quote:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in ('"', "'"):
            quote = char
        elif char == "#":
            return line[:index]
    return line


def _balanced_array(value: str) -> bool:
    depth = 0
    quote = ""
    escaped = False
    for char in value:
        if quote:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in ('"', "'"):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                return False
    return not quote and depth == 0


def _split_toml_array(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    depth = 0
    quote = ""
    escaped = False
    for index, char in enumerate(value):
        if quote:
            if quote == '"' and escaped:
                escaped = False
            elif quote == '"' and char == "\\":
                escaped = True
            elif char == quote:
                quote = ""
        elif char in ('"', "'"):
            quote = char
        elif char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(value[start:index].strip())
            start = index + 1
    tail = value[start:].strip()
    if tail:
        parts.append(tail)
    elif parts and value.rstrip().endswith(","):
        pass
    elif value.strip():
        parts.append("")
    return parts


def _parse_toml_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        raise ValidationError("empty TOML value")
    if value.startswith("["):
        if not value.endswith("]") or not _balanced_array(value):
            raise ValidationError("invalid TOML array")
        body = value[1:-1].strip()
        if not body:
            return []
        return [_parse_toml_scalar(part) for part in _split_toml_array(body)]
    if value[:1] in ('"', "'"):
        if value[-1:] != value[:1]:
            raise ValidationError("unterminated TOML string")
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError) as error:
            raise ValidationError("invalid TOML string") from error
        if not isinstance(parsed, str):
            raise ValidationError("invalid TOML string")
        return parsed
    if value == "true":
        return True
    if value == "false":
        return False
    numeric = value.replace("_", "")
    if re.fullmatch(r"[+-]?[0-9]+", numeric):
        return int(numeric, 10)
    if re.fullmatch(
        r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?",
        numeric,
    ):
        return float(numeric)
    raise ValidationError("unsupported TOML syntax in Python 3.10 fallback")


def _load_toml_subset(text: str) -> dict[str, Any]:
    """Parse only the conservative syntax accepted by campaign contracts."""

    root: dict[str, Any] = {}
    current: dict[str, Any] = root
    logical_lines: list[str] = []
    pending = ""
    for physical in text.splitlines():
        clean = _strip_toml_comment(physical).strip()
        if not clean:
            continue
        pending = f"{pending} {clean}".strip() if pending else clean
        if "=" in pending:
            _, possible_value = pending.split("=", 1)
            if possible_value.lstrip().startswith("[") and not _balanced_array(
                possible_value
            ):
                continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        raise ValidationError("unterminated TOML value")

    for line in logical_lines:
        if line.startswith("[[") and line.endswith("]]"):
            name = line[2:-2].strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
                raise ValidationError("invalid TOML array-table name")
            existing = root.setdefault(name, [])
            if not isinstance(existing, list):
                raise ValidationError(f"duplicate TOML table: {name}")
            item: dict[str, Any] = {}
            existing.append(item)
            current = item
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1].strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", name):
                raise ValidationError("invalid TOML table name")
            if name in root:
                raise ValidationError(f"duplicate TOML table: {name}")
            table: dict[str, Any] = {}
            root[name] = table
            current = table
            continue
        if "=" not in line:
            raise ValidationError("invalid TOML assignment")
        key, value = line.split("=", 1)
        key = key.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*", key):
            raise ValidationError("invalid TOML key")
        if key in current:
            raise ValidationError(f"duplicate TOML key: {key}")
        current[key] = _parse_toml_scalar(value)
    return root


def load_toml(path: Path) -> dict[str, Any]:
    raw = read_limited_bytes(path, MAX_CONTRACT_BYTES, "contract")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValidationError("contract is not UTF-8") from error
    try:
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:  # pragma: no cover - exercised on Python 3.10
            parsed = _load_toml_subset(text)
        else:
            parsed = tomllib.loads(text)
    except ValidationError:
        raise
    except Exception as error:
        raise ValidationError("contract is not valid TOML") from error
    if not isinstance(parsed, dict):
        raise ValidationError("contract root must be a table")
    return parsed


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a table/object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ValidationError(f"{label} must be an array/list")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    required: Iterable[str],
    optional: Iterable[str],
    label: str,
) -> None:
    required_set = set(required)
    optional_set = set(optional)
    missing = sorted(required_set - set(value))
    unknown = sorted(set(value) - required_set - optional_set)
    if missing:
        raise ValidationError(f"{label} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ValidationError(f"{label} has unknown keys: {', '.join(unknown)}")


def _string(value: Any, label: str, *, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValidationError(f"{label} must be a non-empty string")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValidationError(f"{label} contains control characters")
    return value


def _bool(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise ValidationError(f"{label} must be boolean")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValidationError(f"{label} must be an integer >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0) or (
        not positive and result < 0
    ):
        comparator = "> 0" if positive else ">= 0"
        raise ValidationError(f"{label} must be finite and {comparator}")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite")
    return result


def _validate_json_evidence(
    value: Any, label: str, *, depth: int = 0, budget: list[int] | None = None
) -> None:
    """Bound optional diagnostic evidence without trusting any of its claims."""

    if budget is None:
        budget = [10_000]
    budget[0] -= 1
    if budget[0] < 0 or depth > 12:
        raise ValidationError(f"{label} diagnostic evidence is too large or deep")
    if value is None or type(value) is bool or type(value) is int:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValidationError(f"{label} contains a non-finite number")
        return
    if isinstance(value, str):
        _string(value, label, maximum=4096)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_evidence(
                item, f"{label}[{index}]", depth=depth + 1, budget=budget
            )
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _string(key, f"{label} key", maximum=128)
            _validate_json_evidence(
                item, f"{label}.{key}", depth=depth + 1, budget=budget
            )
        return
    raise ValidationError(f"{label} contains an unsupported value")


def _safe_id(value: Any, label: str) -> str:
    result = _string(value, label, maximum=128)
    if SAFE_ID_RE.fullmatch(result) is None:
        raise ValidationError(f"{label} is not a safe identifier")
    return result


def validate_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized schema-v1 campaign contract."""

    _exact_keys(
        contract,
        (
            "schema_version",
            "state",
            "campaign",
            "commands",
            "attestation",
            "runtime",
            "measurement",
            "gates",
            "workloads",
        ),
        ("roofline",),
        "contract",
    )
    if type(contract["schema_version"]) is not int or contract["schema_version"] != 1:
        raise ValidationError("schema_version must be integer 1")
    if contract["state"] != "approved":
        raise ValidationError("state must be exactly 'approved'")

    attestation = _mapping(contract["attestation"], "attestation")
    _exact_keys(
        attestation,
        (
            "mode",
            "scheme",
            "namespace",
            "signer_id",
            "public_key_base64",
            "policy_sha256",
            "signer_build_sha256",
            "verifier_executable",
            "verifier_sha256",
        ),
        (),
        "attestation",
    )
    if attestation["namespace"] != "rhinoforge-full-verdict-v1":
        raise ValidationError(
            "attestation.namespace must be exactly 'rhinoforge-full-verdict-v1'"
        )
    attestation_mode = attestation["mode"]
    if attestation_mode not in {"external", "unavailable"}:
        raise ValidationError("attestation.mode must be external or unavailable")
    if attestation_mode == "unavailable":
        expected_unavailable = {
            "scheme": "none",
            "signer_id": "not-available",
            "public_key_base64": "not-available",
            "policy_sha256": ZERO_SHA256,
            "signer_build_sha256": ZERO_SHA256,
            "verifier_executable": "not-available",
            "verifier_sha256": ZERO_SHA256,
        }
        if any(attestation[field] != value for field, value in expected_unavailable.items()):
            raise ValidationError(
                "unavailable attestation must use exact not-available/zero sentinels"
            )
    else:
        if attestation["scheme"] != "ed25519":
            raise ValidationError("external attestation.scheme must be ed25519")
        _safe_id(attestation["signer_id"], "attestation.signer_id")
        public_key_text = _string(
            attestation["public_key_base64"],
            "attestation.public_key_base64",
            maximum=128,
        )
        try:
            public_key = base64.b64decode(public_key_text, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ValidationError(
                "attestation.public_key_base64 must be canonical base64"
            ) from error
        if (
            len(public_key) != 32
            or base64.b64encode(public_key).decode("ascii") != public_key_text
        ):
            raise ValidationError(
                "attestation.public_key_base64 must encode one raw Ed25519 public key"
            )
        for field in ("policy_sha256", "signer_build_sha256"):
            digest = _string(attestation[field], f"attestation.{field}")
            if SHA256_RE.fullmatch(digest) is None or digest == ZERO_SHA256:
                raise ValidationError(
                    f"attestation.{field} must be a frozen nonzero SHA-256"
                )
        verifier_executable = _string(
            attestation["verifier_executable"],
            "attestation.verifier_executable",
            maximum=4096,
        )
        if not Path(verifier_executable).is_absolute():
            raise ValidationError("attestation.verifier_executable must be absolute")
        verifier_hash = _string(
            attestation["verifier_sha256"], "attestation.verifier_sha256"
        )
        if SHA256_RE.fullmatch(verifier_hash) is None or verifier_hash == ZERO_SHA256:
            raise ValidationError(
                "attestation.verifier_sha256 must be a frozen nonzero SHA-256"
            )

    campaign = _mapping(contract["campaign"], "campaign")
    _exact_keys(
        campaign,
        (
            "id",
            "op",
            "candidate_root",
            "reference_root",
            "reference_tree_sha256",
        ),
        (),
        "campaign",
    )
    _safe_id(campaign["id"], "campaign.id")
    op = _string(campaign["op"], "campaign.op", maximum=64)
    if op not in SUPPORTED_OPS:
        raise ValidationError("campaign.op is not a supported exact operation")
    _string(campaign["candidate_root"], "campaign.candidate_root", maximum=4096)
    _string(campaign["reference_root"], "campaign.reference_root", maximum=4096)
    reference_tree_hash = _string(
        campaign["reference_tree_sha256"], "campaign.reference_tree_sha256"
    )
    if SHA256_RE.fullmatch(reference_tree_hash) is None or reference_tree_hash == ZERO_SHA256:
        raise ValidationError(
            "campaign.reference_tree_sha256 must be a frozen nonzero SHA-256"
        )

    commands = _mapping(contract["commands"], "commands")
    _exact_keys(
        commands,
        (
            "correctness",
            "benchmark",
            "profile_one_call",
            "benchmark_adapter",
            "benchmark_adapter_sha256",
        ),
        (),
        "commands",
    )
    for command_name in ("correctness", "benchmark", "profile_one_call"):
        argv = _sequence(commands[command_name], f"commands.{command_name}")
        if not argv:
            raise ValidationError(f"commands.{command_name} must not be empty")
        for index, arg in enumerate(argv):
            _string(arg, f"commands.{command_name}[{index}]", maximum=4096)
    adapter_path = _string(
        commands["benchmark_adapter"], "commands.benchmark_adapter", maximum=4096
    )
    if any(marker in adapter_path.lower() for marker in UNRESOLVED_MARKERS):
        raise ValidationError("commands.benchmark_adapter is unresolved")
    adapter_hash = _string(
        commands["benchmark_adapter_sha256"], "commands.benchmark_adapter_sha256"
    )
    if SHA256_RE.fullmatch(adapter_hash) is None or adapter_hash == ZERO_SHA256:
        raise ValidationError(
            "commands.benchmark_adapter_sha256 must be a frozen nonzero SHA-256"
        )

    runtime = _mapping(contract["runtime"], "runtime")
    _exact_keys(
        runtime,
        (
            "rhinoforge_commit",
            "runtime_set",
            "torch_version",
            "launch_version",
            "launch_library",
            "launch_library_sha256",
            "operator_version",
            "operator_asset",
            "operator_asset_sha256",
            "kernel_manifest",
            "kernel_manifest_sha256",
            "required_kernels",
            "requires_device_compiler",
            "device_compiler_authorized",
            "device_compiler_id",
            "device_compiler_sha256",
            "compiler_authorization_receipt",
            "compiler_authorization_receipt_sha256",
            "asset_release_owner",
            "trace_schema",
            "trace_event_category",
        ),
        ("fused_kernel",),
        "runtime",
    )
    commit = _string(runtime["rhinoforge_commit"], "runtime.rhinoforge_commit")
    if COMMIT_RE.fullmatch(commit) is None:
        raise ValidationError("runtime.rhinoforge_commit must be a full lowercase commit")
    if runtime["trace_schema"] != "rhinoforge-rpu-chrome-v1":
        raise ValidationError(
            "runtime.trace_schema must be exactly 'rhinoforge-rpu-chrome-v1'"
        )
    if runtime["trace_event_category"] != "rpu_device_program":
        raise ValidationError(
            "runtime.trace_event_category must be exactly 'rpu_device_program'"
        )
    runtime_set = _string(runtime["runtime_set"], "runtime.runtime_set", maximum=256)
    if SAFE_TOKEN_RE.fullmatch(runtime_set) is None or "/" in runtime_set:
        raise ValidationError("runtime.runtime_set must be a path-free exact identifier")
    for field in ("torch_version", "launch_version", "operator_version"):
        _string(runtime[field], f"runtime.{field}", maximum=256)
    _string(runtime["launch_library"], "runtime.launch_library", maximum=4096)
    launch_hash = _string(
        runtime["launch_library_sha256"], "runtime.launch_library_sha256"
    )
    if SHA256_RE.fullmatch(launch_hash) is None:
        raise ValidationError("runtime.launch_library_sha256 must be lowercase SHA-256")
    asset = _string(runtime["operator_asset"], "runtime.operator_asset", maximum=4096)
    asset_hash = _string(
        runtime["operator_asset_sha256"], "runtime.operator_asset_sha256"
    )
    if SHA256_RE.fullmatch(asset_hash) is None:
        raise ValidationError("runtime.operator_asset_sha256 must be lowercase SHA-256")
    manifest = _string(
        runtime["kernel_manifest"], "runtime.kernel_manifest", maximum=4096
    )
    manifest_hash = _string(
        runtime["kernel_manifest_sha256"], "runtime.kernel_manifest_sha256"
    )
    if SHA256_RE.fullmatch(manifest_hash) is None:
        raise ValidationError("runtime.kernel_manifest_sha256 must be lowercase SHA-256")
    if not asset.endswith(".ref"):
        raise ValidationError("runtime.operator_asset must name an opaque .ref asset")
    if manifest != asset + ".kernels":
        raise ValidationError(
            "runtime.kernel_manifest must exactly equal operator_asset + '.kernels'"
        )
    kernels = _sequence(runtime["required_kernels"], "runtime.required_kernels")
    if not kernels:
        raise ValidationError("runtime.required_kernels must not be empty")
    normalized_kernels: list[str] = []
    for index, kernel in enumerate(kernels):
        name = _string(kernel, f"runtime.required_kernels[{index}]", maximum=255)
        if KERNEL_NAME_RE.fullmatch(name) is None:
            raise ValidationError("runtime.required_kernels contains an invalid name")
        if name.upper().startswith("REPLACE_WITH_"):
            raise ValidationError(
                "runtime.required_kernels contains an unresolved fusion sentinel"
            )
        normalized_kernels.append(name)
    if len(set(normalized_kernels)) != len(normalized_kernels):
        raise ValidationError("runtime.required_kernels must be unique")
    if op in FUSION_OPS:
        if "fused_kernel" not in runtime:
            raise ValidationError(
                "fusion contract requires runtime.fused_kernel"
            )
        fused_kernel = _string(
            runtime["fused_kernel"], "runtime.fused_kernel", maximum=255
        )
        if (
            KERNEL_NAME_RE.fullmatch(fused_kernel) is None
            or fused_kernel.upper().startswith("REPLACE_WITH_")
            or fused_kernel not in normalized_kernels
        ):
            raise ValidationError(
                "runtime.fused_kernel must be one explicit resolved required kernel"
            )
    elif "fused_kernel" in runtime:
        raise ValidationError(
            "runtime.fused_kernel is valid only for a fusion campaign"
        )
    requires_compiler = _bool(
        runtime["requires_device_compiler"], "runtime.requires_device_compiler"
    )
    compiler_authorized = _bool(
        runtime["device_compiler_authorized"],
        "runtime.device_compiler_authorized",
    )
    release_owner = _string(
        runtime["asset_release_owner"], "runtime.asset_release_owner", maximum=256
    )
    compiler_id = _string(
        runtime["device_compiler_id"], "runtime.device_compiler_id", maximum=256
    )
    compiler_hash = _string(
        runtime["device_compiler_sha256"],
        "runtime.device_compiler_sha256",
    )
    if SHA256_RE.fullmatch(compiler_hash) is None:
        raise ValidationError("runtime.device_compiler_sha256 must be lowercase SHA-256")
    authorization_receipt = _string(
        runtime["compiler_authorization_receipt"],
        "runtime.compiler_authorization_receipt",
        maximum=4096,
    )
    authorization_hash = _string(
        runtime["compiler_authorization_receipt_sha256"],
        "runtime.compiler_authorization_receipt_sha256",
    )
    if SHA256_RE.fullmatch(authorization_hash) is None:
        raise ValidationError(
            "runtime.compiler_authorization_receipt_sha256 must be lowercase SHA-256"
        )
    if requires_compiler and not compiler_authorized:
        raise ValidationError(
            "compiler-required contract needs device_compiler_authorized=true"
        )
    if requires_compiler and release_owner.strip().upper() in {
        "UNRESOLVED",
        "UNKNOWN",
        "TODO",
        "TBD",
        "NONE",
    }:
        raise ValidationError(
            "compiler-required contract needs a resolved asset_release_owner"
        )
    if requires_compiler and compiler_id.strip().lower() in {
        "not-required",
        "unresolved",
        "unknown",
        "todo",
        "tbd",
    }:
        raise ValidationError(
            "compiler-required contract needs an exact device_compiler_id"
        )
    if requires_compiler and compiler_hash == ZERO_SHA256:
        raise ValidationError(
            "compiler-required contract needs the exact device compiler SHA-256"
        )
    if requires_compiler and (
        authorization_hash == ZERO_SHA256
        or authorization_receipt.strip().lower()
        in {"unresolved", "unknown", "todo", "tbd", "none", "not-required"}
    ):
        raise ValidationError(
            "compiler-required contract needs a hashed authorization receipt"
        )
    if not requires_compiler and compiler_id != "not-required":
        raise ValidationError(
            "non-compiler contract must set device_compiler_id='not-required'"
        )
    if not requires_compiler and (
        compiler_hash != ZERO_SHA256
        or authorization_receipt != "not-required"
        or authorization_hash != ZERO_SHA256
    ):
        raise ValidationError(
            "non-compiler contract must use not-required/zero compiler evidence"
        )

    measurement = _mapping(contract["measurement"], "measurement")
    _exact_keys(
        measurement,
        ("metric", "direction", "warmup", "samples", "verdict_processes"),
        (),
        "measurement",
    )
    if measurement["metric"] != "latency_us":
        raise ValidationError("measurement.metric must be exactly 'latency_us'")
    if measurement["direction"] != "min":
        raise ValidationError("measurement.direction must be exactly 'min'")
    _integer(measurement["warmup"], "measurement.warmup", minimum=1)
    _integer(measurement["samples"], "measurement.samples", minimum=1)
    _integer(
        measurement["verdict_processes"],
        "measurement.verdict_processes",
        minimum=2,
    )

    gates = _mapping(contract["gates"], "gates")
    _exact_keys(
        gates,
        (
            "max_abs",
            "max_rel",
            "anchor_max_abs",
            "anchor_max_rel",
            "max_cv",
            "require_same_dtype",
            "require_fp32_anchor",
            "require_graph_replay",
            "require_independent_outputs",
        ),
        (
            "max_epilogue_tax_pct",
            "fusion_confidence",
            "fusion_bootstrap_trials",
            "fusion_bootstrap_seed",
        ),
        "gates",
    )
    _number(gates["max_abs"], "gates.max_abs")
    _number(gates["max_rel"], "gates.max_rel")
    _number(gates["anchor_max_abs"], "gates.anchor_max_abs")
    _number(gates["anchor_max_rel"], "gates.anchor_max_rel")
    _number(gates["max_cv"], "gates.max_cv")
    for field in (
        "require_same_dtype",
        "require_fp32_anchor",
        "require_graph_replay",
        "require_independent_outputs",
    ):
        _bool(gates[field], f"gates.{field}")
    fusion_gate_fields = {
        "max_epilogue_tax_pct",
        "fusion_confidence",
        "fusion_bootstrap_trials",
        "fusion_bootstrap_seed",
    }
    if op in FUSION_OPS:
        missing_fusion_gates = sorted(fusion_gate_fields - set(gates))
        if missing_fusion_gates:
            raise ValidationError(
                "fusion contract is missing gates: " + ", ".join(missing_fusion_gates)
            )
        _number(gates["max_epilogue_tax_pct"], "gates.max_epilogue_tax_pct")
        confidence = _number(gates["fusion_confidence"], "gates.fusion_confidence")
        if confidence <= 0.5 or confidence >= 1:
            raise ValidationError("gates.fusion_confidence must be between 0.5 and 1")
        _integer(
            gates["fusion_bootstrap_trials"],
            "gates.fusion_bootstrap_trials",
            minimum=1000,
        )
        _integer(
            gates["fusion_bootstrap_seed"],
            "gates.fusion_bootstrap_seed",
        )
    elif fusion_gate_fields & set(gates):
        raise ValidationError("non-fusion contract must not declare fusion-only gates")

    workloads = _sequence(contract["workloads"], "workloads")
    if not workloads:
        raise ValidationError("workloads must not be empty")
    workload_ids: list[str] = []
    for index, raw_workload in enumerate(workloads):
        workload = _mapping(raw_workload, f"workloads[{index}]")
        _exact_keys(
            workload,
            ("id", "m", "n", "k", "dtype", "layout", "semantic_inputs"),
            (),
            f"workloads[{index}]",
        )
        workload_ids.append(_safe_id(workload["id"], f"workloads[{index}].id"))
        for dimension in ("m", "n", "k"):
            value = _integer(
                workload[dimension], f"workloads[{index}].{dimension}", minimum=1
            )
            if value > 2**31 - 1:
                raise ValidationError("workload dimension exceeds signed 32-bit range")
        dtype = _string(
            workload["dtype"], f"workloads[{index}].dtype", maximum=128
        )
        if SAFE_TOKEN_RE.fullmatch(dtype) is None:
            raise ValidationError(f"workloads[{index}].dtype is invalid")
        _string(
            workload["layout"], f"workloads[{index}].layout", maximum=512
        )
        semantics = _sequence(
            workload["semantic_inputs"], f"workloads[{index}].semantic_inputs"
        )
        if not semantics:
            raise ValidationError("workload semantic_inputs must not be empty")
        semantic_names: list[str] = []
        for semantic_index, semantic in enumerate(semantics):
            value = _string(
                semantic,
                f"workloads[{index}].semantic_inputs[{semantic_index}]",
                maximum=128,
            )
            normalized_value = value.strip().lower()
            if normalized_value in {"unknown", "todo", "tbd", "none"} or any(
                marker in normalized_value for marker in UNRESOLVED_MARKERS
            ):
                raise ValidationError(
                    "workload semantic_inputs contains an unresolved sentinel"
                )
            semantic_names.append(value)
        if len(semantic_names) != len(set(semantic_names)):
            raise ValidationError("workload semantic_inputs must be unique")
    if len(workload_ids) != len(set(workload_ids)):
        raise ValidationError("workload ids must be unique")

    roofline = contract.get("roofline")
    if roofline is not None:
        roofline_map = _mapping(roofline, "roofline")
        _exact_keys(
            roofline_map,
            (
                "peak_ops_per_s",
                "bandwidth_bytes_per_s",
                "launch_floor_s",
                "peak_kind",
                "source_id",
                "source_file",
                "source_sha256",
            ),
            (),
            "roofline",
        )
        for field in ("peak_ops_per_s", "bandwidth_bytes_per_s", "launch_floor_s"):
            value = roofline_map[field]
            _number(value, f"roofline.{field}", positive=True)
        if roofline_map["peak_kind"] not in {"authoritative", "empirical"}:
            raise ValidationError("roofline.peak_kind must be authoritative or empirical")
        source_id = _string(roofline_map["source_id"], "roofline.source_id", maximum=256)
        if any(marker in source_id.lower() for marker in UNRESOLVED_MARKERS):
            raise ValidationError("roofline.source_id is unresolved")
        _string(roofline_map["source_file"], "roofline.source_file", maximum=4096)
        source_hash = _string(roofline_map["source_sha256"], "roofline.source_sha256")
        if SHA256_RE.fullmatch(source_hash) is None or source_hash == ZERO_SHA256:
            raise ValidationError("roofline.source_sha256 must be a frozen nonzero SHA-256")

    # A JSON round trip produces mutable plain containers without TOML-specific
    # scalar subclasses and also rejects anything outside strict JSON.
    return json.loads(canonical_json(contract))


def load_contract(path: Path) -> dict[str, Any]:
    return validate_contract(load_toml(path))


def _strict_bool_map(
    value: Any, label: str, *, exact_core: bool = False
) -> dict[str, bool]:
    mapping = _mapping(value, label)
    if not mapping:
        raise ValidationError(f"{label} must not be empty")
    if exact_core:
        _exact_keys(mapping, HARD_GATE_NAMES, (), label)
    result: dict[str, bool] = {}
    for name, value_item in mapping.items():
        if SAFE_TOKEN_RE.fullmatch(_string(name, f"{label} key", maximum=128)) is None:
            raise ValidationError(f"{label} contains an invalid gate name")
        result[name] = _bool(value_item, f"{label}.{name}")
    return result


def _derive_quantized_output_gates(
    correctness: Mapping[str, Any],
) -> dict[str, bool]:
    """Recompute encoded-output gates from signed journal evidence."""

    if "quantized_output_roles" not in correctness:
        raise ValidationError(
            "timed rmsnorm+quant_mxfp8 workload lacks quantized output roles"
        )
    roles_raw = _mapping(
        correctness["quantized_output_roles"],
        "correctness.quantized_output_roles",
    )
    roles: dict[str, str] = {}
    for raw_path, raw_role in roles_raw.items():
        path = _string(raw_path, "quantized output role path", maximum=512)
        role = _string(raw_role, f"quantized output role {path}", maximum=16)
        if not path.startswith("$"):
            raise ValidationError("quantized output role paths must start with '$'")
        if role not in {"payload", "scale", "metadata"}:
            raise ValidationError(
                "quantized output roles must be payload, scale, or metadata"
            )
        roles[path] = role
    if not roles or "payload" not in roles.values() or "scale" not in roles.values():
        raise ValidationError("quantized output roles require payload and scale leaves")

    probes = _mapping(correctness.get("probes"), "correctness.probes")
    if not probes:
        raise ValidationError("quantized correctness probes must not be empty")
    structure_ok = True
    bit_exact_ok = True
    layout_ok = True
    anchor_ok = True
    expected_paths = set(roles)
    for probe_name, raw_probe in probes.items():
        probe = _mapping(raw_probe, f"correctness.probes.{probe_name}")
        comparison = _mapping(
            probe.get("same_dtype"),
            f"correctness.probes.{probe_name}.same_dtype",
        )
        leaves = _sequence(
            comparison.get("leaves"),
            f"correctness.probes.{probe_name}.same_dtype.leaves",
        )
        leaf_count = _integer(
            comparison.get("leaf_count"),
            f"correctness.probes.{probe_name}.same_dtype.leaf_count",
        )
        errors = _sequence(
            comparison.get("errors"),
            f"correctness.probes.{probe_name}.same_dtype.errors",
        )
        for index, error in enumerate(errors):
            _string(
                error,
                f"correctness.probes.{probe_name}.same_dtype.errors[{index}]",
                maximum=4096,
            )
        paths: list[str] = []
        leaf_kinds_ok = True
        probe_bit_exact = True
        probe_layout = True
        for index, raw_leaf in enumerate(leaves):
            leaf = _mapping(
                raw_leaf,
                f"correctness.probes.{probe_name}.same_dtype.leaves[{index}]",
            )
            path = _string(
                leaf.get("path"),
                f"correctness.probes.{probe_name}.same_dtype.leaves[{index}].path",
                maximum=512,
            )
            paths.append(path)
            tensor_kind = _string(
                leaf.get("tensor_kind"),
                f"correctness.probes.{probe_name}.same_dtype.leaves[{index}].tensor_kind",
                maximum=16,
            )
            leaf_kinds_ok = leaf_kinds_ok and tensor_kind in {"integral", "floating"}
            probe_bit_exact = probe_bit_exact and _bool(
                leaf.get("byte_exact"),
                f"correctness.probes.{probe_name}.same_dtype.leaves[{index}].byte_exact",
            )
            probe_layout = probe_layout and _bool(
                leaf.get("layout_match"),
                f"correctness.probes.{probe_name}.same_dtype.leaves[{index}].layout_match",
            )
        structure_ok = bool(
            structure_ok
            and not errors
            and leaf_count == len(leaves) == len(expected_paths)
            and len(paths) == len(set(paths))
            and set(paths) == expected_paths
            and leaf_kinds_ok
        )
        bit_exact_ok = bit_exact_ok and probe_bit_exact and bool(leaves)
        layout_ok = layout_ok and probe_layout and bool(leaves)
        anchor = _mapping(
            probe.get("fp32_anchor"),
            f"correctness.probes.{probe_name}.fp32_anchor",
        )
        anchor_ok = bool(
            anchor_ok
            and anchor.get("kind") == "dequantized_vs_fp32"
            and _bool(
                anchor.get("passed"),
                f"correctness.probes.{probe_name}.fp32_anchor.passed",
            )
        )
    return {
        "quantized_output_structure": structure_ok,
        "quantized_output_bit_exact": bit_exact_ok,
        "quantized_output_layout": layout_ok,
        "quantized_dequantized_anchor": anchor_ok,
    }


def _derive_output_contract_gates(
    correctness: Mapping[str, Any],
) -> dict[str, bool]:
    """Recompute output residency and stride/layout equality for every probe."""

    probes = _mapping(correctness.get("probes"), "correctness.probes")
    if not probes:
        raise ValidationError("timed correctness probes must not be empty")
    layout_ok = True
    residency_ok = True
    for probe_name, raw_probe in probes.items():
        probe = _mapping(raw_probe, f"correctness.probes.{probe_name}")
        comparison = _mapping(
            probe.get("same_dtype"),
            f"correctness.probes.{probe_name}.same_dtype",
        )
        leaves = _sequence(
            comparison.get("leaves"),
            f"correctness.probes.{probe_name}.same_dtype.leaves",
        )
        if not leaves:
            raise ValidationError("timed comparison must contain tensor leaves")
        per_leaf_layout: list[bool] = []
        per_leaf_residency: list[bool] = []
        paths: list[str] = []
        for index, raw_leaf in enumerate(leaves):
            label = f"correctness.probes.{probe_name}.same_dtype.leaves[{index}]"
            leaf = _mapping(raw_leaf, label)
            paths.append(_string(leaf.get("path"), f"{label}.path", maximum=512))
            candidate_strides = [
                _integer(value, f"{label}.candidate_strides[{position}]")
                for position, value in enumerate(
                    _sequence(
                        leaf.get("candidate_strides"), f"{label}.candidate_strides"
                    )
                )
            ]
            reference_strides = [
                _integer(value, f"{label}.reference_strides[{position}]")
                for position, value in enumerate(
                    _sequence(
                        leaf.get("reference_strides"), f"{label}.reference_strides"
                    )
                )
            ]
            leaf_layout = candidate_strides == reference_strides
            if _bool(leaf.get("layout_match"), f"{label}.layout_match") != leaf_layout:
                raise ValidationError("output leaf layout_match is contradictory")
            candidate_device = _string(
                leaf.get("candidate_device"), f"{label}.candidate_device", maximum=128
            )
            reference_device = _string(
                leaf.get("reference_device"), f"{label}.reference_device", maximum=128
            )
            leaf_residency = candidate_device == reference_device
            if (
                _bool(leaf.get("device_match"), f"{label}.device_match")
                != leaf_residency
            ):
                raise ValidationError("output leaf device_match is contradictory")
            per_leaf_layout.append(leaf_layout)
            per_leaf_residency.append(leaf_residency)
        if len(paths) != len(set(paths)):
            raise ValidationError("output comparison leaf paths must be unique")
        probe_layout = all(per_leaf_layout)
        probe_residency = all(per_leaf_residency)
        if (
            _bool(
                comparison.get("layout_match"),
                f"correctness.probes.{probe_name}.same_dtype.layout_match",
            )
            != probe_layout
        ):
            raise ValidationError("output comparison layout_match contradicts leaves")
        if (
            _bool(
                comparison.get("device_match"),
                f"correctness.probes.{probe_name}.same_dtype.device_match",
            )
            != probe_residency
        ):
            raise ValidationError("output comparison device_match contradicts leaves")
        layout_ok = layout_ok and probe_layout
        residency_ok = residency_ok and probe_residency
    return {"layout": layout_ok, "residency": residency_ok}


def _derive_bare_comparison_gate(
    raw_comparison: Any, gates: Mapping[str, Any]
) -> bool:
    """Recompute a bare-operation comparison instead of trusting its booleans."""

    comparison = _mapping(raw_comparison, "correctness.bare_operation.comparison")
    _exact_keys(
        comparison,
        (
            "passed",
            "leaf_count",
            "shape_match",
            "dtype_match",
            "device_match",
            "byte_exact",
            "layout_match",
            "nonfinite_count",
            "max_abs",
            "max_rel",
            "leaves",
            "errors",
        ),
        (),
        "correctness.bare_operation.comparison",
    )
    errors = _sequence(
        comparison["errors"], "correctness.bare_operation.comparison.errors"
    )
    for index, error in enumerate(errors):
        _string(
            error,
            f"correctness.bare_operation.comparison.errors[{index}]",
            maximum=4096,
        )
    leaves = _sequence(
        comparison["leaves"], "correctness.bare_operation.comparison.leaves"
    )
    claimed_leaf_count = _integer(
        comparison["leaf_count"],
        "correctness.bare_operation.comparison.leaf_count",
    )
    if claimed_leaf_count != len(leaves) or not leaves:
        raise ValidationError("bare comparison leaf_count contradicts leaves")
    leaf_shape: list[bool] = []
    leaf_dtype: list[bool] = []
    leaf_residency: list[bool] = []
    leaf_layout: list[bool] = []
    leaf_bytes: list[bool] = []
    leaf_passed: list[bool] = []
    nonfinite_values: list[int] = []
    absolute_values: list[float] = []
    relative_values: list[float] = []
    paths: list[str] = []
    leaf_keys = (
        "path",
        "candidate_shape",
        "reference_shape",
        "candidate_dtype",
        "reference_dtype",
        "candidate_device",
        "reference_device",
        "candidate_strides",
        "reference_strides",
        "tensor_kind",
        "shape_match",
        "dtype_match",
        "device_match",
        "layout_match",
        "byte_exact",
        "nonfinite_count",
        "max_abs",
        "max_rel",
        "anchor_max_abs",
        "anchor_max_rel",
        "passed",
    )
    for index, raw_leaf in enumerate(leaves):
        label = f"correctness.bare_operation.comparison.leaves[{index}]"
        leaf = _mapping(raw_leaf, label)
        _exact_keys(leaf, leaf_keys, (), label)
        path = _string(leaf["path"], f"{label}.path", maximum=512)
        paths.append(path)
        candidate_shape = [
            _integer(value, f"{label}.candidate_shape[{position}]")
            for position, value in enumerate(
                _sequence(leaf["candidate_shape"], f"{label}.candidate_shape")
            )
        ]
        reference_shape = [
            _integer(value, f"{label}.reference_shape[{position}]")
            for position, value in enumerate(
                _sequence(leaf["reference_shape"], f"{label}.reference_shape")
            )
        ]
        candidate_dtype = _string(
            leaf["candidate_dtype"], f"{label}.candidate_dtype", maximum=128
        )
        reference_dtype = _string(
            leaf["reference_dtype"], f"{label}.reference_dtype", maximum=128
        )
        candidate_device = _string(
            leaf["candidate_device"], f"{label}.candidate_device", maximum=128
        )
        reference_device = _string(
            leaf["reference_device"], f"{label}.reference_device", maximum=128
        )
        candidate_strides = [
            _integer(value, f"{label}.candidate_strides[{position}]")
            for position, value in enumerate(
                _sequence(leaf["candidate_strides"], f"{label}.candidate_strides")
            )
        ]
        reference_strides = [
            _integer(value, f"{label}.reference_strides[{position}]")
            for position, value in enumerate(
                _sequence(leaf["reference_strides"], f"{label}.reference_strides")
            )
        ]
        _string(leaf["tensor_kind"], f"{label}.tensor_kind", maximum=16)
        shape_match = candidate_shape == reference_shape
        dtype_match = candidate_dtype == reference_dtype
        device_match = candidate_device == reference_device
        layout_match = candidate_strides == reference_strides
        for field, expected in (
            ("shape_match", shape_match),
            ("dtype_match", dtype_match),
            ("device_match", device_match),
            ("layout_match", layout_match),
        ):
            if _bool(leaf[field], f"{label}.{field}") != expected:
                raise ValidationError(f"bare comparison leaf {field} is contradictory")
        byte_exact = _bool(leaf["byte_exact"], f"{label}.byte_exact")
        nonfinite = _integer(leaf["nonfinite_count"], f"{label}.nonfinite_count")
        max_abs = _number(leaf["max_abs"], f"{label}.max_abs")
        max_rel = _number(leaf["max_rel"], f"{label}.max_rel")
        _number(leaf["anchor_max_abs"], f"{label}.anchor_max_abs")
        _number(leaf["anchor_max_rel"], f"{label}.anchor_max_rel")
        expected_leaf_passed = bool(
            shape_match
            and dtype_match
            and device_match
            and layout_match
            and nonfinite == 0
            and max_abs <= float(gates["max_abs"])
            and max_rel <= float(gates["max_rel"])
        )
        if _bool(leaf["passed"], f"{label}.passed") != expected_leaf_passed:
            raise ValidationError("bare comparison leaf passed flag is contradictory")
        leaf_shape.append(shape_match)
        leaf_dtype.append(dtype_match)
        leaf_residency.append(device_match)
        leaf_layout.append(layout_match)
        leaf_bytes.append(byte_exact)
        leaf_passed.append(expected_leaf_passed)
        nonfinite_values.append(nonfinite)
        absolute_values.append(max_abs)
        relative_values.append(max_rel)
    if len(paths) != len(set(paths)):
        raise ValidationError("bare comparison leaf paths must be unique")
    derived_values: dict[str, Any] = {
        "shape_match": all(leaf_shape),
        "dtype_match": all(leaf_dtype),
        "device_match": all(leaf_residency),
        "layout_match": all(leaf_layout),
        "byte_exact": all(leaf_bytes),
        "nonfinite_count": sum(nonfinite_values),
        "max_abs": max(absolute_values),
        "max_rel": max(relative_values),
    }
    for field in (
        "shape_match",
        "dtype_match",
        "device_match",
        "layout_match",
        "byte_exact",
    ):
        if _bool(comparison[field], f"bare comparison.{field}") != derived_values[field]:
            raise ValidationError(f"bare comparison {field} contradicts leaves")
    if _integer(
        comparison["nonfinite_count"], "bare comparison.nonfinite_count"
    ) != derived_values["nonfinite_count"]:
        raise ValidationError("bare comparison nonfinite_count contradicts leaves")
    for field in ("max_abs", "max_rel"):
        claimed = _number(comparison[field], f"bare comparison.{field}")
        if not _same_number(claimed, float(derived_values[field])):
            raise ValidationError(f"bare comparison {field} contradicts leaves")
    derived_passed = bool(not errors and all(leaf_passed))
    if _bool(comparison["passed"], "bare comparison.passed") != derived_passed:
        raise ValidationError("bare comparison passed flag contradicts leaves")
    return derived_passed


def _parse_graph_state_evidence(raw: Any, label: str) -> dict[str, Any]:
    state = _mapping(raw, label)
    _exact_keys(
        state,
        ("build_count", "replay_count", "cache_size", "invariant_ok"),
        (),
        label,
    )
    return {
        "build_count": _integer(state["build_count"], f"{label}.build_count"),
        "replay_count": _integer(state["replay_count"], f"{label}.replay_count"),
        "cache_size": _integer(state["cache_size"], f"{label}.cache_size"),
        "invariant_ok": _bool(state["invariant_ok"], f"{label}.invariant_ok"),
    }


def _is_steady_replay_transition(
    stable: Mapping[str, Any], final: Mapping[str, Any]
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


def _derive_graph_replay(
    lifecycle: Mapping[str, Any], *, evidence_required: bool
) -> bool:
    """Recompute the global Graph gate from exact stable/final evidence."""

    stable_raw = lifecycle.get("stable_state")
    final_raw = lifecycle.get("final_state")
    if stable_raw is None or final_raw is None:
        if evidence_required:
            raise ValidationError(
                "timed workload requires stable_state and final_state Graph evidence"
            )
        if stable_raw is not None:
            _parse_graph_state_evidence(
                stable_raw, "lifecycle.stable_state"
            )
        if final_raw is not None:
            _parse_graph_state_evidence(final_raw, "lifecycle.final_state")
        if "graph_state_present" in lifecycle:
            _bool(
                lifecycle["graph_state_present"], "lifecycle.graph_state_present"
            )
        return False

    stable = _parse_graph_state_evidence(
        stable_raw, "lifecycle.stable_state"
    )
    final = _parse_graph_state_evidence(final_raw, "lifecycle.final_state")
    if "graph_state_present" in lifecycle and not _bool(
        lifecycle["graph_state_present"], "lifecycle.graph_state_present"
    ):
        raise ValidationError(
            "graph_state_present contradicts Graph lifecycle evidence"
        )

    claimed_build = _integer(
        lifecycle["graph_build_count"], "lifecycle.graph_build_count"
    )
    claimed_replay = _integer(
        lifecycle["graph_replay_count"], "lifecycle.graph_replay_count"
    )
    claimed_cache = _bool(
        lifecycle["cache_size_stable"], "lifecycle.cache_size_stable"
    )
    claimed_invariant = _bool(
        lifecycle["cache_invariant_ok"], "lifecycle.cache_invariant_ok"
    )
    derived_cache = final["cache_size"] == stable["cache_size"]
    derived_invariant = bool(stable["invariant_ok"] and final["invariant_ok"])
    if claimed_build != final["build_count"]:
        raise ValidationError("graph_build_count contradicts final_state")
    if claimed_replay != final["replay_count"]:
        raise ValidationError("graph_replay_count contradicts final_state")
    if claimed_cache != derived_cache:
        raise ValidationError("cache_size_stable contradicts Graph states")
    if claimed_invariant != derived_invariant:
        raise ValidationError("cache_invariant_ok contradicts Graph states")
    return _is_steady_replay_transition(stable, final)


def _derive_paired_arm_replay(lifecycle: Mapping[str, Any]) -> bool:
    """Validate independent candidate/bare BUILD-to-REPLAY transitions."""

    arm_states = _mapping(lifecycle.get("arm_states"), "lifecycle.arm_states")
    _exact_keys(arm_states, ("stable", "final"), (), "lifecycle.arm_states")

    def parse_phase(phase: str) -> dict[str, dict[str, Any]]:
        raw_phase = _mapping(
            arm_states[phase], f"lifecycle.arm_states.{phase}"
        )
        _exact_keys(
            raw_phase,
            ("candidate", "bare_operation"),
            (),
            f"lifecycle.arm_states.{phase}",
        )
        result: dict[str, dict[str, Any]] = {}
        for arm in ("candidate", "bare_operation"):
            result[arm] = _parse_graph_state_evidence(
                raw_phase[arm], f"lifecycle.arm_states.{phase}.{arm}"
            )
        return result

    stable = parse_phase("stable")
    final = parse_phase("final")
    derived = all(
        _is_steady_replay_transition(stable[arm], final[arm])
        for arm in ("candidate", "bare_operation")
    )
    claimed = _bool(
        lifecycle.get("paired_steady_replay"), "lifecycle.paired_steady_replay"
    )
    if claimed != derived:
        raise ValidationError("paired_steady_replay contradicts arm lifecycle evidence")
    return derived


def _samples(value: Any, label: str, minimum_count: int) -> list[float]:
    sequence = _sequence(value, label)
    if len(sequence) < minimum_count:
        raise ValidationError(
            f"{label} must contain at least {minimum_count} samples"
        )
    return [_number(sample, f"{label}[{index}]", positive=True) for index, sample in enumerate(sequence)]


def _same_number(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-6)


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def paired_median_bootstrap_ci(
    candidate_samples: Sequence[float],
    bare_samples: Sequence[float],
    *,
    confidence: float,
    trials: int,
    seed: int,
) -> tuple[float, float]:
    """Deterministically bootstrap the median paired epilogue tax percentage."""

    if len(candidate_samples) != len(bare_samples) or not candidate_samples:
        raise ValidationError("paired bootstrap requires equal non-empty samples")
    if not 0 < confidence < 1 or trials < 1000 or seed < 0:
        raise ValidationError("paired bootstrap parameters are outside the contract")
    paired = [
        (candidate - bare) * 100.0 / bare
        for candidate, bare in zip(candidate_samples, bare_samples)
    ]
    count = len(paired)
    rng = random.Random(seed)
    bootstrapped = [
        float(statistics.median(paired[rng.randrange(count)] for _ in range(count)))
        for _ in range(trials)
    ]
    alpha = (1.0 - confidence) / 2.0
    return (
        _percentile(bootstrapped, alpha),
        _percentile(bootstrapped, 1.0 - alpha),
    )


def _sample_statistics(values: Sequence[float]) -> dict[str, float | int]:
    p50 = float(statistics.median(values))
    mean = float(statistics.fmean(values))
    return {
        "p50_ns": p50,
        "p95_ns": _percentile(values, 0.95),
        "mad_ns": float(statistics.median(abs(value - p50) for value in values)),
        "cv": float(statistics.pstdev(values) / mean),
        "sample_count": len(values),
    }


def _validate_workload_timing(
    raw_samples_value: Any,
    summary_value: Any,
    minimum_count: int,
    *,
    fusion_gates: Mapping[str, Any] | None = None,
    max_cv: float | None = None,
) -> tuple[bool, float | None, bool | None, bool | None]:
    raw_samples = _mapping(raw_samples_value, "raw_samples")
    _exact_keys(
        raw_samples,
        ("candidate_ns", "reference_ns"),
        ("bare_operation_ns",),
        "raw_samples",
    )
    candidate_raw = _sequence(raw_samples["candidate_ns"], "candidate_ns")
    reference_raw = _sequence(raw_samples["reference_ns"], "reference_ns")
    bare_raw = (
        _sequence(raw_samples["bare_operation_ns"], "bare_operation_ns")
        if "bare_operation_ns" in raw_samples
        else None
    )
    timing_available = bool(candidate_raw or reference_raw or bare_raw)
    summary = _mapping(summary_value, "summary")
    if not timing_available:
        if candidate_raw or reference_raw or (bare_raw not in (None, [])):
            raise ValidationError("failed timing receipt must use paired empty samples")
        _exact_keys(
            summary,
            ("candidate_p50_ns", "reference_p50_ns"),
            ("timing_error",),
            "summary",
        )
        if summary["candidate_p50_ns"] is not None or summary["reference_p50_ns"] is not None:
            raise ValidationError("failed timing receipt must use null latency summaries")
        if "timing_error" in summary:
            _string(summary["timing_error"], "summary.timing_error", maximum=4096)
        return False, None, None, None

    candidate_samples = _samples(candidate_raw, "candidate_ns", minimum_count)
    reference_samples = _samples(reference_raw, "reference_ns", minimum_count)
    if len(candidate_samples) != len(reference_samples):
        raise ValidationError("candidate and reference sample counts must match")
    bare_samples: list[float] | None = None
    if bare_raw is not None:
        bare_samples = _samples(bare_raw, "bare_operation_ns", minimum_count)
        if len(bare_samples) != len(candidate_samples):
            raise ValidationError("bare operation and candidate sample counts must match")
    if fusion_gates is not None and bare_samples is None:
        raise ValidationError(
            "timed fusion workload requires paired bare-operation samples"
        )

    candidate_stats = _sample_statistics(candidate_samples)
    reference_stats = _sample_statistics(reference_samples)
    bare_stats = _sample_statistics(bare_samples) if bare_samples else None
    candidate_p50 = float(candidate_stats["p50_ns"])
    reference_p50 = float(reference_stats["p50_ns"])
    statistic_suffixes = ("p50_ns", "p95_ns", "mad_ns", "cv", "sample_count")
    optional_summary = [
        f"{prefix}_{suffix}"
        for prefix in ("candidate", "reference")
        for suffix in statistic_suffixes
        if suffix != "p50_ns"
    ]
    optional_summary.append("timing_order_blocks")
    fusion_tax_ok: bool | None = None
    if bare_samples is not None:
        optional_summary.extend(
            f"bare_operation_{suffix}" for suffix in statistic_suffixes
        )
        optional_summary.extend(
            (
                "epilogue_tax_p50_ns",
                "epilogue_tax_p95_ns",
                "epilogue_tax_p50_pct",
                "epilogue_tax_ci_lower_pct",
                "epilogue_tax_ci_upper_pct",
                "epilogue_tax_confidence",
                "epilogue_tax_bootstrap_trials",
                "epilogue_tax_bootstrap_seed",
            )
        )
    _exact_keys(
        summary,
        ("candidate_p50_ns", "reference_p50_ns"),
        optional_summary,
        "summary",
    )
    claimed_candidate = _number(
        summary["candidate_p50_ns"], "summary.candidate_p50_ns", positive=True
    )
    claimed_reference = _number(
        summary["reference_p50_ns"], "summary.reference_p50_ns", positive=True
    )
    if not _same_number(claimed_candidate, candidate_p50):
        raise ValidationError("summary.candidate_p50_ns does not match raw samples")
    if not _same_number(claimed_reference, reference_p50):
        raise ValidationError("summary.reference_p50_ns does not match raw samples")

    for prefix, expected in (("candidate", candidate_stats), ("reference", reference_stats)):
        for suffix in statistic_suffixes[1:]:
            key = f"{prefix}_{suffix}"
            if key not in summary:
                continue
            if suffix == "sample_count":
                claimed_count = _integer(summary[key], f"summary.{key}", minimum=1)
                if claimed_count != expected[suffix]:
                    raise ValidationError(f"summary.{key} does not match raw samples")
            else:
                claimed = _number(summary[key], f"summary.{key}")
                if not _same_number(claimed, float(expected[suffix])):
                    raise ValidationError(f"summary.{key} does not match raw samples")

    if "timing_order_blocks" in summary:
        order_blocks = _sequence(
            summary["timing_order_blocks"], "summary.timing_order_blocks"
        )
        if not order_blocks:
            raise ValidationError("summary.timing_order_blocks must not be empty")
        for index, block in enumerate(order_blocks):
            _string(block, f"summary.timing_order_blocks[{index}]", maximum=256)

    if bare_samples is not None:
        if "bare_operation_p50_ns" not in summary:
            raise ValidationError("bare_operation summary is missing")
        assert bare_stats is not None
        bare_p50 = float(bare_stats["p50_ns"])
        claimed_bare = _number(
            summary["bare_operation_p50_ns"],
            "summary.bare_operation_p50_ns",
            positive=True,
        )
        if not _same_number(claimed_bare, bare_p50):
            raise ValidationError(
                "summary.bare_operation_p50_ns does not match raw samples"
            )
        for suffix in statistic_suffixes[1:]:
            key = f"bare_operation_{suffix}"
            if key not in summary:
                continue
            if suffix == "sample_count":
                claimed_count = _integer(summary[key], f"summary.{key}", minimum=1)
                if claimed_count != bare_stats[suffix]:
                    raise ValidationError(f"summary.{key} does not match raw samples")
            else:
                claimed = _number(summary[key], f"summary.{key}")
                if not _same_number(claimed, float(bare_stats[suffix])):
                    raise ValidationError(f"summary.{key} does not match raw samples")

        paired_delta = [
            candidate - bare for candidate, bare in zip(candidate_samples, bare_samples)
        ]
        paired_percentage = [
            delta * 100.0 / bare for delta, bare in zip(paired_delta, bare_samples)
        ]
        expected_taxes = {
            "epilogue_tax_p50_ns": float(statistics.median(paired_delta)),
            "epilogue_tax_p95_ns": _percentile(paired_delta, 0.95),
            "epilogue_tax_p50_pct": float(statistics.median(paired_percentage)),
        }
        for key, expected in expected_taxes.items():
            if key in summary:
                claimed = _finite_number(summary[key], f"summary.{key}")
                if not _same_number(claimed, expected):
                    raise ValidationError(f"summary.{key} does not match raw samples")
        if fusion_gates is not None:
            required_ci = (
                "epilogue_tax_p50_pct",
                "epilogue_tax_ci_lower_pct",
                "epilogue_tax_ci_upper_pct",
                "epilogue_tax_confidence",
                "epilogue_tax_bootstrap_trials",
                "epilogue_tax_bootstrap_seed",
            )
            missing_ci = [key for key in required_ci if key not in summary]
            if missing_ci:
                raise ValidationError(
                    "fusion summary is missing paired CI fields: "
                    + ", ".join(missing_ci)
                )
            confidence = float(fusion_gates["fusion_confidence"])
            trials = int(fusion_gates["fusion_bootstrap_trials"])
            seed = int(fusion_gates["fusion_bootstrap_seed"])
            expected_lower, expected_upper = paired_median_bootstrap_ci(
                candidate_samples,
                bare_samples,
                confidence=confidence,
                trials=trials,
                seed=seed,
            )
            claimed_lower = _finite_number(
                summary["epilogue_tax_ci_lower_pct"],
                "summary.epilogue_tax_ci_lower_pct",
            )
            claimed_upper = _finite_number(
                summary["epilogue_tax_ci_upper_pct"],
                "summary.epilogue_tax_ci_upper_pct",
            )
            if not _same_number(claimed_lower, expected_lower) or not _same_number(
                claimed_upper, expected_upper
            ):
                raise ValidationError("fusion paired CI does not match raw samples")
            if not _same_number(
                _number(
                    summary["epilogue_tax_confidence"],
                    "summary.epilogue_tax_confidence",
                ),
                confidence,
            ):
                raise ValidationError("fusion confidence does not match the contract")
            if _integer(
                summary["epilogue_tax_bootstrap_trials"],
                "summary.epilogue_tax_bootstrap_trials",
                minimum=1000,
            ) != trials:
                raise ValidationError("fusion bootstrap trials do not match the contract")
            if _integer(
                summary["epilogue_tax_bootstrap_seed"],
                "summary.epilogue_tax_bootstrap_seed",
            ) != seed:
                raise ValidationError("fusion bootstrap seed does not match the contract")
            fusion_tax_ok = bool(
                claimed_upper <= float(fusion_gates["max_epilogue_tax_pct"])
            )
    stability_ok = None
    if max_cv is not None:
        stability_ok = bool(
            float(candidate_stats["cv"]) <= max_cv
            and float(reference_stats["cv"]) <= max_cv
            and (bare_stats is None or float(bare_stats["cv"]) <= max_cv)
        )
    return True, candidate_p50, fusion_tax_ok, stability_ok


def validate_result(
    raw_result: Any,
    contract: Mapping[str, Any],
    expected_contract_sha256: str,
) -> dict[str, Any]:
    """Validate a raw iteration and return its canonical journal record."""

    result = _mapping(raw_result, "result")
    _exact_keys(
        result,
        (
            "schema_version",
            "candidate_id",
            "candidate_commit",
            "parent_commit",
            "contract_sha256",
            "preflight_sha256",
            "process",
            "verdict",
            "metric",
            "workloads",
            "hard_gates",
        ),
        ("artifacts",),
        "result",
    )
    if type(result["schema_version"]) is not int or result["schema_version"] != 1:
        raise ValidationError("result.schema_version must be integer 1")
    candidate_id = _safe_id(result["candidate_id"], "result.candidate_id")
    candidate_commit = _string(result["candidate_commit"], "candidate_commit")
    parent_commit = _string(result["parent_commit"], "parent_commit")
    if COMMIT_RE.fullmatch(candidate_commit) is None:
        raise ValidationError("candidate_commit must be a full lowercase commit")
    if COMMIT_RE.fullmatch(parent_commit) is None:
        raise ValidationError("parent_commit must be a full lowercase commit")
    sha = _string(result["contract_sha256"], "contract_sha256")
    if SHA256_RE.fullmatch(sha) is None or sha != expected_contract_sha256:
        raise ValidationError("contract_sha256 does not match the exact contract")
    preflight_sha = _string(result["preflight_sha256"], "preflight_sha256")
    if SHA256_RE.fullmatch(preflight_sha) is None:
        raise ValidationError("preflight_sha256 must be lowercase SHA-256")
    process = _mapping(result["process"], "process")
    _exact_keys(
        process,
        ("boot_id", "pid", "start_ticks", "run_uuid", "attested"),
        (),
        "process",
    )
    process_identity = {
        "boot_id": _canonical_uuid(process["boot_id"], "process.boot_id"),
        "pid": _integer(process["pid"], "process.pid", minimum=1),
        "start_ticks": _integer(
            process["start_ticks"], "process.start_ticks", minimum=1
        ),
        "run_uuid": _canonical_uuid(
            process["run_uuid"], "process.run_uuid", version=4
        ),
        "attested": _bool(process["attested"], "process.attested"),
    }
    verdict = result["verdict"]
    if verdict not in ("full", "signal"):
        raise ValidationError("verdict must be exactly 'full' or 'signal'")

    metric = _mapping(result["metric"], "metric")
    _exact_keys(metric, ("name", "direction", "value"), ("aggregation",), "metric")
    if metric["name"] != contract["measurement"]["metric"]:
        raise ValidationError("metric.name does not match the contract")
    if metric["direction"] != contract["measurement"]["direction"]:
        raise ValidationError("metric.direction does not match the contract")
    if "aggregation" in metric and metric["aggregation"] != "median_workload_candidate_p50":
        raise ValidationError("metric.aggregation is unsupported")
    metric_value = (
        None
        if metric["value"] is None
        else _number(metric["value"], "metric.value", positive=True)
    )

    raw_workloads = _sequence(result["workloads"], "result.workloads")
    if not raw_workloads:
        raise ValidationError("result.workloads must not be empty")
    contract_ids = [workload["id"] for workload in contract["workloads"]]
    seen_ids: list[str] = []
    canonical_workloads: list[dict[str, Any]] = []
    candidate_medians_ns: list[float] = []
    timing_available_flags: list[bool] = []
    operation = str(contract.get("campaign", {}).get("op", ""))
    fusion_gates = contract["gates"] if operation in FUSION_OPS else None
    for index, raw_workload in enumerate(raw_workloads):
        workload = _mapping(raw_workload, f"result.workloads[{index}]")
        _exact_keys(
            workload,
            (
                "id",
                "raw_samples",
                "summary",
                "correctness",
                "lifecycle",
                "hard_gates",
                "eligible",
            ),
            ("execution",),
            f"result.workloads[{index}]",
        )
        workload_id = _safe_id(workload["id"], f"result.workloads[{index}].id")
        if workload_id not in contract_ids:
            raise ValidationError(f"unknown workload id: {workload_id}")
        if workload_id in seen_ids:
            raise ValidationError(f"duplicate workload id: {workload_id}")
        seen_ids.append(workload_id)

        count = (
            SIGNAL_MINIMUM_SAMPLES
            if verdict == "signal"
            else int(contract["measurement"]["samples"])
        )
        timing_available, candidate_p50, fusion_tax_ok, stability_ok = _validate_workload_timing(
            workload["raw_samples"],
            workload["summary"],
            count,
            fusion_gates=fusion_gates,
            max_cv=(
                float(contract["gates"]["max_cv"])
                if "max_cv" in contract["gates"]
                else None
            ),
        )
        timing_available_flags.append(timing_available)

        execution = workload.get("execution")
        execution_ok: bool | None = None
        if execution is not None:
            execution_map = _mapping(execution, "execution")
            _exact_keys(
                execution_map,
                (
                    "kernel_names",
                    "launch_count",
                    "single_device_program",
                    "trace_summary_sha256",
                    "preflight_sha256",
                ),
                (),
                "execution",
            )
            raw_kernel_names = _sequence(
                execution_map["kernel_names"], "execution.kernel_names"
            )
            kernel_names = [
                _string(name, "execution.kernel_names[]", maximum=255)
                for name in raw_kernel_names
            ]
            if not kernel_names or len(kernel_names) != len(set(kernel_names)):
                raise ValidationError("execution.kernel_names must be non-empty and unique")
            if any(KERNEL_NAME_RE.fullmatch(name) is None for name in kernel_names):
                raise ValidationError("execution.kernel_names contains an invalid name")
            launch_count = _integer(
                execution_map["launch_count"], "execution.launch_count", minimum=1
            )
            single_program = _bool(
                execution_map["single_device_program"],
                "execution.single_device_program",
            )
            trace_hash = _string(
                execution_map["trace_summary_sha256"],
                "execution.trace_summary_sha256",
            )
            execution_preflight = _string(
                execution_map["preflight_sha256"], "execution.preflight_sha256"
            )
            if SHA256_RE.fullmatch(trace_hash) is None or SHA256_RE.fullmatch(
                execution_preflight
            ) is None:
                raise ValidationError("execution hashes must be lowercase SHA-256")
            if execution_preflight != preflight_sha:
                raise ValidationError("execution preflight hash does not match result")
            required_kernel_names = set(
                contract.get("runtime", {}).get("required_kernels", [])
            )
            expected_names = (
                {contract["runtime"]["fused_kernel"]}
                if fusion_gates is not None
                else required_kernel_names
            )
            execution_ok = bool(
                single_program
                and launch_count == 1
                and len(kernel_names) == 1
                and set(kernel_names) <= expected_names
                and trace_hash != ZERO_SHA256
                and execution_preflight != ZERO_SHA256
            )
        if verdict == "full" and timing_available:
            if execution is None or not execution_ok:
                raise ValidationError(
                    "full workload requires attested one-launch device-program evidence"
                )

        correctness = _mapping(workload["correctness"], "correctness")
        _exact_keys(
            correctness,
            (
                "passed",
                "nonfinite_count",
                "max_abs",
                "max_rel",
                "anchor_max_abs",
                "anchor_max_rel",
                "same_dtype",
                "fp32_anchor",
            ),
            (
                "fp32_anchor_present",
                "probes",
                "quantized_output_roles",
                "bare_operation",
                "retained_output",
                "same_pointer_changed_value",
                "different_pointer_same_value",
                "second_input_changed",
                "input_immutable",
                "error",
            ),
            "correctness",
        )
        nonfinite_count = _integer(
            correctness["nonfinite_count"], "correctness.nonfinite_count"
        )
        max_abs = _number(correctness["max_abs"], "correctness.max_abs")
        max_rel = _number(correctness["max_rel"], "correctness.max_rel")
        anchor_max_abs = _number(
            correctness["anchor_max_abs"], "correctness.anchor_max_abs"
        )
        anchor_max_rel = _number(
            correctness["anchor_max_rel"], "correctness.anchor_max_rel"
        )
        same_dtype = _bool(correctness["same_dtype"], "correctness.same_dtype")
        if correctness["fp32_anchor"] is None:
            if timing_available or correctness["passed"] is not False:
                raise ValidationError(
                    "fp32_anchor may be null only for a failed execution receipt"
                )
            fp32_anchor = False
        else:
            fp32_anchor = _bool(
                correctness["fp32_anchor"], "correctness.fp32_anchor"
            )
        fp32_anchor_present = (
            _bool(
                correctness["fp32_anchor_present"],
                "correctness.fp32_anchor_present",
            )
            if "fp32_anchor_present" in correctness
            else fp32_anchor
        )
        correctness_derived = (
            nonfinite_count == 0
            and max_abs <= float(contract["gates"]["max_abs"])
            and max_rel <= float(contract["gates"]["max_rel"])
            and (
                not contract["gates"]["require_same_dtype"] or same_dtype
            )
            and (
                not contract["gates"]["require_fp32_anchor"]
                or (
                    fp32_anchor_present
                    and fp32_anchor
                    and anchor_max_abs
                    <= float(contract["gates"]["anchor_max_abs"])
                    and anchor_max_rel
                    <= float(contract["gates"]["anchor_max_rel"])
                )
            )
        )
        correctness_passed = _bool(correctness["passed"], "correctness.passed")
        if not timing_available and correctness_passed:
            raise ValidationError("failed timing receipt cannot pass correctness")
        if correctness_passed and not correctness_derived:
            raise ValidationError("correctness.passed contradicts measured gates")
        for detail_name in (
            "probes",
            "quantized_output_roles",
            "bare_operation",
            "retained_output",
            "same_pointer_changed_value",
            "different_pointer_same_value",
        ):
            if detail_name in correctness:
                _validate_json_evidence(
                    correctness[detail_name], f"correctness.{detail_name}"
                )
        output_contract_gates: dict[str, bool] | None = None
        if timing_available:
            output_contract_gates = _derive_output_contract_gates(correctness)

        quantized_output_gates: dict[str, bool] | None = None
        if operation == "rmsnorm+quant_mxfp8" and timing_available:
            quantized_output_gates = _derive_quantized_output_gates(correctness)

        bare_operation_ok: bool | None = None
        if operation in BARE_OPERATION_IDENTITIES and timing_available:
            if "bare_operation" not in correctness:
                raise ValidationError(
                    "timed fusion workload lacks bare-operation correctness evidence"
                )
            bare_evidence = _mapping(
                correctness["bare_operation"], "correctness.bare_operation"
            )
            _exact_keys(
                bare_evidence,
                ("identity", "passed", "input_immutable", "comparison"),
                (),
                "correctness.bare_operation",
            )
            if bare_evidence["identity"] != BARE_OPERATION_IDENTITIES[operation]:
                raise ValidationError(
                    "bare-operation identity does not match the exact profile"
                )
            bare_passed = _bool(
                bare_evidence["passed"], "correctness.bare_operation.passed"
            )
            bare_input_immutable = _bool(
                bare_evidence["input_immutable"],
                "correctness.bare_operation.input_immutable",
            )
            bare_comparison = _mapping(
                bare_evidence["comparison"],
                "correctness.bare_operation.comparison",
            )
            comparison_passed = _derive_bare_comparison_gate(
                bare_comparison, contract["gates"]
            )
            bare_operation_ok = bool(
                bare_passed and bare_input_immutable and comparison_passed
            )
            if bare_passed != bare_operation_ok:
                raise ValidationError(
                    "bare-operation passed flag contradicts its evidence"
                )
        if "second_input_changed" in correctness:
            _bool(
                correctness["second_input_changed"],
                "correctness.second_input_changed",
            )
        if "input_immutable" in correctness:
            input_immutable = _bool(
                correctness["input_immutable"], "correctness.input_immutable"
            )
            if correctness_passed and not input_immutable:
                raise ValidationError("passing correctness requires immutable inputs")
        if "error" in correctness:
            _string(correctness["error"], "correctness.error", maximum=4096)
            if correctness_passed:
                raise ValidationError("passing correctness evidence cannot contain error")

        lifecycle = _mapping(workload["lifecycle"], "lifecycle")
        _exact_keys(
            lifecycle,
            (
                "passed",
                "graph_build_count",
                "graph_replay_count",
                "cache_size_stable",
                "cache_invariant_ok",
                "independent_outputs",
            ),
            (
                "graph_state_present",
                "stable_state",
                "final_state",
                "paired_steady_replay",
                "arm_states",
                "error",
            ),
            "lifecycle",
        )
        build_count = _integer(
            lifecycle["graph_build_count"], "lifecycle.graph_build_count"
        )
        replay_count = _integer(
            lifecycle["graph_replay_count"], "lifecycle.graph_replay_count"
        )
        cache_stable = _bool(
            lifecycle["cache_size_stable"], "lifecycle.cache_size_stable"
        )
        cache_invariant = _bool(
            lifecycle["cache_invariant_ok"], "lifecycle.cache_invariant_ok"
        )
        independent_outputs = _bool(
            lifecycle["independent_outputs"], "lifecycle.independent_outputs"
        )
        graph_ok = _derive_graph_replay(
            lifecycle,
            evidence_required=timing_available,
        )
        paired_arm_replay_ok: bool | None = None
        if operation in FUSION_OPS and timing_available:
            paired_arm_replay_ok = _derive_paired_arm_replay(lifecycle)
        graph_required_for_lifecycle = bool(
            timing_available or contract["gates"]["require_graph_replay"]
        )
        lifecycle_derived = (
            (not graph_required_for_lifecycle or graph_ok)
            and (
                not contract["gates"]["require_independent_outputs"]
                or independent_outputs
            )
            and (paired_arm_replay_ok is None or paired_arm_replay_ok)
        )
        lifecycle_passed = _bool(lifecycle["passed"], "lifecycle.passed")
        if not timing_available and lifecycle_passed:
            raise ValidationError("failed timing receipt cannot pass lifecycle")
        if lifecycle_passed and not lifecycle_derived:
            raise ValidationError("lifecycle.passed contradicts measured gates")
        if "graph_state_present" in lifecycle:
            graph_state_present = _bool(
                lifecycle["graph_state_present"], "lifecycle.graph_state_present"
            )
            if contract["gates"]["require_graph_replay"] and not graph_state_present:
                raise ValidationError("required Graph lifecycle state is absent")
        if "error" in lifecycle:
            _string(lifecycle["error"], "lifecycle.error", maximum=4096)
            if lifecycle_passed:
                raise ValidationError("passing lifecycle evidence cannot contain error")

        core_gates = {
            "finite": nonfinite_count == 0,
            "correctness": correctness_passed and correctness_derived,
            "lifecycle": lifecycle_passed and lifecycle_derived,
            "same_dtype": (
                not contract["gates"]["require_same_dtype"] or same_dtype
            ),
            "fp32_anchor": (
                not contract["gates"]["require_fp32_anchor"]
                or (
                    fp32_anchor_present
                    and fp32_anchor
                    and anchor_max_abs
                    <= float(contract["gates"]["anchor_max_abs"])
                    and anchor_max_rel
                    <= float(contract["gates"]["anchor_max_rel"])
                )
            ),
            "graph_replay": (
                (
                    not timing_available
                    and not contract["gates"]["require_graph_replay"]
                )
                or graph_ok
            ),
            "independent_outputs": (
                not contract["gates"]["require_independent_outputs"]
                or independent_outputs
            ),
        }
        if fusion_tax_ok is not None:
            core_gates["epilogue_tax"] = fusion_tax_ok
        if bare_operation_ok is not None:
            core_gates["bare_operation_correctness"] = bare_operation_ok
        if output_contract_gates is not None:
            core_gates.update(output_contract_gates)
        if quantized_output_gates is not None:
            core_gates.update(quantized_output_gates)
        if execution_ok is not None:
            core_gates["single_device_program"] = execution_ok
        if stability_ok is not None:
            core_gates["stability"] = stability_ok
        if paired_arm_replay_ok is not None:
            core_gates["paired_steady_replay"] = paired_arm_replay_ok
        claimed_gates = _strict_bool_map(workload["hard_gates"], "hard_gates")
        if timing_available:
            if "finite" not in claimed_gates:
                raise ValidationError("timed workload hard_gates must include finite")
            if claimed_gates["finite"] != core_gates["finite"]:
                raise ValidationError("workload finite gate contradicts measured evidence")
            for derived_name in (
                "epilogue_tax",
                "bare_operation_correctness",
                "layout",
                "residency",
                "paired_steady_replay",
                "quantized_output_structure",
                "quantized_output_bit_exact",
                "quantized_output_layout",
                "quantized_dequantized_anchor",
                "single_device_program",
                "stability",
            ):
                if derived_name in core_gates:
                    if derived_name not in claimed_gates:
                        raise ValidationError(
                            f"workload hard_gates must include {derived_name}"
                        )
                    if claimed_gates[derived_name] != core_gates[derived_name]:
                        raise ValidationError(
                            f"workload {derived_name} gate contradicts measured evidence"
                        )
            expected_eligible = all(core_gates.values()) and all(claimed_gates.values())
        else:
            if all(claimed_gates.values()):
                raise ValidationError("failed timing receipt needs an explicit failed gate")
            expected_eligible = False
        if _bool(workload["eligible"], "eligible") != expected_eligible:
            raise ValidationError("workload eligible contradicts hard_gates")

        if candidate_p50 is not None:
            candidate_medians_ns.append(candidate_p50)
        canonical_workloads.append(json.loads(canonical_json(workload)))

    expected_order = [item for item in contract_ids if item in seen_ids]
    if seen_ids != expected_order:
        raise ValidationError("result workloads must preserve contract order")
    if verdict == "full" and seen_ids != contract_ids:
        raise ValidationError("full verdict must include every exact workload")

    all_workloads_eligible = all(
        bool(workload["eligible"]) for workload in canonical_workloads
    )
    metric_complete = (
        all_workloads_eligible
        and len(candidate_medians_ns) == len(canonical_workloads)
        and all(timing_available_flags)
    )
    expected_metric = (
        float(statistics.median(candidate_medians_ns)) / 1000.0
        if metric_complete
        else None
    )
    if expected_metric is None:
        if metric_value is not None:
            raise ValidationError(
                "incomplete or failed workload set must use null metric.value"
            )
    elif metric_value is None or not _same_number(metric_value, expected_metric):
        raise ValidationError("metric.value must equal median workload candidate p50 in us")
    if verdict == "full" and metric_complete:
        if not process_identity["attested"]:
            raise ValidationError(
                "eligible full receipt requires verifier-observed process attestation"
            )
        if preflight_sha == ZERO_SHA256:
            raise ValidationError("eligible full receipt requires a bound preflight receipt")

    claimed_top_gates = _strict_bool_map(result["hard_gates"], "result.hard_gates")
    known_top_gates = {
        "workloads_present": bool(canonical_workloads),
        "all_workloads_eligible": all_workloads_eligible,
        "metric_finite": expected_metric is not None,
    }
    for gate_name, expected in known_top_gates.items():
        if gate_name in claimed_top_gates and claimed_top_gates[gate_name] != expected:
            raise ValidationError(
                f"top-level {gate_name} gate contradicts workload evidence"
            )

    artifacts = result.get("artifacts")
    if artifacts is not None:
        artifact_map = _mapping(artifacts, "artifacts")
        for key, value in artifact_map.items():
            if SAFE_TOKEN_RE.fullmatch(_string(key, "artifact key", maximum=128)) is None:
                raise ValidationError("artifact key is invalid")
            _string(value, f"artifacts.{key}", maximum=4096)

    canonical = json.loads(canonical_json(result))
    canonical["candidate_id"] = candidate_id
    canonical["candidate_commit"] = candidate_commit
    canonical["parent_commit"] = parent_commit
    canonical["preflight_sha256"] = preflight_sha
    canonical["process"] = process_identity
    canonical["metric"]["value"] = expected_metric
    canonical["workloads"] = canonical_workloads
    return canonical


def record_is_promotable(
    record: Mapping[str, Any], metric_name: str, direction: str, contract_hash: str
) -> bool:
    """Return True only for a fully evaluated, all-hard-gates record."""

    try:
        if record.get("schema_version") != SCHEMA_VERSION:
            return False
        if record.get("verdict") != "full":
            return False
        if record.get("contract_sha256") != contract_hash:
            return False
        preflight_hash = str(record.get("preflight_sha256", ""))
        if SHA256_RE.fullmatch(preflight_hash) is None or preflight_hash == ZERO_SHA256:
            return False
        if COMMIT_RE.fullmatch(str(record.get("candidate_commit", ""))) is None:
            return False
        process = _mapping(record.get("process"), "process")
        if process.get("attested") is not True:
            return False
        _canonical_uuid(process.get("boot_id"), "process.boot_id")
        _canonical_uuid(process.get("run_uuid"), "process.run_uuid", version=4)
        _integer(process.get("pid"), "process.pid", minimum=1)
        _integer(process.get("start_ticks"), "process.start_ticks", minimum=1)
        metric = _mapping(record.get("metric"), "metric")
        if metric.get("name") != metric_name or metric.get("direction") != direction:
            return False
        _number(metric.get("value"), "metric.value", positive=True)
        top_gates = _strict_bool_map(record.get("hard_gates"), "hard_gates")
        if not all(top_gates.values()):
            return False
        workloads = _sequence(record.get("workloads"), "workloads")
        if not workloads:
            return False
        for workload in workloads:
            workload_map = _mapping(workload, "workload")
            if workload_map.get("eligible") is not True:
                return False
            if not all(
                _strict_bool_map(workload_map.get("hard_gates"), "hard_gates").values()
            ):
                return False
    except ValidationError:
        return False
    return True
