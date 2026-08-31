#!/usr/bin/env python3
"""Fail-closed preflight for an approved RhinoForge kernel campaign."""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from campaign_common import (
    KERNEL_NAME_RE,
    SHA256_RE,
    ValidationError,
    content_tree_sha256,
    contract_sha256,
    load_contract,
)


MANIFEST_HEADER = "rhinoforge-kernels-v1"
MAX_MANIFEST_BYTES = 1024 * 1024
RPU_DEVICE_NAME_RE = re.compile(r"rpu[0-9]+")
ZERO_SHA256 = "0" * 64


def _resolve_contract_path(contract_path: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = contract_path.parent / path
    return path.resolve(strict=False)


def _find_rhinoforge_root() -> Path | None:
    script = Path(__file__).resolve()
    for candidate in script.parents:
        if _is_rhinoforge_root(candidate):
            return candidate
    return None


def _is_rhinoforge_root(root: Path) -> bool:
    """Accept only the top level of a checkout with RhinoForge markers.

    A clean, unrelated Git repository must not be able to satisfy the runtime
    commit and cleanliness gates supplied by a campaign contract.
    """

    try:
        resolved = root.resolve(strict=True)
    except OSError:
        return False
    if not resolved.is_dir():
        return False
    markers_ok = bool(
        (resolved / ".git").exists()
        and (resolved / "pyproject.toml").is_file()
        and (resolved / "docs" / "architecture.md").is_file()
        and (resolved / "python" / "rpu_backend").is_dir()
    )
    if not markers_ok:
        return False
    try:
        completed = subprocess.run(
            ["git", "-C", str(resolved), "rev-parse", "--show-toplevel"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        top_level = Path(completed.stdout.strip()).resolve(strict=True)
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and top_level == resolved


def _git_head(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    head = completed.stdout.strip()
    if completed.returncode != 0 or len(head) != 40:
        return None
    return head


def _git_clean(root: Path) -> bool:
    try:
        completed = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return completed.returncode == 0 and not completed.stdout.strip()


def _command_available(argv: list[str], contract_path: Path) -> bool:
    executable = argv[0]
    if os.sep in executable or (os.altsep and os.altsep in executable):
        path = _resolve_contract_path(contract_path, executable)
        executable_ok = path.is_file() and os.access(path, os.X_OK)
    else:
        executable_ok = shutil.which(executable) is not None
    if not executable_ok:
        return False
    executable_name = Path(executable).name.lower()
    if executable_name.startswith("python"):
        arguments = argv[1:]
        if not arguments or "-c" in arguments:
            return False
        if "-m" in arguments:
            module_index = arguments.index("-m") + 1
            return module_index < len(arguments) and bool(arguments[module_index])
        script = next((arg for arg in arguments if not arg.startswith("-")), None)
        if script is None:
            return False
        script_path = _resolve_contract_path(contract_path, script)
        return script_path.is_file()
    return True


def _profile_runner_matches_adapter(
    contract_path: Path, commands: dict[str, Any]
) -> bool:
    argv = commands["profile_one_call"]
    executable_name = Path(argv[0]).name.lower()
    runner_index = 1 if executable_name.startswith("python") else 0
    if len(argv) <= runner_index or argv[runner_index].startswith("-"):
        return False
    runner = _resolve_contract_path(contract_path, argv[runner_index])
    adapter = _resolve_contract_path(contract_path, commands["benchmark_adapter"])
    try:
        return (
            runner.resolve(strict=True) == adapter.resolve(strict=True)
            and runner.is_file()
        )
    except OSError:
        return False


def _read_manifest(manifest: Path, asset_size: int) -> set[str]:
    try:
        manifest_stat = manifest.stat()
    except OSError as error:
        raise ValidationError("kernel manifest is unavailable") from error
    if not stat.S_ISREG(manifest_stat.st_mode):
        raise ValidationError("kernel manifest is not a regular file")
    if manifest_stat.st_size > MAX_MANIFEST_BYTES:
        raise ValidationError("kernel manifest exceeds 1 MiB")
    try:
        # Only the adjacent public .kernels metadata is parsed.  The opaque
        # .ref asset is separately streamed into SHA-256 and never decoded.
        lines = manifest.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError("kernel manifest is unreadable UTF-8") from error
    if not lines or lines[0] != MANIFEST_HEADER:
        raise ValidationError("kernel manifest header is invalid")
    if len(lines) < 2 or not lines[1].startswith("asset-size="):
        raise ValidationError("kernel manifest is missing asset-size")
    size_text = lines[1][len("asset-size=") :]
    if not size_text or not size_text.isascii() or not size_text.isdecimal():
        raise ValidationError("kernel manifest asset-size is invalid")
    if int(size_text, 10) != asset_size:
        raise ValidationError("kernel manifest asset-size does not match asset stat")
    names = lines[2:]
    if not names:
        raise ValidationError("kernel manifest has no kernel names")
    if any(KERNEL_NAME_RE.fullmatch(name) is None for name in names):
        raise ValidationError("kernel manifest contains an invalid kernel name")
    if names != sorted(set(names)):
        raise ValidationError("kernel manifest names must be unique and sorted")
    return set(names)


def _stream_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                block = handle.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
    except OSError as error:
        raise ValidationError("runtime identity file is unreadable") from error
    return digest.hexdigest()


def _valid_pinned_sha256(value: Any) -> bool:
    return isinstance(value, str) and value != ZERO_SHA256 and bool(
        SHA256_RE.fullmatch(value)
    )


def _regular_file_matches_sha256(path: Path, expected: Any) -> bool:
    if not _valid_pinned_sha256(expected):
        return False
    try:
        return stat.S_ISREG(path.stat().st_mode) and _stream_sha256(path) == expected
    except (OSError, ValidationError):
        return False


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    """Write a strict JSON receipt atomically in its destination directory."""

    destination = Path(os.path.abspath(os.fspath(path)))
    if destination.is_symlink():
        raise ValidationError("preflight output must not be a symbolic link")
    parent = destination.parent
    if not parent.is_dir():
        raise ValidationError("preflight output directory is unavailable")
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        dir=parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise


def _device_capable(path: Path) -> bool:
    try:
        mode = path.stat().st_mode
    except OSError:
        return False
    return stat.S_ISCHR(mode) and os.access(path, os.R_OK | os.W_OK)


def run_preflight(
    contract_path: Path,
    *,
    probe_torch_rpu: bool = False,
    device_glob: str = "/dev/rpu*",
    dev_mem: Path = Path("/dev/mem"),
    device_compiler: Path | None = None,
    rhinoforge_root: Path | None = None,
) -> dict[str, Any]:
    contract_path = contract_path.resolve(strict=False)
    checks: list[dict[str, Any]] = []
    try:
        exact_contract_sha256: str | None = contract_sha256(contract_path)
    except ValidationError:
        exact_contract_sha256 = None

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append(
            {"name": name, "status": "pass" if passed else "fail", "detail": detail}
        )

    try:
        contract = load_contract(contract_path)
    except ValidationError as error:
        record("contract", False, str(error))
        return {
            "schema_version": 1,
            "ok": False,
            "contract_sha256": exact_contract_sha256,
            "empirical_floor_required": True,
            "checks": checks,
        }
    record("contract", True, "schema v1 and approved state")

    for command_name in ("correctness", "benchmark", "profile_one_call"):
        argv = contract["commands"][command_name]
        command_available = _command_available(argv, contract_path)
        record(
            f"command.{command_name}",
            command_available,
            "non-shell argv and executable resolved"
            if command_available
            else "command executable is unavailable",
        )

    adapter = _resolve_contract_path(
        contract_path, contract["commands"]["benchmark_adapter"]
    )
    adapter_identity_ok = _regular_file_matches_sha256(
        adapter, contract["commands"]["benchmark_adapter_sha256"]
    )
    record(
        "command.benchmark_adapter_identity",
        adapter_identity_ok,
        "immutable benchmark adapter SHA-256 matches"
        if adapter_identity_ok
        else "benchmark adapter is unavailable or SHA-256 mismatched",
    )
    profile_runner_ok = _profile_runner_matches_adapter(
        contract_path, contract["commands"]
    )
    record(
        "command.profile_runner_identity",
        profile_runner_ok,
        "one-call profiler uses the frozen benchmark adapter"
        if profile_runner_ok
        else "one-call profiler runner differs from the frozen adapter",
    )
    if contract["attestation"]["mode"] == "external":
        verifier_path = Path(contract["attestation"]["verifier_executable"])
        verifier_ok = bool(
            verifier_path.is_absolute()
            and _regular_file_matches_sha256(
                verifier_path, contract["attestation"]["verifier_sha256"]
            )
            and os.access(verifier_path, os.X_OK)
        )
        record(
            "attestation.release_verifier",
            verifier_ok,
            "DSSE Ed25519 verifier executable SHA-256 matches"
            if verifier_ok
            else "DSSE Ed25519 verifier is unavailable or mismatched",
        )
    else:
        checks.append(
            {
                "name": "attestation.release_verifier",
                "status": "skip",
                "detail": "local diagnostics allowed; release selection is blocked",
            }
        )

    for root_name in ("candidate_root", "reference_root"):
        root = _resolve_contract_path(contract_path, contract["campaign"][root_name])
        is_directory = root.is_dir()
        record(
            f"campaign.{root_name}",
            is_directory,
            "directory exists" if is_directory else "directory is unavailable",
        )
    reference_root = _resolve_contract_path(
        contract_path, contract["campaign"]["reference_root"]
    )
    try:
        observed_reference_hash = content_tree_sha256(reference_root)
    except ValidationError:
        observed_reference_hash = None
    reference_identity_ok = bool(
        observed_reference_hash
        and observed_reference_hash
        == contract["campaign"]["reference_tree_sha256"]
    )
    record(
        "campaign.reference_tree_identity",
        reference_identity_ok,
        "frozen reference content-tree SHA-256 matches"
        if reference_identity_ok
        else "reference tree is unsafe, unavailable, or SHA-256 mismatched",
    )

    selected_rhinoforge_root = rhinoforge_root or _find_rhinoforge_root()
    root_identity_ok = bool(
        selected_rhinoforge_root
        and _is_rhinoforge_root(selected_rhinoforge_root)
    )
    record(
        "runtime.rhinoforge_root_identity",
        root_identity_ok,
        "Git top-level and RhinoForge repository markers match"
        if root_identity_ok
        else "selected root is not the top level of a RhinoForge checkout",
    )
    head = (
        _git_head(selected_rhinoforge_root)
        if selected_rhinoforge_root and root_identity_ok
        else None
    )
    commit_matches = head == contract["runtime"]["rhinoforge_commit"]
    record(
        "runtime.rhinoforge_commit",
        commit_matches,
        "exact full commit matches"
        if commit_matches
        else "current RhinoForge commit is unavailable or mismatched",
    )
    clean_tree = bool(
        selected_rhinoforge_root
        and root_identity_ok
        and _git_clean(selected_rhinoforge_root)
    )
    record(
        "runtime.rhinoforge_clean_tree",
        clean_tree,
        "RhinoForge worktree is clean"
        if clean_tree
        else "RhinoForge worktree is dirty or unavailable",
    )

    declared_asset = contract["runtime"]["operator_asset"]
    declared_manifest = contract["runtime"]["kernel_manifest"]
    configured_asset = os.environ.get("RPU_KERNEL_LIB_PATH")
    if configured_asset:
        candidate_asset = Path(configured_asset).resolve(strict=False)
        name_matches = candidate_asset.name == Path(declared_asset).name
    else:
        candidate_asset = _resolve_contract_path(contract_path, declared_asset)
        name_matches = True
    candidate_manifest = Path(str(candidate_asset) + ".kernels")
    manifest_name_matches = candidate_manifest.name == Path(declared_manifest).name
    record(
        "runtime.asset_name",
        name_matches,
        "configured opaque asset name matches contract"
        if name_matches
        else "configured opaque asset name mismatches contract",
    )
    record(
        "runtime.manifest_name",
        manifest_name_matches,
        "adjacent manifest name matches contract"
        if manifest_name_matches
        else "adjacent manifest name mismatches contract",
    )

    launch_library = _resolve_contract_path(
        contract_path, contract["runtime"]["launch_library"]
    )
    try:
        launch_regular = stat.S_ISREG(launch_library.stat().st_mode)
        launch_hash_matches = (
            launch_regular
            and _stream_sha256(launch_library)
            == contract["runtime"]["launch_library_sha256"]
        )
    except (OSError, ValidationError):
        launch_hash_matches = False
    record(
        "runtime.launch_library_identity",
        launch_hash_matches,
        "Launch library SHA-256 matches"
        if launch_hash_matches
        else "Launch library is unavailable or SHA-256 mismatched",
    )

    manifest_names: set[str] | None = None
    asset_identity_ok = False
    manifest_identity_ok = False
    asset_size = 0
    try:
        # The opaque asset is streamed only into SHA-256.  No bytes are parsed,
        # decoded, logged, or included in the report.
        asset_stat = candidate_asset.stat()
        if not stat.S_ISREG(asset_stat.st_mode):
            raise ValidationError("operator asset is not a regular file")
        asset_size = asset_stat.st_size
        asset_identity_ok = (
            _stream_sha256(candidate_asset)
            == contract["runtime"]["operator_asset_sha256"]
        )
    except (OSError, ValidationError) as error:
        record("runtime.operator_asset_identity", False, str(error))
    if not any(check["name"] == "runtime.operator_asset_identity" for check in checks):
        record(
            "runtime.operator_asset_identity",
            asset_identity_ok,
            "opaque asset SHA-256 matches"
            if asset_identity_ok
            else "opaque asset SHA-256 mismatched",
        )
    try:
        manifest_identity_ok = (
            _stream_sha256(candidate_manifest)
            == contract["runtime"]["kernel_manifest_sha256"]
        )
    except ValidationError as error:
        record("runtime.kernel_manifest_identity", False, str(error))
    if not any(check["name"] == "runtime.kernel_manifest_identity" for check in checks):
        record(
            "runtime.kernel_manifest_identity",
            manifest_identity_ok,
            "kernel manifest SHA-256 matches"
            if manifest_identity_ok
            else "kernel manifest SHA-256 mismatched",
        )
    if asset_identity_ok and manifest_identity_ok:
        try:
            manifest_names = _read_manifest(candidate_manifest, asset_size)
        except ValidationError as error:
            record("runtime.asset_manifest", False, str(error))
        else:
            record(
                "runtime.asset_manifest",
                True,
                "public manifest header and asset-size match",
            )
    else:
        record(
            "runtime.asset_manifest",
            False,
            "manifest admission is blocked by runtime identity mismatch",
        )
    required_kernels = set(contract["runtime"]["required_kernels"])
    kernels_present = manifest_names is not None and required_kernels <= manifest_names
    record(
        "runtime.required_kernels",
        kernels_present,
        "all required public kernel names are admitted"
        if kernels_present
        else "one or more required kernel names are missing",
    )

    rpu_paths = [
        Path(value)
        for value in sorted(glob.glob(device_glob))
        if Path(value).parent == Path("/dev")
        and RPU_DEVICE_NAME_RE.fullmatch(Path(value).name) is not None
    ]
    usable_rpu_count = sum(_device_capable(path) for path in rpu_paths)
    record(
        "device.rpu",
        usable_rpu_count > 0,
        f"{usable_rpu_count} usable character device(s)"
        if usable_rpu_count
        else "no readable/writable RPU character device",
    )
    mem_path_is_canonical = dev_mem == Path("/dev/mem")
    mem_capable = mem_path_is_canonical and _device_capable(dev_mem)
    record(
        "device.mem",
        mem_capable,
        "readable/writable character device"
        if mem_capable
        else "missing or inaccessible character device",
    )

    if contract["runtime"]["requires_device_compiler"]:
        runtime = contract["runtime"]
        authorization_declared = runtime.get("device_compiler_authorized") is True
        authorization_path_value = runtime.get("compiler_authorization_receipt")
        authorization_hash = runtime.get("compiler_authorization_receipt_sha256")
        authorization_receipt_ok = False
        if (
            authorization_declared
            and isinstance(authorization_path_value, str)
            and bool(authorization_path_value)
            and _valid_pinned_sha256(authorization_hash)
        ):
            authorization_path = _resolve_contract_path(
                contract_path, authorization_path_value
            )
            authorization_receipt_ok = _regular_file_matches_sha256(
                authorization_path, authorization_hash
            )
        record(
            "runtime.device_compiler_authorization",
            authorization_declared,
            "compiler use explicitly authorized by resolved release owner"
            if authorization_declared
            else "contract does not explicitly authorize compiler use",
        )
        record(
            "runtime.device_compiler_authorization_receipt",
            authorization_receipt_ok,
            "pinned compiler authorization receipt SHA-256 matches"
            if authorization_receipt_ok
            else (
                "compiler authorization receipt path/SHA-256 is missing, "
                "unresolved, unavailable, or mismatched"
            ),
        )
        compiler_value = device_compiler
        if compiler_value is None:
            environment_value = os.environ.get("RHINOFORGE_DEVICE_COMPILER")
            compiler_value = Path(environment_value) if environment_value else None
        compiler_ok = bool(
            compiler_value
            and compiler_value.is_file()
            and os.access(compiler_value, os.X_OK)
        )
        record(
            "runtime.device_compiler",
            compiler_ok,
            "explicit compiler executable is available"
            if compiler_ok
            else "required compiler executable is unavailable",
        )
        compiler_binary_identity_ok = False
        if compiler_ok and compiler_value is not None:
            try:
                resolved_compiler = compiler_value.resolve(strict=True)
            except OSError:
                resolved_compiler = compiler_value
            compiler_binary_identity_ok = _regular_file_matches_sha256(
                resolved_compiler,
                runtime.get("device_compiler_sha256"),
            )
        record(
            "runtime.device_compiler_sha256",
            compiler_binary_identity_ok,
            "compiler executable SHA-256 matches the contract"
            if compiler_binary_identity_ok
            else (
                "device_compiler_sha256 is missing, unresolved, or does not "
                "match the executable"
            ),
        )
        compiler_identity_ok = False
        if (
            compiler_ok
            and compiler_binary_identity_ok
            and compiler_value is not None
        ):
            try:
                completed = subprocess.run(
                    [str(compiler_value.resolve(strict=True)), "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                output_lines = completed.stdout.splitlines()
                if not output_lines:
                    output_lines = completed.stderr.splitlines()
                compiler_identity_ok = bool(
                    completed.returncode == 0
                    and output_lines
                    and output_lines[0].strip()
                    == runtime["device_compiler_id"]
                )
            except (OSError, subprocess.SubprocessError):
                pass
        record(
            "runtime.device_compiler_identity",
            compiler_identity_ok,
            "compiler --version first line matches exact identity"
            if compiler_identity_ok
            else "compiler identity is unavailable or mismatched",
        )
    else:
        for check_name in (
            "runtime.device_compiler_authorization",
            "runtime.device_compiler_authorization_receipt",
            "runtime.device_compiler",
            "runtime.device_compiler_sha256",
            "runtime.device_compiler_identity",
        ):
            checks.append(
                {
                    "name": check_name,
                    "status": "skip",
                    "detail": "contract does not require a device compiler",
                }
            )

    if probe_torch_rpu:
        torch_ok = False
        detail = "torch.rpu probe failed"
        try:
            importlib.import_module("rpu_backend")
            torch = importlib.import_module("torch")
            namespace = getattr(torch, "rpu", None)
            available = getattr(namespace, "is_available", None)
            version_ok = str(getattr(torch, "__version__", "")) == contract["runtime"][
                "torch_version"
            ]
            synchronize = getattr(namespace, "synchronize", None)
            namespace_ok = bool(
                callable(available)
                and available()
                and callable(synchronize)
                and version_ok
            )
            if namespace_ok:
                source = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float16)
                device_value = source.to("rpu")
                synchronize()
                host_value = device_value.to("cpu")
                synchronize()
                torch_ok = bool(
                    getattr(host_value, "dtype", None) == torch.float16
                    and host_value.tolist() == source.tolist()
                )
            detail = (
                "torch.rpu FP16 host-device-host roundtrip passed"
                if torch_ok
                else "torch.rpu roundtrip unavailable, mismatched, or incorrect"
            )
        except Exception:
            # Do not copy exception messages: import failures can contain source
            # paths or loader details that do not belong in a campaign record.
            pass
        record("probe.torch_rpu", torch_ok, detail)
    else:
        checks.append(
            {
                "name": "probe.torch_rpu",
                "status": "skip",
                "detail": "optional probe not requested",
            }
        )

    roofline = contract.get("roofline", {})
    roofline_fields = {
        "peak_ops_per_s",
        "bandwidth_bytes_per_s",
        "launch_floor_s",
        "peak_kind",
        "source_id",
        "source_file",
        "source_sha256",
    }
    empirical_floor_required = not roofline_fields <= set(roofline)
    if not empirical_floor_required:
        roofline_source = _resolve_contract_path(
            contract_path, roofline["source_file"]
        )
        source_ok = _regular_file_matches_sha256(
            roofline_source, roofline["source_sha256"]
        )
        record(
            "roofline.source_identity",
            source_ok,
            "roofline source bytes match the frozen SHA-256"
            if source_ok
            else "roofline source is unavailable or SHA-256 mismatched",
        )
    checks.append(
        {
            "name": "roofline",
            "status": "pass",
            "detail": (
                "complete declared roofline inputs"
                if not empirical_floor_required
                else "incomplete roofline; empirical floor evidence is required"
            ),
        }
    )

    ok = all(check["status"] != "fail" for check in checks)
    return {
        "schema_version": 1,
        "ok": ok,
        "contract_sha256": exact_contract_sha256,
        "campaign_id": contract["campaign"]["id"],
        "operator": contract["campaign"]["op"],
        "runtime_set": contract["runtime"]["runtime_set"],
        "workload_count": len(contract["workloads"]),
        "launch_library_name": Path(contract["runtime"]["launch_library"]).name,
        "operator_asset_name": Path(declared_asset).name,
        "kernel_manifest_name": Path(declared_manifest).name,
        "required_kernel_count": len(required_kernels),
        "empirical_floor_required": empirical_floor_required,
        "checks": checks,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a schema-v1 approved RhinoForge kernel campaign.",
        epilog=(
            "Contract tables: campaign, commands, runtime, measurement, gates, "
            "one or more [[workloads]], and optional roofline. Commands are "
            "validated as argv arrays and are never executed. Set "
            "RHINOFORGE_DEVICE_COMPILER to an executable when the contract "
            "requires a device compiler."
        ),
    )
    parser.add_argument("contract", type=Path, help="approved TOML contract")
    parser.add_argument(
        "--probe-torch-rpu",
        action="store_true",
        help=(
            "deprecated compatibility flag; the CLI always requires the "
            "torch.rpu FP16 roundtrip"
        ),
    )
    parser.add_argument(
        "--device-compiler",
        type=Path,
        help="explicit compiler executable for a compiler-required contract",
    )
    parser.add_argument(
        "--rhinoforge-root",
        type=Path,
        help="RhinoForge Git checkout whose exact HEAD is validated",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="atomically write the strict JSON preflight receipt to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_preflight(
        args.contract,
        # A command-line campaign receipt is never allowed to skip the real
        # backend FP16 transfer probe.  The function argument remains
        # injectable so board-free unit tests can isolate other checks.
        probe_torch_rpu=True,
        device_compiler=args.device_compiler,
        rhinoforge_root=args.rhinoforge_root,
    )
    if args.output is not None:
        try:
            _atomic_write_json(args.output, report)
        except (OSError, ValidationError):
            print("preflight receipt could not be written atomically", file=sys.stderr)
            return 2
    print(json.dumps(report, allow_nan=False, ensure_ascii=False, sort_keys=True))
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    sys.exit(main())
