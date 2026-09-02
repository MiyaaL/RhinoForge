#!/usr/bin/env python3
"""Board-free identity and end-to-end smoke test for the official hxcc chain.

The hxcc Python entry point is only a wrapper around clang-17, rpuas, and
rhino_gen_oplib.  A wrapper version or hash alone therefore does not identify
the compiler that produced a REF.  This helper records the wrapper and each
underlying executable, checks the target/optimization/oplib invocation from a
driver dry-run, and (unless ``--identity-only`` is requested) compiles a tiny
clean-room ``.rc`` in a private temporary directory.  It intentionally emits
hashes, sizes, and bounded diagnostics rather than REF/assembly contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


# clang-17 in the official wheel is ~58 MiB; keep a bounded but realistic
# ceiling for executable identity while still rejecting unbounded inputs.
MAX_FILE_BYTES = 256 * 1024 * 1024
MAX_CAPTURE_BYTES = 512 * 1024
MAX_TIMEOUT_S = 300.0
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_TARGET = "rpu-rhino-rpuhsa"
MANUAL_ARCHIVE_SHA256 = (
    "21e8ed1462620b84e4d818eae24b8507817cbeca1a39105cf3863d8c8287eeba"
)
ARCH_ALIASES = {
    "aarch64": "aarch64",
    "arm64": "aarch64",
    "x86_64": "x86_64",
    "amd64": "x86_64",
}
WHEEL_TAGS = {
    "aarch64": "manylinux2014_aarch64",
    "x86_64": "manylinux1_x86_64",
}
ELF_MACHINES = {183: "aarch64", 62: "x86_64"}

# This is deliberately minimal: it exercises the RPU entry-point/type parser
# and the complete clang -> rpuas -> rhino_gen_oplib path without depending on
# a board or on a copyrighted sample from the developer guide.
SMOKE_SOURCE = """#include \"__clang_rpu_builtin_types.h\"
#include \"__clang_rpu_builtin_vars.h\"
__rprog__ void rhinoforge_hxcc_smoke(__local__ f16v16 *out) {
  out[0] = out[0];
}
"""


class ValidationError(ValueError):
    """A toolchain identity or smoke invariant is not admissible."""


def _validated_timeout(timeout: Any) -> float:
    """Return a finite subprocess timeout within the campaign bound.

    Keep this check at the subprocess boundary as well as in ``run_preflight``:
    the helper is intentionally small and is exercised directly by a few
    board-free callers.  In particular, ``float(10**10000)`` raises
    ``OverflowError`` instead of producing a finite value.
    """

    try:
        value = float(timeout)
    except (OverflowError, TypeError, ValueError):
        value = math.nan
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(value)
        or value <= 0
        or value > MAX_TIMEOUT_S
    ):
        raise ValidationError(f"timeout must be in (0, {MAX_TIMEOUT_S}] seconds")
    return value


def _normalize_arch(value: str) -> str:
    return ARCH_ALIASES.get(value.strip().lower(), value.strip().lower())


def _bounded_text(raw: bytes) -> str:
    if len(raw) > MAX_CAPTURE_BYTES:
        raw = raw[:MAX_CAPTURE_BYTES] + b"\n[output truncated]\n"
    return raw.decode("utf-8", errors="replace")


def _run(
    argv: list[str], *, cwd: Path | None = None, timeout: float = 30.0
) -> dict[str, Any]:
    if not argv or any(not isinstance(item, str) for item in argv):
        raise ValidationError("command argv must be a non-empty string list")
    timeout = _validated_timeout(timeout)
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd) if cwd is not None else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout if isinstance(error.stdout, bytes) else b""
        stderr = error.stderr if isinstance(error.stderr, bytes) else b""
        return {
            "returncode": None,
            "timed_out": True,
            "stdout": _bounded_text(stdout),
            "stderr": _bounded_text(stderr),
        }
    except OSError as error:
        return {
            "returncode": None,
            "timed_out": False,
            "error": f"{type(error).__name__}: {error}",
            "stdout": "",
            "stderr": "",
        }
    return {
        "returncode": int(completed.returncode),
        "timed_out": False,
        "stdout": _bounded_text(completed.stdout),
        "stderr": _bounded_text(completed.stderr),
    }


def _resolve_executable(value: str) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value)
    if "/" not in value and (os.altsep is None or os.altsep not in value):
        found = shutil.which(value)
        if not found:
            return None
        candidate = Path(found)
    try:
        resolved = candidate.resolve(strict=True)
        mode = resolved.stat().st_mode
    except OSError:
        return None
    if not stat.S_ISREG(mode) or not os.access(resolved, os.X_OK):
        return None
    return resolved


def _file_snapshot(
    path: Path, *, label: str = "file", maximum: int = MAX_FILE_BYTES
) -> tuple[str, int, bytes]:
    """Hash a regular file through one identity-checked descriptor.

    Tool and manual identities are part of the campaign contract.  Reading a
    path with ``Path.open`` after a separate ``lstat`` leaves a replacement or
    same-size mutation race between the hash and the ELF/header probe.  Keep
    the descriptor open for the complete bounded read and re-check the path
    before accepting the snapshot.
    """

    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical, label)
    absolute = _absolute_path(lexical)
    try:
        initial = os.lstat(absolute)
    except (OSError, ValueError) as error:
        raise ValidationError(f"{label} is unavailable") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise ValidationError(f"{label} must be a regular non-symlink file")
    if initial.st_size > maximum:
        raise ValidationError(f"{label} exceeds {maximum} bytes")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except (OSError, ValueError) as error:
        raise ValidationError(f"{label} cannot be opened safely") from error
    initial_identity = (
        initial.st_dev,
        initial.st_ino,
        initial.st_size,
        initial.st_mtime_ns,
        initial.st_ctime_ns,
    )
    try:
        opened = os.fstat(descriptor)
        opened_identity = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
            opened.st_ctime_ns,
        )
        if not stat.S_ISREG(opened.st_mode) or opened_identity != initial_identity:
            raise ValidationError(f"{label} changed before reading")
        digest = hashlib.sha256()
        header = bytearray()
        observed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            if observed > maximum:
                raise ValidationError(f"{label} exceeds {maximum} bytes")
            digest.update(block)
            if len(header) < 20:
                header.extend(block[: 20 - len(header)])
        final = os.fstat(descriptor)
        final_identity = (
            final.st_dev,
            final.st_ino,
            final.st_size,
            final.st_mtime_ns,
            final.st_ctime_ns,
        )
        if observed != opened.st_size or final_identity != initial_identity:
            raise ValidationError(f"{label} changed while reading")
        try:
            current = os.lstat(absolute)
        except (OSError, ValueError) as error:
            raise ValidationError(f"{label} path changed while reading") from error
        current_identity = (
            current.st_dev,
            current.st_ino,
            current.st_size,
            current.st_mtime_ns,
            current.st_ctime_ns,
        )
        if stat.S_ISLNK(current.st_mode) or not stat.S_ISREG(current.st_mode):
            raise ValidationError(f"{label} path changed while reading")
        if current_identity != initial_identity:
            raise ValidationError(f"{label} path changed while reading")
        return digest.hexdigest(), observed, bytes(header)
    except OSError as error:
        raise ValidationError(f"{label} could not be read") from error
    finally:
        os.close(descriptor)


def _sha256(path: Path) -> str:
    return _file_snapshot(path, label="file")[0]


def _elf_arch_bytes(header: bytes) -> str | None:
    if len(header) < 20 or header[:4] != b"\x7fELF":
        return None
    endian = "<" if header[5] == 1 else ">" if header[5] == 2 else ""
    if not endian:
        return None
    try:
        machine = struct.unpack_from(endian + "H", header, 18)[0]
    except struct.error:
        return None
    return ELF_MACHINES.get(machine)


def _tool_identity(
    path: Path | None, host_arch: str, *, require_elf: bool = False
) -> dict[str, Any]:
    if path is None:
        return {"available": False}
    try:
        digest, size, header = _file_snapshot(path, label="executable")
    except (OSError, ValidationError) as error:
        return {"available": False, "path": str(path), "error": str(error)}
    elf_arch = _elf_arch_bytes(header)
    # hxcc itself is a Python wrapper, while the three package tools are ELF.
    # Requiring ELF for the package tools prevents a shell/Python shim from
    # being mistaken for the compiler that produced a REF.
    arch_ok = (
        (elf_arch == host_arch if host_arch else elf_arch is not None)
        if require_elf
        else (elf_arch is None or not host_arch or elf_arch == host_arch)
    )
    return {
        "available": True,
        "path": str(path),
        "sha256": digest,
        "size": size,
        "elf_arch": elf_arch,
        "elf_required": require_elf,
        "host_arch_match": arch_ok,
    }


def _hash_expectation(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    if SHA256_RE.fullmatch(value) is None:
        raise ValidationError(f"{label} must be lowercase SHA-256")
    return value


def _expected_hash_check(identity: dict[str, Any], expected: str | None) -> bool | None:
    if expected is None:
        return None
    return bool(identity.get("available") and identity.get("sha256") == expected)


def _resource_dir_from_output(text: str) -> Path | None:
    # clang prints the resource directory as a single absolute path.  Select
    # the last existing directory so diagnostics preceding it cannot confuse
    # the resolver.
    candidates: list[Path] = []
    for line in text.splitlines():
        value = line.strip().strip('"')
        if value.startswith("/"):
            path = Path(value)
            if path.is_dir():
                candidates.append(path)
    return candidates[-1] if candidates else None


def _discover_package_tools(resource_dir: Path | None) -> dict[str, Path | None]:
    if resource_dir is None:
        return {"clang": None, "rpuas": None, "rhino_gen_oplib": None}
    # .../hxcc/data/lib/clang/17 -> .../hxcc/data/bin
    data_dir = resource_dir.parent.parent.parent
    bin_dir = data_dir / "bin"
    return {
        "clang": _resolve_executable(str(bin_dir / "clang-17")),
        "rpuas": _resolve_executable(str(bin_dir / "rpuas")),
        "rhino_gen_oplib": _resolve_executable(
            str(bin_dir / "rhino_gen_oplib")
        ),
    }


def _driver_tokens(text: str) -> list[list[str]]:
    """Tokenize each command line of clang's ``-###`` output.

    ``-###`` is diagnostic text rather than a stable machine-readable format,
    so malformed lines are ignored.  Exact command-token matching is still
    preferable to substring searches: a source called ``rpuas`` must not make
    an otherwise incomplete chain look valid.
    """

    commands: list[list[str]] = []
    for line in text.splitlines():
        try:
            tokens = shlex.split(line, posix=True)
        except ValueError:
            continue
        if tokens:
            commands.append(tokens)
    return commands


def _command_name(tokens: list[str]) -> str:
    if not tokens:
        return ""
    return Path(tokens[0]).name


def _has_pair(tokens: list[str], first: str, second: str) -> bool:
    return any(
        tokens[index] == first and index + 1 < len(tokens) and tokens[index + 1] == second
        for index in range(len(tokens))
    )


def _path_token_matches(token: str, expected: Path | None) -> bool:
    if expected is None:
        return True
    try:
        return Path(token).resolve(strict=False) == expected
    except OSError:
        return False


def _parse_driver(
    text: str,
    expected_target: str,
    expected_tools: dict[str, Path | None] | None = None,
) -> dict[str, Any]:
    commands = _driver_tokens(text)
    cc1 = next((tokens for tokens in commands if "-cc1" in tokens), [])
    assembler = next(
        (tokens for tokens in commands if _command_name(tokens) == "rpuas"), []
    )
    oplib = next(
        (tokens for tokens in commands if _command_name(tokens) == "rhino_gen_oplib"),
        [],
    )
    target = None
    for index, token in enumerate(cc1):
        if token == "-triple" and index + 1 < len(cc1):
            target = cc1[index + 1]
            break
    has_o2 = "-O2" in cc1
    has_o0 = "-O0" in cc1
    has_rpuas = bool(assembler)
    has_oplib = bool(oplib)
    has_r1 = _has_pair(oplib, "-m", "r1")
    expected_tools = expected_tools or {}
    path_matches = {
        "clang": bool(cc1)
        and _path_token_matches(cc1[0], expected_tools.get("clang")),
        "rpuas": bool(assembler)
        and _path_token_matches(assembler[0], expected_tools.get("rpuas")),
        "rhino_gen_oplib": bool(oplib)
        and _path_token_matches(
            oplib[0], expected_tools.get("rhino_gen_oplib")
        ),
    }
    return {
        "clang_command_present": bool(cc1),
        "target_triple": target,
        "target_ok": target == expected_target,
        "optimization_o2": has_o2,
        "o0_present": has_o0,
        "assembler_present": has_rpuas,
        "oplib_generator_present": has_oplib,
        "oplib_mode_r1": has_r1,
        "tool_paths_match": path_matches,
        "chain_ok": bool(
            target == expected_target
            and cc1
            and has_o2
            and not has_o0
            and has_rpuas
            and has_oplib
            and has_r1
            and all(path_matches.values())
        ),
    }


def _safe_source(source: Path | None, destination: Path) -> Path:
    if source is None:
        path = destination / "rhinoforge_hxcc_smoke.rc"
        path.write_text(SMOKE_SOURCE, encoding="utf-8")
        return path
    source_lexical = _lexical_absolute(source)
    source_bytes = _read_source_bytes(source_lexical)
    target = destination / source.name
    # Copy the descriptor-validated snapshot: hxcc writes outputs next to the
    # input, while the original source may be concurrently edited by another
    # process.  Never let that race alter what is compiled.
    target.write_bytes(source_bytes)
    return target


def _absolute_path(path: Path) -> Path:
    # ``Path.resolve`` follows a final symlink.  Use an absolute lexical path
    # for output safety so a symlink destination cannot be mistaken for its
    # target and overwritten.
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute spelling without resolving links or ``..``."""

    value = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(value):
        value = os.path.join(os.getcwd(), value)
    return Path(value)


def _reject_symlink_components(path: Path, label: str = "output") -> None:
    current = Path(path.anchor) if path.anchor else Path(".")
    parts = path.parts[1:] if path.anchor else path.parts
    missing = False
    for component_index, part in enumerate(parts):
        if part in {"", "."}:
            continue
        if missing:
            if part == "..":
                raise ValidationError(f"{label} path contains an unavailable component")
            continue
        if part == "..":
            current = current.parent
            continue
        current = current / part
        try:
            info = os.lstat(current)
        except FileNotFoundError:
            # Remaining components cannot exist until a later mkdir; retain a
            # marker so an ensuing ``..`` cannot be silently normalized into a
            # different path.
            missing = True
            continue
        except OSError as error:
            raise ValidationError(f"{label} path cannot be inspected") from error
        # Leave the final component to the regular-file lstat/open checks so
        # callers retain their precise diagnostics; reject every parent link.
        if stat.S_ISLNK(info.st_mode) and component_index != len(parts) - 1:
            raise ValidationError(f"{label} path must not contain symbolic links")


def _read_source_bytes(path: Path) -> bytes:
    """Read a source snapshot with no-follow and descriptor identity checks."""

    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical, "--source")
    absolute = _absolute_path(lexical)
    try:
        initial = os.lstat(absolute)
    except OSError as error:
        raise ValidationError("--source is unavailable") from error
    if stat.S_ISLNK(initial.st_mode) or not stat.S_ISREG(initial.st_mode):
        raise ValidationError("--source must be a regular non-symlink file")
    if initial.st_size > MAX_FILE_BYTES:
        raise ValidationError("--source is too large")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValidationError("--source cannot be opened safely") from error
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
            raise ValidationError("--source changed before reading")
        proc_fd = f"/proc/self/fd/{descriptor}"
        if os.path.lexists(proc_fd):
            try:
                fd_target = os.path.realpath(proc_fd)
            except OSError:
                fd_target = ""
            if fd_target and fd_target != os.path.realpath(os.fspath(absolute)):
                raise ValidationError("--source path changed before reading")
        chunks: list[bytes] = []
        observed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            if observed > MAX_FILE_BYTES:
                raise ValidationError("--source is too large")
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
            raise ValidationError("--source changed while reading")
        return b"".join(chunks)
    except OSError as error:
        raise ValidationError("--source is unreadable") from error
    finally:
        os.close(descriptor)


def _same_path(first: Path, second: Path | None) -> bool:
    if second is None:
        return False
    first_abs = _absolute_path(first)
    second_abs = _absolute_path(second)
    try:
        if first_abs.exists() and second_abs.exists() and os.path.samefile(first_abs, second_abs):
            return True
    except OSError:
        pass
    return first_abs == second_abs


def _safe_output_path(path: Path, forbidden: Iterable[Path | None] = ()) -> Path:
    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical, "output")
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
        if _same_path(destination, candidate):
            raise ValidationError("output must differ from compiler/source inputs")
    return destination


def _output_json(
    path: Path, value: dict[str, Any], *, forbidden: Iterable[Path | None] = ()
) -> None:
    destination = _safe_output_path(path, forbidden)
    payload = (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
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


def run_preflight(
    compiler: str = "hxcc",
    *,
    source: Path | None = None,
    identity_only: bool = False,
    expected_host_arch: str | None = None,
    expected_target: str = DEFAULT_TARGET,
    expected_wrapper_sha256: str | None = None,
    expected_clang_sha256: str | None = None,
    expected_rpuas_sha256: str | None = None,
    expected_oplib_sha256: str | None = None,
    manual_archive: Path | None = None,
    timeout: float = 60.0,
) -> dict[str, Any]:
    if not isinstance(expected_target, str) or not expected_target:
        raise ValidationError("expected target triple must be non-empty")
    # ``subprocess.run(timeout=...)`` raises a raw ``ValueError`` for NaN/inf
    # and can raise ``OverflowError`` while coercing an enormous integer.  Do
    # the finite, bounded check here so both the API and CLI return the same
    # controlled validation error before probing any compiler executable.
    timeout = _validated_timeout(timeout)
    host_arch = _normalize_arch(platform.machine())
    expected_arch = (
        _normalize_arch(expected_host_arch) if expected_host_arch is not None else host_arch
    )
    expected_wrapper_sha256 = _hash_expectation(
        expected_wrapper_sha256, "expected wrapper hash"
    )
    expected_clang_sha256 = _hash_expectation(
        expected_clang_sha256, "expected clang hash"
    )
    expected_rpuas_sha256 = _hash_expectation(
        expected_rpuas_sha256, "expected rpuas hash"
    )
    expected_oplib_sha256 = _hash_expectation(
        expected_oplib_sha256, "expected oplib hash"
    )

    wrapper = _resolve_executable(compiler)
    report: dict[str, Any] = {
        "schema_version": 1,
        "ok": False,
        "manual_archive_sha256": MANUAL_ARCHIVE_SHA256,
        "manual_archive_verified": False,
        "host": {
            "machine": platform.machine(),
            "normalized_arch": host_arch,
            "expected_arch": expected_arch,
            "wheel_tag": WHEEL_TAGS.get(expected_arch),
            "architecture_ok": host_arch == expected_arch,
        },
        "compiler": {"requested": compiler},
        "driver": {},
        "compile_smoke": {"requested": not identity_only},
        "checks": [],
    }

    def check(name: str, passed: bool, detail: str) -> None:
        report["checks"].append(
            {"name": name, "status": "pass" if passed else "fail", "detail": detail}
        )

    if manual_archive is not None:
        try:
            archive_path = _absolute_path(manual_archive)
            info = os.lstat(archive_path)
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise ValidationError("manual archive must be a regular non-symlink file")
            observed = _sha256(archive_path)
            report["manual_archive_observed_sha256"] = observed
            report["manual_archive_bytes"] = info.st_size
            archive_ok = observed == MANUAL_ARCHIVE_SHA256
            report["manual_archive_verified"] = archive_ok
            check(
                "manual.archive",
                archive_ok,
                "archive SHA-256 matches the pinned manual"
                if archive_ok
                else "archive SHA-256 mismatches the pinned manual",
            )
        except (OSError, ValidationError) as error:
            report["manual_archive_error"] = str(error)
            check("manual.archive", False, str(error))

    check(
        "host.architecture",
        host_arch == expected_arch,
        f"host={host_arch or 'unknown'} expected={expected_arch or 'unknown'}",
    )
    if wrapper is None:
        check("wrapper.available", False, "hxcc executable is unavailable")
        report["error"] = "compiler executable is unavailable"
        return report

    wrapper_identity = _tool_identity(wrapper, host_arch, require_elf=False)
    report["compiler"]["wrapper"] = wrapper_identity
    wrapper_hash_ok = _expected_hash_check(wrapper_identity, expected_wrapper_sha256)
    check("wrapper.identity", bool(wrapper_identity.get("available")), str(wrapper))
    if wrapper_hash_ok is not None:
        check("wrapper.sha256", wrapper_hash_ok, "wrapper hash matches expectation")

    version = _run([str(wrapper), "--version"], timeout=timeout)
    version_text = (version.get("stdout") or version.get("stderr") or "").strip()
    version_line = version_text.splitlines()[0] if version_text else ""
    report["compiler"]["version"] = {
        "returncode": version.get("returncode"),
        "first_line": version_line,
    }
    check(
        "wrapper.version",
        version.get("returncode") == 0 and version_line.startswith("hxcc version "),
        version_line or str(version.get("error", "no version output")),
    )

    help_probe = _run([str(wrapper), "--help"], timeout=timeout)
    clang_help_probe = _run([str(wrapper), "--clang-help"], timeout=timeout)
    report["compiler"]["help"] = {
        "wrapper_help_returncode": help_probe.get("returncode"),
        "clang_help_returncode": clang_help_probe.get("returncode"),
    }
    check(
        "wrapper.help",
        help_probe.get("returncode") == 0
        and clang_help_probe.get("returncode") == 0,
        "hxcc --help and --clang-help completed",
    )

    resource_probe = _run([str(wrapper), "--print-resource-dir"], timeout=timeout)
    resource_text = (resource_probe.get("stdout") or "").strip()
    resource_dir = _resource_dir_from_output(resource_text)
    report["compiler"]["resource_dir"] = str(resource_dir) if resource_dir else None
    check(
        "wrapper.resource_dir",
        resource_probe.get("returncode") == 0 and resource_dir is not None,
        str(resource_dir) if resource_dir else "resource directory unavailable",
    )

    tools = _discover_package_tools(resource_dir)
    identities = {
        name: _tool_identity(path, host_arch, require_elf=True)
        for name, path in tools.items()
    }
    report["compiler"]["underlying"] = identities
    for name, identity, expected in (
        ("clang", identities["clang"], expected_clang_sha256),
        ("rpuas", identities["rpuas"], expected_rpuas_sha256),
        ("rhino_gen_oplib", identities["rhino_gen_oplib"], expected_oplib_sha256),
    ):
        check(f"underlying.{name}.available", bool(identity.get("available")), name)
        check(
            f"underlying.{name}.architecture",
            bool(identity.get("available") and identity.get("host_arch_match")),
            f"ELF arch={identity.get('elf_arch') or 'non-ELF/unknown'}",
        )
        expected_ok = _expected_hash_check(identity, expected)
        if expected_ok is not None:
            check(f"underlying.{name}.sha256", expected_ok, "hash matches expectation")

    with tempfile.TemporaryDirectory(prefix="rhinoforge-hxcc-preflight-") as raw_dir:
        workdir = Path(raw_dir)
        smoke_source = _safe_source(source, workdir)
        dry = _run(
            [str(wrapper), "-###", "-O2", "-c", str(smoke_source)],
            cwd=workdir,
            timeout=timeout,
        )
        dry_text = "\n".join(
            item for item in (dry.get("stdout", ""), dry.get("stderr", "")) if item
        )
        driver = _parse_driver(
            dry_text,
            expected_target,
            expected_tools=tools,
        )
        driver["returncode"] = dry.get("returncode")
        report["driver"] = driver
        check(
            "driver.chain",
            dry.get("returncode") == 0 and bool(driver.get("chain_ok")),
            f"target={driver.get('target_triple')!r} -O2={driver.get('optimization_o2')} "
            f"rpuas={driver.get('assembler_present')} oplib-r1={driver.get('oplib_mode_r1')}",
        )

        if not identity_only:
            compile_probe = _run(
                [str(wrapper), "-O2", "-c", str(smoke_source)],
                cwd=workdir,
                timeout=timeout,
            )
            object_path = workdir / f"{smoke_source.stem}.o"
            ref_path = workdir / f"{smoke_source.stem}.ref"
            outputs: dict[str, Any] = {}
            for label, path in (("object", object_path), ("ref", ref_path)):
                if path.is_file() and not path.is_symlink():
                    try:
                        outputs[label] = {
                            "size": path.stat().st_size,
                            "sha256": _sha256(path),
                        }
                    except (OSError, ValidationError) as error:
                        outputs[label] = {"error": str(error)}
                else:
                    outputs[label] = {"missing": True}
            report["compile_smoke"] = {
                "requested": True,
                "returncode": compile_probe.get("returncode"),
                "outputs": outputs,
                # The REF is generated for the smoke test, but remains in the
                # private TemporaryDirectory and is never published by this
                # receipt.
                "opaque_outputs_private": True,
            }
            compile_ok = bool(
                compile_probe.get("returncode") == 0
                and outputs.get("object", {}).get("size", 0) > 0
                and outputs.get("ref", {}).get("size", 0) > 0
            )
            check(
                "compile.smoke",
                compile_ok,
                "isolated .o and .ref were generated"
                if compile_ok
                else "hxcc did not produce both expected outputs",
            )

    core_statuses = [item["status"] for item in report["checks"]]
    report["ok"] = bool(core_statuses) and all(status == "pass" for status in core_statuses)
    if not report["ok"]:
        report["error"] = "one or more hxcc preflight checks failed"
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify hxcc wrapper/clang/rpuas/rhino_gen_oplib identity and run "
            "an isolated -O2 .rc -> .o/.ref smoke."
        )
    )
    parser.add_argument(
        "--compiler",
        default=os.environ.get("RHINOFORGE_HXCC", "hxcc"),
        help="hxcc executable or PATH name (default: RHINOFORGE_HXCC or hxcc)",
    )
    parser.add_argument("--source", type=Path, help="optional source copied into the private build dir")
    parser.add_argument(
        "--identity-only",
        action="store_true",
        help="skip the end-to-end compile, retaining the driver dry-run checks",
    )
    parser.add_argument(
        "--expected-host-arch",
        choices=("aarch64", "x86_64"),
        help="fail if the running host does not have this normalized architecture",
    )
    parser.add_argument("--expected-target", default=DEFAULT_TARGET)
    parser.add_argument("--expected-wrapper-sha256")
    parser.add_argument("--expected-clang-sha256")
    parser.add_argument("--expected-rpuas-sha256")
    parser.add_argument("--expected-oplib-sha256")
    parser.add_argument(
        "--manual-archive",
        type=Path,
        help="optional local hxcc.zip to hash against the pinned manual reference",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output", type=Path, help="write an atomic JSON receipt")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        report = run_preflight(
            args.compiler,
            source=args.source,
            identity_only=args.identity_only,
            expected_host_arch=args.expected_host_arch,
            expected_target=args.expected_target,
            expected_wrapper_sha256=args.expected_wrapper_sha256,
            expected_clang_sha256=args.expected_clang_sha256,
            expected_rpuas_sha256=args.expected_rpuas_sha256,
            expected_oplib_sha256=args.expected_oplib_sha256,
            manual_archive=args.manual_archive,
            timeout=args.timeout,
        )
    except (OSError, ValidationError, ValueError) as error:
        report = {"schema_version": 1, "ok": False, "error": str(error)}
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if args.output:
        try:
            forbidden_paths: list[Path | None] = [
                args.source,
                args.manual_archive,
                _resolve_executable(args.compiler),
            ]
            compiler_report = report.get("compiler", {})
            if isinstance(compiler_report, dict):
                wrapper_report = compiler_report.get("wrapper", {})
                if isinstance(wrapper_report, dict) and isinstance(
                    wrapper_report.get("path"), str
                ):
                    forbidden_paths.append(Path(wrapper_report["path"]))
                underlying_report = compiler_report.get("underlying", {})
                if isinstance(underlying_report, dict):
                    for identity in underlying_report.values():
                        if isinstance(identity, dict) and isinstance(
                            identity.get("path"), str
                        ):
                            forbidden_paths.append(Path(identity["path"]))
            _output_json(
                args.output,
                report,
                forbidden=forbidden_paths,
            )
        except (OSError, ValidationError) as error:
            print(json.dumps({"schema_version": 1, "ok": False, "error": str(error)}, ensure_ascii=False))
            return 2
    else:
        print(encoded)
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    sys.exit(main())
