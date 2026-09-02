#!/usr/bin/env python3
"""Inspect hxcc assembly/source for the manual's performance invariants.

This is a structural gate, not a cycle model.  It records counts and hashes
only, so generated assembly can remain outside the candidate repository.  The
compiler is allowed to silently lower an invalid ``rpu_hwloop`` to ``wjump``;
therefore a campaign should run this after an isolated ``-save-temps`` build
and use ``--strict`` (or explicit profile requirements) before timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable


MAX_INPUT_BYTES = 256 * 1024 * 1024
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
ASYNC_FAMILIES = ("vld", "vst", "vmat", "valu", "vsfu")


class ValidationError(ValueError):
    """Assembly/source evidence is malformed or violates a selected gate."""


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute spelling without resolving links or ``..``."""

    value = os.path.expanduser(os.fspath(path))
    if not os.path.isabs(value):
        value = os.path.join(os.getcwd(), value)
    return Path(value)


def _reject_symlink_components(path: Path, label: str) -> None:
    """Reject links in an existing lexical prefix before path normalization."""

    current = Path(path.anchor) if path.anchor else Path(".")
    parts = path.parts[1:] if path.anchor else path.parts
    missing = False
    for component_index, part in enumerate(parts):
        if part in {"", "."}:
            continue
        if missing:
            # Do not let abspath turn an unavailable/.. spelling into a
            # different existing path.  Ordinary missing final components are
            # left to the caller's regular unavailable diagnostic.
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
            missing = True
            continue
        except OSError as error:
            raise ValidationError(f"{label} path cannot be inspected") from error
        # Let the final entry be diagnosed by lstat/open below, preserving the
        # established "regular non-symlink" error; parent links are rejected
        # here before abspath can hide them.
        if stat.S_ISLNK(info.st_mode) and component_index != len(parts) - 1:
            raise ValidationError(f"{label} path must not contain symbolic links")


def _read_regular(path: Path, label: str) -> bytes:
    lexical = _lexical_absolute(path)
    _reject_symlink_components(lexical, label)
    absolute = _absolute_path(lexical)
    try:
        info = os.lstat(absolute)
    except OSError as error:
        raise ValidationError(f"{label} is unavailable") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise ValidationError(f"{label} must be a regular non-symlink file")
    if info.st_size > MAX_INPUT_BYTES:
        raise ValidationError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(absolute, flags)
    except OSError as error:
        raise ValidationError(f"{label} cannot be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        initial_identity = (
            info.st_dev,
            info.st_ino,
            info.st_size,
            info.st_mtime_ns,
            info.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino)
            or opened.st_size != info.st_size
            or opened.st_mtime_ns != info.st_mtime_ns
            or opened.st_ctime_ns != info.st_ctime_ns
        ):
            raise ValidationError(f"{label} changed before reading")
        proc_fd = f"/proc/self/fd/{descriptor}"
        if os.path.lexists(proc_fd):
            try:
                fd_target = os.path.realpath(proc_fd)
            except OSError:
                fd_target = ""
            if fd_target and fd_target != os.path.realpath(os.fspath(absolute)):
                raise ValidationError(f"{label} path changed before reading")
        chunks: list[bytes] = []
        observed = 0
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            observed += len(block)
            if observed > MAX_INPUT_BYTES:
                raise ValidationError(f"{label} exceeds {MAX_INPUT_BYTES} bytes")
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
            raise ValidationError(f"{label} changed while reading")
        return b"".join(chunks)
    except OSError as error:
        raise ValidationError(f"{label} is unreadable") from error
    finally:
        os.close(descriptor)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _strip_comments_and_literals(source: str) -> str:
    """Blank comments and string/character literals while preserving lines.

    Regex-only comment stripping can mistake ``"//"`` for a comment and can
    report words such as ``"new"`` as language features.  This small lexer is
    intentionally not a C++ parser; it only makes the review checks
    conservative and location-stable.
    """

    result: list[str] = []
    state = "normal"
    escaped = False
    index = 0
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if state == "normal":
            if char == "/" and next_char == "/":
                result.extend((" ", " "))
                index += 2
                state = "line_comment"
                continue
            if char == "/" and next_char == "*":
                result.extend((" ", " "))
                index += 2
                state = "block_comment"
                continue
            if char in {'"', "'"}:
                result.append(" ")
                state = "string" if char == '"' else "char"
                escaped = False
                index += 1
                continue
            result.append(char)
            index += 1
            continue
        if state == "line_comment":
            if char == "\n":
                result.append(char)
                state = "normal"
            else:
                result.append(" ")
            index += 1
            continue
        if state == "block_comment":
            if char == "*" and next_char == "/":
                result.extend((" ", " "))
                index += 2
                state = "normal"
            else:
                result.append("\n" if char == "\n" else " ")
                index += 1
            continue
        # string or character literal
        if char == "\n" and state == "string":
            # An unterminated literal is still blanked; the compiler will
            # report the syntax error separately.
            result.append(char)
            state = "normal"
            escaped = False
            index += 1
            continue
        result.append("\n" if char == "\n" else " ")
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif (state == "string" and char == '"') or (
            state == "char" and char == "'"
        ):
            state = "normal"
        index += 1
    return "".join(result)


def _matching_brace(text: str, opening: int) -> int | None:
    depth = 0
    for index in range(opening, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return index
    return None


def lint_source(source_text: str, *, require_entry: bool = False) -> dict[str, Any]:
    """Return clean-room source restrictions and pragma adjacency metrics."""

    if not isinstance(source_text, str):
        raise ValidationError("source text must be a string")
    text = _strip_comments_and_literals(source_text)
    violations: list[dict[str, Any]] = []

    forbidden = {
        "dynamic_allocation": r"\b(?:new|delete|malloc|calloc|realloc|free)\b",
        "exceptions": r"\b(?:try|catch|throw)\b",
        "virtual_dispatch": r"\bvirtual\b|\bvtable\b|\bvptr\b",
        "stl": r"\bstd\s*::",
        "recursion_hint": r"\b(?:recurse|recursive)\b",
        # Function pointers/indirect calls are not part of the device ABI.
        # The declaration form is intentionally conservative and catches
        # typedefs such as ``void (*fn)(...)`` without flagging ordinary data
        # pointers.
        "function_pointer": r"\(\s*[*&]{1,2}\s*[A-Za-z_]\w*\s*\)",
    }
    for name, pattern in forbidden.items():
        matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))
        if matches:
            violations.append(
                {"kind": name, "count": len(matches), "source_offset": matches[0].start()}
            )

    # A direct self-call is a useful conservative recursion hint.  We only
    # inspect named __rprog__ entries and do not attempt to parse C++ fully.
    for match in re.finditer(
        r"__rprog__\s+[^;{}]+?\b([A-Za-z_]\w*)\s*\([^;{}]*\)\s*\{",
        text,
        flags=re.DOTALL,
    ):
        name = match.group(1)
        body_start = match.end()
        opening = match.end() - 1
        body_end = _matching_brace(text, opening)
        body = text[match.end() :] if body_end is None else text[match.end() : body_end]
        if re.search(rf"\b{re.escape(name)}\s*\(", body):
            violations.append(
                {"kind": "recursive_entry_call", "function": name, "source_offset": body_start}
            )

    # The guide requires a void __rprog__ entry.  A missing match is reported
    # as an informational failure only when the source appears to contain an
    # entry marker; ordinary headers may legitimately define none.
    entry_markers = len(re.findall(r"\b__rprog__\b", text))
    nonvoid_entries = len(
        re.findall(
            r"__rprog__\s+(?!void\b)(?:[A-Za-z_][\w:<>*&\s]*)\s+[A-Za-z_]\w*\s*\(",
            text,
        )
    )
    if nonvoid_entries:
        violations.append({"kind": "nonvoid_rprog_entry", "count": nonvoid_entries})
    if require_entry and entry_markers == 0:
        violations.append({"kind": "missing_rprog_entry"})

    local_scalar_count = len(
        re.findall(
            r"\b__local__\s+(?:bool|char|short|int|long|float|double|half|bfloat16)\b(?!\s*\*)",
            text,
            flags=re.IGNORECASE,
        )
    )
    if local_scalar_count:
        violations.append(
            {"kind": "local_scalar_address_space", "count": local_scalar_count}
        )

    lines = text.splitlines()
    pragma_count = 0
    detached_pragmas = 0
    int_loop_count = 0
    for index, line in enumerate(lines):
        if re.search(r"#\s*pragma\s+rpu_hwloop_[A-Za-z0-9_]+", line):
            pragma_count += 1
            following = ""
            for next_line in lines[index + 1 :]:
                if next_line.strip():
                    following = next_line.strip()
                    break
            if not re.match(r"for\s*\(", following):
                detached_pragmas += 1
        if re.search(
            r"\bfor\s*\(\s*(?:(?:signed|unsigned)\s+)?int\s+[A-Za-z_]",
            line,
        ):
            int_loop_count += 1
    if detached_pragmas:
        violations.append(
            {"kind": "hwloop_pragma_not_immediately_before_for", "count": detached_pragmas}
        )

    return {
        "entry_marker_count": entry_markers,
        "hwloop_pragma_count": pragma_count,
        "detached_hwloop_pragma_count": detached_pragmas,
        "int_loop_count": int_loop_count,
        "violations": violations,
        "source_ok": not violations,
    }


def _instruction_records(assembly_text: str) -> list[tuple[str, int]]:
    result: list[tuple[str, int]] = []
    for raw_line in assembly_text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or line.endswith(":") or line.startswith("."):
            continue
        # GNU-style assembly has an optional label followed by whitespace.
        if ":" in line.split()[0]:
            line = line.split(":", 1)[1].strip()
        if not line:
            continue
        mnemonic = line.split()[0].lower()
        if mnemonic.startswith("."):
            continue
        result.append((mnemonic, len(result)))
    return result


def _instruction_mnemonics(assembly_text: str) -> list[str]:
    return [mnemonic for mnemonic, _ in _instruction_records(assembly_text)]


def _async_source_counts(source_text: str) -> dict[str, int]:
    clean = _strip_comments_and_literals(source_text)
    return {
        family: len(
            re.findall(
                rf"\b{family}[A-Za-z0-9_]*_async\b", clean, flags=re.IGNORECASE
            )
        )
        for family in ASYNC_FAMILIES
    }


def _fence_family(mnemonic: str) -> str | None:
    match = re.match(r"^fence[._](vld|vst|vmat|valu|vsfu)(?:[._]|$)", mnemonic)
    return match.group(1) if match else None


def _is_execute_mnemonic(mnemonic: str, family: str) -> bool:
    # *_m/*_p are configuration/setup instructions.  Require an execute or
    # Repeat form before treating a unit fence as evidence of an async use.
    return bool(re.match(rf"^{family}_(?:e|r|x|xr)(?:[._]|$)", mnemonic))


def inspect_assembly(
    assembly_text: str,
    *,
    source_text: str | None = None,
    strict: bool = False,
    max_wjump: int | None = None,
    require_vmat: bool = False,
    require_lpaddr: bool = False,
    require_async_fence: bool = False,
    max_loop_depth: int = 8,
    require_entry: bool = False,
) -> dict[str, Any]:
    if not isinstance(assembly_text, str):
        raise ValidationError("assembly text must be a string")
    # Keep the programmatic API as strict as the CLI's ``type=int`` parser.
    # In particular, bool is an int subclass and NaN/inf comparisons can
    # otherwise bypass the depth/wjump gates (``count > nan`` is false).
    if type(max_loop_depth) is not int or max_loop_depth < 1:
        raise ValidationError("max_loop_depth must be a positive integer")
    if max_wjump is not None and type(max_wjump) is not int:
        raise ValidationError("max_wjump must be an integer or None")
    if max_wjump is not None and max_wjump < 0:
        raise ValidationError("max_wjump must be non-negative")
    records = _instruction_records(assembly_text)
    mnemonics = [mnemonic for mnemonic, _ in records]
    counts = {
        "hardware_loop": sum(bool(re.fullmatch(r"loop(?:\..*)?", item)) for item in mnemonics),
        "software_wjump": sum(item == "wjump" for item in mnemonics),
        "repeat": sum(
            bool(re.match(r"^(?:vld|vst|vmat|valu|vsfu)_r(?:[._]|$)", item))
            for item in mnemonics
        ),
        "lpaddr": sum("lpaddr" in item for item in mnemonics),
        "vld": sum(item.startswith("vld_") for item in mnemonics),
        "vst": sum(item.startswith("vst_") for item in mnemonics),
        "vmat": sum(item.startswith("vmat_") for item in mnemonics),
        # ``vmat_m``/``vmat_p`` are setup/configuration instructions.  A GEMM
        # gate must see an execute/repeat form, not merely compiler prologue
        # initialization.
        "vmat_compute": sum(
            bool(re.match(r"^vmat_(?:e|r|x|xr)(?:[._]|$)", item))
            for item in mnemonics
        ),
        "valu": sum(item.startswith("valu_") for item in mnemonics),
        "vsfu": sum(item.startswith("vsfu_") for item in mnemonics),
        "fence": sum(item.startswith("fence") for item in mnemonics),
        "tail_strobe": sum("strb" in item or "strobe" in item for item in mnemonics),
    }
    depths = [
        int(value)
        for value in re.findall(r"\bDepth\s*=\s*(\d+)", assembly_text, flags=re.IGNORECASE)
    ]
    max_depth = max(depths, default=0)
    violations: list[dict[str, Any]] = []
    if not mnemonics:
        violations.append({"kind": "empty_assembly"})
    effective_max_wjump = 0 if strict and max_wjump is None else max_wjump
    if effective_max_wjump is not None and counts["software_wjump"] > effective_max_wjump:
        violations.append(
            {
                "kind": "software_loop_wjump",
                "count": counts["software_wjump"],
                "allowed": effective_max_wjump,
            }
        )
    if max_depth > max_loop_depth:
        violations.append(
            {"kind": "hardware_loop_depth", "observed": max_depth, "allowed": max_loop_depth}
        )
    if require_vmat and counts["vmat_compute"] == 0:
        violations.append({"kind": "missing_vmat"})
    if require_lpaddr and counts["lpaddr"] == 0:
        violations.append({"kind": "missing_lpaddr"})

    source_report = None
    if source_text is not None:
        source_report = lint_source(source_text, require_entry=require_entry)
        violations.extend(source_report["violations"])
        async_counts = _async_source_counts(source_text)
        counts["source_async"] = sum(async_counts.values())
        fence_counts = {
            family: sum(
                1
                for mnemonic, _ in records
                if _fence_family(mnemonic) == family
            )
            for family in ASYNC_FAMILIES
        }
        counts["fence_by_unit"] = fence_counts
        async_fence_violations: list[dict[str, Any]] = []
        for family, async_count in async_counts.items():
            if async_count == 0:
                continue
            matching_fences = [
                index
                for mnemonic, index in records
                if _fence_family(mnemonic) == family
            ]
            # A unit fence must exist after at least one instruction from the
            # same unit.  This catches a stray fence.vld being used to cover a
            # VMAT/VALU async operation while remaining tolerant of loop
            # hoisting (one fence can cover a repeated operation).
            unit_instructions = [
                index
                for mnemonic, index in records
                if _is_execute_mnemonic(mnemonic, family)
            ]
            if require_async_fence and (
                not matching_fences
                or not unit_instructions
                or max(matching_fences) <= max(unit_instructions)
            ):
                async_fence_violations.append(
                    {
                        "kind": "async_unit_without_matching_fence",
                        "unit": family,
                        "async_operations": async_count,
                        "matching_fences": len(matching_fences),
                        "execute_instructions": len(unit_instructions),
                    }
                )
        violations.extend(async_fence_violations)
    elif require_async_fence:
        # Without source, only an explicit fence count can be checked.  Keep
        # this a hard opt-in so ordinary assembly inspection remains useful.
        if counts["fence"] == 0:
            violations.append({"kind": "fence_evidence_missing_source_not_supplied"})

    return {
        "schema_version": 1,
        "assembly_sha256": _sha256(assembly_text.encode("utf-8")),
        "assembly_bytes": len(assembly_text.encode("utf-8")),
        "counts": counts,
        "instruction_count": len(mnemonics),
        "max_loop_depth": max_depth,
        "source": source_report,
        "violations": violations,
        "ok": not violations,
    }


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(os.fspath(path))))


def _safe_destination(path: Path, forbidden: Iterable[Path | None] = ()) -> Path:
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
        if candidate is None:
            continue
        candidate_lexical = _lexical_absolute(candidate)
        _reject_symlink_components(candidate_lexical, "input")
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
    path: Path, report: dict[str, Any], *, forbidden: Iterable[Path | None] = ()
) -> None:
    destination = _safe_destination(path, forbidden)
    payload = (
        json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    if len(payload) > MAX_OUTPUT_BYTES:
        raise ValidationError("summary output is too large")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Count RPU hardware-loop/Repeat/VMAT/lpaddr/fence evidence without publishing assembly."
    )
    parser.add_argument("assembly", type=Path)
    parser.add_argument("--source", type=Path, help="optional matching .rc source for async/ABI lint")
    parser.add_argument("--strict", action="store_true", help="reject any software wjump")
    parser.add_argument("--max-wjump", type=int)
    parser.add_argument("--require-vmat", action="store_true")
    parser.add_argument("--require-lpaddr", action="store_true")
    parser.add_argument("--require-async-fence", action="store_true")
    parser.add_argument(
        "--require-entry",
        action="store_true",
        help="require at least one __rprog__ entry in the supplied source",
    )
    parser.add_argument("--max-loop-depth", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(list(argv) if argv is not None else None)
    try:
        assembly_bytes = _read_regular(args.assembly, "assembly")
        try:
            assembly_text = assembly_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValidationError("assembly is not UTF-8 text") from error
        source_text = None
        if args.source is not None:
            source_bytes = _read_regular(args.source, "source")
            try:
                source_text = source_bytes.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValidationError("source is not UTF-8 text") from error
        report = inspect_assembly(
            assembly_text,
            source_text=source_text,
            strict=args.strict,
            max_wjump=args.max_wjump,
            require_vmat=args.require_vmat,
            require_lpaddr=args.require_lpaddr,
            require_async_fence=args.require_async_fence,
            max_loop_depth=args.max_loop_depth,
            require_entry=args.require_entry,
        )
    except (OSError, ValidationError, ValueError, TypeError, OverflowError) as error:
        report = {"schema_version": 1, "ok": False, "error": str(error)}
    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if args.output:
        try:
            _atomic_write(args.output, report, forbidden=(args.assembly, args.source))
        except (OSError, ValidationError, TypeError, OverflowError) as error:
            print(json.dumps({"schema_version": 1, "ok": False, "error": str(error)}, ensure_ascii=False))
            return 2
    else:
        print(encoded)
    return 0 if report.get("ok") is True else 2


if __name__ == "__main__":
    sys.exit(main())
