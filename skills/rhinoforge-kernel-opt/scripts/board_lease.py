#!/usr/bin/env python3
"""Run one command while holding the process-global RhinoForge RPU lease."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CANONICAL_LOCK_PATH = Path("/tmp/rhinoforge-rpu-board-v3.lock")
GUARD_DIRECTORY = Path("/tmp")
MAX_JOURNAL_BYTES = 8 * 1024 * 1024
ZERO_SHA256 = "0" * 64


class LeaseIntegrityError(ValueError):
    """The cooperative lease identity or journal is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LeaseIntegrityError(f"duplicate journal key: {key}")
        result[key] = value
    return result


def _record_hash(record_without_hash: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(record_without_hash).encode("utf-8")).hexdigest()


def _validate_journal(fd: int) -> tuple[str, int]:
    file_stat = os.fstat(fd)
    if file_stat.st_size > MAX_JOURNAL_BYTES:
        raise LeaseIntegrityError("board lease journal exceeds 8 MiB")
    os.lseek(fd, 0, os.SEEK_SET)
    raw = b""
    remaining = file_stat.st_size
    while remaining:
        block = os.read(fd, min(remaining, 1024 * 1024))
        if not block:
            raise LeaseIntegrityError("board lease journal was truncated while reading")
        raw += block
        remaining -= len(block)
    if not raw:
        return ZERO_SHA256, 0
    if not raw.endswith(b"\n"):
        raise LeaseIntegrityError("board lease journal has a partial final record")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise LeaseIntegrityError("board lease journal is not UTF-8") from error
    previous = ZERO_SHA256
    active_lease_id: str | None = None
    record_count = 0
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line:
            raise LeaseIntegrityError(f"blank board lease journal line {line_number}")
        try:
            record = json.loads(line, object_pairs_hook=_unique_object)
        except (json.JSONDecodeError, RecursionError) as error:
            raise LeaseIntegrityError(
                f"invalid board lease journal line {line_number}"
            ) from error
        if not isinstance(record, dict):
            raise LeaseIntegrityError("board lease journal record is not an object")
        if record.get("schema_version") != 3 or record.get("sequence") != line_number:
            raise LeaseIntegrityError("board lease journal sequence is invalid")
        event = record.get("event")
        lease_id = record.get("lease_id")
        if event == "lease_started" and active_lease_id is None:
            active_lease_id = lease_id
        elif event == "lease_finished" and active_lease_id == lease_id:
            active_lease_id = None
        else:
            raise LeaseIntegrityError("board lease journal lease pairing is invalid")
        claimed_hash = record.pop("record_sha256", None)
        if record.get("prev_record_sha256") != previous:
            raise LeaseIntegrityError("board lease journal hash chain is broken")
        expected_hash = _record_hash(record)
        if claimed_hash != expected_hash:
            raise LeaseIntegrityError("board lease journal record hash is invalid")
        previous = expected_hash
        record_count = line_number
    if active_lease_id is not None:
        raise LeaseIntegrityError("board lease journal ends with an incomplete lease")
    return previous, record_count


def _append_record(fd: int, record: dict[str, Any], previous_hash: str) -> str:
    value = dict(record)
    value["prev_record_sha256"] = previous_hash
    record_hash = _record_hash(value)
    value["record_sha256"] = record_hash
    payload = (_canonical_json(value) + "\n").encode("utf-8")
    if os.write(fd, payload) != len(payload):
        raise OSError("short board lease journal append")
    os.fsync(fd)
    return record_hash


def _canonical_lock_path(lock_path: Path) -> Path:
    supplied = Path(lock_path)
    if supplied != CANONICAL_LOCK_PATH:
        raise ValueError(
            f"board lock must be exactly {CANONICAL_LOCK_PATH}; alternate locks are forbidden"
        )
    if supplied.is_symlink():
        raise LeaseIntegrityError("canonical board lease journal must not be a symlink")
    return supplied


def _open_journal(path: Path) -> tuple[int, os.stat_result]:
    flags = os.O_RDWR | os.O_CREAT | os.O_APPEND
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags, 0o600)
    try:
        file_stat = os.fstat(fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise LeaseIntegrityError("canonical board lease journal is not regular")
        if file_stat.st_uid != os.geteuid():
            raise LeaseIntegrityError("canonical board lease journal owner mismatches")
        if stat.S_IMODE(file_stat.st_mode) != 0o600:
            raise LeaseIntegrityError("canonical board lease journal mode must be 0600")
        if file_stat.st_nlink != 1:
            raise LeaseIntegrityError("canonical board lease journal link count must be one")
        return fd, file_stat
    except Exception:
        os.close(fd)
        raise


def _acquire_guard(timeout_s: float) -> int:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    guard = os.open(GUARD_DIRECTORY, flags)
    guard_stat = os.fstat(guard)
    if not stat.S_ISDIR(guard_stat.st_mode) or guard_stat.st_nlink < 1:
        os.close(guard)
        raise LeaseIntegrityError("canonical guard directory identity is invalid")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return guard
        except BlockingIOError:
            if time.monotonic() >= deadline:
                os.close(guard)
                raise TimeoutError("timed out waiting for the RPU board lease")
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))


class BoardLease:
    """Process-global directory guard plus a consistency-checked journal."""

    def __init__(self, timeout_s: float, command: list[str]) -> None:
        if not command:
            raise ValueError("a command identity is required")
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(float(timeout_s))
            or timeout_s < 0
        ):
            raise ValueError("timeout_s must be finite and non-negative")
        self.timeout_s = float(timeout_s)
        self.command = list(command)
        self.lease_id = str(uuid.uuid4())
        self.guard = -1
        self.fd = -1
        self.identity: tuple[int, int] | None = None
        self.head_hash = ZERO_SHA256
        self.previous_head_hash = ZERO_SHA256
        self.start_record_sha256 = ZERO_SHA256
        self.finish_record_sha256 = ZERO_SHA256
        self.start_sequence = 0
        self.finish_sequence = 0
        self.started_monotonic_ns = 0
        self.finished_monotonic_ns = 0
        self.finished = False

    def __enter__(self) -> "BoardLease":
        self.guard = _acquire_guard(self.timeout_s)
        try:
            self.fd, file_stat = _open_journal(CANONICAL_LOCK_PATH)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            self.identity = (file_stat.st_dev, file_stat.st_ino)
            self.check_integrity()
            self.head_hash, record_count = _validate_journal(self.fd)
            self.previous_head_hash = self.head_hash
            self.start_sequence = record_count + 1
            self.started_monotonic_ns = time.monotonic_ns()
            started = {
                "schema_version": 3,
                "sequence": self.start_sequence,
                "event": "lease_started",
                "lease_id": self.lease_id,
                "pid": os.getpid(),
                "started_at": datetime.now(timezone.utc).isoformat(),
                "started_monotonic_ns": self.started_monotonic_ns,
                "command_executable": Path(self.command[0]).name,
                "command_argv_sha256": hashlib.sha256(
                    _canonical_json(self.command).encode("utf-8")
                ).hexdigest(),
            }
            self.head_hash = _append_record(self.fd, started, self.head_hash)
            self.start_record_sha256 = self.head_hash
            return self
        except Exception:
            self.close()
            raise

    def check_integrity(self) -> None:
        if self.fd < 0 or self.identity is None:
            raise LeaseIntegrityError("board lease is not active")
        opened = os.fstat(self.fd)
        try:
            current = os.lstat(CANONICAL_LOCK_PATH)
        except OSError as error:
            raise LeaseIntegrityError("canonical board lease pathname disappeared") from error
        if (
            (opened.st_dev, opened.st_ino) != self.identity
            or (current.st_dev, current.st_ino) != self.identity
            or opened.st_nlink != 1
            or current.st_nlink != 1
            or current.st_uid != os.geteuid()
            or stat.S_IMODE(current.st_mode) != 0o600
            or not stat.S_ISREG(current.st_mode)
        ):
            raise LeaseIntegrityError("canonical board lease pathname/inode was replaced")

    def finish(self, exit_code: int) -> None:
        if self.finished:
            raise LeaseIntegrityError("board lease was already finished")
        self.check_integrity()
        self.finish_sequence = self.start_sequence + 1
        self.finished_monotonic_ns = time.monotonic_ns()
        finished = {
            "schema_version": 3,
            "sequence": self.finish_sequence,
            "event": "lease_finished",
            "lease_id": self.lease_id,
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "finished_monotonic_ns": self.finished_monotonic_ns,
            "exit_code": int(exit_code),
        }
        self.head_hash = _append_record(self.fd, finished, self.head_hash)
        self.finish_record_sha256 = self.head_hash
        self.finished = True

    def close(self) -> None:
        if self.fd >= 0:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = -1
        if self.guard >= 0:
            try:
                fcntl.flock(self.guard, fcntl.LOCK_UN)
            finally:
                os.close(self.guard)
                self.guard = -1

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.fd >= 0 and not self.finished:
            try:
                self.finish(0 if exc_type is None else 2)
            except Exception:
                if exc_type is None:
                    self.close()
                    raise
        self.close()


def run_with_lease(lock_path: Path, timeout_s: float, command: list[str]) -> int:
    _canonical_lock_path(lock_path)
    if not command:
        raise ValueError("a command is required after --")
    with BoardLease(timeout_s, command) as lease:
        print(
            _canonical_json(
                {
                    "schema_version": 3,
                    "lease_id": lease.lease_id,
                    "journal": CANONICAL_LOCK_PATH.name,
                    "journal_head_sha256": lease.head_hash,
                }
            ),
            flush=True,
        )
        try:
            child = subprocess.Popen(command)
        except OSError:
            lease.finish(127)
            return 127
        try:
            while child.poll() is None:
                time.sleep(0.05)
                lease.check_integrity()
            exit_code = int(child.returncode)
            lease.finish(exit_code)
            return exit_code
        except KeyboardInterrupt:
            child.terminate()
            child.wait()
            lease.finish(130)
            return 130
        except LeaseIntegrityError:
            child.terminate()
            try:
                child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=CANONICAL_LOCK_PATH,
        help=f"canonical journal path; must be exactly {CANONICAL_LOCK_PATH}",
    )
    parser.add_argument("--timeout", type=float, default=0.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    try:
        return run_with_lease(args.lock, args.timeout, command)
    except (LeaseIntegrityError, OSError, TimeoutError, ValueError) as error:
        print(f"board lease failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
