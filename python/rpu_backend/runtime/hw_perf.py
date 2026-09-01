"""Process-scoped r4 hardware kernel/DMA trace controls.

The native runtime owns the authoritative state because the enable bit is baked
into each SDK batch.  Every configuration transition invalidates registered
Graphs before returning; callers must therefore change it only between
forwards, preferably before the first forward in a fresh process.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading
from typing import Iterator


_context_lock = threading.RLock()


def _native_api():
    import sys

    package = sys.modules.get("rpu_backend")
    extension = sys.modules.get("rpu_backend._cpp_ext")
    if (
        package is None
        or not bool(getattr(package, "_cpp_loaded", False))
        or extension is None
        or not hasattr(extension, "set_hw_perf_trace")
        or not hasattr(extension, "get_hw_perf_trace")
    ):
        raise RuntimeError(
            "the installed RhinoForge native extension does not provide r4 "
            "hardware tracing; rebuild and reinstall this source tree"
        )
    return extension


def _validated_max_dumps(max_dumps: int) -> int:
    if isinstance(max_dumps, bool) or not isinstance(max_dumps, int):
        raise TypeError("max_dumps must be an integer")
    if max_dumps <= 0:
        raise ValueError("max_dumps must be greater than zero")
    return max_dumps


def _prepare_output_dir(output_dir: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(output_dir)
    except TypeError as exc:
        raise TypeError("output_dir must be a path-like value") from exc
    if not raw:
        raise ValueError("output_dir must be non-empty")
    path = Path(raw).expanduser()
    path = path.absolute()
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise NotADirectoryError(f"hardware trace output is not a directory: {path}")
    return os.fspath(path)


def set_hw_perf_trace(
    enabled: bool,
    output_dir: str | os.PathLike[str] = "",
    max_dumps: int = 32,
) -> None:
    """Enable or disable r4 hardware trace collection between forwards.

    Enabling creates ``output_dir`` and starts a fresh bounded dump session.
    Changing any setting invalidates all registered Graph batches.  Disabling
    ignores ``output_dir`` and ``max_dumps`` and removes the instrumentation
    from subsequently rebuilt batches.
    """
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a bool")
    extension = _native_api()
    if not enabled:
        extension.set_hw_perf_trace(False, "", 0)
        return
    limit = _validated_max_dumps(max_dumps)
    output = _prepare_output_dir(output_dir)
    extension.set_hw_perf_trace(True, output, limit)


def get_hw_perf_trace() -> dict[str, object]:
    """Return enabled state, output directory, dump limit, and dump count."""
    raw = dict(_native_api().get_hw_perf_trace())
    return {
        "enabled": bool(raw["enabled"]),
        "output_dir": str(raw["output_dir"]),
        "max_dumps": int(raw["max_dumps"]),
        "dump_count": int(raw["dump_count"]),
    }


@contextmanager
def hw_perf_trace(
    output_dir: str | os.PathLike[str],
    max_dumps: int = 32,
) -> Iterator[dict[str, object]]:
    """Collect bounded r4 Chrome traces and restore the prior configuration."""
    with _context_lock:
        previous = get_hw_perf_trace()
        set_hw_perf_trace(True, output_dir, max_dumps)
        try:
            yield get_hw_perf_trace()
        finally:
            if bool(previous["enabled"]):
                set_hw_perf_trace(
                    True,
                    str(previous["output_dir"]),
                    int(previous["max_dumps"]),
                )
            else:
                set_hw_perf_trace(False)


__all__ = ["set_hw_perf_trace", "get_hw_perf_trace", "hw_perf_trace"]
