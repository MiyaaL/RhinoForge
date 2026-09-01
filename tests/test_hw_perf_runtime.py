from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "python/rpu_backend/runtime/hw_perf.py"


@pytest.fixture
def hw_perf_module(monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("_rhinoforge_hw_perf_test", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Extension:
        def __init__(self) -> None:
            self.state = {
                "enabled": False,
                "output_dir": "",
                "max_dumps": 0,
                "dump_count": 0,
            }
            self.calls: list[tuple[bool, str, int]] = []

        def set_hw_perf_trace(
            self, enabled: bool, output_dir: str, max_dumps: int
        ) -> None:
            self.calls.append((enabled, output_dir, max_dumps))
            self.state = {
                "enabled": enabled,
                "output_dir": output_dir,
                "max_dumps": max_dumps,
                "dump_count": 0,
            }

        def get_hw_perf_trace(self) -> dict[str, object]:
            return dict(self.state)

    extension = Extension()
    monkeypatch.setitem(sys.modules, "rpu_backend", SimpleNamespace(_cpp_loaded=True))
    monkeypatch.setitem(sys.modules, "rpu_backend._cpp_ext", extension)
    return module, extension


def test_hw_perf_context_creates_directory_and_restores_disabled_state(
    hw_perf_module, tmp_path: Path
) -> None:
    module, extension = hw_perf_module
    output = tmp_path / "nested" / "trace"

    with module.hw_perf_trace(output, max_dumps=7) as active:
        assert output.is_dir()
        assert active == {
            "enabled": True,
            "output_dir": str(output),
            "max_dumps": 7,
            "dump_count": 0,
        }

    assert extension.calls == [
        (True, str(output), 7),
        (False, "", 0),
    ]
    assert module.get_hw_perf_trace()["enabled"] is False


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "2"])
def test_hw_perf_rejects_invalid_dump_limits_before_creating_output(
    hw_perf_module, tmp_path: Path, value
) -> None:
    module, extension = hw_perf_module
    output = tmp_path / "must-not-exist"

    with pytest.raises((TypeError, ValueError)):
        module.set_hw_perf_trace(True, output, value)

    assert not output.exists()
    assert extension.calls == []


def test_hw_perf_disable_canonicalizes_unused_arguments(hw_perf_module) -> None:
    module, extension = hw_perf_module

    module.set_hw_perf_trace(False, "ignored", -1)

    assert extension.calls == [(False, "", 0)]
