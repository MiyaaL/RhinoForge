"""Board-free entry-budget gates; a larger budget is not board parity proof."""

import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def budget_probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("segment-budget")
    source = directory / "probe.cpp"
    source.write_text('''
#include "graph/graph_segment_budget.h"
#include <iostream>
int main(int argc, char** argv) {
    try {
        if (argc > 2) {
            const bool command = std::string(argv[2]) == "command";
            std::cout << rpu_graph_detail::parse_segment_limit(
                argv[1], command ? "RPU_GRAPH_MAX_SEGMENT_COMMAND_MB"
                                 : "RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB",
                command ? 4 : 32, command ? 8 : 64);
            return 0;
        }
        const auto n = rpu_graph_detail::parse_segment_entry_budget(
            argc > 1 ? argv[1] : nullptr);
        std::cout << n << " "
                  << rpu_graph_detail::effective_segment_entry_budget(n, 65536)
                  << " " << rpu_graph_detail::effective_segment_entry_budget(n, 4096);
    } catch (const std::exception& e) {
        std::cerr << e.what();
        return 2;
    }
}
''')
    executable = directory / "probe"
    subprocess.run([compiler, "-std=c++17", "-I", str(ROOT / "src"),
                    str(source), "-o", str(executable)], check=True)
    return executable


@pytest.mark.parametrize("value,expected", [
    (None, "8192 8192 2048"), ("32768", "32768 32768 2048"),
    ("8192", "8192 8192 2048"), ("1", "1 1 1"),
])
def test_entry_override_preserves_sdk_headroom(budget_probe, value, expected):
    command = [str(budget_probe)] + ([] if value is None else [value])
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    assert result.stdout == expected


@pytest.mark.parametrize("value", ["", "0", "-1", "+1", " 8192", "8192x",
                                   "32769", "65536", "999999999999999999999"])
def test_invalid_budget_is_not_unlimited(budget_probe, value):
    result = subprocess.run([str(budget_probe), value], capture_output=True, text=True)
    assert result.returncode == 2
    assert "RPU_GRAPH_MAX_SEGMENT_ENTRIES" in result.stderr


def test_other_resource_and_fence_guards_remain():
    source = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    assert 'std::getenv("RPU_GRAPH_MAX_SEGMENT_ENTRIES")' in source
    assert '"RPU_GRAPH_MAX_SEGMENT_COMMAND_MB", 4, 8' in source
    assert '"RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB", 32, 64' in source
    assert "requested_kd_mb * kMiB, sdk_kd / 2" in source
    assert "requested_instr_mb * kMiB, sdk_instr / 2" in source
    assert "pending_streams.empty()" in source
    assert "seg_queue_state != kn.queue_state" in source
    assert "query_launch_footprint" in source
    assert "seg.entry_count = seg_resources.entries" in source
    wrapper = (ROOT / "run_wall_qwen35_openloop.sh").read_text()
    assert 'WALL_QWEN35_OPT="${WALL_QWEN35_OPT-1}"' in wrapper
    assert '--language-one-graph)' not in wrapper
    assert 'GRAPH_MAX_SEGMENT_ENTRIES=32768' in wrapper
    assert '"RPU_GRAPH_MAX_SEGMENT_ENTRIES=$GRAPH_MAX_SEGMENT_ENTRIES"' in wrapper


@pytest.mark.parametrize("kind,value,expected", [
    ("command", "4", "4"), ("command", "8", "8"),
    ("instruction", "32", "32"), ("instruction", "64", "64"),
])
def test_bounded_memory_override(budget_probe, kind, value, expected):
    result = subprocess.run([str(budget_probe), value, kind],
                            capture_output=True, text=True, check=True)
    assert result.stdout == expected


@pytest.mark.parametrize("kind,value", [
    ("command", "9"), ("instruction", "65"),
    ("command", "0"), ("instruction", "0"),
    ("command", ""), ("instruction", "32MB"),
])
def test_bad_memory_budget_rejected(budget_probe, kind, value):
    result = subprocess.run([str(budget_probe), value, kind],
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert "RPU_GRAPH_MAX_SEGMENT_" in result.stderr


@pytest.mark.parametrize("name,invalid", [
    ("RPU_GRAPH_MAX_SEGMENT_COMMAND_MB", "9"),
    ("RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB", "65"),
    ("LKN_MAX_BATCH_ENTRIES", "4194305"),
    ("LKN_KD_BUF_MB", "257"), ("LKN_INSTR_BUF_MB", "1025"),
    ("RPU_GRAPH_MAX_SEGMENT_COMMAND_MB", ""),
    ("RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB", "0"),
    ("LKN_MAX_BATCH_ENTRIES", "0"), ("LKN_KD_BUF_MB", "16MB"),
    ("LKN_INSTR_BUF_MB", "-128"),
])
def test_unified_wrapper_overrides_individual_resource_controls(name, invalid):
    result = subprocess.run(
        ["bash", str(ROOT / "run_wall_qwen35_openloop.sh"), "--check"],
        env={**os.environ, name: invalid, "ENV_SH": "/must-not-be-read"},
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "environment script is missing" in result.stderr
    assert "must be an integer" not in result.stderr


def test_wrapper_forwards_both_soft_and_sdk_limits():
    wrapper = (ROOT / "run_wall_qwen35_openloop.sh").read_text()
    for name, value in (
        ("RPU_GRAPH_MAX_SEGMENT_COMMAND_MB", "GRAPH_MAX_SEGMENT_COMMAND_MB"),
        ("RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB", "GRAPH_MAX_SEGMENT_INSTRUCTION_MB"),
        ("LKN_MAX_BATCH_ENTRIES", "SDK_MAX_BATCH_ENTRIES"),
        ("LKN_KD_BUF_MB", "SDK_KD_BUF_MB"), ("LKN_INSTR_BUF_MB", "SDK_INSTR_BUF_MB"),
    ):
        assert f'"{name}=${value}"' in wrapper
    assert 'SDK_MAX_BATCH_ENTRIES=65536' in wrapper
    assert 'SDK_KD_BUF_MB=16' in wrapper
    assert 'SDK_INSTR_BUF_MB=128' in wrapper


@pytest.mark.parametrize("opt", ["0", "1"])
def test_unified_preset_overrides_six_budget_settings(opt):
    names = ("RPU_GRAPH_MAX_SEGMENT_ENTRIES", "RPU_GRAPH_MAX_SEGMENT_COMMAND_MB",
             "RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB", "LKN_MAX_BATCH_ENTRIES",
             "LKN_KD_BUF_MB", "LKN_INSTR_BUF_MB")
    result = subprocess.run(
        ["bash", str(ROOT / "run_wall_qwen35_openloop.sh"), "--check"],
        env={**os.environ, **{name: "invalid" for name in names},
             "WALL_QWEN35_OPT": opt,
             "ENV_SH": "/must-not-be-read"},
        capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "environment script is missing" in result.stderr
    assert "must be an integer" not in result.stderr
