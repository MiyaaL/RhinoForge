"""Unified cold presets and same-math FP16 Graph orchestration (board-free)."""

import contextlib
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.wall_qwen35 import action, execution

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("WALL_QWEN35_OPT", "RPU_GRAPH_MAX_SEGMENT_ENTRIES",
        "RPU_GRAPH_MAX_SEGMENT_COMMAND_MB", "RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB",
        "LKN_MAX_BATCH_ENTRIES", "LKN_KD_BUF_MB", "LKN_INSTR_BUF_MB")


@pytest.mark.parametrize("value,opt", [(None, True), ("1", True), ("ON", True),
                                      ("0", False), ("false", False)])
def test_switch_defaults_and_owns_all_budgets(monkeypatch, value, opt):
    for key in KEYS:
        monkeypatch.setenv(key, "stale")
    if value is None:
        monkeypatch.delenv("WALL_QWEN35_OPT")
    else:
        monkeypatch.setenv("WALL_QWEN35_OPT", value)
    assert execution.resolve_wall_qwen35_opt() is opt
    execution.configure_wall_qwen35_execution(opt)
    assert tuple(os.environ[k] for k in KEYS) == (
        ("1", "32768", "8", "64", "65536", "16", "128") if opt else
        ("0", "8192", "4", "32", "65536", "8", "64")
    )


@pytest.mark.parametrize("value", ["", "2", "-1", " 1", "unknown"])
def test_bad_switch_rejected_by_python_and_shell_before_loading(monkeypatch, value):
    monkeypatch.setenv("WALL_QWEN35_OPT", value)
    with pytest.raises(ValueError, match="WALL_QWEN35_OPT"):
        execution.resolve_wall_qwen35_opt()
    result = subprocess.run(
        ["bash", str(ROOT / "run_wall_qwen35_openloop.sh"), "--check"],
        env={**os.environ, "ENV_SH": "/must-not-be-read"}, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert "WALL_QWEN35_OPT must be boolean" in result.stderr
    assert "environment script" not in result.stderr


@pytest.mark.parametrize("flag", ["--language-one-graph", "--action-execution=fp32_host"])
def test_old_independent_flags_are_rejected(flag):
    result = subprocess.run(["bash", str(ROOT / "run_wall_qwen35_openloop.sh"), flag],
                            capture_output=True, text=True)
    assert result.returncode == 2 and "unknown option" in result.stderr


@pytest.mark.parametrize("opt", [None, "0", "1"])
def test_shell_forwards_unified_preset_to_python(tmp_path, opt):
    env_sh = tmp_path / "env.sh"
    env_sh.write_text(f'export CONDA_PREFIX="{tmp_path}"\n')
    fake_python = tmp_path / "python"
    fake_python.write_text(
        '#!/bin/bash\nif [[ "$1" == -c ]]; then printf "/tmp\\n"; else\n'
        'printf "TEST_PRESET=%s,%s,%s,%s,%s,%s,%s\\n" '
        + " ".join(f'"${{{k}}}"' for k in KEYS) + '\nfi\n'
    )
    fake_python.chmod(0o755)
    asset = tmp_path / "placeholder.ref"
    asset.touch()
    env = {**os.environ, **{k: "stale" for k in KEYS},
           "ENV_SH": str(env_sh), "PYTHON_BIN": str(fake_python),
           "DATASET_DIR": str(tmp_path), "CHECKPOINT_PATH": str(tmp_path),
           "RHINO_LAUNCH_LIB_DIR": str(tmp_path), "RPU_KERNEL_LIB_PATH": str(asset)}
    if opt is None:
        env.pop("WALL_QWEN35_OPT")
    else:
        env["WALL_QWEN35_OPT"] = opt
    result = subprocess.run(
        ["bash", str(ROOT / "run_wall_qwen35_openloop.sh"), "--check", "--no-sudo"],
        env=env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    expected = "0,8192,4,32,65536,8,64" if opt == "0" else "1,32768,8,64,65536,16,128"
    assert f"TEST_PRESET={expected}" in result.stdout


@pytest.mark.parametrize("one_graph", [True, False])
def test_fp16_loop_refreshes_steps_inputs_and_retained_outputs(monkeypatch, one_graph):
    import rpu_backend
    # Board-free GraphSignature deliberately has no native key payload.
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", lambda **kw: kw)
    class Cache:
        def __init__(self):
            self.signature = None
            self.builds = self.replays = 0
            self.frozen = False

        def lookup(self, sig):
            return self if sig == self.signature else None

        def is_frozen(self):
            return self.frozen

        def size(self):
            return int(self.signature is not None)

        @contextlib.contextmanager
        def capture(self, sig):
            if self.signature is None:
                self.signature = sig
                self.builds += 1
            else:
                assert sig == self.signature
                self.replays += 1
            yield

    cache = Cache()
    state = SimpleNamespace(action_one_graph=one_graph, action_graph_cache=cache,
                            handle=1, action_k_caches=[], action_v_caches=[])
    expert = SimpleNamespace(_wall_qwen35_action_ready=True, training=False,
                             _rpu_qwen3_5=state)
    original_to = torch.Tensor.to

    def to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            return tensor.clone()
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", to)
    prefix_updates = []
    monkeypatch.setattr(action, "_copy_physical_prefix",
                        lambda *a: prefix_updates.append(a[-1]) or [308] * 24)
    monkeypatch.setattr(action, "_ensure_action_rope", lambda *a: None)
    mods = torch.arange(10, dtype=torch.float16).view(10, 1, 1).expand(10, 49, 3072).contiguous()
    monkeypatch.setattr(action, "_wall_loop_modulation", lambda _: mods)
    calls = []

    def native(handle, x, keep, padding, out, dt, mod, k, v, lens, steps):
        assert x.data_ptr() != out.data_ptr()
        values = mod[:, 0, 0].tolist() if steps == 10 else [mod[0, 0].item()]
        calls.append(values)
        out.copy_(x)
        for value in values:
            out.add_((torch.full_like(out, value) * keep + padding) * dt)
        return out

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_wall_action_loop", native, raising=False)
    outputs = []
    for seed in (3407, 3408, 3407):
        x = torch.randn(1, 32, 26, generator=torch.Generator().manual_seed(seed))
        mask = torch.tensor([1] * 20 + [0] * 6)
        padding = 0.25 - x
        result = action.run_wall_qwen35_action_loop(
            expert, action=x, dof_mask=mask, padding_velocity=padding,
            position_ids=torch.arange(231, 263), prefix_cache=object(), prefix_len=308,
        )
        expected, keep, pv = action._pack_wall_fp16_loop_inputs(x, mask, padding)
        _, dt = action._wall_fp16_times()
        for i in range(10):
            expected.add_((torch.full_like(expected, i) * keep + pv) * dt)
        assert torch.equal(result, expected[..., :26].float())
        assert result.dtype == torch.float32 and result.is_contiguous()
        outputs.append(result)
        cache.frozen = True
    assert cache.builds == 1 and cache.replays == (2 if one_graph else 29)
    assert prefix_updates == [308] * 3
    assert calls == ([list(range(10))] * 4 if one_graph else [[0]] + [[i] for i in range(10)] * 3)
    assert torch.equal(outputs[0], outputs[2]) and not torch.equal(outputs[0], outputs[1])
    assert len({x.data_ptr() for x in outputs}) == 3
    with pytest.raises(RuntimeError, match="READY miss"):
        action.run_wall_qwen35_action_loop(
            expert, action=x, dof_mask=mask, padding_velocity=padding,
            position_ids=torch.arange(231, 263), prefix_cache=object(), prefix_len=309,
        )
    assert prefix_updates == [308] * 3
