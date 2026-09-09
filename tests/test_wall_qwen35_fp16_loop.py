"""Board-free FP16 Action contract, independent of the existing FP32 host arm."""
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.wall_qwen35 import action as wall

ROOT = Path(__file__).resolve().parents[1]


def test_packed_mask_survives_all_ten_positive_fp16_updates():
    x = torch.linspace(-2, 2, 32 * 26).reshape(1, 32, 26)
    mask = torch.tensor([1] * 20 + [0] * 6)
    padding = 0.25 - x
    packed, keep, pv = wall._pack_wall_fp16_loop_inputs(x, mask, padding)
    assert packed.shape == pv.shape == (1, 32, 64)
    assert keep.shape == (64,)
    assert packed.dtype == pv.dtype == keep.dtype == torch.float16
    assert torch.equal(packed[..., 26:52], mask.half().expand(1, 32, 26))
    assert packed[..., 52:].eq(0).all() and pv[..., 26:].eq(0).all()
    assert pv[..., :20].eq(0).all()
    times, dt = wall._wall_fp16_times()
    assert dt == 0.09991455078125
    assert not torch.equal(times[1:] - times[:-1], (times[1] - times[0]).expand(10))
    reference = x.half()
    for _ in range(10):
        velocity = torch.full_like(packed, 0.5)
        velocity[..., 26:] = 0
        packed = packed + (velocity * keep + pv) * dt
        reference = reference + (0.5 * mask.half() + padding.half() * (1 - mask.half())) * dt
    assert torch.equal(packed[..., :26], reference)
    assert torch.equal(packed[..., 26:52], mask.half().expand(1, 32, 26))
    assert packed[..., 52:].eq(0).all()


def test_packed_inputs_have_independent_storage_and_reject_half_overflow():
    x = torch.zeros(1, 32, 26)
    packed, _, _ = wall._pack_wall_fp16_loop_inputs(x, torch.ones(26), x)
    x.fill_(3)
    assert packed[..., :26].eq(0).all()
    with pytest.raises(ValueError, match="after FP16"):
        wall._pack_wall_fp16_loop_inputs(x.fill_(1e10), torch.ones(26), x)


def test_io_weights_pad_without_folding_scale(monkeypatch):
    wi = torch.linspace(-1, 1, 1024 * 52).reshape(1024, 52)
    wo = torch.ones(26, 1024)
    processor = SimpleNamespace(w1=wall.WallHostLinear(wi),
                                action_proj_back=wall.WallHostLinear(wo))
    monkeypatch.setattr(wall, "_processor", lambda _: processor)
    inp, out = wall._wall_fp16_io_weights(None)
    assert inp.shape == (1024, 64) and out.shape == (64, 1024)
    assert torch.equal(inp[:, :52], wi.half())
    assert inp[:, 52:].eq(0).all() and out[26:].eq(0).all()
    assert torch.equal(out[:26], wo.half())


def test_loop_refreshes_semantic_inputs_before_capture_and_checks_ready_first():
    src = inspect.getsource(wall.run_wall_qwen35_action_loop)
    assert src.index("cache.is_frozen()") < src.index("_copy_physical_prefix")
    assert src.count("_copy_physical_prefix(") == 1
    assert src.index("_ensure_action_rope") < src.index("with cache.capture")
    assert 'op_id="rpu_wall_qwen35_action_fp16_loop"' in src
    assert "int(prefix_len)" in src and "dyn_dims=[steps_per_graph, 24" in src
    assert 'torch.empty_like(packed)' in src and '.contiguous().clone()' in src


def test_physical_single_segment_policy_is_per_cache_and_fail_closed():
    infra = (ROOT / "src/graph/graph_infra.cpp").read_text()
    execute = (ROOT / "src/graph/graph_runtime_execute.cpp").read_text()
    assert "make_registered_rpu_kernel_graph(require_single_segment_)" in infra
    assert "segment_resource_budget(require_single_segment_)" in execute
    assert "segments_.size() == 1 && segments_[0].start_idx == 0" in execute
    assert "segments_[0].end_idx == nodes_.size()" in execute
    assert "seg.instruction_bytes <= resource_budget.instr_bytes" in execute
    assert "sdk_instr / 2" in execute and "sdk_kd / 2" in execute
    assert "Single-segment Graph forbids raw-kernel downgrade" in execute
    assert "Single-segment Graph forbids partial execution at sync_point" in execute
    runtime = (ROOT / "src/graph/graph_runtime.cpp").read_text()
    assert "Single-segment Graph cannot be empty" in runtime
    assert "!outer->require_single_segment_" in runtime


def test_native_loop_preserves_wall_rounding_and_persistent_state():
    source = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    inp = source.split("void Qwen3_5Model::emit_action_input_projection()", 1)[1].split(
        "void Qwen3_5Model::emit_action_output_projection()", 1)[0]
    out = source.split("void Qwen3_5Model::emit_action_output_projection()", 1)[1].split(
        "// ── build_full_attention", 1)[0]
    assert inp.index("launch_linear(") < inp.index("c10::Half(0.25f)")
    assert "action_padding_velocity_live_base_" in inp
    assert "rpu_launch_eltwise_binary_scalar_spm_kernel(" in out
    assert 'addr(0, "action_padding_velocity")' in out
    assert 'wall_action_mode_ ? StorageClass::Persistent : StorageClass::Temp' in source
    assert '"action_padding_velocity"' in source


def test_old_native_extension_rejected_before_install(monkeypatch):
    import rpu_backend
    monkeypatch.setattr(rpu_backend, "_cpp_ext", SimpleNamespace(), raising=False)
    with pytest.raises(RuntimeError, match="rebuilt Python/native"):
        wall.require_wall_fp16_loop_runtime()


def test_public_switch_and_removed_action_mode_fail_early(monkeypatch):
    from rpu_backend.api.wall_qwen35 import WallQwen35Policy
    assert "action_execution" not in inspect.signature(WallQwen35Policy.from_checkpoint).parameters
    with pytest.raises(TypeError, match="action_execution"):
        WallQwen35Policy.from_checkpoint("/does/not/exist", action_execution="fp32_host")
    monkeypatch.setenv("WALL_QWEN35_OPT", "invalid")
    with pytest.raises(ValueError, match="WALL_QWEN35_OPT"):
        WallQwen35Policy.from_checkpoint("/does/not/exist")


def test_sdk_budget_guard_rejects_before_install(monkeypatch):
    import rpu_backend
    monkeypatch.setattr(rpu_backend, "_cpp_ext", SimpleNamespace(graph_single_segment_abi=1), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_wall_action_loop", object(), raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_wall_action_prefix_bucket", object(), raising=False)
    monkeypatch.setenv("LKN_MAX_BATCH_ENTRIES", "65536")
    monkeypatch.setenv("LKN_KD_BUF_MB", "4")
    monkeypatch.setenv("LKN_INSTR_BUF_MB", "64")
    with pytest.raises(RuntimeError, match="cold LKN_KD_BUF_MB"):
        wall.require_wall_fp16_loop_runtime()
    monkeypatch.setenv("LKN_KD_BUF_MB", "8")
    monkeypatch.setenv("LKN_INSTR_BUF_MB", "64")
    wall.require_wall_fp16_loop_runtime()


def test_direct_runtime_checks_native_before_processor_or_model_loading():
    from rpu_backend.adapters.wall_qwen35.runtime import WallQwen35Runtime
    source = inspect.getsource(WallQwen35Runtime.install)
    assert source.index("require_wall_fp16_loop_runtime()") < source.index("_load_processor(")
    patch = inspect.getsource(wall.patch_wall_qwen35_action_for_rpu)
    assert patch.index("io_weights = _wall_fp16_io_weights") < patch.index("install_qwen3_5_text_for_rpu(")
