"""Board-free source contract for the certified Wall Qwen3.5 action ABI."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CPP = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text(encoding="utf-8")
HEADER = (ROOT / "src/fused/rpu_qwen3_5_model.h").read_text(encoding="utf-8")
DECLS = (ROOT / "src/core/rpu_kernel_decls.h").read_text(encoding="utf-8")
DISPATCH = (ROOT / "src/core/rpu_dispatch_registrations.inc").read_text(
    encoding="utf-8"
)


def _section(start: str, end: str) -> str:
    begin = CPP.index(start)
    return CPP[begin : CPP.index(end, begin)]


def test_wall_action_abi_is_registered_as_a_cold_handle_setter():
    assert "void enable_wall_action_mode(at::IntArrayRef full_layers);" in HEADER
    assert "void rpu_qwen3_5_enable_wall_action_mode(" in DECLS
    assert (
        'm.def("qwen3_5_enable_wall_action_mode(int handle, '
        'int[] full_layers) -> ()",' in DISPATCH
    )
    setter = _section(
        "void Qwen3_5Model::enable_wall_action_mode(",
        "void Qwen3_5Model::set_action_io_weights(",
    )
    for exact_guard in (
        "kWallActionNumLayers",
        "kWallActionHiddenSize",
        "kWallActionIntermediateSize",
        "kWallActionNumQHeads",
        "kWallActionEffectiveKvHeads",
        "kWallActionHeadDim",
        "!has_gdn_",
        "[3, 7, 11, 15, 19, 23]",
        "action_residual_eps_ = eps_ / 16.0",
        "invalidate_model_state()",
    ):
        assert exact_guard in setter


def test_wall_uses_separate_adaptive_epsilon_and_keeps_qk_epsilon():
    adaptive = _section(
        "void Qwen3_5Model::apply_adaptive_norm(",
        "void Qwen3_5Model::apply_adaptive_residual_gate(",
    )
    assert "wall_action_mode_ ? action_residual_eps_ : eps_" in adaptive

    attention = _section(
        "void Qwen3_5Model::build_full_attention(",
        "void Qwen3_5Model::build_gdn(",
    )
    assert "action_residual_eps_" not in attention
    assert attention.count("hd, eps_") >= 4


def test_wall_zero_residual_is_persistent_and_graph_initialized():
    buffers = _section(
        "std::vector<BufferDecl> Qwen3_5Model::declare_buffers(",
        "ModelStaticConfig Qwen3_5Model::static_config(",
    )
    assert '"zero_resid", res, 1, 8, StorageClass::Persistent' in buffers

    layer = _section(
        "void Qwen3_5Model::build_layer_subgraph(",
        "void Qwen3_5Model::launch_linear(",
    )
    assert "wall_action_mode_ && layer_idx == 0" in layer
    assert "rpu_launch_memset_spm_multicore(" in layer
    assert 'addr(0, "zero_resid"), chunk.len * h' in layer


def test_wall_identity_attention_and_mlp_gate_before_residual_add():
    layer = _section(
        "void Qwen3_5Model::build_layer_subgraph(",
        "void Qwen3_5Model::launch_linear(",
    )
    assert "wall_action_mode_ && !wall_action_full_mask_[layer_idx]" in layer
    assert "apply_wall_residual_gate(" in layer

    wall_gate = _section(
        "void Qwen3_5Model::apply_wall_residual_gate(",
        "void Qwen3_5Model::emit_wall_mlp_pipeline(",
    )
    assert wall_gate.index("ValuOpType::MUL") < wall_gate.index("ValuOpType::ADD")
    assert "ValuOpType::SUB" not in wall_gate

    wall_mlp = _section(
        "void Qwen3_5Model::emit_wall_mlp_pipeline(",
        "void Qwen3_5Model::emit_action_input_projection(",
    )
    assert 'addr(0, "down"), addr(0, "zero_resid")' in wall_mlp
    assert "apply_wall_residual_gate(" in wall_mlp

    attention = _section(
        "void Qwen3_5Model::build_full_attention(",
        "void Qwen3_5Model::build_gdn(",
    )
    assert 'wall_action_mode_ ? addr(0, "zero_resid")' in attention
    assert "if (wall_action_mode_)" in attention


def test_wall_rejects_native_io_and_set_weights_clears_mode():
    io_setter = _section(
        "void Qwen3_5Model::set_action_io_weights(",
        "int64_t Qwen3_5Model::resolve_prefill_chunk_size(",
    )
    assert "!wall_action_mode_" in io_setter

    action_step = _section(
        "at::Tensor Qwen3_5Model::forward_action_step(",
        "// ── forward: stash GDN state",
    )
    assert "!wall_action_mode_" in action_step

    setup = _section(
        "void Qwen3_5Model::set_weights(",
        "}  // namespace v3",
    )
    assert "wall_action_mode_ = false" in setup
    assert "wall_action_full_mask_.clear()" in setup


def test_wall_prefix_and_kv_cache_envelope_is_exact():
    forward = _section(
        "at::Tensor Qwen3_5Model::forward_action(",
        "at::Tensor Qwen3_5Model::forward_action_step(",
    )
    assert "wall_full_prefix = -1" in forward
    assert "prefix <= 384" in forward
    assert "prefix == wall_full_prefix" in forward
    for topology_guard in (
        "kc.sizes() == vc.sizes()",
        "kc.size(0) == 1",
        "kc.size(2) == 1 && kc.size(3) == 16",
        "kc.size(4) == 8 && kc.size(5) == 16",
        "kc.size(6) == 16",
        "prefix + seq_len <= k_capacity",
        "prefix + seq_len <= v_capacity",
    ):
        assert topology_guard in forward


def test_legacy_g05_add_sub_gate_path_remains_available():
    legacy = _section(
        "void Qwen3_5Model::apply_raw_residual_gate(",
        "void Qwen3_5Model::apply_wall_residual_gate(",
    )
    assert "ValuOpType::SUB" in legacy
    assert "ValuOpType::MUL" in legacy
    assert "ValuOpType::ADD" in legacy
