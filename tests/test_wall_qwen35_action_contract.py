from __future__ import annotations

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.wall_qwen35 import action as wall_action
from rpu_backend.api.errors import UnsupportedModelError


ROOT = Path(__file__).resolve().parents[1]
ACTION_SOURCE = (
    ROOT / "python/rpu_backend/adapters/wall_qwen35/action.py"
).read_text(encoding="utf-8")
RUNTIME_SOURCE = (
    ROOT / "python/rpu_backend/adapters/wall_qwen35/runtime.py"
).read_text(encoding="utf-8")
QWEN35_SOURCE = (
    ROOT / "python/rpu_backend/adapters/qwen3_5/__init__.py"
).read_text(encoding="utf-8")


def _parameter_contract(function) -> list[tuple[str, inspect._ParameterKind]]:
    return [
        (parameter.name, parameter.kind)
        for parameter in inspect.signature(function).parameters.values()
    ]


def test_action_helper_signatures_are_exact_and_have_no_kwargs_escape_hatch() -> None:
    positional = inspect.Parameter.POSITIONAL_OR_KEYWORD
    keyword = inspect.Parameter.KEYWORD_ONLY
    assert _parameter_contract(wall_action.load_wall_qwen35_action_module) == [
        ("checkpoint", positional),
        ("manifest", keyword),
        ("action_min", keyword),
        ("action_delta", keyword),
    ]
    assert _parameter_contract(wall_action.patch_wall_qwen35_action_for_rpu) == [
        ("expert", positional),
        ("max_seq_len", keyword),
    ]
    assert _parameter_contract(wall_action.run_wall_qwen35_action) == [
        ("expert", positional),
        ("inputs_embeds", keyword),
        ("action_embeds", keyword),
        ("time_cond", keyword),
        ("position_ids", keyword),
        ("prefix_cache", keyword),
        ("kv_cache", keyword),
        ("prefix_len", keyword),
        ("dof_mask", keyword),
    ]


def test_runtime_does_not_silently_drop_exact_action_helper_arguments() -> None:
    tree = ast.parse(RUNTIME_SOURCE)
    compatibility_calls: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if node.func.id != "_call_supported" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Name):
            compatibility_calls.add(first.id)

    assert compatibility_calls.isdisjoint({"load_action", "patch_action", "run"})


def test_scale4_has_exactly_two_python_owners() -> None:
    modulation = inspect.getsource(wall_action._adaptive_modulation_cpu)
    decoder = inspect.getsource(wall_action.run_wall_qwen35_action)
    host_input = inspect.getsource(wall_action.wall_qwen35_host_action_input)
    host_velocity = inspect.getsource(wall_action.wall_qwen35_host_velocity)
    host_euler = inspect.getsource(wall_action.wall_qwen35_host_euler_step)

    assert "gate / 4.0" in modulation
    assert "hidden_cpu / 4.0" in decoder
    assert ACTION_SOURCE.count("gate / 4.0") == 1
    assert ACTION_SOURCE.count("hidden_cpu / 4.0") == 1
    for unscaled in (host_input, host_velocity, host_euler):
        assert "/ 4.0" not in unscaled
    assert "qwen3_5_enable_wall_action_mode(handle, list(_FULL_LAYERS))" in ACTION_SOURCE
    assert wall_action._FULL_LAYERS == (3, 7, 11, 15, 19, 23)


def test_action_fast_replay_signature_keeps_exact_real_prefix() -> None:
    install = inspect.getsource(wall_action.patch_wall_qwen35_action_for_rpu)
    decoder = inspect.getsource(wall_action.run_wall_qwen35_action)

    assert "GraphCache(max_entries=1)" in install
    assert "ACTION_HIDDEN_SIZE, prefix_len" in decoder
    assert "_select_prefill_bucket" not in decoder
    assert "state.action_graph_cache.clear()" in decoder
    assert decoder.index("state.action_graph_cache.is_frozen()") < decoder.index(
        "state.action_graph_cache.clear()"
    )
    assert decoder.index("state.action_graph_cache.is_frozen()") < decoder.index(
        "_copy_physical_prefix"
    )
    assert decoder.index("state.action_graph_cache.is_frozen()") < decoder.index(
        "torch.ops.rpu.qwen3_5_action_forward"
    )


def _processor_with_w1(weight: torch.Tensor) -> wall_action.WallActionProcessor:
    dummy = wall_action.WallHostLinear(torch.empty((1, 1), dtype=torch.float32))
    bank = wall_action.WallNormalizerBank(
        minimum={},
        delta={},
        selected_key="x2_normal",
        selected_minimum=torch.zeros(26),
        selected_delta=torch.ones(26),
    )
    return wall_action.WallActionProcessor(
        w1=wall_action.WallHostLinear(weight),
        time_mlp_in=dummy,
        time_mlp_out=dummy,
        action_proj_back=dummy,
        propri_proj=dummy,
        normalizer_action=bank,
        normalizer_propri=bank,
    )


def test_host_action_input_is_raw_fp32_before_decoder_scale() -> None:
    weight = torch.zeros((1024, 52), dtype=torch.float32)
    weight[0].fill_(1.0)
    expert = SimpleNamespace(action_processor=_processor_with_w1(weight))
    action = torch.ones((1, 32, 26), dtype=torch.float32)
    dof_mask = torch.tensor([1] * 20 + [0] * 6, dtype=torch.float32)

    result = wall_action.wall_qwen35_host_action_input(
        expert, action, dof_mask
    )

    assert result.dtype is torch.float32
    assert tuple(result.shape) == (1, 32, 1024)
    assert torch.all(result[..., 0] == 46.0)
    assert torch.count_nonzero(result[..., 1:]) == 0


def test_adaptive_rows_pack_scale_shift_and_gate_over_four_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bias = torch.cat(
        (
            torch.full((1024,), 1.0),
            torch.full((1024,), 2.0),
            torch.full((1024,), 8.0),
        )
    )
    projection = wall_action._WallAdaProjection(
        weight=torch.empty((1, 1), dtype=torch.float32),
        bias=bias,
    )
    expert = SimpleNamespace(
        _wall_ada_rows=(projection,) * wall_action._MODULATION_ROWS
    )

    def fake_linear(value, _weight, projected_bias):
        return projected_bias.unsqueeze(0).expand(value.shape[0], -1)

    monkeypatch.setattr(wall_action.F, "linear", fake_linear)
    packed = wall_action._adaptive_modulation_cpu(
        expert, torch.zeros((1, 1024), dtype=torch.float32)
    )

    assert packed.dtype is torch.float32
    assert tuple(packed.shape) == (49, 3072)
    assert torch.all(packed[:, :1024] == 2.0)
    assert torch.all(packed[:, 1024:2048] == 2.0)
    assert torch.all(packed[:, 2048:] == 2.0)


def test_host_euler_step_uses_masked_padding_velocity_in_fp32() -> None:
    action = torch.zeros((1, 32, 26), dtype=torch.float32)
    velocity = torch.ones_like(action)
    padding_velocity = torch.full_like(action, 2.0)
    dof_mask = torch.tensor([1] * 20 + [0] * 6, dtype=torch.float32)

    result = wall_action.wall_qwen35_host_euler_step(
        action,
        velocity,
        dof_mask,
        padding_velocity,
        delta_t=0.1,
    )

    assert result.dtype is torch.float32
    torch.testing.assert_close(result[..., :20], torch.full((1, 32, 20), 0.1))
    torch.testing.assert_close(result[..., 20:], torch.full((1, 32, 6), 0.2))
    with pytest.raises(ValueError, match="exactly 10 steps"):
        wall_action.validate_wall_qwen35_euler_profile(
            action, dof_mask, padding_velocity, num_steps=9
        )


def test_action_constants_freeze_certified_target_profile() -> None:
    assert wall_action.ACTION_BATCH_SIZE == 1
    assert wall_action.ACTION_HORIZON == 32
    assert wall_action.ACTION_DIM == 26
    assert wall_action.ACTION_HIDDEN_SIZE == 1024
    assert wall_action.ACTION_INTERMEDIATE_SIZE == 2048
    assert wall_action.ACTION_NUM_LAYERS == 24
    assert wall_action.ACTION_MAX_SEQ_LEN == 780
    assert wall_action.ACTION_MAX_PREFIX_LEN == 384
    assert wall_action.ACTION_NUM_STEPS == 10


def test_wall_base_decoder_uses_explicit_cache_only_no_lm_head_contract() -> None:
    marker = "_wall_qwen35_cache_only_no_lm_head"
    assert f"self.base_model.{marker} = True" in RUNTIME_SOURCE
    assert f'_WALL_CACHE_ONLY_MARKER = "{marker}"' in QWEN35_SOURCE
    assert "_wall_qwen35_load_complete" in QWEN35_SOURCE
    assert "_WALL_QWEN35_VOCAB_SIZE = 256277" in QWEN35_SOURCE
    assert "Qwen3.5 cache-only forward requires logits_to_keep=1" in QWEN35_SOURCE


def _cache_only_candidate(*, manifest=...):
    weight = torch.nn.Parameter(
        torch.empty((256277, 1), device="meta"), requires_grad=False
    )
    embed = SimpleNamespace(weight=weight)
    output = SimpleNamespace(weight=weight)
    model = SimpleNamespace(
        config=SimpleNamespace(
            text_config=SimpleNamespace(
                vocab_size=256277,
                hidden_size=1,
                tie_word_embeddings=True,
            )
        ),
        model=SimpleNamespace(language_model=SimpleNamespace(embed_tokens=embed)),
        get_output_embeddings=lambda: output,
        _wall_qwen35_cache_only_no_lm_head=True,
        _wall_qwen35_load_complete=True,
    )
    if manifest is not ...:
        model._wall_qwen35_checkpoint_manifest = manifest
    return model


def test_cache_only_admission_requires_process_local_exact_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpu_backend.adapters import qwen3_5
    from rpu_backend.adapters.wall_qwen35 import checkpoint as wall_checkpoint

    with pytest.raises(UnsupportedModelError, match="manifest is invalid"):
        qwen3_5._admit_wall_cache_only_model(_cache_only_candidate())

    manifest = SimpleNamespace(root=Path("/exact-wall"))
    calls = []

    def admit(path, supplied):
        calls.append((path, supplied))
        assert supplied is manifest
        return supplied

    monkeypatch.setattr(wall_checkpoint, "_admit_manifest", admit)
    model = _cache_only_candidate(manifest=manifest)
    assert qwen3_5._admit_wall_cache_only_model(model) is True
    assert model._wall_qwen35_cache_only_admitted is True
    assert calls == [(manifest.root, manifest)]

    to_rpu = inspect.getsource(qwen3_5.Qwen3_5Adapter.to_rpu)
    assert to_rpu.index("_admit_wall_cache_only_model(model)") < to_rpu.index(
        "_claim_live_instance(model)"
    )


def test_action_rope_uses_multimodal_positions_not_physical_prefix() -> None:
    positions = torch.arange(231, 263, dtype=torch.int64)
    three_lanes = positions.view(1, 1, -1).expand(3, 1, -1)

    canonical = wall_action._canonical_action_positions(three_lanes, 308)

    assert tuple(canonical.shape) == (3, 32)
    assert canonical.dtype is torch.int64
    assert canonical[:, 0].tolist() == [231, 231, 231]
    assert canonical[:, -1].tolist() == [262, 262, 262]
    assert not bool((canonical == torch.arange(308, 340)).all())

    four_lanes = torch.cat(
        (torch.full((1, 1, 32), 999, dtype=torch.int64), three_lanes), dim=0
    )
    assert torch.equal(
        wall_action._canonical_action_positions(four_lanes, 308), canonical
    )
    assert torch.equal(
        wall_action._canonical_action_positions(positions, 308), canonical
    )


def test_action_rope_position_profile_fails_closed() -> None:
    positions = torch.arange(231, 263, dtype=torch.int64).expand(3, -1).clone()
    with pytest.raises(ValueError, match="requires checkpoint-derived"):
        wall_action._canonical_action_positions(None, 308)
    with pytest.raises(ValueError, match="integer dtype"):
        wall_action._canonical_action_positions(positions.float(), 308)
    drifted_lane = positions.clone()
    drifted_lane[2] += 1
    with pytest.raises(ValueError, match="lanes must coincide"):
        wall_action._canonical_action_positions(drifted_lane, 308)
    noncontiguous = positions.clone()
    noncontiguous[:, 7] += 1
    with pytest.raises(ValueError, match="must be contiguous"):
        wall_action._canonical_action_positions(noncontiguous, 308)


def test_action_rope_table_matches_reference_positions_on_cpu() -> None:
    positions = torch.arange(231, 263, dtype=torch.int64).expand(3, -1)

    cos, sin = wall_action._build_action_rope_tables(positions, device="cpu")

    inv_freq = 1.0 / (
        wall_action._ROPE_THETA
        ** (
            torch.arange(0, wall_action._ROTARY_DIM, 2, dtype=torch.float64)
            / wall_action._ROTARY_DIM
        )
    )
    expected = positions[0, :, None].double() * inv_freq[None, :]
    torch.testing.assert_close(cos, expected.cos().half(), rtol=0, atol=0)
    torch.testing.assert_close(sin, expected.sin().half(), rtol=0, atol=0)
    assert tuple(cos.shape) == (32, 32)


def test_action_rope_cache_is_keyed_by_positions_and_refreshes_stable_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    def fake_tables(position_ids):
        start = float(position_ids[0, 0])
        return (
            torch.full((32, 32), start, dtype=torch.float16),
            torch.full((32, 32), -start, dtype=torch.float16),
        )

    monkeypatch.setattr(wall_action, "_build_action_rope_tables", fake_tables)
    monkeypatch.setattr(
        torch.ops.rpu,
        "qwen3_5_set_prefill_rope",
        lambda handle, cos, sin: calls.append((handle, cos.clone(), sin.clone())),
        raising=False,
    )
    state = SimpleNamespace(handle=17, action_rope_cache=None)
    first = torch.arange(231, 263, dtype=torch.int64).expand(3, -1)
    second = torch.arange(232, 264, dtype=torch.int64).expand(3, -1)

    wall_action._ensure_action_rope(state, first)
    wall_action._ensure_action_rope(state, first.clone())
    wall_action._ensure_action_rope(state, second)

    assert len(calls) == 2
    assert calls[0][0] == calls[1][0] == 17
    assert calls[0][1][0, 0].item() == 231.0
    assert calls[1][1][0, 0].item() == 232.0
