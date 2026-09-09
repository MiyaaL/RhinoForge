"""Board-free orchestration/math gates; these do not certify RPU parity."""

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.qwen3_5 import vision


@pytest.fixture
def packed_runtime(monkeypatch):
    # Execute the real adapter control flow, redirecting only device transfers
    # and native/Graph endpoints to CPU doubles.
    tensor_to = torch.Tensor.to

    def cpu_to(tensor, *args, **kwargs):
        if args and args[0] == "rpu":
            args = ("cpu", *args[1:])
        if kwargs.get("device") == "rpu":
            kwargs["device"] = "cpu"
        return tensor_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", cpu_to)
    monkeypatch.setattr(torch.rpu, "get_debug_export", lambda: False)
    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary", lambda: None, raising=False)

    class GraphCache:
        def __init__(self):
            self.calls = []
            self.evictions = []

        @contextmanager
        def capture(self, sig):
            self.calls.append(sig)
            yield

        def evict(self, sig):
            self.evictions.append(sig)

    monkeypatch.setattr(vision.rpu_backend.graph, "GraphSignature", SimpleNamespace)

    class Tower(SimpleNamespace):
        def fast_pos_embed_interpolate(self, grid):
            # Image-specific values let the test check that all positions are
            # forwarded in order, including when the geometry changes.
            return torch.cat([
                torch.full((int(t * h * w), 4), float(h + w))
                for t, h, w in grid.tolist()
            ]).half()

    graph = GraphCache()
    kv = SimpleNamespace(k_caches=[object()], v_caches=[object()],
                         reset_to_position=lambda pos: None)
    tower = Tower(
        _rpu_vision_handle=7, _rpu_vision_spatial_merge_size=2,
        _rpu_vision_num_layers=1, _rpu_vision_hidden_size=4,
        _rpu_vision_kv_cache=kv, _rpu_vision_packed_spatial=True,
        _rpu_vision_step0=True, _rpu_vision_has_merger=True,
        _rpu_vision_position_idx_keepalive=torch.zeros((588, 2), dtype=torch.int16),
        _rpu_vision_freq_cos=torch.zeros((14, 16)),
        _rpu_vision_step0_pos_grid=None, _rpu_vision_step0_pos=None,
        _rpu_vision_step0_pos_temporal_num_frames=None,
        _rpu_vision_step0_pos_camera_batch_count=None,
        _rpu_vision_graph_cache=graph, _rpu_vision_graph_max_entries=1,
        _rpu_vision_graph_key=None, _rpu_vision_graph_sig=None,
    )
    calls = []
    buffers = {}

    def native(*args):
        calls.append(args)
        _, pixels, _, _, n, pos, target, starts, _, _, lengths = args
        # CPU oracle for the native contract: independent per-image attention,
        # shared rowwise work, then four-patch merger, preserving pixel order.
        rows = (pixels.squeeze(0) + pos).float()
        lengths = lengths or (n,)
        attended = []
        for part in rows.split(lengths):
            attended.append(torch.nn.functional.scaled_dot_product_attention(
                part[None, None], part[None, None], part[None, None]
            )[0, 0])
        result = torch.cat(attended).half().unsqueeze(0)
        # Reuse the same output buffers, like native. The adapter must clone.
        if n not in buffers:
            buffers[n] = (torch.empty_like(result), torch.empty((1, n // 4, 4)))
        out, merged = buffers[n]
        out.copy_(result)
        merged.copy_(out.reshape(1, n // 4, 4, 4).mean(2))
        if target is not None:
            offset = 0
            for start, count in zip(starts, lengths):
                target[:, start:start + count // 4].copy_(merged[:, offset:offset + count // 4])
                offset += count // 4
        return out

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_forward", native, raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_get_merger_out",
                        lambda handle: buffers[calls[-1][4]][1], raising=False)
    return tower, calls, graph


def test_three_images_share_one_call_and_keep_independent_outputs(packed_runtime):
    tower, calls, graph = packed_runtime
    grid = torch.tensor([[1, 8, 14], [1, 10, 14], [1, 10, 14]])
    pixels = torch.randn((392, 4), generator=torch.Generator().manual_seed(3)).half()
    first = vision._rpu_vision_forward(tower, pixels, grid)
    original = first.last_hidden_state.clone()
    pooled = first.pooler_output.clone()
    assert len(calls) == len(graph.calls) == 1
    assert calls[0][-1] == (112, 140, 140)
    assert calls[0][1].shape == (1, 392, 4)
    assert calls[0][5].shape == (392, 4)
    assert first.pooler_output.shape == (98, 4)
    # Same geometry, fresh pixels: reuse signature, refresh data, and never
    # overwrite a previously returned output. Other cameras cannot affect face.
    changed = pixels.clone()
    changed[112:] += 3
    second = vision._rpu_vision_forward(tower, changed, grid)
    assert graph.calls[0] is graph.calls[1]
    torch.testing.assert_close(first.last_hidden_state, original, rtol=0, atol=0)
    torch.testing.assert_close(first.pooler_output, pooled, rtol=0, atol=0)
    torch.testing.assert_close(second.last_hidden_state[:112], original[:112], rtol=0, atol=0)
    assert not torch.equal(second.last_hidden_state[112:], original[112:])


@pytest.mark.parametrize("packed", [True, False])
@pytest.mark.parametrize("wall_label", [True, False])
def test_graph_labels_are_policy_owned_in_both_vision_modes(
    packed_runtime, packed, wall_label,
):
    tower, calls, graph = packed_runtime
    tower._rpu_vision_packed_spatial = packed
    if wall_label:
        tower._wall_qwen35_vision_graph_op_id = "rpu_wall_qwen35_vision"
    grid = torch.tensor([[1, 8, 14], [1, 10, 14], [1, 10, 14]])
    pixels = torch.zeros((392, 4), dtype=torch.float16)
    for _ in range(2):
        vision._rpu_vision_forward(tower, pixels, grid)
    assert len(calls) == len(graph.calls) == (2 if packed else 6)
    expected = "rpu_wall_qwen35_vision" if wall_label else "qwen3_5_vision"
    assert {sig.op_id for sig in graph.calls} == {expected}
    if packed:
        assert graph.calls[0] is graph.calls[1]


def test_wall_sets_graph_labels_before_adapter_installation():
    source = (Path(__file__).resolve().parents[1]
              / "python/rpu_backend/adapters/wall_qwen35/runtime.py").read_text()
    label = 'self.base_model.model.visual._wall_qwen35_vision_graph_op_id = "rpu_wall_qwen35_vision"'
    assert source.index(label) < source.index("self._base_adapter.to_rpu(")
    assert 'text_state.prefill_graph_op_id = "rpu_wall_qwen35_prefill"' in source


def test_same_total_different_boundaries_change_signature_and_positions(packed_runtime):
    tower, calls, graph = packed_runtime
    pixels = torch.zeros((392, 4), dtype=torch.float16)
    grid = torch.tensor([[1, 8, 14], [1, 10, 14], [1, 10, 14]])
    vision._rpu_vision_forward(tower, pixels, grid)
    vision._rpu_vision_forward(tower, pixels, grid[[1, 0, 2]])
    assert calls[-1][-1] == (140, 112, 140)
    assert graph.calls[0].dyn_dims != graph.calls[1].dyn_dims
    assert len(graph.evictions) == 1
    assert not torch.equal(calls[0][5], calls[1][5])


def test_maximum_envelope_and_fresh_fusion_destinations(packed_runtime):
    tower, calls, graph = packed_runtime
    grid = torch.tensor([[1, 14, 14]] * 3)
    pixels = torch.zeros((588, 4), dtype=torch.float16)
    first = torch.full((1, 200, 4), -1.0)
    second = first.clone()
    starts = [2, 60, 125]
    for target in (first, second):
        out = vision._rpu_vision_forward(
            tower, pixels, grid, _rpu_fusion_target=target,
            _rpu_fusion_run_starts=starts,
        )
        assert out.last_hidden_state is None
    assert len(calls) == 2
    assert calls[0][6] is first and calls[1][6] is second
    assert calls[0][7] == starts
    assert graph.calls[0] is graph.calls[1]
    torch.testing.assert_close(first, second)
    assert torch.all(first[:, 174:] == -1)


@pytest.mark.parametrize("grid,rows,kwargs", [
    ([[1, 8, 14]] * 2, 224, {}),
    ([[1, 16, 14]] * 3, 672, {}),
    ([[2, 8, 14]] * 3, 672, {}),
    ([[1, 8, 14]] * 3, 300, {}),
    ([[1, 8, 14]] * 3, 336, {"_rpu_camera_batch_count": 3}),
])
def test_invalid_packed_profiles_fail_before_native(packed_runtime, grid, rows, kwargs):
    tower, calls, graph = packed_runtime
    with pytest.raises(ValueError, match="vision"):
        vision._rpu_vision_forward(tower, torch.zeros((rows, 4)), torch.tensor(grid), **kwargs)
    assert calls == graph.calls == []


def test_generic_images_remain_separate(packed_runtime):
    tower, calls, graph = packed_runtime
    tower._rpu_vision_packed_spatial = False
    grid = torch.tensor([[1, 8, 14], [1, 10, 14], [1, 10, 14]])
    output = vision._rpu_vision_forward(tower, torch.zeros((392, 4)).half(), grid)
    assert [call[4] for call in calls] == [112, 140, 140]
    assert all(call[-1] == () for call in calls)
    assert output.last_hidden_state.shape == (392, 4)


def test_native_span_attention_and_merger_scatter_source_contract():
    root = Path(__file__).resolve().parents[1]
    source = (root / "src/fused/rpu_qwen3_5_vision_model.cpp").read_text()
    attention = source.split("void Qwen3_5VisionModel::emit_packed_spatial_attention", 1)[1]
    attention = attention.split("void Qwen3_5VisionModel::build_layer_subgraph", 1)[0]
    assert "ctx().stage_plan.spans" in attention
    assert "span.len, nq, nq, hd, span.len" in attention
    assert "spatial_k_caches_[layer_idx][image]" in attention
    assert 'addr_offset("q_comp").value + bytes' in attention
    assert 'addr_offset("sdpa_out").value + bytes' in attention
    assert "plan.num_chunks == 1" in source
    assert "detail::layout_mix(hash, span.len)" in source
    assert "if (views.empty())" in source  # views retain TensorImpl across replay
    assert "(begin - span.offset) / mu * oh * DWIDTH" in source
    assert "(begin - off) / mu * oh * DWIDTH" in source

    reduction = source.split("void Qwen3_5VisionModel::emit_spatial_residual_reduce", 1)[1]
    reduction = reduction.split("void Qwen3_5VisionModel::build_layer_subgraph", 1)[0]
    assert "spatial_image_spans_.empty()" in reduction  # generic path unchanged
    assert "input, residual, output, chunk.len, h" in reduction
    assert "ctx().stage_plan.spans" in reduction
    assert "span.offset * h * DWIDTH" in reduction
    assert "input + bytes, residual + bytes, output + bytes" in reduction
    assert "span.len, h, NUM_CORES, NUM_CORES" in reduction
    assert "chunk.offset == 0 && chunk.len == current_num_patches_" in reduction
    body = source.split("void Qwen3_5VisionModel::build_layer_subgraph", 1)[1]
    body = body.split("Qwen3_5VisionModel::declare_buffers", 1)[0]
    assert body.count("emit_spatial_residual_reduce(") == 2
    assert "rpu_launch_all_reduce_sum_residual_kernel(" not in body
    assert "WALL_DIAG_" not in source  # no temporary diagnostic selectors/taps

    # Independent index oracle for the maximum-envelope cross-chunk scatter.
    scatters = []
    for off in (0, 512):
        end = min(off + 512, 588)
        for image in range(3):
            begin = max(off, image * 196)
            stop = min(end, (image + 1) * 196)
            if begin < stop:
                scatters.append((image, (begin - off) // 4,
                                 (begin - image * 196) // 4, (stop - begin) // 4))
    assert scatters == [(0, 0, 0, 49), (1, 49, 0, 49),
                        (2, 98, 0, 30), (2, 0, 30, 19)]


@pytest.mark.parametrize("has_lengths", [False, True])
def test_packed_native_schema_preflight(monkeypatch, has_lengths):
    schema = torch._C.parse_schema(
        "rpu::qwen3_5_vision_forward(Tensor input"
        + (", int[] image_patch_counts=[]" if has_lengths else "")
        + ") -> Tensor"
    )
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_forward",
                        SimpleNamespace(default=SimpleNamespace(_schema=schema)), raising=False)
    if has_lengths:
        vision._require_packed_spatial_vision_native()
    else:
        with pytest.raises(RuntimeError, match="rebuilding"):
            vision._require_packed_spatial_vision_native()


def test_packed_install_rejects_old_native_without_mutation(monkeypatch):
    tower = SimpleNamespace(blocks=[object()] * 24, merger=object(),
                            config=object(), _wall_qwen35_packed_vision=True)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_forward", object(), raising=False)
    before = vars(tower).copy()
    with pytest.raises(RuntimeError, match="rebuilding"):
        vision.install_qwen3_5_vision_for_rpu(tower)
    assert vars(tower) == before


def test_packed_mode_is_monotonic():
    from rpu_backend.api.errors import RPUConfigError
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator

    tower = torch.nn.Module()
    install_hw_attr_validator(tower)
    tower._rpu_vision_packed_spatial = True
    with pytest.raises(RPUConfigError, match="monotonic"):
        tower._rpu_vision_packed_spatial = False
