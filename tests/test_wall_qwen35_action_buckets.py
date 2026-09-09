"""Action bucketing semantics and retained-state checks without a board."""
import inspect
from pathlib import Path

import pytest
import torch

from rpu_backend.adapters.wall_qwen35 import action

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("prefix,bucket", [
    (1, 64), (63, 64), (64, 64), (65, 128), (299, 320),
    (313, 320), (319, 320), (320, 320), (321, 384), (384, 384),
])
def test_bucket_and_swizzled_gap(prefix, bucket):
    assert action._action_prefix_bucket(prefix) == bucket
    # Distinct K/V layouts: include dirty finite and NaN padding, and preserve
    # prefix/suffix exactly. Use independent logical views for the assertions.
    k = torch.randn(1, 26, 1, 16, 8, 16, 16, dtype=torch.float16)
    v = torch.randn_like(k)
    old_k, old_v = k.clone(), v.clone()
    action._zero_action_prefix_padding(k, v, prefix, bucket)

    def logical(t, is_k):
        order = (0, 4, 2, 1, 5, 3, 6) if is_k else (0, 4, 2, 1, 6, 3, 5)
        return t.permute(order).reshape(1, 8, 416, 256)

    for t, old, is_k in ((k, old_k, True), (v, old_v, False)):
        actual, before = logical(t, is_k), logical(old, is_k)
        assert torch.equal(actual[:, :, :prefix], before[:, :, :prefix])
        assert actual[:, :, prefix:bucket].eq(0).all()
        assert torch.equal(actual[:, :, bucket:], before[:, :, bucket:])
    k.fill_(float("nan"))
    v.fill_(float("nan"))
    action._zero_action_prefix_padding(k, v, prefix, bucket)
    assert logical(k, True)[:, :, prefix:bucket].eq(0).all()
    assert logical(v, False)[:, :, prefix:bucket].eq(0).all()


@pytest.mark.parametrize("value", [0, -1, 385, True, 64.0, "64", None])
def test_reject_invalid_prefix(value):
    with pytest.raises(ValueError, match="integer"):
        action._action_prefix_bucket(value)


def test_bucket_gap_mask_preserves_attention_semantics():
    # Independent FP32 math anchor for relocating action keys, not a claim of
    # native FP16 bitwise parity (kernel tiling/rounding needs board evidence).
    gen = torch.Generator().manual_seed(4901)
    for p in (1, 63, 64, 65, 299, 313, 320, 321, 384):
        b = action._action_prefix_bucket(p)
        q = torch.randn(2, 32, 16, generator=gen)
        k = torch.randn(2, p + 32, 16, generator=gen)
        v = torch.randn(2, p + 32, 16, generator=gen)
        kb = torch.cat((k[:, :p], torch.zeros(2, b-p, 16), k[:, p:]), dim=1)
        vb = torch.cat((v[:, :p], torch.zeros(2, b-p, 16), v[:, p:]), dim=1)
        mask = torch.zeros(32, b + 32)
        mask[:, p:b] = -torch.inf
        ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        got = torch.nn.functional.scaled_dot_product_attention(q, kb, vb, attn_mask=mask)
        torch.testing.assert_close(got, ref, atol=3e-7, rtol=2e-5)


def test_native_mask_shape_and_layout_are_stable_and_update_outside_capture():
    src = (ROOT / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    setter = src.split("void Qwen3_5Model::set_wall_action_prefix_bucket", 1)[1].split(
        "// Per-forward prefill M-RoPE tables", 1)[0]
    assert "sdpa_stable_mask_cache()" in setter
    assert "narrow(1, prefix_len, bucket_len - prefix_len)" in setter
    assert "-std::numeric_limits<float>::infinity()" in setter
    assert 'A(32 * (384 + 32) * DWIDTH)' in src
    assert 'sdpa_dma_mask_to_spm({action_prefix_mask_, 4}' in src
    py = inspect.getsource(action.run_wall_qwen35_action_loop)
    assert py.index("qwen3_5_set_wall_action_prefix_bucket") < py.index("with cache.capture")
    assert 'shapes=[1, 32, 64, bucket_len]' in py
    assert 'if not state.action_one_graph and cache.size()' in py
    patch = inspect.getsource(action.patch_wall_qwen35_action_for_rpu)
    assert 'max_entries=6 if one_graph else 1' in patch


def test_old_native_without_bucket_abi_is_rejected(monkeypatch):
    import rpu_backend
    from types import SimpleNamespace
    monkeypatch.setattr(rpu_backend, "_cpp_ext", SimpleNamespace(graph_single_segment_abi=1))
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_wall_action_loop", object(), raising=False)
    # Supply an operator namespace that really lacks the new entry.
    monkeypatch.setattr(torch.ops, "rpu", SimpleNamespace(qwen3_5_wall_action_loop=object()))
    with pytest.raises(RuntimeError, match="prefix bucketing"):
        action.require_wall_fp16_loop_runtime()


def test_all_buckets_retained_with_aba_and_frozen_replay(monkeypatch):
    import contextlib
    from types import SimpleNamespace
    import rpu_backend
    monkeypatch.setattr(rpu_backend.graph, "GraphSignature", lambda **kw: tuple(kw["shapes"]))

    class Cache:
        frozen = False

        def __init__(self):
            self.counts = {}

        def lookup(self, sig):
            return self.counts.get(sig)

        def is_frozen(self):
            return self.frozen

        def size(self):
            return len(self.counts)

        @contextlib.contextmanager
        def capture(self, sig):
            self.counts[sig] = self.counts.get(sig, 0) + 1
            yield

        def clear(self):
            pytest.fail("optimized cache must never clear on a new bucket")

    cache = Cache()
    state = SimpleNamespace(action_one_graph=True, action_graph_cache=cache,
                            handle=1, action_k_caches=[], action_v_caches=[])
    expert = SimpleNamespace(_wall_qwen35_action_ready=True, training=False,
                             _rpu_qwen3_5=state)
    original_to = torch.Tensor.to
    monkeypatch.setattr(torch.Tensor, "to", lambda t, *a, **kw:
                        t.clone() if a and a[0] == "rpu" else original_to(t, *a, **kw))
    updates, ropes = [], []
    monkeypatch.setattr(action, "_copy_physical_prefix", lambda s, c, p, b: [b]*24)
    monkeypatch.setattr(action, "_ensure_action_rope", lambda s, p: ropes.append(p.clone()))
    monkeypatch.setattr(action, "_wall_loop_modulation", lambda e: torch.zeros(10, 49, 3072))
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_wall_action_prefix_bucket",
                        lambda h, p, b: updates.append((p, b)), raising=False)
    calls = []

    def native(h, x, keep, padding, out, dt, mod, k, v, lens, steps):
        calls.append(lens[0])
        return out.copy_(x)

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_wall_action_loop", native, raising=False)
    prefixes = [299, 313, 321, 299, 1, 64, 65, 129, 193, 384, 320, 299]
    seen = set()
    for i, p in enumerate(prefixes):
        b = action._action_prefix_bucket(p)
        seen.add(b)
        result = action.run_wall_qwen35_action_loop(
            expert, action=torch.full((1,32,26), float(i)), dof_mask=torch.ones(26),
            padding_velocity=torch.zeros(1,32,26), position_ids=torch.arange(32),
            prefix_cache=object(), prefix_len=p,
        )
        assert result.eq(i).all()
        assert len(cache.counts) == len(seen)
        if len(seen) == 6:
            cache.frozen = True
    assert len(calls) == len(prefixes) + 6  # one prime per previously unseen bucket
    assert sum(cache.counts.values()) == len(prefixes)
    assert updates == [(p, action._action_prefix_bucket(p)) for p in prefixes]
    assert all(torch.equal(p[0], torch.arange(32)) for p in ropes)
