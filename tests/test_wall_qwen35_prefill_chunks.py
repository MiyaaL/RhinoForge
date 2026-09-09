"""Cold Wall-only admission and wide-prefill manifest regression checks."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.wall_qwen35.runtime import _configure_prefill_chunks


def test_legacy_does_not_touch_chunk_controls():
    _configure_prefill_chunks(SimpleNamespace(), optimized=False)


def test_native_admission_failure_is_not_silently_split(monkeypatch):
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_chunk_size_cap",
                        lambda *args: None, raising=False)
    def reject(*args):
        raise RuntimeError("must be called before the first forward")
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_chunk_envelope", reject,
                        raising=False)
    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_set_prefill_chunk_size",
                        lambda *args: pytest.fail("must stop on cold guard"),
                        raising=False)
    with pytest.raises(RuntimeError, match="before the first forward"):
        _configure_prefill_chunks(SimpleNamespace(handle=42), optimized=True)


def test_wide_manifest_separates_only_mutually_exclusive_mixers():
    root = Path(__file__).resolve().parents[1]
    cpp = (root / "src/fused/rpu_qwen3_5_model.cpp").read_text()
    manifest = cpp.split("Qwen3_5Model::declare_buffers(", 1)[1].split(
        "return decls;", 1)[0]
    assert "!action_mode_ && cs > 128" in manifest
    assert "mixer_shift = wide_prefill ? 10 : 0" in manifest
    assert "mlp_begin = wide_prefill ? 21 : 7" in manifest
    assert "layer_end = wide_prefill ? 22 : 8" in manifest
    assert '"residual1",  res,  1, layer_end' in manifest
    assert '"input_norm", res,  1, layer_end' in manifest
    assert '"residual2",  0,    0, 0, StorageClass::Temp, 0, "input_norm"' in manifest
    assert "1 + mixer_shift, 6 + mixer_shift" in manifest
    assert "p0 + mixer_shift, p1 + mixer_shift" in manifest
    assert 'BP("gdn_c_tril",   64 * 64, 1, 4)' in manifest
    assert 'BP("gdn_c_strict", 64 * 64, 1, 4)' in manifest
    assert "const int64_t C = 64, N = L / C" in cpp


def test_generic_chunk_envelopes_remain_128():
    from rpu_backend.adapters.qwen3_5.text import (
        CERTIFIED_CHUNK_ENVELOPE, _VISION_CHUNK_ENVELOPE,
    )
    assert {v[1] for v in CERTIFIED_CHUNK_ENVELOPE.values()} == {128}
    assert {v[1] for v in _VISION_CHUNK_ENVELOPE.values()} == {128}
