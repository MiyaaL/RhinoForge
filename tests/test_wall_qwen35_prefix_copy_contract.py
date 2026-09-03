"""Board-free contract for the optional Wall prefix-copy diagnostic arm."""

from pathlib import Path
from types import SimpleNamespace

import torch


ROOT = Path(__file__).resolve().parents[1]
ACTION = (ROOT / "python/rpu_backend/adapters/wall_qwen35/action.py").read_text(
    encoding="utf-8"
)
RUNTIME = (ROOT / "python/rpu_backend/adapters/wall_qwen35/runtime.py").read_text(
    encoding="utf-8"
)


def test_prefix_copy_once_is_cold_and_defaults_to_legacy_copying():
    assert '_PREFIX_COPY_ONCE_ENV = "RPU_QWEN35_WALL_PREFIX_COPY_ONCE"' in ACTION
    assert 'os.environ.get(_PREFIX_COPY_ONCE_ENV) == "1"' in ACTION
    assert "state.action_prefix_generation = 0" in ACTION
    assert "state.action_prefix_copy_generation = -1" in ACTION


def test_prefix_copy_once_requires_request_generation_and_full_layer_prefixes():
    assert "generation > 0" in ACTION
    assert "action_prefix_copy_generation" in ACTION
    assert "state.action_prefix_lens[index]" in ACTION
    assert "dst_k[:, :blocks].copy_(src_k[:, :blocks])" in ACTION
    assert "dst_v[:, :blocks].copy_(src_v[:, :blocks])" in ACTION


def test_runtime_stamps_each_public_prefix_build():
    assert "action_state.action_prefix_generation" in RUNTIME
    assert "int(getattr(action_state, \"action_prefix_generation\", 0)) + 1" in RUNTIME


class _FakeDevice:
    type = "rpu"


class _FakeRpuTensor:
    dtype = torch.float16
    ndim = 7
    device = _FakeDevice()

    def __init__(self, blocks: int, token: int):
        self.shape = (1, blocks, 1, 16, 8, 16, 16)
        self._token = token
        self.copy_count = 0

    def data_ptr(self):
        return self._token

    def __getitem__(self, _index):
        return self

    def copy_(self, _source):
        self.copy_count += 1
        return self


def test_prefix_copy_once_fake_tensor_gate(monkeypatch):
    from rpu_backend.adapters.wall_qwen35 import action

    prefix_len = 320
    blocks = 20
    source_k = [_FakeRpuTensor(blocks, 1000 + i) for i in range(24)]
    source_v = [_FakeRpuTensor(blocks, 2000 + i) for i in range(24)]
    dest_k = [_FakeRpuTensor(blocks, 3000 + i) for i in range(24)]
    dest_v = [_FakeRpuTensor(blocks, 4000 + i) for i in range(24)]
    state = SimpleNamespace(
        action_k_caches=dest_k,
        action_v_caches=dest_v,
        action_prefix_lens=[0] * 24,
        action_prefix_len=None,
        action_prefix_generation=1,
        action_prefix_copy_generation=-1,
    )
    cache = SimpleNamespace(position=prefix_len, k_caches=source_k, v_caches=source_v)

    monkeypatch.setenv("RPU_QWEN35_WALL_PREFIX_COPY_ONCE", "1")
    action._copy_physical_prefix(state, cache, prefix_len)
    assert sum(t.copy_count for t in dest_k + dest_v) == 12
    action._copy_physical_prefix(state, cache, prefix_len)
    assert sum(t.copy_count for t in dest_k + dest_v) == 12

    state.action_prefix_generation = 2
    action._copy_physical_prefix(state, cache, prefix_len)
    assert sum(t.copy_count for t in dest_k + dest_v) == 24

    monkeypatch.setenv("RPU_QWEN35_WALL_PREFIX_COPY_ONCE", "0")
    action._copy_physical_prefix(state, cache, prefix_len)
    assert sum(t.copy_count for t in dest_k + dest_v) == 36
