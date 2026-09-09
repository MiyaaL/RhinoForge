from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.wall_qwen35 import checkpoint as wall_checkpoint
from rpu_backend.api import WallQwen35ActionOutput, WallQwen35Policy


@pytest.fixture(autouse=True)
def _isolate_cold_environment(monkeypatch):
    monkeypatch.setattr(os, "environ", os.environ.copy())


def _bound_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    allow_vision: bool,
) -> tuple[WallQwen35Policy, object]:
    profile = object()
    calls: list[Path] = []

    def fake_preflight(path):
        calls.append(Path(path))
        return profile

    monkeypatch.setattr(
        wall_checkpoint, "preflight_wall_qwen35_checkpoint", fake_preflight
    )
    from rpu_backend.adapters.wall_qwen35 import action, execution
    monkeypatch.setenv("WALL_QWEN35_OPT", "1")
    monkeypatch.setattr(action, "require_wall_fp16_loop_runtime", lambda: None)
    monkeypatch.setattr(execution, "configure_wall_qwen35_execution", lambda opt: None)
    policy = WallQwen35Policy.from_checkpoint(
        tmp_path,
        allow_numeric_blocked_vision=allow_vision,
    )
    assert calls == [tmp_path.resolve()]
    assert policy._profile is profile
    assert policy.robot_id == "10070"
    return policy, profile


def test_public_policy_constructor_and_profile_are_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="use WallQwen35Policy.from_checkpoint"):
        WallQwen35Policy()

    calls = 0

    def forbidden_preflight(_path):
        nonlocal calls
        calls += 1
        pytest.fail("invalid public parameters must fail before checkpoint hashing")

    monkeypatch.setattr(
        wall_checkpoint, "preflight_wall_qwen35_checkpoint", forbidden_preflight
    )
    invalid = (
        {"dataset_key": "other"},
        {"robot_id": "10071"},
        {"robot_id": True},
        {"camera_names": ("left_wrist_view", "face_view", "right_wrist_view")},
        {"max_seq_len": 384},
        {"max_seq_len": True},
        {"allow_numeric_blocked_vision": 1},
    )
    for kwargs in invalid:
        with pytest.raises(ValueError):
            WallQwen35Policy.from_checkpoint(tmp_path, **kwargs)
    assert calls == 0


def test_policy_requires_explicit_numeric_blocked_vision_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = _bound_policy(tmp_path, monkeypatch, allow_vision=False)

    with pytest.raises(RuntimeError, match="numeric-blocked.*vision"):
        policy.to("rpu")
    assert policy._runtime is None
    assert policy._install_started is False
    with pytest.raises(ValueError, match=r"only \.to\('rpu'\)"):
        policy.to("cpu")


def test_failed_install_is_irreversible_for_policy_instance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, _ = _bound_policy(tmp_path, monkeypatch, allow_vision=True)
    factory_calls = 0

    class FailingRuntime:
        @classmethod
        def from_checkpoint(cls, _checkpoint, **_kwargs):
            nonlocal factory_calls
            factory_calls += 1
            return cls()

        @staticmethod
        def to(_device):
            raise RuntimeError("synthetic install failure")

    fake_module = ModuleType("rpu_backend.adapters.wall_qwen35.runtime")
    fake_module.WallQwen35Runtime = FailingRuntime
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    with pytest.raises(RuntimeError, match="synthetic install failure"):
        policy.to("rpu")
    assert policy._runtime is None
    assert policy._install_started is True

    with pytest.raises(RuntimeError, match="already attempted"):
        policy.to("rpu")
    assert factory_calls == 1


def test_policy_install_is_idempotent_and_close_is_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy, profile = _bound_policy(tmp_path, monkeypatch, allow_vision=True)
    records = []

    class FakeRuntime:
        def __init__(self) -> None:
            self.close_calls = 0

        @classmethod
        def from_checkpoint(cls, checkpoint, **kwargs):
            records.append((Path(checkpoint), kwargs))
            return cls()

        def to(self, device):
            assert device == "rpu"
            return self

        def close(self):
            self.close_calls += 1

    fake_module = ModuleType("rpu_backend.adapters.wall_qwen35.runtime")
    fake_module.WallQwen35Runtime = FakeRuntime
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    assert policy.to(torch.device("rpu")) is policy
    runtime = policy._runtime
    assert policy.to("rpu") is policy
    assert len(records) == 1
    checkpoint, kwargs = records[0]
    assert checkpoint == tmp_path.resolve()
    assert kwargs == {
        "profile": profile,
        "dataset_key": "x2_normal",
        "robot_id": "10070",
        "camera_names": (
            "face_view",
            "left_wrist_view",
            "right_wrist_view",
        ),
        "max_seq_len": 780,
        "allow_numeric_blocked_vision": True,
        "wall_qwen35_opt": True,
    }

    policy.close()
    policy.close()
    assert runtime.close_calls == 1
    with pytest.raises(RuntimeError, match="closed"):
        policy.to("rpu")
    with pytest.raises(RuntimeError, match="must be live"):
        policy.predict_action_chunk(
            images={}, instruction="task", proprioception=torch.zeros(26)
        )


def test_policy_close_keeps_runtime_reachable_after_cleanup_failure() -> None:
    class FailOnceRuntime:
        def __init__(self) -> None:
            self.calls = 0

        def close(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("synthetic cleanup failure")

    runtime = FailOnceRuntime()
    policy = _live_policy(runtime)

    with pytest.raises(RuntimeError, match="synthetic cleanup failure"):
        policy.close()
    assert policy._runtime is runtime
    assert policy._closed is False

    policy.close()
    assert runtime.calls == 2
    assert policy._runtime is None
    assert policy._closed is True


def test_runtime_close_retires_vision_graphs_handles_then_live_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpu_backend.adapters.qwen3_5 import vision as qwen35_vision
    from rpu_backend.adapters.wall_qwen35.runtime import WallQwen35Runtime
    from rpu_backend.api import causal_lm

    events: list[str] = []

    class Cache:
        def __init__(self, name: str) -> None:
            self.name = name

        def clear(self) -> None:
            events.append(f"clear:{self.name}")

    class Finalizer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.alive = True

        def __call__(self) -> None:
            events.append(f"finalize:{self.name}")
            self.alive = False

    vision = SimpleNamespace(_rpu_vision_handle=7)
    text_state = SimpleNamespace(
        graph_cache=Cache("text"),
        handle_finalizer=Finalizer("text"),
    )
    text = SimpleNamespace(_rpu_qwen3_5=text_state)
    base_model = SimpleNamespace(
        model=SimpleNamespace(visual=vision, language_model=text)
    )
    action_state = SimpleNamespace(
        action_graph_cache=Cache("action"),
        graph_cache=Cache("action-shared"),
        handle_finalizer=Finalizer("action"),
    )
    action = SimpleNamespace(_rpu_qwen3_5=action_state)

    monkeypatch.setattr(
        qwen35_vision,
        "_rollback_qwen3_5_vision_install",
        lambda owner: events.append("rollback:vision")
        if owner is vision
        else pytest.fail("wrong vision owner"),
    )
    monkeypatch.setattr(
        causal_lm,
        "_release_live_instance",
        lambda owner: events.append("release:base")
        if owner is base_model
        else pytest.fail("wrong live owner"),
    )

    runtime = WallQwen35Runtime.__new__(WallQwen35Runtime)
    runtime._closed = False
    runtime._installed = True
    runtime._owned_vision_env = False
    runtime._owned_coexist_persistent_env = False
    runtime.base_cache = object()
    runtime.action_expert = action
    runtime.base_model = base_model
    runtime._base_adapter = object()
    runtime._processor = object()

    runtime.close()

    assert events == [
        "rollback:vision",
        "clear:action",
        "clear:action-shared",
        "finalize:action",
        "clear:text",
        "finalize:text",
        "release:base",
    ]
    assert runtime._closed is True
    assert runtime._installed is False
    assert runtime.base_model is None
    assert runtime.action_expert is None


def _empty_runtime_for_install_test():
    from rpu_backend.adapters.wall_qwen35.runtime import WallQwen35Runtime

    runtime = WallQwen35Runtime.__new__(WallQwen35Runtime)
    runtime._closed = False
    runtime._installed = False
    runtime.allow_numeric_blocked_vision = True
    runtime.wall_qwen35_opt = True
    runtime._owned_vision_env = False
    runtime._owned_coexist_persistent_env = False
    runtime.base_cache = None
    runtime.action_expert = None
    runtime.base_model = None
    runtime._base_adapter = None
    runtime._processor = None
    return runtime


@pytest.mark.parametrize("failure", [None, "forward", "logits", "position"])
def test_prefix_handoff_retires_previous_component_temps_outside_capture(
    monkeypatch, failure,
) -> None:
    from rpu_backend.adapters.wall_qwen35.runtime import WallQwen35Runtime

    events = []
    # Exact failing board watermark: action Temps plus packed Vision would
    # exceed the available arena, though each component fits by itself.
    arena = {"temporary": 3372288, "capturing": False}
    persistent = object()

    def reset_temporary():
        assert not arena["capturing"]
        arena["temporary"] = 0
        events.append("reset_temp")

    monkeypatch.setattr(torch.ops.rpu, "spm_alloc_reset_temporary",
                        reset_temporary, raising=False)

    def reset_cache():
        assert arena["temporary"] == 0
        cache.position = 0
        events.append("reset_cache")

    cache = SimpleNamespace(position=0, persistent=persistent, reset=reset_cache)
    prepared = SimpleNamespace(
        pixel_values=torch.zeros((392, 4)), prefix_input_ids=object(),
        prefix_attention_mask=object(), prefix_position_ids=object(),
        image_grid_thw=object(), prefix_mm_token_type_ids=object(),
        prefix_length=308,
    )

    def forward(**kwargs):
        assert arena["temporary"] + 4817408 <= 7598080
        assert kwargs["past_key_values"] is cache
        assert kwargs["pixel_values"].device.type == "cpu"
        assert kwargs["pixel_values"].dtype == torch.float16
        arena["capturing"] = True
        arena["temporary"] = 4817408
        events.append("prefix")
        try:
            if failure == "forward":
                raise RuntimeError("synthetic prefix failure")
            cache.position = prepared.prefix_length + int(failure == "position")
            return SimpleNamespace(logits=torch.empty((1, 1, 1 if failure == "logits" else 0)))
        finally:
            arena["capturing"] = False

    runtime = WallQwen35Runtime.__new__(WallQwen35Runtime)
    runtime.base_cache = cache
    runtime.base_model = forward
    if failure is not None:
        with pytest.raises(RuntimeError):
            runtime._build_prefix(prepared)
    else:
        for prefix in (308, 310, 311):
            arena["temporary"] = 3372288  # previous completed action component
            prepared.prefix_length = prefix
            assert runtime._build_prefix(prepared) is cache
            assert arena["temporary"] == 0  # Text -> Action handoff
            assert cache.persistent is persistent
    assert arena["temporary"] == 0
    assert events == ["reset_temp", "reset_cache", "prefix", "reset_temp"] * (
        1 if failure else 3
    )


def test_runtime_rejects_conflicting_cold_multi_handle_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key = "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN"
    monkeypatch.setenv(key, "0")
    runtime = _empty_runtime_for_install_test()

    with pytest.raises(RuntimeError, match=f"{key}=1.*fresh process"):
        runtime.install()

    assert runtime._closed is True
    assert runtime._installed is False
    assert runtime._owned_coexist_persistent_env is False
    assert os.environ[key] == "0"


def test_runtime_restores_owned_cold_environment_after_install_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from rpu_backend.adapters.wall_qwen35 import runtime as wall_runtime
    from rpu_backend.adapters.qwen3_5 import vision as vision_adapter

    monkeypatch.setattr(vision_adapter, "_require_packed_spatial_vision_native", lambda: None)
    from rpu_backend.adapters.wall_qwen35 import action
    monkeypatch.setattr(action, "require_wall_fp16_loop_runtime", lambda: None)

    coexist = "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN"
    vision = "QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED"
    monkeypatch.delenv(coexist, raising=False)
    monkeypatch.delenv(vision, raising=False)
    entries = "QWEN3_5_VISION_GRAPH_MAX_ENTRIES"
    monkeypatch.delenv(entries, raising=False)
    def fail_processor(*_args, **_kwargs):
        assert os.environ[entries] == "1"
        raise RuntimeError("synthetic processor failure")

    monkeypatch.setattr(wall_runtime, "_load_processor", fail_processor)
    runtime = _empty_runtime_for_install_test()
    runtime.checkpoint = Path("/synthetic")

    with pytest.raises(RuntimeError, match="synthetic processor failure"):
        runtime.install()

    assert coexist not in os.environ
    assert vision not in os.environ
    assert entries not in os.environ
    assert runtime._closed is True


def test_runtime_rejects_old_vision_native_before_loading(monkeypatch):
    from rpu_backend.adapters.wall_qwen35 import runtime as wall_runtime

    monkeypatch.setattr(torch.ops.rpu, "qwen3_5_vision_forward", object(), raising=False)
    monkeypatch.setenv("RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN", "1")
    monkeypatch.setattr(wall_runtime, "_load_processor",
                        lambda *a, **kw: pytest.fail("must reject before loading"))
    monkeypatch.setattr(wall_runtime, "_load_base_model",
                        lambda *a, **kw: pytest.fail("must reject before loading"))
    runtime = _empty_runtime_for_install_test()
    with pytest.raises(RuntimeError, match="rebuilding the native extension"):
        runtime.install()
    assert runtime._closed


@pytest.mark.parametrize("opt", [True, False])
def test_runtime_graph_labels_pass_real_install_attribute_validation(monkeypatch, opt):
    import rpu_backend
    from rpu_backend.adapters.qwen3_5 import vision
    from rpu_backend.adapters.wall_qwen35 import action, execution
    from rpu_backend.adapters.wall_qwen35 import runtime as wall_runtime
    from rpu_backend.runtime import RPUConfigError
    from rpu_backend.runtime.hw_attrs import (
        install_hw_attr_validator,
        validate_postinstall,
        validate_preinstall,
    )

    for key in (
        "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN",
        "QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED",
        "QWEN3_5_VISION_GRAPH_MAX_ENTRIES",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(execution, "configure_wall_qwen35_execution", lambda opt: None)
    monkeypatch.setattr(vision, "_require_packed_spatial_vision_native", lambda: None)
    monkeypatch.setattr(action, "require_wall_fp16_loop_runtime", lambda: None)
    chunk_calls = []
    for name in ("qwen3_5_set_chunk_size_cap", "qwen3_5_set_chunk_envelope",
                 "qwen3_5_set_prefill_chunk_size"):
        monkeypatch.setattr(torch.ops.rpu, name,
                            lambda *args, _name=name: chunk_calls.append((_name, args)),
                            raising=False)
    monkeypatch.setattr(
        rpu_backend.graph, "GraphCache",
        lambda **kw: SimpleNamespace(**kw, clear=lambda: None),
    )

    model = torch.nn.Module()
    model.config = SimpleNamespace(text_config=SimpleNamespace())
    model.model = torch.nn.Module()
    model.model.visual = torch.nn.Module()
    model.model.language_model = torch.nn.Module()
    monkeypatch.setattr(wall_runtime, "_load_processor", lambda *a, **kw: object())
    monkeypatch.setattr(wall_runtime, "_load_base_model", lambda *a, **kw: model)
    installs = []

    class Adapter:
        def __init__(self, base):
            self.model = base

        def to_rpu(self, **kwargs):
            # Exercise the real recursive pre/post-install checks and validator
            # stamp at the actual Wall -> Qwen adapter boundary. Only device
            # weight installation is doubled; SimpleNamespace-only towers in
            # forward tests cannot catch a cold attribute namespace violation.
            validate_preinstall(self.model)
            self.model.model.language_model._rpu_qwen3_5 = SimpleNamespace(handle=42)
            validate_postinstall(self.model)
            install_hw_attr_validator(self.model)
            installs.append(kwargs)

    monkeypatch.setattr(wall_runtime, "Qwen3_5Adapter", Adapter)
    monkeypatch.setattr(wall_runtime.Qwen3_5Cache, "from_config", lambda *a, **kw: object())
    expert = SimpleNamespace(_rpu_qwen3_5=SimpleNamespace())
    monkeypatch.setattr(action, "load_wall_qwen35_action_module", lambda *a, **kw: expert)
    monkeypatch.setattr(action, "patch_wall_qwen35_action_for_rpu", lambda obj, **kw: obj)
    runtime = _empty_runtime_for_install_test()
    runtime.wall_qwen35_opt = opt
    runtime.checkpoint = Path("/synthetic")
    runtime._manifest = object()
    runtime.max_seq_len = 512
    runtime._action_min = runtime._action_delta = torch.zeros(26)
    try:
        assert runtime.install() is runtime
        assert runtime.install() is runtime
        assert len(installs) == 1
        assert chunk_calls == ([
            ("qwen3_5_set_chunk_size_cap", (42, 0)),
            ("qwen3_5_set_chunk_envelope", (42, 384, 384)),
            ("qwen3_5_set_prefill_chunk_size", (42, 384)),
        ] if opt else [])
        tower = model.model.visual
        assert tower._wall_qwen35_packed_vision is opt
        assert tower._wall_qwen35_vision_graph_op_id == "rpu_wall_qwen35_vision"
        assert not hasattr(tower, "_rpu_vision_graph_op_id")
        validate_postinstall(model)
        # The fix must not relax unknown/stale hardware-attribute rejection.
        with pytest.raises(RPUConfigError, match="Unknown hardware attribute"):
            tower._rpu_vision_graph_op_id = "stale"
        stale = torch.nn.Module()
        stale._rpu_vision_graph_op_id = "stale"
        with pytest.raises(RPUConfigError, match="BEFORE"):
            validate_preinstall(stale)
    finally:
        runtime.close()


class _PredictRuntime:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.action_storage = torch.zeros((1, 32, 52), dtype=torch.float32)
        self.norm_storage = torch.zeros((1, 32, 52), dtype=torch.float32)

    def predict_action_chunk(self, **kwargs):
        self.calls.append(kwargs)
        call_value = float(len(self.calls))
        self.action_storage.fill_(call_value)
        self.norm_storage.fill_(-call_value)
        return {
            "actions": self.action_storage[..., ::2],
            "actions_norm": self.norm_storage[..., ::2],
            "prefix_length": 374,
            "extra": {"call": len(self.calls)},
        }

    def close(self) -> None:
        pass


def _live_policy(runtime) -> WallQwen35Policy:
    policy = WallQwen35Policy.__new__(WallQwen35Policy)
    policy._checkpoint = Path("/unused")
    policy._profile = object()
    policy._dataset_key = "x2_normal"
    policy._robot_id = "10070"
    policy._camera_names = (
        "face_view",
        "left_wrist_view",
        "right_wrist_view",
    )
    policy._max_seq_len = 780
    policy._allow_numeric_blocked_vision = True
    policy._runtime = runtime
    policy._install_started = True
    policy._closed = False
    return policy


def test_predict_validates_inputs_orders_cameras_and_clones_outputs() -> None:
    runtime = _PredictRuntime()
    policy = _live_policy(runtime)
    images = {
        "right_wrist_view": object(),
        "face_view": object(),
        "left_wrist_view": object(),
    }
    first = policy.predict_action_chunk(
        images=images,
        instruction={"task": "pick", "detail": "slowly"},
        proprioception=torch.arange(26),
        agent_pos_mask=[1] * 20 + [0] * 6,
        dof_mask=[1] * 20 + [0] * 6,
        noise_seed=3407,
    )

    assert isinstance(first, WallQwen35ActionOutput)
    assert tuple(first.actions.shape) == (1, 32, 26)
    assert tuple(first.actions_norm.shape) == (1, 32, 26)
    assert first.actions.is_contiguous() and first.actions_norm.is_contiguous()
    assert first.actions.data_ptr() != runtime.action_storage.data_ptr()
    assert first.actions_norm.data_ptr() != runtime.norm_storage.data_ptr()
    assert first.prefix_length == 374
    assert list(runtime.calls[0]["images"]) == list(policy.camera_names)
    assert runtime.calls[0]["proprioception"].dtype == torch.float32
    assert runtime.calls[0]["proprioception"].device.type == "cpu"
    assert runtime.calls[0]["noise_seed"] == 3407

    second = policy.predict_action_chunk(
        images=images,
        instruction="pick",
        proprioception=torch.zeros(26),
    )
    assert torch.all(first.actions == 1)
    assert torch.all(second.actions == 2)
    assert runtime.calls[1]["agent_pos_mask"].tolist() == [1.0] * 20 + [0.0] * 6
    assert runtime.calls[1]["dof_mask"].tolist() == [1.0] * 20 + [0.0] * 6

    initial_noise = torch.randn((32, 26), dtype=torch.float32)
    explicit = policy.predict_action_chunk(
        images=images,
        instruction="pick",
        proprioception=torch.zeros(26),
        initial_noise=initial_noise,
    )
    assert torch.all(explicit.actions == 3)
    passed_noise = runtime.calls[2]["initial_noise"]
    assert tuple(passed_noise.shape) == (1, 32, 26)
    assert passed_noise.is_contiguous()
    assert passed_noise.data_ptr() != initial_noise.data_ptr()
    torch.testing.assert_close(passed_noise[0], initial_noise)
    assert runtime.calls[2]["noise_seed"] is None


def test_predict_rejects_bad_request_and_runtime_output() -> None:
    policy = _live_policy(_PredictRuntime())
    cameras = {name: object() for name in policy.camera_names}

    with pytest.raises(ValueError, match="exactly"):
        policy.predict_action_chunk(
            images={**cameras, "extra": object()},
            instruction="task",
            proprioception=torch.zeros(26),
        )
    with pytest.raises(ValueError, match="noise_seed"):
        policy.predict_action_chunk(
            images=cameras,
            instruction="task",
            proprioception=torch.zeros(26),
            noise_seed=True,
        )
    with pytest.raises(ValueError, match="either noise_seed or initial_noise"):
        policy.predict_action_chunk(
            images=cameras,
            instruction="task",
            proprioception=torch.zeros(26),
            noise_seed=1,
            initial_noise=torch.zeros((32, 26)),
        )
    with pytest.raises(ValueError, match=r"shape \[32,26\]"):
        policy.predict_action_chunk(
            images=cameras,
            instruction="task",
            proprioception=torch.zeros(26),
            initial_noise=torch.zeros((31, 26)),
        )
    nonfinite_noise = torch.zeros((32, 26))
    nonfinite_noise[0, 0] = float("nan")
    with pytest.raises(ValueError, match="only finite"):
        policy.predict_action_chunk(
            images=cameras,
            instruction="task",
            proprioception=torch.zeros(26),
            initial_noise=nonfinite_noise,
        )
    with pytest.raises(ValueError, match="only 0 or 1"):
        policy.predict_action_chunk(
            images=cameras,
            instruction="task",
            proprioception=torch.zeros(26),
            dof_mask=[1] * 25 + [0.5],
        )
    with pytest.raises(ValueError, match=r"dof_mask must be \[1\]\*20"):
        policy.predict_action_chunk(
            images=cameras,
            instruction="task",
            proprioception=torch.zeros(26),
            dof_mask=[1] * 26,
        )

    class BadRuntime:
        @staticmethod
        def predict_action_chunk(**_kwargs):
            return torch.zeros((1, 31, 26))

    policy._runtime = BadRuntime()
    with pytest.raises(RuntimeError, match="invalid action shape"):
        policy.predict_action_chunk(
            images=cameras,
            instruction="task",
            proprioception=torch.zeros(26),
        )
