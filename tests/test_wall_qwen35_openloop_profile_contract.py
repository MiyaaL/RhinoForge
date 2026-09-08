from __future__ import annotations

import hashlib
import inspect
import json
import numpy as np
from pathlib import Path
import re
import runpy
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "wall_qwen35_openloop.py"
WRAPPER = ROOT / "run_wall_qwen35_openloop.sh"


@pytest.fixture(scope="module")
def openloop_namespace():
    return runpy.run_path(str(EXAMPLE))


def _profile_args(**overrides):
    values = {
        "torch_profile": False,
        "torch_profile_output": None,
        "torch_profile_dir": None,
        "torch_profile_record_shapes": False,
        "torch_profile_memory": False,
        "torch_profile_with_stack": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_flow_noise_schedule_round_trips_common_artifact(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    schedule = openloop_namespace["_flow_noise_schedule"]
    generated, metadata = schedule(None, request_count=2, base_seed=3407)

    assert generated.shape == (2, 32, 26)
    assert generated.dtype == np.float32
    assert metadata["kind"] == "generated_cpu_per_request"
    assert metadata["base_seed"] == 3407
    assert metadata["value_sha256"] == hashlib.sha256(
        generated.tobytes()
    ).hexdigest()

    artifact = tmp_path / "flow_noise.npy"
    np.save(artifact, generated[:, None])
    loaded, loaded_metadata = schedule(
        artifact, request_count=2, base_seed=9999
    )

    np.testing.assert_array_equal(loaded, generated)
    assert loaded_metadata["kind"] == "explicit_npy"
    assert loaded_metadata["source_path"] == str(artifact.resolve())
    assert loaded_metadata["value_sha256"] == metadata["value_sha256"]


def test_flow_noise_schedule_rejects_wrong_shape_and_nonfinite(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    schedule = openloop_namespace["_flow_noise_schedule"]
    wrong = tmp_path / "wrong.npy"
    np.save(wrong, np.zeros((1, 32, 26), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        schedule(wrong, request_count=2, base_seed=3407)

    nonfinite = tmp_path / "nonfinite.npy"
    values = np.zeros((2, 32, 26), dtype=np.float32)
    values[0, 0, 0] = np.nan
    np.save(nonfinite, values)
    with pytest.raises(ValueError, match="only finite"):
        schedule(nonfinite, request_count=2, base_seed=3407)


def test_profile_is_disabled_by_default(openloop_namespace, tmp_path: Path) -> None:
    create = openloop_namespace["_create_torch_profiler"]

    profiler, metadata = create(_profile_args(), tmp_path / "run")

    assert profiler is None
    assert metadata == {
        "enabled": False,
        "mode": "disabled",
        "scope": "post-install open-loop inference requests",
        "includes_graph_build": False,
        "includes_lazy_first_request_setup": False,
        "profiler_acc_events": False,
        "accumulate_request_events": False,
        "latency_is_diagnostic_only": False,
        "record_shapes": False,
        "profile_memory": False,
        "with_stack": False,
        "output": None,
        "summary_output": None,
        "directory": None,
        "activities": [],
        "artifacts": [],
        "sensitive_artifact": False,
        "admission": None,
    }


def test_profile_requests_cpu_and_privateuse1_and_uses_default_trace(
    openloop_namespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = openloop_namespace["torch"]
    create = openloop_namespace["_create_torch_profiler"]
    marker = object()
    captured: dict = {}

    def fake_profile(**kwargs):
        captured.update(kwargs)
        return marker

    monkeypatch.setattr(torch.profiler, "profile", fake_profile)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    profiler, metadata = create(
        _profile_args(
            torch_profile=True,
            torch_profile_record_shapes=True,
            torch_profile_memory=True,
            torch_profile_with_stack=True,
        ),
        output_dir,
    )

    assert profiler is marker
    assert captured["activities"][0] is torch.profiler.ProfilerActivity.CPU
    private_use = getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)
    expected_names = ["CPU"]
    if private_use is not None:
        assert captured["activities"][1] is private_use
        expected_names.append("PrivateUse1")
    assert captured["record_shapes"] is True
    assert captured["profile_memory"] is True
    assert captured["with_stack"] is True
    assert captured["acc_events"] is True
    assert metadata["activities"] == expected_names
    assert metadata["mode"] == "ready_replay"
    trace_path = Path(metadata["output"])
    summary_path = Path(metadata["summary_output"])
    assert trace_path.parent == output_dir
    assert re.fullmatch(
        r"wall_qwen35_torch_profile_\d{8}_\d{6}_\d+\.trace\.json",
        trace_path.name,
    )
    assert summary_path.parent == output_dir
    assert summary_path.name == trace_path.name.replace(
        ".trace.json", ".summary.json"
    )
    assert metadata["directory"] == str(output_dir)
    assert metadata["scope"] == (
        "one repeated first request after an unprofiled warmup"
    )
    assert metadata["includes_graph_build"] is None
    assert metadata["includes_lazy_first_request_setup"] is False
    assert metadata["profiler_acc_events"] is True
    assert metadata["accumulate_request_events"] is False


def test_reference_profile_directory_forces_per_request_shapes_and_stacks(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    create = openloop_namespace["_create_torch_profiler"]
    profile_dir = tmp_path / "profiles"

    profiler, metadata = create(
        _profile_args(torch_profile_dir=profile_dir),
        tmp_path / "run",
    )

    assert profiler is None
    assert profile_dir.is_dir()
    assert metadata["mode"] == "per_request"
    assert metadata["directory"] == str(profile_dir)
    assert metadata["output"] is None
    assert metadata["summary_output"] is None
    assert metadata["record_shapes"] is True
    assert metadata["with_stack"] is True
    assert metadata["profiler_acc_events"] is True
    assert metadata["accumulate_request_events"] is False


def test_profile_modes_are_mutually_exclusive(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    create = openloop_namespace["_create_torch_profiler"]

    with pytest.raises(SystemExit, match="choose either ready-replay"):
        create(
            _profile_args(
                torch_profile=True,
                torch_profile_dir=tmp_path / "profiles",
            ),
            tmp_path / "run",
        )


def test_ready_replay_profile_output_is_a_reusable_directory(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    plan = openloop_namespace["_plan_torch_profile"]
    create = openloop_namespace["_create_torch_profiler"]
    output_dir = tmp_path / "run"
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()

    metadata = plan(
        _profile_args(
            torch_profile=True,
            torch_profile_output=profile_dir,
        ),
        output_dir,
    )
    trace_path = Path(metadata["output"])
    summary_path = Path(metadata["summary_output"])

    assert metadata["directory"] == str(profile_dir)
    assert trace_path.parent == profile_dir
    assert re.fullmatch(
        r"wall_qwen35_torch_profile_\d{8}_\d{6}_\d+\.trace\.json",
        trace_path.name,
    )
    assert summary_path.name == trace_path.name.replace(
        ".trace.json", ".summary.json"
    )

    trace_path.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit, match="already exists"):
        create(
            _profile_args(
                torch_profile=True,
                torch_profile_output=profile_dir,
            ),
            output_dir,
            profile_plan=metadata,
        )
    assert trace_path.read_text(encoding="utf-8") == "keep"


def test_ready_replay_profile_rejects_a_file_as_its_directory(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    plan = openloop_namespace["_plan_torch_profile"]
    output_dir = tmp_path / "run"
    profile_file = tmp_path / "profile.json"
    profile_file.write_text("keep", encoding="utf-8")

    with pytest.raises(SystemExit, match="not a directory"):
        plan(
            _profile_args(
                torch_profile=True,
                torch_profile_output=profile_file,
            ),
            output_dir,
        )


def test_profile_destination_is_planned_before_backend_import() -> None:
    main_source = EXAMPLE.read_text(encoding="utf-8").split("def main() -> int:", 1)[1]

    assert main_source.index("profile_plan = _plan_torch_profile") < main_source.index(
        "from rpu_backend.api import WallQwen35Policy"
    )


def test_profiled_policy_call_marks_one_step(
    openloop_namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = openloop_namespace["torch"]
    predict = openloop_namespace["_profiled_predict_action_chunk"]
    events: list[str] = []

    class Scope:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self):
            events.append(f"scope-enter:{self.name}")

        def __exit__(self, exc_type, exc, traceback):
            events.append(f"scope-exit:{self.name}")

    class Profiler:
        @staticmethod
        def step() -> None:
            events.append("step")

    class Policy:
        @staticmethod
        def predict_action_chunk(**kwargs):
            assert torch.is_inference_mode_enabled()
            events.append(f"predict:{kwargs['noise_seed']}")
            return "result"

    def record_function(name: str):
        assert name in {
            "generate_flow_action_batch",
            "wall_qwen35_openloop::predict_action_chunk",
        }
        return Scope(name)

    monkeypatch.setattr(torch.profiler, "record_function", record_function)
    result = predict(Policy(), Profiler(), mark_step=True, noise_seed=3407)

    assert result == "result"
    assert events == [
        "scope-enter:generate_flow_action_batch",
        "scope-enter:wall_qwen35_openloop::predict_action_chunk",
        "predict:3407",
        "scope-exit:wall_qwen35_openloop::predict_action_chunk",
        "scope-exit:generate_flow_action_batch",
        "step",
    ]


def test_backend_model_stage_scope_is_visible_to_kineto() -> None:
    import torch

    from rpu_backend.adapters.wall_qwen35.runtime import _profile_scope

    names = (
        "wall_qwen35_preprocess",
        "wall_qwen35_vision_text_prefill",
        "wall_qwen35_action_denoise_loop",
        "wall_qwen35_action_decoder",
    )
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        acc_events=True,
    ) as profiler:
        for name in names:
            with _profile_scope(name):
                torch.ones(1)

    observed = {event.key for event in profiler.key_averages()}
    assert set(names) <= observed


def test_unprofiled_policy_call_keeps_default_path_idle(
    openloop_namespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = openloop_namespace["torch"]
    predict = openloop_namespace["_profiled_predict_action_chunk"]

    def unexpected_record_function(_name: str):
        pytest.fail("record_function must stay idle when profiling is disabled")

    class Policy:
        @staticmethod
        def predict_action_chunk(**kwargs):
            assert torch.is_inference_mode_enabled()
            return kwargs["value"]

    monkeypatch.setattr(
        torch.profiler,
        "record_function",
        unexpected_record_function,
    )
    assert predict(Policy(), None, value="unprofiled") == "unprofiled"


def test_profile_export_records_size_and_sha256(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    export = openloop_namespace["_export_torch_profile"]
    trace = b'{"traceEvents": []}\n'
    output = tmp_path / "trace.json"

    class Profiler:
        @staticmethod
        def export_chrome_trace(path: str) -> None:
            Path(path).write_bytes(trace)

    metadata = {"output": str(output)}
    export(Profiler(), metadata)

    assert output.read_bytes() == trace
    assert metadata["trace_bytes"] == len(trace)
    assert metadata["trace_sha256"] == hashlib.sha256(trace).hexdigest()


def test_ready_replay_export_writes_compact_summary(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    export = openloop_namespace["_export_torch_profile"]
    trace = b'{"traceEvents": []}\n'
    output = tmp_path / "profile.trace.json"
    summary = tmp_path / "profile.summary.json"

    class Event:
        key = "wall_qwen35_action_denoise_loop"
        count = 1
        self_cpu_time_total = 10.0
        cpu_time_total = 20.0
        self_device_time_total = 30.0
        device_time_total = 40.0

    class Profiler:
        @staticmethod
        def export_chrome_trace(path: str) -> None:
            Path(path).write_bytes(trace)

        @staticmethod
        def key_averages():
            return [Event()]

    metadata = {
        "output": str(output),
        "summary_output": str(summary),
        "artifacts": [],
        "admission": {"status": "runtime_extension_required"},
    }
    export(Profiler(), metadata)

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["trace"]["path"] == str(output)
    assert payload["profile"]["admission"]["status"] == (
        "runtime_extension_required"
    )
    assert payload["key_averages"] == [
        {
            "name": "wall_qwen35_action_denoise_loop",
            "count": 1,
            "self_cpu_time_us": 10.0,
            "cpu_time_us": 20.0,
            "self_device_time_us": 30.0,
            "device_time_us": 40.0,
        }
    ]
    assert metadata["summary_bytes"] == summary.stat().st_size
    assert metadata["summary_sha256"] == hashlib.sha256(
        summary.read_bytes()
    ).hexdigest()


def test_ready_replay_admission_rejects_oneshot_prefill_and_vision_rebuild(
    openloop_namespace,
) -> None:
    classify = openloop_namespace["_ready_replay_admission"]
    torch = openloop_namespace["torch"]

    def retained(replays: int):
        return {
            "size": 1,
            "max_entries": 1,
            "phase": "READY",
            "replays": replays,
            "recaptures": 0,
            "entries": [
                {
                    "signature": "sig",
                    "kernel_count": 4,
                    "segment_count": 1,
                    "local_spm_slot_count": 0,
                    "data_node_count": 0,
                    "replay_count": replays,
                    "recapture_count": 0,
                    "non_replayable_reason": "",
                }
            ],
            "invariant_ok": True,
        }

    before = {
        "action": retained(9),
        "vision": retained(1),
        "base_prefill": {"contract": "bounded one-shot; no replay claim"},
    }
    after = {
        "action": retained(19),
        # The single-entry cache rebuilt face then wrist, ending with the same
        # signature and replay count instead of replaying all three calls.
        "vision": retained(1),
        "base_prefill": {"contract": "bounded one-shot; no replay claim"},
    }
    warm_output = SimpleNamespace(
        actions=torch.zeros((1, 32, 26)),
        actions_norm=torch.zeros((1, 32, 26)),
        prefix_length=299,
    )
    measured_output = SimpleNamespace(
        actions=warm_output.actions.clone(),
        actions_norm=warm_output.actions_norm.clone(),
        prefix_length=299,
    )

    admission = classify(before, after, warm_output, measured_output)

    assert admission["status"] == "runtime_extension_required"
    assert admission["full_ready"] is False
    assert admission["components"]["action"]["accepted"] is True
    assert admission["components"]["vision"]["accepted"] is False
    assert admission["components"]["base_prefill"]["accepted"] is False
    assert admission["output_parity"]["accepted"] is True


def test_ready_replay_admission_accepts_retained_prefix_bucket(
    openloop_namespace,
) -> None:
    classify = openloop_namespace["_ready_replay_admission"]
    torch = openloop_namespace["torch"]

    def retained(replays: int, *, signature: str = "sig", phase: str = "READY"):
        return {
            "size": 1,
            "max_entries": 7,
            "phase": phase,
            "replays": replays,
            "recaptures": 0,
            "entries": [
                {
                    "signature": signature,
                    "kernel_count": 4,
                    "segment_count": 1,
                    "local_spm_slot_count": 0,
                    "data_node_count": 2,
                    "replay_count": replays,
                    "recapture_count": 0,
                    "non_replayable_reason": "",
                }
            ],
            "invariant_ok": True,
        }

    before = {
        "action": retained(9),
        "vision": retained(1),
        "base_prefill": retained(0, signature="bucket=320"),
    }
    after = {
        "action": retained(19),
        "vision": retained(2),
        "base_prefill": retained(1, signature="bucket=320"),
    }
    warm_output = SimpleNamespace(
        actions=torch.zeros((1, 32, 26)),
        actions_norm=torch.zeros((1, 32, 26)),
        prefix_length=299,
    )
    measured_output = SimpleNamespace(
        actions=warm_output.actions.clone(),
        actions_norm=warm_output.actions_norm.clone(),
        prefix_length=299,
    )

    admission = classify(before, after, warm_output, measured_output)

    assert admission["status"] == "accepted"
    assert admission["full_ready"] is True
    assert admission["output_parity"]["actions_norm_exact"] is True
    assert admission["components"]["base_prefill"]["accepted"] is True
    assert admission["components"]["base_prefill"]["contract"] == (
        "retained prefix bucket Graph"
    )

    configuring_before = {
        name: retained(
            int(stats["replays"]),
            signature=stats["entries"][0]["signature"],
            phase="CONFIGURING",
        )
        for name, stats in before.items()
    }
    configuring = classify(
        configuring_before, after, warm_output, measured_output
    )
    assert configuring["status"] == "runtime_extension_required"
    assert configuring["full_ready"] is False
    assert all(
        component["ready_phase"] is False
        for component in configuring["components"].values()
    )

    missing_normalized = SimpleNamespace(
        actions=warm_output.actions.clone(),
        actions_norm=None,
        prefix_length=299,
    )
    missing = classify(before, after, warm_output, missing_normalized)
    assert missing["full_ready"] is False
    assert missing["output_parity"]["actions_norm_present"] is False

    wrong_normalized_shape = SimpleNamespace(
        actions=warm_output.actions.clone(),
        actions_norm=torch.zeros((1, 31, 26)),
        prefix_length=299,
    )
    wrong_shape_warm = SimpleNamespace(
        actions=warm_output.actions.clone(),
        actions_norm=wrong_normalized_shape.actions_norm.clone(),
        prefix_length=299,
    )
    wrong_shape = classify(
        before, after, wrong_shape_warm, wrong_normalized_shape
    )
    assert wrong_shape["full_ready"] is False
    assert wrong_shape["output_parity"]["actions_norm_same_shape"] is False

    drifted_normalized = SimpleNamespace(
        actions=warm_output.actions.clone(),
        actions_norm=warm_output.actions_norm.clone(),
        prefix_length=299,
    )
    drifted_normalized.actions_norm[0, 0, 0] = 1.0
    drifted = classify(before, after, warm_output, drifted_normalized)
    assert drifted["full_ready"] is False
    assert drifted["output_parity"]["actions_norm_same_shape"] is True
    assert drifted["output_parity"]["actions_norm_finite"] is True
    assert drifted["output_parity"]["actions_norm_exact"] is False
    assert drifted["output_parity"]["actions_norm_max_abs"] == 1.0


def test_ready_probe_freezes_then_restores_all_three_caches(
    openloop_namespace,
) -> None:
    freeze = openloop_namespace["_frozen_policy_graph_caches"]
    events = []

    class Cache:
        phase = "CONFIGURING"

        def __init__(self, name: str) -> None:
            self.name = name

        def freeze(self):
            events.append(f"freeze:{self.name}")
            self.phase = "READY"

        def begin_warmup(self):
            events.append(f"warm:{self.name}")
            self.phase = "WARMING"

    action = Cache("action")
    vision = Cache("vision")
    prefill = Cache("prefill")
    policy = SimpleNamespace(
        _runtime=SimpleNamespace(
            action_expert=SimpleNamespace(
                _rpu_qwen3_5=SimpleNamespace(action_graph_cache=action)
            ),
            base_model=SimpleNamespace(
                model=SimpleNamespace(
                    visual=SimpleNamespace(_rpu_vision_graph_cache=vision),
                    language_model=SimpleNamespace(
                        _rpu_qwen3_5=SimpleNamespace(
                            prefill_graph_cache=prefill
                        )
                    ),
                )
            ),
        )
    )

    with freeze(policy) as caches:
        assert all(cache.phase == "READY" for cache in caches.values())
    assert all(cache.phase == "WARMING" for cache in caches.values())
    assert events == [
        "freeze:action",
        "freeze:vision",
        "freeze:prefill",
        "warm:prefill",
        "warm:vision",
        "warm:action",
    ]

    events.clear()
    action.phase = vision.phase = prefill.phase = "CONFIGURING"

    def fail_freeze():
        events.append("freeze:vision-failed")
        raise RuntimeError("freeze failed")

    vision.freeze = fail_freeze
    with pytest.raises(RuntimeError, match="freeze failed"):
        with freeze(policy):
            pytest.fail("partial freeze must not enter the READY body")
    assert action.phase == "WARMING"
    assert vision.phase == "CONFIGURING"
    assert prefill.phase == "CONFIGURING"
    assert events == [
        "freeze:action",
        "freeze:vision-failed",
        "warm:action",
    ]


def test_prefix_bucket_selection_and_plan_are_fixed_shape() -> None:
    from rpu_backend.adapters.qwen3_5.text import (
        _plan_prefill_bucket,
        _prefill_bucket_tail_class,
        _select_prefill_bucket,
    )

    buckets = (64, 128, 192, 256, 320, 384)
    assert _select_prefill_bucket(299, buckets) == 320
    assert _select_prefill_bucket(320, buckets) == 320
    assert _select_prefill_bucket(321, buckets) == 384
    assert _select_prefill_bucket(384, buckets) == 384
    with pytest.raises(RuntimeError, match="largest retained bucket 384"):
        _select_prefill_bucket(385, buckets)

    calls = []

    def resolve(length: int) -> int:
        calls.append(length)
        return 128

    assert _plan_prefill_bucket(
        313,
        320,
        848,
        64,
        resolve,
    ) == (320, 128)
    assert calls == [320]
    assert _plan_prefill_bucket(320, 320, 848, 64, resolve) == (320, 128)
    with pytest.raises(RuntimeError, match="over the configured budget"):
        _plan_prefill_bucket(255, 320, 848, 64, resolve)

    # These three real lengths share bucket=320/chunk=128 but cross native
    # convolution node/grid envelopes, so they must not share Graph identity.
    assert _prefill_bucket_tail_class(257, 320, 128) == 1
    assert _prefill_bucket_tail_class(258, 320, 128) == 2
    assert _prefill_bucket_tail_class(287, 320, 128) == 2
    assert _prefill_bucket_tail_class(288, 320, 128) == 3
    assert _prefill_bucket_tail_class(320, 320, 128) == 3


def test_prefill_bucket_plan_cache_keeps_real_length_in_its_key() -> None:
    from rpu_backend.adapters.qwen3_5.text import run_qwen3_5_text

    source = inspect.getsource(run_qwen3_5_text)
    assert re.search(
        r"plan_key\s*=\s*\(\s*real_len,\s*prefill_bucket,",
        source,
    )


def test_vision_graph_capacity_is_monotonic_after_install() -> None:
    import torch

    from rpu_backend.api.errors import RPUConfigError
    from rpu_backend.runtime.hw_attrs import install_hw_attr_validator

    module = torch.nn.Module()
    install_hw_attr_validator(module)
    module._rpu_vision_graph_max_entries = 2
    with pytest.raises(RPUConfigError, match="monotonic"):
        module._rpu_vision_graph_max_entries = 1


def test_per_request_profile_uses_reference_filename_and_artifact_manifest(
    openloop_namespace,
    tmp_path: Path,
) -> None:
    export = openloop_namespace["_export_torch_profile"]
    request_path = openloop_namespace["_request_trace_path"]
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir()
    metadata = {"directory": str(profile_dir), "artifacts": []}
    output = request_path(metadata, 1)
    trace = b'{"traceEvents": []}\n'

    class Profiler:
        @staticmethod
        def export_chrome_trace(path: str) -> None:
            Path(path).write_bytes(trace)

    export(Profiler(), metadata, trace_path=output, request_index=1)

    assert output.parent == profile_dir
    assert output.name.startswith("qwen35_generate_flow_action_batch_")
    assert "_000002_" in output.name
    assert output.name.endswith(".trace.json.gz")
    assert metadata["artifacts"] == [
        {
            "path": str(output),
            "trace_bytes": len(trace),
            "trace_sha256": hashlib.sha256(trace).hexdigest(),
            "request_idx": 1,
        }
    ]


def test_wrapper_exposes_and_forwards_profile_options() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--torch-profile" in result.stdout
    assert "--torch-profile-output" in result.stdout
    assert "--torch-profile-dir" in result.stdout
    assert "--torch-profile-record-shapes" in result.stdout
    assert "--torch-profile-memory" in result.stdout
    assert "--torch-profile-with-stack" in result.stdout
    assert "READY-probe output directory" in result.stdout
    assert "Vision defaults to one Graph" in result.stdout
    assert "No extra enable flag is needed" in result.stdout

    source = WRAPPER.read_text(encoding="utf-8")
    assert '--torch-profile-output "$TORCH_PROFILE_OUTPUT"' in source
    assert '--torch-profile-dir "$TORCH_PROFILE_DIR"' in source
    assert "PYTHON_ARGS+=(--torch-profile-record-shapes)" in source
    assert "PYTHON_ARGS+=(--torch-profile-memory)" in source
    assert "PYTHON_ARGS+=(--torch-profile-with-stack)" in source


def test_runner_rejects_old_installed_vision_before_checkpoint_load(
    openloop_namespace, monkeypatch,
) -> None:
    from rpu_backend.adapters.qwen3_5 import vision

    monkeypatch.delattr(vision, "_require_packed_spatial_vision_native")
    with pytest.raises(RuntimeError, match="rebuild and reinstall"):
        openloop_namespace["_require_packed_vision_runtime"]()
    source = inspect.getsource(openloop_namespace["main"])
    assert source.index("_require_packed_vision_runtime()") < source.index(
        "WallQwen35Policy.from_checkpoint("
    )
    assert "if not args.check_config:\n        _require_packed_vision_runtime()" in source


def test_runner_validates_native_and_reports_loaded_adapter(
    openloop_namespace, monkeypatch, capsys,
) -> None:
    from rpu_backend.adapters.qwen3_5 import vision

    calls = []
    monkeypatch.setattr(vision, "_require_packed_spatial_vision_native", lambda: calls.append(1))
    openloop_namespace["_require_packed_vision_runtime"]()
    assert calls == [1]
    assert str(Path(vision.__file__).resolve()) in capsys.readouterr().out

    def stale_native():
        raise RuntimeError("native ABI mismatch")

    monkeypatch.setattr(vision, "_require_packed_spatial_vision_native", stale_native)
    with pytest.raises(RuntimeError, match="native ABI mismatch"):
        openloop_namespace["_require_packed_vision_runtime"]()


def test_wrapper_rejects_profile_in_check_mode_before_environment_setup() -> None:
    result = subprocess.run(
        ["bash", str(WRAPPER), "--check", "--torch-profile"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Torch profiling requires real execution" in result.stderr

    directory_result = subprocess.run(
        [
            "bash",
            str(WRAPPER),
            "--check",
            "--torch-profile-dir",
            "/tmp/qwen35-profile-contract-test",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert directory_result.returncode == 2
    assert "Torch profiling requires real execution" in directory_result.stderr
