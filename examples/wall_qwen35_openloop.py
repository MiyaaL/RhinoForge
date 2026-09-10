#!/usr/bin/env python3
"""Run the exact Wall Qwen3.5 checkpoint on one recorded open-loop episode.

The runner follows the checkpoint-matched Harrix data, prompt, segmentation,
and action-decoding contract: annotated events are split into 32-step chunks
without crossing event boundaries, one RGB frame per camera is sampled at each
chunk start, and the physical relative 26D policy output is decoded to an
absolute dual-arm 14D trajectory for comparison with the recorded master arms.
Its deterministic host-noise schedule is exported and can be replaced by an
explicit common-noise artifact for device-to-device parity. It never executes
robot actions.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import time
from typing import Any

# Keep the repository-wide dependency-free example probe ahead of OpenCV,
# NumPy, Torch, YAML, PIL, and rpu_backend imports.  The root wrapper supplies
# dataset/checkpoint arguments, so its --check path continues into the complete
# host preflight below.
if __name__ == "__main__" and sys.argv[1:] == ["--check-config"]:
    print("configuration OK: wall_qwen35_openloop exact-profile CLI")
    raise SystemExit(0)

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


CAMERA_FILES = {
    "face_view": "faceImg.mp4",
    "left_wrist_view": "leftImg.mp4",
    "right_wrist_view": "rightImg.mp4",
}
EXPECTED_VIDEO_GEOMETRY = {
    "face_view": (1280, 720),
    "left_wrist_view": (640, 480),
    "right_wrist_view": (640, 480),
}
EXPECTED_IMAGE_GRID_THW = [[1, 8, 14], [1, 10, 14], [1, 10, 14]]
EXPECTED_AGENT_POS_CONFIG = {
    "follow_left_ee_cartesian_pos": 3,
    "follow_left_ee_rotation_6D": 6,
    "follow_left_gripper": 1,
    "follow_right_ee_cartesian_pos": 3,
    "follow_right_ee_rotation_6D": 6,
    "follow_right_gripper": 1,
    "velocity_decomposed": 3,
    "height": 1,
    "head_actions": 2,
}
EXPECTED_DOF_CONFIG = {
    "master_left_ee_cartesian_pos_relative": 3,
    "master_left_ee_rotation_6D_relative": 6,
    "master_left_gripper": 1,
    "master_right_ee_cartesian_pos_relative": 3,
    "master_right_ee_rotation_6D_relative": 6,
    "master_right_gripper": 1,
    "velocity_decomposed": 3,
    "height": 1,
    "head_actions": 2,
}
MASK26 = np.asarray([1] * 20 + [0] * 6, dtype=np.float32)
ACTION_HORIZON = 32
SEED_DEFAULT = 3407
def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _flow_noise_schedule(
    path: Path | None,
    *,
    request_count: int,
    base_seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    expected = (request_count, ACTION_HORIZON, 26)
    if path is None:
        chunks = []
        for request_index in range(request_count):
            generator = torch.Generator(device="cpu").manual_seed(
                base_seed + request_index
            )
            chunks.append(
                torch.randn(
                    (ACTION_HORIZON, 26),
                    generator=generator,
                    dtype=torch.float32,
                ).numpy()
            )
        values = np.stack(chunks).astype(np.float32, copy=False)
        source = {
            "kind": "generated_cpu_per_request",
            "base_seed": int(base_seed),
            "schedule": "torch CPU generator re-seeded with base_seed + request_idx",
        }
    else:
        resolved = path.expanduser().resolve(strict=True)
        if not resolved.is_file():
            raise ValueError(f"flow noise is not a regular file: {resolved}")
        try:
            values = np.load(resolved, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ValueError(f"cannot load flow noise {resolved}: {exc}") from exc
        if values.ndim == 4 and values.shape[1] == 1:
            values = values[:, 0]
        source = {
            "kind": "explicit_npy",
            "source_path": str(resolved),
            "source_sha256": _sha256(resolved),
        }
    if tuple(values.shape) != expected:
        raise ValueError(
            f"flow noise must have shape {expected}, got {tuple(values.shape)}"
        )
    try:
        values = np.asarray(values, dtype=np.float32, order="C")
    except (TypeError, ValueError) as exc:
        raise ValueError("flow noise must be convertible to FP32") from exc
    if not np.isfinite(values).all():
        raise ValueError("flow noise must contain only finite values")
    source["value_sha256"] = hashlib.sha256(values.tobytes()).hexdigest()
    source["shape"] = list(values.shape)
    source["dtype"] = str(values.dtype)
    return values, source


def _require_packed_vision_runtime() -> None:
    # The wrapper executes this source file but imports the installed package.
    # Reject an old per-image adapter/native ABI before loading the checkpoint.
    from rpu_backend.adapters.qwen3_5 import vision

    require_native = getattr(vision, "_require_packed_spatial_vision_native", None)
    if not callable(require_native):
        raise RuntimeError(
            "the installed RhinoForge package lacks packed one-Graph Wall Vision; "
            "rebuild and reinstall this source tree with the wrapper's Python: "
            "python -m pip install . --no-build-isolation"
        )
    require_native()
    print(
        "[wall-qwen35-openloop] packed Vision runtime OK: "
        f"{Path(vision.__file__).resolve()}",
        flush=True,
    )


def _require_graph_budget_runtime() -> None:
    # Older native extensions silently ignore the new budget controls. A
    # conservative run remains compatible; any changed soft limit needs ABI 1.
    defaults = {
        "RPU_GRAPH_MAX_SEGMENT_ENTRIES": (8192, 32768),
        "RPU_GRAPH_MAX_SEGMENT_COMMAND_MB": (4, 8),
        "RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB": (32, 64),
    }
    changed = False
    for name, (default, maximum) in defaults.items():
        value = os.environ.get(name, str(default))
        if (not value.isascii() or not value.isdecimal()
                or not 1 <= int(value) <= maximum):
            raise ValueError(f"{name} must be a decimal integer in [1, {maximum}]")
        changed |= int(value) != default
    if not changed:
        return
    import rpu_backend

    extension = getattr(rpu_backend, "_cpp_ext", None)
    if getattr(extension, "graph_segment_budget_abi", None) != 1:
        raise RuntimeError(
            "the installed RhinoForge native extension lacks Graph segment budget "
            "controls; rebuild and reinstall this source tree with the wrapper's "
            "Python: python -m pip install . --no-build-isolation"
        )


def _configure_execution_runtime() -> bool:
    # The source wrapper may be newer than the installed package. Do not let
    # an old adapter silently use an obsolete execution preset.
    try:
        from rpu_backend.adapters.wall_qwen35 import execution
    except ImportError as exc:
        raise RuntimeError("rebuild and reinstall RhinoForge for WALL_QWEN35_OPT") from exc
    if getattr(execution, "WALL_QWEN35_OPT_ABI", None) != 2:
        raise RuntimeError("rebuild and reinstall RhinoForge for WALL_QWEN35_OPT")
    opt = execution.resolve_wall_qwen35_opt()
    execution.configure_wall_qwen35_execution(opt)
    return opt


def _require_action_execution_runtime() -> None:
    from rpu_backend.adapters.wall_qwen35 import action

    guard = getattr(action, "require_wall_fp16_loop_runtime", None)
    if not callable(guard) or not callable(getattr(action, "_action_prefix_bucket", None)):
        raise RuntimeError(
            "the installed Wall Action adapter lacks the FP16 loop/prefix-bucket implementation; "
            "rebuild and reinstall this source tree with the wrapper's Python"
        )
    guard()


def _runtime_provenance(policy: Any) -> dict[str, Any]:
    import rpu_backend

    package_dir = Path(rpu_backend.__file__).resolve().parent
    extension = package_dir / "rpu_backend.so"
    operator_asset = Path(os.environ["RPU_KERNEL_LIB_PATH"]).resolve(strict=True)
    launch_dir = Path(os.environ["RHINO_LAUNCH_LIB_DIR"]).resolve(strict=True)
    launch_library = (launch_dir / "librhino_launch.so.1.0.0").resolve(strict=True)
    manifest = getattr(policy, "_profile", None)
    return {
        "rhinoforge_version": importlib.metadata.version("rhinoforge"),
        "python_package": str(package_dir),
        "extension": {"path": str(extension), "sha256": _sha256(extension)},
        "operator_asset": {
            "path": str(operator_asset),
            "sha256": _sha256(operator_asset),
        },
        "launch_library": {
            "path": str(launch_library),
            "sha256": _sha256(launch_library),
        },
        "checkpoint_file_sha256": dict(manifest.file_sha256),
        "checkpoint_tensor_manifest_sha256": manifest.tensor_manifest_sha256,
        "checkpoint_action_manifest_sha256": manifest.action_manifest_sha256,
        "checkpoint_base_manifest_sha256": manifest.base_manifest_sha256,
    }


def _torch_profile_requested(args: argparse.Namespace) -> bool:
    return _torch_profile_mode(args) != "disabled"


def _torch_profile_mode(args: argparse.Namespace) -> str:
    return "per_request" if args.torch_profile_dir is not None else "disabled"


def _torch_profiler_activities() -> tuple[list[Any], list[str]]:
    activities = [torch.profiler.ProfilerActivity.CPU]
    activity_names = ["CPU"]
    private_use = getattr(torch.profiler.ProfilerActivity, "PrivateUse1", None)
    if private_use is not None:
        activities.append(private_use)
        activity_names.append("PrivateUse1")
    return activities, activity_names


def _new_torch_profiler(
    metadata: dict[str, Any],
    *,
    acc_events: bool,
) -> Any:
    activities, _ = _torch_profiler_activities()
    return torch.profiler.profile(
        activities=activities,
        record_shapes=bool(metadata["record_shapes"]),
        profile_memory=bool(metadata["profile_memory"]),
        with_stack=bool(metadata["with_stack"]),
        acc_events=acc_events,
    )


def _profile_directory_arg(value: str) -> Path:
    if not value:
        raise argparse.ArgumentTypeError("profile directory must not be empty")
    return Path(value)


def _resolve_output_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve(strict=False)


def _validate_profile_directory(profile_dir: Path) -> None:
    if profile_dir.exists():
        if not profile_dir.is_dir():
            raise SystemExit(
                f"profile output is not a directory: {profile_dir}"
            )
        return

    existing_parent = profile_dir.parent
    while not existing_parent.exists() and existing_parent != existing_parent.parent:
        existing_parent = existing_parent.parent
    if existing_parent.exists() and not existing_parent.is_dir():
        raise SystemExit(
            "profile output parent is not a directory: "
            f"{existing_parent}"
        )


def _plan_torch_profile(
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    """Resolve and validate profile artifacts without initializing the RPU."""

    mode = _torch_profile_mode(args)
    enabled = mode != "disabled"
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "mode": mode,
        "scope": (
            "one trace per open-loop inference request"
            if enabled
            else "post-install open-loop inference requests"
        ),
        "includes_graph_build": enabled,
        "includes_lazy_first_request_setup": enabled,
        "profiler_acc_events": enabled,
        "accumulate_request_events": False,
        "latency_is_diagnostic_only": enabled,
        "record_shapes": enabled,
        "profile_memory": False,
        "with_stack": enabled,
        "output": None,
        "summary_output": None,
        "directory": None,
        "activities": [],
        "artifacts": [],
        "sensitive_artifact": enabled,
        "admission": None,
    }
    if not enabled:
        return metadata

    profile_dir = _resolve_output_path(args.torch_profile_dir)
    _validate_profile_directory(profile_dir)
    metadata["directory"] = str(profile_dir)
    return metadata


def _plan_hw_perf(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    """Resolve the bounded r4 hardware trace directory without touching RPU."""

    enabled = args.hw_perf_dir is not None
    if args.hw_perf_max_dumps <= 0:
        raise SystemExit("--hw-perf-max-dumps must be greater than zero")
    metadata: dict[str, Any] = {
        "enabled": enabled,
        "output_dir": None,
        "max_dumps": int(args.hw_perf_max_dumps),
        "dump_count": 0,
        "artifacts": [],
        "sensitive_artifact": enabled,
        "latency_is_diagnostic_only": enabled,
        "frequency_mhz": os.environ.get("LKN_RPU_FREQ_MHZ", "800"),
        "session_started_ns": time.time_ns(),
    }
    if not enabled:
        return metadata
    profile_dir = _resolve_output_path(args.hw_perf_dir)
    _validate_profile_directory(profile_dir)
    metadata["output_dir"] = str(profile_dir)
    return metadata


def _finalize_hw_perf_metadata(metadata: dict[str, Any]) -> None:
    if not metadata["enabled"]:
        return
    output_dir = Path(metadata["output_dir"])
    artifacts = []
    for path in sorted(output_dir.glob("rpu_hwperf_*.json")):
        if (
            path.is_file()
            and path.stat().st_size > 0
            and path.stat().st_mtime_ns >= int(metadata["session_started_ns"])
        ):
            artifacts.append(
                {
                    "path": str(path),
                    "trace_bytes": path.stat().st_size,
                    "trace_sha256": _sha256(path),
                }
            )
    metadata["artifacts"] = artifacts
    metadata["dump_count"] = len(artifacts)


def _create_torch_profiler(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    profile_plan: dict[str, Any] | None = None,
) -> tuple[Any | None, dict[str, Any]]:
    metadata = (
        _plan_torch_profile(args, output_dir)
        if profile_plan is None
        else dict(profile_plan)
    )
    if not metadata["enabled"]:
        return None, metadata

    _, metadata["activities"] = _torch_profiler_activities()
    profile_dir = Path(metadata["directory"])
    _validate_profile_directory(profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)
    # Each request gets its own profiler; preparation must not start recording.
    return None, metadata


def _request_trace_path(metadata: dict[str, Any], request_index: int) -> Path:
    profile_dir = Path(metadata["directory"])
    return profile_dir / (
        "qwen35_generate_flow_action_batch_"
        f"{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_"
        f"{request_index + 1:06d}_{time.time_ns()}.trace.json.gz"
    )


def _export_torch_profile(
    profiler: Any,
    metadata: dict[str, Any],
    *,
    trace_path: Path | None = None,
    request_index: int | None = None,
) -> None:
    if trace_path is None:
        trace_path = Path(metadata["output"])
    if trace_path.exists():
        raise RuntimeError(f"refusing to overwrite Torch trace: {trace_path}")
    profiler.export_chrome_trace(str(trace_path))
    if not trace_path.is_file() or trace_path.stat().st_size <= 0:
        raise RuntimeError(f"Torch profiler did not create a non-empty trace: {trace_path}")
    artifact = {
        "path": str(trace_path),
        "trace_bytes": trace_path.stat().st_size,
        "trace_sha256": _sha256(trace_path),
    }
    if request_index is None:
        metadata.update(artifact)
    else:
        artifact["request_idx"] = request_index
        metadata["artifacts"].append(artifact)
    summary_output = metadata.get("summary_output")
    if request_index is None and summary_output:
        summary_path = Path(summary_output)
        if summary_path.exists():
            raise RuntimeError(
                f"refusing to overwrite Torch profile summary: {summary_path}"
            )
        _json_dump(
            summary_path,
            {
                "schema_version": 1,
                "trace": artifact,
                "profile": {
                    key: value
                    for key, value in metadata.items()
                    if key not in {"artifacts"}
                },
                "key_averages": _torch_profile_key_averages(profiler),
            },
        )
        metadata["summary_bytes"] = summary_path.stat().st_size
        metadata["summary_sha256"] = _sha256(summary_path)
    print(f"[wall-qwen35-openloop] torch profile: {trace_path}", flush=True)


def _torch_profile_key_averages(profiler: Any) -> list[dict[str, Any]]:
    """Return a compact, version-tolerant summary of the hottest trace rows."""

    try:
        events = list(profiler.key_averages())
    except (AttributeError, RuntimeError):
        return []

    def number(event: Any, *names: str) -> float:
        for name in names:
            value = getattr(event, name, None)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    rows = []
    for event in events:
        rows.append(
            {
                "name": str(getattr(event, "key", "<unknown>")),
                "count": int(getattr(event, "count", 0)),
                "self_cpu_time_us": number(event, "self_cpu_time_total"),
                "cpu_time_us": number(event, "cpu_time_total"),
                "self_device_time_us": number(
                    event,
                    "self_device_time_total",
                    "self_privateuse1_time_total",
                ),
                "device_time_us": number(
                    event,
                    "device_time_total",
                    "privateuse1_time_total",
                ),
            }
        )
    rows.sort(
        key=lambda row: (
            row["device_time_us"],
            row["cpu_time_us"],
            row["self_device_time_us"],
            row["self_cpu_time_us"],
        ),
        reverse=True,
    )
    return rows[:200]


def _profiled_predict_action_chunk(
    policy: Any,
    profiler: Any | None,
    *,
    mark_step: bool = False,
    **kwargs: Any,
) -> Any:
    reference_range = (
        torch.profiler.record_function("generate_flow_action_batch")
        if profiler is not None
        else contextlib.nullcontext()
    )
    wall_range = (
        torch.profiler.record_function(
            "wall_qwen35_openloop::predict_action_chunk"
        )
        if profiler is not None
        else contextlib.nullcontext()
    )
    with torch.inference_mode(), reference_range, wall_range:
        output = policy.predict_action_chunk(**kwargs)
    if profiler is not None and mark_step:
        profiler.step()
    return output


def _finite_vector(row: dict, key: str, width: int, frame_index: int) -> np.ndarray:
    value = np.asarray(row.get(key), dtype=np.float32).reshape(-1)
    if value.shape != (width,):
        raise ValueError(
            f"trajectory frame {frame_index} field {key!r} must have "
            f"shape ({width},), got {value.shape}"
        )
    if not np.isfinite(value).all():
        raise ValueError(
            f"trajectory frame {frame_index} field {key!r} is not finite"
        )
    return value


def _euler_zyx_to_rows6(value: Any) -> np.ndarray:
    """[roll,pitch,yaw] -> top two rows of Rz(yaw)Ry(pitch)Rx(roll)."""

    roll, pitch, yaw = np.asarray(value, dtype=np.float64).reshape(3)
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.asarray(
        [
            cy * cp,
            cy * sp * sr - sy * cr,
            cy * sp * cr + sy * sr,
            sy * cp,
            sy * sp * sr + cy * cr,
            sy * sp * cr - cy * sr,
        ],
        dtype=np.float32,
    )


def _make_state26(frame: dict) -> np.ndarray:
    parts: list[np.ndarray] = []
    for side in ("left", "right"):
        parts.extend(
            (
                np.asarray(frame[f"follow_{side}_position"], dtype=np.float32),
                _euler_zyx_to_rows6(frame[f"follow_{side}_rotation"]),
                np.asarray([frame[f"follow_{side}_gripper"]], dtype=np.float32),
            )
        )
    parts.append(np.zeros(6, dtype=np.float32))
    result = np.concatenate(parts).astype(np.float32)
    if result.shape != (26,) or not np.isfinite(result).all():
        raise ValueError(f"constructed state must be finite [26], got {result.shape}")
    return result


def _rows6_to_matrix(value: Any) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    row0 = value[..., :3]
    row1 = value[..., 3:6]
    row0 = row0 / np.maximum(np.linalg.norm(row0, axis=-1, keepdims=True), 1e-9)
    row1 = row1 - np.sum(row0 * row1, axis=-1, keepdims=True) * row0
    row1 = row1 / np.maximum(np.linalg.norm(row1, axis=-1, keepdims=True), 1e-9)
    row2 = np.cross(row0, row1)
    return np.stack((row0, row1, row2), axis=-2)


def _matrix_to_euler_zyx(matrix: Any) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    roll = np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2])
    pitch = np.arctan2(
        -matrix[..., 2, 0],
        np.hypot(matrix[..., 0, 0], matrix[..., 1, 0]),
    )
    yaw = np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0])
    return np.stack((roll, pitch, yaw), axis=-1).astype(np.float32)


def _relative26_to_absolute14(relative: Any, state26: Any) -> np.ndarray:
    relative = np.asarray(relative, dtype=np.float32)
    state = np.asarray(state26, dtype=np.float32).reshape(26)
    if relative.ndim != 2 or relative.shape[1] != 26:
        raise ValueError(f"relative action must have shape [T,26], got {relative.shape}")
    result = np.empty((relative.shape[0], 14), dtype=np.float32)
    for src_pos, src_rot, src_grip, dst in (
        (0, 3, 9, 0),
        (10, 13, 19, 7),
    ):
        result[:, dst : dst + 3] = (
            relative[:, src_pos : src_pos + 3]
            + state[src_pos : src_pos + 3]
        )
        delta_rotation = _rows6_to_matrix(
            relative[:, src_rot : src_rot + 6]
        )
        state_rotation = _rows6_to_matrix(
            state[None, src_rot : src_rot + 6]
        )[0]
        result[:, dst + 3 : dst + 6] = _matrix_to_euler_zyx(
            delta_rotation @ state_rotation
        )
        result[:, dst + 6] = relative[:, src_grip]
    if not np.isfinite(result).all():
        raise RuntimeError("decoded absolute action contains non-finite values")
    return result


def _ground_truth_absolute14(frame: dict) -> np.ndarray:
    parts: list[np.ndarray] = []
    for side in ("left", "right"):
        parts.extend(
            (
                np.asarray(frame[f"master_{side}_position"], dtype=np.float32),
                np.asarray(frame[f"master_{side}_rotation"], dtype=np.float32),
                np.asarray([frame[f"master_{side}_gripper"]], dtype=np.float32),
            )
        )
    result = np.concatenate(parts).astype(np.float32)
    if result.shape != (14,) or not np.isfinite(result).all():
        raise ValueError(f"ground truth must be finite [14], got {result.shape}")
    return result


def _euler_zyx_to_matrix(value: Any) -> np.ndarray:
    """Vectorized ``[roll,pitch,yaw]`` to ``Rz(yaw)Ry(pitch)Rx(roll)``."""

    value = np.asarray(value, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(value, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    result[..., 0, 0] = cy * cp
    result[..., 0, 1] = cy * sp * sr - sy * cr
    result[..., 0, 2] = cy * sp * cr + sy * sr
    result[..., 1, 0] = sy * cp
    result[..., 1, 1] = sy * sp * sr + cy * cr
    result[..., 1, 2] = sy * sp * cr - cy * sr
    result[..., 2, 0] = -sp
    result[..., 2, 1] = cp * sr
    result[..., 2, 2] = cp * cr
    return result


def _absolute14_metrics(predicted: Any, target: Any) -> dict[str, float]:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if predicted.shape != target.shape or predicted.ndim != 2 or predicted.shape[1] != 14:
        raise ValueError("absolute action metrics require matching [T,14] arrays")
    position_indices = (0, 1, 2, 7, 8, 9)
    gripper_indices = (6, 13)
    angles: list[np.ndarray] = []
    for offset in (3, 10):
        pred_rotation = _euler_zyx_to_matrix(predicted[:, offset : offset + 3])
        target_rotation = _euler_zyx_to_matrix(target[:, offset : offset + 3])
        relative = pred_rotation @ np.swapaxes(target_rotation, -1, -2)
        cosine = np.clip(
            (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        angles.append(np.degrees(np.arccos(cosine)))
    return {
        "abs14_l1": float(np.abs(predicted - target).mean()),
        "position_mae": float(
            np.abs(predicted[:, position_indices] - target[:, position_indices]).mean()
        ),
        "rotation_geodesic_deg": float(np.concatenate(angles).mean()),
        "gripper_mae": float(
            np.abs(predicted[:, gripper_indices] - target[:, gripper_indices]).mean()
        ),
    }


def _video_metadata(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open camera video: {path}")
        return {
            "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        capture.release()


def _decode_selected_frames(path: Path, requested: set[int]) -> dict[int, Image.Image]:
    capture = cv2.VideoCapture(str(path))
    selected: dict[int, Image.Image] = {}
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open camera video: {path}")
        frame_index = 0
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            if frame_index in requested:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                selected[frame_index] = Image.fromarray(rgb)
                if len(selected) == len(requested):
                    break
            frame_index += 1
    finally:
        capture.release()
    missing = sorted(requested - set(selected))
    if missing:
        raise RuntimeError(f"could not decode frames {missing} from {path}")
    return selected


def _parse_episode(
    dataset_dir: Path,
    checkpoint: Path,
    *,
    instruction_source: str,
    max_requests: int,
) -> dict[str, Any]:
    episode = dataset_dir.name
    trajectory_path = dataset_dir / f"{episode}.json"
    instruction_path = dataset_dir / "instruction.json"
    config_path = next(
        (checkpoint / name for name in ("config.yml", "config.yaml") if (checkpoint / name).is_file()),
        None,
    )
    if config_path is None:
        raise FileNotFoundError(f"checkpoint has no config.yml/config.yaml: {checkpoint}")
    for path in (trajectory_path, instruction_path, *[dataset_dir / name for name in CAMERA_FILES.values()]):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("model_type") != "qwen3_5":
        raise ValueError(f"checkpoint model_type must be qwen3_5, got {config.get('model_type')!r}")
    task = config.get("task") or {}
    if int(task.get("action_horizon_flow") or 0) != ACTION_HORIZON:
        raise ValueError("checkpoint action_horizon_flow must be 32")
    if task.get("agent_pos_config") != EXPECTED_AGENT_POS_CONFIG:
        raise ValueError(
            "checkpoint agent_pos_config does not match the admitted Wall 26D layout"
        )
    if task.get("dof_config") != EXPECTED_DOF_CONFIG:
        raise ValueError(
            "checkpoint dof_config does not match the admitted Wall 26D layout"
        )

    trajectory_document = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory = trajectory_document.get("data")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"trajectory has no non-empty data list: {trajectory_path}")
    required_vectors = {
        **{f"{role}_{side}_position": 3 for role in ("follow", "master") for side in ("left", "right")},
        **{f"{role}_{side}_rotation": 3 for role in ("follow", "master") for side in ("left", "right")},
    }
    for frame_index, row in enumerate(trajectory):
        for key, width in required_vectors.items():
            _finite_vector(row, key, width, frame_index)
        for role in ("follow", "master"):
            for side in ("left", "right"):
                value = float(row[f"{role}_{side}_gripper"])
                if not math.isfinite(value):
                    raise ValueError(
                        f"trajectory frame {frame_index} non-finite {role}_{side}_gripper"
                    )

    instruction_document = json.loads(instruction_path.read_text(encoding="utf-8"))
    episode_instruction = instruction_document.get(episode)
    if not isinstance(episode_instruction, dict):
        raise ValueError(f"instruction.json has no episode key {episode!r}")
    base_instruction = str(episode_instruction.get("instruction") or "").strip()
    if not base_instruction:
        raise ValueError("episode base instruction is empty")
    raw_events = episode_instruction.get(instruction_source)
    if not isinstance(raw_events, dict) or not raw_events:
        raise ValueError(
            f"episode has no instruction source {instruction_source!r}"
        )
    events: list[dict[str, Any]] = []
    for bounds, caption in raw_events.items():
        pieces = str(bounds).split()
        if len(pieces) != 2:
            raise ValueError(f"invalid event bounds {bounds!r}")
        start, stop = (int(piece) for piece in pieces)
        start = max(0, min(start, len(trajectory)))
        stop = max(0, min(stop, len(trajectory)))
        if stop > start:
            events.append(
                {"start": start, "stop": stop, "caption": str(caption or "").strip()}
            )
    events.sort(key=lambda item: (item["start"], item["stop"]))
    if not events:
        raise ValueError("instruction source has no valid in-range events")

    segments: list[dict[str, Any]] = []
    for event_index, event in enumerate(events):
        start = int(event["start"])
        segment_index = 0
        while start < int(event["stop"]):
            stop = min(start + ACTION_HORIZON, int(event["stop"]))
            segments.append(
                {
                    "event_idx": event_index,
                    "segment_idx_in_event": segment_index,
                    "event_bounds": [int(event["start"]), int(event["stop"])],
                    "start": start,
                    "stop": stop,
                    "eval_steps": stop - start,
                    "caption": event["caption"],
                }
            )
            if max_requests and len(segments) >= max_requests:
                break
            start = stop
            segment_index += 1
        if max_requests and len(segments) >= max_requests:
            break

    video_paths = {name: dataset_dir / filename for name, filename in CAMERA_FILES.items()}
    video_metadata = {name: _video_metadata(path) for name, path in video_paths.items()}
    bad_counts = {
        name: metadata["frames"]
        for name, metadata in video_metadata.items()
        if metadata["frames"] != len(trajectory)
    }
    if bad_counts:
        raise ValueError(
            f"trajectory/video frame counts differ: trajectory={len(trajectory)}, "
            f"videos={bad_counts}"
        )
    for name, metadata in video_metadata.items():
        geometry = (metadata["width"], metadata["height"])
        if geometry != EXPECTED_VIDEO_GEOMETRY[name]:
            raise ValueError(
                f"{name} geometry {geometry} != exact episode profile "
                f"{EXPECTED_VIDEO_GEOMETRY[name]}"
            )
        if not math.isclose(metadata["fps"], 20.0, rel_tol=0.0, abs_tol=1e-3):
            raise ValueError(
                f"{name} media FPS {metadata['fps']} != exact episode profile 20"
            )
    requested = {int(segment["start"]) for segment in segments}
    frames = {
        name: _decode_selected_frames(path, requested)
        for name, path in video_paths.items()
    }
    return {
        "episode": episode,
        "trajectory": trajectory,
        "trajectory_path": trajectory_path,
        "instruction_path": instruction_path,
        "config_path": config_path,
        "base_instruction": base_instruction,
        "events": events,
        "segments": segments,
        "frames": frames,
        "video_metadata": video_metadata,
    }


def _host_prefix_preflight(
    checkpoint: Path,
    parsed: dict[str, Any],
    *,
    dataset_key: str,
    robot_id: str,
) -> list[dict[str, Any]]:
    """Validate all real images/prompts before irreversible RPU installation."""

    from transformers import AutoConfig, AutoProcessor
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    from rpu_backend.adapters.wall_qwen35.action import (
        _canonical_action_positions,
    )
    from rpu_backend.adapters.wall_qwen35.checkpoint import (
        extend_wall_qwen35_tokenizer,
    )
    from rpu_backend.adapters.wall_qwen35.preprocessing import (
        MAX_ACTION_PREFIX_LENGTH,
        prepare_wall_qwen35_input,
    )

    normalizer = torch.load(
        checkpoint / "normalizer_propri.pth",
        map_location="cpu",
        weights_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True
    )
    extend_wall_qwen35_tokenizer(processor.tokenizer, target_vocab=256277)
    processor.tokenizer.padding_side = "right"
    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    config._attn_implementation = "eager"
    getattr(config, "text_config", config)._attn_implementation = "eager"
    # A meta model supplies the exact HF get_rope_index implementation without
    # allocating checkpoint weights.  Action positions are therefore admitted
    # before the irreversible streamed RPU installation, not inferred from the
    # larger physical KV-cache extent.
    with torch.device("meta"):
        position_model = Qwen3_5ForConditionalGeneration(config)
    results: list[dict[str, Any]] = []
    try:
        for request_index, segment in enumerate(parsed["segments"]):
            start = int(segment["start"])
            state = _make_state26(parsed["trajectory"][start])
            prepared = prepare_wall_qwen35_input(
                processor,
                position_model,
                images={
                    name: frames[start]
                    for name, frames in parsed["frames"].items()
                },
                instruction={
                    "task": parsed["base_instruction"],
                    "detail": segment["caption"],
                },
                state=state,
                state_min=normalizer[f"min.{dataset_key}"],
                state_delta=normalizer[f"delta.{dataset_key}"],
                dataset_key=dataset_key,
                robot_id=robot_id,
                agent_pos_mask=MASK26,
                dof_mask=MASK26,
            )
            action_positions = _canonical_action_positions(
                prepared.action_position_ids, prepared.prefix_length
            )
            results.append(
                {
                    "request_idx": request_index,
                    "frame": start,
                    "prefix_length": int(prepared.prefix_length),
                    "logical_length": int(prepared.attention_mask.sum().item()),
                    "image_grid_thw": prepared.image_grid_thw.tolist(),
                    "action_position_start": int(action_positions[0, 0]),
                    "action_position_end": int(action_positions[0, -1]),
                }
            )
            if prepared.image_grid_thw.tolist() != EXPECTED_IMAGE_GRID_THW:
                raise RuntimeError(
                    "Dataset-V2 image grid drift: "
                    f"{prepared.image_grid_thw.tolist()} != "
                    f"{EXPECTED_IMAGE_GRID_THW}"
                )
            if prepared.prefix_length > MAX_ACTION_PREFIX_LENGTH:
                raise RuntimeError("prefix preflight accepted an out-of-profile input")
    finally:
        del position_model
        del processor
        gc.collect()
    return results


def _graph_stats(cache: Any) -> dict[str, Any] | None:
    if cache is None:
        return None
    entries = list(cache.snapshot())
    return {
        "size": int(cache.size()),
        "max_entries": int(cache.max_entries()),
        "phase": str(getattr(cache, "phase", "unknown")),
        "replays": sum(int(getattr(entry, "replay_count", 0)) for entry in entries),
        "recaptures": sum(int(getattr(entry, "recapture_count", 0)) for entry in entries),
        "entries": [
            {
                "signature": repr(getattr(entry, "signature", None)),
                "kernel_count": int(getattr(entry, "kernel_count", 0)),
                "segment_count": int(getattr(entry, "segment_count", 0)),
                "local_spm_slot_count": int(
                    getattr(entry, "local_spm_slot_count", 0)
                ),
                "data_node_count": int(getattr(entry, "data_node_count", 0)),
                "replay_count": int(getattr(entry, "replay_count", 0)),
                "recapture_count": int(getattr(entry, "recapture_count", 0)),
                "non_replayable_reason": str(
                    getattr(entry, "non_replayable_reason", "")
                ),
            }
            for entry in entries
        ],
        "invariant_ok": bool(cache.cache_invariant_ok()),
    }


def _validate_retained_graph_stats(
    name: str, stats: dict[str, Any] | None, *, calls: int
) -> None:
    """Require one build per retained signature and replay for the rest.

    Wall deliberately retains more than one signature for Vision geometry and
    prefix buckets.  Therefore ``size == 1`` is not a valid lifecycle check;
    with ``calls`` total invocations, the minimum replay count is
    ``calls - size`` (one BUILD for each signature).
    """
    if stats is None:
        raise RuntimeError(f"{name} Graph lifecycle is unavailable")
    size = int(stats.get("size", 0))
    max_entries = int(stats.get("max_entries", 0))
    entries = list(stats.get("entries", ()))
    if (
        not bool(stats.get("invariant_ok", False))
        or size <= 0
        or size != len(entries)
        or size > max_entries
        or int(stats.get("recaptures", 0)) != 0
        or int(stats.get("replays", 0)) < max(0, int(calls) - size)
        or any(
            int(entry.get("kernel_count", 0)) <= 0
            or bool(entry.get("non_replayable_reason"))
            or int(entry.get("recapture_count", 0)) != 0
            for entry in entries
        )
    ):
        raise RuntimeError(
            f"{name} Graph lifecycle did not prove one BUILD per retained "
            f"signature and stable replay: {stats}"
        )


def _oneshot_graph_stats(graph: Any) -> dict[str, Any] | None:
    if graph is None:
        return None
    raw = dict(graph.debug_stats())
    stats: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, (list, tuple)):
            stats[str(key)] = [int(item) for item in value]
        elif isinstance(value, (bool, int, float, str)):
            stats[str(key)] = value
        else:
            stats[str(key)] = str(value)
    return {
        "contract": "bounded one-shot; no replay claim",
        "state": int(graph.state()),
        "replayable": bool(graph.replayable()),
        "has_built_signature": bool(graph.has_built_signature()),
        "graph_size": int(graph.graph_size()),
        "debug_stats": stats,
    }


def _policy_graph_stats(policy: Any) -> dict[str, Any]:
    runtime = getattr(policy, "_runtime", None)
    action_state = getattr(
        getattr(runtime, "action_expert", None), "_rpu_qwen3_5", None
    )
    base_model = getattr(runtime, "base_model", None)
    fusion = getattr(base_model, "model", None)
    vision = getattr(fusion, "visual", None)
    text = getattr(fusion, "language_model", None)
    text_state = getattr(text, "_rpu_qwen3_5", None)
    prefill_cache = getattr(text_state, "prefill_graph_cache", None)
    if prefill_cache is not None:
        base_prefill = _graph_stats(prefill_cache)
    else:
        base_prefill = _oneshot_graph_stats(
            getattr(text_state, "prefill_graph", None)
        )
    return {
        "action": _graph_stats(getattr(action_state, "action_graph_cache", None)),
        "vision": _graph_stats(getattr(vision, "_rpu_vision_graph_cache", None)),
        "base_prefill": base_prefill,
    }


def _policy_retained_graph_caches(policy: Any) -> dict[str, Any]:
    """Resolve the three caches required by the exact-repeat READY probe."""

    runtime = getattr(policy, "_runtime", None)
    action_state = getattr(
        getattr(runtime, "action_expert", None), "_rpu_qwen3_5", None
    )
    fusion = getattr(getattr(runtime, "base_model", None), "model", None)
    vision = getattr(fusion, "visual", None)
    text_state = getattr(
        getattr(fusion, "language_model", None), "_rpu_qwen3_5", None
    )
    caches = {
        "action": getattr(action_state, "action_graph_cache", None),
        "vision": getattr(vision, "_rpu_vision_graph_cache", None),
        "base_prefill": getattr(text_state, "prefill_graph_cache", None),
    }
    missing = [name for name, cache in caches.items() if cache is None]
    if missing:
        raise RuntimeError(
            "Wall Qwen3.5 READY probe is missing retained GraphCache owners: "
            + ", ".join(missing)
        )
    return caches


@contextlib.contextmanager
def _frozen_policy_graph_caches(policy: Any):
    """Freeze every required cache and restore WARMING on all exit paths."""

    caches = _policy_retained_graph_caches(policy)
    frozen = []
    try:
        for cache in caches.values():
            cache.freeze()
            frozen.append(cache)
        yield caches
    finally:
        # Later open-loop requests may have another exact Action prefix. Keep
        # the warmed entries, but explicitly allow a bounded BUILD again.
        for cache in reversed(frozen):
            cache.begin_warmup()


def _retained_replay_admission(
    before: dict[str, Any] | None,
    after: dict[str, Any] | None,
    *,
    expected_replays: int,
) -> dict[str, Any]:
    """Classify one retained cache without inferring READY from a clean log."""

    before_replays = int(before["replays"]) if before is not None else 0
    after_replays = int(after["replays"]) if after is not None else 0
    before_recaptures = int(before["recaptures"]) if before is not None else 0
    after_recaptures = int(after["recaptures"]) if after is not None else 0
    replay_delta = after_replays - before_replays
    recapture_delta = after_recaptures - before_recaptures
    ready_phase = bool(
        str((before or {}).get("phase", "")) == "READY"
        and str((after or {}).get("phase", "")) == "READY"
    )
    before_signatures = (
        tuple(entry["signature"] for entry in before["entries"])
        if before is not None
        else ()
    )
    after_signatures = (
        tuple(entry["signature"] for entry in after["entries"])
        if after is not None
        else ()
    )
    accepted = bool(
        before is not None
        and after is not None
        and before["size"] > 0
        and after["size"] == before["size"]
        and after_signatures == before_signatures
        and ready_phase
        and before["invariant_ok"]
        and after["invariant_ok"]
        and before_recaptures == 0
        and after_recaptures == 0
        and recapture_delta == 0
        and replay_delta == expected_replays
        and all(
            entry["kernel_count"] > 0
            and not entry["non_replayable_reason"]
            for entry in after["entries"]
        )
    )
    return {
        "accepted": accepted,
        "expected_replays": expected_replays,
        "replay_delta": replay_delta,
        "recapture_delta": recapture_delta,
        "stable_signatures": after_signatures == before_signatures,
        "ready_phase": ready_phase,
        "before": before,
        "after": after,
    }


def _profile_output_parity(warm: Any, measured: Any) -> dict[str, Any]:
    expected_shape = (1, ACTION_HORIZON, 26)
    warm_actions = warm.actions.detach().to(device="cpu", dtype=torch.float32)
    measured_actions = measured.actions.detach().to(
        device="cpu", dtype=torch.float32
    )
    actions_pair_shape_match = tuple(warm_actions.shape) == tuple(
        measured_actions.shape
    )
    same_shape = bool(
        tuple(warm_actions.shape) == expected_shape
        and tuple(measured_actions.shape) == expected_shape
    )
    finite = bool(
        same_shape
        and torch.isfinite(warm_actions).all()
        and torch.isfinite(measured_actions).all()
    )
    exact = bool(
        same_shape and finite and torch.equal(warm_actions, measured_actions)
    )
    max_abs = (
        float((warm_actions - measured_actions).abs().max())
        if actions_pair_shape_match and warm_actions.numel()
        else None
    )
    warm_actions_norm = getattr(warm, "actions_norm", None)
    measured_actions_norm = getattr(measured, "actions_norm", None)
    normalized_present = bool(
        warm_actions_norm is not None and measured_actions_norm is not None
    )
    if normalized_present:
        warm_actions_norm = warm_actions_norm.detach().to(
            device="cpu", dtype=torch.float32
        )
        measured_actions_norm = measured_actions_norm.detach().to(
            device="cpu", dtype=torch.float32
        )
        normalized_pair_shape_match = tuple(warm_actions_norm.shape) == tuple(
            measured_actions_norm.shape
        )
        normalized_same_shape = bool(
            tuple(warm_actions_norm.shape) == expected_shape
            and tuple(measured_actions_norm.shape) == expected_shape
        )
        normalized_finite = bool(
            normalized_same_shape
            and torch.isfinite(warm_actions_norm).all()
            and torch.isfinite(measured_actions_norm).all()
        )
        normalized_exact = bool(
            normalized_finite
            and torch.equal(warm_actions_norm, measured_actions_norm)
        )
        normalized_max_abs = (
            float((warm_actions_norm - measured_actions_norm).abs().max())
            if normalized_pair_shape_match and warm_actions_norm.numel()
            else None
        )
    else:
        normalized_same_shape = False
        normalized_finite = False
        normalized_exact = False
        normalized_max_abs = None
    return {
        # Keep the exact parity contract in the admission record.  Numerical
        # repeatability is evidence, not a reason to discard an otherwise
        # valid diagnostic trace; the caller separately hard-fails shape,
        # prefix, or non-finite mismatches.
        "accepted": bool(
            exact
            and normalized_exact
            and warm.prefix_length == measured.prefix_length
        ),
        "expected_shape": expected_shape,
        "same_shape": same_shape,
        "finite": finite,
        "prefix_equal": bool(warm.prefix_length == measured.prefix_length),
        "actions_exact": exact,
        "actions_max_abs": max_abs,
        "actions_norm_present": normalized_present,
        "actions_norm_same_shape": normalized_same_shape,
        "actions_norm_finite": normalized_finite,
        "actions_norm_exact": normalized_exact,
        "actions_norm_max_abs": normalized_max_abs,
        "warm_prefix_length": warm.prefix_length,
        "measured_prefix_length": measured.prefix_length,
    }


def _ready_replay_admission(
    before: dict[str, Any],
    after: dict[str, Any],
    warm: Any,
    measured: Any,
) -> dict[str, Any]:
    action = _retained_replay_admission(
        before.get("action"), after.get("action"),
        expected_replays=getattr(measured, "extra", {}).get("action_graph_calls", 10),
    )
    extra = getattr(measured, "extra", {})
    if extra.get("action_execution") in ("fp16_one_graph", "fp16_steps"):
        entries = (after.get("action") or {}).get("entries", [])
        action["single_physical_segment"] = bool(
            len(entries) == 1 and entries[0].get("segment_count") == 1
        )
        action["accepted"] &= action["single_physical_segment"]
    vision = _retained_replay_admission(
        before.get("vision"), after.get("vision"),
        expected_replays=extra.get("vision_graph_calls", 1),
    )
    parity = _profile_output_parity(warm, measured)
    base_before = before.get("base_prefill")
    base_after = after.get("base_prefill")
    if (
        isinstance(base_before, dict)
        and isinstance(base_after, dict)
        and "entries" in base_before
        and "entries" in base_after
    ):
        base_prefill = _retained_replay_admission(
            base_before, base_after, expected_replays=1
        )
        base_prefill["contract"] = "retained prefix bucket Graph"
    else:
        base_prefill = {
            "accepted": False,
            "contract": "bounded one-shot; no retained READY entry",
            "before": base_before,
            "after": base_after,
        }
    # Check physical submissions, not just logical capture calls. Split Vision
    # can retain two distinct shapes yet replay three times per request.
    if "wall_qwen35_opt" in extra:
        expected_segments = (1, 1, 1) if extra["wall_qwen35_opt"] else (3, 3, 10)
        for name, component, expected in zip(
            ("vision", "base_prefill", "action"),
            (vision, base_prefill, action), expected_segments,
        ):
            prior = {e["signature"]: e for e in (before.get(name) or {}).get("entries", [])}
            launches = sum(
                (e["replay_count"] - prior.get(e["signature"], {}).get("replay_count", 0))
                * e["segment_count"]
                for e in (after.get(name) or {}).get("entries", [])
            )
            component["physical_segments"] = launches
            component["expected_physical_segments"] = expected
            component["accepted"] &= launches == expected
    full_ready = bool(
        parity["accepted"]
        and action["accepted"]
        and vision["accepted"]
        and base_prefill["accepted"]
    )
    blockers = []
    if not vision["accepted"]:
        blockers.append(
            "Vision did not replay all three calls from stable retained signatures"
        )
    if not base_prefill["accepted"]:
        blockers.append(
            "Base Prefill did not replay a stable retained prefix bucket Graph"
        )
    if not action["accepted"]:
        blockers.append("Action did not prove the selected profile's stable replay/segment count")
    if not parity["accepted"]:
        blockers.append("warmup and measured action outputs are not exactly equal")
    return {
        "status": "accepted" if full_ready else "runtime_extension_required",
        "full_ready": full_ready,
        "output_parity": parity,
        "components": {
            "vision": vision,
            "base_prefill": base_prefill,
            "action": action,
        },
        "blockers": blockers,
    }


def _write_plot(
    output_dir: Path,
    predictions: list[np.ndarray],
    ground_truth: list[np.ndarray],
    times: list[np.ndarray],
) -> str | None:
    try:
        os.environ.setdefault("MPLBACKEND", "Agg")
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/rhinoforge-matplotlib")
        from matplotlib import pyplot as plt
    except Exception as exc:
        print(f"[wall-qwen35-openloop] plot skipped: {exc}", flush=True)
        return None
    labels = (
        "left_x", "left_y", "left_z", "left_roll", "left_pitch",
        "left_yaw", "left_gripper", "right_x", "right_y", "right_z",
        "right_roll", "right_pitch", "right_yaw", "right_gripper",
    )
    figure, axes = plt.subplots(14, 1, figsize=(14, 32), sharex=True)
    for dimension, axis in enumerate(axes):
        for index, (predicted, target, frame_numbers) in enumerate(
            zip(predictions, ground_truth, times)
        ):
            axis.plot(
                frame_numbers,
                target[:, dimension],
                "b-",
                linewidth=1.5,
                label="GT master" if index == 0 else None,
            )
            axis.plot(
                frame_numbers,
                predicted[:, dimension],
                "r--",
                linewidth=1.2,
                label="RPU prediction" if index == 0 else None,
            )
        axis.set_ylabel(labels[dimension])
        axis.grid(True, alpha=0.25)
        if dimension == 0:
            axis.legend(loc="upper right")
    axes[-1].set_xlabel("recorded frame (20 FPS media index)")
    figure.tight_layout()
    path = output_dir / "openloop_abs14.png"
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return str(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--dataset-key", choices=("x2_normal",), default="x2_normal")
    parser.add_argument("--robot-id", default="10070")
    parser.add_argument("--instruction-source", default="distribute")
    parser.add_argument("--inference-steps", type=int, choices=(10,), default=10)
    parser.add_argument("--max-requests", "--max-events", dest="max_requests", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=SEED_DEFAULT)
    parser.add_argument(
        "--flow-noise",
        type=Path,
        help=(
            "common FP32 .npy noise with shape [requests,32,26] (or "
            "[requests,1,32,26]); bypasses --noise-seed"
        ),
    )
    parser.add_argument(
        "--torch-profile-dir",
        type=_profile_directory_arg,
        help=(
            "enable CPU + PrivateUse1 profiling: one *.trace.json.gz per "
            "inference request, including first-request setup/BUILD (no extra "
            "warmup); shapes and stacks on, memory events off"
        ),
    )
    parser.add_argument(
        "--hw-perf-dir",
        type=_profile_directory_arg,
        help="enable bounded r4 kernel/DMA Chrome traces in this directory",
    )
    parser.add_argument("--hw-perf-max-dumps", type=int, default=32)
    parser.add_argument("--check-config", action="store_true")
    parser.add_argument(
        "--allow-numeric-blocked-vision",
        action="store_true",
        help="required controlled-evaluation opt-in for real RPU execution",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    profile_requested = _torch_profile_requested(args)
    hw_perf_requested = args.hw_perf_dir is not None
    if args.check_config and (profile_requested or hw_perf_requested):
        raise SystemExit("profiling requires real execution; remove --check-config")
    if args.max_requests < 0:
        raise SystemExit("--max-requests must be non-negative")
    if args.noise_seed < 0 or args.noise_seed > (2**63 - 1):
        raise SystemExit("--noise-seed must be in [0, 2^63-1]")
    dataset_dir = args.dataset_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"dataset directory is missing: {dataset_dir}")
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint directory is missing: {checkpoint}")

    output_dir: Path | None = None
    profile_plan: dict[str, Any] | None = None
    hw_perf_meta: dict[str, Any] | None = None
    if not args.check_config:
        if not args.allow_numeric_blocked_vision:
            raise SystemExit(
                "real execution requires --allow-numeric-blocked-vision; this is a "
                "controlled evaluation and not a release-support claim"
            )
        if args.output_dir is None:
            raise SystemExit("real execution requires a fresh --output-dir")
        output_dir = args.output_dir.expanduser().resolve()
        if output_dir.exists():
            raise SystemExit(f"output directory already exists: {output_dir}")
        profile_plan = _plan_torch_profile(args, output_dir)
        hw_perf_meta = _plan_hw_perf(args, output_dir)

    from rpu_backend.api import WallQwen35Policy

    opt = _configure_execution_runtime()
    action_mode = "fp16_one_graph" if opt else "fp16_steps"
    if not args.check_config:
        _require_packed_vision_runtime()
        _require_graph_budget_runtime()
        _require_action_execution_runtime()

    # Exact checkpoint admission is intentionally completed before image decode
    # and before any irreversible model installation.
    policy = WallQwen35Policy.from_checkpoint(
        checkpoint,
        dataset_key=args.dataset_key,
        robot_id=args.robot_id,
        allow_numeric_blocked_vision=args.allow_numeric_blocked_vision,
    )
    parsed = _parse_episode(
        dataset_dir,
        checkpoint,
        instruction_source=args.instruction_source,
        max_requests=args.max_requests,
    )
    prefix_preflight = _host_prefix_preflight(
        checkpoint, parsed, dataset_key=args.dataset_key, robot_id=args.robot_id
    )
    prefix_lengths = [item["prefix_length"] for item in prefix_preflight]
    if (
        args.flow_noise is None
        and args.noise_seed + len(parsed["segments"]) - 1 > (2**63 - 1)
    ):
        raise SystemExit("per-request noise seed schedule exceeds 2^63-1")
    try:
        flow_noise, flow_noise_meta = _flow_noise_schedule(
            args.flow_noise,
            request_count=len(parsed["segments"]),
            base_seed=int(args.noise_seed),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(
        "[wall-qwen35-openloop] preflight OK: "
        f"frames={len(parsed['trajectory'])}, events={len(parsed['events'])}, "
        f"requests={len(parsed['segments'])}, "
        f"prefix=[{min(prefix_lengths)},{max(prefix_lengths)}]",
        flush=True,
    )
    if args.check_config:
        policy.close()
        return 0
    assert output_dir is not None
    assert profile_plan is not None
    assert hw_perf_meta is not None
    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise SystemExit(f"output directory already exists: {output_dir}") from exc
    _, torch_profile_meta = _create_torch_profiler(
        args,
        output_dir,
        profile_plan=profile_plan,
    )
    flow_noise_path = output_dir / "flow_noise.npy"
    np.save(flow_noise_path, flow_noise)
    flow_noise_meta = {
        **flow_noise_meta,
        "artifact": str(flow_noise_path),
        "artifact_sha256": _sha256(flow_noise_path),
    }

    common_meta = {
        "mode": "wall_qwen35_openloop",
        "status": "installing",
        "dataset": str(dataset_dir),
        "episode": parsed["episode"],
        "checkpoint": str(checkpoint),
        "checkpoint_config": str(parsed["config_path"]),
        "dataset_key": args.dataset_key,
        "normalizer": args.dataset_key,
        "robot_id": args.robot_id,
        "instruction_source": args.instruction_source,
        "base_instruction": parsed["base_instruction"],
        "action_horizon": ACTION_HORIZON,
        "num_inference_steps": args.inference_steps,
        "action_dim": 26,
        "noise_seed_base": (
            int(args.noise_seed) if args.flow_noise is None else None
        ),
        "flow_noise": flow_noise_meta,
        "model_requests": len(parsed["segments"]),
        "recorded_frames": len(parsed["trajectory"]),
        "annotated_events": len(parsed["events"]),
        "recorded_media_fps": parsed["video_metadata"]["face_view"]["fps"],
        "prompt_frequency_contract": "32HZ",
        "alignment": "video frame i == trajectory row i; no resampling",
        "video_metadata": parsed["video_metadata"],
        "image_resize": {
            "algorithm": "dataset_v2_smart_resize",
            "resample": "PIL BICUBIC",
            "target_long_edge": 224,
            "image_factor": 32,
            "min_pixels": 4096,
            "max_pixels": 16777216,
            "downstream_processor_resize": False,
        },
        "prefix_preflight": prefix_preflight,
        "agent_pos_mask": MASK26.tolist(),
        "dof_mask": MASK26.tolist(),
        "controlled_evaluation": True,
        "wall_qwen35_opt": opt,
        "graph_plan": [1, 1, 1] if opt else [3, 3, 10],
        "action_execution": {
            "mode": action_mode,
            "steps_per_graph": 10 if opt else 1,
            "require_single_segment": True,
            "time_ada_precompute": "cpu_fp32_then_fp16",
        },
        "numeric_blocked_vision_opt_in": True,
        "vision_execution": {
            "mode": "packed_spatial" if opt else "per_image",
            "images_per_graph": 3 if opt else 1,
            "attention_calls_per_layer": 3,
            "encoder_residual_reduce": "per_image_spans",
        },
        "cold_runtime_profile": {
            "WALL_QWEN35_OPT": os.environ.get("WALL_QWEN35_OPT"),
            "RPU_GRAPH_MAX_SEGMENT_ENTRIES": os.environ.get(
                "RPU_GRAPH_MAX_SEGMENT_ENTRIES"
            ),
            "RPU_GRAPH_MAX_SEGMENT_COMMAND_MB": os.environ.get(
                "RPU_GRAPH_MAX_SEGMENT_COMMAND_MB"
            ),
            "RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB": os.environ.get(
                "RPU_GRAPH_MAX_SEGMENT_INSTRUCTION_MB"
            ),
            "LKN_MAX_BATCH_ENTRIES": os.environ.get("LKN_MAX_BATCH_ENTRIES"),
            "LKN_KD_BUF_MB": os.environ.get("LKN_KD_BUF_MB"),
            "LKN_INSTR_BUF_MB": os.environ.get("LKN_INSTR_BUF_MB"),
            "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN": os.environ.get(
                "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN"
            ),
            # Wall decode fusion arms are cold process-level selectors.  Keep
            # their effective values in every run receipt so an A/B result is
            # auditable even when the launcher rebuilds its environment.
            "RPU_QWEN35_WALL_FUSED_SILU_MUL": os.environ.get(
                "RPU_QWEN35_WALL_FUSED_SILU_MUL"
            ),
            "RPU_QWEN35_WALL_PREREDUCE_RESIDUAL_GATE": os.environ.get(
                "RPU_QWEN35_WALL_PREREDUCE_RESIDUAL_GATE"
            ),
        },
        "torch_profile": torch_profile_meta,
        "hw_perf": hw_perf_meta,
        "provenance": _runtime_provenance(policy),
    }
    _json_dump(output_dir / "meta.json", common_meta)
    runtime_stack = contextlib.ExitStack()
    runtime_stack.callback(policy.close)
    if hw_perf_meta["enabled"]:
        if not hasattr(torch.rpu, "hw_perf_trace"):
            runtime_stack.close()
            raise RuntimeError(
                "the installed RhinoForge build lacks r4 hardware trace "
                "support; rebuild and reinstall this source tree"
            )
        try:
            runtime_stack.enter_context(
                torch.rpu.hw_perf_trace(
                    hw_perf_meta["output_dir"],
                    max_dumps=hw_perf_meta["max_dumps"],
                )
            )
        except BaseException:
            runtime_stack.close()
            raise
        print(
            "[wall-qwen35-openloop] RPU hardware trace enabled: "
            f"{hw_perf_meta['output_dir']}",
            flush=True,
        )
    print("[wall-qwen35-openloop] installing policy on RPU", flush=True)
    install_started = time.perf_counter()
    try:
        policy.to("rpu")
    except BaseException:
        runtime_stack.close()
        _finalize_hw_perf_metadata(hw_perf_meta)
        raise
    install_seconds = time.perf_counter() - install_started
    print(
        f"[wall-qwen35-openloop] RPU install complete in {install_seconds:.3f}s",
        flush=True,
    )

    predictions_relative: list[np.ndarray] = []
    predictions_absolute: list[np.ndarray] = []
    ground_truth_chunks: list[np.ndarray] = []
    time_chunks: list[np.ndarray] = []
    segment_results: list[dict[str, Any]] = []
    try:
        for request_index, segment in enumerate(parsed["segments"]):
            start = int(segment["start"])
            state = _make_state26(parsed["trajectory"][start])
            request_noise_seed = (
                int(args.noise_seed) + request_index
                if args.flow_noise is None
                else None
            )
            request_started = time.perf_counter()
            request_kwargs = {
                "images": {
                    name: frames[start]
                    for name, frames in parsed["frames"].items()
                },
                "instruction": {
                    "task": parsed["base_instruction"],
                    "detail": segment["caption"],
                },
                "proprioception": state,
                "agent_pos_mask": MASK26,
                "dof_mask": MASK26,
                "initial_noise": flow_noise[request_index],
            }
            if torch_profile_meta["mode"] == "per_request":
                request_profiler = _new_torch_profiler(
                    torch_profile_meta,
                    acc_events=True,
                )
                with request_profiler:
                    output = _profiled_predict_action_chunk(
                        policy,
                        request_profiler,
                        **request_kwargs,
                    )
                request_seconds = time.perf_counter() - request_started
                _export_torch_profile(
                    request_profiler,
                    torch_profile_meta,
                    trace_path=_request_trace_path(
                        torch_profile_meta,
                        request_index,
                    ),
                    request_index=request_index,
                )
            else:
                output = _profiled_predict_action_chunk(
                    policy,
                    None,
                    **request_kwargs,
                )
                request_seconds = time.perf_counter() - request_started
            relative = output.actions.numpy()[0].astype(np.float32, copy=True)
            if relative.shape != (ACTION_HORIZON, 26) or not np.isfinite(relative).all():
                raise RuntimeError(f"invalid relative action shape/content: {relative.shape}")
            expected_prefix = prefix_preflight[request_index]["prefix_length"]
            if output.prefix_length != expected_prefix:
                raise RuntimeError(
                    "live/preflight prefix length mismatch: "
                    f"{output.prefix_length} != {expected_prefix}"
                )
            absolute = _relative26_to_absolute14(relative, state)
            eval_steps = int(segment["eval_steps"])
            predicted_eval = absolute[:eval_steps].copy()
            target = np.stack(
                [
                    _ground_truth_absolute14(parsed["trajectory"][frame])
                    for frame in range(start, start + eval_steps)
                ]
            ).astype(np.float32)
            frame_numbers = np.arange(start, start + eval_steps, dtype=np.int64)
            metrics = _absolute14_metrics(predicted_eval, target)
            graph = _policy_graph_stats(policy)
            action_graph = graph["action"]
            vision_graph = graph["vision"]
            completed_requests = request_index + 1
            _validate_retained_graph_stats(
                # Optimized Action retains all prefix buckets: require one
                # BUILD per bucket and cumulative REPLAY for every other call.
                # Only the fallback evicts exact-prefix per-step entries.
                "action", action_graph,
                calls=completed_requests if opt else 10
            )
            _validate_retained_graph_stats(
                "vision", vision_graph, calls=completed_requests * (1 if opt else 3)
            )
            _validate_retained_graph_stats(
                "base prefill", graph["base_prefill"], calls=completed_requests
            )
            print(
                "[wall-qwen35-openloop] Action Graph: "
                f"mode={action_mode}, entries={action_graph['size']}, "
                f"prefix_bucket={getattr(output, 'extra', {}).get('action_prefix_bucket')}, "
                f"segments={[entry['segment_count'] for entry in action_graph['entries']]}, "
                f"replays={action_graph['replays']}, "
                f"recaptures={action_graph['recaptures']}, "
                f"invariant_ok={action_graph['invariant_ok']}",
                flush=True,
            )
            print(
                "[wall-qwen35-openloop] Vision Graph: "
                f"entries={vision_graph['size']}, "
                f"segments={[entry['segment_count'] for entry in vision_graph['entries']]}, "
                f"replays={vision_graph['replays']}, "
                f"recaptures={vision_graph['recaptures']}, "
                f"invariant_ok={vision_graph['invariant_ok']}",
                flush=True,
            )
            print(
                "[wall-qwen35-openloop] Prefill Graph: "
                f"entries={graph['base_prefill']['size']}, "
                f"segments={[entry['segment_count'] for entry in graph['base_prefill']['entries']]}, "
                f"replays={graph['base_prefill']['replays']}, "
                f"recaptures={graph['base_prefill']['recaptures']}, "
                f"invariant_ok={graph['base_prefill']['invariant_ok']}",
                flush=True,
            )
            predictions_relative.append(relative)
            predictions_absolute.append(predicted_eval)
            ground_truth_chunks.append(target)
            time_chunks.append(frame_numbers)
            segment_results.append(
                {
                    **segment,
                    "request_idx": request_index,
                    "prefix_length": prefix_preflight[request_index]["prefix_length"],
                    "action_prefix_bucket": getattr(output, "extra", {}).get("action_prefix_bucket"),
                    "noise_seed": request_noise_seed,
                    "flow_noise_sha256": hashlib.sha256(
                        flow_noise[request_index].tobytes()
                    ).hexdigest(),
                    "prediction_steps": eval_steps,
                    "seconds": request_seconds,
                    **metrics,
                    "graph": graph,
                }
            )
            np.save(
                output_dir / "pred_relative26.npy",
                np.stack(predictions_relative).astype(np.float32),
            )
            np.save(
                output_dir / "pred_abs14.npy",
                np.concatenate(predictions_absolute).astype(np.float32),
            )
            np.save(
                output_dir / "pred_concat.npy",
                np.concatenate(predictions_absolute).astype(np.float32),
            )
            np.save(
                output_dir / "gt_abs14.npy",
                np.concatenate(ground_truth_chunks).astype(np.float32),
            )
            np.save(
                output_dir / "gt_concat.npy",
                np.concatenate(ground_truth_chunks).astype(np.float32),
            )
            np.save(
                output_dir / "time.npy",
                np.concatenate(time_chunks).astype(np.int64),
            )
            np.save(
                output_dir / "time_concat.npy",
                np.concatenate(time_chunks).astype(np.int64),
            )
            _json_dump(output_dir / "segments.json", segment_results)
            print(
                f"[wall-qwen35-openloop] request {request_index + 1}/"
                f"{len(parsed['segments'])} frames=[{start},{start + eval_steps}) "
                f"prefix={prefix_preflight[request_index]['prefix_length']} "
                f"L1={metrics['abs14_l1']:.6f} "
                f"rot={metrics['rotation_geodesic_deg']:.3f}deg "
                f"time={request_seconds:.3f}s",
                flush=True,
            )
    finally:
        if hw_perf_meta["enabled"]:
            hw_perf_meta["native_state_before_disable"] = dict(
                torch.rpu.get_hw_perf_trace()
            )
        runtime_stack.close()
        _finalize_hw_perf_metadata(hw_perf_meta)

    prediction_concat = np.concatenate(predictions_absolute).astype(np.float32)
    ground_truth_concat = np.concatenate(ground_truth_chunks).astype(np.float32)
    overall_metrics = _absolute14_metrics(prediction_concat, ground_truth_concat)
    plot = _write_plot(
        output_dir,
        predictions_absolute,
        ground_truth_chunks,
        time_chunks,
    )
    final_meta = {
        **common_meta,
        "status": "complete",
        "install_seconds": install_seconds,
        "evaluated_steps": int(len(prediction_concat)),
        "metrics": overall_metrics,
        "abs14_l1_overall": overall_metrics["abs14_l1"],
        "prediction_space": (
            "absolute dual-arm [L_xyz,L_rpy,L_gripper,R_xyz,R_rpy,R_gripper]"
        ),
        "raw_policy_output": "physical relative [requests,32,26]",
        "ground_truth_source": "recorded master arm fields at the same frame index",
        "plot": plot,
        "final_graph": segment_results[-1]["graph"],
    }
    _json_dump(output_dir / "meta.json", final_meta)
    print(
        f"[wall-qwen35-openloop] DONE: requests={len(segment_results)}, "
        f"steps={len(prediction_concat)}, overall abs14 "
        f"L1={overall_metrics['abs14_l1']:.6f}",
        flush=True,
    )
    print(f"[wall-qwen35-openloop] output: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
