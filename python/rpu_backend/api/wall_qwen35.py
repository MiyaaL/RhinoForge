"""Public controlled-evaluation facade for the private Wall Qwen3.5 policy.

This entry point is intentionally bound to one checkpoint profile.  It is not
an architecture-wide Qwen3.5 support claim: the vision tower remains
numeric-blocked and requires an explicit controlled-evaluation opt-in.
"""

from __future__ import annotations

import dataclasses
import os
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import torch


_CAMERA_NAMES = (
    "face_view",
    "left_wrist_view",
    "right_wrist_view",
)
_DATASET_KEYS = frozenset({"x2_normal"})
_ACTION_DIM = 26
_ACTION_HORIZON = 32
_MAX_SEQ_LEN = 780
_ROBOT_ID = "10070"
_PROFILE_MASK = (1.0,) * 20 + (0.0,) * 6


def _finite_vector(value: Any, *, name: str, binary: bool = False) -> torch.Tensor:
    try:
        tensor = (value if isinstance(value, torch.Tensor) and value.dtype == torch.float32
                  else torch.as_tensor(value, dtype=torch.float32))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f"WallQwen35Policy {name} must be tensor-like") from exc
    if tensor.device.type != "cpu":
        raise ValueError(f"WallQwen35Policy {name} must be on CPU")
    tensor = tensor if tensor.ndim == 1 and tensor.is_contiguous() else tensor.reshape(-1).contiguous()
    if tensor.numel() != _ACTION_DIM:
        raise ValueError(
            f"WallQwen35Policy {name} must contain {_ACTION_DIM} values, "
            f"got {tensor.numel()}"
        )
    array = (tensor.detach().resolve_neg().numpy() if tensor.requires_grad or tensor.is_neg() else tensor.numpy())
    if not bool(np.isfinite(array).all()):
        raise ValueError(f"WallQwen35Policy {name} must contain only finite values")
    if binary and not bool(((array == 0) | (array == 1)).all()):
        raise ValueError(f"WallQwen35Policy {name} must contain only 0 or 1")
    return tensor


def _finite_initial_noise(value: Any) -> torch.Tensor:
    try:
        tensor = (value if isinstance(value, torch.Tensor) and value.dtype == torch.float32
                  else torch.as_tensor(value, dtype=torch.float32))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "WallQwen35Policy initial_noise must be tensor-like"
        ) from exc
    if tensor.device.type != "cpu":
        raise ValueError("WallQwen35Policy initial_noise must be on CPU")
    if tuple(tensor.shape) == (_ACTION_HORIZON, _ACTION_DIM):
        tensor = tensor.unsqueeze(0)
    if tuple(tensor.shape) != (1, _ACTION_HORIZON, _ACTION_DIM):
        raise ValueError(
            "WallQwen35Policy initial_noise must have shape [32,26] or "
            f"[1,32,26], got {tuple(tensor.shape)}"
        )
    array = (tensor.detach().resolve_neg().numpy() if tensor.requires_grad or tensor.is_neg() else tensor.numpy())
    if not bool(np.isfinite(array).all()):
        raise ValueError(
            "WallQwen35Policy initial_noise must contain only finite values"
        )
    return (tensor.contiguous().clone() if tensor.requires_grad else
            torch.from_numpy(np.array(array, copy=True, order="C")))


def _camera_mapping(images: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(images, Mapping):
        raise ValueError(
            "WallQwen35Policy images must be a mapping keyed by the three "
            "canonical camera names"
        )
    missing = [name for name in _CAMERA_NAMES if name not in images]
    extra = sorted(set(images) - set(_CAMERA_NAMES))
    if missing or extra:
        raise ValueError(
            "WallQwen35Policy images must contain exactly "
            f"{list(_CAMERA_NAMES)}; missing={missing}, extra={extra}"
        )
    return {name: images[name] for name in _CAMERA_NAMES}


@dataclasses.dataclass(frozen=True)
class WallQwen35ActionOutput:
    """One physical-unit action chunk returned on CPU.

    ``actions`` is contiguous FP32 with shape ``[1, 32, 26]``.  It has already
    been de-normalized with the checkpoint's selected dataset statistics.
    """

    actions: torch.Tensor
    actions_norm: torch.Tensor | None = None
    prefix_length: int | None = None
    extra: dict[str, Any] = dataclasses.field(default_factory=dict)


class WallQwen35Policy:
    """Exact-checkpoint Wall Qwen3.5 flow policy for controlled RPU evaluation."""

    def __init__(self) -> None:
        raise RuntimeError(
            "WallQwen35Policy() is not a public constructor; use "
            "WallQwen35Policy.from_checkpoint(...).to('rpu')"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | os.PathLike[str],
        *,
        dataset_key: str = "x2_normal",
        robot_id: str | int = _ROBOT_ID,
        camera_names: Sequence[str] = _CAMERA_NAMES,
        max_seq_len: int = _MAX_SEQ_LEN,
        allow_numeric_blocked_vision: bool = False,
    ) -> "WallQwen35Policy":
        """Validate and bind the one admitted checkpoint without loading weights."""
        if not isinstance(dataset_key, str) or dataset_key not in _DATASET_KEYS:
            raise ValueError(
                "WallQwen35Policy initial profile requires dataset_key='x2_normal'"
            )
        if isinstance(robot_id, bool) or not isinstance(robot_id, (str, Integral)):
            raise ValueError("WallQwen35Policy robot_id must be '10070'")
        robot_id = str(robot_id)
        if robot_id != _ROBOT_ID:
            raise ValueError(
                f"WallQwen35Policy initial profile requires robot_id={_ROBOT_ID!r}"
            )
        if isinstance(camera_names, (str, bytes)) or tuple(camera_names) != _CAMERA_NAMES:
            raise ValueError(
                "WallQwen35Policy initial profile requires camera_names="
                f"{_CAMERA_NAMES!r}"
            )
        if (
            isinstance(max_seq_len, bool)
            or not isinstance(max_seq_len, Integral)
            or int(max_seq_len) != _MAX_SEQ_LEN
        ):
            raise ValueError(
                f"WallQwen35Policy initial profile requires max_seq_len={_MAX_SEQ_LEN}"
            )
        if not isinstance(allow_numeric_blocked_vision, bool):
            raise ValueError(
                "WallQwen35Policy allow_numeric_blocked_vision must be bool"
            )
        from rpu_backend.adapters.wall_qwen35.execution import resolve_wall_qwen35_opt
        opt = resolve_wall_qwen35_opt()

        checkpoint_path = Path(checkpoint).expanduser().resolve()
        from rpu_backend.adapters.wall_qwen35.checkpoint import (
            preflight_wall_qwen35_checkpoint,
        )

        profile = preflight_wall_qwen35_checkpoint(checkpoint_path)
        instance = cls.__new__(cls)
        instance._checkpoint = checkpoint_path
        instance._profile = profile
        instance._dataset_key = dataset_key
        instance._robot_id = robot_id
        instance._camera_names = _CAMERA_NAMES
        instance._max_seq_len = _MAX_SEQ_LEN
        instance._allow_numeric_blocked_vision = allow_numeric_blocked_vision
        instance._wall_qwen35_opt = opt
        instance._runtime = None
        instance._install_started = False
        instance._closed = False
        return instance

    @property
    def camera_names(self) -> tuple[str, str, str]:
        return self._camera_names

    @property
    def action_dim(self) -> int:
        return _ACTION_DIM

    @property
    def action_horizon(self) -> int:
        return _ACTION_HORIZON

    @property
    def robot_id(self) -> str:
        return self._robot_id

    def to(self, device: str | torch.device) -> "WallQwen35Policy":
        if str(device) != "rpu":
            raise ValueError("WallQwen35Policy supports only .to('rpu')")
        if self._closed:
            raise RuntimeError("WallQwen35Policy is closed")
        if self._runtime is not None:
            return self
        if self._install_started:
            raise RuntimeError(
                "WallQwen35Policy installation was already attempted; create a "
                "fresh policy after a failed irreversible weight transform"
            )
        if not self._allow_numeric_blocked_vision:
            raise RuntimeError(
                "Wall Qwen3.5 uses the experimental/numeric-blocked Qwen3.5 "
                "vision path. Recreate the policy with "
                "allow_numeric_blocked_vision=True for controlled evaluation."
            )

        from rpu_backend.adapters.wall_qwen35.execution import configure_wall_qwen35_execution
        from rpu_backend.adapters.wall_qwen35.action import require_wall_fp16_loop_runtime
        configure_wall_qwen35_execution(self._wall_qwen35_opt)
        require_wall_fp16_loop_runtime()
        self._install_started = True
        from rpu_backend.adapters.wall_qwen35.runtime import WallQwen35Runtime

        runtime = WallQwen35Runtime.from_checkpoint(
            self._checkpoint,
            profile=self._profile,
            dataset_key=self._dataset_key,
            robot_id=self._robot_id,
            camera_names=self._camera_names,
            max_seq_len=self._max_seq_len,
            allow_numeric_blocked_vision=True,
            wall_qwen35_opt=self._wall_qwen35_opt,
        )
        self._runtime = runtime.to("rpu")
        return self

    def predict_action_chunk(
        self,
        *,
        images: Mapping[str, Any],
        instruction: str | Mapping[str, str],
        proprioception: Any,
        agent_pos_mask: Any | None = None,
        dof_mask: Any | None = None,
        noise_seed: int | None = None,
        initial_noise: Any | None = None,
    ) -> WallQwen35ActionOutput:
        if self._runtime is None or self._closed:
            raise RuntimeError(
                "WallQwen35Policy must be live; call .to('rpu') before inference"
            )
        ordered_images = _camera_mapping(images)
        state = _finite_vector(proprioception, name="proprioception")
        state_mask = (
            torch.from_numpy(np.asarray(_PROFILE_MASK, dtype=np.float32))
            if agent_pos_mask is None
            else _finite_vector(agent_pos_mask, name="agent_pos_mask", binary=True)
        )
        action_mask = (
            torch.from_numpy(np.asarray(_PROFILE_MASK, dtype=np.float32))
            if dof_mask is None
            else _finite_vector(dof_mask, name="dof_mask", binary=True)
        )
        if not np.array_equal(state_mask.numpy(force=True), _PROFILE_MASK):
            raise ValueError(
                "WallQwen35Policy agent_pos_mask must be [1]*20 + [0]*6"
            )
        if not np.array_equal(action_mask.numpy(force=True), _PROFILE_MASK):
            raise ValueError("WallQwen35Policy dof_mask must be [1]*20 + [0]*6")
        if noise_seed is not None and (
            isinstance(noise_seed, bool) or not isinstance(noise_seed, Integral)
        ):
            raise ValueError("WallQwen35Policy noise_seed must be an integer or None")
        if initial_noise is not None and noise_seed is not None:
            raise ValueError(
                "WallQwen35Policy accepts either noise_seed or initial_noise, not both"
            )
        explicit_noise = (
            None if initial_noise is None else _finite_initial_noise(initial_noise)
        )
        resolved_noise_seed = (
            0 if initial_noise is None and noise_seed is None else noise_seed
        )
        if not isinstance(instruction, (str, Mapping)):
            raise ValueError(
                "WallQwen35Policy instruction must be a string or task/detail mapping"
            )

        result = self._runtime.predict_action_chunk(
            images=ordered_images,
            instruction=instruction,
            proprioception=state,
            agent_pos_mask=state_mask,
            dof_mask=action_mask,
            noise_seed=(
                None
                if resolved_noise_seed is None
                else int(resolved_noise_seed)
            ),
            initial_noise=explicit_noise,
        )
        if isinstance(result, WallQwen35ActionOutput):
            output = result
        elif isinstance(result, Mapping):
            output = WallQwen35ActionOutput(**result)
        else:
            output = WallQwen35ActionOutput(actions=result)
        actions = output.actions.detach().to(device="cpu", dtype=torch.float32)
        if tuple(actions.shape) != (1, _ACTION_HORIZON, _ACTION_DIM):
            raise RuntimeError(
                "Wall Qwen3.5 runtime returned an invalid action shape: "
                f"{tuple(actions.shape)}"
            )
        if not bool(torch.isfinite(actions).all()):
            raise RuntimeError("Wall Qwen3.5 runtime returned non-finite actions")
        actions_norm = output.actions_norm
        if actions_norm is not None:
            actions_norm = actions_norm.detach().to(
                device="cpu", dtype=torch.float32
            ).contiguous().clone()
        return dataclasses.replace(
            output,
            actions=actions.contiguous().clone(),
            actions_norm=actions_norm,
        )

    infer = predict_action_chunk

    def close(self) -> None:
        if self._closed:
            return
        runtime = self._runtime
        close = getattr(runtime, "close", None)
        if callable(close):
            close()
        self._runtime = None
        self._closed = True

    def __enter__(self) -> "WallQwen35Policy":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.close()


__all__ = ["WallQwen35ActionOutput", "WallQwen35Policy"]
