"""Exact-checkpoint Wall Qwen3.5 action decoder.

The FP16 loop moves action I/O and all ten Euler updates onto RPU, using FP16
operands/results with ACC32 GEMMs. Time/Ada conditioning is precomputed once
in host FP32 and uploaded in FP16. Both Graph arms use these same semantics;
the host FP32 helpers below remain numerical references only. The shell uses the existing
Qwen3.5 installer as all-full attention; Wall native mode then executes the six
checkpoint attention layers and treats the remaining token mixers as identity
branches while retaining every checkpoint MLP.

Scale-4 has one owner at each physical boundary:

* ``w1`` is divided by four exactly once, after GEMM (FP16 loop) or immediately
  before upload (decoder-only reference helper);
* AdaLN rows are packed as ``[1 + scale, shift, gate / 4]`` in FP32 and the
  complete row is cast once to FP16;
* the native Wall mode uses ``rms_norm_eps / 16`` for adaptive RMSNorm.

No other input, output, or Euler helper applies scale-4.
"""

from __future__ import annotations

import math
import os
import re
import threading
import types
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .checkpoint import (
    WallQwen35CheckpointManifest,
    _admit_manifest,
    _file_identity,
    _is_action_tensor,
)


ACTION_BATCH_SIZE = 1
ACTION_HORIZON = 32
ACTION_DIM = 26
ACTION_HIDDEN_SIZE = 1024
ACTION_INTERMEDIATE_SIZE = 2048
ACTION_NUM_LAYERS = 24
ACTION_MAX_SEQ_LEN = 780
ACTION_MAX_PREFIX_LEN = 384
ACTION_NUM_STEPS = 10
_NUM_Q_HEADS = 8
_NUM_KV_HEADS = 2
_HEAD_DIM = 256
_ROTARY_DIM = 64
_ROPE_THETA = 10_000_000.0
_FULL_LAYERS = (3, 7, 11, 15, 19, 23)
_FULL_LAYER_SET = frozenset(_FULL_LAYERS)
_MODULATION_ROWS = 2 * ACTION_NUM_LAYERS + 1
_MODULATION_WIDTH = 3 * ACTION_HIDDEN_SIZE
_ACTION_INSTALL_LOCK = threading.Lock()
_ACTION_PACKED_DIM = 64
_NORMALIZER_RE = re.compile(
    r"^model\.action_processor\.normalizer_(action|propri)\."
    r"(min|delta)\.(.+)$"
)


@dataclass(frozen=True, slots=True)
class WallHostLinear:
    """Plain host weights, deliberately outside ``nn.Module`` traversal."""

    weight: torch.Tensor
    bias: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class WallNormalizerBank:
    minimum: Mapping[str, torch.Tensor]
    delta: Mapping[str, torch.Tensor]
    selected_key: str
    selected_minimum: torch.Tensor
    selected_delta: torch.Tensor


@dataclass(frozen=True, slots=True)
class WallActionProcessor:
    """Host-FP32 action projections and checkpoint normalization vectors."""

    w1: WallHostLinear
    time_mlp_in: WallHostLinear
    time_mlp_out: WallHostLinear
    action_proj_back: WallHostLinear
    propri_proj: WallHostLinear
    normalizer_action: WallNormalizerBank
    normalizer_propri: WallNormalizerBank


@dataclass(frozen=True, slots=True)
class _WallAdaProjection:
    weight: torch.Tensor
    bias: torch.Tensor


class _VectorWeight(nn.Module):
    def __init__(self, width: int, *, device: str = "meta") -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(width, device=device, dtype=torch.float16),
            requires_grad=False,
        )


class _WallAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(
            ACTION_HIDDEN_SIZE,
            _NUM_Q_HEADS * _HEAD_DIM * 2,
            bias=False,
            device="meta",
            dtype=torch.float16,
        )
        self.k_proj = nn.Linear(
            ACTION_HIDDEN_SIZE,
            _NUM_KV_HEADS * _HEAD_DIM,
            bias=False,
            device="meta",
            dtype=torch.float16,
        )
        self.v_proj = nn.Linear(
            ACTION_HIDDEN_SIZE,
            _NUM_KV_HEADS * _HEAD_DIM,
            bias=False,
            device="meta",
            dtype=torch.float16,
        )
        self.o_proj = nn.Linear(
            _NUM_Q_HEADS * _HEAD_DIM,
            ACTION_HIDDEN_SIZE,
            bias=False,
            device="meta",
            dtype=torch.float16,
        )
        self.q_norm = _VectorWeight(_HEAD_DIM)
        self.k_norm = _VectorWeight(_HEAD_DIM)


class _WallMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            ACTION_HIDDEN_SIZE,
            ACTION_INTERMEDIATE_SIZE,
            bias=False,
            device="meta",
            dtype=torch.float16,
        )
        self.up_proj = nn.Linear(
            ACTION_HIDDEN_SIZE,
            ACTION_INTERMEDIATE_SIZE,
            bias=False,
            device="meta",
            dtype=torch.float16,
        )
        self.down_proj = nn.Linear(
            ACTION_INTERMEDIATE_SIZE,
            ACTION_HIDDEN_SIZE,
            bias=False,
            device="meta",
            dtype=torch.float16,
        )


class _WallLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = _WallAttention()
        self.mlp = _WallMLP()


class WallQwen35ActionModule(nn.Module):
    """Minimal Qwen3.5-shaped shell consumed by the shared RPU installer."""

    def __init__(self) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(
            hidden_size=ACTION_HIDDEN_SIZE,
            intermediate_size=ACTION_INTERMEDIATE_SIZE,
            num_hidden_layers=ACTION_NUM_LAYERS,
            num_attention_heads=_NUM_Q_HEADS,
            num_key_value_heads=_NUM_KV_HEADS,
            head_dim=_HEAD_DIM,
            attention_bias=False,
            attn_output_gate=True,
            hidden_act="silu",
            rms_norm_eps=1.0e-6,
            adaptive_mode="adaLN",
            layer_types=("full_attention",) * ACTION_NUM_LAYERS,
            rope_parameters={
                "rope_theta": _ROPE_THETA,
                "rope_type": "default",
                "mrope_section": (11, 11, 10),
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.25,
            },
            # Structurally required by the shared installer/cache factory even
            # though the all-full shell never builds or allocates GDN state.
            linear_num_value_heads=16,
            linear_num_key_heads=16,
            linear_key_head_dim=128,
            linear_value_head_dim=128,
            linear_conv_kernel_dim=4,
        )
        self.layers = nn.ModuleList(_WallLayer() for _ in range(ACTION_NUM_LAYERS))
        self.norm = _VectorWeight(ACTION_HIDDEN_SIZE)

        # These are plain dataclasses/tuples, not Modules: the shared weight
        # converter must never swizzle host-FP32 Ada or action-processor weights.
        self.action_processor: WallActionProcessor | None = None
        self._wall_ada_rows: tuple[_WallAdaProjection, ...] = ()

    def forward(self, *args, **kwargs):  # pragma: no cover - replaced by run helper
        del args, kwargs
        raise RuntimeError("use run_wall_qwen35_action() after RPU installation")


def _unsupported(message: str) -> BaseException:
    try:
        from rpu_backend.api.errors import UnsupportedModelError

        return UnsupportedModelError(message)
    except ImportError:
        return ValueError(message)


def _expected_structural_shapes() -> dict[str, tuple[int, ...]]:
    expected: dict[str, tuple[int, ...]] = {}
    for index in range(ACTION_NUM_LAYERS):
        stem = f"model.language_model.layers.{index}"
        for norm_name in (
            "input_layernorm_action",
            "post_attention_layernorm_action",
        ):
            expected[f"{stem}.{norm_name}.dense.weight"] = (
                _MODULATION_WIDTH,
                ACTION_HIDDEN_SIZE,
            )
            expected[f"{stem}.{norm_name}.dense.bias"] = (_MODULATION_WIDTH,)
        expected[f"{stem}.mlp_action_expert.gate_up_proj.weight"] = (
            2 * ACTION_INTERMEDIATE_SIZE,
            ACTION_HIDDEN_SIZE,
        )
        expected[f"{stem}.mlp_action_expert.down_proj.weight"] = (
            ACTION_HIDDEN_SIZE,
            ACTION_INTERMEDIATE_SIZE,
        )
        if index in _FULL_LAYER_SET:
            attention = f"{stem}.self_attn"
            expected[f"{attention}.q_proj_action.weight"] = (
                _NUM_Q_HEADS * _HEAD_DIM * 2,
                ACTION_HIDDEN_SIZE,
            )
            expected[f"{attention}.k_proj_action.weight"] = (
                _NUM_KV_HEADS * _HEAD_DIM,
                ACTION_HIDDEN_SIZE,
            )
            expected[f"{attention}.v_proj_action.weight"] = (
                _NUM_KV_HEADS * _HEAD_DIM,
                ACTION_HIDDEN_SIZE,
            )
            expected[f"{attention}.o_proj_action.weight"] = (
                ACTION_HIDDEN_SIZE,
                _NUM_Q_HEADS * _HEAD_DIM,
            )
            expected[f"{attention}.q_norm_action.weight"] = (_HEAD_DIM,)
            expected[f"{attention}.k_norm_action.weight"] = (_HEAD_DIM,)
    expected["model.language_model.norm_action.dense.weight"] = (
        _MODULATION_WIDTH,
        ACTION_HIDDEN_SIZE,
    )
    expected["model.language_model.norm_action.dense.bias"] = (
        _MODULATION_WIDTH,
    )
    expected.update(
        {
            "model.action_processor.w1.weight": (ACTION_HIDDEN_SIZE, 52),
            "model.action_processor.time_mlp_in.weight": (
                ACTION_HIDDEN_SIZE,
                ACTION_HIDDEN_SIZE,
            ),
            "model.action_processor.time_mlp_in.bias": (ACTION_HIDDEN_SIZE,),
            "model.action_processor.time_mlp_out.weight": (
                ACTION_HIDDEN_SIZE,
                ACTION_HIDDEN_SIZE,
            ),
            "model.action_processor.time_mlp_out.bias": (ACTION_HIDDEN_SIZE,),
            "model.action_processor.action_proj_back.weight": (
                ACTION_DIM,
                ACTION_HIDDEN_SIZE,
            ),
            "model.action_processor.propri_proj.weight": (2560, 52),
        }
    )
    return expected


def _validate_action_metadata(handle) -> tuple[set[str], tuple[str, ...]]:
    action_names = {name for name in handle.keys() if _is_action_tensor(name)}
    expected = _expected_structural_shapes()
    missing = sorted(set(expected) - action_names)
    if missing:
        raise _unsupported(f"Wall action checkpoint is missing keys: {missing}")
    bad: list[str] = []
    for name, shape in expected.items():
        value = handle.get_slice(name)
        actual = tuple(int(dimension) for dimension in value.get_shape())
        if str(value.get_dtype()) != "F32" or actual != shape:
            bad.append(f"{name}: F32{shape} expected, got {value.get_dtype()}{actual}")

    groups: dict[tuple[str, str], set[str]] = {
        (normalizer, statistic): set()
        for normalizer in ("action", "propri")
        for statistic in ("min", "delta")
    }
    extras = action_names - set(expected)
    for name in sorted(extras):
        match = _NORMALIZER_RE.fullmatch(name)
        if match is None:
            bad.append(f"unexpected action key {name}")
            continue
        normalizer, statistic, dataset = match.groups()
        value = handle.get_slice(name)
        shape = tuple(int(dimension) for dimension in value.get_shape())
        if str(value.get_dtype()) != "F32" or shape != (ACTION_DIM,):
            bad.append(f"{name}: expected F32({ACTION_DIM},), got {value.get_dtype()}{shape}")
        groups[(normalizer, statistic)].add(dataset)

    dataset_sets = tuple(groups[key] for key in sorted(groups))
    if not dataset_sets or any(values != dataset_sets[0] for values in dataset_sets):
        bad.append("action/propri min/delta normalizer dataset keys differ")
    datasets = tuple(sorted(dataset_sets[0])) if dataset_sets else ()
    if len(datasets) != 77 or not {"x2_normal", "ex_normal"}.issubset(datasets):
        bad.append(
            "Wall checkpoint must contain the exact 77-dataset normalizer bank"
        )
    if len(action_names) != 497:
        bad.append(f"expected 497 action tensors, got {len(action_names)}")
    if bad:
        raise _unsupported("Wall action key/dtype/shape profile mismatch: " + "; ".join(bad))
    return action_names, datasets


def _cpu_tensor(handle, name: str, *, dtype: torch.dtype) -> torch.Tensor:
    value = handle.get_tensor(name)
    return value.detach().to(device="cpu", dtype=dtype).contiguous()


def _set_weight(module: nn.Module, value: torch.Tensor) -> None:
    old = module._parameters.get("weight")
    if old is None or tuple(old.shape) != tuple(value.shape):
        raise RuntimeError(
            f"Wall action shell weight shape drift: {tuple(value.shape)} versus "
            f"{None if old is None else tuple(old.shape)}"
        )
    module._parameters["weight"] = nn.Parameter(value, requires_grad=False)


def _host_linear(handle, stem: str, *, bias: bool) -> WallHostLinear:
    weight = _cpu_tensor(handle, f"{stem}.weight", dtype=torch.float32)
    bias_value = (
        _cpu_tensor(handle, f"{stem}.bias", dtype=torch.float32)
        if bias
        else None
    )
    return WallHostLinear(weight=weight, bias=bias_value)


def _normalizer_bank(
    handle,
    kind: str,
    datasets: tuple[str, ...],
    *,
    selected_minimum: torch.Tensor | None,
    selected_delta: torch.Tensor | None,
    selected_key_hint: str | None = None,
) -> WallNormalizerBank:
    minimum = {
        dataset: _cpu_tensor(
            handle,
            f"model.action_processor.normalizer_{kind}.min.{dataset}",
            dtype=torch.float32,
        )
        for dataset in datasets
    }
    delta = {
        dataset: _cpu_tensor(
            handle,
            f"model.action_processor.normalizer_{kind}.delta.{dataset}",
            dtype=torch.float32,
        )
        for dataset in datasets
    }
    if selected_minimum is None and selected_delta is None:
        selected_key = (
            "x2_normal" if selected_key_hint is None else selected_key_hint
        )
        if selected_key != "x2_normal":
            raise ValueError("Wall selected normalizer key is outside the profile")
        selected_minimum = minimum[selected_key]
        selected_delta = delta[selected_key]
    elif selected_minimum is None or selected_delta is None:
        raise ValueError("selected normalizer minimum and delta must be provided together")
    else:
        selected_minimum = selected_minimum.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        selected_delta = selected_delta.detach().to(
            device="cpu", dtype=torch.float32
        ).contiguous()
        if tuple(selected_minimum.shape) != (ACTION_DIM,) or tuple(
            selected_delta.shape
        ) != (ACTION_DIM,):
            raise ValueError("Wall selected normalizer tensors must have shape [26]")
        matches = [
            dataset
            for dataset in datasets
            if torch.equal(selected_minimum, minimum[dataset])
            and torch.equal(selected_delta, delta[dataset])
        ]
        preferred = [dataset for dataset in ("x2_normal",) if dataset in matches]
        if len(preferred) != 1:
            raise ValueError(
                "selected action normalizer does not identify exactly one admitted "
                "Wall inference dataset"
            )
        # rd_normal intentionally aliases x2_normal in this checkpoint.  The
        # public inference profile admits only x2_normal, so prefer
        # that exact profile label instead of demanding global bank uniqueness.
        selected_key = preferred[0]
    if not bool(
        torch.isfinite(selected_minimum).all()
        and torch.isfinite(selected_delta).all()
        and selected_delta.ne(0).all()
    ):
        raise ValueError("Wall selected normalizer must be finite with nonzero delta")
    return WallNormalizerBank(
        minimum=minimum,
        delta=delta,
        selected_key=selected_key,
        selected_minimum=selected_minimum.clone(),
        selected_delta=selected_delta.clone(),
    )


def load_wall_qwen35_action_module(
    checkpoint: str | Path,
    *,
    manifest: WallQwen35CheckpointManifest | None = None,
    action_min: torch.Tensor | None = None,
    action_delta: torch.Tensor | None = None,
) -> WallQwen35ActionModule:
    """Admit metadata first, then stream only the 497 action tensors.

    The returned decoder weights are CPU FP16.  Host Ada, action processor,
    and normalizer tensors remain CPU FP32 and are plain objects outside the
    synthetic module traversal.
    """

    admitted = _admit_manifest(checkpoint, manifest)
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("Wall Qwen3.5 action loading requires safetensors") from error

    weights_path = admitted.root / "model.safetensors"
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        action_names, datasets = _validate_action_metadata(handle)

        # No module or host tensor is mutated/materialized until the complete
        # action key/dtype/shape profile above has closed.
        expert = WallQwen35ActionModule()
        shared_zeros = {
            "q": torch.zeros(
                _NUM_Q_HEADS * _HEAD_DIM * 2,
                ACTION_HIDDEN_SIZE,
                dtype=torch.float16,
            ),
            "k": torch.zeros(
                _NUM_KV_HEADS * _HEAD_DIM,
                ACTION_HIDDEN_SIZE,
                dtype=torch.float16,
            ),
            "v": torch.zeros(
                _NUM_KV_HEADS * _HEAD_DIM,
                ACTION_HIDDEN_SIZE,
                dtype=torch.float16,
            ),
            "o": torch.zeros(
                ACTION_HIDDEN_SIZE,
                _NUM_Q_HEADS * _HEAD_DIM,
                dtype=torch.float16,
            ),
            "q_norm": torch.zeros(_HEAD_DIM, dtype=torch.float16),
            "k_norm": torch.zeros(_HEAD_DIM, dtype=torch.float16),
        }
        ada_rows: list[_WallAdaProjection] = []
        for index, layer in enumerate(expert.layers):
            stem = f"model.language_model.layers.{index}"
            for norm_name in (
                "input_layernorm_action",
                "post_attention_layernorm_action",
            ):
                ada_stem = f"{stem}.{norm_name}.dense"
                ada_rows.append(
                    _WallAdaProjection(
                        weight=_cpu_tensor(
                            handle, f"{ada_stem}.weight", dtype=torch.float32
                        ),
                        bias=_cpu_tensor(
                            handle, f"{ada_stem}.bias", dtype=torch.float32
                        ),
                    )
                )

            gate_up = _cpu_tensor(
                handle,
                f"{stem}.mlp_action_expert.gate_up_proj.weight",
                dtype=torch.float16,
            )
            gate, up = gate_up.chunk(2, dim=0)
            _set_weight(layer.mlp.gate_proj, gate.contiguous())
            _set_weight(layer.mlp.up_proj, up.contiguous())
            _set_weight(
                layer.mlp.down_proj,
                _cpu_tensor(
                    handle,
                    f"{stem}.mlp_action_expert.down_proj.weight",
                    dtype=torch.float16,
                ),
            )

            attention = layer.self_attn
            if index in _FULL_LAYER_SET:
                attention_stem = f"{stem}.self_attn"
                for target, source in (
                    (attention.q_proj, "q_proj_action"),
                    (attention.k_proj, "k_proj_action"),
                    (attention.v_proj, "v_proj_action"),
                    (attention.o_proj, "o_proj_action"),
                    (attention.q_norm, "q_norm_action"),
                    (attention.k_norm, "k_norm_action"),
                ):
                    _set_weight(
                        target,
                        _cpu_tensor(
                            handle,
                            f"{attention_stem}.{source}.weight",
                            dtype=torch.float16,
                        ),
                    )
            else:
                # Distinct Parameter wrappers share the immutable zero storage
                # until the installer creates each layer's transformed copy.
                for target, key in (
                    (attention.q_proj, "q"),
                    (attention.k_proj, "k"),
                    (attention.v_proj, "v"),
                    (attention.o_proj, "o"),
                    (attention.q_norm, "q_norm"),
                    (attention.k_norm, "k_norm"),
                ):
                    _set_weight(target, shared_zeros[key])

        final_stem = "model.language_model.norm_action.dense"
        ada_rows.append(
            _WallAdaProjection(
                weight=_cpu_tensor(
                    handle, f"{final_stem}.weight", dtype=torch.float32
                ),
                bias=_cpu_tensor(
                    handle, f"{final_stem}.bias", dtype=torch.float32
                ),
            )
        )
        _set_weight(
            expert.norm,
            torch.zeros(ACTION_HIDDEN_SIZE, dtype=torch.float16),
        )
        if len(ada_rows) != _MODULATION_ROWS:
            raise RuntimeError(f"Wall Ada row count {len(ada_rows)} != 49")

        action_bank = _normalizer_bank(
            handle,
            "action",
            datasets,
            selected_minimum=action_min,
            selected_delta=action_delta,
            selected_key_hint=None,
        )
        propri_bank = _normalizer_bank(
            handle,
            "propri",
            datasets,
            selected_minimum=None,
            selected_delta=None,
            selected_key_hint=action_bank.selected_key,
        )
        expert.action_processor = WallActionProcessor(
            w1=_host_linear(handle, "model.action_processor.w1", bias=False),
            time_mlp_in=_host_linear(
                handle, "model.action_processor.time_mlp_in", bias=True
            ),
            time_mlp_out=_host_linear(
                handle, "model.action_processor.time_mlp_out", bias=True
            ),
            action_proj_back=_host_linear(
                handle, "model.action_processor.action_proj_back", bias=False
            ),
            propri_proj=_host_linear(
                handle, "model.action_processor.propri_proj", bias=False
            ),
            normalizer_action=action_bank,
            normalizer_propri=propri_bank,
        )
        expert._wall_ada_rows = tuple(ada_rows)

        # The admitted exact manifest plus the semantic profile above means all
        # 497 action keys are intentionally consumed by the shell/host banks.
        consumed_count = len(_expected_structural_shapes()) + 4 * len(datasets)
        if consumed_count != len(action_names):
            raise RuntimeError(
                f"Wall action loader consumed {consumed_count} of {len(action_names)} keys"
            )

    if _file_identity(weights_path) != admitted.stat_for("model.safetensors"):
        raise RuntimeError("Wall model.safetensors changed during action loading")
    expert.eval()
    expert._wall_qwen35_checkpoint_manifest = admitted
    expert._wall_qwen35_action_load_complete = True
    return expert


def _check_action_shell(expert: WallQwen35ActionModule) -> None:
    config = expert.config
    rope = config.rope_parameters
    profile = (
        int(config.hidden_size),
        int(config.intermediate_size),
        int(config.num_hidden_layers),
        int(config.num_attention_heads),
        int(config.num_key_value_heads),
        int(config.head_dim),
        bool(config.attention_bias),
        bool(config.attn_output_gate),
        str(config.hidden_act),
        float(config.rms_norm_eps),
        str(config.adaptive_mode),
        tuple(config.layer_types),
        float(rope.get("rope_theta", 0.0)),
        str(rope.get("rope_type", "")),
        tuple(int(value) for value in rope.get("mrope_section", ())),
        bool(rope.get("mrope_interleaved", False)),
        float(rope.get("partial_rotary_factor", 0.0)),
    )
    expected = (
        ACTION_HIDDEN_SIZE,
        ACTION_INTERMEDIATE_SIZE,
        ACTION_NUM_LAYERS,
        _NUM_Q_HEADS,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        False,
        True,
        "silu",
        1.0e-6,
        "adaLN",
        ("full_attention",) * ACTION_NUM_LAYERS,
        _ROPE_THETA,
        "default",
        (11, 11, 10),
        True,
        0.25,
    )
    if profile != expected:
        raise _unsupported(f"Wall action shell profile mismatch: {profile}")
    if len(expert._wall_ada_rows) != _MODULATION_ROWS:
        raise _unsupported("Wall action shell must carry exactly 49 Ada projections")
    if not isinstance(expert.action_processor, WallActionProcessor):
        raise _unsupported("Wall action shell has no admitted host processor")


def _drop_installed_shell_weights(expert: WallQwen35ActionModule) -> None:
    """Release CPU decoder weights after the native handle owns RPU copies."""

    for parameter in expert.parameters():
        parameter.data = torch.empty(0, dtype=torch.float16, device="cpu")


def patch_wall_qwen35_action_for_rpu(
    expert: WallQwen35ActionModule,
    *,
    max_seq_len: int = ACTION_MAX_SEQ_LEN,
    one_graph: bool = True,
) -> WallQwen35ActionModule:
    """Install the exact decoder shell and select native Wall action mode."""

    if not isinstance(expert, WallQwen35ActionModule):
        raise TypeError("Wall action patch requires WallQwen35ActionModule")
    if type(one_graph) is not bool:
        raise ValueError("one_graph must be bool")
    require_wall_fp16_loop_runtime()
    if getattr(expert, "_wall_qwen35_action_ready", False):
        state = getattr(expert, "_rpu_qwen3_5", None)
        if state is not None and getattr(state, "action_graph_cache", None) is not None:
            if state.action_one_graph != one_graph:
                raise RuntimeError("Action execution cannot change after installation")
            return expert
        raise RuntimeError("Wall action ready marker has incomplete runtime state")
    if getattr(expert, "_wall_qwen35_install_started", False):
        raise RuntimeError(
            "Wall action installation was already attempted; reload a fresh module"
        )
    if expert.training:
        raise NotImplementedError("Wall action decoder is inference-only; call eval()")
    if isinstance(max_seq_len, bool) or not isinstance(max_seq_len, Integral):
        raise ValueError("Wall action max_seq_len must be an integer")
    if int(max_seq_len) != ACTION_MAX_SEQ_LEN:
        raise ValueError("initial Wall action profile requires max_seq_len=780")
    _check_action_shell(expert)
    if any(
        parameter.device.type != "cpu" or parameter.dtype is not torch.float16
        for parameter in expert.parameters()
    ):
        raise ValueError("Wall action shell parameters must be clean CPU FP16")
    io_weights = _wall_fp16_io_weights(expert)
    if not _ACTION_INSTALL_LOCK.acquire(blocking=False):
        raise RuntimeError("another Wall action installation is in progress")

    state_before = getattr(expert, "_rpu_qwen3_5", None)
    try:
        if state_before is not None or getattr(
            expert, "_rpu_qwen3_5_text_install_started", False
        ):
            raise RuntimeError("Wall action shell already has a Qwen3.5 runtime")
        expert._wall_qwen35_install_started = True
        from rpu_backend.adapters.qwen3_5.text import install_qwen3_5_text_for_rpu
        from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

        handle = install_qwen3_5_text_for_rpu(
            expert,
            expert.config,
            max_seq_len=int(max_seq_len),
            _cpu_stage_weights=True,
        )
        # All cold native controls precede the first forward/capture.
        torch.ops.rpu.qwen3_5_set_linear_acc32(handle, True)
        torch.ops.rpu.qwen3_5_enable_wall_action_mode(handle, list(_FULL_LAYERS))
        torch.ops.rpu.qwen3_5_set_fast_replay(handle, True)

        import rpu_backend as rpu_backend

        state = expert._rpu_qwen3_5
        state.action_one_graph = one_graph
        from rpu_backend.runtime.weights import tp_col_swizzle_mc_weight

        input_w, output_w = io_weights
        state.action_io_keepalive = (
            tp_col_swizzle_mc_weight(input_w, num_cores=1).contiguous().to("rpu"),
            torch.zeros(ACTION_HIDDEN_SIZE, dtype=torch.float16).to("rpu"),
            tp_col_swizzle_mc_weight(output_w, num_cores=1).contiguous().to("rpu"),
            torch.zeros(_ACTION_PACKED_DIM, dtype=torch.float16).to("rpu"),
        )
        torch.ops.rpu.qwen3_5_set_action_io_weights(
            handle, *state.action_io_keepalive,
            ACTION_DIM, _ACTION_PACKED_DIM, ACTION_HORIZON,
        )
        # Optimized Action fixes KV insertion/SDPA lengths per 64-row bucket.
        # Real prefix visibility and RoPE are refreshed before every capture.
        # Keep all six buckets: returning to an earlier bucket must REPLAY.
        state.action_graph_cache = rpu_backend.graph.GraphCache(
            max_entries=6 if one_graph else 1, require_single_segment=True
        )
        state.action_cache = Qwen3_5Cache.from_config(
            expert.config,
            max_seq_len=int(max_seq_len),
            device="rpu",
            dtype=torch.float16,
        )
        empty_cache = torch.empty(0, device="rpu", dtype=torch.float16)
        for index in range(ACTION_NUM_LAYERS):
            if index not in _FULL_LAYER_SET:
                state.action_cache.k_caches[index] = empty_cache
                state.action_cache.v_caches[index] = empty_cache
        state.action_k_caches = state.action_cache.k_caches
        state.action_v_caches = state.action_cache.v_caches
        state.action_prefix_lens = [0] * ACTION_NUM_LAYERS
        state.action_prefix_len = None
        state.action_prefix_bucket = None
        state.action_rope_cache = None
        state.action_modulation_cache = {}
        state.action_loop_modulation = None
        state.action_graph_signature = None
        _drop_installed_shell_weights(expert)
        expert._wall_qwen35_action_ready = True
        return expert
    except BaseException:
        state = getattr(expert, "_rpu_qwen3_5", None)
        if state is not None and state is not state_before:
            for name in ("graph_cache", "action_graph_cache"):
                cache = getattr(state, name, None)
                if cache is not None:
                    try:
                        cache.clear()
                    except Exception:
                        pass
            finalizer = getattr(state, "handle_finalizer", None)
            if finalizer is not None and getattr(finalizer, "alive", False):
                try:
                    finalizer()
                except Exception:
                    pass
        # Install-started markers intentionally remain: shared conversion is
        # irreversible and retrying this object would double-swizzle weights.
        raise
    finally:
        _ACTION_INSTALL_LOCK.release()


def _processor(expert: WallQwen35ActionModule) -> WallActionProcessor:
    processor = expert.action_processor
    if not isinstance(processor, WallActionProcessor):
        raise RuntimeError("Wall action module has no host processor")
    return processor


def _cpu_fp32(value: torch.Tensor, name: str, shape: tuple[int, ...]) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or tuple(value.shape) != shape:
        raise ValueError(f"{name} must be CPU with shape {shape}")
    result = value.detach().to(dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _dof_mask(value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.device.type != "cpu":
        raise ValueError("Wall dof_mask must be a CPU tensor")
    mask = value.detach().to(dtype=torch.float32)
    if tuple(mask.shape) == (ACTION_DIM,):
        mask = mask.view(1, 1, ACTION_DIM)
    elif tuple(mask.shape) == (1, ACTION_DIM):
        mask = mask.unsqueeze(1)
    if tuple(mask.shape) != (1, 1, ACTION_DIM):
        raise ValueError("Wall dof_mask must have shape [1,1,26]")
    if not bool(((mask == 0) | (mask == 1)).all()):
        raise ValueError("Wall dof_mask must contain only zero or one")
    return mask.contiguous()


def wall_qwen35_host_action_input(
    expert: WallQwen35ActionModule,
    action: torch.Tensor,
    dof_mask: torch.Tensor,
) -> torch.Tensor:
    """Return raw FP32 ``w1`` output; decoder upload owns the sole ``/4``."""

    action = _cpu_fp32(
        action, "Wall normalized action", (1, ACTION_HORIZON, ACTION_DIM)
    )
    mask = _dof_mask(dof_mask).expand_as(action)
    processor = _processor(expert)
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        return F.linear(
            torch.cat((action, mask), dim=-1), processor.w1.weight
        ).contiguous()


def _sinusoidal_time(timestep: torch.Tensor) -> torch.Tensor:
    half = ACTION_HIDDEN_SIZE // 2
    exponent = torch.log(torch.tensor(10_000.0, dtype=torch.float32)) / (half - 1)
    frequency = torch.exp(torch.arange(half, dtype=torch.float32) * -exponent)
    phase = timestep.reshape(-1, 1) * frequency.reshape(1, -1)
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


def wall_qwen35_host_time_condition(
    expert: WallQwen35ActionModule,
    timestep: torch.Tensor,
) -> torch.Tensor:
    timestep = _cpu_fp32(timestep, "Wall timestep", (1,))
    processor = _processor(expert)
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        value = _sinusoidal_time(timestep)
        value = F.linear(
            value, processor.time_mlp_in.weight, processor.time_mlp_in.bias
        )
        value = F.silu(value)
        value = F.linear(
            value, processor.time_mlp_out.weight, processor.time_mlp_out.bias
        )
        return F.silu(value).contiguous()


def wall_qwen35_host_velocity(
    expert: WallQwen35ActionModule,
    hidden: torch.Tensor,
) -> torch.Tensor:
    hidden = _cpu_fp32(
        hidden,
        "Wall action decoder hidden",
        (ACTION_BATCH_SIZE, ACTION_HORIZON, ACTION_HIDDEN_SIZE),
    )
    processor = _processor(expert)
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        return F.linear(hidden, processor.action_proj_back.weight).contiguous()


def validate_wall_qwen35_euler_profile(
    action: torch.Tensor,
    dof_mask: torch.Tensor,
    padding_velocity: torch.Tensor,
    *,
    num_steps: int = ACTION_NUM_STEPS,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(num_steps, bool) or not isinstance(num_steps, Integral):
        raise ValueError("Wall Euler num_steps must be an integer")
    if int(num_steps) != ACTION_NUM_STEPS:
        raise ValueError("Wall Euler profile requires exactly 10 steps")
    action = _cpu_fp32(action, "Wall Euler action", (1, 32, 26))
    mask = _dof_mask(dof_mask)
    padding_velocity = _cpu_fp32(
        padding_velocity, "Wall padding_velocity", (1, 32, 26)
    )
    return action, mask, padding_velocity


def wall_qwen35_host_euler_step(
    action: torch.Tensor,
    velocity: torch.Tensor,
    dof_mask: torch.Tensor,
    padding_velocity: torch.Tensor,
    *,
    delta_t: float,
) -> torch.Tensor:
    action, mask, padding_velocity = validate_wall_qwen35_euler_profile(
        action, dof_mask, padding_velocity
    )
    velocity = _cpu_fp32(velocity, "Wall action velocity", (1, 32, 26))
    if not isinstance(delta_t, (float, int)) or not math.isfinite(float(delta_t)):
        raise ValueError("Wall Euler delta_t must be finite")
    if float(delta_t) <= 0.0:
        raise ValueError("Wall Euler delta_t must be positive")
    active = mask.expand_as(action)
    mixed = padding_velocity * (1.0 - active) + velocity * active
    return (action + float(delta_t) * mixed).contiguous().clone()


def _adaptive_modulation_cpu(
    expert: WallQwen35ActionModule,
    time_cond: torch.Tensor,
) -> torch.Tensor:
    cond = _cpu_fp32(
        time_cond, "Wall time_cond", (ACTION_BATCH_SIZE, ACTION_HIDDEN_SIZE)
    )
    rows: list[torch.Tensor] = []
    with torch.inference_mode(), torch.autocast("cpu", enabled=False):
        for projection in expert._wall_ada_rows:
            raw = F.linear(cond, projection.weight, projection.bias)
            scale, shift, gate = raw.chunk(3, dim=-1)
            # Pack the physical row completely in FP32.  There is exactly one
            # dtype boundary for the full [49, 3072] schedule below.
            rows.append(torch.cat((scale + 1.0, shift, gate / 4.0), dim=-1))
    packed = torch.stack(rows, dim=0)[:, 0].contiguous()
    if tuple(packed.shape) != (_MODULATION_ROWS, _MODULATION_WIDTH):
        raise RuntimeError(f"Wall modulation shape drift: {tuple(packed.shape)}")
    return packed


def _adaptive_modulation(
    expert: WallQwen35ActionModule,
    time_cond: torch.Tensor,
) -> torch.Tensor:
    state = expert._rpu_qwen3_5
    cond = _cpu_fp32(
        time_cond, "Wall time_cond", (ACTION_BATCH_SIZE, ACTION_HIDDEN_SIZE)
    )
    key = cond.numpy().tobytes()
    cached = state.action_modulation_cache.get(key)
    if cached is not None:
        return cached
    modulation = _adaptive_modulation_cpu(expert, cond).to(
        device="rpu", dtype=torch.float16
    ).contiguous()
    if len(state.action_modulation_cache) < ACTION_NUM_STEPS:
        state.action_modulation_cache[key] = modulation
    return modulation


def require_wall_fp16_loop_runtime() -> None:
    """Reject stale native packages before loading/transforming model weights."""
    import rpu_backend

    if (getattr(getattr(rpu_backend, "_cpp_ext", None), "graph_single_segment_abi", 0) != 1
            or not hasattr(torch.ops.rpu, "qwen3_5_wall_action_loop")
            or not hasattr(torch.ops.rpu, "qwen3_5_set_wall_action_prefix_bucket")):
        raise RuntimeError(
            "FP16 Wall Action loop requires a rebuilt Python/native RhinoForge package "
            "with Wall Action prefix bucketing and single-segment Graph support"
        )
    for name, default, minimum, maximum in (
        ("LKN_MAX_BATCH_ENTRIES", 65536, 65536, 4194304),
        ("LKN_KD_BUF_MB", 8, 8, 256),
        ("LKN_INSTR_BUF_MB", 64, 64, 1024),
    ):
        value = os.environ.get(name, str(default))
        if (not value.isascii() or not value.isdecimal()
                or not minimum <= int(value) <= maximum):
            raise RuntimeError(
                f"Wall FP16 loop requires cold {name} in [{minimum},{maximum}]; "
                "use run_wall_qwen35_openloop.sh or set it before starting Python"
            )


def _finite_half(value: torch.Tensor, name: str) -> torch.Tensor:
    result = value.detach().to(device="cpu", dtype=torch.float16).contiguous()
    if not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} is not finite after FP16 conversion")
    return result


def _wall_fp16_io_weights(expert) -> tuple[torch.Tensor, torch.Tensor]:
    processor = _processor(expert)
    wi = _cpu_fp32(processor.w1.weight, "Wall w1", (1024, 52))
    wo = _cpu_fp32(processor.action_proj_back.weight, "Wall projection", (26, 1024))
    if processor.w1.bias is not None or processor.action_proj_back.bias is not None:
        raise ValueError("Wall FP16 I/O profile requires bias-free projections")
    # Preserve the /4 as a separate post-GEMM FP16 operation, not a weight fold.
    return (F.pad(_finite_half(wi, "Wall w1"), (0, 12)),
            F.pad(_finite_half(wo, "Wall projection"), (0, 0, 0, 38)))


def _pack_wall_fp16_loop_inputs(action, dof_mask, padding_velocity):
    action, mask, padding_velocity = validate_wall_qwen35_euler_profile(
        action, dof_mask, padding_velocity
    )
    action = _finite_half(action, "Wall action")
    mask = mask.half()
    padding = _finite_half(padding_velocity, "Wall padding velocity")
    packed = F.pad(torch.cat((action, mask.expand_as(action)), dim=-1), (0, 12))
    keep = F.pad(mask.reshape(26), (0, 38))
    padding = F.pad(padding * (1 - mask), (0, 38))
    return packed.contiguous(), keep.contiguous(), padding.contiguous()


def _wall_fp16_times() -> tuple[torch.Tensor, float]:
    times = torch.linspace(0.0, 1.0, 11, dtype=torch.float32) * 0.999
    steps = (times[1:] - times[:-1]).half()
    if not bool(steps.eq(steps[0]).all()) or float(steps[0]) <= 0:
        raise RuntimeError("Wall FP16 Euler schedule is not a uniform positive dt")
    return times, float(steps[0])


def _wall_loop_modulation(expert):
    state = expert._rpu_qwen3_5
    if state.action_loop_modulation is None:
        times, _ = _wall_fp16_times()
        rows = [_adaptive_modulation_cpu(
            expert, wall_qwen35_host_time_condition(expert, times[i:i + 1])
        ) for i in range(ACTION_NUM_STEPS)]
        state.action_loop_modulation = _finite_half(
            torch.stack(rows), "Wall time/Ada modulation"
        ).to("rpu")
    return state.action_loop_modulation


def run_wall_qwen35_action_loop(
    expert, *, action, dof_mask, padding_velocity, position_ids, prefix_cache,
    prefix_len: int,
) -> torch.Tensor:
    """Ten FP16 steps in one Graph or ten one-step calls; fresh CPU FP32 result.

    Time/Ada projections are precomputed in CPU FP32 once and uploaded in FP16.
    GEMMs use half inputs/weights, ACC32, and half outputs. Euler uses separate
    half mask/add/multiply/add operations, with no host step boundary.
    """
    if not getattr(expert, "_wall_qwen35_action_ready", False) or expert.training:
        raise RuntimeError("install an inference-only Wall action module first")
    state = expert._rpu_qwen3_5
    steps_per_graph = ACTION_NUM_STEPS if state.action_one_graph else 1
    if (isinstance(prefix_len, bool) or not isinstance(prefix_len, Integral)
            or not 1 <= prefix_len <= ACTION_MAX_PREFIX_LEN):
        raise ValueError("Wall action prefix_len must be an integer in [1,384]")
    positions = _canonical_action_positions(position_ids, prefix_len)
    bucket_len = _action_prefix_bucket(prefix_len) if state.action_one_graph else int(prefix_len)
    packed, keep, padding = _pack_wall_fp16_loop_inputs(action, dof_mask, padding_velocity)
    import rpu_backend

    signature = rpu_backend.graph.GraphSignature(
        op_id="rpu_wall_qwen35_action_fp16_loop",
        shapes=[1, 32, 64, bucket_len],
        dyn_dims=[steps_per_graph, 24, *_FULL_LAYERS], dtypes=[torch.float16],
    )
    cache = state.action_graph_cache
    cold = cache.lookup(signature) is None
    if cold:
        if cache.is_frozen():
            raise RuntimeError(f"Wall action GraphCache READY miss for prefix extent {bucket_len}")
        if not state.action_one_graph and cache.size():
            cache.clear()
    prefix_lens = _copy_physical_prefix(state, prefix_cache, int(prefix_len), bucket_len)
    if state.action_one_graph:
        # This setter uploads to a model-owned, per-bucket stable DDR slot.
        # It runs even on fast REPLAY (which skips the native layer builder).
        torch.ops.rpu.qwen3_5_set_wall_action_prefix_bucket(
            state.handle, int(prefix_len), bucket_len,
        )
    state.action_prefix_bucket = bucket_len
    _ensure_action_rope(state, positions)
    modulation = _wall_loop_modulation(expert)
    packed, keep, padding = (t.to("rpu") for t in (packed, keep, padding))
    _, dt = _wall_fp16_times()
    for step in range(0, ACTION_NUM_STEPS, steps_per_graph):
        # Keep the evolving x on RPU in both arms. The split arm refreshes the
        # modulation DMA view on each replay, without recomputing host math.
        step_modulation = modulation if state.action_one_graph else modulation[step].contiguous()
        output = torch.empty_like(packed)

        def forward():
            return torch.ops.rpu.qwen3_5_wall_action_loop(
                state.handle, packed, keep, padding, output, dt, step_modulation,
                state.action_k_caches, state.action_v_caches, prefix_lens, steps_per_graph,
            )

        if cold and step == 0:
            # Prime persistent buffers before BUILD. This is outside READY;
            # replay/capture reloads the original packed x, not the prime result.
            forward()
        with cache.capture(signature):
            forward()
        packed = output
    state.action_graph_signature = signature
    result = output.to(device="cpu", dtype=torch.float32)[..., :26].contiguous().clone()
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Wall FP16 loop returned non-finite actions")
    return result


def _canonical_action_positions(
    position_ids: torch.Tensor | None,
    prefix_len: int,
) -> torch.Tensor:
    """Return the reference action RoPE lanes as CPU int64 ``[3, 32]``.

    ``prefix_len`` is the physical KV-cache extent and is deliberately not a
    rotary position.  Multimodal compression makes the action token positions
    smaller than that extent (for example, frame 0 is 231..262 while its
    physical prefix contains 308 tokens).
    """

    if position_ids is None:
        raise ValueError(
            "Wall action requires checkpoint-derived multimodal position_ids"
        )
    if not isinstance(position_ids, torch.Tensor):
        raise TypeError("Wall action position_ids must be a tensor or None")
    value = position_ids.detach().to(device="cpu")
    if value.ndim == 3:
        if value.shape[1] != 1:
            raise ValueError("Wall action position_ids batch dimension must be one")
        value = value[:, 0]
    elif value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[0] not in (1, 3, 4):
        raise ValueError("Wall action position_ids must have 1, 3, or 4 lanes")
    if value.shape[1] != ACTION_HORIZON:
        raise ValueError("Wall action position_ids must contain exactly 32 positions")
    if value.dtype not in {
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    }:
        raise ValueError("Wall action position_ids must use an integer dtype")
    lanes = value[1:] if value.shape[0] == 4 else value
    if lanes.shape[0] == 1:
        lanes = lanes.expand(3, -1)
    if lanes.shape[0] > 1 and not bool(lanes.eq(lanes[0:1]).all()):
        raise ValueError("Wall action M-RoPE lanes must coincide for text actions")
    if not bool((lanes[:, 1:] - lanes[:, :-1]).eq(1).all()):
        raise ValueError("Wall action position_ids must be contiguous")
    if prefix_len < 1:
        raise ValueError("Wall action prefix_len must be positive")
    if int(lanes[0, 0]) < 0 or int(lanes[0, 0]) > prefix_len:
        raise ValueError(
            "Wall action position_ids must start within the physical prefix"
        )
    return lanes.to(dtype=torch.int64).contiguous()


def _build_action_rope_tables(
    position_ids: torch.Tensor,
    *,
    device: str = "rpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    from rpu_backend.adapters.qwen3_5.text import (
        _build_interleaved_mrope_cos_sin,
    )

    return _build_interleaved_mrope_cos_sin(
        position_ids,
        _ROTARY_DIM,
        _ROPE_THETA,
        (11, 11, 10),
        dtype=torch.float16,
        device=device,
    )


def _ensure_action_rope(state, position_ids: torch.Tensor) -> None:
    key = tuple(int(value) for value in position_ids.reshape(-1).tolist())
    cached = state.action_rope_cache
    if cached is not None and cached[0] == key:
        return
    cos, sin = _build_action_rope_tables(position_ids)
    if tuple(cos.shape) != (ACTION_HORIZON, _ROTARY_DIM // 2):
        raise RuntimeError("Wall compact action RoPE shape drift")
    # The native setter copies into address-stable DDR when the shape is
    # unchanged.  That is the documented mutable input path for retained
    # action Graph replay; no position value is frozen in the signature.
    torch.ops.rpu.qwen3_5_set_prefill_rope(state.handle, cos, sin)
    state.action_rope_cache = (key, cos, sin)


def _action_prefix_bucket(prefix_len: int) -> int:
    if (isinstance(prefix_len, bool) or not isinstance(prefix_len, Integral)
            or not 1 <= prefix_len <= ACTION_MAX_PREFIX_LEN):
        raise ValueError("Wall action prefix_len must be an integer in [1,384]")
    return ((int(prefix_len) + 63) // 64) * 64


def _zero_action_prefix_padding(k, v, prefix_len: int, bucket_len: int) -> None:
    """Clear only the gap, including a partial swizzled block (K/V differ)."""
    if prefix_len == bucket_len:
        return
    block, tail = divmod(prefix_len, 16)
    if tail:
        k[:, block:block + 1, :, :, :, tail:, :].zero_()
        v[:, block:block + 1, :, :, :, :, tail:].zero_()
        block += 1
    if block < bucket_len // 16:
        k[:, block:bucket_len // 16].zero_()
        v[:, block:bucket_len // 16].zero_()


def _copy_physical_prefix(
    state, prefix_cache, prefix_len: int, bucket_len: int | None = None,
) -> list[int]:
    if bucket_len is None:
        bucket_len = prefix_len
    if bucket_len not in (prefix_len, _action_prefix_bucket(prefix_len)):
        raise ValueError("Wall action physical prefix must be exact or its 64-row bucket")
    if prefix_cache is None:
        raise TypeError("Wall action requires a base physical prefix cache")
    position = getattr(prefix_cache, "position", None)
    if isinstance(position, bool) or not isinstance(position, Integral):
        raise TypeError("Wall base prefix cache has no integer position")
    if int(position) != prefix_len:
        raise ValueError(
            f"Wall base cache position {position} != prefix_len {prefix_len}"
        )
    source_k = getattr(prefix_cache, "k_caches", None)
    source_v = getattr(prefix_cache, "v_caches", None)
    if source_k is None or source_v is None:
        physical = getattr(prefix_cache, "_rpu_qwen3_5_cache", None)
        source_k = getattr(physical, "k_caches", None)
        source_v = getattr(physical, "v_caches", None)
    if source_k is None or source_v is None:
        raise TypeError("Wall base prefix cache does not expose physical K/V")
    if len(source_k) != ACTION_NUM_LAYERS or len(source_v) != ACTION_NUM_LAYERS:
        raise ValueError("Wall base physical cache must contain 24 layers")

    blocks = (prefix_len + 15) // 16
    prefix_lens = [0] * ACTION_NUM_LAYERS
    for index in _FULL_LAYERS:
        src_k = source_k[index]
        src_v = source_v[index]
        dst_k = state.action_k_caches[index]
        dst_v = state.action_v_caches[index]
        expected_tail = (1, 16, 8, 16, 16)
        for name, source, destination in (
            ("K", src_k, dst_k),
            ("V", src_v, dst_v),
        ):
            if (
                source.device.type != "rpu"
                or destination.device.type != "rpu"
                or source.dtype is not torch.float16
                or destination.dtype is not torch.float16
                or source.ndim != 7
                or destination.ndim != 7
                or tuple(source.shape[2:]) != expected_tail
                or tuple(destination.shape[2:]) != expected_tail
                or source.shape[0] != 1
                or destination.shape[0] != 1
                or source.shape[1] < blocks
                or destination.shape[1] * 16 < bucket_len + ACTION_HORIZON
                or source.data_ptr() == destination.data_ptr()
            ):
                raise ValueError(
                    f"Wall physical {name} cache layer {index} has invalid topology"
                )
        # Always refresh the six physical prefixes.  The base cache is reused
        # in place and intentionally has no generation marker.
        dst_k[:, :blocks].copy_(src_k[:, :blocks])
        dst_v[:, :blocks].copy_(src_v[:, :blocks])
        _zero_action_prefix_padding(dst_k, dst_v, prefix_len, bucket_len)
        prefix_lens[index] = bucket_len
    state.action_prefix_lens = prefix_lens
    state.action_prefix_len = prefix_len
    return prefix_lens


def run_wall_qwen35_action(
    expert: WallQwen35ActionModule,
    *,
    inputs_embeds: torch.Tensor | None = None,
    action_embeds: torch.Tensor | None = None,
    time_cond: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    prefix_cache=None,
    kv_cache=None,
    prefix_len: int,
    dof_mask: torch.Tensor,
) -> torch.Tensor:
    """Run one Wall decoder step and return a fresh CPU FP32 hidden tensor.

    ``inputs_embeds``/``action_embeds`` is the *raw* host-FP32 ``w1`` output.
    This function owns its sole ``/4`` and the FP16 upload.  The base cache is
    read-only; its six full-layer physical prefixes are copied on every call to
    independent, address-stable action storage.  Same prefix shape replays the
    single Graph entry while input and modulation addresses are refreshed by
    the established Qwen3.5 action-forward path.
    """

    if not isinstance(expert, WallQwen35ActionModule) or not getattr(
        expert, "_wall_qwen35_action_ready", False
    ):
        raise RuntimeError("patch a loaded Wall action module before inference")
    if expert.training:
        raise NotImplementedError("Wall action decoder is inference-only")
    if inputs_embeds is None:
        inputs_embeds = action_embeds
    elif action_embeds is not None and action_embeds is not inputs_embeds:
        if not torch.equal(inputs_embeds, action_embeds):
            raise ValueError("inputs_embeds and action_embeds disagree")
    if inputs_embeds is None:
        raise TypeError("Wall action requires inputs_embeds")
    if prefix_cache is None:
        prefix_cache = kv_cache
    elif kv_cache is not None and kv_cache is not prefix_cache:
        raise ValueError("prefix_cache and kv_cache must refer to the same object")
    if isinstance(prefix_len, bool) or not isinstance(prefix_len, Integral):
        raise ValueError("Wall action prefix_len must be an integer")
    prefix_len = int(prefix_len)
    if not 1 <= prefix_len <= ACTION_MAX_PREFIX_LEN:
        raise ValueError(
            f"Wall action prefix_len must be in [1, {ACTION_MAX_PREFIX_LEN}]"
        )

    hidden_cpu = _cpu_fp32(
        inputs_embeds,
        "Wall raw w1 output",
        (ACTION_BATCH_SIZE, ACTION_HORIZON, ACTION_HIDDEN_SIZE),
    )
    time_cond = _cpu_fp32(
        time_cond, "Wall time_cond", (ACTION_BATCH_SIZE, ACTION_HIDDEN_SIZE)
    )
    _dof_mask(dof_mask)
    action_position_ids = _canonical_action_positions(position_ids, prefix_len)

    import rpu_backend as rpu_backend

    state = expert._rpu_qwen3_5
    signature = rpu_backend.graph.GraphSignature(
        op_id="rpu_wall_qwen35_action_decoder",
        shapes=[ACTION_BATCH_SIZE, ACTION_HORIZON, ACTION_HIDDEN_SIZE, prefix_len],
        dyn_dims=[ACTION_NUM_LAYERS, *_FULL_LAYERS],
        dtypes=[torch.float16],
    )
    cold = state.action_graph_cache.lookup(signature) is None
    if cold:
        # Reject a frozen miss before prefix copies, RoPE updates, uploads, or
        # the uncaptured priming forward. READY is strictly lookup-only.
        if state.action_graph_cache.is_frozen():
            raise RuntimeError(
                "Wall action GraphCache READY miss for exact real prefix "
                f"{prefix_len}; call begin_warmup() before changing prefix"
            )
        if state.action_graph_cache.size() != 0:
            state.action_graph_cache.clear()

    prefix_lens = _copy_physical_prefix(state, prefix_cache, prefix_len)
    _ensure_action_rope(state, action_position_ids)
    modulation = _adaptive_modulation(expert, time_cond)
    # Sole action-input scale boundary: host FP32 division, then one cast/upload.
    hidden_rpu = (hidden_cpu / 4.0).to(
        device="rpu", dtype=torch.float16
    ).contiguous()
    if cold:
        # Prime persistent SPM state before BUILD.  The action suffix is then
        # deterministically overwritten by the captured call.
        torch.ops.rpu.qwen3_5_action_forward(
            state.handle,
            hidden_rpu,
            modulation,
            state.action_k_caches,
            state.action_v_caches,
            prefix_lens,
        )
    with state.action_graph_cache.capture(signature):
        hidden = torch.ops.rpu.qwen3_5_action_forward(
            state.handle,
            hidden_rpu,
            modulation,
            state.action_k_caches,
            state.action_v_caches,
            prefix_lens,
        )
    state.action_graph_signature = signature
    result = hidden.to(device="cpu", dtype=torch.float32).contiguous().clone()
    if tuple(result.shape) != (
        ACTION_BATCH_SIZE,
        ACTION_HORIZON,
        ACTION_HIDDEN_SIZE,
    ):
        raise RuntimeError(f"Wall action decoder returned shape {tuple(result.shape)}")
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Wall action decoder returned non-finite hidden states")
    return result


__all__ = [
    "ACTION_DIM",
    "ACTION_HORIZON",
    "ACTION_MAX_SEQ_LEN",
    "ACTION_NUM_STEPS",
    "WallActionProcessor",
    "WallHostLinear",
    "WallNormalizerBank",
    "WallQwen35ActionModule",
    "load_wall_qwen35_action_module",
    "patch_wall_qwen35_action_for_rpu",
    "run_wall_qwen35_action",
    "run_wall_qwen35_action_loop",
    "validate_wall_qwen35_euler_profile",
    "wall_qwen35_host_action_input",
    "wall_qwen35_host_euler_step",
    "wall_qwen35_host_time_condition",
    "wall_qwen35_host_velocity",
]
