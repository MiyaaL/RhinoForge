"""End-to-end orchestration for the exact Wall Qwen3.5 checkpoint.

The base Qwen3.5 vision/text path uses the normal RhinoForge adapter.  The
action expert remains a separate Qwen3.5 native handle; its input projection,
time conditioning, output projection, Euler integration, and normalization run
in host FP32 for the intentionally simple first port.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from rpu_backend.adapters.qwen3_5 import Qwen3_5Adapter
from rpu_backend.api.qwen3_5_cache import Qwen3_5Cache

from .preprocessing import (
    ACTION_HORIZON,
    CHECKPOINT_VOCAB_SIZE,
    DEFAULT_CAMERAS,
    DEFAULT_ROBOT_ID,
    MAX_SEQ_LENGTH,
    STATE_DIM,
    SPECIAL_TOKENS,
    WALL_PREFIX_BUCKETS,
    WallQwen35PreparedInput,
    prepare_wall_qwen35_input,
)


_INSTALL_LOCK = threading.Lock()
_VISION_OPT_IN = "QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED"
_VISION_GRAPH_MAX_ENTRIES = "QWEN3_5_VISION_GRAPH_MAX_ENTRIES"
_COEXIST_PERSISTENT_ENV = "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN"
_TRUE_ENV_VALUES = frozenset({"1", "true", "True", "on"})

def _profile_scope(name: str):
    """Return a stable model-stage range for Torch profiler attribution.

    PyTorch's private ``_profiler_enabled`` predicate does not report Kineto's
    CPU + PrivateUse1 session reliably on this release. These four coarse model
    scopes are therefore always constructed, matching GraphCache's public range
    behavior and avoiding a false-negative trace with only generic ATen names.
    """

    return torch.profiler.record_function(name)


def _wall_modules():
    # Imports are delayed so importing rpu_backend remains cheap and so the
    # checkpoint/action modules can keep board-only dependencies private.
    from . import action as action_module
    from . import checkpoint as checkpoint_module

    return checkpoint_module, action_module


def _required(module, name):
    value = getattr(module, name, None)
    if not callable(value):
        raise RuntimeError(
            f"wall_qwen35 integration requires {module.__name__}.{name}()"
        )
    return value


def _text_config(config):
    return getattr(config, "text_config", config)


def _load_base_model(checkpoint: Path, checkpoint_module, *, manifest):
    """Instantiate the standard HF class and stream only its base tensors."""

    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5ForConditionalGeneration,
    )

    config = AutoConfig.from_pretrained(checkpoint, local_files_only=True)
    tc = _text_config(config)
    tc.vocab_size = CHECKPOINT_VOCAB_SIZE
    tc._attn_implementation = "eager"
    config.vocab_size = CHECKPOINT_VOCAB_SIZE
    config._attn_implementation = "eager"
    stream = _required(checkpoint_module, "stream_load_wall_qwen35_base_model")

    # The exact helper owns meta construction and streams only admitted base
    # tensors. Never fall back to from_pretrained or mask a helper TypeError.
    model = stream(
        None,
        checkpoint,
        config=config,
        model_class=Qwen3_5ForConditionalGeneration,
        dtype=torch.float16,
        manifest=manifest,
    )
    if model is None:
        raise RuntimeError("base-model streaming helper returned no model")
    return model


def _load_processor(checkpoint: Path, checkpoint_module):
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        checkpoint, local_files_only=True, use_fast=True,
    )
    tokenizer = processor.tokenizer
    extender = _required(checkpoint_module, "extend_wall_qwen35_tokenizer")
    extender(tokenizer, target_vocab=CHECKPOINT_VOCAB_SIZE)
    if len(tokenizer) != CHECKPOINT_VOCAB_SIZE:
        raise ValueError(
            f"Wall tokenizer width {len(tokenizer)} != {CHECKPOINT_VOCAB_SIZE}"
        )
    expected_ids = tuple(range(248077, 248083))
    actual_ids = tuple(tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS)
    if actual_ids != expected_ids:
        raise ValueError(
            f"Wall special-token IDs {actual_ids} do not match {expected_ids}"
        )
    tokenizer.padding_side = "right"
    return processor


def _load_norm_vectors(checkpoint: Path, kind: str, dataset_key: str):
    if dataset_key != "x2_normal":
        raise ValueError(
            "initial Wall Qwen3.5 profile requires dataset_key='x2_normal'"
        )
    payload = torch.load(
        checkpoint / f"normalizer_{kind}.pth",
        map_location="cpu", weights_only=True,
    )
    try:
        minimum = payload[f"min.{dataset_key}"].detach().float().contiguous()
        delta = payload[f"delta.{dataset_key}"].detach().float().contiguous()
    except KeyError as error:
        raise KeyError(
            f"normalizer_{kind}.pth has no dataset key {dataset_key!r}"
        ) from error
    if tuple(minimum.shape) != (STATE_DIM,) or tuple(delta.shape) != (STATE_DIM,):
        raise ValueError(f"{kind} normalizer must contain 26 values")
    return minimum, delta


class WallQwen35Runtime:
    """One-process owner of the base and action Qwen3.5 RPU handles."""

    def __init__(
        self,
        checkpoint: str | Path,
        *,
        dataset_key: str = "x2_normal",
        robot_id: str | int = DEFAULT_ROBOT_ID,
        camera_names: Sequence[str] = DEFAULT_CAMERAS,
        max_seq_len: int = MAX_SEQ_LENGTH,
        allow_numeric_blocked_vision: bool = False,
        profile: Any | None = None,
    ):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.dataset_key = str(dataset_key)
        if isinstance(robot_id, bool) or str(robot_id) != DEFAULT_ROBOT_ID:
            raise ValueError(
                "initial Wall Qwen3.5 profile requires robot_id='10070'"
            )
        self.robot_id = str(robot_id)
        self.camera_names = tuple(camera_names)
        self.max_seq_len = int(max_seq_len)
        if self.camera_names != DEFAULT_CAMERAS:
            raise ValueError(f"Wall Qwen3.5 requires cameras {DEFAULT_CAMERAS!r}")
        if self.max_seq_len != MAX_SEQ_LENGTH:
            raise ValueError("initial Wall Qwen3.5 profile requires max_seq_len=780")
        if not isinstance(allow_numeric_blocked_vision, bool):
            raise ValueError("allow_numeric_blocked_vision must be bool")
        self.allow_numeric_blocked_vision = allow_numeric_blocked_vision
        self._closed = False
        self._installed = False
        self._owned_vision_env = False
        self._owned_vision_graph_entries_env = False
        self._owned_coexist_persistent_env = False
        self._base_adapter = None
        self.base_model = None
        self.base_cache = None
        self.action_expert = None
        self._manifest = profile
        self._processor = None

        checkpoint_module, _ = _wall_modules()
        if self._manifest is None:
            self._manifest = _required(
                checkpoint_module, "preflight_wall_qwen35_checkpoint"
            )(self.checkpoint)
        else:
            self._manifest = _required(
                checkpoint_module, "_admit_manifest"
            )(self.checkpoint, self._manifest)
        self._state_min, self._state_delta = _load_norm_vectors(
            self.checkpoint, "propri", self.dataset_key
        )
        self._action_min, self._action_delta = _load_norm_vectors(
            self.checkpoint, "action", self.dataset_key
        )

    @classmethod
    def from_checkpoint(cls, checkpoint, **kwargs):
        return cls(checkpoint, **kwargs)

    def install(self):
        if self._closed:
            raise RuntimeError("WallQwen35Runtime is closed")
        if self._installed:
            return self
        if not self.allow_numeric_blocked_vision:
            raise NotImplementedError(
                "Wall Qwen3.5 image inference uses RhinoForge's numeric-blocked "
                "Qwen3.5 vision path; pass allow_numeric_blocked_vision=True for "
                "controlled evaluation"
            )
        if not _INSTALL_LOCK.acquire(blocking=False):
            raise RuntimeError("another Wall Qwen3.5 installation is in progress")
        checkpoint_module, action_module = _wall_modules()
        try:
            # Wall alternates three live FusedModelBase handles (vision, base
            # text, and action).  The native switch is sampled once, so bind it
            # before the first handle is installed.  Without it, each handle's
            # first allocation advances the process-wide persistent generation
            # and forces the other two back through allocation/Graph BUILD.
            coexist_value = os.environ.get(_COEXIST_PERSISTENT_ENV)
            if coexist_value is None:
                os.environ[_COEXIST_PERSISTENT_ENV] = "1"
                self._owned_coexist_persistent_env = True
            elif coexist_value not in _TRUE_ENV_VALUES:
                raise RuntimeError(
                    "Wall Qwen3.5 requires the cold runtime setting "
                    f"{_COEXIST_PERSISTENT_ENV}=1; start a fresh process "
                    f"instead of using {coexist_value!r}"
                )
            if os.environ.get(_VISION_OPT_IN) != "1":
                os.environ[_VISION_OPT_IN] = "1"
                self._owned_vision_env = True
            # Wall's exact prefix has two distinct image geometries (face and
            # wrist).  The shared Qwen3.5 installer defaults to one retained
            # Vision graph for generic workloads; that setting would evict one
            # geometry on every request and make a repeated profile call BUILD
            # again.  Require a bounded two-entry cache for this controlled
            # profile, while honoring a larger caller-provided budget.
            vision_entries = os.environ.get(_VISION_GRAPH_MAX_ENTRIES)
            if vision_entries is None:
                os.environ[_VISION_GRAPH_MAX_ENTRIES] = "2"
                self._owned_vision_graph_entries_env = True
            else:
                try:
                    parsed_entries = int(vision_entries)
                except (TypeError, ValueError) as exc:
                    raise RuntimeError(
                        "Wall Qwen3.5 requires "
                        f"{_VISION_GRAPH_MAX_ENTRIES} to be a positive integer"
                    ) from exc
                if parsed_entries < 2:
                    raise RuntimeError(
                        "Wall Qwen3.5 requires at least two retained Vision "
                        "graph entries; unset the variable or set it to >=2"
                    )
            self._processor = _load_processor(self.checkpoint, checkpoint_module)
            self.base_model = _load_base_model(
                self.checkpoint, checkpoint_module, manifest=self._manifest
            )
            self.base_model.config._attn_implementation = "eager"
            _text_config(self.base_model.config)._attn_implementation = "eager"
            self.base_model.eval()
            # This VLA uses the base decoder only to populate six physical K/V
            # prefixes for the action expert.  It never consumes language
            # logits, and its 256277-row tied head cannot satisfy the generic
            # RPU col-swizzle multiple.  Keep that exception explicit and local.
            self.base_model._wall_qwen35_cache_only_no_lm_head = True
            self._base_adapter = Qwen3_5Adapter(self.base_model)
            self._base_adapter.to_rpu(max_seq_len=self.max_seq_len)
            # The generic adapter deliberately keeps variable-length prefill as
            # a one-shot graph.  Wall has a finite <=384 prefix envelope, so
            # opt into a retained bucket cache after the text handle is fully
            # installed.  This is process-local policy state, not a public
            # Qwen3.5 execution option.
            import rpu_backend as _rb

            text_owner = getattr(
                getattr(self.base_model, "model", None),
                "language_model",
                None,
            )
            text_state = getattr(text_owner, "_rpu_qwen3_5", None)
            if text_state is None:
                raise RuntimeError(
                    "Wall Qwen3.5 base install did not publish text runtime state"
                )
            text_state.prefill_graph_cache = _rb.graph.GraphCache(
                max_entries=len(WALL_PREFIX_BUCKETS)
            )
            text_state.prefill_graph_sig = None
            text_state.prefill_bucket_sizes = WALL_PREFIX_BUCKETS
            text_state.prefill_graph_op_id = "rpu_wall_qwen35_prefill"
            # Enable exact same-shape M-RoPE table reuse where the prompt
            # position tensor repeats; tables are still refreshed outside graph
            # capture whenever a request has different positions.
            text_state.prefill_rope_cache = None
            self.base_cache = Qwen3_5Cache.from_config(
                _text_config(self.base_model.config),
                max_seq_len=self.max_seq_len,
            )

            load_action = _required(
                action_module, "load_wall_qwen35_action_module"
            )
            self.action_expert = load_action(
                self.checkpoint,
                manifest=self._manifest,
                action_min=self._action_min,
                action_delta=self._action_delta,
            )
            patch_action = _required(
                action_module, "patch_wall_qwen35_action_for_rpu"
            )
            patched = patch_action(
                self.action_expert,
                max_seq_len=self.max_seq_len,
            )
            if patched is not None:
                self.action_expert = patched
            action_state = getattr(self.action_expert, "_rpu_qwen3_5", None)
            if action_state is None:
                raise RuntimeError(
                    "Wall Qwen3.5 action install did not publish runtime state"
                )
            action_state.action_prefix_bucket_sizes = WALL_PREFIX_BUCKETS
            self._installed = True
            return self
        except BaseException:
            self.close()
            raise
        finally:
            _INSTALL_LOCK.release()

    def to(self, device):
        if str(device) not in {"rpu", "rpu:0"}:
            raise ValueError("WallQwen35Runtime supports only device='rpu'")
        return self.install()

    def _prepare(self, *, images, instruction, state, agent_pos_mask, dof_mask):
        with _profile_scope("wall_qwen35_preprocess"):
            return prepare_wall_qwen35_input(
                self._processor,
                self.base_model,
                images=images,
                instruction=instruction,
                state=state,
                state_min=self._state_min,
                state_delta=self._state_delta,
                dataset_key=self.dataset_key,
                robot_id=self.robot_id,
                agent_pos_mask=agent_pos_mask,
                dof_mask=dof_mask,
                camera_names=self.camera_names,
                max_seq_length=self.max_seq_len,
            )

    def _build_prefix(self, prepared: WallQwen35PreparedInput):
        # Stamp before reset/prefill so a failed or partial rebuild cannot
        # accidentally reuse the previous request's action-owned prefix.
        action_state = getattr(self.action_expert, "_rpu_qwen3_5", None)
        if action_state is not None:
            action_state.action_prefix_generation = (
                int(getattr(action_state, "action_prefix_generation", 0)) + 1
            )
        self.base_cache.reset()
        # Qwen3.5 vision STEP0 accepts host folded patches and performs the
        # explicit CPU/RPU handoff itself.  Keep this pointer off RPU before
        # entering the lazy vision installer/captured graph.
        pixel_values = prepared.pixel_values.detach().to(
            device="cpu", dtype=torch.float16
        ).contiguous()
        with torch.inference_mode(), _profile_scope(
            "wall_qwen35_vision_text_prefill"
        ):
            output = self.base_model(
                input_ids=prepared.prefix_input_ids,
                attention_mask=prepared.prefix_attention_mask,
                position_ids=prepared.prefix_position_ids,
                past_key_values=self.base_cache,
                pixel_values=pixel_values,
                image_grid_thw=prepared.image_grid_thw,
                mm_token_type_ids=prepared.prefix_mm_token_type_ids,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
        logits = getattr(output, "logits", None)
        if not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 1, 0):
            raise RuntimeError(
                "Wall base decoder must use the explicit cache-only/no-lm-head "
                f"contract, got logits shape {getattr(logits, 'shape', None)}"
            )
        if int(self.base_cache.position) != prepared.prefix_length:
            raise RuntimeError(
                "base prefix cache position drift: "
                f"{self.base_cache.position} != {prepared.prefix_length}"
            )
        return self.base_cache

    def _run_action_layer(self, *, action_embed, time_cond, prepared, prefix_cache):
        _, action_module = _wall_modules()
        run = _required(action_module, "run_wall_qwen35_action")
        # The action helper owns the six-layer physical cache copy.  Supplying
        # both the cache and exact prefix length keeps that ownership explicit.
        with _profile_scope("wall_qwen35_action_decoder"):
            return run(
                self.action_expert,
                action_embeds=action_embed,
                time_cond=time_cond,
                position_ids=prepared.action_position_ids,
                prefix_cache=prefix_cache,
                prefix_len=prepared.prefix_length,
                dof_mask=prepared.dof_mask,
            )

    def predict_action_chunk(
        self,
        *,
        images: Mapping[str, Any],
        instruction: str | Mapping[str, Any],
        state: Any | None = None,
        proprioception: Any | None = None,
        agent_pos_mask: Any | None = None,
        dof_mask: Any | None = None,
        noise_seed: int | None = None,
        initial_noise: Any | None = None,
    ) -> dict[str, Any]:
        if not self._installed:
            raise RuntimeError("call runtime.to('rpu') before inference")
        if self._closed:
            raise RuntimeError("WallQwen35Runtime is closed")
        if noise_seed is not None and (
            isinstance(noise_seed, bool) or not isinstance(noise_seed, int)
        ):
            raise ValueError("noise_seed must be an integer or None")
        if initial_noise is not None and noise_seed is not None:
            raise ValueError("provide either noise_seed or initial_noise, not both")
        if (state is None) == (proprioception is None):
            raise ValueError(
                "provide exactly one of state or proprioception"
            )
        state_value = state if state is not None else proprioception
        prepared = self._prepare(
            images=images,
            instruction=instruction,
            state=state_value,
            agent_pos_mask=agent_pos_mask,
            dof_mask=dof_mask,
        )
        prefix_cache = self._build_prefix(prepared)
        _, action_module = _wall_modules()
        if initial_noise is None:
            resolved_noise_seed = 0 if noise_seed is None else noise_seed
            generator = torch.Generator(device="cpu").manual_seed(
                resolved_noise_seed
            )
            action = torch.randn(
                (1, ACTION_HORIZON, STATE_DIM),
                generator=generator, dtype=torch.float32,
            )
            noise_source = "cpu_seed"
        else:
            try:
                action = torch.as_tensor(initial_noise, dtype=torch.float32)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError("initial_noise must be tensor-like") from exc
            if action.device.type != "cpu":
                raise ValueError("initial_noise must be on CPU")
            if tuple(action.shape) == (ACTION_HORIZON, STATE_DIM):
                action = action.unsqueeze(0)
            if tuple(action.shape) != (1, ACTION_HORIZON, STATE_DIM):
                raise ValueError(
                    "initial_noise must have shape [32,26] or [1,32,26], "
                    f"got {tuple(action.shape)}"
                )
            if not bool(torch.isfinite(action).all()):
                raise ValueError("initial_noise must contain only finite values")
            action = action.contiguous().clone()
            resolved_noise_seed = None
            noise_source = "explicit"
        noise_snapshot = action.clone()
        padding_action = (
            ((-self._action_min) / self._action_delta) * 2.0 - 1.0
        ).clamp(-1.0, 1.0).view(1, 1, STATE_DIM)
        padding_velocity = padding_action - noise_snapshot
        times = torch.linspace(0.0, 1.0, 11, dtype=torch.float32) * 0.999
        with torch.inference_mode(), _profile_scope(
            "wall_qwen35_action_denoise_loop"
        ):
            for index in range(10):
                action_embed = action_module.wall_qwen35_host_action_input(
                    self.action_expert, action, prepared.dof_mask
                )
                time_cond = action_module.wall_qwen35_host_time_condition(
                    self.action_expert, times[index:index + 1]
                )
                hidden = self._run_action_layer(
                    action_embed=action_embed,
                    time_cond=time_cond,
                    prepared=prepared,
                    prefix_cache=prefix_cache,
                )
                velocity = action_module.wall_qwen35_host_velocity(
                    self.action_expert, hidden
                )
                if tuple(velocity.shape) != (1, ACTION_HORIZON, STATE_DIM):
                    raise RuntimeError(
                        f"action velocity shape {tuple(velocity.shape)} is invalid"
                    )
                action = action_module.wall_qwen35_host_euler_step(
                    action,
                    velocity,
                    prepared.dof_mask,
                    padding_velocity,
                    delta_t=float(times[index + 1] - times[index]),
                )
        result = (action + 1.0) * 0.5
        result = result * self._action_delta.view(1, 1, -1)
        result = result + self._action_min.view(1, 1, -1)
        actions = result.to(
            device="cpu", dtype=torch.float32
        ).contiguous().clone()
        base_owner = getattr(
            getattr(self.base_model, "model", None), "language_model", None
        )
        base_plan = getattr(base_owner, "_rpu_last_execution_plan", None)
        plan_extra = {}
        if isinstance(base_plan, dict):
            for key in ("bucket_len", "execution_len", "padding_rows", "chunk_size"):
                if key in base_plan:
                    plan_extra[key] = base_plan[key]
        return {
            "actions": actions,
            "actions_norm": action.to(
                device="cpu", dtype=torch.float32
            ).contiguous().clone(),
            "prefix_length": prepared.prefix_length,
            "extra": {
                "noise_seed": resolved_noise_seed,
                "noise_source": noise_source,
                "prefix_bucket": plan_extra.pop("bucket_len", None),
                "base_prefill": plan_extra,
            },
        }

    def close(self):
        if self._closed:
            return
        base_model = self.base_model

        # Vision is installed lazily by the first prefix forward.  Retire it
        # explicitly before dropping the model; otherwise its Graph queue and
        # native handle survive until cyclic GC happens to collect the patched
        # HF model.
        if base_model is not None:
            fusion = getattr(base_model, "model", None)
            vision = getattr(fusion, "visual", None)
            if vision is not None and any(
                hasattr(vision, name)
                for name in (
                    "_rpu_vision_handle",
                    "_rpu_vision_installing_handle",
                    "_rpu_vision_graph_cache",
                )
            ):
                from rpu_backend.adapters.qwen3_5.vision import (
                    _rollback_qwen3_5_vision_install,
                )

                _rollback_qwen3_5_vision_install(vision)

        # Both synthetic action and base text handles use the shared Qwen3.5
        # state shape.  Clear every retained GraphCache before invoking the
        # handle finalizer.  Cleanup errors deliberately propagate: releasing
        # process ownership after an unproven teardown could admit a second
        # policy over leaked native state.
        for owner, cache_names in (
            (self.action_expert, ("action_graph_cache", "graph_cache")),
            (
                getattr(
                    getattr(base_model, "model", None),
                    "language_model",
                    None,
                ),
                (
                    "graph_cache",
                    "prefill_graph_cache",
                    "prefill_debug_graph_cache",
                ),
            ),
        ):
            state = getattr(owner, "_rpu_qwen3_5", None)
            if state is None:
                continue
            for cache_name in cache_names:
                cache = getattr(state, cache_name, None)
                if cache is not None:
                    cache.clear()
            finalizer = getattr(state, "handle_finalizer", None)
            if finalizer is not None and getattr(finalizer, "alive", False):
                finalizer()

        if base_model is not None:
            from rpu_backend.api.causal_lm import _release_live_instance

            _release_live_instance(base_model)

        self.base_cache = None
        self.action_expert = None
        self.base_model = None
        self._base_adapter = None
        self._processor = None
        self._installed = False
        self._closed = True
        if getattr(self, "_owned_vision_env", False):
            os.environ.pop(_VISION_OPT_IN, None)
            self._owned_vision_env = False
        if getattr(self, "_owned_vision_graph_entries_env", False):
            os.environ.pop(_VISION_GRAPH_MAX_ENTRIES, None)
            self._owned_vision_graph_entries_env = False
        if getattr(self, "_owned_coexist_persistent_env", False):
            os.environ.pop(_COEXIST_PERSISTENT_ENV, None)
            self._owned_coexist_persistent_env = False

    def __enter__(self):
        return self.install()

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = ["WallQwen35Runtime"]
