"""Checkpoint admission and CPU streaming for the exact Wall Qwen3.5 profile.

The training checkpoint is a single, local safetensors file containing the
ordinary Qwen3.5 vision/text weights, the Wall action expert, and host action
processor weights.  Admission is intentionally exact: file digests and the
complete safetensors key/dtype/shape manifest are checked before a tensor is
materialized or a model parameter is replaced.
"""

from __future__ import annotations

import hashlib
import json
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


CHECKPOINT_VOCAB_SIZE = 256_277
_BASE_TOKENIZER_SIZE = 248_077
_FLOW_SPECIAL_TOKENS = (
    "<|propri|>",
    "<|action|>",
    "<point>",
    "</point>",
    "<box>",
    "</box>",
)
_AUDIO_TAIL = (
    "<|audio_start|>",
    "<|audio_end|>",
    "<tts_pad>",
    "<tts_text_bos>",
    "<tts_text_eod>",
    "<tts_text_bos_single>",
    "<|audio_pad|>",
)
_RESERVED_COUNT = CHECKPOINT_VOCAB_SIZE - _BASE_TOKENIZER_SIZE - len(
    _FLOW_SPECIAL_TOKENS
)

_EXPECTED_FILE_SHA256 = {
    "model.safetensors": (
        "82f9d583bf33f713c941b1c3aac60a6be5ada9eb02cf8a04f9f21e61bc0e7b2c"
    ),
    "config.json": (
        "76aa51dd1cc450bf753c9790ffae668e21467964b4cb3dc8b0035853b6227c62"
    ),
    "config.yml": (
        "3e84029377a5878fe4a91fbcd74100cca67d1ed938f5e81e3b31fd84a7f86828"
    ),
    "tokenizer.json": (
        "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"
    ),
    "tokenizer_config.json": (
        "49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c"
    ),
    "preprocessor_config.json": (
        "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516"
    ),
    "vocab.json": (
        "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003"
    ),
    "normalizer_action.pth": (
        "6318b943be0186286de0b1c11e1c76158e604bf7513251c2ba9242f8a10dab5d"
    ),
    "normalizer_propri.pth": (
        "882a4a219618800fe8754a7aa1c4b9548e2085feea793679bbb2f79cb2e32691"
    ),
}

_EXPECTED_TENSOR_COUNT = 1_090
_EXPECTED_TENSOR_MANIFEST_SHA256 = (
    "31c3d1d02949ec38209f7274f684c0841b8d5a28d7ad49a58c7e983d0a5de14d"
)
_EXPECTED_ACTION_TENSOR_COUNT = 497
_EXPECTED_ACTION_MANIFEST_SHA256 = (
    "f15783b5d953e4af99bae4607bda6b20f2dbcd282d9df8b89182dca2bfdb0300"
)
_EXPECTED_BASE_TENSOR_COUNT = 593
_EXPECTED_BASE_MANIFEST_SHA256 = (
    "1f1232048e4772a46023e7a0302489e581ebf3913d2e5c26b9b28433bbc2bba9"
)
_ADMISSION_TOKEN = object()


@dataclass(frozen=True, slots=True)
class WallQwen35CheckpointManifest:
    """Immutable result of exact checkpoint admission."""

    root: Path
    file_sha256: tuple[tuple[str, str], ...]
    file_stats: tuple[tuple[str, int, int, int], ...]
    tensor_count: int
    tensor_manifest_sha256: str
    action_tensor_count: int
    action_manifest_sha256: str
    base_tensor_count: int
    base_manifest_sha256: str
    _admission_token: object = field(
        default=None, init=False, repr=False, compare=False
    )

    def digest_for(self, name: str) -> str:
        return dict(self.file_sha256)[name]

    def stat_for(self, name: str) -> tuple[int, int, int]:
        values = {
            key: (size, mtime_ns, inode)
            for key, size, mtime_ns, inode in self.file_stats
        }
        size, mtime_ns, inode = values[name]
        return int(size), int(mtime_ns), int(inode)


def _unsupported(message: str) -> BaseException:
    try:
        from rpu_backend.api.errors import UnsupportedModelError

        return UnsupportedModelError(message)
    except ImportError:
        return ValueError(message)


def _local_checkpoint_root(checkpoint: str | Path) -> Path:
    if not isinstance(checkpoint, (str, Path)):
        raise TypeError("Wall Qwen3.5 checkpoint must be a local path")
    root = Path(checkpoint).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"Wall Qwen3.5 checkpoint is not a directory: {root}")
    return root


def _local_regular_file(root: Path, name: str) -> Path:
    path = root / name
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise ValueError(f"Wall Qwen3.5 checkpoint is missing {name}") from error
    if resolved.parent != root:
        raise ValueError(
            f"Wall Qwen3.5 checkpoint file {name} resolves outside {root}"
        )
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError(f"Wall Qwen3.5 checkpoint entry {name} is not a file")
    return resolved


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(path: Path) -> tuple[int, int, int]:
    value = path.stat()
    return int(value.st_size), int(value.st_mtime_ns), int(value.st_ino)


def _is_action_tensor(name: str) -> bool:
    if name.startswith("model.action_processor."):
        return True
    if not name.startswith("model.language_model."):
        return False
    return (
        "_action" in name
        or ".action_" in name
        or ".mlp_action_expert." in name
        or ".input_layernorm_action." in name
        or ".post_attention_layernorm_action." in name
        or name.startswith("model.language_model.norm_action.")
    )


def _tensor_manifest_row(handle, name: str) -> bytes:
    value = handle.get_slice(name)
    shape = ",".join(str(int(dimension)) for dimension in value.get_shape())
    return f"{name}|{value.get_dtype()}|{shape}\n".encode("utf-8")


def _check_config_profile(config: dict[str, Any]) -> None:
    text = config.get("text_config") or {}
    vision = config.get("vision_config") or {}
    vla = config.get("vla_config") or {}
    rope = text.get("rope_parameters") or {}
    layer_types = tuple(text.get("layer_types") or ())
    expected_layers = tuple(
        "full_attention" if index % 4 == 3 else "linear_attention"
        for index in range(24)
    )
    profile = (
        tuple(config.get("architectures") or ()),
        config.get("model_type"),
        int(text.get("hidden_size", -1)),
        int(text.get("intermediate_size", -1)),
        int(text.get("num_hidden_layers", -1)),
        int(text.get("num_attention_heads", -1)),
        int(text.get("num_key_value_heads", -1)),
        int(text.get("head_dim", -1)),
        bool(text.get("attention_bias", True)),
        bool(text.get("attn_output_gate", False)),
        str(text.get("hidden_act", "")),
        float(text.get("rms_norm_eps", 0.0)),
        layer_types,
        float(rope.get("rope_theta", 0.0)),
        str(rope.get("rope_type", "")),
        tuple(int(value) for value in rope.get("mrope_section", ())),
        bool(rope.get("mrope_interleaved", False)),
        float(rope.get("partial_rotary_factor", 0.0)),
        int(vision.get("depth", -1)),
        int(vision.get("hidden_size", -1)),
        int(vision.get("intermediate_size", -1)),
        int(vision.get("out_hidden_size", -1)),
        int(vision.get("num_heads", -1)),
        tuple(int(value) for value in vla.get("dim_inputs", ())),
        int(vla.get("action_hidden_size", -1)),
        int(vla.get("state_hidden_size", -1)),
        bool(vla.get("norm_moe", False)),
        bool(vla.get("mlp_moe", False)),
        bool(vla.get("use_mot", False)),
        bool(vla.get("use_adarms", False)),
        int(vla.get("adarms_cond_dim", -1)),
        bool(vla.get("action_linear_attention", True)),
        int(vla.get("action_mlp_intermediate_size", -1)),
        bool(vla.get("use_state_string_representation", False)),
        bool(vla.get("causal_action_attention_mask", True)),
    )
    expected = (
        ("Qwen3_5ForConditionalGeneration",),
        "qwen3_5",
        2048,
        6144,
        24,
        8,
        2,
        256,
        False,
        True,
        "silu",
        1.0e-6,
        expected_layers,
        10_000_000.0,
        "default",
        (11, 11, 10),
        True,
        0.25,
        24,
        1024,
        4096,
        2048,
        16,
        (2048, 1024),
        1024,
        2560,
        True,
        True,
        True,
        True,
        1024,
        False,
        2048,
        True,
        False,
    )
    if profile != expected:
        raise _unsupported(
            "Wall Qwen3.5 supports only the exact TR-038 checkpoint profile; "
            f"got {profile}"
        )
    scheduler = vla.get("noise_scheduler") or {}
    scheduler_profile = (
        float(scheduler.get("beta_alpha", 0.0)),
        float(scheduler.get("beta_beta", 0.0)),
        float(scheduler.get("s", 0.0)),
        int(scheduler.get("num_inference_timesteps", -1)),
    )
    if scheduler_profile != (1.5, 1.0, 0.999, 10):
        raise _unsupported(
            "Wall Qwen3.5 requires the exact beta(1.5,1.0), s=0.999, "
            f"10-step scheduler; got {scheduler_profile}"
        )


def preflight_wall_qwen35_checkpoint(
    checkpoint: str | Path,
) -> WallQwen35CheckpointManifest:
    """Verify the exact local checkpoint without materializing any tensor."""

    root = _local_checkpoint_root(checkpoint)
    file_hashes: list[tuple[str, str]] = []
    file_stats: list[tuple[str, int, int, int]] = []
    for name, expected in _EXPECTED_FILE_SHA256.items():
        path = _local_regular_file(root, name)
        before = _file_identity(path)
        actual = _file_sha256(path)
        after = _file_identity(path)
        if before != after:
            raise RuntimeError(f"Wall Qwen3.5 checkpoint file changed while hashing: {name}")
        if actual != expected:
            raise _unsupported(
                f"Wall Qwen3.5 checkpoint digest mismatch for {name}: "
                f"expected {expected}, got {actual}"
            )
        file_hashes.append((name, actual))
        file_stats.append((name, *after))

    config_path = root / "config.json"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid Wall Qwen3.5 config.json: {error}") from error
    if not isinstance(config, dict):
        raise ValueError("Wall Qwen3.5 config.json must contain an object")
    _check_config_profile(config)

    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("Wall Qwen3.5 requires safetensors") from error

    weights_path = root / "model.safetensors"
    all_digest = hashlib.sha256()
    action_digest = hashlib.sha256()
    base_digest = hashlib.sha256()
    action_count = 0
    base_count = 0
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        names = sorted(handle.keys())
        for name in names:
            row = _tensor_manifest_row(handle, name)
            all_digest.update(row)
            if _is_action_tensor(name):
                action_digest.update(row)
                action_count += 1
            else:
                base_digest.update(row)
                base_count += 1
    manifest_values = (
        len(names),
        all_digest.hexdigest(),
        action_count,
        action_digest.hexdigest(),
        base_count,
        base_digest.hexdigest(),
    )
    expected_values = (
        _EXPECTED_TENSOR_COUNT,
        _EXPECTED_TENSOR_MANIFEST_SHA256,
        _EXPECTED_ACTION_TENSOR_COUNT,
        _EXPECTED_ACTION_MANIFEST_SHA256,
        _EXPECTED_BASE_TENSOR_COUNT,
        _EXPECTED_BASE_MANIFEST_SHA256,
    )
    if manifest_values != expected_values:
        raise _unsupported(
            "Wall Qwen3.5 safetensors key/dtype/shape manifest mismatch: "
            f"expected {expected_values}, got {manifest_values}"
        )
    if _file_identity(weights_path) != dict(
        (name, (size, mtime_ns, inode))
        for name, size, mtime_ns, inode in file_stats
    )["model.safetensors"]:
        raise RuntimeError("Wall Qwen3.5 model.safetensors changed during preflight")

    result = WallQwen35CheckpointManifest(
        root=root,
        file_sha256=tuple(file_hashes),
        file_stats=tuple(file_stats),
        tensor_count=len(names),
        tensor_manifest_sha256=manifest_values[1],
        action_tensor_count=action_count,
        action_manifest_sha256=manifest_values[3],
        base_tensor_count=base_count,
        base_manifest_sha256=manifest_values[5],
    )
    # A manifest is a process-local admission capability, not a serializable
    # claim supplied by callers. This prevents a hand-constructed dataclass
    # from replacing the expensive exact-file preflight.
    object.__setattr__(result, "_admission_token", _ADMISSION_TOKEN)
    return result


def _admit_manifest(
    checkpoint: str | Path,
    manifest: WallQwen35CheckpointManifest | None,
) -> WallQwen35CheckpointManifest:
    root = _local_checkpoint_root(checkpoint)
    if manifest is None:
        return preflight_wall_qwen35_checkpoint(root)
    if not isinstance(manifest, WallQwen35CheckpointManifest):
        raise TypeError("manifest must come from preflight_wall_qwen35_checkpoint")
    if manifest._admission_token is not _ADMISSION_TOKEN:
        raise ValueError(
            "manifest was not issued by preflight_wall_qwen35_checkpoint in "
            "this process"
        )
    if manifest.root != root:
        raise ValueError(
            f"checkpoint {root} does not match admitted manifest {manifest.root}"
        )
    manifest_profile = (
        manifest.tensor_count,
        manifest.tensor_manifest_sha256,
        manifest.action_tensor_count,
        manifest.action_manifest_sha256,
        manifest.base_tensor_count,
        manifest.base_manifest_sha256,
    )
    expected_profile = (
        _EXPECTED_TENSOR_COUNT,
        _EXPECTED_TENSOR_MANIFEST_SHA256,
        _EXPECTED_ACTION_TENSOR_COUNT,
        _EXPECTED_ACTION_MANIFEST_SHA256,
        _EXPECTED_BASE_TENSOR_COUNT,
        _EXPECTED_BASE_MANIFEST_SHA256,
    )
    if manifest_profile != expected_profile:
        raise ValueError("manifest tensor profile does not match Wall Qwen3.5")
    if manifest.file_sha256 != tuple(_EXPECTED_FILE_SHA256.items()):
        raise ValueError("manifest file digest profile does not match Wall Qwen3.5")
    for name in _EXPECTED_FILE_SHA256:
        path = _local_regular_file(root, name)
        if _file_identity(path) != manifest.stat_for(name):
            raise RuntimeError(
                f"Wall Qwen3.5 checkpoint file changed after preflight: {name}"
            )
        if manifest.digest_for(name) != _EXPECTED_FILE_SHA256[name]:
            raise ValueError(f"manifest digest is invalid for {name}")
    return manifest


def extend_wall_qwen35_tokenizer(
    tokenizer,
    *,
    target_vocab: int = CHECKPOINT_VOCAB_SIZE,
):
    """Install the accepted flow-only token ABI with exact IDs and order."""

    if isinstance(target_vocab, bool) or int(target_vocab) != CHECKPOINT_VOCAB_SIZE:
        raise ValueError(
            f"Wall Qwen3.5 tokenizer target must be {CHECKPOINT_VOCAB_SIZE}"
        )
    current = len(tokenizer)
    if current not in (_BASE_TOKENIZER_SIZE, CHECKPOINT_VOCAB_SIZE):
        raise ValueError(
            "Wall Qwen3.5 tokenizer must be pristine or already exactly extended; "
            f"got length {current}"
        )
    for offset, token in enumerate(_AUDIO_TAIL):
        token_id = _BASE_TOKENIZER_SIZE - len(_AUDIO_TAIL) + offset
        if tokenizer.convert_ids_to_tokens(token_id) != token:
            raise ValueError(
                f"Wall Qwen3.5 base tokenizer ID {token_id} is not {token!r}"
            )

    if current == _BASE_TOKENIZER_SIZE:
        added = tokenizer.add_special_tokens(
            {"additional_special_tokens": list(_FLOW_SPECIAL_TOKENS)},
            replace_extra_special_tokens=False,
        )
        if added != len(_FLOW_SPECIAL_TOKENS):
            raise ValueError(
                "Wall Qwen3.5 could not add the six flow special tokens exactly"
            )
        reserved = [
            f"<|flow_only_reserved_{index}|>" for index in range(_RESERVED_COUNT)
        ]
        added = tokenizer.add_tokens(reserved, special_tokens=False)
        if added != _RESERVED_COUNT:
            raise ValueError(
                f"Wall Qwen3.5 reserved-token extension added {added}, "
                f"expected {_RESERVED_COUNT}"
            )

    if len(tokenizer) != CHECKPOINT_VOCAB_SIZE:
        raise ValueError(
            f"Wall Qwen3.5 tokenizer length {len(tokenizer)} != "
            f"{CHECKPOINT_VOCAB_SIZE}"
        )
    for offset, token in enumerate(_FLOW_SPECIAL_TOKENS):
        expected_id = _BASE_TOKENIZER_SIZE + offset
        if tokenizer.convert_tokens_to_ids(token) != expected_id:
            raise ValueError(
                f"Wall Qwen3.5 token {token!r} must have ID {expected_id}"
            )
    all_special = set(getattr(tokenizer, "all_special_tokens", ()))
    missing_special = sorted(set(_FLOW_SPECIAL_TOKENS) - all_special)
    if missing_special:
        raise ValueError(
            f"Wall Qwen3.5 flow tokens are not special: {missing_special}"
        )
    reserve_start = _BASE_TOKENIZER_SIZE + len(_FLOW_SPECIAL_TOKENS)
    all_special = set(getattr(tokenizer, "all_special_tokens", ()))
    for index in range(_RESERVED_COUNT):
        token = f"<|flow_only_reserved_{index}|>"
        expected_id = reserve_start + index
        if tokenizer.convert_tokens_to_ids(token) != expected_id:
            raise ValueError(
                f"Wall Qwen3.5 token {token!r} must have ID {expected_id}"
            )
        if token in all_special:
            raise ValueError(
                f"Wall Qwen3.5 reserve token {token!r} must remain plain"
            )
    return tokenizer


def _resolve_submodule(root: torch.nn.Module, qualified_name: str):
    parent_name, _, leaf = qualified_name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    return parent, leaf


def _replace_parameter(
    model: torch.nn.Module,
    name: str,
    value: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> None:
    parent, leaf = _resolve_submodule(model, name)
    old = parent._parameters.get(leaf)
    if old is None:
        raise KeyError(f"Wall Qwen3.5 base model has no parameter {name}")
    if tuple(old.shape) != tuple(value.shape):
        raise ValueError(
            f"Wall Qwen3.5 parameter {name} shape {tuple(value.shape)} != "
            f"{tuple(old.shape)}"
        )
    tensor = value.detach().to(device="cpu", dtype=dtype).contiguous()
    parent._parameters[leaf] = torch.nn.Parameter(tensor, requires_grad=False)


def _materialize_rotary_buffers(model: torch.nn.Module) -> None:
    for module in model.modules():
        class_name = type(module).__name__
        if class_name == "Qwen3_5VisionRotaryEmbedding":
            fresh = type(module)(int(module.dim), float(module.theta))
            module._buffers["inv_freq"] = fresh.inv_freq.detach().cpu().clone()
        elif class_name == "Qwen3_5TextRotaryEmbedding":
            fresh = type(module)(module.config, device="cpu")
            module._buffers["inv_freq"] = fresh.inv_freq.detach().cpu().clone()
            module._buffers["original_inv_freq"] = (
                fresh.original_inv_freq.detach().cpu().clone()
            )


def _prepare_hf_config(config):
    text = getattr(config, "text_config", config)
    text.vocab_size = CHECKPOINT_VOCAB_SIZE
    if hasattr(config, "vocab_size"):
        config.vocab_size = CHECKPOINT_VOCAB_SIZE
    # The checkpoint's training-only flash_mask string is not a public
    # Transformers attention implementation.  RhinoForge replaces the forward
    # after CPU load, so eager is the inert construction-time choice.
    config._attn_implementation = "eager"
    text._attn_implementation = "eager"
    return config


def stream_load_wall_qwen35_base_model(
    model_or_none,
    checkpoint: str | Path,
    *,
    config=None,
    model_class=None,
    dtype: torch.dtype = torch.float16,
    manifest: WallQwen35CheckpointManifest | None = None,
):
    """Stream the 593 ordinary HF tensors without loading action tensors.

    ``model_or_none`` may be ``None`` (a meta model is constructed), an already
    meta-constructed model, or a clean CPU model.  Every parameter is replaced
    exactly once; the caller must discard the model after any failure.
    """

    if dtype is not torch.float16:
        raise ValueError("Wall Qwen3.5 base execution supports only torch.float16")
    admitted = _admit_manifest(checkpoint, manifest)
    root = admitted.root
    if config is None:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            root, local_files_only=True, trust_remote_code=False
        )
    config = _prepare_hf_config(config)
    if model_class is None:
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5ForConditionalGeneration,
        )

        model_class = Qwen3_5ForConditionalGeneration
    if model_or_none is None:
        with torch.device("meta"):
            model = model_class(config)
    else:
        model = model_or_none
        if getattr(model, "_wall_qwen35_load_started", False):
            raise RuntimeError(
                "Wall Qwen3.5 base loading was already attempted; construct a "
                "fresh model"
            )
    if type(model).__name__ != "Qwen3_5ForConditionalGeneration":
        raise TypeError(
            "Wall Qwen3.5 streaming requires the standard "
            "Qwen3_5ForConditionalGeneration class"
        )
    model._wall_qwen35_load_started = True

    target_keys = set(model.state_dict().keys())
    expected_missing = {"lm_head.weight"}
    loaded: set[str] = set()
    try:
        from safetensors import safe_open

        with safe_open(
            str(root / "model.safetensors"), framework="pt", device="cpu"
        ) as handle:
            base_names = sorted(
                name for name in handle.keys() if not _is_action_tensor(name)
            )
            for name in base_names:
                value = handle.get_tensor(name)
                if name.endswith(".mlp.gate_up_proj.weight"):
                    if value.ndim != 2 or value.shape[0] % 2:
                        raise ValueError(f"invalid fused MLP weight {name}")
                    gate, up = value.chunk(2, dim=0)
                    gate_name = name.replace("gate_up_proj", "gate_proj")
                    up_name = name.replace("gate_up_proj", "up_proj")
                    _replace_parameter(model, gate_name, gate, dtype=dtype)
                    _replace_parameter(model, up_name, up, dtype=dtype)
                    loaded.update((gate_name, up_name))
                else:
                    _replace_parameter(model, name, value, dtype=dtype)
                    loaded.add(name)
                del value
        if target_keys - loaded != expected_missing:
            raise RuntimeError(
                "Wall Qwen3.5 base streaming did not close the target state: "
                f"missing={sorted(target_keys - loaded)}, "
                f"unexpected={sorted(loaded - target_keys)}"
            )
        model.tie_weights()
        _materialize_rotary_buffers(model)
        meta_parameters = [
            name for name, value in model.named_parameters() if value.is_meta
        ]
        meta_buffers = [name for name, value in model.named_buffers() if value.is_meta]
        if meta_parameters or meta_buffers:
            raise RuntimeError(
                "Wall Qwen3.5 streaming left meta tensors: "
                f"parameters={meta_parameters}, buffers={meta_buffers}"
            )
        if any(
            value.device.type != "cpu" or value.dtype != torch.float16
            for value in model.parameters()
        ):
            raise RuntimeError("Wall Qwen3.5 base parameters are not CPU FP16")
        model.eval()
        model._wall_qwen35_checkpoint_manifest = admitted
        model._wall_qwen35_load_complete = True
        return model
    except BaseException:
        # The marker deliberately remains set.  A partially replaced model is
        # not a safe retry target even though no swizzle has happened yet.
        raise


__all__ = [
    "CHECKPOINT_VOCAB_SIZE",
    "WallQwen35CheckpointManifest",
    "extend_wall_qwen35_tokenizer",
    "preflight_wall_qwen35_checkpoint",
    "stream_load_wall_qwen35_base_model",
]
