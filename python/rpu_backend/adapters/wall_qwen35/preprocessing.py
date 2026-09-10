"""Self-contained input preparation for the exact Wall Qwen3.5 profile.

This module intentionally does not import Wall-X, Harrix, x2robot_utils, or
qwen_vl_utils.  It reproduces the Dataset-V2 inference branch used to train the
checkpoint: three ordered camera placeholders, right padding without
truncation, 256-bin state strings, and one PIL BICUBIC resize before the Qwen
image processor.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from concurrent.futures import Executor, wait
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image


DEFAULT_CAMERAS = ("face_view", "left_wrist_view", "right_wrist_view")
CAMERA_LABELS = {
    "face_view": "front view",
    "left_wrist_view": "left wrist view",
    "right_wrist_view": "right wrist view",
}
ACTION_HORIZON = 32
MAX_ACTION_PREFIX_LENGTH = 384
# Fixed execution shapes used by the controlled prefix-bucket runtime.  Every
# admitted real prefix is at most 63 rows below its selected bucket.
WALL_PREFIX_BUCKETS = (64, 128, 192, 256, 320, 384)
# Each execution bucket can produce three native GDN tail envelopes: one real
# row, 2..31 unsegmented rows, or 32+ segmented rows. Graph identity must retain
# that topology class in addition to the physical bucket shape.
WALL_PREFILL_GRAPH_MAX_ENTRIES = 3 * len(WALL_PREFIX_BUCKETS)
STATE_DIM = 26
STATE_BINS = 256
MAX_SEQ_LENGTH = 780
CHECKPOINT_VOCAB_SIZE = 256277
DEFAULT_ROBOT_ID = "10070"
_ROBOT_EMBODIMENTS = {
    ("x2_normal", DEFAULT_ROBOT_ID): "X2 ex001_desktop",
}
PROFILE_MASK = (1.0,) * 20 + (0.0,) * 6
# Exact per-byte conversion for the fixed mean/std profile. Conversion before
# patch permutation/temporal duplication is identical and avoids strided casts.
_UINT8_TO_FP16 = ((np.arange(256, dtype=np.float32) - np.float32(127.5))
                 / np.float32(127.5)).astype(np.float16)
_UINT8_TO_FP16.setflags(write=False)
SPECIAL_TOKENS = (
    "<|propri|>",
    "<|action|>",
    "<point>",
    "</point>",
    "<box>",
    "</box>",
)


@dataclass(frozen=True)
class WallQwen35PreparedInput:
    """A single, unpadded Wall Qwen3.5 request and its action suffix."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    full_position_ids: torch.Tensor | None
    prefix_length: int
    action_position_ids: torch.Tensor | None
    normalized_state: torch.Tensor
    agent_pos_mask: torch.Tensor
    dof_mask: torch.Tensor
    dataset_key: str

    @property
    def prefix_input_ids(self) -> torch.Tensor:
        return self.input_ids[:, : self.prefix_length].contiguous()

    @property
    def prefix_attention_mask(self) -> torch.Tensor:
        return self.attention_mask[:, : self.prefix_length].contiguous()

    @property
    def prefix_mm_token_type_ids(self) -> torch.Tensor:
        return self.mm_token_type_ids[:, : self.prefix_length].contiguous()

    @property
    def prefix_position_ids(self) -> torch.Tensor | None:
        if self.full_position_ids is None:
            return None
        return self.full_position_ids[..., : self.prefix_length].contiguous()


def smart_resize(
    height: int,
    width: int,
    *,
    factor: int = 32,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 4096 * 4096,
) -> tuple[int, int]:
    """Official Qwen-VL resize rule, kept local to avoid qwen_vl_utils."""

    height, width, factor = int(height), int(width), int(factor)
    if height <= 0 or width <= 0 or factor <= 0:
        raise ValueError("image dimensions and resize factor must be positive")
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must not exceed 200")
    h = max(factor, round(height / factor) * factor)
    w = max(factor, round(width / factor) * factor)
    if h * w > max_pixels:
        scale = math.sqrt(max_pixels / (height * width))
        h = max(factor, round(height * scale / factor) * factor)
        w = max(factor, round(width * scale / factor) * factor)
    if h * w < min_pixels:
        scale = math.sqrt(min_pixels / (height * width))
        h = max(factor, round(height * scale / factor) * factor)
        w = max(factor, round(width * scale / factor) * factor)
    return int(h), int(w)


def resize_dataset_v2_image(
    image: Image.Image,
    *,
    target: int = 224,
    factor: int = 32,
    min_pixels: int = 4 * 32 * 32,
    max_pixels: int = 4096 * 4096,
) -> Image.Image:
    """One BICUBIC resize matching Dataset-V2 validation."""

    if not isinstance(image, Image.Image):
        raise TypeError(f"expected PIL.Image.Image, got {type(image).__name__}")
    if image.mode != "RGB":
        image = image.convert("RGB")
    out_h, out_w = _dataset_v2_size(
        image.size, target=target, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    return image.resize((out_w, out_h), Image.Resampling.BICUBIC)


def _dataset_v2_size(size, *, target=224, factor=32,
                     min_pixels=4 * 32 * 32, max_pixels=4096 * 4096):
    width, height = size
    if target > 0:
        if width > height:
            width, height = target, max(1, int(target * height / width))
        else:
            width, height = max(1, int(target * width / height)), target
    return smart_resize(
        height, width, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )


def extend_flow_tokenizer(tokenizer, *, target_vocab: int = CHECKPOINT_VOCAB_SIZE) -> None:
    """Delegate to the package's single exact flow-token ABI owner."""

    from .checkpoint import extend_wall_qwen35_tokenizer

    extend_wall_qwen35_tokenizer(tokenizer, target_vocab=target_vocab)


def normalize_vector(
    value: Any,
    minimum: torch.Tensor,
    delta: torch.Tensor,
    *,
    name: str,
) -> torch.Tensor:
    if (isinstance(value, torch.Tensor) and value.device.type == "cpu"
            and value.dtype == torch.float32 and not value.requires_grad
            and not value.is_neg() and all(
                item.device.type == "cpu" and item.dtype == torch.float32
                and not item.requires_grad and not item.is_neg()
                for item in (minimum, delta))):
        if tuple(value.shape) not in ((STATE_DIM,), (1, STATE_DIM), (1, 1, STATE_DIM)):
            raise ValueError(f"{name} must have shape [26], [1,26], or [1,1,26]")
        array = value.numpy().reshape(1, 1, STATE_DIM)
        low, span = minimum.numpy().reshape(1, 1, -1), delta.numpy().reshape(1, 1, -1)
        if low.shape[-1] != STATE_DIM or span.shape[-1] != STATE_DIM:
            raise ValueError(f"{name} normalizer must contain 26 values")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")
        if (span <= 0).any():
            raise ValueError(f"{name} normalizer delta must be positive")
        normalized = array - low
        normalized /= span
        normalized *= np.float32(2.0)
        normalized -= np.float32(1.0)
        np.clip(normalized, -1.0, 1.0, out=normalized)
        return torch.from_numpy(normalized)
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tuple(tensor.shape) == (STATE_DIM,):
        tensor = tensor.view(1, 1, STATE_DIM)
    elif tuple(tensor.shape) == (1, STATE_DIM):
        tensor = tensor.unsqueeze(1)
    if tuple(tensor.shape) != (1, 1, STATE_DIM):
        raise ValueError(f"{name} must have shape [26], [1,26], or [1,1,26]")
    if not bool(np.isfinite(tensor.detach().resolve_neg().numpy()).all()):
        raise ValueError(f"{name} must contain only finite values")
    minimum = minimum.detach().to(device="cpu", dtype=torch.float32).reshape(1, 1, -1)
    delta = delta.detach().to(device="cpu", dtype=torch.float32).reshape(1, 1, -1)
    if tuple(minimum.shape[-1:]) != (STATE_DIM,) or tuple(delta.shape[-1:]) != (STATE_DIM,):
        raise ValueError(f"{name} normalizer must contain 26 values")
    if bool((delta.resolve_neg().numpy() <= 0).any()):
        raise ValueError(f"{name} normalizer delta must be positive")
    if tensor.requires_grad:
        return (((tensor - minimum) / delta) * 2.0 - 1.0).clamp(-1.0, 1.0)
    normalized = tensor.resolve_neg().numpy() - minimum.resolve_neg().numpy()
    normalized /= delta.resolve_neg().numpy()
    normalized *= np.float32(2.0)
    normalized -= np.float32(1.0)
    np.clip(normalized, -1.0, 1.0, out=normalized)
    return torch.from_numpy(normalized)


def digitize_state(
    normalized_state: torch.Tensor,
    agent_pos_mask: torch.Tensor,
    *,
    bins: int = STATE_BINS,
) -> str:
    """Exact np.digitize branch used by x2robot_utils.preprocesser_call."""

    state = normalized_state if (normalized_state.device.type == "cpu"
        and normalized_state.dtype == torch.float32 and not normalized_state.requires_grad
        and not normalized_state.is_neg()) else normalized_state.detach().to(device="cpu", dtype=torch.float32).resolve_neg()
    mask = agent_pos_mask if (agent_pos_mask.device.type == "cpu"
        and not agent_pos_mask.requires_grad and not agent_pos_mask.is_neg()
        ) else agent_pos_mask.detach().to(device="cpu").resolve_neg()
    if tuple(state.shape) != (1, 1, STATE_DIM) or tuple(mask.shape) != (1, 1, STATE_DIM):
        raise ValueError("normalized state and agent_pos_mask must be [1,1,26]")
    if int(bins) == STATE_BINS:
        boundaries = np.arange(STATE_BINS, dtype=np.float32) / np.float32(128.0) - np.float32(1.0)
        discretized = np.searchsorted(boundaries, state.numpy(), side="right") - 1
        return " ".join(map(str, discretized[0, 0, mask.numpy()[0, 0].astype(bool)].tolist()))
    boundaries = torch.linspace(-1.0, 1.0, int(bins) + 1)[:-1]
    # np.digitize(..., right=False) is torch.bucketize(..., right=True).
    discretized = torch.bucketize(state, boundaries, right=True) - 1
    return " ".join(map(str, discretized[0, 0, mask[0, 0].bool()].tolist()))


def build_prompt(
    instruction: str | Mapping[str, Any],
    camera_names: Sequence[str] = DEFAULT_CAMERAS,
    *,
    state_string: str,
    dataset_key: str = "x2_normal",
    robot_id: str | int = DEFAULT_ROBOT_ID,
    action_horizon: int = ACTION_HORIZON,
) -> tuple[str, str]:
    if tuple(camera_names) != DEFAULT_CAMERAS:
        raise ValueError(f"Wall Qwen3.5 requires camera order {DEFAULT_CAMERAS!r}")
    if int(action_horizon) != ACTION_HORIZON:
        raise ValueError("Wall Qwen3.5 action horizon must be 32")
    def _sanitize(value: Any) -> str:
        words = str(value or "").strip().split()
        if len(words) > 256:
            raise ValueError("runtime instruction exceeds the 256-word profile")
        return " ".join(words)

    if isinstance(instruction, Mapping):
        task = _sanitize(instruction.get("task", ""))
        detail = _sanitize(instruction.get("detail", ""))
        lines = ([f"Task: {task}"] if task else []) + (
            [f"Instruction: {detail}"] if detail else []
        )
        instruction_text = "\n".join(lines)
    else:
        instruction_text = f"Task: {_sanitize(instruction)}"
    if isinstance(robot_id, bool):
        raise ValueError("robot_id must be the admitted decimal ID 10070")
    robot_id = str(robot_id)
    embodiment = _ROBOT_EMBODIMENTS.get((dataset_key, robot_id))
    if embodiment is None:
        if dataset_key != "x2_normal":
            raise ValueError(
                "initial Wall Qwen3.5 profile requires dataset_key='x2_normal'"
            )
        raise ValueError(
            "initial Wall Qwen3.5 profile admits only robot_id='10070'"
        )
    camera_setup = "".join(f" {CAMERA_LABELS[name]}," for name in camera_names)
    prologue = (
        "<|im_start|>system\n"
        "You are an embodied vision-language-action (VLA) model controlling "
        "the robot with language instructions."
        f"\n Embodiment: {embodiment}"
        f"\n Camera Setup:{camera_setup}"
        "\n Frequency: 32HZ"
        "\n Action Space: Rel EEF"
        "\n<|im_end|>\n"
    )
    observation = "<|im_start|>user\nObservation:"
    for name in camera_names:
        observation += (
            f" {CAMERA_LABELS[name]}: "
            "<|vision_start|><|image_pad|><|vision_end|>"
        )
    user = (
        f"{observation}\n{instruction_text}\n"
        "Predict the next action in robot action.\n"
        f"Proprioception: {state_string}\n<|im_end|>\n"
    )
    prefix = prologue + user + "<|im_start|>assistant\n"
    return prefix, "<|action|>" * ACTION_HORIZON


def _mask26(value: Any | None, *, name: str) -> torch.Tensor:
    if value is None:
        tensor = torch.tensor(PROFILE_MASK, dtype=torch.float32)
    else:
        tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tuple(tensor.shape) == (STATE_DIM,):
        tensor = tensor.view(1, 1, STATE_DIM)
    elif tuple(tensor.shape) == (1, STATE_DIM):
        tensor = tensor.unsqueeze(1)
    if tuple(tensor.shape) != (1, 1, STATE_DIM):
        raise ValueError(f"{name} must have 26 entries")
    array = tensor.detach().resolve_neg().numpy()
    if not bool(((array == 0) | (array == 1)).all()):
        raise ValueError(f"{name} entries must be zero or one")
    if not np.array_equal(array.reshape(-1), PROFILE_MASK):
        raise ValueError(
            f"{name} must match the exact X2 profile [1]*20 + [0]*6"
        )
    return tensor.contiguous()


def _resize_camera(value: Image.Image | str | Path) -> Image.Image:
    if isinstance(value, (str, Path)):
        with Image.open(value) as opened:
            return resize_dataset_v2_image(opened)
    return resize_dataset_v2_image(value)




def _configure_atomic_action_suffix(processor):
    """Cold tokenizer metadata only; no prompt or observation result caching."""
    tokenizer = processor.tokenizer
    action_id = tokenizer.convert_tokens_to_ids("<|action|>")
    token = getattr(tokenizer, "added_tokens_decoder", {}).get(action_id)
    processor._wall_atomic_action_id = (
        action_id if getattr(tokenizer, "is_fast", False)
        and token is not None and str(token) == "<|action|>"
        and token.special and not token.lstrip and not token.rstrip
        and not token.normalized and not token.single_word else None
    )
    _configure_prefix_scaffold(processor)


def _configure_prefix_scaffold(processor):
    """Compile only immutable system/camera text, never request content."""
    import json
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase
    tokenizer = processor.tokenizer
    processor._wall_prefix_plan = None
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if (not getattr(tokenizer, "is_fast", False) or backend is None
            or getattr(tokenizer.__call__, "__func__", None) is not PreTrainedTokenizerBase.__call__):
        return
    normalizer = backend.normalizer
    if normalizer is not None and json.loads(normalizer.__getstate__()) != {"type": "NFC"}:
        return
    # The immutable ASCII scaffold ends at an atomic token and the live tail
    # starts with a newline; NFC cannot compose across this boundary.
    post = backend.post_processor
    post_state = None if post is None else json.loads(post.__getstate__())
    byte_level = post_state == {
        "type": "ByteLevel", "add_prefix_space": False,
        "trim_offsets": False, "use_regex": False,
    }
    identity_template = (isinstance(post_state, dict)
        and post_state.get("type") == "TemplateProcessing"
        and post_state.get("single") == [{"Sequence": {"id": "A", "type_id": 0}}]
        and post_state.get("special_tokens") == {})
    if post_state is not None and not byte_level and not identity_template:
        return
    pre = backend.pre_tokenizer
    pre_state = None if pre is None else json.loads(pre.__getstate__())
    def admissible(state):
        if not isinstance(state, dict):
            return False
        if state.get("type") == "Sequence":
            return all(admissible(item) for item in state.get("pretokenizers", []))
        return (state.get("type") == "Split" or
                state.get("type") == "ByteLevel" and not state.get("add_prefix_space", True))
    if not admissible(pre_state):
        return
    decoder = tokenizer.added_tokens_decoder
    for text in ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>"):
        token = decoder.get(tokenizer.convert_tokens_to_ids(text))
        if (token is None or str(token) != text or not token.special or token.lstrip
                or token.rstrip or token.normalized or token.single_word):
            return
    prefix, _ = build_prompt("", state_string="")
    end = prefix.rfind("<|vision_end|>") + len("<|vision_end|>")
    scaffold = prefix[:end]
    ids = tokenizer(scaffold, add_special_tokens=False)["input_ids"]
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    chunks = []
    start = 0
    for index, token_id in enumerate(ids):
        if token_id == image_id:
            chunks.append(tuple(ids[start:index]))
            start = index + 1
    chunks.append(tuple(ids[start:]))
    if len(chunks) == 4:
        processor._wall_prefix_plan = (scaffold, tuple(chunks), image_id)


def _pack_image_array(image, dtype, output=None):
    h, w = image.height // 16, image.width // 16
    if dtype == np.float16:
        array = np.take(_UINT8_TO_FP16, np.frombuffer(image.tobytes(), dtype=np.uint8),
                        mode="clip").reshape(image.height, image.width, 3)
    else:
        array = np.asarray(image, dtype=np.float32)
        array -= np.float32(127.5)
        array /= np.float32(127.5)
    spatial = array.reshape(h // 2, 2, 16, w // 2, 2, 16, 3)
    spatial = spatial.transpose(0, 3, 1, 4, 6, 2, 5)
    pixels = np.empty((h * w, 1536), dtype=dtype) if output is None else output
    pixels.reshape(h // 2, w // 2, 2, 2, 3, 2, 16, 16)[...] = spatial[..., None, :, :]
    return pixels


def _resize_and_pack(image, size, box, dtype, output=None):
    resized = (_resize_camera(image) if box is None else
               image.resize(size, Image.Resampling.BICUBIC, box))
    return _pack_image_array(resized, dtype, output)


def _submit_image_patches(values, sizes, executor, dtype, output=None):
    """Fuse resize and packing; bands end on complete 32-pixel merge rows."""
    if dtype not in (torch.float16, torch.float32):
        raise ValueError("Wall pixel dtype must be float32 or float16")
    numpy_dtype = np.float16 if dtype == torch.float16 else np.float32
    if output is None:
        output = np.empty((sum(h * w // 256 for h, w in sizes), 1536), dtype=numpy_dtype)
    offset = 0
    futures = []
    try:
        for image, (height, width) in zip(values, sizes):
            split = (image.mode == "RGB" and image.height >= 512
                     and image.height % 4 == 0 and height >= 128
                     and height & (height - 1) == 0)
            if split or (image.mode == "RGB" and height >= 128
                         and image.height % height == 0):
                image.load()
                # Integer source/output scale also preserves exact sample
                # centers. Two uneven wrist bands align on 32-pixel merge rows.
                boundaries = ([0, height // 2, height]
                              if split else [0, (height // 64) * 32, height])
                for start, end in zip(boundaries, boundaries[1:]):
                    box = (0, start * image.height // height, image.width,
                           end * image.height // height)
                    count = width * (end - start) // 256
                    futures.append(executor.submit(
                        _resize_and_pack, image, (width, end - start), box, numpy_dtype,
                        output[offset:offset + count]))
                    offset += count
            else:
                count = width * height // 256
                futures.append(executor.submit(
                    _resize_and_pack, image, (width, height), None, numpy_dtype,
                    output[offset:offset + count]))
                offset += count
        return futures
    except BaseException:
        for future in futures:
            future.cancel()
        wait(futures)
        raise


def _submit_image_resizes(values, sizes, executor):
    jobs = []
    futures = []
    try:
        for image, (height, width) in zip(values, sizes):
            # A power-of-two output height and integral quarter boundaries
            # preserve PIL's vertical sample centers exactly. Each resize
            # receives the FULL source image/halo: never crop the input.
            split = (
                image.mode == "RGB" and image.height >= 512
                and image.height % 4 == 0 and height >= 32
                and height & (height - 1) == 0
            )
            group = []
            if split:
                image.load()  # finish lazy decoding before concurrent reads
                for index in range(4):
                    future = executor.submit(
                        image.resize, (width, height // 4),
                        Image.Resampling.BICUBIC,
                        (0, index * (image.height // 4), image.width,
                         (index + 1) * (image.height // 4)),
                    )
                    group.append(future)
                    futures.append(future)
            else:
                future = executor.submit(_resize_camera, image)
                group.append(future)
                futures.append(future)
            jobs.append(((width, height), group))
        return jobs, futures
    except BaseException:
        for future in futures:
            future.cancel()
        wait(futures)
        raise


def _collect_image_resizes(jobs):
    images = []
    for size, group in jobs:
        if len(group) == 1:
            images.append(group[0].result())
        else:
            image = Image.new("RGB", size)
            for index, future in enumerate(group):
                image.paste(future.result(), (0, index * (size[1] // 4)))
            images.append(image)
    return images



@lru_cache(maxsize=64)
def _vision_position_template(height, width):
    # Shape-only immutable metadata, never an observation or request output.
    template = np.empty((3, height * width), dtype=np.int64)
    template[0] = 0
    template[1] = np.repeat(np.arange(height), width)
    template[2] = np.tile(np.arange(width), height)
    template.setflags(write=False)
    return template


def _compute_wall_positions(fusion, compute, full, mm, grids):
    """Equivalent batch-one image-only M-RoPE; preserve HF rope_deltas."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    if (
        getattr(compute, "__func__", None) is not Qwen3_5Model.compute_3d_position_ids
        or getattr(fusion.get_rope_index, "__func__", None) is not Qwen3_5Model.get_rope_index
        or getattr(fusion.get_vision_position_ids, "__func__", None) is not Qwen3_5Model.get_vision_position_ids
        or fusion.config.vision_config.spatial_merge_size != 2
        or full.input_ids.shape[0] != 1
        or not bool(full.attention_mask.numpy().all())
    ):
        return compute(
            input_ids=full.input_ids, inputs_embeds=None,
            image_grid_thw=grids, video_grid_thw=None,
            attention_mask=full.attention_mask, past_key_values=None,
            mm_token_type_ids=mm,
        )
    types = mm.numpy()[0]
    boundaries = [0, *(np.flatnonzero(types[1:] != types[:-1]) + 1).tolist(), len(types)]
    spans = zip(boundaries, boundaries[1:])
    grid_values = grids.tolist()
    if any(t != 1 or h % 2 or w % 2 for t, h, w in grid_values):
        raise ValueError("Wall M-RoPE requires single-frame even image grids")
    positions = np.empty((3, 1, len(types)), dtype=np.int64)
    current = 0
    image_index = 0
    for start, end in spans:
        length = end - start
        if types[start] == 0:
            positions[:, 0, start:end] = np.arange(current, current + length)
            current += length
        elif types[start] == 1 and image_index < len(grid_values):
            _, height, width = grid_values[image_index]
            height, width = height // 2, width // 2
            if length != height * width:
                raise ValueError("Wall image token/grid length mismatch")
            positions[:, 0, start:end] = _vision_position_template(height, width) + current
            current += max(height, width)
            image_index += 1
        else:
            raise ValueError("Wall M-RoPE image/grid count mismatch")
    if image_index != len(grid_values):
        raise ValueError("Wall M-RoPE image/grid count mismatch")
    fusion.rope_deltas = torch.from_numpy(np.asarray([[current - len(types)]], dtype=np.int64))
    return torch.from_numpy(positions)

def _use_numpy_image_inputs(image_processor):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )

    return (
        type(image_processor) is Qwen2VLImageProcessor
        and image_processor.patch_size == 16
        and image_processor.merge_size == 2
        and image_processor.temporal_patch_size == 2
        and image_processor.do_rescale
        and image_processor.do_normalize
        and image_processor.rescale_factor == 1.0 / 255.0
        and isinstance(image_processor.image_mean, (list, tuple))
        and isinstance(image_processor.image_std, (list, tuple))
        and tuple(image_processor.image_mean) == (0.5, 0.5, 0.5)
        and tuple(image_processor.image_std) == (0.5, 0.5, 0.5)
    )


def _numpy_image_inputs(image_processor, images, *, dtype=torch.float32, executor=None):
    """Exact CPU specialization of the admitted Torchvision image profile.

    Keep PIL's single BICUBIC resize. Normalize before duplicating the static
    frame's temporal plane, and materialize the final patch order just once.
    Unknown processors/settings retain their own reference implementation.
    """
    if dtype not in (torch.float32, torch.float16):
        raise ValueError("Wall pixel dtype must be float32 or float16")
    if not _use_numpy_image_inputs(image_processor):
        result = image_processor(images=images, return_tensors="pt")
        result["pixel_values"] = result["pixel_values"].to(dtype=dtype)
        return result

    if executor is not None and len(images) > 1:
        # Resizes have completed; reuse the same bounded pool for independent
        # camera packing. Include joins and final materialization in this call.
        futures = []
        try:
            for image in images:
                futures.append(executor.submit(
                    _numpy_image_inputs, image_processor, [image], dtype=dtype))
            parts = [future.result() for future in futures]
            return {
                "pixel_values": torch.from_numpy(np.concatenate(
                    [part["pixel_values"].numpy() for part in parts], axis=0)),
                "image_grid_thw": torch.from_numpy(np.concatenate(
                    [part["image_grid_thw"].numpy() for part in parts], axis=0)),
            }
        finally:
            for future in futures:
                future.cancel()
            wait(futures)

    grids = [(1, image.height // 16, image.width // 16) for image in images]
    pixels = np.empty((sum(h * w for _, h, w in grids), 1536),
                      dtype=np.float16 if dtype == torch.float16 else np.float32)
    offset = 0
    for image, (_, h, w) in zip(images, grids):
        # Torchvision fuses mean/std with 1/rescale_factor: (u8-127.5)/127.5.
        # Preserve the two FP32 operations; u8/255*2-1 rounds differently.
        if dtype == torch.float16:
            array = _UINT8_TO_FP16[np.asarray(image)]
        else:
            array = np.asarray(image, dtype=np.float32)
            array -= np.float32(127.5)
            array /= np.float32(127.5)
        spatial = array.reshape(h // 2, 2, 16, w // 2, 2, 16, 3)
        spatial = spatial.transpose(0, 3, 1, 4, 6, 2, 5)
        target = pixels[offset:offset + h * w].reshape(h // 2, w // 2, 2, 2, 3, 2, 16, 16)
        target[...] = spatial[..., None, :, :]
        offset += h * w
    return {
        "pixel_values": torch.from_numpy(pixels),
        "image_grid_thw": torch.tensor(grids, dtype=torch.int64),
    }


def prepare_wall_qwen35_input(
    processor,
    model,
    *,
    images: Mapping[str, Image.Image | str | Path],
    instruction: str | Mapping[str, Any],
    state: Any,
    state_min: torch.Tensor,
    state_delta: torch.Tensor,
    dataset_key: str,
    robot_id: str | int = DEFAULT_ROBOT_ID,
    agent_pos_mask: Any | None = None,
    dof_mask: Any | None = None,
    camera_names: Sequence[str] = DEFAULT_CAMERAS,
    max_seq_length: int = MAX_SEQ_LENGTH,
    image_executor: Executor | None = None,
    pixel_dtype: torch.dtype = torch.float32,
) -> WallQwen35PreparedInput:
    """Prepare one exact-profile image/state/action request on CPU."""

    if dataset_key != "x2_normal":
        raise ValueError(
            "initial Wall Qwen3.5 profile requires dataset_key='x2_normal'"
        )

    apos_mask = _mask26(agent_pos_mask, name="agent_pos_mask")
    action_mask = _mask26(dof_mask, name="dof_mask")
    normalized = normalize_vector(state, state_min, state_delta, name="state")
    state_string = digitize_state(normalized, apos_mask)
    prefix, postfix = build_prompt(
        instruction, camera_names, state_string=state_string,
        dataset_key=dataset_key, robot_id=robot_id,
    )

    values = []
    for name in camera_names:
        if name not in images:
            raise KeyError(f"missing camera image {name!r}")
        values.append(images[name])

    image_processor = processor.image_processor
    patch_size = int(getattr(image_processor, "patch_size", 0) or 0)
    merge_size = int(getattr(image_processor, "merge_size", 0) or 0)
    if patch_size * merge_size != 32:
        raise ValueError(
            "Wall Qwen3.5 image processor requires patch_size * merge_size = 32, "
            f"got {patch_size} * {merge_size}"
        )
    if not hasattr(image_processor, "do_resize"):
        raise ValueError(
            "Wall Qwen3.5 image processor has no do_resize switch; cannot "
            "enforce Dataset-V2 single-stage image resampling"
        )
    image_processor.do_resize = False
    image_futures = None
    if (
        image_executor is not None
        and _use_numpy_image_inputs(image_processor)
        and all(isinstance(value, Image.Image) for value in values)
    ):
        # Grid metadata depends only on dimensions, so text and M-RoPE can run
        # while PIL releases the GIL for the three independent resizes.
        sizes = [_dataset_v2_size(value.size) for value in values]
        grids = torch.tensor([(1, h // patch_size, w // patch_size)
                              for h, w in sizes], dtype=torch.int64)
        pixel_tensor = torch.empty((sum(h * w // 256 for h, w in sizes), 1536), dtype=pixel_dtype)
        image_futures = _submit_image_patches(
            values, sizes, image_executor, pixel_dtype, pixel_tensor.numpy())
    else:
        ordered_images = [_resize_camera(value) for value in values]
        image_inputs = _numpy_image_inputs(
                image_processor, ordered_images, dtype=pixel_dtype, executor=image_executor)
        grids = image_inputs["image_grid_thw"]
    try:
        if prefix.count("<|image_pad|>") != len(grids):
            raise ValueError("image placeholder/grid count mismatch")
        expanded_prefix = prefix
        merge_length = merge_size ** 2
        for grid_t, grid_h, grid_w in grids.tolist():
            count = grid_t * grid_h * grid_w // merge_length
            expanded_prefix = expanded_prefix.replace(
                "<|image_pad|>", "<|placeholder|>" * count, 1
            )
        # Keep expanded rows inert until every original camera placeholder has
        # been consumed; otherwise the next ``replace(..., 1)`` expands the first
        # image run again and shifts all following camera spans.
        expanded_prefix = expanded_prefix.replace(
            "<|placeholder|>", "<|image_pad|>"
        )
        tokenizer = processor.tokenizer
        tensor_type = None if getattr(tokenizer, "is_fast", False) else "pt"
        plan = getattr(processor, "_wall_prefix_plan", None)
        if (tensor_type is None and plan is not None and prefix.startswith(plan[0])
                and tokenizer.backend_tokenizer.padding is None
                and tokenizer.backend_tokenizer.truncation is None):
            # The split is immediately after an admitted atomic vision-end
            # token. Its no-strip metadata and backend guards preserve BPE
            # boundaries. Instruction/state and all following text stay live.
            encoded = tokenizer.backend_tokenizer.encode_batch(
                [prefix[len(plan[0]):] + postfix], add_special_tokens=False)[0]
            leading = []
            for chunk, (grid_t, grid_h, grid_w) in zip(plan[1], grids.tolist()):
                leading.extend(chunk)
                leading.extend([plan[2]] * (grid_t * grid_h * grid_w // merge_length))
            leading.extend(plan[1][-1])
            full = SimpleNamespace(
                input_ids=[leading + encoded.ids],
                attention_mask=[[1] * len(leading) + encoded.attention_mask])
        else:
            full = tokenizer(
                [expanded_prefix + postfix], return_tensors=tensor_type,
                padding=True, padding_side="right", truncation=False,
            )
        atomic_id = getattr(processor, "_wall_atomic_action_id", None)
        if (tensor_type is None and atomic_id is not None
                and full.input_ids[0][-ACTION_HORIZON:] == [atomic_id] * ACTION_HORIZON):
            # Cold admission proves the suffix tokens cannot consume prefix
            # whitespace or merge across the boundary. Check all 32 IDs live.
            prefix_length = len(full.input_ids[0]) - ACTION_HORIZON
        else:
            prefix_tokens = tokenizer(
                [expanded_prefix], return_tensors=tensor_type,
                padding=False, truncation=False,
            )
            prefix_length = (len(prefix_tokens.input_ids[0]) if tensor_type is None
                             else int(prefix_tokens.input_ids.shape[1]))
        if tensor_type is None:
            # Materialize known batch-one int64 arrays without recursive
            # Python flatten/dispatch in HF's generic conversion.
            full = SimpleNamespace(
                input_ids=torch.from_numpy(np.asarray(full.input_ids, dtype=np.int64)),
                attention_mask=torch.from_numpy(np.asarray(full.attention_mask, dtype=np.int64)))
        if prefix_length > MAX_ACTION_PREFIX_LENGTH:
            raise ValueError(
                "Wall Qwen3.5 initial RPU action profile admits at most "
                f"{MAX_ACTION_PREFIX_LENGTH} prefix tokens, got {prefix_length}"
            )
        logical_len = int(full.attention_mask.numpy().sum())
        if logical_len > int(max_seq_length):
            raise ValueError(
                f"inference input length {logical_len} exceeds profile {max_seq_length}"
            )
        if int(full.input_ids.shape[1]) - prefix_length != ACTION_HORIZON:
            raise ValueError("tokenizer did not encode each <|action|> as one token")

        image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        mm = torch.from_numpy((full.input_ids.numpy() == image_id).astype(np.int32))
        position_ids = None
        fusion = getattr(model, "model", model)
        compute = getattr(fusion, "compute_3d_position_ids", None)
        if callable(compute):
            position_ids = _compute_wall_positions(fusion, compute, full, mm, grids)
        if image_futures is not None:
            # Future order is camera then merge-row band, exactly the flattened
            # spatial patch order. No intermediate image assembly or second pool.
            for future in image_futures:
                future.result()
            image_inputs = {"pixel_values": pixel_tensor}
        action_pos = None if position_ids is None else position_ids[..., prefix_length:]
        return WallQwen35PreparedInput(
            input_ids=full.input_ids.contiguous(),
            attention_mask=full.attention_mask.contiguous(),
            mm_token_type_ids=mm.contiguous(),
            pixel_values=image_inputs["pixel_values"].contiguous(),
            image_grid_thw=grids.contiguous(),
            full_position_ids=position_ids,
            prefix_length=prefix_length,
            action_position_ids=action_pos,
            normalized_state=normalized.contiguous(),
            agent_pos_mask=apos_mask,
            dof_mask=action_mask,
            dataset_key=dataset_key,
        )
    finally:
        # On rejection, finish outstanding reads before returning control to a
        # caller that may immediately reuse/mutate its image buffers.
        if image_futures is not None:
            for future in image_futures:
                future.cancel()
            wait(image_futures)



__all__ = [
    "ACTION_HORIZON",
    "CHECKPOINT_VOCAB_SIZE",
    "DEFAULT_CAMERAS",
    "DEFAULT_ROBOT_ID",
    "MAX_ACTION_PREFIX_LENGTH",
    "WALL_PREFILL_GRAPH_MAX_ENTRIES",
    "WALL_PREFIX_BUCKETS",
    "MAX_SEQ_LENGTH",
    "PROFILE_MASK",
    "STATE_DIM",
    "WallQwen35PreparedInput",
    "build_prompt",
    "digitize_state",
    "extend_flow_tokenizer",
    "normalize_vector",
    "prepare_wall_qwen35_input",
    "resize_dataset_v2_image",
    "smart_resize",
]
