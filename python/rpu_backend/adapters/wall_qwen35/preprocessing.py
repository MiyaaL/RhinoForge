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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    image = image.convert("RGB")
    width, height = image.size
    if target > 0:
        if width > height:
            width, height = target, max(1, int(target * height / width))
        else:
            width, height = max(1, int(target * width / height)), target
    out_h, out_w = smart_resize(
        height, width, factor=factor,
        min_pixels=min_pixels, max_pixels=max_pixels,
    )
    return image.resize((out_w, out_h), Image.Resampling.BICUBIC)


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
    tensor = torch.as_tensor(value, dtype=torch.float32, device="cpu")
    if tuple(tensor.shape) == (STATE_DIM,):
        tensor = tensor.view(1, 1, STATE_DIM)
    elif tuple(tensor.shape) == (1, STATE_DIM):
        tensor = tensor.unsqueeze(1)
    if tuple(tensor.shape) != (1, 1, STATE_DIM):
        raise ValueError(f"{name} must have shape [26], [1,26], or [1,1,26]")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    minimum = minimum.detach().to(device="cpu", dtype=torch.float32).reshape(1, 1, -1)
    delta = delta.detach().to(device="cpu", dtype=torch.float32).reshape(1, 1, -1)
    if tuple(minimum.shape[-1:]) != (STATE_DIM,) or tuple(delta.shape[-1:]) != (STATE_DIM,):
        raise ValueError(f"{name} normalizer must contain 26 values")
    if bool((delta <= 0).any()):
        raise ValueError(f"{name} normalizer delta must be positive")
    return (((tensor - minimum) / delta) * 2.0 - 1.0).clamp(-1.0, 1.0)


def digitize_state(
    normalized_state: torch.Tensor,
    agent_pos_mask: torch.Tensor,
    *,
    bins: int = STATE_BINS,
) -> str:
    """Exact np.digitize branch used by x2robot_utils.preprocesser_call."""

    state = normalized_state.detach().to(device="cpu", dtype=torch.float32)
    mask = agent_pos_mask.detach().to(device="cpu", dtype=torch.bool)
    if tuple(state.shape) != (1, 1, STATE_DIM) or tuple(mask.shape) != (1, 1, STATE_DIM):
        raise ValueError("normalized state and agent_pos_mask must be [1,1,26]")
    boundaries = torch.linspace(-1.0, 1.0, int(bins) + 1)[:-1]
    # np.digitize(..., right=False) is torch.bucketize(..., right=True).
    discretized = torch.bucketize(state, boundaries, right=True) - 1
    return " ".join(map(str, discretized[0, 0, mask[0, 0]].tolist()))


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
    if not bool(((tensor == 0) | (tensor == 1)).all()):
        raise ValueError(f"{name} entries must be zero or one")
    expected = torch.tensor(PROFILE_MASK, dtype=torch.float32).view(1, 1, STATE_DIM)
    if not torch.equal(tensor, expected):
        raise ValueError(
            f"{name} must match the exact X2 profile [1]*20 + [0]*6"
        )
    return tensor.contiguous()


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

    ordered_images = []
    for name in camera_names:
        if name not in images:
            raise KeyError(f"missing camera image {name!r}")
        value = images[name]
        if isinstance(value, (str, Path)):
            with Image.open(value) as opened:
                image = opened.convert("RGB").copy()
        else:
            image = value
        ordered_images.append(resize_dataset_v2_image(image))

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
    image_inputs = image_processor(images=ordered_images, return_tensors="pt")
    grids = image_inputs["image_grid_thw"]
    if prefix.count("<|image_pad|>") != len(grids):
        raise ValueError("image placeholder/grid count mismatch")
    expanded_prefix = prefix
    merge_length = merge_size ** 2
    for grid in grids:
        count = int(grid.prod().item()) // merge_length
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
    full = tokenizer(
        [expanded_prefix + postfix], return_tensors="pt",
        padding=True, padding_side="right", truncation=False,
    )
    prefix_tokens = tokenizer(
        [expanded_prefix], return_tensors="pt",
        padding=False, truncation=False,
    )
    prefix_length = int(prefix_tokens.input_ids.shape[1])
    if prefix_length > MAX_ACTION_PREFIX_LENGTH:
        raise ValueError(
            "Wall Qwen3.5 initial RPU action profile admits at most "
            f"{MAX_ACTION_PREFIX_LENGTH} prefix tokens, got {prefix_length}"
        )
    logical_len = int(full.attention_mask.sum().item())
    if logical_len > int(max_seq_length):
        raise ValueError(
            f"inference input length {logical_len} exceeds profile {max_seq_length}"
        )
    if int(full.input_ids.shape[1]) - prefix_length != ACTION_HORIZON:
        raise ValueError("tokenizer did not encode each <|action|> as one token")

    mm = torch.zeros_like(full.input_ids, dtype=torch.int32)
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    mm.masked_fill_(full.input_ids == image_id, 1)
    position_ids = None
    fusion = getattr(model, "model", model)
    compute = getattr(fusion, "compute_3d_position_ids", None)
    if callable(compute):
        position_ids = compute(
            input_ids=full.input_ids,
            inputs_embeds=None,
            image_grid_thw=grids,
            video_grid_thw=None,
            attention_mask=full.attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm,
        )
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
