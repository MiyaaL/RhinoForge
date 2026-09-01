from __future__ import annotations

import bisect
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from rpu_backend.adapters.wall_qwen35 import preprocessing as wall_preprocessing


def test_digitize_state_matches_numpy_right_false_boundaries_and_mask() -> None:
    step = 2.0 / 256.0
    values = torch.tensor(
        [-1.0, -1.0 + step, 0.0, 1.0] + [0.25] * 22,
        dtype=torch.float32,
    ).view(1, 1, 26)
    mask = torch.tensor([1, 1, 1, 1] + [0] * 22).view(1, 1, 26)

    assert wall_preprocessing.digitize_state(values, mask) == "0 1 128 255"

    boundaries = torch.linspace(-1.0, 1.0, 257)[:-1].tolist()
    reference = [
        bisect.bisect_right(boundaries, float(value)) - 1
        for value in values.reshape(-1)
    ]
    full_mask = torch.ones((1, 1, 26), dtype=torch.float32)
    actual = [
        int(value)
        for value in wall_preprocessing.digitize_state(values, full_mask).split()
    ]
    assert actual == reference

    empty_mask = torch.zeros((1, 1, 26), dtype=torch.float32)
    assert wall_preprocessing.digitize_state(values, empty_mask) == ""


def test_prompt_freezes_camera_order_embodiment_and_action_suffix() -> None:
    prefix, postfix = wall_preprocessing.build_prompt(
        "  pick   the block  ",
        state_string="0 1 2",
        dataset_key="x2_normal",
    )

    assert "Embodiment: X2 ex001_desktop" in prefix
    assert "Frequency: 32HZ" in prefix
    assert "Action Space: Rel EEF" in prefix
    assert "Task: pick the block" in prefix
    assert "Proprioception: 0 1 2" in prefix
    assert prefix.endswith("<|im_start|>assistant\n")
    assert prefix.count("<|vision_start|><|image_pad|><|vision_end|>") == 3
    assert prefix.index("front view") < prefix.index("left wrist view")
    assert prefix.index("left wrist view") < prefix.index("right wrist view")
    assert postfix == "<|action|>" * 32

    mapped, _ = wall_preprocessing.build_prompt(
        {"task": "move", "detail": "slowly"}, state_string=""
    )
    assert "Embodiment: X2 ex001_desktop" in mapped
    assert "Task: move\nInstruction: slowly" in mapped


def test_prompt_rejects_out_of_profile_inputs() -> None:
    with pytest.raises(ValueError, match="camera order"):
        wall_preprocessing.build_prompt(
            "task",
            tuple(reversed(wall_preprocessing.DEFAULT_CAMERAS)),
            state_string="0",
        )
    with pytest.raises(ValueError, match="horizon must be 32"):
        wall_preprocessing.build_prompt(
            "task", state_string="0", action_horizon=31
        )
    with pytest.raises(ValueError, match="dataset_key"):
        wall_preprocessing.build_prompt(
            "task", state_string="0", dataset_key="unknown"
        )
    with pytest.raises(ValueError, match="robot_id='10070'"):
        wall_preprocessing.build_prompt(
            "task", state_string="0", robot_id="10071"
        )
    with pytest.raises(ValueError, match="256-word"):
        wall_preprocessing.build_prompt(
            " ".join(["word"] * 257), state_string="0"
        )


def test_dataset_v2_resize_is_one_rgb_bicubic_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_resize = Image.Image.resize
    calls: list[tuple[tuple[int, int], Image.Resampling]] = []

    def tracked_resize(self, size, resample=None, box=None, reducing_gap=None):
        calls.append((tuple(size), resample))
        return original_resize(
            self,
            size,
            resample=resample,
            box=box,
            reducing_gap=reducing_gap,
        )

    monkeypatch.setattr(Image.Image, "resize", tracked_resize)
    image = Image.new("RGBA", (400, 300), (12, 34, 56, 78))
    result = wall_preprocessing.resize_dataset_v2_image(image)

    assert result.mode == "RGB"
    assert result.size == (224, 160)
    assert calls == [((224, 160), Image.Resampling.BICUBIC)]


def test_smart_resize_enforces_factor_and_aspect_envelope() -> None:
    height, width = wall_preprocessing.smart_resize(168, 224)
    assert (height, width) == (160, 224)
    assert height % 32 == width % 32 == 0
    assert height * width >= 4 * 32 * 32

    with pytest.raises(ValueError, match="aspect ratio"):
        wall_preprocessing.smart_resize(1, 201)
    with pytest.raises(ValueError, match="positive"):
        wall_preprocessing.smart_resize(0, 224)


class _FakeImageProcessor:
    patch_size = 16
    merge_size = 2
    do_resize = True

    def __call__(self, *, images, return_tensors):
        assert return_tensors == "pt"
        assert len(images) == 3
        assert all(image.mode == "RGB" for image in images)
        assert self.do_resize is False
        return {
            "image_grid_thw": torch.tensor(
                [[1, 2, 2], [1, 2, 2], [1, 2, 2]], dtype=torch.int64
            ),
            "pixel_values": torch.zeros((3, 4, 8), dtype=torch.float32),
        }


class _LengthTokenizer:
    def __init__(self, prefix_length: int) -> None:
        self.prefix_length = prefix_length

    def __call__(
        self,
        values,
        *,
        return_tensors,
        padding,
        truncation,
        padding_side=None,
    ):
        assert return_tensors == "pt"
        assert truncation is False
        assert len(values) == 1
        suffix = 32 if "<|action|>" in values[0] else 0
        length = self.prefix_length + suffix
        return SimpleNamespace(
            input_ids=torch.zeros((1, length), dtype=torch.int64),
            attention_mask=torch.ones((1, length), dtype=torch.int64),
        )

    @staticmethod
    def convert_tokens_to_ids(token: str) -> int:
        assert token == "<|image_pad|>"
        return 99


class _FakeProcessor:
    def __init__(self, prefix_length: int) -> None:
        self.image_processor = _FakeImageProcessor()
        self.tokenizer = _LengthTokenizer(prefix_length)


def _prepare_with_prefix(prefix_length: int):
    images = {
        name: Image.new("RGB", (400, 300), (index, index, index))
        for index, name in enumerate(wall_preprocessing.DEFAULT_CAMERAS)
    }
    return wall_preprocessing.prepare_wall_qwen35_input(
        _FakeProcessor(prefix_length),
        object(),
        images=images,
        instruction="pick the block",
        state=torch.zeros(26),
        state_min=torch.full((26,), -1.0),
        state_delta=torch.full((26,), 2.0),
        dataset_key="x2_normal",
        agent_pos_mask=torch.tensor([1] * 20 + [0] * 6),
        dof_mask=torch.tensor([1] * 20 + [0] * 6),
    )


def test_prepare_accepts_prefix_384_and_preserves_padded_aux_masks() -> None:
    prepared = _prepare_with_prefix(384)

    assert prepared.prefix_length == 384
    assert tuple(prepared.input_ids.shape) == (1, 416)
    assert tuple(prepared.prefix_input_ids.shape) == (1, 384)
    assert prepared.agent_pos_mask[0, 0, :20].tolist() == [1.0] * 20
    assert prepared.agent_pos_mask[0, 0, 20:].tolist() == [0.0] * 6
    assert torch.equal(prepared.agent_pos_mask, prepared.dof_mask)
    assert prepared.full_position_ids is None
    assert prepared.action_position_ids is None


def test_prepare_rejects_prefix_385_before_model_execution() -> None:
    with pytest.raises(
        ValueError,
        match="initial RPU action profile admits at most 384 prefix tokens, got 385",
    ):
        _prepare_with_prefix(385)


def test_each_camera_placeholder_is_expanded_once_for_distinct_grids() -> None:
    class VariableGridImageProcessor:
        patch_size = 16
        merge_size = 2
        do_resize = True

        def __call__(self, *, images, return_tensors):
            assert len(images) == 3 and return_tensors == "pt"
            assert self.do_resize is False
            return {
                # merge_length=4, hence the three image-token counts are 1/2/4.
                "image_grid_thw": torch.tensor(
                    [[1, 2, 2], [1, 2, 4], [1, 4, 4]], dtype=torch.int64
                ),
                "pixel_values": torch.zeros((7, 4, 8), dtype=torch.float32),
            }

    class RecordingTokenizer(_LengthTokenizer):
        def __init__(self) -> None:
            super().__init__(prefix_length=100)
            self.texts: list[str] = []

        def __call__(self, values, **kwargs):
            self.texts.append(values[0])
            return super().__call__(values, **kwargs)

    processor = SimpleNamespace(
        image_processor=VariableGridImageProcessor(),
        tokenizer=RecordingTokenizer(),
    )
    images = {
        name: Image.new("RGB", (400, 300))
        for name in wall_preprocessing.DEFAULT_CAMERAS
    }
    wall_preprocessing.prepare_wall_qwen35_input(
        processor,
        object(),
        images=images,
        instruction="pick",
        state=torch.zeros(26),
        state_min=torch.full((26,), -1.0),
        state_delta=torch.full((26,), 2.0),
        dataset_key="x2_normal",
    )

    expanded_prefix = processor.tokenizer.texts[1]
    pad = "<|image_pad|>"
    vision_start = "<|vision_start|>"
    vision_end = "<|vision_end|>"
    assert f"front view: {vision_start}{pad}{vision_end}" in expanded_prefix
    assert (
        f"left wrist view: {vision_start}{pad * 2}{vision_end}"
        in expanded_prefix
    )
    assert (
        f"right wrist view: {vision_start}{pad * 4}{vision_end}"
        in expanded_prefix
    )
    assert expanded_prefix.count(pad) == 7
    assert "<|placeholder|>" not in expanded_prefix
    assert processor.tokenizer.texts[0] == (
        expanded_prefix + "<|action|>" * wall_preprocessing.ACTION_HORIZON
    )
