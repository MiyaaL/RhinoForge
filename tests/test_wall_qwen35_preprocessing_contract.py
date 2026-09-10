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


def _real_image_processor():
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessor,
    )
    return Qwen2VLImageProcessor(
        patch_size=16, merge_size=2, temporal_patch_size=2,
        image_mean=[0.5] * 3, image_std=[0.5] * 3, do_resize=False,
    )


def test_numpy_patches_exact_all_byte_values_mixed_grids_and_owned_output():
    import numpy as np
    ip = _real_image_processor()
    images = [
        Image.fromarray((np.arange(h * w * 3).reshape(h, w, 3) + i * 71)
                        .astype(np.uint8))
        for i, (h, w) in enumerate(((128, 224), (160, 224), (224, 128)))
    ]
    expected = ip(images=images, return_tensors="pt")
    actual = wall_preprocessing._numpy_image_inputs(ip, images)
    for key in expected:
        assert actual[key].dtype == expected[key].dtype
        assert torch.equal(actual[key], expected[key]), key
    snapshot = actual["pixel_values"].clone()
    images[0].paste((255, 0, 0), (0, 0, 224, 128))
    changed = wall_preprocessing._numpy_image_inputs(ip, images)
    assert not torch.equal(changed["pixel_values"], snapshot)
    assert torch.equal(actual["pixel_values"], snapshot)


@pytest.mark.parametrize("setting,value", [
    ("do_normalize", False), ("rescale_factor", 1.0 / 256.0),
    ("image_mean", [0.1, 0.2, 0.3]), ("temporal_patch_size", 1),
])
def test_unknown_image_settings_retain_reference_processor(setting, value):
    ip = _real_image_processor()
    setattr(ip, setting, value)
    images = [Image.new("RGB", (64, 32), (12, 34, 56))] * 3
    assert not wall_preprocessing._use_numpy_image_inputs(ip)
    expected = ip(images=images, return_tensors="pt")
    actual = wall_preprocessing._numpy_image_inputs(ip, images)
    for key in expected:
        assert torch.equal(actual[key], expected[key])


def test_parallel_images_equal_serial_and_finish_on_prompt_rejection(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    import time
    processor = _FakeProcessor(100)
    processor.image_processor = _real_image_processor()
    kwargs = dict(
        images={name: Image.new("RGB", (400, 300), (i * 20, 4, 8))
                for i, name in enumerate(wall_preprocessing.DEFAULT_CAMERAS)},
        instruction="pick", state=torch.zeros(26),
        state_min=torch.full((26,), -1.0), state_delta=torch.full((26,), 2.0),
        dataset_key="x2_normal",
    )
    serial = wall_preprocessing.prepare_wall_qwen35_input(processor, object(), **kwargs)
    with ThreadPoolExecutor(max_workers=3) as pool:
        parallel = wall_preprocessing.prepare_wall_qwen35_input(
            processor, object(), image_executor=pool, **kwargs)
        assert torch.equal(serial.pixel_values, parallel.pixel_values)
        assert torch.equal(serial.image_grid_thw, parallel.image_grid_thw)
        assert torch.equal(serial.input_ids, parallel.input_ids)
        finished = []
        original = wall_preprocessing._resize_camera
        def slow_resize(value):
            time.sleep(0.02)
            try:
                return original(value)
            finally:
                finished.append(threading.get_ident())
        monkeypatch.setattr(wall_preprocessing, "_resize_camera", slow_resize)
        processor.tokenizer = _LengthTokenizer(385)
        with pytest.raises(ValueError, match="at most 384"):
            wall_preprocessing.prepare_wall_qwen35_input(
                processor, object(), image_executor=pool, **kwargs)
        count = len(finished)
        time.sleep(0.04)
        assert len(finished) == count


@pytest.mark.parametrize("height,width", [(720, 1280), (1080, 1920), (480, 640), (481, 641)])
def test_parallel_bicubic_bands_preserve_full_image_halos(height, width):
    from concurrent.futures import ThreadPoolExecutor
    import numpy as np
    rng = np.random.default_rng(314)
    random = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    edges = np.zeros((height, width, 3), dtype=np.uint8)
    for center in (height // 4, height // 2, 3 * height // 4):
        edges[max(0, center - 3):center + 4, :, :] = 255
    checker = np.broadcast_to(
        (((np.indices((height, width)).sum(axis=0) % 2) * 255)[..., None]),
        (height, width, 3),
    ).astype(np.uint8)
    images = [Image.fromarray(a) for a in (random, edges, checker)]
    sizes = [wall_preprocessing._dataset_v2_size(image.size) for image in images]
    expected = [wall_preprocessing.resize_dataset_v2_image(image) for image in images]
    with ThreadPoolExecutor(max_workers=6) as pool:
        jobs, futures = wall_preprocessing._submit_image_resizes(images, sizes, pool)
        actual = wall_preprocessing._collect_image_resizes(jobs)
        assert all(future.done() for future in futures)
    for reference, candidate in zip(expected, actual):
        assert np.array_equal(np.asarray(reference), np.asarray(candidate))


def _position_only_model(model_class=None):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    cls = model_class or Qwen3_5Model
    model = cls.__new__(cls)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=2))
    model.rope_deltas = None
    return model


@pytest.mark.parametrize("tail", [1, 32, 100, 220])
def test_numpy_mrope_exact_and_updates_rope_delta(tail):
    model = _position_only_model()
    grids = torch.tensor([[1, 8, 14], [1, 10, 14], [1, 14, 14]])
    types = ([0] * 5 + [1] * 28 + [0] * 3 + [1] * 35
             + [0] * 7 + [1] * 49 + [0] * tail)
    mm = torch.tensor([types], dtype=torch.int32)
    full = SimpleNamespace(
        input_ids=torch.zeros((1, len(types)), dtype=torch.int64),
        attention_mask=torch.ones((1, len(types)), dtype=torch.int64),
    )
    reference = model.compute_3d_position_ids(
        full.input_ids, None, image_grid_thw=grids,
        attention_mask=full.attention_mask, mm_token_type_ids=mm,
    )
    reference_delta = model.rope_deltas.clone()
    model.rope_deltas = torch.tensor([[999]])
    actual = wall_preprocessing._compute_wall_positions(
        model, model.compute_3d_position_ids, full, mm, grids)
    assert torch.equal(actual, reference)
    assert torch.equal(model.rope_deltas, reference_delta)


def test_custom_vision_position_method_uses_reference_path():
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    class CustomModel(Qwen3_5Model):
        def get_vision_position_ids(self, *args, **kwargs):
            return super().get_vision_position_ids(*args, **kwargs) + 2
    model = _position_only_model(CustomModel)
    mm = torch.tensor([[0, 1, 1, 1, 1, 0]], dtype=torch.int32)
    grids = torch.tensor([[1, 4, 4]])
    full = SimpleNamespace(input_ids=torch.zeros((1, 6), dtype=torch.int64),
                           attention_mask=torch.ones((1, 6), dtype=torch.int64))
    expected = model.compute_3d_position_ids(
        full.input_ids, None, image_grid_thw=grids,
        attention_mask=full.attention_mask, mm_token_type_ids=mm)
    actual = wall_preprocessing._compute_wall_positions(
        model, model.compute_3d_position_ids, full, mm, grids)
    assert torch.equal(actual, expected)


def test_fast_tokenizer_direct_materialization_matches_hf_tensor_conversion():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|image_pad|>", "<|vision_start|>", "<|vision_end|>", "<|action|>",
    ]})
    tokenizer.pad_token = "[UNK]"
    processor = SimpleNamespace(image_processor=_FakeImageProcessor(), tokenizer=tokenizer)
    kwargs = dict(
        images={name: Image.new("RGB", (64, 64)) for name in wall_preprocessing.DEFAULT_CAMERAS},
        instruction="pick", state=torch.zeros(26),
        state_min=torch.full((26,), -1.0), state_delta=torch.full((26,), 2.0),
        dataset_key="x2_normal",
    )
    wall_preprocessing._configure_atomic_action_suffix(processor)
    assert processor._wall_atomic_action_id is not None
    fast = wall_preprocessing.prepare_wall_qwen35_input(processor, object(), **kwargs)
    class ReferenceTokenizer:
        is_fast = False
        def __call__(self, *args, **kwargs):
            return tokenizer(*args, **kwargs)
        def convert_tokens_to_ids(self, token):
            return tokenizer.convert_tokens_to_ids(token)
    processor.tokenizer = ReferenceTokenizer()
    reference = wall_preprocessing.prepare_wall_qwen35_input(processor, object(), **kwargs)
    for name in ("input_ids", "attention_mask", "mm_token_type_ids"):
        assert torch.equal(getattr(fast, name), getattr(reference, name))
    assert fast.prefix_length == reference.prefix_length


def test_direct_fp16_patches_match_reference_fp32_conversion():
    import numpy as np
    ip = _real_image_processor()
    images = [Image.fromarray(
        np.arange(128 * 224 * 3, dtype=np.uint8).reshape(128, 224, 3)
    )] * 3
    reference = ip(images=images, return_tensors="pt")["pixel_values"].half()
    candidate = wall_preprocessing._numpy_image_inputs(
        ip, images, dtype=torch.float16)["pixel_values"]
    assert candidate.dtype == torch.float16
    assert candidate.is_contiguous()
    assert torch.equal(candidate, reference)


def test_numpy_state_normalization_matches_eager_torch_and_retains_grad():
    generator = torch.Generator().manual_seed(82)
    for _ in range(30):
        state = torch.randn(26, generator=generator) * 10
        minimum = torch.randn(26, generator=generator)
        delta = torch.rand(26, generator=generator) + 0.01
        expected = (((state - minimum) / delta) * 2.0 - 1.0).clamp(-1, 1).view(1, 1, 26)
        actual = wall_preprocessing.normalize_vector(state, minimum, delta, name="state")
        assert torch.equal(actual, expected)
    state = torch.zeros(26, requires_grad=True)
    actual = wall_preprocessing.normalize_vector(state, torch.zeros(26), torch.ones(26), name="state")
    assert actual.requires_grad
    actual.sum().backward()
    assert state.grad is not None


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_resize_patches_match_complete_images(dtype):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.default_rng(91)
    images = [Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
              for h, w in [(720, 1280), (480, 640), (481, 641), (1080, 1920)]]
    sizes = [wall_preprocessing._dataset_v2_size(im.size) for im in images]
    reference = wall_preprocessing._numpy_image_inputs(
        _real_image_processor(),
        [wall_preprocessing.resize_dataset_v2_image(im) for im in images], dtype=dtype)
    with ThreadPoolExecutor(max_workers=6) as pool:
        actual = torch.empty_like(reference["pixel_values"])
        futures = wall_preprocessing._submit_image_patches(images, sizes, pool, dtype, actual.numpy())
        for future in futures:
            future.result()
    assert torch.equal(actual, reference["pixel_values"])
    retained = actual.clone()
    images[0].paste((0, 0, 0), (0, 0, images[0].width, images[0].height))
    assert torch.equal(actual, retained)


def test_digitize_state_negative_view_matches_materialized_values():
    value = torch.linspace(-1, 1, 26).reshape(1, 1, 26)
    negative = torch._neg_view(value)
    mask = torch.ones_like(value)
    assert wall_preprocessing.digitize_state(negative, mask) == wall_preprocessing.digitize_state(-value, mask)


def test_static_scaffold_matches_live_full_tokenization():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import ByteLevel
    from transformers import PreTrainedTokenizerFast
    backend = Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    backend.pre_tokenizer = ByteLevel(add_prefix_space=False)
    from tokenizers.normalizers import NFC
    backend.normalizer = NFC()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]")
    tokenizer.add_special_tokens({"additional_special_tokens": [
        "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>",
        "<|image_pad|>", "<|action|>",
    ]})
    processor = SimpleNamespace(image_processor=_real_image_processor(), tokenizer=tokenizer)
    wall_preprocessing._configure_atomic_action_suffix(processor)
    plan = processor._wall_prefix_plan
    assert plan is not None
    for text, shift in [("pick up cup", 0.0), ("move! <|im_start|> then stop e\u0301", 0.2)]:
        kwargs = dict(
            images={name: Image.new("RGB", (64, 64)) for name in wall_preprocessing.DEFAULT_CAMERAS},
            instruction=text, state=torch.full((26,), shift),
            state_min=torch.full((26,), -1.0), state_delta=torch.full((26,), 2.0),
            dataset_key="x2_normal",
        )
        processor._wall_prefix_plan = plan
        tokenizer.backend_tokenizer.no_padding()
        tokenizer.backend_tokenizer.no_truncation()
        actual = wall_preprocessing.prepare_wall_qwen35_input(processor, object(), **kwargs)
        processor._wall_prefix_plan = None
        reference = wall_preprocessing.prepare_wall_qwen35_input(processor, object(), **kwargs)
        for name in ("input_ids", "attention_mask", "mm_token_type_ids", "normalized_state"):
            assert torch.equal(getattr(actual, name), getattr(reference, name))
        assert actual.prefix_length == reference.prefix_length
