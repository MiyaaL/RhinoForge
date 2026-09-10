"""Board-free checks for Wall's RPU-specific startup diagnostics."""

import inspect
import logging
from pathlib import Path
import runpy
import threading
import tomllib
from types import SimpleNamespace

import pytest
import torch

from rpu_backend.adapters.wall_qwen35 import checkpoint as wall_checkpoint
from rpu_backend.adapters.wall_qwen35 import runtime as wall_runtime


ROOT = Path(__file__).resolve().parents[1]
HF_LOGGER = "transformers.models.qwen3_5.modeling_qwen3_5"
CUDA_HINT = (
    "The fast path is not available because one of the required library is not installed. Falling back to "
    "torch implementation. To install follow https://github.com/fla-org/flash-linear-attention#installation and"
    " https://github.com/Dao-AILab/causal-conv1d"
)


@pytest.mark.parametrize("fail", [False, True])
def test_meta_construction_filters_only_unused_cuda_hint(caplog, monkeypatch, fail):
    logger = logging.getLogger(HF_LOGGER)
    monkeypatch.setattr(logger, "propagate", True)
    previous_filters = list(logger.filters)
    config = object()

    def construct(received):
        assert received is config
        result = torch.empty(1)
        assert result.device.type == "meta"
        logger.warning(CUDA_HINT)
        logger.warning("keep other Qwen warnings")
        logger.error(CUDA_HINT)
        logging.getLogger("other.model").warning(CUDA_HINT)
        worker = threading.Thread(target=logger.warning, args=(CUDA_HINT,))
        worker.start()
        worker.join()
        if fail:
            raise ValueError("construction failed")
        return result

    with caplog.at_level(logging.WARNING):
        if fail:
            with pytest.raises(ValueError, match="construction failed"):
                wall_checkpoint._build_wall_qwen35_meta_model(config, construct)
        else:
            result = wall_checkpoint._build_wall_qwen35_meta_model(config, construct)
            assert result.device.type == "meta"
        assert logger.filters == previous_filters
        assert torch.empty(1).device.type == "cpu"
        logger.warning(CUDA_HINT)  # Outside Wall construction this remains visible.

    assert [(r.name, r.levelno, r.getMessage()) for r in caplog.records] == [
        (HF_LOGGER, logging.WARNING, "keep other Qwen warnings"),
        (HF_LOGGER, logging.ERROR, CUDA_HINT),
        ("other.model", logging.WARNING, CUDA_HINT),
        (HF_LOGGER, logging.WARNING, CUDA_HINT),  # Other thread.
        (HF_LOGGER, logging.WARNING, CUDA_HINT),  # Outside the construction scope.
    ]


def test_actual_hf_gated_delta_constructor_keeps_torch_fallback(caplog, monkeypatch):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as hf

    monkeypatch.setattr(hf.logger, "propagate", True)
    # Avoid warning_once's cross-test cache; exercise the real constructor text.
    monkeypatch.setattr(hf.logger, "warning_once", hf.logger.warning)
    monkeypatch.setattr(hf, "is_fast_path_available", False)
    monkeypatch.setattr(hf, "FusedRMSNormGated", None)
    config = hf.Qwen3_5TextConfig(
        hidden_size=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
    )
    with caplog.at_level(logging.WARNING):
        model = wall_checkpoint._build_wall_qwen35_meta_model(
            config, lambda c: hf.Qwen3_5GatedDeltaNet(c, layer_idx=0)
        )
    assert CUDA_HINT not in caplog.messages
    assert all(p.device.type == "meta" for p in model.parameters())
    assert hf.is_fast_path_available is False
    assert model.chunk_gated_delta_rule is (
        hf.chunk_gated_delta_rule or hf.torch_chunk_gated_delta_rule
    )


def test_runtime_processor_uses_non_deprecated_equivalent_backend(monkeypatch):
    from transformers import AutoProcessor

    class Tokenizer:
        def __len__(self):
            return wall_runtime.CHECKPOINT_VOCAB_SIZE

        def convert_tokens_to_ids(self, token):
            return 248077 + wall_runtime.SPECIAL_TOKENS.index(token)

    calls = []
    extensions = []
    processor = SimpleNamespace(
        tokenizer=Tokenizer(), image_processor=SimpleNamespace(backend="torchvision")
    )

    def load(*args, **kwargs):
        calls.append((args, kwargs))
        return processor

    monkeypatch.setattr(AutoProcessor, "from_pretrained", load)
    module = SimpleNamespace(
        _load_wall_qwen35_hf_processor=wall_checkpoint._load_wall_qwen35_hf_processor,
        extend_wall_qwen35_tokenizer=lambda *a, **kw: extensions.append((a, kw))
    )
    path = Path("local-checkpoint")
    assert wall_runtime._load_processor(path, module) is processor
    assert calls == [((path,), {"local_files_only": True})]
    assert extensions == [((processor.tokenizer,), {"target_vocab": 256277})]
    assert processor.tokenizer.padding_side == "right"


@pytest.mark.parametrize("backend", ["pil", None])
def test_processor_rejects_silent_backend_change(monkeypatch, backend):
    from transformers import AutoProcessor

    processor = SimpleNamespace(image_processor=SimpleNamespace(backend=backend))
    monkeypatch.setattr(AutoProcessor, "from_pretrained", lambda *a, **kw: processor)
    with pytest.raises(RuntimeError, match="requires the torchvision image backend"):
        wall_checkpoint._load_wall_qwen35_hf_processor(Path("local-checkpoint"))


def test_preflight_and_stream_loader_share_rpu_meta_construction():
    namespace = runpy.run_path(str(ROOT / "examples/wall_qwen35_openloop.py"))
    source = inspect.getsource(namespace["_host_prefix_preflight"])
    assert "_load_wall_qwen35_hf_processor(checkpoint)" in source
    assert "use_fast=" not in source
    assert "_build_wall_qwen35_meta_model(config, Qwen3_5ForConditionalGeneration)" in source
    source = inspect.getsource(wall_checkpoint.stream_load_wall_qwen35_base_model)
    assert "_build_wall_qwen35_meta_model(config, model_class)" in source


def test_vla_dependencies_include_headless_opencv_not_cuda_fast_paths():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert "opencv-python-headless>=4.11,<5" in project["optional-dependencies"]["vla"]
    requirements = project["dependencies"] + [
        item for extra in project["optional-dependencies"].values() for item in extra
    ]
    assert not any("flash-linear-attention" in r or "causal-conv1d" in r for r in requirements)
