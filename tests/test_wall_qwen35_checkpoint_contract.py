from __future__ import annotations

import copy
import dataclasses
import hashlib
from pathlib import Path

import pytest

from rpu_backend.adapters.wall_qwen35 import checkpoint as wall_checkpoint
from rpu_backend.api.errors import UnsupportedModelError


def _exact_config() -> dict:
    return {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_hidden_layers": 24,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "head_dim": 256,
            "attention_bias": False,
            "attn_output_gate": True,
            "hidden_act": "silu",
            "rms_norm_eps": 1.0e-6,
            "layer_types": [
                "full_attention" if index % 4 == 3 else "linear_attention"
                for index in range(24)
            ],
            "rope_parameters": {
                "rope_theta": 10_000_000.0,
                "rope_type": "default",
                "mrope_section": [11, 11, 10],
                "mrope_interleaved": True,
                "partial_rotary_factor": 0.25,
            },
        },
        "vision_config": {
            "depth": 24,
            "hidden_size": 1024,
            "intermediate_size": 4096,
            "out_hidden_size": 2048,
            "num_heads": 16,
        },
        "vla_config": {
            "dim_inputs": [2048, 1024],
            "action_hidden_size": 1024,
            "state_hidden_size": 2560,
            "norm_moe": True,
            "mlp_moe": True,
            "use_mot": True,
            "use_adarms": True,
            "adarms_cond_dim": 1024,
            "action_linear_attention": False,
            "action_mlp_intermediate_size": 2048,
            "use_state_string_representation": True,
            "causal_action_attention_mask": False,
            "noise_scheduler": {
                "beta_alpha": 1.5,
                "beta_beta": 1.0,
                "s": 0.999,
                "num_inference_timesteps": 10,
            },
        },
    }


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("text_config", "hidden_size", 1024),
        ("text_config", "num_key_value_heads", 8),
        ("text_config", "layer_types", ["full_attention"] * 24),
        ("vision_config", "num_heads", 8),
        ("vla_config", "action_linear_attention", True),
    ],
)
def test_exact_checkpoint_profile_rejects_architecture_drift(
    section: str,
    field: str,
    value,
) -> None:
    config = _exact_config()
    wall_checkpoint._check_config_profile(config)

    drifted = copy.deepcopy(config)
    drifted[section][field] = value
    with pytest.raises(UnsupportedModelError, match="exact TR-038"):
        wall_checkpoint._check_config_profile(drifted)


def test_exact_checkpoint_profile_rejects_scheduler_drift() -> None:
    config = _exact_config()
    config["vla_config"]["noise_scheduler"]["num_inference_timesteps"] = 11
    with pytest.raises(UnsupportedModelError, match="10-step scheduler"):
        wall_checkpoint._check_config_profile(config)


def _tiny_manifest(
    root: Path,
    name: str,
    digest: str,
    *,
    token: bool,
    tensor_count: int | None = None,
) -> wall_checkpoint.WallQwen35CheckpointManifest:
    stat = wall_checkpoint._file_identity(root / name)
    manifest = wall_checkpoint.WallQwen35CheckpointManifest(
        root=root.resolve(),
        file_sha256=((name, digest),),
        file_stats=((name, *stat),),
        tensor_count=(
            wall_checkpoint._EXPECTED_TENSOR_COUNT
            if tensor_count is None
            else tensor_count
        ),
        tensor_manifest_sha256=(
            wall_checkpoint._EXPECTED_TENSOR_MANIFEST_SHA256
        ),
        action_tensor_count=wall_checkpoint._EXPECTED_ACTION_TENSOR_COUNT,
        action_manifest_sha256=wall_checkpoint._EXPECTED_ACTION_MANIFEST_SHA256,
        base_tensor_count=wall_checkpoint._EXPECTED_BASE_TENSOR_COUNT,
        base_manifest_sha256=wall_checkpoint._EXPECTED_BASE_MANIFEST_SHA256,
    )
    if token:
        object.__setattr__(
            manifest, "_admission_token", wall_checkpoint._ADMISSION_TOKEN
        )
    return manifest


def test_manifest_capability_reuses_preflight_without_rehashing_10gb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"small profile fixture"
    asset = tmp_path / "tiny.bin"
    asset.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        wall_checkpoint, "_EXPECTED_FILE_SHA256", {asset.name: digest}
    )
    manifest = _tiny_manifest(
        tmp_path, asset.name, digest, token=True
    )
    monkeypatch.setattr(
        wall_checkpoint,
        "preflight_wall_qwen35_checkpoint",
        lambda _path: pytest.fail("an admitted manifest must not rehash assets"),
    )

    assert wall_checkpoint._admit_manifest(tmp_path, manifest) is manifest

    asset.write_bytes(payload + b" changed")
    with pytest.raises(RuntimeError, match="changed after preflight"):
        wall_checkpoint._admit_manifest(tmp_path, manifest)


def test_manifest_rejects_hand_constructed_and_profile_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "tiny.bin"
    asset.write_bytes(b"fixture")
    digest = hashlib.sha256(asset.read_bytes()).hexdigest()
    monkeypatch.setattr(
        wall_checkpoint, "_EXPECTED_FILE_SHA256", {asset.name: digest}
    )

    forged = _tiny_manifest(tmp_path, asset.name, digest, token=False)
    with pytest.raises(ValueError, match="was not issued by preflight"):
        wall_checkpoint._admit_manifest(tmp_path, forged)

    drifted = _tiny_manifest(
        tmp_path,
        asset.name,
        digest,
        token=True,
        tensor_count=wall_checkpoint._EXPECTED_TENSOR_COUNT - 1,
    )
    with pytest.raises(ValueError, match="tensor profile"):
        wall_checkpoint._admit_manifest(tmp_path, drifted)


class _FakeTokenizer:
    def __init__(self) -> None:
        self._length = wall_checkpoint._BASE_TOKENIZER_SIZE
        self._token_to_id: dict[str, int] = {}
        self._id_to_token: dict[int, str] = {}
        self._special: set[str] = set()
        self.special_calls: list[bool] = []
        self.plain_calls: list[bool] = []
        first_audio = self._length - len(wall_checkpoint._AUDIO_TAIL)
        for offset, token in enumerate(wall_checkpoint._AUDIO_TAIL):
            self._token_to_id[token] = first_audio + offset
            self._id_to_token[first_audio + offset] = token
            self._special.add(token)

    def __len__(self) -> int:
        return self._length

    @property
    def all_special_tokens(self) -> list[str]:
        return sorted(self._special)

    def convert_ids_to_tokens(self, token_id: int) -> str | None:
        return self._id_to_token.get(token_id)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._token_to_id.get(token)

    def add_special_tokens(
        self,
        values: dict[str, list[str]],
        *,
        replace_extra_special_tokens: bool,
    ) -> int:
        self.special_calls.append(replace_extra_special_tokens)
        return self._add(values["additional_special_tokens"], special=True)

    def add_tokens(self, values: list[str], *, special_tokens: bool = False) -> int:
        self.plain_calls.append(special_tokens)
        return self._add(values, special=special_tokens)

    def _add(self, values: list[str], *, special: bool) -> int:
        added = 0
        for token in values:
            if token in self._token_to_id:
                continue
            token_id = self._length
            self._length += 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
            if special:
                self._special.add(token)
            added += 1
        return added


def test_tokenizer_exact_six_special_and_8194_plain_token_abi() -> None:
    tokenizer = _FakeTokenizer()
    result = wall_checkpoint.extend_wall_qwen35_tokenizer(tokenizer)

    assert result is tokenizer
    assert len(tokenizer) == wall_checkpoint.CHECKPOINT_VOCAB_SIZE
    assert tokenizer.special_calls == [False]
    assert tokenizer.plain_calls == [False]
    for offset, token in enumerate(wall_checkpoint._FLOW_SPECIAL_TOKENS):
        assert tokenizer.convert_tokens_to_ids(token) == 248_077 + offset
        assert token in tokenizer.all_special_tokens
    reserve_start = 248_077 + 6
    assert tokenizer.convert_tokens_to_ids("<|flow_only_reserved_0|>") == reserve_start
    assert (
        tokenizer.convert_tokens_to_ids("<|flow_only_reserved_8193|>")
        == 256_276
    )
    assert not any(
        token.startswith("<|flow_only_reserved_")
        for token in tokenizer.all_special_tokens
    )

    wall_checkpoint.extend_wall_qwen35_tokenizer(tokenizer)
    assert tokenizer.special_calls == [False]
    assert tokenizer.plain_calls == [False]


def test_tokenizer_rejects_reserved_token_marked_special() -> None:
    tokenizer = _FakeTokenizer()
    wall_checkpoint.extend_wall_qwen35_tokenizer(tokenizer)
    tokenizer._special.add("<|flow_only_reserved_0|>")

    with pytest.raises(ValueError, match="reserved.*special|plain"):
        wall_checkpoint.extend_wall_qwen35_tokenizer(tokenizer)


def test_tokenizer_checks_frozen_audio_tail_before_mutation() -> None:
    tokenizer = _FakeTokenizer()
    bad_id = wall_checkpoint._BASE_TOKENIZER_SIZE - 1
    tokenizer._id_to_token[bad_id] = "not-the-audio-tail"

    with pytest.raises(ValueError, match="base tokenizer ID"):
        wall_checkpoint.extend_wall_qwen35_tokenizer(tokenizer)
    assert len(tokenizer) == wall_checkpoint._BASE_TOKENIZER_SIZE
    assert tokenizer.special_calls == []
    assert tokenizer.plain_calls == []


def test_checkpoint_manifest_constants_are_frozen() -> None:
    assert wall_checkpoint._EXPECTED_TENSOR_COUNT == 1_090
    assert wall_checkpoint._EXPECTED_ACTION_TENSOR_COUNT == 497
    assert wall_checkpoint._EXPECTED_BASE_TENSOR_COUNT == 593
    assert (
        wall_checkpoint._EXPECTED_ACTION_TENSOR_COUNT
        + wall_checkpoint._EXPECTED_BASE_TENSOR_COUNT
        == wall_checkpoint._EXPECTED_TENSOR_COUNT
    )
    assert wall_checkpoint._EXPECTED_FILE_SHA256["model.safetensors"] == (
        "82f9d583bf33f713c941b1c3aac60a6be5ada9eb02cf8a04f9f21e61bc0e7b2c"
    )
    assert dataclasses.fields(wall_checkpoint.WallQwen35CheckpointManifest)[-1].init is False
