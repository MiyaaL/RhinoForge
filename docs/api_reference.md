# API reference

[简体中文](api_reference.zh.md) | English

The distribution is named `rhinoforge`; its Python package remains
`rpu_backend`. Importing the package registers PyTorch's `PrivateUse1` device as
`rpu` and creates the `torch.rpu` namespace.

Support is profile-specific. An importable class, adapter, registry alias, or
native schema does not by itself make a model supported. Check
[Model support](model_support.md) before selecting an entry point.

## Top-level package

`rpu_backend.__all__` contains exactly four names:

| Name | Contract |
|---|---|
| `__version__` | Installed RhinoForge distribution version |
| `RPUCache` | FP16 KV cache for fused causal attention |
| `RPUModelForCausalLM` | Loader for registered decoder-only CausalLM profiles |
| `reset_graph_cache()` | Clear the process graph cache; a no-op without the native backend |

The API compatibility level is available as `rpu_backend.api.API_VERSION` and
currently reports `5.0.0`; it is independent of the distribution version.

## Causal language models

```python
import torch

from rpu_backend import RPUCache, RPUModelForCausalLM

model = RPUModelForCausalLM.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",              # omit to load on CPU, then call model.to("rpu")
    rpu_execution={
        "prefill": {
            "chunk_size": "auto",
            "padding_rows": "auto",
            "padding_budget": 64,
        }
    },
)
cache = RPUCache.from_model(
    model,
    input_ids=input_ids,
    max_new_tokens=32,
)
```

### `RPUModelForCausalLM.from_pretrained`

```python
RPUModelForCausalLM.from_pretrained(
    hf_repo_or_path,
    *,
    dtype=torch.float16,
    device=None,
    rpu_execution=None,
    **hf_kwargs,
)
```

The loader reads the Hugging Face configuration before the full weight load,
selects an adapter by `config.architectures[0]`, and rejects unsupported
profiles early. It currently admits the built-in Qwen3 and Llama CausalLM
architectures. Only `torch.float16` is accepted by this entry point; a
quantized checkpoint carries its own checked metadata.

`rpu_execution` is a read-only per-handle mapping. The CausalLM entry accepts a
`prefill` stage with:

- `chunk_size`: `"auto"` or a positive multiple of 16;
- `padding_rows`: `"auto"` or a non-negative exact row count; and
- `padding_budget`: a non-negative integer, mutually exclusive with an exact
  `padding_rows` value.

The adapter may narrow these values to the exact profile envelope. Execution
settings are cold: create a new model to change them after RPU installation.

The generic CausalLM and image-text loaders, and `Pi05Policy.from_pretrained`,
support only built-in model code. `trust_remote_code` must be exactly `False`;
custom Hugging Face model code is rejected before configuration or weight
loading.

### `RPUCache`

Prefer `RPUCache.from_model` over the low-level constructor:

```python
RPUCache.from_model(
    model,
    *,
    input_ids=None,
    max_new_tokens=...,
    max_seq_len=None,
    batch_size=1,
    device="rpu",
)
```

Pass exactly one sizing form:

- `input_ids` plus `max_new_tokens`; or
- an explicit positive `max_seq_len`.

The cache exposes `reset()`, `reset_to_position(pos)`, `get_seq_length()`,
`get_max_length()`, `get_cache(layer_idx)`, and `to_dynamic_cache(device="cpu")`.
Batch sizes above one are limited to the explicitly admitted Qwen3 path and
require equal real-token lengths across rows.

`rpu_backend.api.Qwen3_5Cache.from_config(text_config, max_seq_len, ...)` adds
the recurrent and convolution state used by Qwen3.5 text. It may reset fully to
position zero but cannot reconstruct a non-zero recurrent state by rewinding.

## Image-text models

`rpu_backend.api.RPUModelForConditionalGeneration` is the Hugging Face-style
loader for the built-in Qwen3-VL conditional-generation path:

```python
from rpu_backend.api import RPUModelForConditionalGeneration

model = RPUModelForConditionalGeneration.from_pretrained(
    checkpoint,
    dtype=torch.float16,
    device="rpu",
    rpu_execution={
        "prefill": {"chunk_size": "auto"},
        "vision": {"chunk_size": "auto"},
    },
)
```

It has the same CPU-first loading and fail-fast profile behavior as the
CausalLM loader. The `vision` stage accepts `chunk_size`; the `prefill` stage
accepts the fields described above. Video and profiles outside
[Model support](model_support.md) are not implied by this class.

Qwen3.5 text/vision and Gemma4 use their documented model-specific adapters;
their source presence does not widen the public support envelope.

## Policy APIs

The following classes are exported from `rpu_backend.api`:

| Class | Primary construction and inference methods |
|---|---|
| `Pi05Policy` | `from_pretrained(...)` or `from_lerobot_policy(...)`; `to("rpu")`; `prepare_graphs(...)`; `predict_action_chunk(...)`; `select_action(...)` |
| `RhinoVLAPolicy` | `from_runtime(...)`, `from_factory(...)`, or `from_pretrained(..., runtime_factory=...)`; `prepare_graphs(...)`; `predict(...)`; `predict_action_chunk(...)` |
| `WallOssPolicy` | `from_checkpoint(...)` or `from_pretrained(...)`; `to("rpu")`; `prepare_graphs(...)`; `infer(...)`; `predict_action_chunk(...)` |
| `WallQwen35Policy` | `from_checkpoint(...)`; `to("rpu")`; `predict_action_chunk(...)` / `infer(...)`; `close()` |
| `Lingbot2Policy` | `from_checkpoint(...)`; `to("rpu")`; `prepare_graphs(...)`; `infer(...)`; `predict_action_chunk(...)`; `close()` |
| `HyEmbodiedPolicy` | `from_checkpoint(...)`; `to("rpu")`; `infer(...)`; `predict_action_chunk(...)`; `close()` |

`WallOssActionOutput`, `WallQwen35ActionOutput`, `Lingbot2ActionOutput`, and
`HyEmbodiedActionOutput` are the corresponding structured return types. A
physical-unit action field only means that the configured normalization was
applied; it is not a robot-safety or coordinate-frame certification.

`WallQwen35Policy` is a Source-only controlled-evaluation API bound to one
exact locally admitted checkpoint. Its initial envelope is FP16 batch 1 with
exactly the three canonical cameras, an initial multimodal prefix no longer
than 384 tokens, Dataset-V2 single-stage BICUBIC image preprocessing, robot ID
`10070`, normalizer `x2_normal`, state/action masks `[1]*20+[0]*6`, an action
shape of `[1,32,26]`, and 10 Euler steps. The controlled runtime maps base-text
prefill into fixed 64-row execution buckets from 64 through 384. A retained
Prefill signature also distinguishes the native final-chunk topology classes
(one row, 2--31 rows, or at least 32 rows); valid prefix length and M-RoPE
values remain per-call mutable state within a class. Native pad-zeroing emits a
stable fill-node envelope even for an exact bucket boundary. With optimization
enabled, Action also uses 64-row prefix buckets (`64..384`): KV insertion is at
the bucket boundary, an additive mask hides the gap after the real prefix,
and logical RoPE positions remain unchanged. Six retained entries share a fixed
SPM layout; model-owned per-bucket mask DDR slots and prefix KV are refreshed
outside capture, so both same-bucket changes and A/B/A requests can replay.
The fallback retains exact-prefix, per-step Action execution. Padding can change
FP16 attention rounding; this is not a bitwise-equivalence claim.
These optimizations are internal to the Wall
profile and do not broaden the generic Qwen3.5 execution API.
The single cold environment switch `WALL_QWEN35_OPT` is sampled by
`from_checkpoint()`: unset/`1` enables Vision1 + Prefill1 + Action1;
`0` restores per-image Vision and per-step Action (3+3+10 on the recorded
open-loop episode). There is no separate `action_execution` argument.
Both modes use half I/O weights, half GEMM outputs with ACC32, and half positive
Euler updates. Time/Ada stays a one-time CPU FP32 precompute uploaded in FP16;
normalization and public outputs remain CPU FP32. The historical FP32 host
profile is no longer selectable. This is an intentional precision downgrade,
not a numerical/task-quality certification.
The profile owns the six cold Graph/SDK budgets: enabled 32768/8/64 MiB with SDK
65536/16/128 MiB, disabled 8192/4/32 with SDK 65536/8/64. Individual overrides
are replaced; recreate the policy in a fresh process to switch. A matching
Python/native build is required. Optimized Vision/Prefill and both Action arms
require one physical segment per Graph invocation, failing rather than silently
splitting.

Wall packs its three single-frame images into one retained Vision Graph call;
each even patch grid is bounded by 14x14 (588 total patches maximum). Dense
operations share the packed rows, while each layer still calls the existing
attention kernel three times with isolated real-length image spans. Encoder
residual reductions retain per-image ring geometry inside that same Graph to
preserve the per-image reference's accumulation boundaries. Rebuild
the native extension when updating this adapter. This source implementation
does not change the numeric-blocked status or certify a speedup.
`predict_action_chunk(..., initial_noise=...)` accepts a finite CPU FP32-
convertible tensor with shape `[32,26]` or `[1,32,26]` for exact cross-device
flow-input parity. It is mutually exclusive with `noise_seed`; omitting both
preserves the deterministic seed-0 behavior.
`from_checkpoint(..., allow_numeric_blocked_vision=True)` is the required
explicit opt-in before `.to("rpu")`, because its Qwen3.5 vision path remains
numeric-blocked. `WallQwen35ActionOutput.actions` is a fresh contiguous CPU
FP32 tensor in physical units with shape `[1,32,26]`. The runtime owns the cold
multi-handle setting `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN=1`; an explicitly
false value is rejected before weight loading and execution must start in a
fresh process. On-board numerical parity, independent Graph-lifecycle, and
thresholded task validation remain pending.

`RhinoVLAPolicy` deliberately delegates checkpoint composition and preprocessing
to an explicit model-repository runtime factory. The runtime must expose the
execution capabilities and effective `rpu_execution` configuration it
consumed; the facade rejects a silent mismatch.

A string `runtime_factory="module:callable"` imports and executes Python code
from that module. Treat it as a trusted extension point: use only an installed,
reviewed model integration and do not copy the value from an untrusted TOML or
request. Passing an already imported callable has the same trust requirement.

`HyEmbodiedPolicy.from_checkpoint` does not read `norm_stats.pkl` by default.
Physical-action decoding requires `norm_stats_path`, the explicit
`trust_norm_stats_pickle=True` opt-in, and `norm_stats_sha256` for the exact
file bytes. The hash is checked before those same bytes are deserialized.

Several policy profiles are Experimental, Component-only, or Source-only. Use
the exact constructor, input schema, and precision in the release profile; do
not infer neighboring configurations from method availability.

## Error hierarchy

All user-facing errors are available from `rpu_backend.api`:

```text
RPUBackendError
├── RPUConfigError
├── UnsupportedModelError
├── RPUUnsupportedDtypeError
├── RPUSingleHandleError
├── SPMExhaustionError
└── WeightShapeMismatchError
```

Catch `RPUBackendError` for backend-level failures, or a subclass when the
caller has a specific recovery action. Standard `TypeError` and `ValueError`
remain in use for malformed Python arguments.

## Graph API

The public graph namespace is `rpu_backend.graph`:

```python
from rpu_backend.graph import GraphCache, GraphSignature

cache = GraphCache(max_entries=4)
sig = GraphSignature(
    op_id="my_model_forward",
    shapes=(seq_len, hidden_size),
    dyn_dims=(num_layers,),
    dtypes=(torch.float16,),
)

with cache.capture(sig):
    output = torch.ops.rpu.my_model_forward(...)
```

`GraphCache` provides `capture`, `begin_warmup`, `freeze`, `is_frozen`,
`lookup`, `evict`, `clear`, `size`, `max_entries`, `snapshot`, and
`cache_invariant_ok`. A frozen cache is lookup-only and rejects an online
BUILD. Most applications should let the model adapter own its graph cache;
manual graph construction is primarily a porting interface.
`GraphCache(..., require_single_segment=True)` is an immutable opt-in requiring
one physical segment covering all recorded nodes. It uses half the SDK entry,
command and instruction capacities (capped at 65536 / 32 MiB / 256 MiB), instead
of global soft budgets, without affecting other caches. Empty captures, partial
sync/downgrades, nested execution, host-node boundaries and resource-driven
splits are rejected before submission. It does not remove SDK hard limits.

User-level diagnostics remain public. On a `GraphCache` instance,
`debug_bucket_counts()`, `debug_branch_counts()`, `dump_signature_tree()`, and
`explain_miss(signature)` report cache organization, signature state, and miss
reasons. `Graph.debug_stats()` returns an aggregate mapping of execution
counters for the most recent graph submission. These methods are useful for
lifecycle diagnosis, but their output is not numerical or support evidence.
`Graph.dump_replay_plan()` returns a pointer-free linear node/segment summary;
`Graph.dump_tree()` presents the same execution structure grouped by segment.
Neither includes launch argument words, device instructions, or raw addresses.
Each segment summary also reports `entries`, `command_bytes`, and
`instruction_bytes` from the public launch-footprint query. These are reserved
batch resources, not tensor traffic or device execution time.
The low-level native module exposes `graph_segment_budget_abi=1` so controlled
runners can reject older extensions that would ignore the budget controls.

`Graph` is the lower-level capture object and `get_default_graph_cache()`
returns a thread-local cache. New adapters should normally use a dedicated
per-model `GraphCache` so ownership and teardown are explicit.

## Adapter registry and plugins

The registry lives in `rpu_backend.runtime.registry`:

```python
from rpu_backend.runtime.registry import (
    get_adapter,
    list_adapters,
    register_adapter,
)
```

Built-in architecture bindings are declared in
`rpu_backend.adapters._manifest.BUILTIN_ADAPTERS`. The current manifest binds:

- `Qwen3ForCausalLM`;
- `LlamaForCausalLM`;
- `DINOv3ViTModel`;
- `Gemma4ForConditionalGeneration`;
- `Qwen3_5ForConditionalGeneration`; and
- `Qwen3VLForConditionalGeneration`.

An external distribution may declare an entry point in the
`rpu_backend.plugins` group. The loaded callable receives a small namespace
whose `register_adapter(architecture, adapter_class)` method adds bindings:

```toml
[project.entry-points."rpu_backend.plugins"]
my_adapter = "my_package.rpu_plugin:register"
```

```python
def register(api):
    api.register_adapter("MyModelForCausalLM", MyModelAdapter)
```

Plugins are loaded lazily after an unknown-architecture lookup, or explicitly
with `discover_plugins()`. They cannot override a built-in architecture or a
previously registered plugin binding. Loading an entry point executes code from
the installed plugin distribution; install and enable only plugins you trust.

The registry is process-wide and has no public unregister operation. Once a
built-in or plugin architecture is bound, keep that binding for the process
lifetime; start a fresh process after changing the installed plugin set.

## Model-path registry

`rpu_backend.model_registry` provides local path resolution:

```python
from rpu_backend.model_registry import MODELS, cache_root, model_path

path = model_path("qwen3-0.6b")
```

`cache_root()` uses `RPU_MODEL_CACHE` when set and otherwise resolves to
`~/.cache/rhinoforge/models`. `model_path(name)` appends the checked-in relative
path and raises `KeyError` for an unknown alias. An alias is only a path mapping;
it is not a support or asset-availability claim.

## Weight-layout API for adapter authors

The canonical helpers are in `rpu_backend.runtime.weights`:

- `get_linear_partition(in_features, out_features, dwidth=2, num_cores=8)`;
- `swizzle_linear(weight, partition, num_cores=8)`; and
- `swizzle_model_inplace(model, ..., skip_names=...)`.

Swizzling changes parameter storage in place and must run exactly once. A
specialized linear owned by a fused subsystem belongs in `skip_names` and must
be transformed by that subsystem's one authoritative conversion path.

## `torch.rpu` device controls

The commonly useful device-level functions include:

- `is_available()`, `device_count()`, and `current_device()`;
- `set_caching_allocator(bool)`, `empty_cache()`, and `memory_stats()`;
- `set_ddr_flush(bool)` and `set_ddr_flush_force(bool)`;
- `set_debug_level(0..5)` and `get_debug_level()`;
- `set_cross_layer_batch_prefill(bool)` and
  `set_cross_layer_batch_size(int)`; and
- `shutdown()`.

Developer diagnostics are also available through `torch.rpu`:

- `set_debug`, `get_debug`, `set_profile`, `get_profile`, and
  `reset_profile_accumulators`;
- `set_debug_export`, `get_debug_export`, `list_debug_tensors`,
  `get_debug_tensor`, and `clear_debug_tensors`;
- `set_spm_debug`, `get_spm_debug`, and `spm_alloc_dump`.
- `set_hw_perf_trace(enabled, output_dir, max_dumps)`,
  `get_hw_perf_trace()`, and the
  `hw_perf_trace(output_dir, max_dumps=32)` context manager for r4 hardware
  kernel/DMA Chrome traces.

Debug tensor exports may contain model inputs or intermediate activations.
Store diagnostics as sensitive application artifacts and do not attach them to
public issue reports. These APIs do not expose raw Graph register, resource, or
plan payloads.

Hardware trace configuration is process-global and may change only between
forwards. Each change invalidates every registered Graph because Rhino Launch
bakes instrumentation into `build_batch()`. Enable it before the first forward,
preferably in a fresh process. `get_hw_perf_trace()` returns `enabled`,
`output_dir`, `max_dumps`, and `dump_count`. Files are timestamped
`rpu_hwperf_*_<graph>_{build,replay,oneshot}_segN.json`; analyze all segments for a
Graph, and use replay files for steady-state structure. r4 Release traces retain
kernel/DMA timing and scheduling metadata but intentionally omit readable
kernel names, operation types, and raw addresses. They are sensitive
application diagnostics, not full cache/stall/utilization PMU profiles.

CPU/RPU boundary flushing is enabled by default and should not be disabled in
normal inference. Chunk size is a per-handle `rpu_execution` setting, not a
process-global `torch.rpu` setter. Internal `torch.ops.rpu.*` schemas are host
wrapper interfaces used by adapters; unless a function is documented here or
in a model example, treat it as an implementation detail.
