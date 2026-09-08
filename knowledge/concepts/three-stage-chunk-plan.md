# Three-stage chunk execution

Some fused models pack several independent streams or use different shapes for
input loading, Q/K/V production, and the remaining layer computation. Treating
all three phases as one chunk list either loses stream boundaries or duplicates
model-specific execution code.

## Contract

- Build one `FmbThreeStageChunkPlan` with `input`, `qkv`, and `compute` chunk
  lists.
- Record semantic spans that contiguously partition the full physical
  sequence. A stage chunk may cross a span boundary only when the model's exact
  profile certifies that packed operation.
- Reuse the common `FusedModelBase` layer executor. Models contribute their
  buffers and layer operations, not another traversal loop.
- Bind physical preparation and Graph reuse to the complete plan fingerprint,
  including spans and the resolved attention-storage policy.
- Leave attention storage on `AUTO` unless an exact profile owns the choice.
  `AUTO` uses the certified SPM K/V path only when the model is eligible and
  the joint layout fits; otherwise it uses DDR. An explicit unsupported SPM
  request fails closed.

The one-chunk input, QKV, and compute case is valid and follows the same path.
Do not add a second executor for it.

Wall Qwen3.5's source-only packed-spatial candidate uses one shared dense chunk
and three real image spans (at most 196 patches each). Its encoder residual
reductions also preserve each image's geometry inside that Graph: changing the
ring's flattened row partition can change rounding despite identical partials.
Use the shared reduction wrapper per span; do not add a new route selector.
Spans do not create an
attention mask: the current host wrapper explicitly inserts and attends to
each image's independent KV slot. Ordered lengths belong in Graph identity and
the workspace layout hash. Retain narrow KV views across replay so their
TensorImpl identities remain stable. The merger processes packed rows and
scatters intersections to independent mutable text destinations, including an
image crossing the 512-patch merger chunk boundary. This is a source contract,
not board numerical, lifecycle, or performance evidence.

## Failure signals

- a chunk crosses a semantic stream boundary;
- a rebuilt physical layout does not change Graph identity;
- the first call works but replay uses stale span or plan data;
- an ineligible or oversized request silently selects SPM attention; or
- a model contains a copied three-stage traversal loop.

## Public sources

- [Architecture: three-stage chunk execution](../../docs/architecture.md#three-stage-chunk-execution)
- [`FmbThreeStageChunkPlan`](../../src/core/fused_model_base.h)
- [Plan construction and attention policy](../../src/core/fmb_three_stage_chunk_plan.cpp)
- [Shared execution lifecycle](../../src/core/fused_model_base.cpp)
- [Wall packed-spatial host wrapper](../../src/fused/rpu_qwen3_5_vision_model.cpp)
- [Qwen3.5 Vision adapter and signatures](../../python/rpu_backend/adapters/qwen3_5/vision.py)
