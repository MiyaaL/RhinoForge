# GraphCache capture

`GraphSignature` is the key for one admitted execution shape and semantic mode.
`GraphCache.capture(signature)` builds the graph on the first call and reuses it
on later calls with the same signature. [Graph API](../../docs/api_reference.md#graph-api)

## Correct reuse

- A production adapter normally owns a dedicated per-model `GraphCache` and
  wraps its top-level native forward in `capture`.
  [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- Every value that can change the result must be encoded in the signature or
  updated at a stable contract point through a mutable transfer.
  [Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)
- A repeated signature is healthy when it builds once, later replay counters
  increase, cache size stays fixed, no second build occurs, and
  `cache_invariant_ok()` remains true.
  [Architecture](../../docs/architecture.md#graph-and-batch-execution)
- `freeze()` changes a warmed cache to lookup-only mode; a missing or invalid
  entry is rejected instead of being built online.
  [GraphCache implementation](../../python/rpu_backend/graph/_runtime.py)
- Cache ownership includes teardown. Prefer an explicit per-model cache over a
  shared default when writing a new adapter.
  [Graph API](../../docs/api_reference.md#graph-api)

A bounded one-shot graph is an explicit model-contract exception for a
variable-shape path. It must execute one non-trivial graph, return to
passthrough state, and leave retained-cache size unchanged.
[Model porting](../../docs/model_porting.md#7-graph-and-cache-invariants)

## Segment budgets

For entry-driven segmentation, the generic soft budget is 8192 nodes. The cold
`RPU_GRAPH_MAX_SEGMENT_ENTRIES` control can request up to 32768 while retaining
SDK headroom and stream fences. Independent command/instruction controls default
to 4/32 MiB and permit at most 8/64 MiB, always clamped to half SDK capacities.
The Wall runner's experimental `--language-one-graph` preset requests 32768 entries and 8/64 MiB, with SDK capacities of
65536 entries and 16/128 MiB; defaults stay conservative pending numerical parity.
Graph count and physical segment count must be checked
independently; `dump_replay_plan()` reports each segment's reserved entry,
command and instruction footprint. See [runtime configuration](../../docs/runtime_config.md#graph-and-replay).

For the FP16 Wall Action loop, `GraphCache(require_single_segment=True)` is an
immutable per-cache opt-in, with SDK-half budgets and bounded maxima. It rejects
empty scopes, partial sync/raw-kernel fallback, nested execution and any plan
that is not one segment covering all nodes. This does not enlarge other caches'
soft budgets. The ten device Euler steps are one FMB body-iteration sequence;
the packed action and padding velocity must remain persistent across bodies.
Time/Ada is precomputed outside capture, and mutable DMA refreshes each request's
noise/mask/padding/output. Cold priming is outside the READY one-submit claim.

## Sources

- [Architecture: graph and batch execution](../../docs/architecture.md#graph-and-batch-execution)
- [Graph API](../../docs/api_reference.md#graph-api)
- [Canonical Python runtime](../../python/rpu_backend/graph/_runtime.py)
- [Model porting: graph invariants](../../docs/model_porting.md#7-graph-and-cache-invariants)
