# Wall Qwen3.5 READY-profile capability assessment

This assessment is limited to the exact controlled-evaluation profile below. It
does not promote the model-support status or transfer evidence from Wall-OSS.

## 1. Candidate identity

| Field | Value |
|---|---|
| Model source and immutable revision | Local checkpoint `/mnt/miyaa/work/ckpt/0_200000`; checkpoint file digests are enforced by `preflight_wall_qwen35_checkpoint()` and emitted by the runner |
| Architecture and family | Qwen3.5 VLA: 24-layer hybrid text backbone, 24-layer vision tower, six-layer action expert, ten-step flow update |
| Configuration hashes | `config.json`: `76aa51dd1cc450bf753c9790ffae668e21467964b4cb3dc8b0035853b6227c62`; `preprocessor_config.json`: `27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516`; `tokenizer_config.json`: `49e2b6e395f959f077f1e992b338919c0d4a9732fc6e613995e06557f843500c` |
| Precision | Controlled RPU execution in FP16; no quantized support claim |
| Input envelope | Batch 1; robot `10070`; `x2_normal`; mask `[1]*20+[0]*6`; `face_view`, `left_wrist_view`, `right_wrist_view`; Dataset-V2 grids `[[1,8,14],[1,10,14],[1,10,14]]`; real prefix `<=384`; horizon 32; action width 26; ten Euler steps |
| Returned output | Fresh contiguous CPU FP32 physical actions `[1,32,26]` |
| Runtime set inspected | RhinoForge source base `ad5319650816ce02ee78216283a812aa6b0a365d`; installed `rhinoforge 1.0.0`; PyTorch `2.10.0+cpu` with PrivateUse1; runtime/operator asset `runtime-v1.0.0-r4`; Rhino Launch `1.0.0` |

## 2. Source closure

- Candidate preprocessing, checkpoint admission, VLA orchestration, action
  shell, Qwen3.5 Vision/Text adapters, GraphCache, and public fused host wrappers
  were inspected.
- The vendor Wall-OSS profile archive was inspected as a behavioral reference,
  not as support evidence for this different checkpoint and architecture.
- Vendor guidance now confirms the r4 Queue HWPerf ABI and required host-side
  integration. The binding and BUILD/REPLAY/oneshot dump paths are implemented
  in this uncommitted source tree but remain pending rebuild and board evidence.
  Other missing evidence is an approved retained-Prefill contract for this Wall
  Qwen3.5 profile; an approved Graph-memory budget for two Vision signatures and
  the admitted action-prefix envelope; independent assessment review.

## 3. Semantic review triggers

- Multimodal Vision/Text fusion and M-RoPE are present and profile-owned.
- The base model is cache-only and intentionally omits the language `lm_head`.
- Vision, base Text, and Action alternate three live fused handles under
  `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN=1`.
- Action copies only the six physical full-attention prefix caches and performs
  ten host-orchestrated Euler steps.
- Prefix length and both image geometries affect execution and must not be
  omitted from Graph admission.

## 4. Capability table

| Requirement | Public/source evidence | Mapping | Status / open question |
|---|---|---|---|
| One clean measured Torch replay | The runner can warm one exact request outside the profiler and profile one identical repeat | Diagnostic orchestration | Covered by the current change; output parity and lifecycle are written to the summary |
| Readable model-stage stack | Runtime ranges name preprocessing, Vision/Text Prefill, Action loop, and Action decoder | Diagnostic orchestration | Covered |
| Action READY replay for one exact prefix | `action_graph_cache`; ten calls per request; mutable action inputs and cache bases | Existing adapter | Covered for one repeated prefix, subject to board validation |
| Vision READY for the production request | Face and wrist geometries produce two signatures while the shared adapter owns one cache entry and explicitly evicts on shape change | Adapter/runtime planning | Gap: the repeated request rebuilds the two shapes instead of replaying three calls |
| Base Prefill READY | Qwen3.5 Text defaults to a bounded raw-Graph one-shot; fixed-profile retained Prefill exists only as another profile-specific path | Runtime extension / profile admission | Gap: no Wall Qwen3.5 retained-Prefill contract is admitted |
| Finite prefix envelope in READY | Action signature contains the real prefix length and its cache capacity is one | Runtime planning | Gap: requires an approved finite cache, padding/bucketing rule, or mutable semantic-prefix contract |
| Vendor-style HWPerf dumps | Vendor confirmed the r4 Queue API; this source exposes bounded `set/get/hw_perf_trace`, invalidates Graphs on transitions, and dumps every BUILD/REPLAY/oneshot segment | Launch/runtime integration | Implemented in source; build and board artifact validation pending |
| Release support | Vision is numeric-blocked and same-dtype, FP32-anchor, task, and independent lifecycle gates remain pending | Validation | Not covered |

## 5. Resource and lifecycle fit

- Action already proves the local one-BUILD/nine-REPLAY structure within a ten
  step request. A repeated identical request should add ten replays without
  changing its signature or cache size.
- Vision needs two simultaneously retained entries for the exact face/wrist
  geometry set, or a validated single-geometry preprocessing contract. Increasing
  capacity without an approved persistent-SPM/queue memory budget is not an
  adapter-only support claim.
- Base Prefill currently has an explicit bounded-one-shot lifecycle. Retention
  must prove stable hidden-input, M-RoPE, KV-cache, valid-length, output, and DMA
  ownership across replay.
- Full-dataset READY additionally needs a finite prefix execution plan. The
  observed episode spans real prefixes 299 through 313, but the public profile
  admits up to 384 and cannot silently specialize to this one recording.
- The currently installed wheel predates the HWPerf binding. The uncommitted
  integration must be rebuilt against the same r4 Launch/runtime, then verified
  by non-empty BUILD/REPLAY/oneshot JSON artifacts before it becomes evidence.

## 6. Verification contract

Before a complete READY claim:

1. Run the exact first request unprofiled, then require bit-exact physical and
   normalized action output from the identical repeat.
2. Freeze Vision, Base Prefill, and Action caches before measurement. Reject any
   READY miss instead of building online.
3. Prove fixed cache sizes, unchanged signatures, zero recaptures,
   `cache_invariant_ok()`, and replay deltas of Vision 3, Prefill 1, and Action 10
   for the measured policy call.
4. Exercise prefix-envelope minimum/maximum and A/B/A same-shape input refresh;
   retain the first returned output across later calls and verify independent
   storage.
5. Run same-dtype parity, FP32-anchor, maximum-envelope, safe-rejection,
   teardown/reload, and representative open-loop task gates without a profiler.
6. In a fresh process, profile only one admitted READY replay. Record exact
   source, checkpoint, Launch, operator asset, board/runtime, input, and output
   hashes. Treat timing as diagnostic while the numerical profile is pending.

### Controlled board probe (2026-09-01)

The warmup/one-repeat diagnostic completed on runtime
`EA_R1SDK_2026_03C-cf33df86` for real prefix 308. Warmup and measured physical
actions were bit-exact (`max_abs=0`). Action added ten replays with zero
recaptures; Vision added zero retained replays because it rebuilt face and wrist
signatures; Base Prefill reported its bounded-one-shot contract. The resulting
admission was therefore `runtime_extension_required`, not READY. The measured
trace contained one each of the top-level preprocessing, Vision/Text Prefill,
and Action-loop ranges, ten Action-decoder ranges, three Vision graph calls, one
base Prefill call, and ten Action graph calls. This result validates the probe's
classification; it is not numerical or performance certification.

### Controlled r4 HWPerf probe (2026-09-01)

After rebuilding this uncommitted integration, one real open-loop request
completed on the same runtime at the vendor-reference 800 MHz conversion. It
produced 16 non-empty, parseable Chrome JSON files: two Vision BUILDs and one
Vision REPLAY; three Base-Prefill oneshot segments; one Action BUILD and nine
Action REPLAYs. Across the files the runtime emitted 155,812 trace events with
balanced duration/flow pairs (`B=E=37,971`, `s=f=39,935`) and `Compute`, `DMA`,
and `SYNC` categories. The run metadata records every artifact size and SHA256
plus the native pre-disable state (`dump_count=16`, `max_dumps=64`).

This proves the host binding reaches actual r4 queue kernel/DMA events; it is
not a latency or model-support claim. Unlike the supplied vendor archive, which
contains readable names from its build profile, this installed r4 Release trace
uses sanitized `kernel_<id>` names. That matches the vendor's Release
sanitization statement. Readable kernel names require a vendor-authorized
Debug/ReleaseForDev build or supported symbolization artifact; they cannot be
reconstructed by this public host integration.

## 7. Vendor requirements

The vendor supplied the release-matched r4 Queue contract for item 1 below. The
remaining model-lifecycle questions still require a release-matched response for
`runtime-v1.0.0-r4`, Rhino Launch `1.0.0`, and the installed RhinoForge ABI:

1. **HWPerf public API — vendor response received, implementation pending board
   verification.** r4 requires `set_enable_hw_perf` before batch construction and
   `dump_hw_perf_chrome` after synchronous enqueue and before the next build. A
   setting transition must reset Graph caches. Release JSON retains timing,
   stream/core/channel metadata while redacting readable names, op types, and
   raw addresses. `LKN_RPU_FREQ_MHZ` only controls cycle conversion.
2. **Retained Qwen3.5 Prefill.** Confirm the supported host-side contract for
   cache-only/no-`lm_head` Prefill with variable real length, padded execution
   length, M-RoPE, GDN/full-attention caches, and three coexisting fused handles.
   Identify every signature field and every mutable/fixed DMA address. Provide
   either an approved patch or precise public wrapper changes plus expected
   BUILD/REPLAY counters.
3. **Vision two-signature residency.** Confirm that the exact `[1,8,14]` face and
   `[1,10,14]` wrist profiles may retain two prepared queues simultaneously under
   the three-handle persistent-SPM setting. State the queue/SPM memory ceiling,
   eviction restrictions after freeze, and required teardown order. If this is
   unsupported, provide the approved fixed-geometry preprocessing envelope.
4. **Action prefix execution plan.** Provide the supported approach for the
   public real-prefix envelope `1..384`: finite cache capacity, execution-length
   padding/bucketing with a mutable real-prefix mask, or another approved plan.
   Define semantic inputs, maximum entries/memory, and safe rejection behavior.
5. **Acceptance artifacts.** Provide board/runtime identity, exact compatible
   operator asset, a vendor-known-good READY trace/summary for this Qwen3.5
   checkpoint profile (not the Qwen2.5-VL Wall-OSS checkpoint), expected Graph
   counts/replay deltas, and any golden output hashes needed for repeatability.

Device-program implementation or opaque operator-asset internals are not
requested; public host contracts, release-matched binaries/assets, and
verification evidence are sufficient.

## 8. Decision

```text
outcome:       Runtime extension
certification: uncertified
```

- Assumption: the current host source and installed runtime are the intended
  starting profile.
- Blockers: full Vision/Prefill READY, finite action-prefix planning, board
  verification of the new HWPerf binding, numerical/task evidence, and
  independent review.
- Smallest next step: rebuild this source against r4 and collect one bounded
  Wall Qwen3.5 hardware-trace session, then inspect every replay segment before
  changing Graph topology.
