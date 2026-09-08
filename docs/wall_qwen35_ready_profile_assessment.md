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
| Source candidate under review | Git baseline `ac63345007132a2cef4f73079611519434cdfecd` plus this assessment's containing commit. The retained-Graph changes were reviewed and tested board-free; they have no transferred board result |
| Historical board runtime | Installed `rhinoforge 1.0.0`; PyTorch `2.10.0+cpu` with PrivateUse1; runtime/operator asset `runtime-v1.0.0-r4`; Rhino Launch `1.0.0` |

## 2. Source closure

- Candidate preprocessing, checkpoint admission, VLA orchestration, action
  shell, Qwen3.5 Vision/Text adapters, GraphCache, and public fused host wrappers
  were inspected.
- The vendor Wall-OSS profile archive was inspected as a behavioral reference,
  not as support evidence for this different checkpoint and architecture.
- Vendor guidance confirms the r4 Queue HWPerf ABI. The checked-in host binding
  and BUILD/REPLAY/oneshot dump paths were exercised by the historical r4 probe
  below; that result predates this candidate's retained Vision/Prefill lifecycle
  and cannot validate it.
- Remaining evidence includes an approved Graph-memory budget for two Vision
  signatures and the lazy 18-entry Prefill capacity; board replay and numerical
  validation of the current candidate; exact-prefix Action lifecycle evidence;
  and independent assessment review.

## 3. Semantic review triggers

- Multimodal Vision/Text fusion and M-RoPE are present and profile-owned.
- The base model is cache-only and intentionally omits the language `lm_head`.
- Vision, base Text, and Action alternate three live fused handles under
  `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN=1`.
- Action copies only the six physical full-attention prefix caches and performs
  ten host-orchestrated Euler steps.
- Prefix length and both image geometries affect execution. Base-text Prefill
  uses a fixed bucket plus native tail-topology class and a mutable valid-length
  update. Action remains keyed by exact real prefix; the face/wrist geometries
  remain two separate Vision signatures across three calls.

## 4. Capability table

| Requirement | Public/source evidence | Mapping | Status / open question |
|---|---|---|---|
| One clean measured Torch replay | The runner warms one exact request, freezes Vision/Prefill/Action caches, and profiles one identical lookup-only repeat | Diagnostic orchestration | Source contract covered; output parity, READY phase, and lifecycle are written to the summary; board execution remains pending |
| Readable model-stage stack | Runtime ranges name preprocessing, Vision/Text Prefill, Action loop, and Action decoder | Diagnostic orchestration | Covered |
| Action READY replay for one exact prefix | `action_graph_cache`; ten calls per request; mutable action inputs and cache bases | Existing adapter | Covered for one repeated prefix, subject to board validation |
| Vision READY for the production request | Face and wrist geometries produce two signatures; the controlled Wall runtime reserves a bounded two-entry cache while generic Qwen3.5 retains its one-entry default | Adapter/runtime planning | Rebuild gap addressed in source; board replay and memory-budget validation remain pending |
| Base Prefill READY | Wall attaches a bounded retained Prefill GraphCache with 64-row buckets through 384. Its identity also includes the final GDN chunk's one-row, 2--31-row, or 32+-row native topology class; capacity is the lazy 6 x 3 cross-product. `set_valid_prefill_len` and M-RoPE updates stay outside capture, while native halo/fill registers remain mutable within a class | Runtime extension / profile admission | Source contract implemented; board replay, numerical parity, and memory-budget validation remain pending |
| Finite prefix envelope in READY | Wall admits every prefix `<=384` into one of `64,128,192,256,320,384` plus its native tail-topology class. Action fast replay remains keyed by the exact real prefix because KV insertion and SDPA bake that value | Runtime planning | Source contract implemented for identical-request READY; maximum-envelope, cross-class, and cross-bucket lifecycle validation remain pending |
| Vendor-style HWPerf dumps | Vendor confirmed the r4 Queue API; this source exposes bounded `set/get/hw_perf_trace`, invalidates Graphs on transitions, and dumps every BUILD/REPLAY/oneshot segment | Launch/runtime integration | Host binding was validated by the historical r4 probe; current retained-Graph candidate still needs a fresh build and trace |
| Release support | Vision is numeric-blocked and same-dtype, FP32-anchor, task, and independent lifecycle gates remain pending | Validation | Not covered |

## 5. Resource and lifecycle fit

- Action already proves the local one-BUILD/nine-REPLAY structure within a ten
  step request. A repeated identical request should add ten replays without
  changing its signature or cache size.
- Vision now requests two simultaneously retained entries for the exact
  face/wrist geometry set on the controlled Wall path. The generic Qwen3.5
  default remains one entry. This bounded increase still requires an approved
  persistent-SPM/queue memory budget and board replay validation; it is not an
  adapter-only support claim.
- Base Prefill now uses a retained bucket-plus-tail-topology lifecycle for
  Wall. The extra class discriminator is required because one-row,
  unsegmented, and segmented GDN tails emit different nodes or grids. It still
  must prove stable hidden-input, M-RoPE, KV-cache, valid-length, output, and
  DMA ownership across replay; generic Qwen3.5 keeps its bounded one-shot
  default.
- Action retains fast replay only for an exact real prefix. A changed prefix
  deliberately replaces its one cache entry so KV-insert positions and SDPA
  lengths cannot remain frozen from an earlier request.
- Base Prefill now has a finite execution plan for the public envelope. The
  observed episode spans real prefixes 299 through 313, which share bucket 320
  and the 32+-row tail class. Action intentionally leaves READY and may BUILD
  again when the exact prefix changes, so this is not a dataset-wide frozen-
  READY claim. Longer admitted prefixes use the remaining bucket/topology
  identities.
- The historical wheel and traces predate these Graph-lifecycle changes. The
  containing commit must be rebuilt against the same r4 Launch/runtime and
  produce fresh non-empty BUILD/REPLAY/oneshot artifacts before its lifecycle
  or timing can be treated as board evidence.

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

The following probes are historical diagnostics from the original integration
lineage subsequently recorded by `ffe04ece7eda5ca0448440df0da41757154209df`.
They do not validate the retained-Graph candidate identified above.

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

### Controlled r4 HWPerf probe (2026-09-02)

After rebuilding the then-current integration, one real open-loop request
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

1. **HWPerf public API — vendor response received; historical host binding
   verified.** r4 requires `set_enable_hw_perf` before batch construction and
   `dump_hw_perf_chrome` after synchronous enqueue and before the next build. A
   setting transition must reset Graph caches. Release JSON retains timing,
   stream/core/channel metadata while redacting readable names, op types, and
   raw addresses. `LKN_RPU_FREQ_MHZ` only controls cycle conversion. A fresh
   trace is still required for the current retained-Graph candidate.
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
4. **Action prefix execution plan.** Confirm whether exact-prefix BUILD followed
   by a frozen same-prefix replay is the supported approach for the public
   real-prefix envelope `1..384`, including the required WARMING boundary between
   different prefixes. Otherwise provide an approved finite cache or mutable-
   prefix plan. Define semantic inputs, memory, and safe rejection behavior.
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

- Assumption: the containing commit and the named historical runtime are the
  intended next validation pair.
- Blockers: current-candidate board replay and memory evidence for two Vision
  signatures and retained Prefill; frozen exact-prefix Action validation;
  maximum-envelope and numerical/task evidence; and independent review. A
  dataset-wide frozen READY state across changing Action prefixes is not claimed.
- Smallest next step: rebuild the containing commit against r4, warm and freeze
  one exact request, then collect one lookup-only replay trace. After that,
  exercise the base-Prefill topology boundaries and a different Action prefix
  only after returning the caches to WARMING.
