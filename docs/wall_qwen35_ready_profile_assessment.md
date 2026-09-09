# Wall Qwen3.5 READY-profile capability assessment

This assessment is limited to the exact controlled-evaluation profile below. It
does not promote the model-support status or transfer evidence from Wall-OSS.

## Profiling CLI consolidation (2026-09-09)

The current open-loop runner exposes only `--torch-profile-dir DIR` and
`--hw-perf-dir DIR` for enabling the respective profilers. Torch uses one
compressed trace per request, with shapes/stacks enabled and memory events
disabled; it includes first-request setup/BUILD and does not insert an
unprofiled warmup or frozen READY repeat. The hardware dump bound remains
`--hw-perf-max-dumps`. See [model testing](model_testing.md) for current commands.
Commands using `--torch-profile`, `--torch-profile-output` or `--hw-perf-output`
below are historical receipts for the revisions measured at the time, not
commands for the consolidated CLI. This CLI-only change does not supply new
RPU numerical, READY or performance evidence.

## Packed-spatial source update (2026-09-08)

The new candidate supersedes the two-signature/three-call Vision design
described in the historical assessment below. Wall now submits one packed
Vision Graph call per request. Dense work shares all image rows; each layer
still makes three independent real-length attention calls using persistent KV
views. The signature includes ordered image lengths. The envelope is three
single-frame even grids up to 14x14 each; the usual lengths are `[112,140,140]`,
and the maximum is `[196,196,196]`. Default Vision cache capacity is one.
The shared FusedModelBase executor, mutable input/output DMA, and maximum-
envelope cross-chunk merger scatter are retained. A future varlen kernel can
replace the isolated attention emitter without changing the image contract.
The encoder's two residual AllReduce sites also retain per-image boundaries
inside that same Graph. This preserves the old ring geometry and accumulation
order while leaving dense work packed; changing the reduction's row count can
otherwise introduce FP16 rounding differences even from identical partials.
The open-loop wrapper checks the installed packed-Vision Python/native ABI
before weight loading and prints the loaded adapter and Vision cache counters.
The runtime closes the Action-to-Vision and Text-to-Action temporary-SPM
boundaries outside Graph capture, including the prefix error path. This avoids
appending packed Vision's workspace to dead Action priming allocations while
preserving retained Graphs and persistent state.

Classification remains Runtime extension / uncertified. Board-free checks
do not transfer the historical r4 timings or Graph receipts to this candidate.
Fresh native build, board numerical/FP32-anchor/task gates, maximum-envelope
SPM admission, one-BUILD/stable-REPLAY, changing inputs/destinations, and clean
latency measurements are required. No device kernel or operator asset was
changed. Sections below retain the previous candidate's context and evidence.

### Installed-candidate diagnostic (2026-09-08)

The rebuilt installed candidate completed the original open-loop wrapper's
full 16-request / 468-frame episode with `--torch-profile` (prefixes 299--313).
Vision had one BUILD, one cache entry, one segment, replay counts 1 through 16,
zero recaptures, and a true invariant throughout. The first-request READY probe
accepted exact Vision/Text/Action replay deltas of 1/1/10; physical and
normalized actions were bit-exact. Later requests return to WARMING and may
rebuild Action for a changed exact prefix, so this is not dataset-wide frozen
READY. The previously reproduced third-request temporary-SPM OOM did not recur
after closing the component handoff boundaries.

Standalone component checks using the exact checkpoint and seed 7241 compared
packed and per-image paths for `[112,140,140]`, `[140,112,140]`, and
`[196,196,196]`. Final hidden and merger outputs were bit-exact in all cases;
repeats, retained outputs, image isolation, and maximum-envelope fusion
scatter/fresh destinations/canaries passed. Native SHA256:
`0907000f7ce6c0577936c79c0b548b43dd56fe9766da4d7a79f74b565b6bf8af`;
installed Wall runtime Python SHA256:
`20230e4c40c6b9cffae2e3b2519459c94443f4e459243b0b024762d9131ef792`.
The related board-free suite passed 123 tests. Raw application/profile
artifacts remain outside Git. These bounded diagnostics do not certify the
FP32-anchor/task gates, eliminate every prior intermittent numerical issue,
promote the support status, or establish a speedup.

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
   `cache_invariant_ok()`, and replay deltas of Vision 1 (3 in the historical
   per-image design), Prefill 1, and Action 10
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

## 9. Local Language segment-budget diagnostic (2026-09-08)

This is a local, uncertified scheduling experiment, not a READY or release
promotion. It uses the existing checkpoint/profile, FP16 decoder boundary and
FP32 host Action projections/Euler updates. No model operator, accumulation
mode, cast boundary or 128/128/64 SPM chunk topology was changed.

The exact Language bucket `[320,2048]` contains 24,367 nodes (10,281 kernels,
14,086 data nodes). The single-segment public footprint is 4,344,736 command
bytes and 42,184,704 instruction bytes.

| Arm | Soft entries / command MiB / instruction MiB | SDK capacities | Actual segments |
|---|---|---|---|
| Conservative baseline | 8192 / 4 / 32 | 65536 / 8 / 64 | 3 |
| Entry-only override | 32768 / 4 / 32 | 65536 / 8 / 64 | 2; first segment reaches exactly 32 MiB instruction |
| Full experimental preset | 32768 / 8 / 64 | 65536 / 16 / 128 | 1 |

The full preset completed all 16 requests (prefix 299--313): one retained
Language entry, one BUILD and 15 REPLAYs, zero recaptures, and true cache
invariants. These are normal retained-cache observations, not a frozen READY
admission. Eight requests had bit-exact KV/recurrent/conv states against the
baseline; eight did not. Maximum Language-state absolute difference was
0.42333984375; maximum returned-action absolute difference was
0.005868434906005859. All compared values were finite. Noise, frame indices,
and ground truth were exact. A separate three-segment repeat also differed in
one of its first three Language boundaries (max absolute 0.29443359375), while
the other two were exact. Thus baseline instability exists, but this does not
establish that the one-segment candidate is correct or equivalent. The first
request's earliest differing captured state was layer 20 for the full preset
and layer 5 for the entry-only arm; these are layer-boundary observations,
not identification of the first incorrect kernel.

The entry-only arm was interrupted on its second request by an incomplete
activation-file write when the temporary filesystem filled. Its first request
proves two segments and a Language-boundary difference; it is not a completed
16-request result. Complete baseline activations were retained privately in
memory-backed temporary storage; subsequent comparisons wrote statistics only.
No activations, device traces or operator assets are checked into the repository.

Reproducibility anchors for the full diagnostic:

- Source base: `c4e01dff40e40007b8178c96d051a410d9bd0ec4` plus the local budget
  controls/resource census; no commit or release selection was made.
- Tested native SHA-256:
  `3c7596bbe71809350ebb99db1fb1e13d60bdca1d77526221199ad6cf589a50c1`.
- r4 operator asset SHA-256:
  `538f49e256814d104c98aa8df4b6d3af3d89b2d636a1fe22b0731df3ce39ac99`.
- Rhino Launch library SHA-256:
  `10910756d373171c74ce80ecfe590324c9e2db32788a6367034a310b771772a6`.
- Private local receipt directory: `/tmp/wall-fp32-capability-QsSK8J`;
  `full-comparison.json`, `full-one-segment/meta.json`,
  `full-one-segment/segments.json`, and `repeat-three-segments`.

The subsequent native capability-marker rebuild only adds the runner's old-ABI
rejection; full-model results above belong to the explicitly hashed executable,
not to another rebuild. The final native hash
`491cb17e9f6124154465b0af0d8a1bb99dfa8068fef36a479b1b2e9224fa0964`
passed actual capability-marker loading, packed-Vision ABI checks and a bit-exact
FP16 host/device/host roundtrip; its full-model gate was not rerun. The related
board-free suite passed 150 tests. The wrapper keeps conservative defaults and exposes
`--language-one-graph` only as an experimental preset after a matching rebuild.
No timing/speedup, FP32-anchor, maximum-envelope or task-quality gate is claimed.
At the time of that diagnostic, Action remained ten Euler-step decoder replays: preserving its FP32 host
boundaries requires an independently validated device implementation before
it can become one physical batch.

## 10. FP16 Action single-Graph extension (2026-09-09)

Historical evidence below predates the unified switch in section 11. Its old
CLI commands identify the measured revision, not current selectable options.

The user explicitly replaced the FP32-boundary requirement with FP16 Action.
The accepted implementation assessment is a **Runtime extension**: reuse the
existing Qwen3.5 FMB ten-body loop, release ACC32/FP16 GEMMs, scalar/broadcast
half arithmetic and mutable DMA. No new device kernel or operator asset is
introduced. Independent read-only review checked the Wall-specific positive
Euler formula, six attention/eighteen identity topology, SPM lifetimes, input
refresh and strict physical-segment admission. This is not release certification.

The exact envelope remains checkpoint `0_200000`, batch 1, horizon 32, width 26,
real prefix 1..384. Pack `[action26, mask26, zeros12]` into width 64. Pad `w1` to
1024x64 and output projection to 64x1024; inputs/weights/results are half, GEMM
accumulation remains ACC32. Divide the half input-projection result by4 as a
separate operation. Each Euler step computes half `velocity*mask + padding`,
then a separate half multiply by `0.09991455078125` and half add to action.
Padding velocity is the original CPU FP32 value cast to half and pre-masked;
packed mask columns never update. Ten CPU FP32 time/Ada rows are prepared once
and uploaded in half. Normalization and returned CPU tensors remain FP32.

`fp16_one_graph` is the wrapper/Policy default; `fp32_host` retains the old
projection/Euler reference. New ABI and cold SDK capacities (at least 65536 /
8 MiB command / 64 MiB instruction) are checked before full model loading.
Action alone uses immutable `require_single_segment`: SDK-half bounded
budgets, no partial sync/downgrade/nested submission, one segment covering all
nodes or failure before execution. Language still needs `--language-one-graph`
to request its larger soft budgets. The three-submission claim applies only
after warmup; persistent priming and BUILD are not included.

Final board-free checks: all 274 repository tests passed, including independent FP16
packing/rounding, overflow rejection, legacy profile, lifecycle guards and
budget tests. Local board results follow. Broader input-envelope numerical
coverage and task-quality/release gates remain pending; do not infer a speedup
or release acceptance from these local diagnostics.

Initial local board evidence (not a release/quality certificate):

- Native SHA256 `f4dc2a1e29a5ea8791fc36069d2b84391e1d16986edcb066789b80edb77d6f35`;
  Torch `2.10.0+cpu`, Rhino Launch `1.0.0`, unchanged r4 operator asset
  SHA256 `538f49e256814d104c98aa8df4b6d3af3d89b2d636a1fe22b0731df3ce39ac99`.
- FP16 host/device/host roundtrip is bit-exact; strict empty-Graph capture is
  rejected. The independent Action probe uses checkpoint weights, synthetic
  physical prefixes, and A/B/A seeds 3407/3408/3407, changing source K/V for B.
- Prefix 308 ten-step Graph: 17520 nodes = 4360 kernels + 13160 data nodes;
  one segment, 2110400 command bytes, 18104320 instruction bytes. These fit
  within half the default SDK capacities, so the provisional 32/256 MiB SDK
  requirement was removed from Python/wrapper; native arithmetic is unchanged.
- A/B/A: one BUILD, two REPLAY, zero recaptures, invariant true, A repeat
  bit-exact, B changes the output. The same FP16 operators executed in ten
  separate calls produce bit-exact final actions for all three inputs
  (`max_abs=0`). That split reference has one BUILD and 29 REPLAY.
- The first probe computed/saved correct results but failed its own teardown
  because a diagnostic Graph lookup still co-owned the cache entry. Releasing
  that local reference fixes teardown without changing inference. Corrected
  probe SHA256 `54b5d4080e6a8924f40fab932825d29881c9f4c7cbffe42d040e0772e4d1836c`;
  split-reference process exits cleanly. This is not a candidate-runtime fix.
- Independent CPU half-boundary reference (same RPU decoder): A/B/A repeats
  are exact and the process exits cleanly at SDK 8/64 MiB. Compared with the
  device-loop output, max absolute normalized-action errors are 0.0029296875 /
  0.00244140625 / 0.0029296875 (means 0.000271476 / 0.000386605 / 0.000271476).
  This reference is not bit-exact. It changes CPU/RPU projection and Euler
  boundaries; the device split/unrolled comparison above remains exact, so
  these numbers must not be labeled graph-merging error or quality acceptance.
- Raw diagnostic artifacts are sensitive temporary files under
  `/dev/shm/rhinoforge-wall-fp16-aaONUI`; no activation data is checked in.
- Maximum-prefix 384 probe also has 17520 nodes in one segment at default SDK
  8/64 MiB, with one BUILD/two REPLAY, zero recaptures, invariant true, exact A
  repeat and clean teardown.
- Real first episode request (prefix 308), using the wrapper with
  `--language-one-graph --max-requests 1 --torch-profile`: Vision, Language and
  Action each have one segment, one REPLAY, zero recaptures and a true invariant.
  The exported trace contains exactly three `rpu_graph::segment_launch`, three
  `rpu_graph::execute_replay` and three `rpu_graph::replay_segment_loop` events.
  Frozen READY admission is **accepted**; both normalized and physical outputs
  are bit-exact between warmup and measured repeat. Process teardown is clean.
- The verified wheel is installed in the wrapper's `hx` Python environment;
  installed native/Action Python hashes match the isolated tested package.
  A recoverable old-package archive is `installed-before.tgz` in the temporary
  diagnostic directory. No operator asset or dependency was changed.
- Installed-environment FP32 host baseline, same real request, Language preset
  and noise SHA256 `9bff273106fba94a31a550264c18e7186dbea33d73c93286a1c65ddecbaaa0d0`:
  the trace has 12 physical segment launches (1 Vision + 1 Language + 10 Action),
  versus 3 for FP16. All cache invariants/lifecycle checks pass and teardown is
  clean. The old FP32 arm is **not** repeat-bit-exact: warm/measured differences
  are 0.002384424 physical and 0.002546400 normalized; its READY numerical gate
  remains unaccepted.
- FP16 versus the FP32 measured output: physical max/mean absolute differences
  0.002058029 / 0.000199179; normalized max/mean 0.003824115 / 0.000272003 across
  all 26 dimensions. For the 20 active dimensions, normalized max/mean are
  0.000897913 / 0.000178360; the largest all-dimension differences are in padding.
  Because the FP32 baseline itself drifts, this comparison cannot isolate all
  errors as dtype effects. One-request recorded-trajectory abs14 L1 is
  0.039540580 (FP16) versus 0.039562110 (FP32); this is not a task-quality gate.
  Profiled request times are not unprofiled performance evidence.
- Reproduce the three-Graph path with
  `bash run_wall_qwen35_openloop.sh --language-one-graph`; append
  `--max-requests 1 --torch-profile` to reproduce the READY trace check.
  Append `--action-execution fp32_host` only for the old precision baseline.

## 11. Unified Wall execution switch (2026-09-09)

This supersedes the selectable options in sections 9 and 10; their measurements
remain historical evidence. `WALL_QWEN35_OPT` is the only public Graph-mode
switch. Unset/`1` selects Vision1 + Prefill1 + Action1. `0` restores per-image
Vision, conservative Prefill segments and ten single-step Action invocations
(3+3+10 for this recorded episode). Both arms use the same FP16 Action operators,
ACC32 GEMMs, positive half Euler step and FP32 time/Ada precomputation. There is
no selectable FP32 host mode and no independent Language CLI option.

Outcome: Adapter-only orchestration of the existing packed/per-image Vision and
native one-/ten-step FP16 Action paths; no new operator or asset. The cold
preset owns all six Graph/SDK budget variables, overriding stale individual
values: enabled 32768/8/64 with SDK 65536/16/128; disabled 8192/4/32 with SDK
65536/8/64 (memory values in MiB). Policy binding snapshots the switch; recreate
in a fresh process to change it. Native SDK limits and stream fences remain.
Optimized Vision/Prefill and both Action arms enforce single-segment capture.
The split arm retains up to three Vision shapes and one exact-prefix Action
signature, refreshing step modulation and evolving action through mutable DMA.
Public outputs retain independent CPU FP32 storage.

The runner rejects stale installed switch/FP16/native ABIs before weight loading
and records the switch, budgets and graph plan. READY admission checks physical
submission counts (3 versus 16), in addition to stable signatures, zero recapture,
invariants and repeated-output parity. Cold priming/BUILD is excluded. Validation
results for this revision are recorded below; no release or speedup claim is made.

Validation: `bash -n`, `git diff --check`, and all 295 board-free tests passed.
New tests cover both shell/Python presets, removal of the independent flags,
stale-package rejection, exact physical submission admission and FP16 step
modulation/input refresh with independent A/B/A outputs. The RPU FP16 roundtrip
passed and both end-to-end processes shut down cleanly.

Both public-runner diagnostics used the real first request (prefix 308, grids
8x14 / 10x14 / 10x14), one unrecorded warmup and one frozen-cache measured repeat,
under the exclusive board lease. Common FP32 noise value SHA-256:
`9bff273106fba94a31a550264c18e7186dbea33d73c93286a1c65ddecbaaa0d0`.

| Gate | OPT=1 (default) | OPT=0 |
|---|---|---|
| Physical Vision / Prefill / Action submissions | 1 / 1 / 1 | 3 / 3 / 10 |
| Independent trace `segment_launch` count | 3 | 16 |
| Final retained cache sizes, V / P / A | 1 / 1 / 1 | 2 / 1 / 1 |
| Final replay counts, V / P / A | 1 / 1 / 1 | 4 / 1 / 19 |
| Recaptures / invariants | 0 / all true | 0 / all true |
| Warmup/repeat physical action max-abs | 0, bit-exact | 0.003958910704 |
| Warmup/repeat normalized action max-abs | 0, bit-exact | 0.00146484375 |
| Full READY admission | accepted | numerical repeatability failed |

The OPT=0 graph/lifecycle gates passed, but it is **not** numerically READY.
All compared outputs were finite. Across the two measured modes, returned
physical-action max/mean absolute difference was 0.003958910704 / 0.000108192602.
The split run itself drifts, so this does not establish a graph-merge error or
same-dtype equivalence. Earlier conservative-path drift is separately documented
in sections 9 and 10; this run did not locate its first divergent operator.
No thresholds were relaxed. FP32-anchor, representative task and broader
end-to-end envelope gates remain pending; profiled time is not a speedup claim.

Reproduction (fresh processes):

```bash
bash run_wall_qwen35_openloop.sh --max-requests 1 --torch-profile
WALL_QWEN35_OPT=0 bash run_wall_qwen35_openloop.sh --max-requests 1 --torch-profile \
  --flow-noise /path/to/first-run/flow_noise.npy
```

Local private artifacts: `/dev/shm/rhinoforge-wall-fp16-aaONUI/e2e-opt-on` and
`e2e-opt-off` (ephemeral; tensor/trace artifacts are not committed). Trace hashes:
`77cb4c97b0656b67c2a1143dd87683de3ff79132df7a1ebed397284ac9a67fca` (on),
`3738ebd91ad3522b38cf4e85c17f90e58cecf0500dad2051a055029c8def76f0` (off).
Native hash remains section 10's
`f4dc2a1e29a5ea8791fc36069d2b84391e1d16986edcb066789b80edb77d6f35`;
Launch and the r4 operator/manifest hashes are unchanged. Wrapper/runner hashes:
`9c5f1c21fcc82db6b1fc73c3b2461e633f9fd24ad25fcfdc4785493e55d5be9f` /
`4cda7b195d64a90185c914683887081d63915e2f09f788c73918943311b1d16f`.
The final wheel hash is
`91a8cd8eab69e1a2d16ec792d72d4db2942ffaf60f11dfa2dba033f3107d8406`;
it differs from the OPT=1 diagnostic wheel only in the runtime's module
docstring, and was used for OPT=0. The prior installed package is backed up as
`installed-before-opt.tgz` in the same private directory (SHA-256
`88be2e95060826c07ed6e228078359487ac9c94c344762cb35737508ea77b5ac`).
