# hxcc optimization playbook

This is an execution order for source-level RPU work.  It turns the developer
guide's hardware-loop, Repeat, address, and asynchronous-unit rules into
measurable candidate hypotheses.  It is not a promise that the installed
runtime asset accepts a new `.ref` or that a board reaches a headline peak.

## Freeze the physical and ABI budget first

Record the exact target/toolchain receipt before changing a kernel.  For the
observed guide/SDK pair, the useful starting budget is:

| resource | starting constraint | review evidence |
|---|---|---|
| parallelism | up to 8 cores; 8 warps/core; 16 threads/warp | Launch queue/core receipt |
| local storage | 8 MiB SPM/core; per-thread VLM has 800 physical 16B entries | allocation and DMA map |
| scalar registers | 40 thread GPRs × 16 bit; 64 warp-shared SGPRs × 16 bit | assembly register/spill review |
| functional units | Scalar, VALU/VSFU, VMAT, and LD/ST are shared within a warp | issue/fence schedule |
| vector type | prefer `f16v16`; budget it as 32 B (two 16 B entries) until emitted layout is checked | assembly/object receipt |
| parameters | compiler uses param0..param61 in the guide; pointers/32-bit values use two 16-bit slots | exact compiler + SDK header |
| overflow arguments | no custom structs as input; host-populated Constant Memory plus `ldc`/`sldc<T>` | frozen byte-offset map |

Keep VLM indices, SPM addresses, and external byte/line offsets in separate
variables.  A `DDR_16B{n}` address is a byte offset of `16*n`; an automatic
`LoopStepInfo` step is multiplied by the selected type's `stepbyte`.  Every
candidate records the chosen granularity and the maximum address for each
operand, including tail tiles.

## Candidate ladder

Change one rung at a time and retain the assembly receipt for the exact clean
commit.

1. **ABI and reference:** compile a synchronous, explicit-loop baseline; check
   output dtype/layout/residency and input immutability before timing.
2. **Tile and residency:** start with the guide's 32x32x64 FP16 GEMM structure
   (16-lane vectors, VMAT FP32 accumulation), then search only contract-admitted
   tiles.  Stage DDR→SPM→VLM once and keep the accumulator in VLM/VMAT; count
   every mandatory transfer even when DMA overlaps compute.
3. **Zero-overhead control:** put each `rpu_hwloop_*` immediately before its
   `for`, use constant/warp starts and constant steps, and inspect for emitted
   `loop` rather than `wjump`.  Reserve eight loop levels; a Repeat consumes
   two.  Use Repeat only for one regular instruction with no branch.
4. **Overlap:** issue independent async VLD/VMAT/VALU/VSFU/VST work, then place
   the matching unit fence before its consumer.  A synchronous operation does
   not need a fence; an async operation without one is a correctness failure.
5. **Address instruction count:** after correctness is stable, try `lpaddr`
   (optionally combined with `add`) and explicit `Tsize`/`Tstep`.  Verify
   `dir`/`add`/`dlt` exclusivity, `dlt`'s use-then-update behavior, 16/32-byte
   granularity, and tail strobes in assembly.  Do not call a source pragma a
   result until the inspector sees the instruction.
6. **Launch and fusion:** only after the bare path is warm, place the epilogue
   on the accumulator/output path and retest launch count, bytes, and Graph
   state.  A host Graph that keeps an intermediate in SPM is a transfer-saving
   schedule, not automatically one fused device program.

At every rung, keep the scalar working set below the exact compiler's TGPR/
SGPR budget as well as the VLM budget.  Short loop/address variables can reduce
register pressure, but an apparent reduction that introduces a spill or an
extra SPM/DDR round trip is a regression.  For multi-core tiles, account for
the warp-shared functional units and use the documented `watom`/`watoms`
protocol when cross-core synchronization is part of the contract.

For the canonical FP16 GEMM, start the roofline ledger with
`required_ops = 2*M*N*K` (one multiply and one add per fused multiply-add),
then add any separately specified epilogue operations.  Derive mandatory bytes
from the frozen DDR→SPM→VLM/output residency map, including tails and scales;
do not substitute tensor size or a rounded profiler bandwidth.  Compare the
measured steady-state latency with both the calibrated compute and transfer
floors, and label the result empirical until the exact board mode supplies an
authoritative peak.

At each rung, run the fast correctness/lifecycle signal before timing.  If a
candidate adds a software `wjump`, an unmatched fence, an over-depth loop, an
unexpected VLM spill, or a new DDR round trip, reject the hypothesis even if a
single host timing happens to improve.

## Profile-specific fusion hypotheses

Use these as separate assignments; do not combine them into an unconstrained
"make GEMM fast" request.

- **GEMM:** `A[M,K] @ B[N,K]^T`, with the registered accumulation and output
  conversion.  Keep VMAT's FP32 accumulator live through the output loop; use
  `outlp1`/`clrlp1` and `lpaddr` only when the emitted addresses match the
  contract.  The bare baseline is one GEMM with identical residency.
- **GEMM+SiLU-mul:** keep gate and up accumulators in VLM, evaluate the exact
  SiLU/rounding order with VALU/VSFU, multiply before either result is stored,
  and write the registered output once.  The bare arm is the two projection
  GEMMs; an additional standalone activation launch or DDR spill is counted,
  not hidden as an epilogue.
- **GEMM+add:** choose bias *or* residual before implementation.  Load a bias
  tile or residual tile at the documented address granularity, broadcast it in
  the warp, and add during output conversion.  Test a same-address changed
  value and a different-address equal-value input so a captured parameter cannot
  masquerade as fusion.
- **GEMM+RoPE:** treat position, cosine, and sine tables as semantic inputs.
  Keep the Q/K pair in VLM, rotate in place (or in a second VLM tile) before
  the single registered materialization, and test first/adjacent/maximum and
  changed positions.  A Q-only experiment is a different profile.
- **RMSNorm+quant_mxfp8:** remain `Blocked` until the exact block-scale
  packing, scale format, rounding, saturation, and asset name are authorized.
  Once supplied, fuse reduction/normalization with payload and scale writes,
  but compare every encoded leaf bit-for-bit and dequantize only for the
  independent FP32 anchor.  The portable int8 fixture in this skill is not
  evidence for MXFP8 hardware support.

### Wall Qwen3.5 action decode (current RPU path)

Treat the Wall action path as its own exact-shape campaign, not as generic
one-token text decode.  The admitted action graph is batch `1`, action length
`32`, hidden `1024`, intermediate `2048`, `24` layers, and full-attention
layers `{3,7,11,15,19,23}`.  A public action request performs ten Euler steps;
the latency target therefore needs an explicit scope (for example, the ten
`wall_qwen35_action_decoder` calls) and must not be mixed with cold model load,
vision, or host preprocessing.

The current graph emits 429 device-program events per action call.  The first
board-free candidate reuses the installed `llama_silu_mul` asset for the Wall
SwiGLU gate/up pair.  It is guarded by the cold process variable
`RPU_QWEN35_WALL_FUSED_SILU_MUL=1`; the default remains the certified
GEMM → SILU → MUL sequence.  The candidate is expected to remove one unary
launch per MLP site (405 events per call, 240 fewer over ten steps), but this
is a hypothesis until a paired board trace confirms numerical parity, DMA
bytes, replay invariants, and fusion tax.

A second, independent diagnostic arm,
`RPU_QWEN35_WALL_PREREDUCE_RESIDUAL_GATE=1`, gates each row-partitioned
`down`/`o_proj` partial before the existing ring all-reduce and supplies the
real residual to that ring's terminal epilogue.  With the exact six full-
attention layers and 24 MLPs this would remove 30 post-reduce ADD launches;
it also skips the now-unused Wall `zero_resid` memset.  The structural
estimate is therefore 398 events per call for pre-reduce alone, or 374 with
`silu_mul` enabled as well.
Because the gate is applied before cross-core FP16 reduction, its rounding
order is intentionally treated as a separate candidate and requires the same
parity gate; it is not enabled by default.

The paired r4 campaign (same six-request flow-noise artifact, fresh process per
arm) measured the following action Graph census and steady whole-request
times:

| Arm | Action kernels/call | Steady request mean | Output parity vs 0/0 |
|---|---:|---:|---|
| 0/0 baseline | 429 | 455.860 ms | reference |
| `FUSED_SILU_MUL=1` | 405 | 433.797 ms | max_abs 0.002417; mean_abs 3.42e-5 |
| `PREREDUCE_RESIDUAL_GATE=1` | 398 | 432.922 ms | max_abs 0.009149; mean_abs 1.62e-4 |
| both arms | 374 | 450.859 ms | max_abs 0.006916; mean_abs 1.50e-4 |

All four arms kept one action Graph entry, 59 replays, zero recaptures, and
`cache_invariant_ok=true`; all outputs were finite and used the same
`flow_noise` SHA-256 `078e697b…`.  The pre-reduce arms fail the current
unfrozen same-dtype parity review and remain diagnostic-only.  `silu_mul` is
the only arm with the smaller observed numerical drift, but its FP16 rounding
change also needs a profile-owned threshold and a larger repeated set before
promotion.  A separate profiler-only 1/0 run reported
`wall_qwen35_action_denoise_loop` CPU 188.291 ms for ten steps (the
screenshot/150 ms scope), while the nested `wall_qwen35_action_decoder`
wrapper was 123.148 ms; the old 0/0 loop/decoder diagnostics were 210.173 /
128.537 ms.  The complete repeated `predict_action_chunk` scope was 439.531
ms.  Profiler/device time is diagnostic (`device_time=0`) and none of these
values passes a board latency gate; the 150 ms denoise-loop goal remains
unmet.

A third candidate was explored in an isolated worktree,
`RPU_QWEN35_WALL_PREFIX_COPY_ONCE=1`, targets the capture-external prefix DMA
boundary rather than the device Graph.  The fixed Wall action contract writes
only the 32-token suffix at `prefix + chunk.offset` (`chunk.offset=0`) for the
six full-attention layers; the independent action K/V cache therefore retains
the physical prefix across the ten Euler steps.  The arm stamps a generation
before each public base-prefill rebuild, copies the six K/V prefixes once, and
keeps the default copy-on-every-call path for direct low-level callers.  It
must retain the independent destination cache, the mutable `prefix_lens`
input, and the existing bucketed Graph signature.  A board-free fake-tensor
test proves 12 copies on the first call, zero on the next nine calls, and 12
again after a generation change.  The expected saving is nine repetitions of
the six-layer K/V prefix DMA per public request; this is a request-boundary
optimization, not a fused GEMM epilogue and not a reason to change the Graph
kernel census.

The r4 board smoke falsified this optimization: with the exact first-request
noise, `RPU_QWEN35_WALL_PREFIX_COPY_ONCE=1` kept the expected 429-kernel Graph
but changed the action result from `abs14 L1=0.039562` to `0.202670`
(`max_abs=1.52877`, `mean_abs=0.114975`).  The arm is therefore rejected and
must remain disabled and is not present in the release path; the suffix-write source inspection was insufficient to
prove that the complete action cache state is safe to reuse.  Preserve the
failure receipt and do not report the theoretical DMA saving as achieved.

### Independent Wall GEMM tile search (2026-09-03)

The release-manifest tile is only a legal starting point.  A separate cold
process sweep was run for each exact local GEMM shape, keeping the compiler,
runtime-v1.0.0-r4 asset, 800 MHz board mode, flow-noise SHA-256
`078e697b…`, request sequence, and Graph checks fixed.  The candidate hook is
default-off (`RPU_QWEN35_WALL_GEMM_TILES`) and accepts only the release-admitted
ACC32 `n_tile` values `{128,112,96,80,64,48,32}`.  Candidate chain:
`93cd5a3` (hook), `b248508` (contract tests), and `b075efd` (runtime-config
documentation), all based on ABI-clean `1413c7f`.

| local GEMM shape | n=128 | n=112 | n=96 | n=80 | n=64 | n=48 | n=32 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[32,256,1024]` | 439.0 | 434.7 | 434.0 | 434.3 | 437.1 | 435.9 | 439.6 |
| `[32,1024,256]` | 440.2 | 435.7 | 434.5 | 434.7 | 431.1 | 434.9 | 439.7 |

Values are steady whole-request means in ms from requests 2–6; the unset
baseline was 431.926 ms.  Every arm was finite, kept the action census at 429
kernels/call, and passed the one-BUILD/REPLAY, fixed-cache, and invariant
checks.  The apparent `[32,1024,256]`, `n=64` lead was not reproducible in a
fresh confirmation (437.2 ms), so no tile is promoted.  This is a negative
optimization result, not evidence that the manifest tile is theoretically
optimal: the remaining uncertainty is below the run-to-run board noise and
requires a longer interleaved campaign before changing the default.

The table above is **whole-request E2E evidence only**.  It includes the Wall
action graph, attention, launches, synchronization, and host-side request
work; it is not a GEMM operator timing and must not be used as one.

### Corrected standalone pure-GEMM panel (2026-09-03)

The operator-only measurement was rerun with the Launch SDK harness
`scripts/pure_gemm_bench.cpp`.  Inputs/outputs were mapped once to per-core
`LocalSPM_t` windows, transformed weights were placed once in `HostDDR`, one
Launch batch was built, and only synchronous resident replays were timed.
Host replay and native Compute duration were recorded separately; `build_us`
and any host materialization were excluded from steady samples.  Every sample
below had zero parity mismatches.  Each tile was run in a fresh process with
five warmups and 40 timed replays at 800 MHz; the Compute column is the
critical path (the maximum of overlapping core streams), not a sum.  The
packet is bias-free (`bias_addr=0`): these rows are GEMM only, not a projection
plus bias/activation result.
The harness can persist unrounded replay samples with `--raw <path>` for
interleaved/paired analysis; those files remain private diagnostic artifacts.

For q/k/v/gate/up (global `[32,2048,1024]`, column partition, local
`[32,256,1024]`, FP16-weight ACC32):

Here q/k/v use the effective post-replication width consumed by the eight-core
Wall asset; the raw logical K/V head width is smaller.

| `n_tile` | `m_tile` | host replay p50 (µs) | Compute critical (µs) |
|---:|---:|---:|---:|
| 128 | 208 | 69.771 | 36.263 |
| 112 | 240 | 68.194 | 35.111 |
| 96 | 272 | 70.578 | 36.028 |
| 80 | 304 | 68.232 | 34.943 |
| 64 | 352 | 71.578 | 36.133 |
| 48 | 416 | 70.309 | 35.020 |
| 32 | 496 | 68.347 | 34.693 |

For o/down (global `[32,1024,2048]`, row partition, local `[32,1024,256]`):

| `n_tile` | `m_tile` | host replay p50 (µs) | Compute critical (µs) |
|---:|---:|---:|---:|
| 128 | 208 | 69.924 | 34.288 |
| 112 | 240 | 69.347 | 34.140 |
| 96 | 272 | 67.232 | 34.182 |
| 80 | 304 | 71.155 | 34.106 |
| 64 | 352 | 68.540 | 34.014 |
| 48 | 416 | 67.040 | 33.754 |
| 32 | 496 | 67.886 | 33.761 |

The current release selector remains `n=112/m=240` for q/k/v and
`n=32/m=496` for o/down.  Across two diagnostic trace rounds, the individual
minima alternate between qkv `n=32`/`n=48` and o/down `n=48`/`n=32`; the
spread versus the release choice is below about 1.2% and is within observed
board/run noise, so no override is promoted.  These are seven prebuilt
release variants, not a proof that every source-level loop, pipeline, or
address schedule has been searched.  The board has no frozen authoritative
peak in this campaign, so the achieved TFLOP figures and any roofline are
empirical.

As a three-GEMM sanity check, q, k, and v were run independently with the
selected qkv tile (`n=112/m=240`, 30 replays, fresh process, deterministic
seeds; a separate 50-replay panel was consistent):

| projection (seed) | parity max-abs | host replay p50 (µs) | Compute critical (µs) |
|---|---:|---:|---:|
| q (11) | 2.43187e-4 | 67.924 | 35.005 |
| k (17) | 2.17915e-4 | 69.002 | 35.292 |
| v (23) | 2.33114e-4 | 69.232 | 35.015 |

The seeds change synthetic operand values only; q/k/v use the same shape,
layout, ABI, and device program.  Therefore these three rows are independent
operator measurements, not three different kernel implementations.  They also
do not establish GEMM+SiLU, GEMM+add, GEMM+RoPE, or attention fusion: the
release asset currently contains no authorized one-launch epilogue fusion.

### Independent Wall attention candidate (2026-09-03)

Attention was measured as a separate task, not folded into either GEMM
epilogue arm.  Candidate `2ba4620` selects the release asset
`llm_fp16_32b_prefill_flash_attn_univ_vctxlen` only during cold Wall handle
setup (`RPU_QWEN35_WALL_SDPA_VCTXLEN=1`); its candidate object receipt was
`32a5b29c…`.  With the same r4 runtime, 800 MHz mode, six-request flow-noise
artifact, and fresh process per arm:

| arm | steady requests 2–6 | overall abs14 L1 | action Graph |
|---|---:|---:|---|
| standard DP SDPA | 431.520 ms | 0.120425233669 | 429 kernels, 59 replays |
| vctxlen SDPA | 433.721 ms | 0.120425233669 | 429 kernels, 59 replays |

The vctxlen output was bit-exact to the baseline (`max_abs=0`, `mean_abs=0`,
finite outputs); both arms had zero recaptures, cache size one, and true cache
invariants.  It is nevertheless rejected: the wrapper writes K/V DDR
registers directly (`set_regs(50/53)`) and has no typed
`stage_kernel_ddr_registers()` provenance receipt.  It also loses 2.201 ms
(+0.51%), so neither performance nor the ABI/provenance gate supports changing
the default.  Keep the environment opt-in and the source-contract test as a
regression guard until the missing provenance path is implemented and a new
interleaved campaign proves a gain.

Do not call this GEMM-epilogue fusion: `llama_silu_mul` is a standalone
elementwise device program placed between two existing GEMMs.  The release
manifest currently has no authorized one-launch GEMM+add, GEMM+RoPE, or
GEMM+epilogue asset.  A host Graph that keeps intermediates in SPM is a
transfer-saving schedule, not proof of a fused device kernel.  Promotion of
the Wall arm requires a clean candidate commit, release-matched Launch/asset
receipt, interleaved bare/candidate measurements under one board lease, and a
reported upper confidence bound on the epilogue tax.

## What “near-zero epilogue” means

Use a paired, interleaved comparison with the same shape, output tree,
residency, warmup, and replay transition:

```text
tax_pct = 100 * (fused_latency - bare_latency) / bare_latency
```

The claim is admissible only when both `candidate` and `bare_operation` arms
independently show BUILD=1, a warm REPLAY that grows, fixed cache size, and true
invariants.  Register a confidence level and threshold before measuring; call
the epilogue *indistinguishable at the registered resolution* only when the
upper confidence bound of paired tax is below that threshold.  Report launch
count, DDR/SPM bytes, absolute delta, raw samples, and the bound.  Never call a
rounded median or an uncalibrated CPU result literal zero cost.

## Delegated iteration schedule

The orchestrator assigns one profile, one exact shape set, and one primary axis
to each optimizer.  A practical first round is: tile/residency, hardware-loop
legality, Repeat eligibility, async double-buffer depth, address/lpaddr map,
then epilogue placement.  The optimizer commits before measurement; a separate
board runner checks out that commit and owns the canonical lease.  Compile and
assembly inspection may run in parallel, but Launch/Graph measurements never
share a live RPU process or execute concurrently.  After three non-improving
attempts, re-profile and choose a different rung.  Record failed builds and
blocked hardware gates as results rather than silently skipping them.
