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
`wall_qwen35_action_decoder` CPU 123.148 ms for ten steps versus the old 0/0
diagnostic 128.537 ms; profiler/device time is diagnostic (`device_time=0`)
and not a board latency gate.

A third diagnostic arm,
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
must remain disabled; the suffix-write source inspection was insufficient to
prove that the complete action cache state is safe to reuse.  Preserve the
failure receipt and do not report the theoretical DMA saving as achieved.

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
