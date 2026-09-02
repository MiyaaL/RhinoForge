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
