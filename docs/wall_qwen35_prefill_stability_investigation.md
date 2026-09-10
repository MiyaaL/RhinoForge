# Wall Qwen3.5 Prefill stability investigation

Status: **unresolved; not a numerical or performance admission** (2026-09-10).
The controlled profile is the existing FP16/ACC32 Wall checkpoint, eight cores,
the r4 operator asset, and the release-matched Launch library. These diagnostics
do not change the public support status or enable a new default execution path.

## What is established

After the [GDN padding correction](qwen35_gdn_padding_safety.md), independently
retained CPU snapshots still expose intermittent differences in both the
128-row and whole-bucket Prefill paths. Thus chunk merging is not a necessary
condition for the remaining instability. Snapshot references must use an
independent clone: a contiguous RPU-to-CPU conversion can alias shared DDR.

One whole-bucket run localized the first recorded difference to a GDN input
RMSNorm output: a single FP16 sign bit changed, while the captured residual
input matched. The following convolution consumed that changed value. Equality
of a layer's final recurrent state or short convolution carry does not establish
equality of all earlier tokens in that layer.

A separate standalone host-driven Launch verifier reproduced intermittent corruption with
just the released `llama_rms_norm`, shape `[320,2048]`, epsilon `1e-6`, eight
cores, and one repeatedly submitted batch. It does not load a model or use
RhinoForge Graph capture. Valid input rows and effective weights came from an
independent healthy model snapshot; padding rows were zeroed for this probe.
The effective weights follow the actual installation order: convert raw
weights to FP16, then add one in FP16.

- Repeated computation produced changed finite FP16 values, including sign-bit
  flips, despite unchanged input/weight snapshots and intact buffer guards.
- NaN-poisoning the destination before each call did not eliminate the failure.
- An independent SPM-to-DDR DMA read reproduced the bad core-0 mapped output.
- Moving the output by 256 bytes did not eliminate the failure.
- A no-compute, mapped-SPM resident-read control completed 1,000 repetitions
  unchanged. This is **not** a DMA-under-load or hardware-health test.

These observations rule out ordinary deterministic FP16 rounding and establish
a failure below the model/Graph orchestration boundary for this reproducer.
They do **not** distinguish the released device program from the SDK/driver,
SPM access path, or physical board. Do not label the board faulty without that
additional evidence.

A frozen address-placement verifier then kept weights/output fixed and moved
only the input by 0, 4 KiB, and 64 KiB. Its base-kernel runs failed at calls 42,
4, and 68 respectively. After the 64 KiB move, errors still occurred at logical
rows 31–32, including a previously failing element's sign bit; core-0 DMA again
confirmed the bad result. This argues against a single fixed physical input
cell as the sole explanation; bank/address aliasing and periodic hardware
effects are not excluded. It does not identify which internal compute/storage
component is at fault. These calls include guarded host staging and are not
latency samples.

The newer shape verifier's default `CopyMemory` loop did not independently
validate other SPM banks: all eight DMA exports matched the core-0 mapped
output, including when another core differed. Do not treat those other-core
DMA reads as confirmation or rejection of corruption. A future verifier needs
per-core sentinel validation of the explicit-channel managed DMA path used by
the production scatter wrapper. This does not invalidate the earlier positive
core-0 DMA confirmation.

## Rejected workaround

The existing `llama_rms_norm_v16` completed 1,000 bitwise-matching repetitions
of the `[320,2048]` out-of-place probe. However, a fresh full-model process with
`RPU_RMSNORM_VWARP=16` and `RPU_RMSNORM_NEWTON=0` failed at its 68th Prefill call,
length 256. Final hidden relative L2 changed by 0.0032119234; the first stored
K/V difference was at layer 3. This is not sufficient to attribute that second
failure to RMSNorm itself.

V16 therefore remains a diagnostic arm, **not a complete fix or a default**.
Extra diagnostic taps can alter timing: a later instrumented run completed
195 calls with no differences in its bucket-256 GDN taps, while a length-159
call without those bucket-specific taps still differed. Neither negative result erases the positive
production-path reproductions.

## Separate timeout symptom

The earlier roughly 3.1-second Prefill anomaly coincided with one driver task
reporting timeout counts 1, 2, and 3. The synchronous SDK submission returned
success despite the bad result. Existing Graph submission code already checks
the returned status. Adding the same check again, a latency cutoff, or a silent
retry is not a root-cause repair. The current release exposes no documented
terminal-task-status/cancel-and-drain path for this situation.

Subsequent numerical failures also occurred at ordinary call latency with no
driver timeout. The timeout and numerical symptoms must not be assumed to have
one proven cause. No valid performance improvement is claimed while these
correctness gates fail.

## Evidence and next boundary

Private host verifiers, frozen input hashes, all failed samples, and run records
are retained under `/tmp/wall-prefill-fix-NX9Usp`; model snapshots are in the
private NVMe campaign directory of the same name. Activations, weights, addresses,
opaque assets, and raw traces are intentionally excluded from this repository.

The fast standalone verifier source SHA-256 is
`ac8996118586eed80b8212997b4d1beca64e3c5d17c060d199d7945861e8b98a`.
The immutable release asset SHA-256 is
`538f49e256814d104c98aa8df4b6d3af3d89b2d636a1fe22b0731df3ce39ac99`.
The clean uninstrumented whole-chunk source tested is `5ec9163`.

Further diagnostics must distinguish the device implementation from the
SDK/driver/board using an authorized matching source or lower-level diagnostic
interface. A genuine repair needs a passing shared-input operator
test followed by full-model numerical, bucket-history, output-lifetime and Graph
lifecycle checks. Only then should unprofiled wrapper performance be compared.
If the failure requires an operator/SDK/driver replacement, obtain the matching
implementation and supported delivery path; do not bypass restricted SDK APIs
or substitute an unvalidated kernel.

## Authorized source iteration (2026-09-10)

The first source candidate is isolated in an `rpu_ops` worktree at commit
`61ae973b6c8a60fb89a71dd06cda77418fc3e59f`. Kernel implementation and generated
assets remain outside RhinoForge. The optional host selector is
`RPU_SOURCE_OPS=rmsnorm`, requiring a matching source SDK and rebuilt native
backend; it is **not enabled by default**. The Wall wrapper now defaults to the
approved sibling `rpu_ops/runtime` REF and Launch instead of historical external
operator packages. Explicit dependency overrides remain available.

The new independent verifier establishes all eight DMA-bank identities with
distinct per-core sentinels, changed data and reversed submission. Its initial
host reader faulted in libc `memcmp` on mapped device SPM; using aligned scalar
SPM snapshots before host comparison fixes that instrumentation defect. This
is separate from the previously observed compute instability. All original
failed probes are retained.

The unchanged released baseline passed a new 32-call real-input A/B/A short
test with guards and independent eight-core DMA. This does not supersede its
earlier long-run failures. The first source call passed finite-output,
input/weight preservation, guard, same-dtype CPU and both FP32-anchor gates.
Its FP32-anchor relative L2 was 0.0004071 versus the baseline's 0.0011928.
However, one corresponding element on every core differed from the healthy
released output by 0.03125, exceeding the pre-registered compatibility bound
0.0278984. This consistent difference is ordinary implementation parity, not
evidence of the earlier intermittent bit change being fixed.

The candidate therefore stops at the compatibility gate: no long-run stability,
Graph/end-to-end admission or performance claim follows from that first call.
The numerical thresholds have not been relaxed. Operator and Wall steady-state
performance must both pass a no-regression comparison before default selection.
Private source-build receipts, immutable oracles and all results are under
`/tmp/rpu-rmsnorm-campaign-BH6vPY`.

The second source candidate, clean commit
`2884291c3a01f8917a4c372293d27b0183b8e4d0`, attempts one reduction for normal
variance and a second scaled reduction only for tiny variance. Its first device
call failed the independent mathematical and finite-output gates, not just old
output compatibility: FP32-anchor relative L2 was about 0.57635 and 49,152
outputs were nonfinite. Guards, inputs, weights and all eight DMA comparisons
passed. The inferred normalization factor closely tracks the block's Gram
column zero rather than the own-row diagonal. Because the Gram matrix is
symmetric, that observation alone does not identify which scalar/vector axis
was selected incorrectly. A separate non-symmetric interface probe is required.
This deterministic candidate defect is distinct from the original intermittent
corruption; neither candidate is admitted or enabled by default.

An independent non-symmetric interface probe subsequently identified a working
extraction path across all eight cores and two input epochs, with 12 repeated
calls and intact guards/DMA. The minimal V2b correction is clean source commit
`a6540ec67c58a7da2b04fbcc8c1130125c0fbcb3`. On the same real-input first call,
nonfinite outputs are now zero and the unchanged staged-CPU and both FP32-anchor
gates pass. Encoded-anchor relative L2 is 0.000412699. The original healthy-Y
gate still fails at two corresponding elements per core (16 total); each differs
by 0.03125, despite being closer to FP32 than the old output. No gate is waived.

Seven separate small-value, independent-core and mixed-row branch signals each
completed 32 calls, with zero repeat-bit or unchanged-neighbor differences and
all their mathematical, finite, input and guard gates passing. These 224 calls
are synthetic branch diagnostics, not a passing real-input long-run test. The
real-input run stopped after its first compatibility failure. Original random
instability, full Graph/model admission and operator/end-to-end no-regression
remain unproven. No timing or default replacement has occurred. Resolving the
old-output compatibility policy requires user direction, not tolerance tuning.

### User-authorized FP32-only acceptance

The user subsequently clarified that old-output pointwise compatibility is not
required; acceptable accuracy is measured against FP32. A new frozen acceptance
contract and V5 independent evaluator therefore keep healthy-output and staged
CPU differences as report-only diagnostics. Both existing FP32 anchor bounds
remain unchanged, as do finite-output, repeat-bit stability, memory/DMA,
Graph-lifecycle and performance no-regression requirements. Earlier failed runs
are retained under their original rules, not retroactively reclassified.

Under the new evaluator, a fresh V2b signal completed all eight cases, 32 calls
each, including the captured Prefill input. Its captured-input encoded-FP32
relative L2 is approximately 0.0004127 (0.04127%). All short-test FP32 and hard
control gates passed. This permits long-run validation; it is not yet evidence
of full-model stability or performance non-regression. Source selection remains
opt-in, and neither the system installation nor the paused service is changed.

The user then requested stopping prolonged testing and accepted the accumulated
stability evidence for continued engineering. At interruption, the first
captured M320/C2048 process had completed 989 logged calls with zero recorded
bit-repeat, finite-output or memory-control failures. The process and its lease
have exited. This run is **stopped by user**, not a completed 3-process/1000-call
panel: its final DMA check and the remaining processes were not run. Do not
restart the prolonged panel under the previous plan. Graph/model accuracy and
performance evidence remain distinct from this stability observation.
