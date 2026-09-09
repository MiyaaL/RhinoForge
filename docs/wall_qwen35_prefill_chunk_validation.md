# Wall Qwen3.5 whole-bucket Prefill

This is a bounded optimization of the existing Source-only controlled profile,
not a new device kernel or a numerical/release certification.
The candidate is **not promoted**: one full-wrapper run encountered a driver
task timeout and a corresponding prediction anomaly. The running installation
is not replaced by this candidate.

## What changed

In this candidate, `WALL_QWEN35_OPT=1` uses one outer compute chunk per
64-row prefix bucket, through 384 rows. The recorded episode's bucket 320 now
runs as 320 rather than `128 + 128 + 64`. One physical Graph segment alone did
not previously remove those outer chunks. Gated DeltaNet's internal 64-row
recurrence remains part of the existing algorithm.

The Wall adapter sets its native chunk envelope/maximum before the first
forward. The maximum is clamped to each actual bucket, not used to pad every
request to 384. Wide chunks separate mutually exclusive Full Attention and
GDN scratch lifetimes; shared residual/MLP lifetimes and persistent ownership
remain explicit. The native SPM guard still rejects an overflow. One outer
chunk also selects the existing SPM-resident inter-layer I/O route, so a
speedup cannot be attributed solely to fewer host launches.

Generic Qwen3.5 and `WALL_QWEN35_OPT=0` retain their 128-row admission ceiling.
The latter still uses the recorded episode's 3+3+10 Graph organization.
No new public switch, FP32-to-FP16 conversion, or operator asset was introduced.

## Repaired baseline and numerical evidence

Investigation exposed the existing [GDN padding-fill overrun](qwen35_gdn_padding_safety.md).
Original revision `597d81a` and the first candidate are diagnosis-only artifacts:
their outputs/timings are not accepted optimization references. Both measured
arms include guarded fixed-length padding writes and the ABI-sized cumsum
workspace. Baseline A is `6dfc603`; candidate B is its sole child `5ec9163`.
Both production wheels exclude temporary localization taps.

Board-free validation passed 229 tests. Each final package completed the same
39 frozen-input cases, including bucket/tail boundaries, 299/308/313/320/321/384,
changed same-shape inputs, and cross-bucket A/B/A returns. Vision, language and
Action persistent allocations were primed together before maximum-size tests.
Native resolution admitted every bucket 64..384 as one outer chunk in B.
Consumed hidden outputs, six full-attention KV pairs, 18 GDN states and meaningful
convolution carry were finite. Each retained entry had one segment, zero
recaptures and an intact cache invariant. A ended with 11 entries/30 replays;
B ended with 7 entries/34 replays, including two initial policy calls.

For frozen real Prefill inputs, hidden-output relative L2 against repaired A was
0.004599 at 308 tokens and 0.005378 at 313 tokens; cosine was 0.9999894 and
0.9999855, respectively. These are measurements, not declared pass thresholds.
Lengths through 128 and the tested 193..256 cases matched bitwise. Across the
seven repeat pairs, maximum hidden relative L2 was 0.003798 within A and
0.003711 within B. Whole-model bitwise repeatability must not be claimed.

The 191-token synthetic case in the 39-case sequence is an unresolved outlier:
A/B hidden relative L2 was 0.02002 and layer-16 GDN-state relative L2 was 0.5087.
Fresh A itself differs from that original A state by about 0.5087; therefore
whole-chunk execution is not necessary to reproduce the discrepancy. A separate
fresh-process `191,191,160,191,192,191` sequence had maximum A/B hidden relative
L2 0.003629 and aggregate GDN relative L2 0.001458. Its B repeats were bitwise
identical; A repeats reached hidden relative L2 0.003146. This narrows the issue
to a baseline-local, potentially history-dependent anomaly but does not establish its cause or
excuse it as rounding. Full-envelope parity remains open.
An additional fresh `160,191,191,192,191` sequence tested first BUILD at 160:
both arms' repeated 191 outputs matched bitwise within that run, and maximum
A/B hidden/GDN relative L2 was 0.003429/0.001353. Thus this short first-BUILD
history alone did not reproduce the original 39-case anomaly.

Private first-layer diagnostics after the padding fix matched bitwise between
128 and 320 through projection, convolution, normalized Q/K, gates, cumsum,
matrix outputs and all five recurrent states. That local result does not
extend to every layer. CPU-FP32 anchor, calibrated same-dtype/task thresholds,
and representative task-quality gates remain pending; the numeric-blocked
Vision opt-in and Source-only status are unchanged.

## Performance protocol

Use the unchanged public open-loop script, the same checkpoint `0_200000`,
recorded 16-request/468-frame episode, FP16 profile, OPT=1 and 800 MHz clock.
Both profiling-directory flags are absent. Fresh-process order is A1,B1,B2,A2;
all exported flow-noise arrays must match byte-for-byte (seed 3407). No compiler
or other board test overlaps these runs; a single canonical board lease covers
the sequence. Package selection uses isolated final-wheel installations.

A frozen external observer times the synchronous original `GraphCache.capture`
scope, forwarding the unchanged scope body/enter/exit. It does not replace
operators, record a profiler trace or read back intermediate tensors. Report
that boundary as Graph-scope wall time, not pure hardware kernel time. Public
request latency also includes preprocessing and host work. Pool requests 2..16
from each arm's two runs; report BUILD, installation and Bash `time -p` separately.

Raw timing, activation and trace artifacts remain outside Git in the private
`wall-prefill-merge-SCmpB1` campaign directory.

## Observed timings and failed stability gate (2026-09-09)

The following describes all 30 post-first-request samples per arm, including
the failed sample. It is not an accepted speedup benchmark: a parity/execution
failure invalidates promotion from that run. No slow sample was silently dropped.

| Observed boundary | Repaired 128-chunk A | Whole-bucket B |
|---|---:|---:|
| Prefill Graph-scope median | 122.818 ms | 54.307 ms |
| Prefill Graph-scope mean | 123.134 ms | 155.570 ms |
| Prefill Graph-scope p95 | 124.324 ms | 54.744 ms |
| Prefill Graph-scope maximum | 135.297 ms | 3097.445 ms |
| Full-request median | 274.844 ms | 208.088 ms |
| Full-request mean | 276.151 ms | 309.745 ms |
| Full-request maximum | 290.594 ms | 3257.021 ms |

Typical Prefill latency is lower by 55.8% (2.26x at the median), and median
full-request latency is lower by 24.3%. However, including the timeout makes
mean Prefill and request time worse by 26.3% and 12.2%. Thirty observations are
too few for p95 to describe a single extreme tail reliably; maximum and failure
count are therefore shown explicitly.
Exactly 1 of B's 30 post-first requests stalled; the three driver timeout
counts describe that single task, not three separate failed requests.
The second B run had no such stall:
its 15 Prefill replays averaged 53.821 ms. This does not retroactively validate B1.

| Fresh process | A1 | B1 | B2 | A2 |
|---|---:|---:|---:|---:|
| Whole script (`time -p`) | 58.64 s | 60.87 s | 59.22 s | 58.79 s |
| Explicit installation | 17.118 s | 17.571 s | 18.450 s | 14.894 s |
| First request, including lazy setup/BUILD | 2.784 s | 2.780 s | 3.302 s | 2.571 s |
| Episode absolute-14D L1 | 0.119455 | 0.124035 | 0.119497 | 0.119474 |

Whole-script time did not improve in these trials. Startup/host work remains
outside the Prefill merge, and the timing observer is identical in all arms.
The episode metric is a diagnostic against recorded actions, not an FP32 model
anchor or a calibrated quality pass.

B1's thirteenth request (prefix 303) took 3097.445 ms inside Prefill. The driver
recorded task-timeout counts 1, 2 and 3 for the same process/task at local
20:38:39..20:38:41. That request's absolute-14D L1 was 0.211010 versus 0.031378
in B2 with identical flow noise. No timeout exception reached the public runner;
its finite-output and Graph-invariant checks alone did not detect the bad
execution. These facts establish a driver-visible stalled execution with a
numerical anomaly, not its underlying SDK/kernel/hardware cause. Private timeout
register dumps are retained outside Git. The earlier 191-token discrepancy had
no corresponding timeout record and is tracked separately.
The host replay path already checks the SDK enqueue return code for zero;
investigation must include why driver-visible failure did not reach that check,
not simply add another unchecked retry or suppress the bad output.

All four runs retained exactly one entry per stage, with one BUILD, 15 REPLAYs,
zero recaptures and true invariants. Prefill changed from 24,367 nodes
(10,281 kernels + 14,086 data nodes) to 10,144
(5,611 kernels + 4,533 data nodes). Its signature changed from chunk 128 to 320;
Vision and Action kernel counts stayed at 682 and 4,360. Fewer graph nodes and
passing lifecycle counters do not establish correct execution after a timeout.

Before promotion, resolve/reproduce the stalled execution, establish an error
propagation path for device failure, rerun independent numerical and repeated
input checks (including first-BUILD history), then repeat a fresh profiler-off
paired benchmark. Current disposition: implementation and structural/lifecycle
evidence available; performance/stability and full numerical promotion blocked.

## Separate final diagnostics

After the failed-run investigation, a fresh FP16 RPU roundtrip passed. The
additional first-BUILD-history test above, two-request hardware-trace run and
two-request OPT=0 smoke run completed without new driver timeout records.
These diagnostic runs are not included in the timing table.

The OPT=1 trace contains Vision/Prefill/Action BUILD followed by REPLAY, each
with `seg0` only. Prefill has 5,611 Compute intervals in either phase; the raw
Chrome file expresses them as 11,222 begin/end records, not twice as many
kernel launches. Files share a local-time session timestamp and consistent
`rpu_wall_qwen35_{vision,prefill,action}` names. OPT=0 retained 3+3+10 on both
requests and all reported cache invariants remained true. A successful smoke
run does not resolve the earlier timeout or establish full repeatability.

## Frozen identities

- A wheel SHA256: `7125cfc61263ff996504162bbff39cb3e2a368aa6dda6cf2070e48978d46635e`.
- B wheel SHA256: `9e3abc1fd0689588f455e8b7083fc5ec817226744af9c32d74bfe45ac3799122`.
- A native SHA256: `c960fb3839babb527cfa0344fb01669e30647f6c82eaa1b4e48c62f85853b594`.
- B native SHA256: `7bdbfcd98af0639f45cc8e957406a70cee16ed8900a0e56e43c05fa55776843d`.
- r4 operator asset SHA256: `538f49e256814d104c98aa8df4b6d3af3d89b2d636a1fe22b0731df3ce39ac99`.
- Launch library SHA256: `10910756d373171c74ce80ecfe590324c9e2db32788a6367034a310b771772a6`.
- Frozen timing observer SHA256: `becf588387f761656d0924d82a0d75645cfbce9755885246fdbb403c31614d22`.
- Frozen 39-case harness SHA256: `0854c1cc5f87fcae2ad37ef2c590ecf632b8d4f1d75b44ade1dae4afc37b19b7`.
