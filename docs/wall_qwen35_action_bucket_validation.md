# Wall Qwen3.5 Action prefix buckets

This is a host Graph-runtime extension of the Source-only controlled profile,
not a new device kernel, FP32 implementation, or release-quality certification.

## Execution contract

`WALL_QWEN35_OPT=1` remains the default: one physical Vision, Prefill and Action
segment per request after cold priming. Action rounds real prefix `P` up to
`B = ceil(P/64)*64`, with six admitted buckets from 64 to 384. All ten FP16
Euler steps remain inside the Action graph.

Physical KV is `[prefix 0:P | gap P:B | action B:B+32]`. The gap is zeroed in
both swizzled layouts (K token axis 5, V token axis 6) and hidden with an
additive negative-infinity mask. Zero padding alone is not sufficient: without
the mask it would affect the softmax denominator. Logical checkpoint RoPE
positions are unchanged and do not use `B` as their starting position.

The cache retains all six bucket signatures. Masks occupy per-bucket,
model-owned stable DDR slots, refreshed outside capture even on fast REPLAY.
Every SDPA reloads its mask into a fixed maximum `[32,416]` temporary SPM
allocation (26 KiB/core). All buckets share the same SPM layout; a return to
an earlier bucket does not invalidate that graph. Masked and unmasked paths use
their actual mask type in SDPA scratch sizing and planner validity checks.
For the existing canonical G0.5/Wall horizon-32, head-dimension-256 Action
profile, mask types 0/1/4 all select v16 tiles `(2,16,16)` and 128 KiB/core
scratch. Thus unbucketed Action scratch/layout is unchanged; this does not
admit wider native Action shapes.

`WALL_QWEN35_OPT=0` retains exact-prefix, per-step Action and the previous
3+3+10 organization on the recorded episode. There is no new public switch.
The wrapper rejects older adapters before weight loading, and the adapter
rejects native extensions without the new prefix-mask setter.

## Validation status

Action-only functional/lifecycle verification, two fresh-process full-wrapper
performance runs and the two-request OPT=0 fallback smoke check passed.
Candidate native sources are frozen in private
validation snapshot `bf40faf` (parent `70d7567`), including the existing
profiling CLI consolidation. Subsequent wrapper admission/tests/docs changes
do not change the compiled model or installed adapter sources.

Board-free checks cover bucket boundaries, illegal inputs, K/V swizzle axes,
FP32 attention visibility equivalence, mutable-input ordering, retained A/B/A
cache ownership, output independence, and stale-package rejection.
The current board-free suite passes 342 tests.

The board plan uses the unchanged r4 operator asset, a serial canonical board
lease, fixed input hashes and separate fresh reference/candidate processes.
Action-only tests cover all six buckets and 24 calls, including 1/63/64/65,
299/313/319/320/321/384, changing noise/RoPE, output retention and repeated
inputs. Candidate diagnostics additionally read back actual KV prefixes/gaps
outside timing. Full performance measurements use the public wrapper without
either profiler, the same 16-request episode and identical exported flow noise.

The 24-call Action-only candidate retained six entries, with exactly six BUILDs
and 18 REPLAYs, zero recaptures and invariant true throughout. Every entry had
one segment; the full-wrapper bucket-320 entry had 18,840 nodes and 4,360 kernels.
Six same-input repeat pairs were
bit-identical, including A/B/A returns; actual prefix/gap readbacks and retained
output-storage checks passed. This is functional/lifecycle evidence, not a
parity-certification stamp.

Against the same-input old FP16 Action reference, consumed coordinates 0:20
have max absolute normalized error 0.00439453125, relative L2 0.00145515,
row-cosine mean/p01/min 0.99999897/0.99999401/0.99999060, and no nonfinite
outputs. Padding-only coordinates 20:26 match exactly. Every tested `P=B`
case matches the reference bitwise; nonzero errors appear when the suffix is
relocated across padding. The FP32 visibility-equivalence test and this pattern
are consistent with changed FP16 attention reduction/rounding. Repeat and
A/B/A checks found no stale-input behavior; tighter kernel-level error
attribution and release parity remain pending.

FP16 mask/tiling rounding is not assumed bitwise equal to exact-prefix
attention. Report numerical distributions and baseline variability separately.
This profile has no calibrated release-level action/task tolerance; full FP32
anchor and task certification remain pending regardless of latency results.

## Unprofiled full-wrapper result (2026-09-09)

The pair uses the same checkpoint/episode, 16 requests / 468 evaluated frames,
FP16 execution, OPT=1, and byte-identical flow noise. Both profiler modes are
disabled. The reference is the pre-bucketing installed package; the candidate
is the installed wheel built from the frozen snapshot. No compilation ran
during either full-wrapper timing run.

| Measurement | Exact-prefix reference | Bucketed candidate |
|---|---:|---:|
| Action BUILD / REPLAY across 16 requests | 14 / 2 | 1 / 15 |
| Mean request latency after the first | 540.908 ms | 271.993 ms |
| Median request latency after the first | 579.688 ms | 273.659 ms |
| First request (includes lazy setup/BUILD) | 8.167 s | 7.789 s |
| Sum of 16 measured inference calls | 16.281 s | 11.869 s |
| Explicit policy installation | 100.386 s | 100.615 s |
| Whole script, Bash `time -p` | 260.37 s | 257.69 s |

Post-first-request mean latency falls 49.7% (1.99x throughput at that boundary),
and summed inference time falls 27.1%. The gain is avoided priming/capture/BUILD
when real prefix changes, not fewer Action compute kernels. Initialization,
checkpoint checks/loading and later host work dominate total script time and
were not optimized. Cold-start differences are not a separate speedup claim.

All real prefixes in this episode (299–313) use Action bucket 320. Each stage
retains one entry, executes one physical segment per call, ends at 15 replays,
has zero recaptures, and keeps its invariant true. The runner now enforces
cumulative Action replay rather than accepting repeated one-entry rebuilds.

Same-noise candidate/reference physical 26D output max-abs is 0.02241153;
consumed-coordinate relative L2 is 0.00151822, row-cosine mean/p01/min
0.99999902/0.99998884/0.99995299. Masked coordinates match exactly and all
outputs are finite. Episode absolute-14D trajectory L1 changes from
0.09787918 to 0.09796325. These are observed distributions, **not** a passed
task or parity threshold.

There is also pre-existing full-model variability: this fresh old-version
reference differs from the user's earlier same-native/same-noise run by up to
0.76010895 in physical 26D output (request 11, last row, coordinate 19).
The two old runs share the same checkpoint/launch/operator provenance and
noise bytes. This is not evidence that any candidate error is acceptable or
that bucketing fixes the prior instability. Action-only same-input repeats
were exact; complete full-model attribution remains outside this result.

A second fresh candidate process reproduced 1 BUILD + 15 REPLAY per stage,
268.808 ms post-first mean, 268.103 ms median and 11.922 s summed inference
(first request 7.890 s, installation 100.371 s, whole script 255.56 s).
Across the two candidate runs, post-first means are 268.8–272.0 ms, versus
540.9 ms for the fresh reference. The two full-model outputs are not bitwise
identical: physical max-abs 0.00808287, consumed relative L2 0.000822159;
masked coordinates remain exact. Episode L1 is 0.09791788 in the second run.
The Action-only replay result must not be extended to full-model numerical READY.

The fresh-process OPT=0 smoke test used the first two requests and their same
flow-noise rows. Real prefixes 308 and 310 remained exact-prefix Action keys.
Each request executed three single-segment Vision calls, three Prefill segments
and ten single-step Action calls (one BUILD plus nine REPLAYs per prefix).
All cache invariants stayed true, no recaptures occurred and both relative-26D
and absolute-14D outputs were finite. This checks fallback behavior, not a
separate performance or parity claim.

Raw outputs, input hashes, lifecycle receipts and comparisons are private under
`/tmp/wall-action-buckets-ZT2EPI/`. Reference and candidate native SHA-256:
`f4dc2a1e29a5ea8791fc36069d2b84391e1d16986edcb066789b80edb77d6f35`
and `6ce94ce6915ce7e878b763c5840f699fa47d38cce350a74201d81c7d0088f573`.

## Reproduction

After rebuilding/installing the matching package in the wrapper's environment:

```bash
bash run_wall_qwen35_openloop.sh
# Exact-prefix/per-step fallback, fresh process:
WALL_QWEN35_OPT=0 bash run_wall_qwen35_openloop.sh
```

For a paired run, pass the earlier run's exported `flow_noise.npy` through
`--flow-noise` and choose a fresh `--output-dir`. Do not pass
`--torch-profile-dir` or `--hw-perf-dir` when measuring latency. Check
`segments.json` for `action_prefix_bucket` and cumulative graph counters, not
only a fast wall-clock sample. Cold model admission/installation and the first
BUILD are separate from steady-state inference latency.
