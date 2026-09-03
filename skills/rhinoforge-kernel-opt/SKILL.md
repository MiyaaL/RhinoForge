---
name: rhinoforge-kernel-opt
description: Run evidence-driven optimization campaigns for RhinoForge RPU operators and fused inference paths, including the local hxcc/Rhino Launch toolchain and assembly-level loop/fusion checks. Use for exact-shape RPU kernel benchmarking, profiling, fusion-cost analysis, Graph/SPM/DMA tuning, candidate iteration, or theoretical/empirical roofline work. Do not use for CUDA-only kernels or for claiming a new device kernel when the authorized RPU compiler, source interface, board access, or release-matched operator asset is unavailable.
---

# Optimize RhinoForge RPU kernels

Treat one exact operator profile as the unit of work. A family name such as
"GEMM" is not a benchmark contract.

## Select a mode

- **Assess** inventories the operator semantics, workload envelope, reference,
  board access, compiler path, operator-asset manifest, profiler, peak
  calibration, and validation commands. Make no kernel or RhinoForge changes.
- **Campaign** requires an accepted assessment with an executable reference,
  immutable harness, usable RPU, and a legal build or tuning path. It creates
  and measures candidates.
- **Resume** reads the contract, Git history, local diagnostic journal, and any
  externally signed release proof before doing new work. Never infer prior
  results from prose or treat a user-owned journal as tamper-proof.

If an assessment lacks device access, an authorized compiler/source interface,
or a required operator in the release-matched asset, stop the affected device
work as `Blocked`. Continue only with board-free harness or skill work, and keep
hardware gates explicitly pending.

Read [the RPU runtime boundary](references/runtime-boundary.md) for every mode.
For Campaign or Resume, also read
[the measurement and promotion protocol](references/benchmark-protocol.md).
When a candidate compiles `.rc`, inspects generated RPU assembly, or uses
Rhino Launch, also read the
[hxcc/RPU developer-guide overlay](references/hxcc-manual.md).  It is a
version-pinned, clean-room digest of the local official manual and does not
widen RhinoForge's release profile.
For the concrete tile → loop → pipeline → epilogue sequence and the five
requested operation profiles, use the
[hxcc optimization playbook](references/hxcc-optimization-playbook.md).
Before any release selection, read
[the external release-attestation boundary](references/release-attestation.md).
Read [the kernel-family contracts](references/kernel-profiles.md) only for the
requested operation. When delegating, read
[the subagent protocol](references/subagent-protocol.md).

## Bootstrap a campaign

Use RhinoForge's configured environment, then create a non-overwriting
workspace from one exact profile:

```bash
source /home/hx/miyaa/work/env.sh
cd /home/hx/miyaa/work/src/RhinoForge/skills/rhinoforge-kernel-opt
python scripts/bootstrap_campaign.py \
  --workspace <campaign-dir> --profile 'gemm+rope' --git-init
```

The bootstrap intentionally does not invent `<campaign-dir>/runner.py` or an
RPU adapter. Every adapter requires `make_case(workload, seed, device)`,
`reference(case)`, `candidate(case)`, and `synchronize()`. A fusion profile
also requires `bare_operation(case)` plus the independent
`bare_reference(case)` and `arm_graph_state(arm)` for the exact arms
`candidate` and `bare_operation`; the bare output tree, residency, strides, and
layout must match the profile's registered contract. Every exact `--profile`
adapter also requires `fp32_anchor(case)` and `graph_state()` because the five
shipped contracts require both gates. Only a generic CPU harness without
`--profile` may omit them; a missing exact-profile hook records a failed gate
and prevents timing. `rmsnorm+quant_mxfp8` additionally requires
`dequantize(output, case)` and `quantized_output_roles()`. Freeze all command
argv entries around that immutable evaluator. `assets/demo_cpu_adapter.py` is
only a CPU protocol smoke test; it is not an RPU kernel or performance reference
and cannot support a performance claim. Run `python scripts/bench.py --help`
for the full interface.

With `--git-init`, only `<campaign-dir>/solution` becomes a Git repository.
`candidate_root` must equal that isolated Git top level; the contract, adapter,
reference, traces, and receipt journal stay outside the optimizer-writable
repository and are protected/tracked by the orchestrator separately.

Review every sentinel and semantic field, retain reviewed hashes in the
campaign receipts, and change `state` to `approved` only last.
`campaign.reference_tree_sha256` binds the symlink-free reference contents
(excluding `.git` metadata); preflight checks it, and both signal and full
launchers check it before and after execution. For a fusion,
`runtime.fused_kernel` must be one resolved, exact member of
`runtime.required_kernels`; token or substring inference is forbidden. A
missing runner, unresolved fusion name, compiler authorization receipt/hash,
compiler binary hash, or asset owner must remain a failed gate.

Compute the supported reference identity, then copy its single digest into the
contract before approval:

```bash
python scripts/hash_reference_tree.py <reference_root>
```

Do not call `campaign_common` internals to manufacture this field.

The local full-diagnostic launcher executes only `commands.benchmark_adapter`
and requires `commands.profile_one_call` to name that same runner; freeze its
exact path and SHA-256. Supplying a different adapter is an identity failure.
Also freeze the external release authority's Ed25519 public key, signer ID,
policy hash, signer-build hash, and local OpenSSL verifier binary SHA-256. The
corresponding private key must be outside the campaign UID and candidate trust
boundary.

The templates deliberately use `attestation.mode = "unavailable"`. That mode
allows approved local diagnostics after every other contract field is resolved,
but `select_best.py` always rejects it. Change to `external` only when a real
separately administered authority and trust anchor exist; never substitute a
locally generated test key merely to clear the release gate.

## Freeze the contract

Record all of the following before the first candidate:

1. Mathematical function, output tree, accumulation order and dtype,
   rounding/saturation rules, layout, aliasing, and mutation rules. Every
   profile has exact output device residency and stride/layout gates; input
   immutability covers storage identity, device, strides, and bytes.
2. Every exact workload: shapes, strides, batch/head geometry, positions,
   quantization block size, semantic inputs, and admitted edge cases.
3. Same-dtype reference and a separate FP32 anchor. Freeze `max_abs`/`max_rel`
   for the same-dtype comparison independently from
   `anchor_max_abs`/`anchor_max_rel`. Quantized output requires the frozen
   payload/scale tree plus bit-exact and stride-exact encoded leaves; a single
   `allclose` is not sufficient.
4. RhinoForge commit, PyTorch, board/driver, Rhino Launch, operator asset and
   kernel-manifest identities, plus the hxcc wrapper *and* underlying
   `clang-17`/`rpuas`/`rhino_gen_oplib` identities, target triple, `-O2`, and
   `-m r1` chain evidence.
5. Correctness, lifecycle, performance, stability, roofline, and fusion-tax
   gates, including the commands that produce each receipt.
6. The candidate-owned paths and evaluator-owned read-only paths.
7. Assembly/source inspection receipts (hardware-loop/Repeat/`lpaddr`/fence/
   `wjump` counts) and the selected Launch-vs-Graph lifecycle.
8. The external release authority's public key, signer identity, policy/build
   hashes, and challenge/sequence owner. Never store its private key here.

Use an approved contract rather than filling missing facts with nearby model
or kernel assumptions. The template under `assets/campaign/contract.toml` is a
draft until every field is reviewed.

From this skill directory, run the deterministic preflight before Campaign
mode:

```bash
python scripts/board_lease.py -- \
  python scripts/preflight.py <workspace>/contract.toml \
    --rhinoforge-root /home/hx/miyaa/work/src/RhinoForge \
    --output <workspace>/runs/preflight.json
```

The CLI always performs the FP16 `torch.rpu` host-device-host roundtrip; it is
not optional. When an approved contract genuinely requires device compilation,
pass the explicit compiler with `--device-compiler`. The contract also binds
the compiler binary SHA-256 and a reviewed authorization receipt SHA-256;
executable presence or a matching version string is not authorization.

The preflight parses only the adjacent public `.kernels` manifest. It may stream
the opaque `.ref` into SHA-256 to verify its frozen identity, but must never
parse, decode, log, rewrite, split, or copy the asset.

## Source-first kernel rule

Do not conflate an executable binary with the implementation that produced it.
There are two deliberately different `.ref` roles in a campaign:

1. The release `.ref` is an immutable, owner-signed reference asset. It is a
   comparator for the production ABI, layout, parity, and timing; it is never
   edited, decompiled, copied into a candidate, or used as evidence that a
   source hypothesis was implemented.
2. hxcc emits a temporary `.ref` when it compiles a reviewed `.rc` source. That
   binary is only the board-load artifact for that source candidate. Generate
   it in a private temporary directory, delete it after the run unless a
   restricted external evidence store explicitly retains its hash-bound copy,
   and never put it in Git. The pure-GEMM harness keeps its historical `--ref`
   spelling as a compatibility alias for “binary to load”; source-derived
   commands should use the unambiguous `--binary` spelling, and every result
   must state which role was used.

Call a kernel *handwritten/source-derived* only when the receipt binds all of
the following to the same clean source snapshot: source SHA-256, exact hxcc
argv and toolchain hashes, generated-assembly SHA-256 plus strict loop/address/
fence counts, a zero-mismatch board parity run, and steady host/native timing.
A run that points `--ref` at the release asset is baseline evidence only, even
if the surrounding host code was changed. A compiled source candidate is still
not a shippable runtime kernel until the release owner admits its symbol,
parameter/grid ABI, manifest entry, asset signature, and held-out profile.
Keep the `.rc` snapshot in an external candidate workspace; this repository may
track the host verifier and pointer-free receipt, but must not become a store
for device-program implementation details or generated operator assets.

The following is an exact column-GEMM example of that source-to-board order;
fusion profiles use their own source entry point and four-pointer verifier.
For every profile, preserve the source basename so hxcc's generated symbol
name remains deterministic:

```bash
src=/path/to/candidates/gemm_ddr_col_m32n256k1024_acc32.rc
build=$(mktemp -d /dev/shm/rhinoforge-rc-XXXXXX)
cp "$src" "$build/$(basename "$src")"
(cd "$build" && hxcc -O2 -save-temps -DRF_N_TILES=4 \
    -DRF_GROUPS=2 -c "$(basename "$src")")
scripts/build_pure_gemm_bench.sh "$build/pure_gemm_bench"
python scripts/inspect_rpu_asm.py \
  "$build"/*-host-rpu-rhino-rpuhsa.s \
  --source "$build/$(basename "$src")" --require-entry --strict \
  --require-vmat --require-lpaddr --require-async-fence
python scripts/board_lease.py -- \
  sudo -n env "$build/pure_gemm_bench" \
  --binary "$build/$(basename "$src" .rc).ref" \
  --kernel gemm_ddr_col_m32n256k1024_acc32 --m 32 --n 2048 --k 1024 \
  --partition 1 --cores 8 --acc32 --tile 128 --typed-weight \
  --warmup 5 --iters 40
rm -rf "$build"
```

For a production `ParallelLinear` packet, verify the source declaration and
host register packing together: the transformed weight pointer is a
`DDR_large` value encoded in 256-byte units, and any `LoopStepInfo` steps from
that base must use the reviewed explicit address granularity (the observed
ABI requires `a_g1B`). A source that compiles with a plain `__DDR` pointer but
has not passed this address-unit/parity test is not a valid replacement.

For an authorized `.rc` campaign, run the board-free compiler identity and
end-to-end smoke before `preflight.py`:

```bash
python scripts/hxcc_preflight.py \
  --compiler "${RHINOFORGE_HXCC:-hxcc}" \
  --expected-host-arch "${RHINOFORGE_HXCC_HOST_ARCH:-aarch64}" \
  --manual-archive "${RHINOFORGE_HXCC_ARCHIVE:-/home/hx/miyaa/work/src/hxcc.zip}" \
  --output <workspace>/runs/hxcc-preflight.json
```

The helper discovers and hashes the wrapper, `clang-17`, `rpuas`, and
`rhino_gen_oplib`, checks an input-bearing `-###` chain for
`rpu-rhino-rpuhsa`, `-O2`, and `-m r1`, then compiles a tiny source in a private
directory. The discovery command above intentionally leaves the expected
hashes unset; before Campaign mode, rerun it with the four frozen values from
the reviewed receipt, for example:

```bash
python scripts/hxcc_preflight.py --compiler /home/hx/.local/bin/hxcc \
  --expected-host-arch aarch64 \
  --expected-wrapper-sha256 <wrapper-sha256> \
  --expected-clang-sha256 <clang-17-sha256> \
  --expected-rpuas-sha256 <rpuas-sha256> \
  --expected-oplib-sha256 <rhino_gen_oplib-sha256> \
  --manual-archive /home/hx/miyaa/work/src/hxcc.zip \
  --output <workspace>/runs/hxcc-preflight.json
```

Attach that pinned receipt (and an external authorization for any device
program) to the contract. A wrapper version, an empty-input target query, or
`rpuas --help` is not sufficient evidence. Keep generated `.o`, `.ref`, and
assembly outside `candidate_root`.

After an isolated compile with `-save-temps`, inspect the assembly before
timing. For a regular GEMM hypothesis use, for example:

```bash
python scripts/inspect_rpu_asm.py <private-build>/kernel-host-rpu-rhino-rpuhsa.s \
  --source <private-build>/kernel.rc --require-entry --strict --require-vmat --require-lpaddr \
  --require-async-fence --output <workspace>/runs/asm-inspection.json
```

`--strict` rejects compiler-lowered software `wjump` loops; the report counts
hardware loops, Repeat forms, `lpaddr`, VLD/VST/VMAT/VALU/VSFU, fences, tail
strobes, and loop depth without publishing source or assembly. Relax a gate
only when the contract explicitly admits a tail/software-loop path and record
the reason. Source pragmas alone never prove a hardware loop. The source
checks are a conservative lexical review, not a complete C++/ABI parser;
successful inspection must still be paired with the hxcc compile and the
release-matched header/asset review.

For a standalone Rhino Launch profile, normalize its native `B`/`E` plus
barrier `s`/`f` trace before the sanitizer:

```bash
python scripts/normalize_hwperf.py <private>/native.json <private>/normalized.json \
  --producer-sha256 <frozen-adapter-sha256> --require-frequency \
  --report <workspace>/runs/hwperf.json
```

The converter fails on unpaired boundaries/flows, strips private event args,
and leaves `device_program_events_exhaustive=false` unless the verifier (or an
external authority) explicitly supplies `--assert-exhaustive`. It is a format
adapter, not a launch-count or manifest attestation; feed its normalized file
to `summarize_hwperf.py` and retain the native hash/frequency/barrier evidence.

The native field mapping is fixed: `cat=Compute` becomes
`rpu_device_program`, `cat=DMA` becomes `rpu_dma` (DMA is not a device-program
launch), and DMA `args.size_bytes` becomes normalized `args.bytes`. The native
`bandwidth_GBps` value is rounded for display; derive the normalized byte rate
from the exact byte count and `duration_us`. In the Launch SDK, the
`Kernel_t` constructor name or `set_name()` is the exact Compute Chrome name;
an omitted name falls back to `kernel_<batch_idx>`. `set_op_type()` only emits a
coarse `args.op_type` label and cannot satisfy an exact manifest-name gate.
Use `--require-frequency` only to require positive frequency metadata. Use
`--assert-exhaustive` only after an independent verifier has checked every
device launch and active-stream barrier; balanced pairs or metadata counters
alone are not exhaustive coverage evidence.

## Establish evidence baselines

Use a clean process and immutable inputs to establish:

- a correctness baseline over public cases plus held-out seeds/shapes;
- a bare-operation latency baseline;
- an equivalent unfused-chain baseline for a fusion campaign;
- Graph BUILD followed by stable REPLAY when the path is repeated;
- output independence under two same-shape calls with different values; and
- empirical compute, memory-bandwidth, and launch/Graph floors when an
  authoritative theoretical specification is unavailable.

For the official Launch path, label the lifecycle explicitly: a
`rhino_launch_batch` (`build_batch` → mutable patch/sync → `enqueu_batch`) is
not the same evidence as a RhinoForge PyTorch Graph BUILD/REPLAY. Record queue
core/warp/broadcast settings and DMA barriers separately. Use the manual's
tuning order—`f16v16`/short address arithmetic, valid hardware loops, eligible
Repeat, async pipeline plus matching fences, then `lpaddr`/tile search—before
attributing a gain to an epilogue.

For GEMM-backed paths, a release-manifest tile is a legal starting point, not a
proof of optimality. Freeze each exact `(M, N_local, K, dtype, accumulation,
core/warp)` shape, enumerate every manifest or compiler-authorized tile that
can implement it, and compare generated loop/Repeat/fence evidence before
choosing a winner. Keep GEMM tile/loop search independent from epilogue or
attention-fusion search so a tile change cannot be misattributed to fusion;
promote only a clean candidate whose same-dtype parity and Graph lifecycle
remain unchanged.

Do not time a candidate that fails a hard semantic or lifecycle gate. Separate
startup/build latency from steady REPLAY latency, and profile in a different
process from the reported latency run.

## Standalone pure-GEMM measurement boundary

When the question is an operator comparison, do not substitute a model or
request wall time.  A Wall Qwen3.5 decode GEMM has two exact contracts:

- q/k/v (and gate/up): global `[M,N,K]=[32,2048,1024]`, column partition over
  eight cores, local `[32,256,1024]`;
- o/down: global `[32,1024,2048]`, row partition over eight cores, local
  `[32,1024,256]`.

The q/k/v dimensions above are the effective post-replication tensor-parallel
width used by the Wall asset (raw logical KV-head widths are smaller).

The current Wall setup explicitly enables FP16-weight ACC32.  The pure-op
protocol is therefore: transform and place the weight in `HostDDR` once;
place each core's input/output windows in its `LocalSPM_t` once; publish the
direct mapped writes with a host barrier; load the exact release kernel; build
one Launch batch; perform one correctness replay, registered warmups, and
synchronous timed replays.  DDR↔SPM staging and host output materialization
must be outside the timed boundary.  Report both `replay_host_us` (API launch
plus wait) and the native Compute critical path (device work); report
`build_us` separately.  The Compute path still includes the operand traffic
required by the kernel's frozen DDR/SPM residency contract.  The standalone
GEMM ABI sets `bias_addr=0`; bias, activation, residual, RoPE, and other
epilogues are intentionally separate operations.

For this Launch SDK release, multi-core staging uses one `LocalSPM_t` per core
and direct mapped `memcpy` plus a barrier.  Repeated `CopyToDevice`/
`CopyFromDevice` calls on a `GlobalSPM_t` do not select independent banks in
this packet ABI and can leave core 0 apparently correct while other cores read
overwritten data.  A parity failure invalidates all timing from that run.

Enumerate every legal release tile (`n_tile` `{128,112,96,80,64,48,32}`),
check same-dtype parity before timing, and keep q/k/v as independent fresh
process measurements even though their shape and kernel are identical.  A
standalone `build_batch`/`enqueue_batch` is Launch-batch evidence, not a
PyTorch Graph BUILD/REPLAY receipt; never print Graph cache or invariant
claims for it.  A tile with the lowest single-run latency is only a measured
candidate.  Without an authoritative board peak plus source/assembly
evidence for all loop, pipeline, and address variants, label the result
“best measured candidate,” never “theoretical optimum.”

Reproduce the board-free build and board run with the checked-in harness:

```bash
scripts/build_pure_gemm_bench.sh /tmp/rhinoforge-pure-gemm-build/pure_gemm_bench
python scripts/board_lease.py -- \
  sudo -n env /tmp/rhinoforge-pure-gemm-build/pure_gemm_bench \
  --binary <release-matched-rhinoOpLib.ref> --m 32 --n 2048 --k 1024 \
  --partition 1 --cores 8 --acc32 --tile 112 --warmup 5 --iters 30 \
  --trace /tmp/pure-gemm-qkv.json --raw /tmp/pure-gemm-qkv.raw.json
```

The `--raw` file contains the unrounded host replay samples and parity/trace
summary, but no addresses or tensor data.  Keep raw files, traces, addresses,
weights, and activation dumps outside Git; commit only the harness, reviewed
summaries, and hashes.

## Delegate safely

The orchestrator freezes the contract, harness, and promotion rule. Give each
optimizer one operation/profile, one Git worktree and branch, and one
measurable hypothesis at a time. Optimizers may change only their candidate
tree. A single board runner owns an exclusive RPU lease and serializes all
hardware runs; parallel agents must not concurrently use the board or share a
live fused model process.

An independent verifier checks a frozen candidate commit in a clean process.
The verifier, not the optimizer, owns final seeds, held-out workloads, timing
order, and promotion. Never let an optimizer edit the reference, evaluator,
timer, result parser, or earlier receipts.

## Iterate transactionally

For every candidate, in this order:

1. State one hypothesis and its expected counter or latency effect.
2. Modify only the candidate implementation.
3. Run the fast correctness/lifecycle signal, then the measurement signal.
4. Commit the candidate code alone, then measure that exact clean Git HEAD and
   its sole parent. Measurement output must be outside the candidate tree or
   in an already ignored run directory so it cannot dirty the measured commit.
5. Immediately append the machine-readable result and human summary to the
   separate campaign evidence store, including failures, raw samples, parent
   commit, environment identity, and artifact hashes. Never amend the measured
   candidate commit to add its own receipt.

Use `scripts/board_lease.py -- <command>` around every standalone RPU command.
Its canonical lock serializes cooperating local processes and detects malformed
or unpaired records; it is not protection from a hostile same-UID process.
`scripts/record_iteration.py` journals signal and failed receipts as local
diagnostics. Neither its `attested` field nor a Linux PID identity authorizes a
release.

For a local full diagnostic, use the launcher below. It invokes the frozen
`commands.profile_one_call` once per workload while holding the lease. The
runner must create the new path named by
`RHINOFORGE_PROFILE_TRACE_OUTPUT`; caller-supplied trace files are forbidden.
It reads the frozen `RHINOFORGE_PROFILE_EVENT_CATEGORY` and
`RHINOFORGE_PROFILE_PRODUCER_SHA256`. The trace object handed to the sanitizer
contains exact `rhinoforgeTrace` metadata whose producer hash equals the
contract-pinned `commands.benchmark_adapter_sha256`; a native SDK B/E trace
must be normalized first (its root carries `otherData`, not this metadata).
See the measurement protocol for the normalized schema. The contract requires trace schema `rhinoforge-rpu-chrome-v1` and
category `rpu_device_program`; the frozen producer must put every and only RPU
device-program launch in that category. Other-category complete events are
host scopes and are ignored, except an allowlisted device-program name outside
the category is rejected. Every trace is re-sanitized against the frozen public
manifest and must contain exactly one event in the device category and one
exact admitted launch; an unknown event in that category fails the gate.
The launcher then measures fresh child processes and writes
`LOCAL-DIAGNOSTIC.json` with `release_eligible=false`. It passes the
verifier-created trace evidence to each child through an identity-bound
`trusted-execution.json`; the adapter cannot self-assert that evidence:

```bash
python scripts/run_full_verdict.py <workspace>/contract.toml \
  --adapter <immutable-rpu-adapter.py> \
  --preflight <workspace>/runs/preflight.json \
  --rhinoforge-root /home/hx/miyaa/work/src/RhinoForge \
  --device-compiler <approved-compiler-if-required> \
  --output-dir <workspace>/runs/full-verdict \
  --candidate-id-prefix <candidate-id>
```

The local launcher re-runs preflight and requires exact equality with the bound
receipt; a hand-written `{"ok": true}` file is not evidence. Omit
`--device-compiler` only when the contract explicitly does not require it.
Both `run_full_verdict.py` and `run_signal.py` accept only `--device rpu`.
Full profiling and benchmark children have finite positive timeouts (defaults
600 and 3600 seconds); signal has a finite non-negative lease timeout and a
finite positive benchmark timeout (defaults 0 and 3600 seconds). A timeout
terminates the child and fails collection.
`collection_ok=true` means the session was recorded; `ok` and
`all_gates_passed` are true only when every full receipt passed, and a collected
failed candidate exits nonzero.

Release promotion is a separate authority-controlled run. The authority must
own the board/lease, immutable snapshot, processes, profiler output, journal,
challenge, sequence, and Ed25519 private key; it must not merely sign hashes
submitted by this launcher. `scripts/select_best.py` accepts only its DSSE
proof, rechecks the exact journal and source tree, recomputes selection, counts
distinct process identities, and refuses unsigned local receipts. If that
authority is unavailable, the correct status is `Blocked`, even when every
local diagnostic gate passes.

After three consecutive candidates fail to improve the current best beyond the
contract's noise threshold, re-profile and change direction. A direction gets
a bounded number of attempts. Stop only at an iteration cap, a measured floor,
or after the contract records that viable directions were exhausted.

## Quantify the claim

For shape `(M,N,K)`, compute achieved operations from the frozen semantic
definition and raw steady-state latency. Define the lower bound as:

```text
max(required_ops / measured_or_authoritative_peak_ops,
    mandatory_bytes / measured_or_authoritative_bandwidth,
    measured_launch_or_replay_floor)
```

Label a result `theoretical` only when the peak specification is authoritative
for the exact device mode. Otherwise label it `empirical roofline` and retain
the calibration results.

`scripts/analyze_efficiency.py` requires a reviewed source ID, its declared
SHA-256, and `--source-file`; the CLI hashes the actual bytes and fails if they
do not match. It also rejects nonphysical efficiency above one instead of
turning an inconsistent floor into a peak claim.

For a fusion, pair it with the profile's bare operation under the same shape,
layout, dtype, output materialization, and REPLAY state. That is one GEMM for
`gemm+add`, the two gate/up GEMMs for `gemm+silu_mul`, the Q/K projection pair
for `gemm+rope`, and RMSNorm for `rmsnorm+quant_mxfp8`:

```text
epilogue_tax_pct = 100 * (latency_fused - latency_bare) / latency_bare
```

The tax is valid only when `arm_graph_state('candidate')` and
`arm_graph_state('bare_operation')` independently show BUILD fixed at one,
REPLAY increasing from an already-warm state, unchanged cache size, and true
invariants before and after paired timing.

Use interleaved trials and a confidence interval. Say "indistinguishable at
the registered measurement resolution" only when the interval's upper bound
is below the contract's pre-registered threshold. Never report absolute zero
cost from a rounded median or one best run.

## Promote and hand off

Promote only a frozen commit that passes the full same-dtype, FP32-anchor,
hard-semantic, held-out, repeatability, Graph lifecycle, maximum-envelope, and
performance/stability gates. Restore candidate code from that exact commit;
do not reconstruct it from memory.

The handoff identifies passed, failed, and unrun gates separately. Keep raw
activation/weight dumps, opaque assets, device addresses, and full hardware
traces out of Git; retain only reviewed summaries, hashes, and approved local
artifact locations.

For a source-level handoff, include the hxcc archive/toolchain hashes, isolated
compile receipt, assembly inspection receipt, and any guide-vs-SDK ABI
discrepancy (for example parameter-register or `LoopOutInfo` types). A clean
compile of a corrected sample is not permission to ship a new `.ref`; the
release owner must still integrate and sign the exact asset.

For design/source audits, see [provenance](references/provenance.md). The
checked-in [local capability receipt](validation/local-capability-2026-08-31.md)
records why offline compilation and host building passed while the initial
device campaign remained blocked; re-run assessment rather than treating that
dated result as current hardware state.
