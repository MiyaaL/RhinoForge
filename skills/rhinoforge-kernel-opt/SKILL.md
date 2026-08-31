---
name: rhinoforge-kernel-opt
description: Run evidence-driven optimization campaigns for RhinoForge RPU operators and fused inference paths. Use for exact-shape RPU kernel benchmarking, profiling, fusion-cost analysis, Graph/SPM/DMA tuning, candidate iteration, or theoretical/empirical roofline work. Do not use for CUDA-only kernels or for claiming a new device kernel when the authorized RPU compiler, source interface, board access, or release-matched operator asset is unavailable.
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
   kernel-manifest identities, compiler identity, environment, and hashes.
5. Correctness, lifecycle, performance, stability, roofline, and fusion-tax
   gates, including the commands that produce each receipt.
6. The candidate-owned paths and evaluator-owned read-only paths.
7. The external release authority's public key, signer identity, policy/build
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

## Establish evidence baselines

Use a clean process and immutable inputs to establish:

- a correctness baseline over public cases plus held-out seeds/shapes;
- a bare-operation latency baseline;
- an equivalent unfused-chain baseline for a fusion campaign;
- Graph BUILD followed by stable REPLAY when the path is repeated;
- output independence under two same-shape calls with different values; and
- empirical compute, memory-bandwidth, and launch/Graph floors when an
  authoritative theoretical specification is unavailable.

Do not time a candidate that fails a hard semantic or lifecycle gate. Separate
startup/build latency from steady REPLAY latency, and profile in a different
process from the reported latency run.

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
`RHINOFORGE_PROFILE_PRODUCER_SHA256`. The raw trace object contains exact
`rhinoforgeTrace` metadata whose producer hash equals the contract-pinned
`commands.benchmark_adapter_sha256`; see the measurement protocol for its
schema. The contract requires trace schema `rhinoforge-rpu-chrome-v1` and
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

For design/source audits, see [provenance](references/provenance.md). The
checked-in [local capability receipt](validation/local-capability-2026-08-31.md)
records why offline compilation and host building passed while the initial
device campaign remained blocked; re-run assessment rather than treating that
dated result as current hardware state.
