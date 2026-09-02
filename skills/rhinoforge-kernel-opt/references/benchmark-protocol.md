# Measurement and promotion protocol

Use this reference in Campaign and Resume modes. Correctness and runtime
lifecycle are prerequisites for performance evidence.

## Evidence identity

Identity is split across three objects; do not attribute all of it to the raw
benchmark result:

- The result directly binds the contract hash, workload IDs, candidate and
  sole-parent commits, preflight hash, Linux process identity, raw latency
  samples, correctness/lifecycle evidence, and sanitized trace-summary hash.
- The frozen contract binds the adapter hash, exact workload semantics and
  gates, symlink-free reference content-tree SHA-256,
  RhinoForge/PyTorch/Launch/operator/manifest identities, and any authorized
  compiler receipt.
- The external authority's signed proof binds the exact contract, candidate
  content tree, result-journal bytes, command/environment hashes, board ID,
  board lease, benchmark/profile processes, raw traces, and sanitized
  summaries.

Schema v1 has no first-class driver-version, board-serial, date, per-input hash,
or full environment-snapshot fields in the result itself. If one is material
to the support profile, freeze it in a reviewed contract/artifact and extend
the authority policy before making the claim. Results from a changed material
identity are a different campaign. Store raw latency samples without rounding.
Keep full traces and tensors outside Git; record only a reviewed summary and
hash.

## Correctness gates

Run each exact public workload, boundary cases, randomized seeds, and verifier-
owned holdouts. Check independently:

- output tree, shape, dtype, exact device residency, exact stride/layout,
  finite values, and semantic ordering for every profile;
- same-dtype absolute/relative error against `max_abs`/`max_rel`, and a
  separately reported FP32 anchor against independent
  `anchor_max_abs`/`anchor_max_rel` thresholds;
- quantized payload, scale tensor, scale layout, block membership, rounding,
  saturation, signed zero, special-value policy, and dequantized error;
- input immutability—including storage identity, device, strides, and exact
  bytes—unless mutation is part of the contract;
- retained output bytes and storage independence after another call;
- same address with changed contents, and different address with equal
  contents;
- fresh-process repeatability and registered warmup variants; and
- Graph BUILD/REPLAY, cache size, semantic-input visibility, and DMA ownership.

Do not flatten a multi-output quantizer into one tensor or let a high cosine
waive a hard mismatch. Thresholds are profile-specific and frozen before the
candidate run.

For `rmsnorm+quant_mxfp8`, `fp32_anchor(case)` returns the unencoded FP32
RMSNorm target, while `dequantize(output, case)` maps the encoded candidate
tree back to that comparison domain. `quantized_output_roles()` maps every
encoded tensor-leaf path exactly once to `payload`, `scale`, or `metadata`, and
must include at least one payload and one scale. The output tree must preserve
the frozen reference structure; every encoded leaf must match bit-for-bit and
stride-for-stride on every anti-cache probe, including signed zero. The
dequantized result must independently pass the FP32-anchor thresholds. This
supports packed-integer or native-FP8 payload representations without inferring
semantics from dtype.

## Timing protocol

Measure startup/BUILD separately from steady REPLAY. The reported steady-state
boundary includes all synchronization and output materialization required by
the public caller.

Every exact `--profile` adapter implements `fp32_anchor(case)` and
`graph_state()` because all five shipped contracts set
`require_fp32_anchor=true` and `require_graph_replay=true`. The only adapter
mode allowed to omit them is the generic CPU protocol harness with no profile.
Missing required hooks become explicit failed gates and no timing samples are
taken; they never silently produce an eligible result. `graph_state()` returns
`build_count`, `replay_count`, `cache_size`, and `invariant_ok` for the overall
profile lifecycle.

1. Run correctness without a profiler.
2. Start a clean process and stabilize the registered warmup/Graph state.
3. Randomize paired order in ABBA or BAAB blocks so temperature and frequency
   drift affect baseline and candidate symmetrically.
4. Synchronize before and after every sample. Generate fresh inputs outside
   the timed region unless input generation is part of the public contract.
5. Repeat in more than one clean process. Record p50, p95, MAD, CV, sample count,
   and all raw values. Reject unstable runs using the pre-registered gate.
6. Collect hardware profiling in a separate diagnostic process. Verify that
   actual named operations, bytes, and launches match the claimed work.

The per-iteration signal may use fewer trials, but it must keep the same input
regime and timing boundary. Schema-v1 signal receipts require at least three
paired positive raw samples per measured arm, and their summaries are
recomputed from those samples. A full verdict still requires at least the
contract's exact `measurement.samples` count per arm, then repeats full
correctness, holdout, anti-gaming, stability, and performance gates. Never
promote a signal receipt.

The contract-driven signal and full launchers accept only `rpu`; direct CPU
adapter runs are protocol tests, never RPU latency evidence. The signal
benchmark timeout and the full profile/benchmark timeouts must be finite and
positive (defaults 3600, 600, and 3600 seconds respectively); the signal lease
timeout is finite and non-negative (default zero). On child timeout the launcher
terminates, then kills if necessary, and fails collection.

Use the contract-driven launcher for an optimization-loop signal:

```bash
python scripts/run_signal.py campaign.toml \
  --candidate-id candidate-0001 \
  --output-dir campaign-results
```

`run_signal.py` has no adapter, profile, workload, warmup, numerical-tolerance,
stability, or fusion-CI override. It derives those values from the exact
approved contract, verifies the frozen adapter and clean isolated candidate
HEAD/sole parent before and after measurement, and observes the `bench.py`
child while holding the canonical board lease. It journals a schema-valid
signal even when a measured gate fails. In that case the JSON report has
`collection_ok=true`, `signal_gates_passed=false`, and the command exits 2.
Collection or identity failures use `collection_ok=false` and are not
journaled. A signal remains non-promotable regardless of either field.

## Adapter and one-call profile runner

`commands.benchmark_adapter` is one frozen Python file with two interfaces:

1. when imported by `bench.py`, it exposes the required
   `make_case(workload, seed, device)`, `reference(case)`, `candidate(case)`,
   and `synchronize()` hooks. Every fused profile also exposes
   `bare_operation(case)`, the independent `bare_reference(case)`, and
   `arm_graph_state(arm)`. The latter accepts only `candidate` or
   `bare_operation` and returns `build_count`, `replay_count`, `cache_size`, and
   `invariant_ok` for that timing arm. The two bare hooks return the same
   registered output tree with matching dtype/device residency, strides, and
   layout; the bare identity is profile-specific (one GEMM, two gate/up GEMMs,
   a Q/K projection pair, or RMSNorm), not inferred from the fusion name. Every
   exact profile also exposes `fp32_anchor(case)` and `graph_state()`;
   `rmsnorm+quant_mxfp8` exposes the remaining quantization hooks described
   above;
2. when executed by `commands.profile_one_call`, it performs exactly one public
   candidate call and writes a new raw trace to the requested path.

The executable mode reads these verifier-owned variables and must fail if a
required one is absent or inconsistent:

```text
RHINOFORGE_PROFILE_WORKLOAD
RHINOFORGE_PROFILE_TRACE_OUTPUT
RHINOFORGE_PROFILE_CONTRACT_SHA256
RHINOFORGE_PROFILE_PREFLIGHT_SHA256
RHINOFORGE_PROFILE_CANDIDATE_COMMIT
RHINOFORGE_PROFILE_PARENT_COMMIT
RHINOFORGE_PROFILE_DEVICE
RHINOFORGE_PROFILE_EVENT_CATEGORY
RHINOFORGE_PROFILE_PRODUCER_SHA256
```

The local full launcher creates the output path, re-sanitizes the trace, writes
an identity-bound `trusted-execution.json`, and passes that file to the fresh
benchmark child with `--trusted-execution`. The adapter does not self-assert a
trace hash. This handoff is local diagnostic evidence only; release still
requires the external authority's signed process/trace statement.

## Anti-gaming checks

The final evaluator is immutable and outside the optimizer's writable tree.
It chooses final seeds, call counts, order, and holdouts only after the
candidate commit is frozen. It checks:

- deferred/asynchronous work by synchronizing and fingerprinting after return;
- cached-output tricks with pointer/content mutations and retained outputs;
- harness-shape or call-index branching with hidden workload/call schedules;
- cross-call batching or global result caches with clean subprocesses;
- canaries around outputs and unchanged input storage; and
- implausible speedups against the hardware trace and mandatory work count.

Python process separation is not a security sandbox for malicious native code.
Run untrusted candidates only inside an administrator-approved isolation
boundary. This protocol is aimed at trustworthy engineering and accidental or
reward-driven benchmark errors.

## Roofline and fusion evidence

For each workload, state required operations and mandatory bytes from the
mathematical contract. Calibrate peak compute, sustained bandwidth, and
launch/REPLAY floor on the same board mode, or cite an authoritative exact-mode
specification. Report:

```text
compute_floor = required_ops / peak_ops_per_s
memory_floor  = mandatory_bytes / bandwidth_bytes_per_s
lower_bound   = max(compute_floor, memory_floor, launch_or_replay_floor)
efficiency    = lower_bound / measured_latency
```

If the inputs came from microbenchmarks, call this an empirical roofline. Do
not silently substitute a vendor headline number.

For fusion, retain both the profile's bare operation and equivalent unfused
chain. The bare operation is GEMM (or the Q/K projection pair) for GEMM
epilogues, two gate/up GEMMs for `gemm+silu_mul`, and RMSNorm for
`rmsnorm+quant_mxfp8`. Bare and fused measurements use the same input and
residency regime, registered output materialization, Graph state, and
interleaved order. Report absolute delta, epilogue tax in percent, and a paired
confidence interval. The claim "indistinguishable at measurement resolution"
requires the deterministic median-of-paired-percentages bootstrap used by both
`bench.py` and `analyze_efficiency.py`; confidence, trial count, seed, and
maximum tax are frozen in the contract. The interval's upper bound, not a
rounded median, must pass that threshold.

An epilogue-tax result is valid only when both arm states independently pass
the steady transition: the stable and final `build_count` are one, the stable
`replay_count` is already at least one and the final count grows, cache size is
unchanged, and `invariant_ok` is true at both observations. A shared aggregate
Graph counter cannot substitute for `arm_graph_state`.

A local full diagnostic invokes the frozen `commands.profile_one_call` itself
under the board lease; it never accepts a caller-provided trace path. The
runner writes a fresh output named by `RHINOFORGE_PROFILE_TRACE_OUTPUT` and
reads `RHINOFORGE_PROFILE_EVENT_CATEGORY` and
`RHINOFORGE_PROFILE_PRODUCER_SHA256`. The latter equals the contract-pinned
`commands.benchmark_adapter_sha256`. Both frozen runtime fields are exact:
`runtime.trace_schema = "rhinoforge-rpu-chrome-v1"` and
`runtime.trace_event_category = "rpu_device_program"`. The normalized trace
handed to the sanitizer (not the native SDK input) must have an object root
with `traceEvents` and this exact four-field metadata object:

The standalone Rhino Launch SDK is a native producer of `B`/`E` duration
events and `s`/`f` barrier-flow events (and its raw buffer is paired 8-byte
START/END records), whereas the schema below is the skill's normalized
`ph=X` representation. The normalized file is still private diagnostic input
(names and timestamps are retained until sanitization); only
`summarize_hwperf.py` produces the pointer-free public summary. A trusted
profiler adapter may perform that normalization in a private directory, but it
must verify balanced pairs, stream barriers, and `LKN_RPU_FREQ_MHZ` (default
800 MHz), bind the raw hash, and never fabricate a launch or silently drop a
trailing stream. The native format is therefore an input format, not a reason
to weaken the one-launch manifest gate.

The supplied converter is intentionally explicit about the trust boundary:

```bash
python scripts/normalize_hwperf.py native.json normalized.json \
  --producer-sha256 <frozen-adapter-sha256> --report hwperf-normalization.json
```

Without `--assert-exhaustive`, its metadata keeps
`device_program_events_exhaustive=false` and the release sanitizer will reject
it. Only a verifier/authority that has checked coverage may set that flag.

For the native Launch input, normalize `Compute` to
`rpu_device_program` and `DMA` to `rpu_dma`; map DMA `args.size_bytes` to
`args.bytes`, and derive an exact byte-rate field from `bytes / duration_us`
because native `bandwidth_GBps` is rounded. The SDK's `Kernel_t` constructor
name and `set_name()` value are the exact Compute event name (otherwise the
default is `kernel_<batch_idx>`); `set_op_type()` is only a coarse argument
label, never a manifest identity. `--require-frequency` checks positive
frequency metadata only. `--assert-exhaustive` is permitted only with an
independent launch/barrier coverage audit, not merely balanced B/E or s/f pairs
or `batch_events`/`records_captured` counters.

```json
{
  "rhinoforgeTrace": {
    "schema": "rhinoforge-rpu-chrome-v1",
    "device_event_category": "rpu_device_program",
    "device_program_events_exhaustive": true,
    "producer_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
  },
  "traceEvents": [
    {
      "name": "release_approved_fused_kernel",
      "cat": "rpu_device_program",
      "ph": "X",
      "ts": 0,
      "dur": 1
    }
  ]
}
```

Replace the illustrative producer digest and kernel name with their exact
contract values. The frozen trace producer must categorize all and only
device-program launches that way. Complete `X` events in other categories are
host scopes and are ignored, except that an allowlisted device-program name in
another category is a hard failure. The launcher allowlists every
release-manifest name, then requires exactly one complete event in the device
category and one manifest launch. An unknown event in that category, a known
second kernel, or a repeated fused event therefore fails. For fusion, the
selected name must equal the explicit
`runtime.fused_kernel`, which must itself be an exact member of
`runtime.required_kernels`; substring/token matching is never evidence. An
external release authority must audit that this exact producer really marks
every device launch exhaustively, repeat collection from an authority-owned
output, and bind it in the signed proof. The root metadata is a hash-bound
producer declaration, not independent hardware proof of exhaustiveness.

## Machine-readable result

`scripts/bench.py` and `scripts/record_iteration.py` exchange schema version 1:

```text
schema_version
candidate_id, candidate_commit, parent_commit, contract_sha256, verdict
preflight_sha256
process {boot_id, pid, start_ticks, run_uuid, attested}
metric {name, direction, value, aggregation}
workloads [
  id
  raw_samples {candidate_ns, reference_ns, mandatory-fusion bare_operation_ns}
  summary
  correctness
  lifecycle
  execution {kernel_names, launch_count, single_device_program,
             trace_summary_sha256, preflight_sha256}
  hard_gates
  eligible
]
hard_gates
artifacts (optional reviewed hashes/paths only)
```

Preflight verifies `campaign.reference_tree_sha256`, a bounded content hash of
the symlink-free reference tree excluding `.git`. Signal and full launchers
verify the same tree before and after the child session; a change fails closed.
The supported way to obtain the contract value is:

```bash
python scripts/hash_reference_tree.py <reference_root>
```

Copy the emitted digest into `campaign.reference_tree_sha256`; do not import a
private helper from the scripts package.

The local journal uses cooperative locking, safe file opens, and unique
candidate IDs, but it is mutable by its owner and is not release-grade
append-only storage. Non-finite samples, missing workloads, too few samples,
inconsistent medians, failed required gates, duplicate IDs, malformed commits,
or changed contract hashes fail closed. Release selection additionally requires
an external signature over the ordered record hashes and exact journal bytes.

A failed execution is still journalable: it has paired empty raw samples,
null latency summaries and metric, explicit failed gates, and can never be
promoted. Do not invent a large sentinel latency for a failure.

## Promotion

Local `process.attested=true` means only that the cooperative wrapper observed
the child; it is insufficient for promotion. Selection requires a fresh
challenge and DSSE/Ed25519 proof from the contract-pinned external authority,
as specified in [release attestation](release-attestation.md). The signed
statement must cover the exact receipt snapshot, source tree, lease, profiler,
and benchmark processes. Within that signed set, only `verdict = "full"`, a
nonzero successful-preflight hash, all required hard gates true, and every
workload eligible may enter selection. A commit also needs at least
`measurement.verdict_processes` distinct authority-bound process triples.
Selection groups by exact candidate/sole-parent/contract/preflight identity,
rechecks the clean Git checkout and independent source-tree hash, uses the
median process metric, and records the release-proof hash in `BEST.json`.

Checkout the selected commit into a clean tree and rerun the full verifier.
The final handoff reports failures and unrun gates as prominently as passes.
