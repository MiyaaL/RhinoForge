# Local RPU kernel capability assessment — revalidated 2026-09-02

This is a local execution receipt, not a reusable support claim.

Device admission was rechecked on 2026-09-02: `/dev/rpu*` and `/dev/mem`
remained absent, and `examples/verify_install.py` again exited 1 with
`RPU is unavailable`.

## Candidate identity

| Field | Observed value |
|---|---|
| Requested suite | GEMM; GEMM+SiLU-mul; GEMM+add; GEMM+RoPE; RMSNorm+MXFP8 |
| RhinoForge source snapshot | `3f268894e6bad843beceda8c97aee5cbded0d805` (inspection baseline; the checkout was `82f944e60243672d5d8a97d41d5ea4a14f91ed29` with this skill plus unrelated worktree changes) |
| Distribution / API | `rhinoforge 1.0.0`; public API `5.0.0` |
| Python / PyTorch | Python 3.12.13; PyTorch `2.10.0+cpu` |
| Runtime set | active `runtime-v1.0.0-r4` |
| Launch / operator | Rhino Launch `1.0.0`; operator asset `1.0.0` |
| Driver | loaded `rpu_drv` version `1.0.1`; kernel 5.10.220-rt112, aarch64 |
| Offline compiler | `hxcc version 1.0+0a6ba7e4` at `/home/hx/.local/bin/hxcc`; wrapper SHA-256 `b887d6686bf588ca173e965ff8ff97c6dda4e12b89717aca11eeab7eda5096cc` |
| SPM/public execution | eight cores; 8 MiB SPM/core; FP16 primary; common 16-element alignment |

The runtime bundle checksum contract and all three named payloads were present.
The opaque asset was SHA-256 verified but never parsed, decoded, or modified.

## Official hxcc manual overlay and board-free compiler evidence

The supplied `/home/hx/miyaa/work/src/hxcc.zip` was read without copying its
contents into the repository. Its archive SHA-256 is
`21e8ed1462620b84e4d818eae24b8507817cbeca1a39105cf3863d8c8287eeba`; the HTML
developer guide is `ff0ad7bee5f2f1ebda9cf0cabb0d3f9dd0872534a33fd907185028fdb02e8594`
and the compiler guide PDF is
`0a34979ed65099ea73f6c2bae0f8272568d7d695ee55b10a1918c1349f8a6114`.
The skill's clean-room digest is
[`references/hxcc-manual.md`](../references/hxcc-manual.md).

`scripts/hxcc_preflight.py` was run with the installed ARM wrapper and all
four tool identities pinned. The strict run returned `ok=true` with 18/18
checks and receipt SHA-256
`7c7e6dfb43f6b978cb37b1560c6aedec9ae45df59de85d1e9a3f489d55b809ce`. It
verified host `aarch64`, wheel tag `manylinux2014_aarch64`, wrapper hash
`b887d6686bf588ca173e965ff8ff97c6dda4e12b89717aca11eeab7eda5096cc`, underlying
`clang-17` hash
`999f36abd6b7920e9705c827ad12aa41dca9af5735df033e8af773a4d28ad72d`, `rpuas`
hash `b43aeb9cedc19cdd80210ff59d5805aa3f1eec99f8e597a14c564abc94703e02`, and
`rhino_gen_oplib` hash
`e502faf9958561a9e8240244ca3079d9d87db04f8a10a37d2345d0c891d70d45`. The
input-bearing dry-run resolved target `rpu-rhino-rpuhsa`, `-O2`, and
`rhino_gen_oplib -m r1`; an isolated clean-room `.rc` produced deterministic
`.o` (200 bytes, SHA-256
`ad5780c484c345072ed074dce4aa6626dfaf659b532978ad776094c16bdd4643`) and
opaque `.ref` (1187 bytes, SHA-256
`ec72625beac205d047d3a4280bede70e022f20e8d4a26a1bdbd0012a5f43c0e9`). The
`.ref` bytes were not retained or inspected.

The guide's 32x32x64 GEMM sample was also compiled in a private directory with
the shipped headers. The unmodified sample failed because its `unsigned short`
loop metadata did not match the SDK's `LoopOutInfo(short*)` constructor. A
clean-room `short` correction then compiled successfully with `-O2 -save-temps`.
`inspect_rpu_asm.py --strict --require-vmat --require-lpaddr
--require-async-fence` passed: six hardware-loop records, five Repeat forms, six
`lpaddr` uses, one VMAT execute, five fences, zero `wjump`, and maximum loop
depth two. The private assembly SHA-256 was
`d99c54229874d3be5e920c67d4b5606f9f59963a227b28be0e690a730bf9f83e`; the
inspection receipt SHA-256 is
`5204a80917914017b4901569aa8697552232c796125c619ea7ee9a8c07610bbe`. No source,
assembly, object, or `.ref` was added to Git. This mismatch is why the skill
requires resolving compiler and SDK header versions together rather than
hard-coding the guide's parameter-register count.

## Source and capability closure

Inspected public RhinoForge architecture, runtime-assets, validation,
performance, model-porting, kernel declarations/registrations, public operator
wrappers, kernel-name manifest, and installed runtime metadata.

Observed public asset capabilities include FP16 SPM GEMM tiles, shared parallel
Linear variants, binary elementwise kernels, RMSNorm variants, RoPE variants,
and standalone `llama_silu_mul`.

Missing or unavailable evidence:

- the current container has neither `/dev/rpu*` nor `/dev/mem`;
- `/sys/devices/virtual/misc/rpu/dev` reports the host driver device as `10:61`,
  but device nodes are not mapped into the container;
- the approved minimal `examples/verify_install.py` probe reports
  `RPU is unavailable` both normally and through the permitted sudo probe;
- the delivered runtime bundle explicitly excludes the board SDK;
- the locally installed `hxcc` has no established license or documented public
  production-asset integration authority in this repository; and
- the release manifest contains no kernel name containing `mxfp8` and no named
  GEMM+SiLU-mul, GEMM+residual-add, or GEMM+RoPE device fusion.

No privileged device-node creation was performed. The execution environment
must be configured by its administrator.

## Board-free execution evidence

- `hxcc` compiled an existing local `.rc` sample outside this repository to
  `.o` and `.ref`; repeat outputs matched the hashes of the corresponding
  existing local artifacts. No source or binary was copied into this skill.
- The strict hxcc preflight checked all four frozen tool identities, the
  input-bearing target/`-O2`/`-m r1` chain, and generated deterministic
  isolated outputs. A `-save-temps` compile of the guide's GEMM structure was
  inspected with the new assembly gate; the report passed with zero `wjump` and
  finite hardware-loop depth after the documented header-compatible `short`
  correction.
- The exact Rhino Launch bundle configured and built the full `rpu_backend`
  host extension successfully in an isolated temporary tree.
- The final skill harness suite ran 138 tests successfully under the configured
  Python 3.12/PyTorch environment (46 subtests). The same 138-test suite
  completed under system Python 3.10 with 17 PyTorch-dependent cases
  explicitly skipped.
- RhinoForge's existing board-free kernel-contract and quantization subset
  (`python -m pytest -q tests/test_kernel_contracts.py tests/test_quantization.py`)
  reported `19 passed in 9.21s` after the skill hardening.
- The supported reference-tree hash command produced the same
  `aecb0f7cae8ac6aafd613e30e62dccdcb3b5017d878a6db2691f51b8e66f4ad2`
  digest under both Python environments.
- Runtime bundle checksums and the three named payload identities passed.

An approved temporary GEMM preflight then passed the contract, command/root,
Git commit, Launch, asset SHA-256, manifest SHA-256/schema, and required-kernel
checks. It failed the dirty source tree plus unavailable `/dev/rpu*`,
`/dev/mem`, and FP16 `torch.rpu` roundtrip gates; no benchmark was launched. A
separate CPU-only exact-profile protocol smoke exercised all five adapter
profiles, nested quantized payload/scale output, input/output lifetime probes,
paired timing, dual-output Q/K RoPE, profile-specific bare references, and
independently warm candidate/bare Graph REPLAY transitions. All five signal
workloads were locally eligible, `execution` remained null, and
`process.attested` remained false. The earlier per-profile strict JSON receipt
hashes (retained as input/protocol history) were:

- GEMM: `9e6bec6569d3e8e4816cfe418d5316fa8a3bd77917351eed20fa53903e1f88ae`;
- GEMM+add: `01343901ffa688fd9a8e015c824dfbd3bceb4750cc85364c977dfe819c649ac6`;
- GEMM+RoPE: `2136197fafae43598f502db422bd5d56071454cb50db51a1d6720e6bf327aa6d`;
- GEMM+SiLU-mul: `80bef804f1d90d400f402a3fabc81186a298d9af51403b09b2ede63660b2ecb4`;
- RMSNorm+quant-MXFP8 protocol fixture:
  `fcc46ef77b1bf4312193b9ab6676242b2a000f3eebcf30ff55ed340955eeea70`.

Those smokes are harness evidence, not RPU performance evidence. In
particular, the portable int8 fixture is not an MXFP8 implementation.

An independent delegated CPU-only signal pass (`cpu-signal-v4`, receipt
`/tmp/rhinoforge-kernel-opt-cpu-signal-v4.json`, SHA-256
`a4ed8a5c88df06d6ff8cc9cb31331d3f2d7a299874c478c7ea5824c88b39cf7a`) then
ran four interleaved candidate/bare timing samples for each requested profile.
The preceding contract-gated iteration summary
(`/tmp/rhinoforge-forward-iteration-summary-v2.json`, SHA-256
`b3138c7b75df31d847a925867d41915be76746135c9d830ef37e8eb9b0398329`) was
kept as a failed-result record: GEMM exceeded the registered stability CV
bound, the three fusion profiles exceeded the epilogue-tax CI bound (and also
stability), and the portable quantized fixture failed its anchor/lifetime
gates before timing. The v4 run was deliberately a cheaper signal protocol
with those production stability/tax gates disabled; therefore “eligible” in
the table below means eligible for this CPU signal only, never release or
hardware eligibility.
Every profile passed the correctness, same-dtype, FP32-anchor, input-lifetime,
output-lifetime, and one-BUILD/stable-REPLAY gates (`BUILD=1`, `REPLAY=9`,
fixed cache size, invariant true). The measured values below are CPU protocol
signals in nanoseconds; `tax upper` is the paired 95% bootstrap upper bound.

| Profile | candidate p50 | bare p50 | tax p50 | tax upper | protocol result |
|---|---:|---:|---:|---:|---|
| GEMM | 13,019.5 | 12,211.5 | 2.78% | 11.02% | eligible; CPU only |
| GEMM+SiLU-mul | 27,116.5 | 10,904.0 | 146.11% | 152.74% | eligible; CPU only |
| GEMM+add | 20,461.5 | 10,576.5 | 92.06% | 96.70% | eligible; CPU only |
| GEMM+RoPE | 217,138.0 | 18,211.5 | 1,087.79% | 1,140.83% | eligible; CPU only |
| RMSNorm+quant-MXFP8 | 153,041.0 | 68,078.0 | 119.43% | 130.06% | eligible; int8 protocol fixture only |

These numbers validate the iteration and fusion-tax machinery, not RPU speed or
theoretical peak. The candidate and bare arms reported no device execution and
were not used for release promotion. The large positive taxes are an explicit
signal to re-profile the corresponding fusion rung once board access and an
authorized device-program path are supplied; they are not hidden as “zero-cost”
epilogues.

The native Launch-format adapter was exercised on the supplied all-reduce
sample (`/tmp/allreduce_baseline_32_trace.json`, input SHA-256
`f330e9359f01d0199af1b5e0f9fb20eddebcd0a8bbcbbae378f39ea78d84ba7f`). With the
fixture's known-complete one Compute pair, `normalize_hwperf.py` produced one
`rpu_device_program` event at 800 MHz and preserved the exact byte/timing
derivation. The normalized trace SHA-256 is
`b03ea074c259a4f9ee85f92aec901b880a80ac27c73f18bf8a5f60a2dbbc974f`, the
normalization report SHA-256 is
`c59e8085eebf27b57e87c4ed373ba920ba62e9b56e968cfcfa25e05bdda8ed29`, and the
downstream pointer-free summary SHA-256 is
`b41c0e6965ad2b8db0f652ca35dbcf32275b6646e324b65f51237fd51b3d300f`.
This is a format/converter regression fixture, not a board launch or peak
throughput claim.

The hardened profiler sanitizer consumed a previously collected local Chrome
trace with 34,401 complete events. With no explicit kernel allowlist, every
name and category was emitted only as `<redacted>` while the summary retained
the exact raw-trace SHA-256. All non-allowlisted names/categories now collapse
into one constant bucket, so their equality and group cardinality are not
published. Unit tests cover arbitrary secrets, embedded addresses,
whitespace/UNC paths, equality-leak attempts, and exact allowlist behavior;
raw traces and generated summaries remain outside Git.
For the 34,401-event trace, the input SHA-256 was
`e49bd20916fd94856502ba213a115cf1c9d192cd6549beed58585e8a6fc9898d`
and the new one-bucket summary SHA-256 was
`14d4241ff91d9ce39d7326ec82c6ca382ed8470afe3f9cfa4042c3ae82a9fbe4`.

An independent adversarial pass also exercised self-attested receipts,
duplicate candidate IDs from one process, mismatched preflight evidence,
known or unknown multi-node Graph traces pretending to be fusion, alternate
board-lock paths, lease pathname/inode replacement, tampered journal hash
chains, forged roofline source hashes, missing bare-operation samples, and
fusion-tax CI mismatch. The final pass additionally rejects forged global
Graph aggregates, non-warm stable states, inconsistent stable/final evidence,
and fusion arms that do not independently remain in steady REPLAY. These local
cases fail closed.

A second audit showed that same-UID trace and journal files are forgeable, so
local process attestation was deliberately demoted to diagnostic evidence.
`BEST.json` additionally requires a fresh challenge plus a DSSE/Ed25519 proof
from the contract-pinned external release authority. No such authority or proof
is available on this machine, so release promotion remains blocked. This is
harness evidence, not RPU performance evidence.

The proof verifier uses `/usr/bin/openssl` through a verified open file
descriptor; its frozen local binary SHA-256 is
`56fd8066c150a7e46521c05f12f7c5a9eeaf1ded600446cac92ee0d5dfe0b813`.
Tests cover valid DSSE/Ed25519 selection, payload tampering, stale challenges,
journal rewrites, and unsigned hand-written receipts.

These prove offline compilation, host integration, and contract checks only.
The compiler's use/redistribution authority remains unknown; the Launch headers
and libraries are available only in the externally supplied archive and are not
integrated into the RhinoForge runtime set; and the production combined asset
still requires its external release owner.

## Operation gap table

| Operation | Existing public pieces | Missing campaign capability | Status |
|---|---|---|---|
| GEMM | FP16 SPM GEMM and parallel Linear names/wrappers; offline compiler works | mapped RPU device and exact workload/peak calibration | hardware-blocked |
| GEMM+SiLU-mul | GEMM plus standalone `llama_silu_mul`; offline compiler works | authorized source/ABI and one-program production asset path | authorization/asset-blocked |
| GEMM+add | Linear bias and standalone binary paths; offline compiler works | exact residual-vs-bias contract and general fused epilogue asset entry | contract/asset-blocked |
| GEMM+RoPE | GEMM plus RoPE/partial-RoPE/M-RoPE paths; offline compiler works | exact position profile and named fused asset entry | contract/asset-blocked |
| RMSNorm+MXFP8 | RMSNorm variants; offline compiler works | exact MXFP8 format, quantizer ABI, and production asset entry | contract/asset-blocked |

Keeping intermediates in SPM and Graph can reduce transfers and host overhead,
but it is not evidence of a single fused device program.

## Verification contract for resumption

Before a candidate iteration, require:

1. administrator-provided `/dev/rpu*` and `/dev/mem` mapping, followed by the
   standard host/device copy check;
2. exact workload shapes/layouts/dtypes/semantics and same-dtype plus FP32
   references;
3. confirmation that the observed `hxcc` toolchain and the exact kernel-source
   interface are authorized, plus an owned asset release path for every new
   device program;
4. an updated release-matched manifest for MXFP8 and any fused kernel name;
5. empirical compute, bandwidth, and launch/REPLAY calibration on the same
   board mode;
6. correctness, output lifetime, warmup, Graph BUILD/REPLAY, multi-input,
   maximum-envelope, interleaved latency, and profiler gates; and
7. independent full-verdict reruns from frozen candidate commits.

## Decision

```text
outcome:       Blocked
certification: uncertified
```

Offline compilation and host-wrapper construction passed. Device-kernel
Campaign mode and performance claims may not proceed from this container. The
smallest next step is administrator device mapping, explicit `hxcc`/source
authorization, a production asset release owner, and exact per-operation
contracts.
