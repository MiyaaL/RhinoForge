# Local RPU kernel capability assessment — revalidated 2026-09-01

This is a local execution receipt, not a reusable support claim.

Device admission was rechecked on 2026-09-01: `/dev/rpu*` and `/dev/mem`
remained absent, and `examples/verify_install.py` again exited 1 with
`RPU is unavailable`.

## Candidate identity

| Field | Observed value |
|---|---|
| Requested suite | GEMM; GEMM+SiLU-mul; GEMM+add; GEMM+RoPE; RMSNorm+MXFP8 |
| RhinoForge source | `3f268894e6bad843beceda8c97aee5cbded0d805` with pre-existing unrelated worktree changes |
| Distribution / API | `rhinoforge 1.0.0`; public API `5.0.0` |
| Python / PyTorch | Python 3.12.13; PyTorch `2.10.0+cpu` |
| Runtime set | active `runtime-v1.0.0-r4` |
| Launch / operator | Rhino Launch `1.0.0`; operator asset `1.0.0` |
| Driver | loaded `rpu_drv` version `1.0.1`; kernel 5.10.220-rt112, aarch64 |
| Offline compiler | `hxcc version 1.0+0a6ba7e4` at `/home/hx/.local/bin/hxcc`; binary SHA-256 `b887d6686bf588ca173e965ff8ff97c6dda4e12b89717aca11eeab7eda5096cc` |
| SPM/public execution | eight cores; 8 MiB SPM/core; FP16 primary; common 16-element alignment |

The runtime bundle checksum contract and all three named payloads were present.
The opaque asset was SHA-256 verified but never parsed, decoded, or modified.

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
- The exact Rhino Launch bundle configured and built the full `rpu_backend`
  host extension successfully in an isolated temporary tree.
- The final skill harness suite ran 106 tests successfully under the configured
  Python 3.12/PyTorch environment. The same 106 tests completed under system
  Python 3.10 with 17 PyTorch-dependent cases explicitly skipped.
- RhinoForge's existing board-free kernel-contract and quantization subset
  reported `19 passed in 9.19s` after the skill hardening.
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
`process.attested` remained false. Their refreshed strict JSON receipt hashes
were:

- GEMM: `9e6bec6569d3e8e4816cfe418d5316fa8a3bd77917351eed20fa53903e1f88ae`;
- GEMM+add: `01343901ffa688fd9a8e015c824dfbd3bceb4750cc85364c977dfe819c649ac6`;
- GEMM+RoPE: `2136197fafae43598f502db422bd5d56071454cb50db51a1d6720e6bf327aa6d`;
- GEMM+SiLU-mul: `80bef804f1d90d400f402a3fabc81186a298d9af51403b09b2ede63660b2ecb4`;
- RMSNorm+quant-MXFP8 protocol fixture:
  `fcc46ef77b1bf4312193b9ab6676242b2a000f3eebcf30ff55ed340955eeea70`.

Those smokes are harness evidence, not RPU performance evidence. In
particular, the portable int8 fixture is not an MXFP8 implementation.

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
The compiler's use/redistribution authority remains unknown, the Launch public
headers expose no device-program API, and the production combined asset still
requires its external release owner.

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
