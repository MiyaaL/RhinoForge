# RhinoForge RPU runtime boundary

Use this reference before assessing or running any campaign. The canonical
contracts remain RhinoForge's `docs/architecture.md`, `docs/runtime_assets.md`,
`docs/validation_policy.md`, `docs/performance.md`, and
`knowledge/concepts/launch-runtime.md`.

## What public source can change

RhinoForge exposes host-side tensor validation, layouts, addresses, SPM
planning, DMA ownership, core selection, synchronization, parameter packing,
Graph capture/replay, and named-kernel lookup. Those are valid campaign targets
when the exact release asset already contains the required device operation.

Device-program implementation and the combined operator asset are outside the
public repository. The `.ref` file is opaque. Its adjacent `.kernels` file is a
release-generated list of size and public kernel names, not source and not a
compiler input. Never inspect the `.ref` payload, generate a replacement
manifest, rename or split the pair, or commit either file.

A genuinely new device kernel requires all of these before Campaign mode:

- an authorized device SDK/compiler and documented source interface;
- an exact mathematical/tensor ABI contract;
- an asset build owner and versioned delivery path;
- a release-matched updated asset and manifest; and
- board validation against the new immutable identities.

Without those capabilities, host scheduling experiments may continue, but the
new device kernel is `Blocked` rather than approximated with a nearby name.

Compiler discovery has separate capability, identity, and authorization gates.
A successful offline compile proves
that a toolchain can translate a source file; it does not prove permission to
use or redistribute that compiler, source, headers, object, or `.ref`, and it
does not install the result into RhinoForge's combined production asset. Record
the exact compiler binary SHA-256, a sanitized compile receipt, and a hashed
authorization receipt naming the release owner; then obtain an approved asset
integration path before a device-kernel campaign.

For the local hxcc release, read
[the hxcc manual overlay](hxcc-manual.md) and run
`scripts/hxcc_preflight.py`. It checks the wrapper plus the underlying
`clang-17`/`rpuas`/`rhino_gen_oplib` chain, an input-bearing target triple, and
an isolated `-O2` compile. The compiler and Launch header versions are part of
the ABI: the guide's 0..61 compiler parameter range and the shipped SDK's
0..67 comment must be resolved together, never guessed.

## Hardware and memory contract

The current public runtime targets eight cores. Each core has 8 MiB SPM;
planning uses a conservative usable budget and treats capacity as a hard gate.
FP16 is the primary execution dtype and many dimensions require 16-element
alignment.

SPM manifests must be deterministic and side-effect free. Declare values that
survive replay as `Persistent` or `PersistentPerLayer`; temporary storage may
be aliased only after proving non-overlapping lifetimes. Ordinary wrappers use
the exact absolute-address or offset form in their declaration. Never cast one
form into the other to satisfy a call.

Immediate multi-core launches require host-parameter broadcast. Prefer the
framework batch/Graph path, which owns broadcast and per-node core selection.
When using the standalone Launch SDK, distinguish its
`build_batch`/`sync_mutable_params`/`enqueu_batch` replay from a RhinoForge
PyTorch Graph replay. Record SPM stride/alignment, DMA size/channel limits,
stream barriers, core IDs, and queue warp/broadcast settings in the evidence.

## Graph, DMA, and output lifetime

For a repeated signature, prove one BUILD followed by stable REPLAY:

- BUILD count remains one;
- replay count grows;
- cache size stays fixed; and
- `cache_invariant_ok()` remains true.

Every result-changing semantic input belongs in the signature or is refreshed
through a documented mutable DMA path. Fixed DMA is only for model-owned
storage whose address remains stable. Caller inputs and fresh outputs need
mutable DMA. Immediate DMA is a one-shot path outside retained capture.

Return independent Python-visible storage whenever a caller can retain an
output. A successful one-call comparison does not cover output aliasing or
replay address ownership.

## Required environment preflight

Check, without modifying system state:

1. the exact RhinoForge, PyTorch, Launch, asset, manifest, driver, and board
   identities;
2. readable release metadata and checksums for the approved runtime set;
3. the required kernel names in the adjacent manifest;
4. accessible `/dev/rpu*` and `/dev/mem` nodes;
5. `torch.rpu.is_available()` and a mandatory FP16 host/device/host roundtrip;
6. an authorized compiler/build path, exact compiler binary hash, and hashed
   authorization receipt when candidate code changes a device
   program; and
7. the profiler and counter schema needed by the registered metric; and
8. for release promotion, the external authority public key, policy hash, and
   signer-build hash. The private key and board-control service must be outside
   the candidate/campaign UID boundary.

The approved campaign also freezes a nonzero
`campaign.reference_tree_sha256`. It hashes the bounded, symlink-free reference
content tree while excluding `.git`; preflight verifies it, and signal/full
launchers verify it both before and after child execution. The full trace
contract is likewise exact: schema `rhinoforge-rpu-chrome-v1`, event category
`rpu_device_program`, and—when fused—one explicit `runtime.fused_kernel` that
is an exact `runtime.required_kernels` member. The frozen producer must reserve
that category for all and only device-program launches; host scopes use other
categories.

Populate the reference field only with the supported public command:

```bash
python scripts/hash_reference_tree.py <reference_root>
```

The one-call trace handed to the sanitizer carries exact `rhinoforgeTrace`
metadata: schema, device category, `device_program_events_exhaustive=true`, and
a producer hash equal to `commands.benchmark_adapter_sha256`. A native SDK
B/E input instead carries `otherData` and must first pass the trusted
normalizer. The normalized declaration binds the artifact to the frozen
producer; it does not replace the external authority's independent audit that
device launches are exhaustively categorized.

The Launch SDK's native Chrome dump uses `B`/`E` duration records and `s`/`f`
barrier-flow records, with cycle conversion controlled by
`LKN_RPU_FREQ_MHZ` (800 MHz by default). A trusted adapter may normalize this
to the skill's `ph=X` schema before sanitization, but must retain raw hash,
frequency, pairing, and stream/barrier completeness. Never reject or count a
launch solely because the native trace is not already `ph=X`.
The adapter maps native `Compute`/`DMA` to `rpu_device_program`/`rpu_dma` and
maps DMA `size_bytes` to `bytes`; its normalized byte rate must use exact bytes
and duration rather than the rounded native `bandwidth_GBps`. The SDK
`Kernel_t` name (`set_name()` included) is the exact Compute event name,
whereas `set_op_type()` is only a coarse label and cannot identify a manifest
kernel. Require positive frequency metadata with `--require-frequency`; reserve
`--assert-exhaustive` for an independent launch-and-barrier coverage audit.

Contract-driven signal and full verdicts run only on `rpu` and bound every
child with finite timeouts. The included CPU adapter is for board-free protocol
testing only and cannot establish RPU performance, a theoretical-limit claim,
or a zero-cost epilogue.

Do not create device nodes, alter device permissions, or bypass a container's
device policy as part of a kernel campaign. Ask the board administrator to map
the required devices into the execution environment.

## Current local assessment snapshot

The repository contains public FP16 GEMM, RMSNorm, RoPE, elementwise, and
standalone `llama_silu_mul` host paths. Availability still belongs to the exact
runtime manifest, not to source presence. The inspected runtime-v1.0.0-r4
manifest does not name MXFP8 or a GEMM+epilogue device kernel. Re-run preflight
for every runtime update; do not inherit this snapshot into a new asset.

In the 2026-09-02 revalidated snapshot, `/home/hx/.local/bin/hxcc` version
`1.0+0a6ba7e4` completed a reproducible offline compile of an existing local
`.rc` sample outside this repository. Its licensing and production asset
integration authority were not established. This is compile-only evidence, not
a support or performance claim.
