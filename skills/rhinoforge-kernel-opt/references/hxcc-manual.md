# hxcc / RPU Developer Guide overlay

Read this reference before a Campaign or Resume that compiles `.rc`, inspects
RPU assembly, or uses the Rhino Launch SDK.  It is a compact, clean-room
operational digest of the official files in `/home/hx/miyaa/work/src/hxcc.zip`;
the archive itself is not vendored into this repository and its opaque `.ref`
and hardware traces must stay outside Git.

## Frozen source identity

The source archive observed on 2026-09-02 has SHA-256
`21e8ed1462620b84e4d818eae24b8507817cbeca1a39105cf3863d8c8287eeba`.
Relevant member hashes are recorded here so a future assessment can detect a
different manual or toolchain without trusting a filename:

| artifact | SHA-256 |
|---|---|
| `RPU_Developer_Guide_V1.0.html` | `ff0ad7bee5f2f1ebda9cf0cabb0d3f9dd0872534a33fd907185028fdb02e8594` |
| `RPU_Developer_Guide_V1.0.pdf` | `c6ed99cc9c46eb585cd2505007351a04717bfc86a67a8c5572322388450263f0` |
| `hxcc - RPU Clang 编译器工具链.pdf` | `0a34979ed65099ea73f6c2bae0f8272568d7d695ee55b10a1918c1349f8a6114` |
| `hxcc-1.0+0a6ba7e4-py3-none-manylinux2014_aarch64.whl` | `04f99c79025e48769aff78fee97c797234af40f27783001dc8eb7faecac1376e` |
| `hxcc-1.0+0a6ba7e4-py3-none-manylinux1_x86_64.whl` | `921d7a54dd4e9aaefb377e16885eaebf73bc0592db09ae52a14b43b2322cfeb8` |
| `rhino-launch-kernel-unknown-linux-aarch64.tar.gz` | `2cb9f96f3a6c921936ec421ad0061253b438ff97f331c3bdd186cb97c5412d1b` |
| `mm_test.tar.gz` | `e93c0ab97116066d0fb8856a83227bb5aaf1dc859357fb1ee6b76b4e084e544c` |

The wheel metadata reports `License: UNKNOWN`; a local `file://` install URL
is provenance only, not an authorization to compile, redistribute, or ship a
device program.  Pin the exact wheel, package tree, and binaries in a reviewed
authorization receipt.  For the observed ARM package, the underlying binary
hashes were:

```text
clang-17          999f36abd6b7920e9705c827ad12aa41dca9af5735df033e8af773a4d28ad72d
rpuas             b43aeb9cedc19cdd80210ff59d5805aa3f1eec99f8e597a14c564abc94703e02
rhino_gen_oplib  e502faf9958561a9e8240244ca3079d9d87db04f8a10a37d2345d0c891d70d45
```

These are evidence for that installation, not universal release constants.

Useful anchors in the HTML guide are §2.3.1 (entry ABI), §2.4.3–§2.5.2
(offline/online compile and Launch deployment), §7.2 (optimization advice),
§7.3.2/§8.2 (blocked GEMM and the 32x32x64 worked structure), and §9.1–§9.3
(limits and API table).  The SDK header comments are the authority for the
exact Launch library shipped with a campaign; where they disagree with the
guide, preserve both observations and resolve the target release explicitly.

## Compiler contract

The supported board-free compile path is:

```bash
hxcc -O2 -c kernel.rc
```

`-O2` is the documented/recommended level; `-O0` is not supported.  The
wrapper emits an object and an opaque `.ref`; `-S` is not a supported output
mode and `-o` cannot safely replace the wrapper's dual outputs.  Run every
compile in a fresh, private build directory and retain only output hashes and
sizes.  Never put `.ref`, `.o`, generated assembly, or source-derived traces in
the candidate Git tree.

The wrapper delegates to the package's `clang-17`, `rpuas`, and
`rhino_gen_oplib`.  A dry driver invocation with a real (even tiny) `.rc` must
show all of the following before it is accepted as toolchain evidence:

```text
target triple: rpu-rhino-rpuhsa
optimization:  -O2
assembler:     rpuas
oplib command: rhino_gen_oplib ... -m r1
```

Do not use `hxcc --print-target-triple` without an input as target evidence: it
can report `unknown`.  `rpuas` treats `--help`/`--version` as input filenames
in this release, so check its executable/ELF identity and the end-to-end hxcc
smoke instead of interpreting those flags as a capability probe.  The
`scripts/hxcc_preflight.py` helper performs these checks without a board.

The host architecture selects the wheel (`manylinux2014_aarch64` on this
machine; `manylinux1_x86_64` on x86-64).  The wrapper's version is
`hxcc version 1.0+0a6ba7e4`; the underlying compiler reports LLVM 17.0.2 with
commit `0a6ba7e4`.  Pin wrapper and underlying binary hashes separately: a
wrapper hash alone does not identify the compiler that it execs.

The assembly/source inspector in this skill is intentionally a conservative
lexical/structural check rather than a complete C++ parser or an ABI proof.
Treat a passing report as a necessary review receipt only after the exact
source has compiled with the pinned headers. For a release handoff, the
resulting asset must additionally be admitted by the release manifest. A
source-derived diagnostic may keep the passing assembly receipt before that
admission, but it is not a promotion or production-runtime claim.

## Device source and ABI rules

Use these as source-review gates, not as a substitute for compiling the exact
candidate:

- A kernel entry is `__rprog__` and must return `void`.  Device helper
  functions are forcibly inlined; there are no function pointers or indirect
  calls.
- Device code has no heap, exceptions, virtual dispatch, STL containers, or
  recursion.  Do not use `new/delete`, `try/catch`, virtual methods, or
  recursive call graphs in an RPU source candidate.
- Vector data belongs in `__local__` VLM; unqualified local SRAM pointers,
  `__SRAM` global SRAM, and `__DDR` DDR are distinct address spaces.  Do not
  decorate scalar locals with `__local__`.
- Pointers and 32-bit scalars consume two 16-bit parameter slots.  The guide
  describes 64 parameter registers and says compiler parameters occupy 0..61;
  the shipped `rhino_launch_kernel.h` comments describe a 0..67 range while
  also calling it 34 16-bit slots.  Treat this as a release mismatch: resolve
  the budget with the exact compiler *and* launch headers used by the campaign,
  record the result, and leave a safety margin.  Never silently assume either
  number.
- Do not pass a user-defined struct or custom data type as a kernel argument.
  When the reviewed scalar/pointer arguments exceed the available parameter
  slots, put the overflow in host-populated Constant Memory (the compiler can
  emit `ldc`, or device code can use the documented `sldc<T>` helper with byte
  offsets).  Count and freeze that Constant-Memory layout as part of the host
  ABI; it is not a free replacement for a register argument.
- The worked GEMM source declares loop metadata with an unsigned-short shape,
  while the shipped header's `LoopOutInfo` constructor accepts `short*`.  A
  compile failure at this boundary is an ABI/version signal; do not paper it
  over in the evaluator or claim that the unmodified sample is portable.
- `f16v16` is the preferred vector type.  The guide defines an `i8v16`/FP8
  vector as 16 bytes (one 16B entry) and an `f16v16`/`bf16v16` vector as
  32 bytes (two 16B entries, also described as a g32B entry).  Its capacity
  section says that each thread owns 800 physical 16B entries, i.e. 12.5 KiB.
  Therefore budget an all-`f16v16` allocation as at most 400 vectors before
  reserving compiler temporaries and a safety margin; never interpret the
  headline as 800 `f16v16` vectors or 25 KiB.  Keep typed indices and external
  byte offsets explicit, and confirm the final allocation in the emitted
  assembly for the exact compiler release.  The shipped
  `f8e4m3v16`/`f8e2m5v16` declarations are not, by themselves, proof of native
  MXFP8 block-scale quantization.

## Loops, Repeat, addresses, and pipelines

The compiler can silently lower an invalid hardware-loop request to a software
`wjump`.  Inspect generated assembly; source pragmas alone are not evidence.

- Put `#pragma rpu_hwloop_*` immediately before its `for`.  Start must be a
  constant or warp-level value, step must be constant, and end may be
  thread-level.  Maximum nesting is eight; a Repeat consumes two loop levels.
- Use Repeat only for a single regular instruction/body with no branch.  Set
  inner dimension 0 and outer dimension 1 (`Tsize`/`Tstep`); account for the
  selected 16B/32B address granularity and `stepbyte` conversion in every
  `Tstep`.  A complicated body is usually better expressed as explicit
  hardware loops.
- Address modes `dir`, `add`, and `dlt` are mutually exclusive; `lpaddr` can
  be combined with them.  `dlt` updates after use.  Check granularity and
  stride units, especially for tail tiles and non-16-aligned rows.
- Prefer the documented overlap pipeline: async VLD/VMAT/VALU/VSFU/VST, then
  the matching `fenceXxx` (`fenceVld`, `fenceVst`, `fenceValu`, `fenceVmat`, or
  `fenceVsfu`) before a consumer.  Synchronous Execute does not need a fence,
  but cross-unit dependencies do.  A hot-loop `wjump`, an async use without a
  fence, or an over-depth loop is a failed optimization hypothesis, not a
  performance win.

For the canonical FP16 GEMM, the guide uses row-major `A[M,K]` and `B[N,K]`
and computes `A * B^T`; the worked tile is 32x32x64 with 16-lane vectors and
three M/N/K hardware loops.  It uses VMAT's internal FP32 accumulator, output
loop control (`outlp1`/`clrlp1`), `lpaddr`, and a staged load/compute/store
pipeline.  Start tuning around this structure, then search only the shapes
admitted by the campaign contract.  It is not evidence that every shape or
fusion exists in the production asset.

### Address-unit checklist

An address wrapper stores a logical address in its selected granularity, not
always a byte address: for example, `DDR_16B{5}` denotes byte offset 80.  The
default `LoopStepInfo::automatic` mode multiplies a step by the type's
`stepbyte`; an explicit `a_g1B`/`a_g16B`/`a_glarge` (or paired `__ABGran`) can
override that conversion.  Keep VLM tile steps and external byte/line steps
separate, and record the chosen granularity for every operand.  A stride that
looks correct in elements can otherwise become a 16B/32B/256B overrun.

## Host Launch SDK and replay

The official single-kernel path is `Program_t::create_with_binary_file(ref,
kernel_name)` → `Kernel_t` register setup → `LocalSPM_t` buffers → queue
enqueue.  The public SDK links `librhino_launch.so`, `rpu_api`, and `Hxil`.
`LocalSPM_t`/`GlobalSPM_t`/`HostDDR_t` describe different spaces; direct DDR
allocation is needed only when the device program actually addresses DDR.

Transfers must be 32-byte aligned (and DMA lengths multiples of 16 bytes),
each transfer is at most 8 MiB, and up to eight DMA channels can be batched.
Pointers are packed as two 16-bit registers.  For repeated work, construct the
queue once with `build_batch()`, patch only reviewed mutable kernel/DMA fields,
call `sync_mutable_params()`, and use `enqueu_batch()`; this is a
`rhino_launch_batch` lifecycle and must not be mislabeled as a PyTorch Graph
BUILD/REPLAY.  Multi-core work needs explicit stream barriers and, where
appropriate, `set_broadcast_mode`; record core IDs and queue settings.

The queue also exposes `set_warp_num(1..8)`,
`set_warp_schedule_mode(0..3)`, instruction-cache/fetch controls, and
`set_enable_hw_perf`.  Freeze these knobs in the campaign when they can affect
latency.

The `Kernel_t` constructor's optional `name` and `set_name()` value is emitted
as the Compute event's exact Chrome `name`; an empty name falls back to
`kernel_<batch_idx>`.  `set_op_type()` is different: it writes only the coarse
`args.op_type` label (for example `matmul`, `elementwise`, or `reduce`).  It is
not a manifest identity and must never be promoted to the fused-kernel name.
For an exact launch gate, set the reviewed manifest name explicitly (or use a
separately reviewed, deterministic name map) and fail closed when only the
default `kernel_N` or an `op_type` label is available.

## Hardware perf evidence

Call `set_enable_hw_perf(true)` before adding kernels/DMA or building a batch.
Each kernel/DMA emits START and END `R1Message_t` records (8 bytes each,
16 bytes total); disabling the feature is documented as zero-overhead.  Read
the buffer only after a finished batch.  For parallel streams, barrier every
active stream to stream 0 before the implicit final packet or trailing slots
can be zero-filled.

`dump_hw_perf_chrome` uses `LKN_RPU_FREQ_MHZ` (default 800 MHz) and emits
`B`/`E` duration events plus `s`/`f` barrier-flow events.  In the shipped SDK,
duration events use native categories `Compute` and `DMA` (barriers use
`SYNC`); `otherData.frequency_hz`, `base_cycle`, `batch_events`, and
`records_captured` are decimal strings.  Convert this input format to the
skill's normalized `ph=X` representation in a separate private diagnostic
step with `scripts/normalize_hwperf.py`; the normalized file still contains
review-only names/timestamps, while `summarize_hwperf.py` creates the final
pointer-free summary.  Apply this fixed mapping:

| native field | normalized representation |
|---|---|
| `cat: "Compute"` | `cat: "rpu_device_program"` |
| `cat: "DMA"` | `cat: "rpu_dma"` (excluded from the one-program gate) |
| DMA `args.size_bytes` | `args.bytes` |
| DMA `args.bandwidth_GBps` | retain the rounded native value in the report; derive normalized `bandwidth_bytes_per_s` from exact `bytes / duration` |

The converter must pair every `B`/`E` and `s`/`f`, retain the raw-trace hash,
frequency, and stream/barrier counts in its private report, and never expose
addresses, raw timestamps, or traces in Git.  A missing `ph=X` event is not
evidence of no launch.  The `--require-frequency` option requires positive
frequency metadata; it is an input-integrity check, not a peak-frequency or
coverage proof.  `--assert-exhaustive` may be used only after an independent
verifier has checked that every device launch and active-stream barrier was
captured; pairing and counter lower bounds alone cannot establish that claim.

## Exact-profile implications

- GEMM: freeze M/N/K, tail policy, A/B orientation, SPM/DDR residency, VMAT
  accumulation and output conversion.  Count DMA and mandatory bytes at the
  actual hierarchy (DDR → SPM → VLM), not just arithmetic.
- GEMM+SiLU-mul, GEMM+add, and GEMM+RoPE: SPM/Graph reuse can remove a DDR
  round trip, but it is not a one-program fusion.  Require an exact manifest
  name and one sanitized device-program launch before calling it a fused
  kernel.  Pair each candidate with its independently warmed bare operation.
- RMSNorm+quant_mxfp8: the guide/headers expose FP8 enums and placeholder
  vector types but no complete MXFP8 block-scale packing, rounding, or
  saturation ABI.  Keep this profile Blocked until the exact quantizer and a
  release-matched asset are supplied; the portable int8 adapter is only a
  protocol fixture.

Before a timed candidate, require: toolchain identity receipt; source lint;
`-###` target/`-m r1` evidence; end-to-end compile in an isolated directory;
assembly counts for hardware loops/Repeat/`lpaddr`/async/fence/`wjump`; and a
Launch or Graph lifecycle receipt appropriate to the selected `launch_mode`.
Only then can empirical peak, bandwidth, and epilogue-tax measurements be
compared.  No manual headline number is a theoretical peak for this board.
