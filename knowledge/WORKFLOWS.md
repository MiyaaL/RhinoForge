# Knowledge workflows

## Query

1. Start at [INDEX.md](INDEX.md).
2. Read the smallest relevant concept or synthesis page.
3. Follow its public source links before changing code or making a support
   claim.
4. Use `rg` only after the indexed sources do not answer the question.

See [RETRIEVAL.md](RETRIEVAL.md) for the Markdown-first retrieval contract and
the boundary for optional semantic indexing.

For model assessment or implementation, use the
[`rhinoforge-port` skill](../.agents/skills/rhinoforge-port/SKILL.md) with
[model porting](../docs/model_porting.md).

For exact RPU operator or fusion performance work, use the
[`rhinoforge-kernel-opt` skill](../skills/rhinoforge-kernel-opt/SKILL.md) with
[performance measurement](../docs/performance.md). Its device campaign must
stop when board access, an authorized compiler/source interface, or the
release-matched operator asset is unavailable.

## Add or update knowledge

For source-operator migration, keep the complete reference asset and its
admission checks. Register source Programs under distinct names, use the
framework's named-kernel Graph path, and freeze selection for the process.
Validate parameter/address units and fallback geometry before board runs;
record standalone Launch-batch results separately from Graph BUILD/REPLAY and
model quality. [Source-library boundary](concepts/launch-runtime.md).
For local framework/operator dependency closure, keep REF and Launch in the
approved `rpu_ops/runtime` tree and select that Launch root explicitly at build
time. Test caller/`ENV_SH` path precedence, missing sidecar/library rejection,
and sudo forwarding of source selection and a matching Python/native package.
Do not copy opaque payloads into RhinoForge or infer admission from path
selection alone. [Local runtime setup](../docs/runtime_assets.md#local-development-with-an-approved-rpu_ops-checkout).

When merging Graph segments, record the node, command and instruction census
before changing budgets. Preserve stream fences and SDK headroom, and verify
same-input parity plus stable replay in a fresh process. A single GraphCache
entry is not proof of a single device submission; use the segment counters and
the [capture concept](concepts/graphcache-capture.md).
For a multi-step Action merge, separately validate same-dtype split/unrolled
execution and any intentional FP32-to-FP16 downgrade. Use A/B/A with changed
noise, padding and prefix contents to detect stale mutable inputs; keep public
outputs independently owned. Wall uses only `WALL_QWEN35_OPT`: default `1`
selects Vision1 + Prefill1 + Action1; `0` restores 3+3+10 on the recorded episode.
Both arms use FP16 Action math. The switch owns all six cold Graph/SDK budgets;
verify both physical submission counts and the loaded package ABI. Historical
FP32 measurements remain precision anchors, not a selectable runtime profile.
The runner requires Wall execution ABI 2: ABI 1 installations can still use the
old Prefill preset. Verify the wrapper's actual Python/native installation,
not only the source checkout, after rebuilding.

For Wall startup diagnostics, retain the installed-Python/native guards while
keeping successful validation silent. Verify the default torchvision processor
backend without forwarding `backend` to Transformers 5.5's video processor;
filter only the unused HF CUDA-kernel installation hint during Wall
meta construction, not all Transformers warnings or its fallback functions.
OpenCV video decoding is declared through the headless package in the `vla`
extra. Reinstall Python changes before testing the installed-package wrapper.

For the Wall open-loop runner, profiling uses only `--torch-profile-dir DIR`
and `--hw-perf-dir DIR` (with the existing hardware dump bound). Torch records
one compressed trace per request with shapes/stacks enabled and memory events
disabled. The first trace includes lazy setup/BUILD; do not infer frozen READY
or steady-state latency from it. Keep historical READY-probe commands bound to
their measured revision when updating the [testing guide](../docs/model_testing.md).
For Wall host-preparation changes, compare every prepared tensor against the
reference processor (including FP32 pixel values, consumed FP16 values, and M-RoPE), preserve PIL's
single-stage resampling, and wait for image workers on failures and teardown.
Retain full cache clearing for the first prefix and after prefix failures.
Cache restart without DDR clearing is restricted to subsequent complete position-zero
multi-token prefill; prove stale-state independence and stable Graph replay
against the clearing path. Keep general cache reset semantics intact and time
request entry through the model forward, not only the named image/text range.
For local asset migration, copy the exact checkpoint inference files and the
complete recorded episode, plus any explicit common-noise artifact. The local
wrapper defaults to `/mnt/nvme/miyaa/work/{ckpt,dataset,prof}`. Preserve hashed
checkpoint configs, check symlink and Python/library resolution, and run the
full host `--check` with the remote mount hidden in a private mount namespace.
Record copy hashes and any execution smoke check separately from model-quality
or release-lifecycle evidence.
Wall's three Graph labels use the common `rpu_wall_qwen35_` prefix with
`vision`, `prefill` and `action` suffixes in both OPT modes. Labels are assigned
before capture; generic Qwen3.5 naming and historical trace artifacts remain
unchanged. Keep precision/step mode in execution metadata, not stage labels.
Optimized Wall Prefill uses one outer chunk per64..384-row prefix bucket, not
just one Graph segment. Its native cold override must retain kernel/SPM guards;
the internal GDN64-row recurrence is unchanged. Validate maximum384-row SPM
with Vision/Action co-resident, consumed KV/recurrent/conv state parity and
cross-bucket A/B/A before citing speedups. Generic and split-mode envelopes stay
at128. Compare profiler-off warm Prefill and end-to-end timing separately from
BUILD/install, because one chunk also selects SPM-resident inter-layer I/O.

For GDN padding fills, keep the grid equal to `ceil(actual_elements/2048)`;
extra blocks in the release fill asset can corrupt adjacent live Q/K rows.
Stable replay uses fixed writes into each target's contiguous trailing guard,
not an oversized launch grid. See [the padding safety receipt](../docs/qwen35_gdn_padding_safety.md).
Also allocate the cumsum workspace by its kernel ABI, not input tensor size.

Cold Wall Vision labels belong to policy metadata (`_wall_qwen35_*`), not the
reserved `_rpu_*` hardware-attribute namespace. Test the real recursive
pre/post-install validators at the Wall-to-Qwen adapter boundary in both OPT
modes; forward-only doubles do not catch installation-time name violations.
Hardware filename timestamps use the process-local session start (`TZ` or
system timezone), not the per-dump write time; older native builds used UTC.
Keep dump ordering and JSON event time bases unchanged. Cover UTC, positive,
fractional and negative offsets with the compiled native formatter test, and
rebuild/reinstall the extension before verifying new files on the board.

1. Confirm the statement in public source or documentation.
2. Update the existing page that owns the fact; create a page only for a new,
   reusable concept.
3. Follow [SCHEMA.md](SCHEMA.md), add direct source links, and update
   [INDEX.md](INDEX.md).
4. Check relative links and scan the changed files for private infrastructure,
   credentials, restricted implementation detail, opaque asset internals, and
   diagnostic payloads.

## Save a reusable answer

When packing independent image streams, update the
[three-stage contract](concepts/three-stage-chunk-plan.md), Graph replay-count
expectations, and runtime capacity documentation together. Keep historical
multi-call timing/traffic ledgers explicitly separate from an unmeasured
single-call candidate.
For packed numerical parity, compare row-parallel partials and residuals before
and after reduction. Preserve semantic reduction spans when the reference
depends on ring geometry; matching GEMM outputs alone does not establish final
parity. Verify the installed Python/native package used by the public launcher,
not only a temporary validation package.
Also exercise changing real prefixes within a bucket and A/B/A bucket returns:
Wall optimized Action retains all six 64-row buckets, masks the gap before its
bucket-offset suffix, and refreshes real RoPE outside capture. Keep mask SPM
size fixed across buckets, retain per-bucket DDR slots, and clear partial KV
blocks using K axis 5 versus V axis 6. Require cumulative BUILD/REPLAY counts,
not merely a single successful warm repeat. A cold Action prime can leave a
temporary-SPM watermark: check the [component handoff](concepts/component-handoff.md)
boundaries before changing workspace sizes or clearing Graph caches. Compare
same-noise FP16 outputs and unprofiled full-wrapper timings separately; this
host Graph extension does not certify numerical or robot-task quality.

1. First update an existing concept or synthesis page when it owns the result.
2. Add a short page under [queries/](queries/README.md) only when the original
   question and decision context remain useful.
3. Keep direct public source links and mark unresolved points; do not store raw
   chat transcripts or diagnostic artifacts.

## Add an external source

Prefer a stable public URL. When an offline snapshot materially improves
provenance or availability, follow [raw/README.md](raw/README.md), confirm its
license or redistribution basis, and cite it from a maintained page.

## Resolve conflicts

Use public API documentation for user-visible behavior and source for actual
implementation. Treat model-support status as profile-specific and defer to
[model support](../docs/model_support.md) and
[model validation policy](../docs/validation_policy.md). Update stale knowledge
in the same change; do not preserve conflicting summaries.
