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
Also exercise consecutive requests with changing exact prefixes: a cold Action
priming call can leave a temporary-SPM watermark that a single warmup/replay
probe misses. Check the [component handoff](concepts/component-handoff.md)
boundaries before changing workspace sizes or clearing Graph caches.

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
