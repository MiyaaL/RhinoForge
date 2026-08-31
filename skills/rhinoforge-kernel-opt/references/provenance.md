# Design provenance

This skill was independently written for RhinoForge. It uses public workflow
ideas from the following frozen sources; it does not vendor their repositories,
prompts, evaluator, or generated artifacts.

## Kernel Design Agents

- Repository: <https://github.com/mit-han-lab/kernel-design-agents>
- Reviewed revision:
  `dda6be3cf1baedd3ed9c76511ef02f72243cc14c`
- Relevant public concept: define a task contract, keep candidate evidence, and
  promote only after correctness plus measurement.
- License finding on 2026-08-31: the reviewed root and pinned skill submodules
  had no merged `LICENSE`, `COPYING`, or `NOTICE`. The open license issue and
  draft licensing pull request were not treated as permission.
- Public evidence: [license issue #1](https://github.com/mit-han-lab/kernel-design-agents/issues/1)
  and [draft pull request #2](https://github.com/mit-han-lab/kernel-design-agents/pull/2).

Consequently, this skill uses a clean-room design and original wording rather
than copying KDA prompts or skill files.

## AKO4ALL

- Repository: <https://github.com/TongmingLAIC/AKO4ALL>
- Reviewed revision:
  `8d3065a7588c124182fa0fe0b3e589879641e1b7`
- License: MIT, Copyright 2026 TongmingLAIC.
- License file: <https://github.com/TongmingLAIC/AKO4ALL/blob/8d3065a7588c124182fa0fe0b3e589879641e1b7/LICENSE>
- Relevant public ideas: resolved campaign inventory, baseline before
  optimization, measured candidate transactions, explicit stall reassessment,
  full verdict after a cheaper iteration signal, and restoring the exact best
  commit.

AKO4ALL's bundled evaluator is CUDA-specific: it depends on CUDA device APIs,
CUDA events, cache-thrashing, and Nsight Compute. It also does not implement
RhinoForge Graph/DMA/output-lifetime gates or quantized multi-output semantics.
This skill therefore implements an RPU-specific contract and runner instead of
porting that evaluator.

## RhinoForge authority

For RPU behavior, the checked-in RhinoForge architecture, runtime assets,
validation policy, performance guide, public host wrappers, and release-matched
`.kernels` manifest are authoritative. External GPU workflow repositories do
not widen the RPU operator asset, public profile, precision, or device-program
boundary.
