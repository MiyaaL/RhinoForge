# Subagent campaign protocol

Use this reference when more than one agent participates.

## Roles

- **Orchestrator:** owns the immutable contract, evaluator, campaign DAG,
  board queue, stall decisions, and final synthesis. It does not silently edit
  an optimizer's measured candidate.
- **Optimizer:** owns one worktree/branch and one hypothesis. It may edit only
  its candidate path and candidate-local notes.
- **Board runner:** owns the exclusive RPU lease and invokes the read-only
  harness. It never accepts evaluator changes from an optimizer branch.
- **Profile analyst:** reads sanitized counter summaries and recommends a new
  direction without modifying candidate code.
- **Verifier:** checks a frozen commit with final seeds, holdouts, clean process,
  lifecycle gates, and full measurement. It tries to refute promotion.

One agent may fill multiple read-only roles only when independence is not a
certification requirement. The final verifier must not be the optimizer that
authored the candidate.

## Isolation

Create one isolated candidate Git repository (or worktree) and
`opt/<op>/<profile>/<direction>` branch per optimizer. `candidate_root` must be
that repository's Git top level. The campaign root is outside it and owns the
contract, evaluator, references, board lease, and local diagnostic result
store. Optimizer worktrees mount or receive those paths read-only.

Do not run two board measurements concurrently. RhinoForge fused paths may
permit only one live RPU-resident model or policy per process, and concurrent
work also invalidates timing and thermal assumptions. Compile or analyze in
parallel; serialize board runs through the skill's one canonical lease.
Alternate lock paths are forbidden because they create independent lock
domains. The chained lease journal is coordination evidence, not a sandbox
against hostile same-UID native code.

Use collision-resistant run IDs, not second-resolution timestamps. Each run
gets its own directory outside Git for stdout, raw trace, and sensitive
artifacts; Git stores only the reviewed result receipt and hashes.

## Assignment contract

Every optimizer receives:

- exact operation/profile and writable paths;
- exact candidate parent commit;
- reference and evaluator paths, read-only, plus the frozen symlink-free
  `campaign.reference_tree_sha256`;
- public and fast-signal workload IDs;
- correctness and performance gates;
- for fusion, the candidate/bare arm Graph-state hook and its independent
  BUILD/REPLAY/cache/invariant gates;
- allowed language/build interface and dependency policy;
- maximum attempts for the direction; and
- the command that records a completed attempt.

Do not ask an optimizer to "make GEMM fast" without exact shapes and semantics.
One assignment changes one primary axis so its outcome is interpretable.

## Iteration receipt

Every attempt, including build failure, records:

- ID, parent, hypothesis, changed axis, and commit;
- build/launch result;
- correctness and lifecycle gates;
- raw fast-signal samples and summary;
- profile evidence when collected;
- keep/revise/reject decision; and
- next direction or blocker.

The optimizer commits candidate code before measurement. The board runner
measures that clean frozen commit; the orchestrator then records the receipt in
a separate evidence store. A receipt is never added by amending the candidate
commit it identifies. The orchestrator rejects a new attempt when the prior
measured attempt lacks either frozen candidate identity or a receipt.

## Promotion flow

1. Orchestrator chooses finalists only from complete signal receipts.
2. Board runner checks out each exact candidate commit into a clean verifier
   tree, verifies the reference content tree, and starts a timeout-bounded RPU
   process. Signal and full sessions recheck the reference identity afterward.
3. Verifier runs full public plus held-out correctness, pointer/content
   mutation, output lifetime, warmup, Graph, stability, and performance gates.
4. The local launcher invokes the frozen profiler command and rejects any
   trace other than one complete admitted launch; this remains diagnostic.
5. A separately administered release authority repeats the entire session,
   owns device access and trace collection, and signs its statement with the
   contract-pinned Ed25519 identity.
6. Orchestrator selects only from that exact signed receipt snapshot and writes
   `BEST.json`; an unsigned local journal is never promotable.
7. Restore code from the exact selected commit and rerun one final clean
   verifier pass.

Conflicts, missing artifacts, unavailable board access, and unrun gates remain
explicit. No agent votes a missing measurement into a pass.
