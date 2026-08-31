# External release attestation

Local Python processes, `/proc` identities, `flock`, trace JSON, and a hash
chain owned by the campaign user are diagnostic evidence. They are not a
security boundary: that user can fabricate or replace all of them. Therefore
`BEST.json` requires a DSSE envelope signed by the Ed25519 public key frozen in
the approved contract.

Contracts support two explicit modes. `unavailable` uses the exact
`not-available`/zero sentinels, lets local preflight and diagnostics proceed,
and makes selection fail immediately. `external` requires the pinned key,
policy/build hashes, and verifier binary identity described below. Moving from
one mode to the other changes the contract hash and starts a distinct evidence
campaign.

## Authority boundary

The corresponding private key must be unavailable to the campaign UID and to
candidate native code. Use a separately administered service, UID plus device
proxy, TPM/HSM, or remote runner. A generic `sign(bytes)` endpoint is not
admissible. The authority accepts only a release challenge, contract digest,
and candidate request; it constructs the statement itself after it:

1. acquires exclusive device access and a persistent, authority-owned lease;
2. snapshots the exact clean candidate commit into read-only content-addressed
   storage and independently hashes the source tree;
3. reruns preflight under that lease and independently verifies the
   symlink-free reference content tree against
   `campaign.reference_tree_sha256` before and after the session;
4. launches and observes every profile and benchmark process;
5. creates each profiler output itself, rather than accepting a trace path;
6. audits that the contract-pinned benchmark-adapter producer really
   categorizes all and only device-program launches as `rpu_device_program`,
   checks the exact `rhinoforgeTrace` declaration and producer SHA-256, parses
   the `rhinoforge-rpu-chrome-v1` raw trace, and requires exactly one complete
   event in that category and one exact admitted manifest launch; host-scope
   `X` events in other categories are ignored, while an unknown event in the
   device category fails;
7. revalidates every result and selects from the exact signed journal snapshot;
8. advances its persistent proof sequence/transparency head; and
9. signs only the statement it constructed from those observations.

The local `run_full_verdict.py` exercises the same checks for debugging, but
always emits `release_eligible=false`. Its `collection_ok` says the session was
recorded; `all_gates_passed` and `ok` are true only when every full receipt
passed. A collected failed candidate exits nonzero. Do not submit that
diagnostic JSON to a blind signing service.

For a fusion, the admitted identity is the contract's exact
`runtime.fused_kernel`, itself an exact member of `runtime.required_kernels`.
Neither the local verifier nor the authority may substitute token or substring
matching for this identity.

`rhinoforgeTrace.device_program_events_exhaustive=true` is a declaration by the
producer whose hash equals `commands.benchmark_adapter_sha256`; it is not an
independent hardware attestation. The authority must inspect and control that
exact producer and establish its exhaustive instrumentation before relying on
the trace.

## DSSE envelope

The verifier admits one signature, the exact payload type
`application/vnd.rhinoforge.kernel-release-proof.v1+json`, canonical JSON
payload bytes, and the contract-pinned raw Ed25519 public key. The payload is
signed using DSSE v1 pre-authentication encoding. Verification executes the
contract-pinned OpenSSL binary by its already-open `/proc/self/fd` descriptor
after checking the exact binary SHA-256, avoiding a path hash/exec race. The
exact statement schema is implemented in `scripts/select_best.py`; it binds:

- proof ID, signer policy/build, fresh release challenge, sequence, and prior proof;
- contract, preflight, adapter, benchmark argv, and profile argv hashes;
- candidate/parent commit, Git tree OID, and an independent SHA-256 source-tree
  identity;
- board identity and a complete lease interval;
- authority-observed preflight, profiler, and benchmark processes;
- one fresh raw trace and sanitized summary per exact workload; and
- the ordered receipt digests plus exact results-journal byte snapshot.

The release pipeline supplies `--expected-challenge` from state outside the
campaign workspace and records the accepted sequence/proof hash. Passing the
challenge copied from an old envelope does not establish freshness. Key
revocation and monotonic sequence enforcement also belong to that external
pipeline; if it cannot provide them, release remains blocked.

## Source-tree SHA-256

The verifier enumerates the selected commit with `git ls-tree -r -z`. Starting
with `SHA256("rhinoforge-source-tree-v1\0")`, it appends, for each Git entry in
tree order, an eight-byte big-endian length followed by the object mode, raw
path bytes, and SHA-256 of the blob. A gitlink uses SHA-256 of
`"gitlink\0" + object-id` instead. The authority must use the identical
algorithm.

## Promotion command

After the external authority returns a proof for the exact journal snapshot:

```bash
python scripts/select_best.py <workspace>/contract.toml \
  <authority-results>/results.jsonl \
  --release-proof <authority-results>/release-proof.dsse.json \
  --expected-challenge "$RELEASE_PIPELINE_CHALLENGE" \
  --output <authority-results>/BEST.json
```

Never place a private signing key, signer implementation, reusable challenge,
or raw trace in the campaign repository.
