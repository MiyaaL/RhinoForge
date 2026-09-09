# Qwen3.5 GDN padding-fill safety

The 2026-09-09 Wall Prefill investigation found an actual SPM write overrun in
the old fixed-grid padding fill, not a large-row L2-normalization failure.
The release `fill` ABI requires `grid.x = ceil(num_elements / 2048)`.
Supplying a bucket-sized grid while reducing `num_elements` for a short pad
does not safely disable the extra blocks.

For a frozen 308-token input, per-core first-layer snapshots showed correct
projection, convolution and immediate L2-normalization outputs. The subsequent
five padding fills changed 14,336 valid Q halfs in a 128-row chunk and 77,823
valid K halfs in a 320-row chunk. This also affected the old path, so its outputs
are not an implementation-parity reference for the repaired path.

Each potentially zero-filled GDN buffer now owns contiguous trailing guard rows:
`G = min(chunk_size - 1, 127)`, matching the adapter's 63 mandatory plus 64 optional
padding cap. All mathematical operations still use the original logical shape.
A constant `G` rows are filled beginning at the actual valid-row count `Lv`:

- `[0, Lv)` is untouched;
- all logical padding `[Lv, L)` is cleared;
- excess writes stay within the same buffer's trailing guard;
- exact-length chunks write only guard storage;
- five fixed-shape fill nodes remain replayable as valid lengths change.

Native code rejects padding beyond this bound and rejects a fill grid containing
more blocks than its actual write extent. The cumsum workspace is also corrected
to its existing ABI: `64 * 17 * 16` halfs (34,816 bytes), independent of batch size.
The fixes apply to both split and optimized GDN Prefill; Action has no GDN mixer.
Neither a new device kernel nor an SPM-capacity bypass is used.

Board-free tests compile and exercise the production fill wrapper's rejection
path, check guarded extents for all 64-row buckets through 384, and check the
manifest. Private diagnostic snapshots after repair match bitwise between
128-row chunks and the 320-row whole chunk through the first GDN layer's
projections, convolution, normalized Q/K, gates, cumsum, decay, matrix outputs,
and all five 64-row recurrent states. Raw activation artifacts stay outside Git.
This is a memory-safety/localization result, not full-model numerical or task
quality certification. Regenerate old traces and predictions before comparison.
