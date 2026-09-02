# Kernel-family contracts

Read only the section for the requested operation. These are semantic starting
points, not support claims. Freeze exact shapes, precision, layouts, and asset
names in the campaign contract.

Every fusion profile defines a profile-specific `bare_operation(case)` and an
independent `bare_reference(case)`, plus
`arm_graph_state('candidate'|'bare_operation')`. The two bare hooks' returned
tree, dtype/device residency, strides, and layout must match the registered bare
baseline. Each arm state returns `build_count`, `replay_count`, `cache_size`,
and `invariant_ok`; fusion tax is admissible only when both arms independently
hold BUILD at one, grow REPLAY from a warm state, keep cache size fixed, and
keep the invariant true. The fusion itself is identified only by the resolved
`runtime.fused_kernel`, which must exactly equal one entry in
`runtime.required_kernels`; never infer fusion from name tokens or substrings.

For source-level candidates, apply the device/loop/address rules in
[the hxcc manual overlay](hxcc-manual.md) and attach an assembly inspection
receipt. In particular, a pragma is not a hardware-loop result until the
generated assembly has no unapproved `wjump`; Repeat consumes two loop levels,
and async unit boundaries require matching fences. Count the DDR→SPM→VLM
traffic implied by the exact residency contract when building a roofline.
Use the [optimization playbook](hxcc-optimization-playbook.md) to order tile,
pipeline, address, and epilogue hypotheses without changing more than one
primary axis per candidate.

## GEMM

Default mathematical contract:

```text
Y[M,N] = X[M,K] @ W[N,K]^T
```

Freeze transposition, weight swizzle, accumulation dtype, output dtype,
rounding, bias absence/presence, and whether inputs/outputs begin/end in DDR or
SPM. Count `2*M*N*K` operations only when one multiply plus one add are the
registered convention. Mandatory bytes must reflect the actual admitted
residency and cannot omit required DMA merely because a profiler overlaps it.

Use at least decode-like small M, a representative prefill tile, non-multiple
tail behavior admitted by the profile, and the maximum envelope. Compare the
generated Linear tile plan with the exact asset manifest; do not add a model-
level tile selector.

The runtime-v1.0.0-r4 snapshot contains FP16 SPM GEMM tiles and shared parallel
Linear paths. That does not provide device source or admit every `(M,N,K)`.

## GEMM + SiLU-mul

Default gated-MLP contract:

```text
G = X @ W_gate^T
U = X @ W_up^T
Y = silu(G) * U
```

Freeze the exact SiLU definition, evaluation/accumulation dtype, order of
rounding, whether gate/up GEMMs are independently materialized, and aliasing.
The bare baseline is the same two GEMMs with identical output residency. The
unfused baseline adds the exact standalone activation and multiply. A valid
fusion result must match this registered low-precision order; matching only an
algebraically rearranged FP32 expression is insufficient.

The inspected asset names standalone `llama_silu_mul` plus GEMM/Linear kernels,
but does not name a GEMM-epilogue SiLU-mul device kernel. Host Graph/SPM
scheduling can remove DDR round trips while still launching more than one
device kernel; report that accurately. A one-launch device fusion remains an
asset/compiler gap until a release-matched name is delivered.

## GEMM + add

Choose exactly one add contract:

```text
bias:     Y = X @ W^T + B[N]
residual: Y = X @ W^T + alpha * R[M,N]
```

Bias and residual add have different traffic and broadcast semantics and are
not interchangeable. Freeze `alpha`, broadcasting, dtype, accumulation,
rounding, in-place behavior, and output ownership. Include same-shape changed-
value tests for both `B` or `R` so a captured address/value cannot go stale.

Some public Linear wrappers accept bias and the asset contains standalone
binary kernels. That does not prove a general residual-add GEMM epilogue.
Classify each exact contract from its wrapper and manifest evidence.

## GEMM + RoPE

Default Q/K projection contract:

```text
Q = X @ W_q^T
K = X @ W_k^T
(Q_rot, K_rot) = rope(Q, K, cos, sin, position)
```

Freeze Q/K head counts, head dimension, partial rotary dimension, interleaving,
position convention, cos/sin layout and precision, padding, batch packing, and
returned layout. Positions and tables are semantic inputs: include them in the
Graph signature or refresh them through the validated mutable path.

The bare baseline is the identical Q/K projection pair. The unfused baseline
uses the exact registered RoPE kernel and the same SPM/DDR residency. Test the
first, adjacent, maximum, repeated, and same-shape changed positions. For
multimodal RoPE, freeze all axes and section routing; 1-D RoPE evidence does not
transfer.

This skill's `gemm+rope` template and portable adapter both mean this two-
projection Q/K contract. A single-projection experiment is a different exact
profile and must not reuse its operation counts, bare baseline, or receipts.

The inspected asset contains separate `llama_rope`, `llama_rope_ddr`, partial
RoPE/M-RoPE, and GEMM paths, but no named GEMM+RoPE epilogue kernel.

## RMSNorm + quant_mxfp8

Default normalization contract:

```text
Z = W * X * rsqrt(mean(X^2, last_dim) + eps)
(Q, S[, metadata]) = quantize_mxfp8(Z, block_spec, rounding, saturation)
```

Freeze epsilon, reduction and reciprocal-square-root precision, learned-weight
placement, normalized width, block orientation/size, MXFP8 element format,
scale format, scale encoding, rounding, saturation, special values, signed
zero, padding, payload packing, scale layout, and metadata. The output is a
tree. `quantized_output_roles()` must cover every tensor-leaf path exactly once
with `payload`, `scale`, or `metadata`, including at least one payload and one
scale. Compare every encoded leaf bit-for-bit and stride-for-stride. Separately,
`fp32_anchor(case)` returns the unencoded FP32 RMSNorm target and
`dequantize(output, case)` converts the encoded candidate tree for comparison
against it. The anchor uses `anchor_max_abs`/`anchor_max_rel`, independently of
the same-dtype encoded-tree `max_abs`/`max_rel` gates.

Here `bare_operation` and `bare_reference` are RMSNorm, not GEMM. Measure that
bare RMSNorm, standalone exact quantization, the unfused chain, and the fused
path. Count required normalization operations and quantization bytes from the
frozen format rather than applying a generic GEMM roofline.

The runtime-v1.0.0-r4 manifest names RMSNorm variants but no kernel containing
`mxfp8`. NVFP4 or W8A16 support is not MXFP8 evidence. This profile is
`Blocked` until an exact quantization definition, authorized compiler/source
path, and release-matched asset entry exist.

The public eager RMSNorm wrapper in the inspected source uses a CPU fallback
for an output-address limitation, while fused direct-address SPM paths can run
RMSNorm on RPU. Freeze which path is under test; evidence from the fused SPM
path must not be attributed to eager RMSNorm or vice versa.

## Fusion-cost verdict

For every fused profile, report all three where applicable:

- bare compute latency;
- equivalent unfused-chain latency; and
- fused latency.

State device-launch count and DDR/SPM traffic for each. A Graph that keeps
intermediates in SPM can have a small marginal tax without being a single
device kernel. Reserve "kernel fusion" for evidence that the release asset and
profile execute the registered work in one device program. Full promotion
requires a separate one-call trace re-sanitized with the complete manifest
allowlist. Its selected identity must be the exact fused name and its total
manifest-kernel launch count must be one, plus a paired-bootstrap epilogue-tax
CI upper bound below the frozen gate.
