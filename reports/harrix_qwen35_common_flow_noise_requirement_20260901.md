# Harrix Qwen3.5 common flow-noise requirement

## Problem

The original RPU/Thor accuracy report was not a same-input comparison. The Wall
RPU runner re-seeded a CPU generator for every request, while Harrix seeded the
process-global CUDA RNG once. The resulting initial flow samples differed and
could dominate per-frame action MAE, especially the gripper dimensions.

The updated 2026-09-01 report is a valid same-noise comparison: a temporary,
non-invasive Python hook captured Harrix BF16's actual CUDA noise and injected
the exact array into Thor MXFP8 and Wall-RPU. The hook was removed after the
run. Harrix still needs the native runner support below so this workflow does
not depend on an out-of-tree evaluation hook.

For request 1 (frames 32–63), changing only the RPU flow seed from 3408 to
3407 changed abs14 MAE from `0.289871` to `0.173857`, peak frame MAE from
`0.490640` to `0.243915`, and gripper MAE from `1.546932` to `0.819060`.
Therefore unmatched-noise results cannot be attributed to device precision.

## Required Harrix behavior

1. Add `--flow-noise PATH` to `run_qwen35_openloop.sh`.
2. Load an unpickled NumPy `.npy` array with exact shape
   `[model_requests, 32, 26]` and dtype convertible to finite FP32. Also accept
   `[model_requests, 1, 32, 26]` by removing the singleton batch dimension.
3. Select `flow_noise[request_idx]` for each request and pass it as the native
   Qwen3.5 `init_noise` tensor. Both `optimized` and `thor_mxfp8` must consume
   the exact supplied values; neither may call `torch.randn` when the artifact
   is present.
4. Save the exact consumed array as `flow_noise.npy` in the result directory.
   Record the source-file SHA-256, saved-artifact SHA-256, contiguous FP32
   value SHA-256, per-request value SHA-256, shape, and dtype in `meta.json`
   and `segments.json`.
5. Reject a missing file, wrong request count, wrong horizon/dimension,
   non-numeric input, or NaN/Inf before model weight loading.
6. Keep the current seeded CUDA RNG behavior only when `--flow-noise` is not
   supplied, but export the samples actually drawn so the run remains
   auditable.

## Existing implementation hook

Harrix already has the required model-level ABI:

- `python/harrix/model_executors/qwen3_5/engines/native/model_opt.py` declares
  `_generate_flow_action_native(..., init_noise=None, ...)`.
- The same file uses the supplied tensor directly at the current
  `init_noise` branch and only samples `_sample_flow_noise(...)` when it is
  absent.
- `thor_mxfp8` inherits the native generate path, so the same `init_noise`
  plumbing should cover BF16 and MXFP8.

The current blocker is the evaluation plumbing:

- `python/harrix/e2e_infer/adapters/variants/qwen3_5.py` explicitly rejects
  `payload["noise"]`.
- The request/preprocessor/engine projection currently does not carry the
  external tensor into `generate_flow_action(init_noise=...)`.

Please remove that rejection and plumb the validated tensor through the typed
request/prepared input to the existing native argument. Do not implement a
second denoise path or regenerate a seed on CUDA; the artifact values
themselves are the comparison contract.

## Acceptance checks

- Running one engine twice with the same artifact produces bit-identical
  `pred_concat.npy` and identical per-request noise hashes.
- BF16 and MXFP8 result metadata report the same FP32 value SHA-256 as the RPU
  result.
- A one-element perturbation in the artifact changes its hash and is observed
  by the model path, proving the value is not silently ignored.
- Invalid artifacts fail before checkpoint loading.
- Accuracy runs use a fresh process with Torch/CUDA/HWPerf profiling disabled.

The RPU side already passes the corresponding round-trip check: generated and
re-injected common-noise runs are bit-identical for `flow_noise.npy`,
`pred_relative26.npy`, and `pred_abs14.npy` (maximum absolute difference 0).
