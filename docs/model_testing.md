# Model execution and profiling

[简体中文](model_testing.zh.md) | English

RhinoForge provides two manual execution layers:

- the scripts in `examples/` are the shortest readable model entry points;
- `examples/run_model.py` runs one of those same scripts while applying a
  documented runtime environment and optional Torch and RPU hardware profiles
  from TOML.

These are board smoke and diagnostic entry points. A successful run proves
that the selected local profile executes; it does not replace a CPU/golden
numerical comparison or expand the support status in
[Model support](model_support.md).

## Entry-point inventory

| Model/profile | Direct entry | Full-runner target | Availability |
|---|---|---|---|
| Qwen3 FP16, Llama-3.2-1B | `examples/causal_lm.py` | `causal_lm` | Runnable with a matching exact release checkpoint/profile |
| Qwen3 14B W8A16 | `examples/causal_lm.py` | `causal_lm` | Source-only local-conversion entry; v1.0.0 binds no public immutable derived checkpoint identity or hash |
| Qwen3.5 text | `examples/qwen3_5_text.py` | `qwen3_5_text` | Runnable text entry |
| Qwen3.5 Vision 2B/4B | `examples/qwen3_5_vision.py` | `qwen3_5_vision` | Experimental/numeric-blocked image entry; exact `QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1` is required for controlled evaluation and is not a support claim |
| Qwen3-VL 2B/4B | `examples/qwen3_vl.py` | `qwen3_vl` | Runnable image-and-text entry |
| Qwen3-VL 8B dense FP16 | `examples/qwen3_vl.py` | `qwen3_vl` | Limited exact padded-profile entry; 4/4 numeric/task/Graph legs pass, including dual-full, with base 5/5 and long-text 9/9 tokens exact; nearby lengths, video, and quantized paths do not inherit |
| Qwen3-VL 32B W8A16 | `examples/qwen3_vl.py` | `qwen3_vl` | Source-only diagnostic entry; the explicit graph-blocked opt-in only reproduces the known all-zero logits from the third decode step |
| DINOv3 ViT-B | `examples/dinov3.py` | `dinov3` | Runnable vision entry |
| SigLIP | `examples/siglip.py` | `siglip` | Runnable component entry using the vision tower from a matching Pi0.5 bundle |
| Pi0.5 Libero FP16 | `examples/pi05.py` | `pi05` | Supported exact profile; 9/9 numeric/task and 27/27 execution-plan records pass, plus an independent fresh-process Graph lifecycle cell |
| Pi0.5 Libero W8A16 | `examples/pi05.py` | `pi05` | Source-only local-conversion entry; no public immutable derived checkpoint identity or hash is bound |
| Pi0.5 Libero W4A16 | `examples/pi05.py` | `pi05` | Source-only local-conversion entry; no public immutable derived checkpoint identity or hash is bound |
| Pi0.5 reduced-step / closed-loop | Public policy API | None | Source-only and validation pending; no public smoke claim and no inheritance from the exact FP16 or quantized paths |
| Pi0.5 base | None | None | Source-only alias; no independent task profile or smoke claim |
| Wall-OSS FP16/W8A16/W4A16 | `examples/wall_oss.py` | `wall_oss` | Public `wall-x` source-only integration |
| Wall Qwen3.5 exact flow policy | `examples/wall_qwen35.py`; `run_wall_qwen35_openloop.sh` | None | Source-only controlled-evaluation entry for one locally admitted checkpoint: FP16 batch 1, robot ID `10070`, `x2_normal`, fixed mask `[1]*20+[0]*6`, three canonical Dataset-V2 camera inputs, initial prefix `<=384`, action `[1,32,26]`, and 10 Euler steps; on-board numerical, Graph-lifecycle, and task validation remain pending |
| Hy-Embodied-0.5-VLA FP16/W16 | `examples/hy_embodied.py` | `hy_embodied` | Runnable normalized-action entry for the exact public profile |
| Hy-Embodied-0.5-VLA W8/W4 | `examples/hy_embodied.py` | `hy_embodied` | Source-only runtime-conversion entry; no public immutable derived checkpoint identity or hash is bound |
| RhinoVLA | `examples/rhinovla.py` | `rhinovla` | Integration entry; the model repository supplies its runtime factory |
| Gemma4-E4B text | `examples/gemma4.py` | `gemma4` | Source-only text entry; requires a compatible local checkpoint and dependency set |
| GR00T-N1.7-3B | `examples/gr00t.py` | `gr00t` | Source-only builder entry with caller-preprocessed tensor inputs |
| LingBot-VLA-V2 | `examples/lingbot2.py` | `lingbot2` | Public source-only integration with caller-supplied images |
| Galaxea G0.5 | `examples/g05.py` | `g05` | Configuration and integration-boundary check; execution fails closed until the model repository supplies the official CPU policy object |
| InternVLA-N1 + NavDP | `examples/internvla_navdp.py` | `internvla_navdp` | Controlled NavDP component with caller-pinned manifest and embeddings; not composite E2E |
| Qwen3 0.6B/1.7B/4B/8B W8A16, Qwen3 32B | Public API named in `model_support.md` | None | Source-only; no released runnable profile or smoke configuration |
| Unsupported profiles | None | None | Not runnable |

The table reports actual public entry points; source presence alone is not
listed as a test path.

## Checked-in profile templates

Every stem below has one `.toml` under
[`examples/configs`](../examples/configs). That file contains the direct model
request, runner environment, Torch/hardware profiling, and every accepted
execution stage. Configuration contents do not change the profile's support
status.

| Family | Configuration stems |
|---|---|
| Qwen3 | `qwen3_0_6b`, `qwen3_1_7b`, `qwen3_4b`, `qwen3_8b`, `qwen3_14b_w8a16` |
| Llama | `llama_3_2_1b` |
| Qwen3.5 text | `qwen3_5_0_8b`, `qwen3_5_2b`, `qwen3_5_4b`, `qwen3_5_9b` |
| Qwen3.5 Vision | `qwen3_5_vision_2b`, `qwen3_5_vision_4b` |
| Qwen3-VL | `qwen3_vl_2b`, `qwen3_vl_4b`, `qwen3_vl_8b`, `qwen3_vl_32b_w8a16` |
| DINOv3 / SigLIP | `dinov3_vit_b`, `siglip` |
| Pi0.5 Libero | `pi05_libero`, `pi05_libero_w8a16`, `pi05_libero_w4` |
| Wall-OSS | `wall_oss`, `wall_oss_w8a16`, `wall_oss_w4a16` |
| Hy-Embodied / RhinoVLA | `hy_embodied`, `rhinovla` |
| Gemma4 / GR00T | `gemma4`, `gr00t` |
| LingBot2 / G0.5 | `lingbot2`, `g05` |
| InternVLA-N1 + NavDP | `internvla_navdp` |

Quantized templates exercise local converter/runtime paths only. Their presence
does not bind a derived checkpoint identity or change a Source-only status.

## Direct examples

Every runner-backed direct example accepts a TOML file and can validate it
without loading a checkpoint:

```bash
python examples/qwen3_vl.py --config examples/configs/qwen3_vl_2b.toml --check-config
python examples/qwen3_vl.py --config qwen3_vl.local.toml
```

Copy the closest file under `examples/configs/`, then set local checkpoint and
input paths. See [Model assets](model_assets.md) for checkpoint acquisition and
hashing.

The generic vision and VLM templates use `assets/example.ppm`, a tiny synthetic
image included only so the public entry points run without a private input.
Replace it with representative images before numerical or task validation.
VLA camera inputs remain caller supplied because their names and preprocessing
belong to the exact model profile.

The Wall Qwen3.5 entry is a standalone exact-checkpoint CLI rather than a TOML
runner target. Supply exactly the three canonical cameras and a 26-value state:

```bash
python examples/wall_qwen35.py \
  --checkpoint /path/to/exact/checkpoint \
  --image face_view=/path/to/face.png \
  --image left_wrist_view=/path/to/left.png \
  --image right_wrist_view=/path/to/right.png \
  --instruction "perform the requested action" \
  --state-json /path/to/state-26.json \
  --allow-numeric-blocked-vision
```

The flag is an explicit controlled-evaluation opt-in for the numeric-blocked
Qwen3.5 vision path; it does not promote this Source-only profile. State and
degree-of-freedom masks, when supplied, must both equal the exact 26-value
`[1]*20+[0]*6` profile. Adding
`--check-config` validates supplied command-line paths and input shapes; with
no request arguments it performs the repository's dependency-free CLI probe.
Neither mode performs checkpoint admission, RPU execution, numerical parity,
Graph lifecycle, or task-quality validation.

For the pinned put-spoon-to-bowl episode, the repository-root wrapper mirrors
the Harrix data, prompt, segmentation, and action-decoding contract and never
sends a robot command:

```bash
bash run_wall_qwen35_openloop.sh --check --no-sudo
bash run_wall_qwen35_openloop.sh --max-events 1
bash run_wall_qwen35_openloop.sh --max-events 1 \
  --torch-profile-dir /tmp/qwen35-openloop-profile
bash run_wall_qwen35_openloop.sh --max-events 1 \
  --hw-perf-dir /tmp/qwen35-openloop-hwperf --hw-perf-max-dumps 32
bash run_wall_qwen35_openloop.sh
bash run_wall_qwen35_openloop.sh \
  --flow-noise /path/to/common_flow_noise.npy
```

The local wrapper defaults to the inference checkpoint at
`/mnt/nvme/miyaa/work/ckpt/0_200000`, the recorded episode under
`/mnt/nvme/miyaa/work/dataset`, and a fresh output directory under
`/mnt/nvme/miyaa/work/prof`. These defaults do not require the `/mnt/miyaa`
remote mount. Copy the checkpoint weights, configs, tokenizer/preprocessor,
normalizers, and complete episode (trajectory, captions, and three videos).
Training optimizer and resume state are not inference dependencies. Keep
checkpoint configs byte-identical: admission checks their hashes; training
paths recorded inside `config.yml` are not loaded by this runner. The existing
local Python environment and installed RPU runtime assets are still used.
Explicit CLI arguments and environment overrides retain precedence.

The wrapper defaults to **one Vision Graph for the three images**; no opt-in
switch beyond the wrapper's existing controlled-evaluation mode is needed.
It uses the installed `rpu_backend`, not the source-tree Python package. After
changing the adapter or native code, use the same Python environment to rebuild:

```bash
source /home/hx/miyaa/work/env.sh
CMAKE_PREFIX_PATH=/home/hx/.local/opt/rhino-launch-kernel-v1.0.0-linux-aarch64 \
  python -m pip install . --no-build-isolation -Cbuild.tool-args=-j2
bash run_wall_qwen35_openloop.sh --max-requests 1
```

Real execution rejects an old packed-Vision Python/native ABI before checkpoint
loading and prints the loaded adapter path. `Vision Graph: entries=1` confirms
the retained cache for the usual image triple. A fresh `--max-requests 1
--torch-profile-dir DIR` run records that request including setup/BUILD; it
does not insert a warmup/repeat or certify whole-policy READY/parity.
Graph details are in `segments.json`, and
`meta.json` records the package/native identities and `vision_execution` mode.

The unified cold switch is `WALL_QWEN35_OPT` (default `1`). No per-stage CLI
flags are needed:

```bash
# Vision 1 + Language/Prefill 1 + Action 1
bash run_wall_qwen35_openloop.sh --max-requests 1 \
  --torch-profile-dir /tmp/qwen35-opt-profile

# Vision 3 + Language/Prefill 3 + Action 10 on this recorded episode
WALL_QWEN35_OPT=0 bash run_wall_qwen35_openloop.sh --max-requests 1 \
  --torch-profile-dir /tmp/qwen35-split-profile
```

Optimized Action maps real prefixes to fixed 64-row buckets (`64..384`),
retains up to six one-segment graphs, and refreshes padding visibility and
real RoPE outside capture. The runner checks cumulative replay counts and
records `action_prefix_bucket` per request. For performance tests, omit both
profiling-directory flags and reuse the same `--flow-noise` artifact. Validate
same-bucket length changes and A/B/A returns separately from cold startup;
more cached buckets do not mean more graph submissions per request.
See the [Action bucket validation receipt](wall_qwen35_action_bucket_validation.md)
for the bounded local lifecycle, numerical-difference and unprofiled timings;
it does not promote the model's Source-only status.

Optimized Prefill also uses the whole prefix bucket as one outer compute chunk:
the recorded episode uses 320 rows instead of `128 + 128 + 64`. This is distinct
from Graph segmentation. Gated DeltaNet keeps its internal 64-row recurrence.
For chunks above128, mutually exclusive Full Attention/GDN scratch lifetimes
are separated; persistent buffers and the hard co-resident SPM limit are unchanged.
One outer chunk lets the existing planner keep inter-layer hidden states in SPM.
The Wall-only cold override fails on insufficient memory rather than silently
splitting; generic Qwen3.5 and `WALL_QWEN35_OPT=0` keep their128-row ceiling.
Check the Prefill signature's chunk dimension against the bucket, finite consumed
KV/state outputs, and cross-bucket A/B/A replay when validating this path. Different
FP16 GEMM/SDPA/reduction geometry is not a bitwise-equivalence guarantee.

Both modes use FP16 Action projections/Euler with ACC32 GEMMs and one-time CPU
FP32 time/Ada precomputation. Disabling optimization changes Graph organization,
not precision; it does not restore the historical FP32 host path. The former
`--language-one-graph` and `--action-execution` options are removed.

Both modes include the [GDN padding safety fix](qwen35_gdn_padding_safety.md).
Old fixed-grid fill traces/predictions can contain overwritten live Q/K data;
regenerate a repaired 128-chunk reference before comparing chunk-merge numerics.

The switch overrides all six Graph/SDK budget environment values: enabled uses
32768 entries / 8 MiB command / 64 MiB instruction with SDK capacities
65536 / 16 MiB / 128 MiB; disabled uses 8192 / 4 / 32 with SDK 65536 / 8 / 64.
Generic non-Wall defaults are unchanged. Use a fresh process to change modes.

Count physical segments, not cache entries. Optimized Vision, Prefill and Action
require one physical segment per Graph; the split Action reuses one single-step
Graph ten times, and split Vision retains separate camera shapes. The historical
READY probe checked 3 versus 16 physical submissions, stable replay and
same-input output parity, excluding cold priming/BUILD. The current per-request
profiler includes cold work and does not perform that probe. Other input envelopes can produce
different conservative Prefill segment counts; the `3+3+10` check is for this
runner's recorded episode. Metadata records the switch, plan and effective budgets.
Stale Python/native packages fail before loading. Compare outputs with the same
`--flow-noise`; these controlled paths are not release-quality certifications.

The encoder retains per-image AllReduce boundaries inside the single Graph:
GEMMs are packed, each layer has three isolated attention calls, and its two
residual reductions each use three image spans. Changing ring geometry by
reducing all packed rows together can change FP16 rounding even with identical
partials. Generic single-image and temporal execution remain unchanged.
The Wall runtime also resets temporary SPM outside capture before Vision/Text
and after Text, preserving persistent state while preventing a previous Action
priming call's workspace from accumulating under the next packed Vision call.

The live commands use `sudo` by default for board access and source
`/home/hx/miyaa/work/env.sh`. Their output includes reference-compatible NumPy
filenames, physical-unit metrics, exact-profile provenance, and retained-Graph
diagnostics. Before Python starts, the wrapper fixes the cold multi-handle
setting `RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN=1`. Every run exports the exact
`flow_noise.npy` it consumed. The default deterministic CPU seed schedule is
not bit-identical to the reference CUDA RNG stream, so device precision claims
must use `--flow-noise` with the same `[requests,32,26]` artifact on every
backend.

The checked-in 2026-09-01 accuracy report is reproduced by a fresh, unprofiled
full run with the captured Harrix BF16 CUDA noise:

```bash
bash run_wall_qwen35_openloop.sh \
  --flow-noise /mnt/nvme/miyaa/work/prof/harrix_qwen35_bf16_flow_noise_20260901.npy \
  --output-dir /mnt/nvme/miyaa/work/prof/wall_qwen35_accuracy_report_repro
sha256sum /mnt/nvme/miyaa/work/prof/wall_qwen35_accuracy_report_repro/{flow_noise,pred_concat}.npy
```

The expected hashes are `6b01140934037843ef2364a5da5dd864bf0ee7f27df3820ed1ff370bf866f5d3`
for `flow_noise.npy` and
`5fb358fbe7ea148b0e7374137fb4ad6cd44f77c9a228c8ab71d3dc706a4b6e3b`
for `pred_concat.npy`. The output directory must not already exist. The full
same-noise BF16/MXFP8/RPU evidence and FP64 aggregate metrics are in
`reports/wall_qwen35_accuracy_comparison_20260901.html` and its sibling JSON.

Torch profiling has one entry point: `--torch-profile-dir DIR`. It requests
CPU and `PrivateUse1` activities and exports each post-install inference request
as a separate `qwen35_generate_flow_action_batch_*.trace.json.gz`. Filenames
include the timestamp, PID, request index and nanosecond timestamp. Shapes and
source stacks are always enabled; memory events are disabled. A fresh profiler
is used for each request with `acc_events=True`, without accumulating previous
requests' events. The reference `generate_flow_action_batch` range and the
`wall_qwen35_preprocess`, `wall_qwen35_vision_text_prefill`,
`wall_qwen35_action_denoise_loop`, and `wall_qwen35_action_decoder` stage ranges
remain visible.

There is no automatic unprofiled warmup or identical-repeat READY probe.
Explicit `policy.to("rpu")` installation is outside the Torch trace, but lazy
first-request setup and Graph BUILD are included. Later requests may BUILD
when their signature changes. A trace is diagnostic evidence, not a frozen
READY or numerical-parity claim; `meta.json` keeps the per-request artifact
manifest and does not report READY admission. Historical exact-repeat
measurements remain in the
[Wall Qwen3.5 READY-profile assessment](wall_qwen35_ready_profile_assessment.md).
The old `--torch-profile`, `--torch-profile-output`,
`--torch-profile-record-shapes`, `--torch-profile-memory`, and
`--torch-profile-with-stack` flags are removed from this Wall runner.

`--hw-perf-dir DIR` is the sole hardware-trace enable/destination option and
enables r4 device tracing before policy installation. The former `--hw-perf`
and `--hw-perf-output` options are removed. The wrapper passes
`LKN_RPU_FREQ_MHZ` through sudo (default 800 MHz) and bounds the number of
segment JSON files with `--hw-perf-max-dumps` (default 32). Filenames include a
timestamp, PID, BUILD/REPLAY/oneshot phase, and segment index. Use the
`*_replay_segN.json` files in Perfetto and inspect every segment belonging to
the inference region. r4 Release traces are intentionally redacted.
The timestamp is the session start in the process's local timezone, shared by
all dumps; it is not each file's write time. Earlier native builds used UTC
(eight hours behind Asia/Shanghai). Rebuild/reinstall the native extension to
use local names. Existing filenames and JSON event timing remain unchanged.

Wall's Graph labels are consistently `rpu_wall_qwen35_vision`,
`rpu_wall_qwen35_prefill` and `rpu_wall_qwen35_action` in both OPT modes. They
appear in hardware filenames and Graph capture ranges/BUILD/REPLAY logs;
`fp16_one_graph` versus `fp16_steps` remains in execution metadata, not the
stage name. Generic Qwen3.5 keeps its `qwen3_5_vision` label. This is a Python
adapter naming change: install the matching Python sources and start a fresh
process; the native ABI, compute and Graph topology are unchanged. Existing
trace files and their recorded provenance are not renamed.

Both directory options can be used together, or independently. Omitting a
directory leaves that profiler disabled; the wrapper also accepts
`TORCH_PROFILE_DIR` and `HW_PERF_DIR` environment equivalents. No other
profiling enable/output/shapes/memory/stack environment overrides are read.
Destinations are validated before checkpoint loading or RPU initialization,
and existing directories can be reused without overwriting trace files.
Neither profiler can be combined with `--check`. Keep these sensitive traces
outside the repository: they can contain application shapes, source paths and
operation metadata. Both profilers perturb latency; rerun without profiling
for final latency measurements.

The Pi0.5 `batch_file` is not bundled. Produce it with the checkpoint-compatible
LeRobot policy preprocessing pipeline, then save the tensor dictionary with
`torch.save`. At minimum it contains one batched image tensor for every key in
the checkpoint's `image_features` configuration, plus
`observation.language.tokens` and `observation.language.attention_mask`; the
preprocessing pipeline may retain other profile-owned fields. Camera names,
resolution, and token length are profile-owned; take them from the
checkpoint/configuration rather than copying another profile's shapes.

The GR00T entry similarly expects a caller-produced tensor mapping. Its
`input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, and `state`
must come from the checkpoint-compatible preprocessing pipeline. The NavDP
component entry requires a caller-pinned path-to-SHA256 JSON manifest plus a
tensor mapping containing `goal_embed [1,1,384]` and
`rgbd_embed [1,32,384]`. These files are inputs, not model assets distributed
by RhinoForge.

Qwen3.5 Vision rejects execution by default before loading model weights in the
public example and before creating its Vision handle or transforming Vision
weights in the shared adapter. For controlled evaluation only, explicitly set
the gate before starting a fresh process:

```bash
QWEN3_5_VISION_ALLOW_NUMERIC_BLOCKED=1 \
  python examples/run_model.py --config examples/configs/qwen3_5_vision_2b.toml
```

This does not waive the failed official real-image numerical gate or enable
video. The Wall Qwen3.5 CLI has the same fail-closed controlled-evaluation
principle through its dedicated flag; it does not use the full runner or claim
that its pending board gates pass. The Qwen3-VL 32B template retains an explicit
opt-in only to reproduce its known hard Graph failure; it is not a passing
hard-gate or runnable-support claim. LingBot2 and InternVLA/NavDP templates
contain explicit source-only or controlled-evaluation acknowledgements.
Removing a gate must fail
rather than fall back to an ordinary profile. G0.5
has no stable standalone public policy constructor: its example validates the
configuration and then stops with the exact integration step for the official
model repository. A successful `--check-config` for any of these entries
validates configuration structure only; it does not promote the support state.

## Complete TOML runner

Every runnable target has one TOML under `examples/configs/`. The same file
contains model/request fields, common runtime settings, Torch profile, hardware
profile, and every applicable `rpu_execution` stage. The complete field and
ownership map is in
[Runtime configuration](runtime_config.md#toml-parameter-catalog).

For example, start from `examples/configs/qwen3_0_6b.toml`:

```bash
python examples/run_model.py --list-targets
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml \
  --check-config
python examples/run_model.py \
  --config examples/configs/qwen3_0_6b.toml
```

The configuration combines runtime environment, per-handle execution planning,
model input, and Torch profiling:

```toml
[runner]
target = "causal_lm"

[runner.env]
RPU_LOG_LEVEL = 3
RPU_WARMUP = 1

[runner.torch_profile]
enabled = false
output = "profiles/torch.json"
record_shapes = false
profile_memory = false
with_stack = false

[runner.hw_perf]
enabled = false
output_dir = "profiles/rpu_hwperf"
max_dumps = 32
```

The remaining `[model]`, `[generation]`, or `[request]` tables are consumed by
the selected direct example. Only variables documented in
[Runtime configuration](runtime_config.md) are accepted under `[runner.env]`.
Boolean TOML values become portable `1`/`0` environment values. The runner
applies them before importing `torch` or `rpu_backend`; run one model/profile
per process. Credential variables, unrelated process-loader variables, and
`RPU_KERNEL_LIB_PATH` are rejected from TOML; set the approved Rhino Launch
library and combined operator asset in the deployment environment.
Variables under the Runtime configuration page's **Diagnostic-only inventory**
are also rejected; a developer must opt into one explicitly in the shell and
review any resulting local artifacts before sharing them.

Per-handle execution planning belongs in the top-level table:

```toml
[rpu_execution.prefill]
chunk_size = "auto"
padding_budget = 64
```

The comprehensive runner validates and forwards this table to the existing
model entry point. The accepted stages are:

| Target | Accepted `rpu_execution` stages |
|---|---|
| `causal_lm`, `qwen3_5_text`, `qwen3_5_vision` | `prefill` |
| `gemma4` | `prefill` (`chunk_size` only) |
| `qwen3_vl` | `prefill`, `vision` |
| `dinov3`, `siglip` | `vision` |
| `pi05`, `wall_oss`, `gr00t` | `prefill`, `vision`, `action` |
| `rhinovla` | Declared by the trusted runtime factory and checked by its capability handshake |
| `hy_embodied`, `lingbot2`, `g05`, `internvla_navdp` | None; a non-empty table is rejected |

Except for Gemma4's narrower row, `prefill` accepts `chunk_size`,
`padding_rows`, and `padding_budget`; `vision` and `action` accept
`chunk_size`. The public loader remains the authoritative profile-envelope
check and rejects unsupported values before RPU installation.

## Torch profile

Set `[runner.torch_profile].enabled = true` to export one Chrome trace for the
whole command. CPU and `PrivateUse1` activities are requested when the
installed PyTorch exposes both. The trace includes model initialization as well
as inference; filter for `rpu::` and named graph ranges when investigating the
device path.

`record_shapes`, `profile_memory`, and `with_stack` all default to `false`.
Enable them individually only when needed: shapes, allocation events, stack
frames, and source paths add application detail to the trace.

Open the JSON in Perfetto or another Chrome-trace viewer. Profiling adds
overhead, so use a profiler-disabled run for latency measurement.

## RPU hardware profile

Set `[runner.hw_perf].enabled = true` only with an r4 Rhino Launch build that
provides the hardware trace API. The output directory may already exist; each
session writes non-overwriting `rpu_hwperf_*.json` filenames. Configuration
validation accepts this table without initializing the device. At execution,
the runner fails clearly if the installed RhinoForge extension predates the
integration. The context is entered before the model target and resets live
Graph batches both when enabling and disabling collection.

## Debugging an incorrect or unstable run

Debug correctness before collecting performance traces. Keep one immutable
run record containing the checkpoint/configuration hashes, TOML, input tensors,
execution settings, and dependency versions. For a stochastic policy, sample
the initial noise or other random state once and clone that exact tensor into
the CPU and RPU paths; two successive RNG calls are different inputs.

Use the smallest public checks that distinguish the failure:

1. Compare the first call with a repeated call of the same signature. A
   repeat-only change points first to Graph lifecycle, persistent SPM, or DMA
   address ownership.
2. Retain the first output, run a different input of the same shape, and verify
   that the retained bytes and storage remain independent.
3. Inspect `GraphCache.snapshot()`, `cache_invariant_ok()`,
   `debug_bucket_counts()`, and `explain_miss(signature)` before changing the
   capture path. A clean log alone does not prove BUILD followed by REPLAY.
4. Isolate the first divergent public component or operator and compare its
   shapes, dtypes, layouts, and value against the same CPU input before
   debugging the full model.
5. Increase `torch.rpu.set_debug_level(...)` only as needed. Use
   `torch.rpu.spm_alloc_dump("label")` for allocator summaries and
   debug-tensor export only for a controlled local comparison.

Debug exports can contain weights, user-derived inputs, and activations. Keep
them outside the repository, clear them after the investigation, and share a
sanitized summary rather than the raw tensors. Once correctness and replay are
stable, rerun in a fresh process with only the profiler needed for the measured
question.

## What a release-quality model check still needs

For a support claim, bind all results to the same source commit, Rhino Launch
package, combined operator asset, checkpoint revision, input envelope, TOML,
and output hashes. Compare against a CPU or approved golden result, exercise
prefill and decode or the complete policy call, repeat after warmup, and verify
the applicable graph lifecycle. Record any unrun gate as pending rather than
inferring it from a smoke run. Use the independent hard semantic, same-dtype
parity, FP32-anchor, and representative task gates in
[Model validation policy](validation_policy.md); a single worst-row cosine is
diagnostic evidence, not a repository-wide product verdict.
