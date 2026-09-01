#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ENV_SH="${ENV_SH:-/home/hx/miyaa/work/env.sh}"
DATASET_DIR="${DATASET_DIR:-/mnt/miyaa/work/dataset/20260122-day-put_spoon_to_bowl@MASTER_SLAVE_MODE@2026_01_22_18_19_18}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-/mnt/miyaa/work/ckpt/0_200000}"
RPU_KERNEL_LIB_PATH="${RPU_KERNEL_LIB_PATH:-/home/hx/.local/share/rhinoforge/runtime/runtime-v1.0.0-r4/rhinoOpLib_rhinoforge_v1.0.0.ref}"
RHINO_LAUNCH_LIB_DIR="${RHINO_LAUNCH_LIB_DIR:-/home/hx/.local/opt/rhino-launch-kernel-v1.0.0-linux-aarch64/lib}"
INSTRUCTION_SOURCE="${INSTRUCTION_SOURCE:-distribute}"
ROBOT_ID="${ROBOT_ID:-10070}"
NORM_KEY="${NORM_KEY:-x2_normal}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-10}"
MAX_REQUESTS="${MAX_REQUESTS:-${MAX_EVENTS:-0}}"
NOISE_SEED="${NOISE_SEED:-${SEED:-3407}}"
FLOW_NOISE="${FLOW_NOISE:-}"
RUN_WITH_SUDO="${RUN_WITH_SUDO:-1}"
TORCH_PROFILE="${TORCH_PROFILE:-0}"
TORCH_PROFILE_OUTPUT="${TORCH_PROFILE_OUTPUT:-}"
TORCH_PROFILE_DIR="${TORCH_PROFILE_DIR:-}"
TORCH_PROFILE_RECORD_SHAPES="${TORCH_PROFILE_RECORD_SHAPES:-0}"
TORCH_PROFILE_MEMORY="${TORCH_PROFILE_MEMORY:-0}"
TORCH_PROFILE_WITH_STACK="${TORCH_PROFILE_WITH_STACK:-0}"
HW_PERF="${HW_PERF:-0}"
HW_PERF_OUTPUT="${HW_PERF_OUTPUT:-}"
HW_PERF_MAX_DUMPS="${HW_PERF_MAX_DUMPS:-32}"
LKN_RPU_FREQ_MHZ="${LKN_RPU_FREQ_MHZ:-800}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$$}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/wall_qwen35_openloop_${RUN_ID}}"

usage() {
    cat <<'EOF'
Usage:
  bash run_wall_qwen35_openloop.sh [OPTIONS]

Run the exact Wall Qwen3.5 checkpoint on the recorded open-loop episode. The
script mirrors the checkpoint-matched Harrix reference: it splits annotated
events into 32-step chunks, never crosses an event boundary, samples the three
camera frames at each chunk start, and compares decoded RPU predictions with
the recorded master-arm absolute trajectory. It never commands a robot.

Options:
      --dataset-dir PATH       Recorded flat episode directory.
      --checkpoint PATH        Exact Wall Qwen3.5 checkpoint directory.
      --output-dir PATH        Fresh output directory (default: /tmp/...RUN_ID).
      --instruction-source KEY Event captions in instruction.json
                               (default: distribute).
      --robot-id ID            Dataset-V2 embodiment ID (default: 10070).
      --norm-key KEY           Checkpoint normalizer key (fixed: x2_normal).
      --inference-steps N      Flow Euler steps (fixed: 10).
      --max-requests N         Run at most N chunks; 0 runs all (default: 0).
      --max-events N           Reference-compatible alias for --max-requests.
      --noise-seed N           Base for the deterministic per-request seed
                               schedule (base + request index; default: 3407).
      --flow-noise PATH        Common FP32 .npy noise [requests,32,26]; bypasses
                               --noise-seed and is copied into the run output.
      --torch-profile          Warm the first request without recording, then
                               profile exactly one identical repeat with CPU +
                               PrivateUse1 activities and Graph admission.
      --torch-profile-output D READY-probe output directory; implies
                               --torch-profile (default: OUTPUT_DIR). Trace and
                               summary filenames include a timestamp and PID.
      --torch-profile-dir DIR  Reference-compatible mode: profile every action
                               request into a separate *.trace.json.gz file.
      --torch-profile-record-shapes
                               Include tensor shapes in the trace.
      --torch-profile-memory   Include Torch memory events in the trace.
      --torch-profile-with-stack
                               Include source stacks in the trace.
      --hw-perf                Collect r4 hardware kernel/DMA Chrome traces.
      --hw-perf-output DIR     Hardware trace directory; implies --hw-perf
                               (default: OUTPUT_DIR/rpu_hwperf). Filenames carry
                               timestamp, PID, phase, and segment index.
      --hw-perf-max-dumps N    Maximum segment trace files (default: 32).
      --check                  Validate checkpoint, data, videos, mappings, and
                               every real prompt prefix without RPU execution.
      --dry-run                Alias for --check.
      --sudo                   Run through sudo (default).
      --no-sudo                Run as the current user.
  -h, --help                   Show this help.

Environment overrides:
  ENV_SH, DATASET_DIR, CHECKPOINT_PATH, OUTPUT_DIR, INSTRUCTION_SOURCE, ROBOT_ID,
  NORM_KEY, NUM_INFERENCE_STEPS, MAX_REQUESTS/MAX_EVENTS, NOISE_SEED/SEED,
  FLOW_NOISE,
  RUN_WITH_SUDO, TORCH_PROFILE, TORCH_PROFILE_OUTPUT, TORCH_PROFILE_DIR,
  TORCH_PROFILE_RECORD_SHAPES, TORCH_PROFILE_MEMORY, TORCH_PROFILE_WITH_STACK,
  HW_PERF, HW_PERF_OUTPUT, HW_PERF_MAX_DUMPS, LKN_RPU_FREQ_MHZ,
  RPU_KERNEL_LIB_PATH, RHINO_LAUNCH_LIB_DIR, RUN_ID.

Examples:
  bash run_wall_qwen35_openloop.sh --check
  bash run_wall_qwen35_openloop.sh --max-requests 1
  bash run_wall_qwen35_openloop.sh --max-requests 1 --torch-profile
  bash run_wall_qwen35_openloop.sh --max-requests 1 \
    --hw-perf-output /home/hx/miyaa/work/prof/wall_qwen35_hwperf
  bash run_wall_qwen35_openloop.sh --max-events 1 \
    --torch-profile-dir /tmp/wall_qwen35-openloop-profile
  bash run_wall_qwen35_openloop.sh \
    --flow-noise /mnt/miyaa/work/prof/harrix_qwen35_bf16_flow_noise_20260901.npy \
    --output-dir /tmp/wall_qwen35_accuracy_report_repro
  bash run_wall_qwen35_openloop.sh
EOF
}

fatal() {
    printf '[wall-qwen35-openloop] FATAL: %s\n' "$*" >&2
    exit 2
}

require_value() {
    [[ -n "${2:-}" ]] || fatal "$1 requires a value"
}

normalize_bool() {
    local output_name="$1"
    local setting_name="$2"
    local value="$3"
    case "${value,,}" in
        1|true|yes|on) printf -v "$output_name" '%s' 1 ;;
        0|false|no|off) printf -v "$output_name" '%s' 0 ;;
        *) fatal "$setting_name must be boolean: $value" ;;
    esac
}

check_only=0
while (($#)); do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --dataset-dir)
            require_value "$1" "${2:-}"
            DATASET_DIR="$2"
            shift 2
            ;;
        --dataset-dir=*)
            DATASET_DIR="${1#*=}"
            shift
            ;;
        --checkpoint)
            require_value "$1" "${2:-}"
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --checkpoint=*)
            CHECKPOINT_PATH="${1#*=}"
            shift
            ;;
        --output-dir)
            require_value "$1" "${2:-}"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --instruction-source)
            require_value "$1" "${2:-}"
            INSTRUCTION_SOURCE="$2"
            shift 2
            ;;
        --instruction-source=*)
            INSTRUCTION_SOURCE="${1#*=}"
            shift
            ;;
        --robot-id)
            require_value "$1" "${2:-}"
            ROBOT_ID="$2"
            shift 2
            ;;
        --robot-id=*)
            ROBOT_ID="${1#*=}"
            shift
            ;;
        --norm-key)
            require_value "$1" "${2:-}"
            NORM_KEY="$2"
            shift 2
            ;;
        --norm-key=*)
            NORM_KEY="${1#*=}"
            shift
            ;;
        --inference-steps)
            require_value "$1" "${2:-}"
            NUM_INFERENCE_STEPS="$2"
            shift 2
            ;;
        --inference-steps=*)
            NUM_INFERENCE_STEPS="${1#*=}"
            shift
            ;;
        --max-requests|--max-events)
            require_value "$1" "${2:-}"
            MAX_REQUESTS="$2"
            shift 2
            ;;
        --max-requests=*|--max-events=*)
            MAX_REQUESTS="${1#*=}"
            shift
            ;;
        --noise-seed)
            require_value "$1" "${2:-}"
            NOISE_SEED="$2"
            shift 2
            ;;
        --noise-seed=*)
            NOISE_SEED="${1#*=}"
            shift
            ;;
        --flow-noise)
            require_value "$1" "${2:-}"
            FLOW_NOISE="$2"
            shift 2
            ;;
        --flow-noise=*)
            FLOW_NOISE="${1#*=}"
            [[ -n "$FLOW_NOISE" ]] || fatal "--flow-noise requires a value"
            shift
            ;;
        --torch-profile)
            TORCH_PROFILE=1
            shift
            ;;
        --torch-profile-output)
            require_value "$1" "${2:-}"
            TORCH_PROFILE_OUTPUT="$2"
            TORCH_PROFILE=1
            shift 2
            ;;
        --torch-profile-output=*)
            TORCH_PROFILE_OUTPUT="${1#*=}"
            [[ -n "$TORCH_PROFILE_OUTPUT" ]] \
                || fatal "--torch-profile-output requires a value"
            TORCH_PROFILE=1
            shift
            ;;
        --torch-profile-dir)
            require_value "$1" "${2:-}"
            TORCH_PROFILE_DIR="$2"
            shift 2
            ;;
        --torch-profile-dir=*)
            TORCH_PROFILE_DIR="${1#*=}"
            [[ -n "$TORCH_PROFILE_DIR" ]] \
                || fatal "--torch-profile-dir requires a value"
            shift
            ;;
        --torch-profile-record-shapes)
            TORCH_PROFILE_RECORD_SHAPES=1
            shift
            ;;
        --torch-profile-memory)
            TORCH_PROFILE_MEMORY=1
            shift
            ;;
        --torch-profile-with-stack)
            TORCH_PROFILE_WITH_STACK=1
            shift
            ;;
        --hw-perf)
            HW_PERF=1
            shift
            ;;
        --hw-perf-output)
            require_value "$1" "${2:-}"
            HW_PERF_OUTPUT="$2"
            HW_PERF=1
            shift 2
            ;;
        --hw-perf-output=*)
            HW_PERF_OUTPUT="${1#*=}"
            [[ -n "$HW_PERF_OUTPUT" ]] \
                || fatal "--hw-perf-output requires a value"
            HW_PERF=1
            shift
            ;;
        --hw-perf-max-dumps)
            require_value "$1" "${2:-}"
            HW_PERF_MAX_DUMPS="$2"
            shift 2
            ;;
        --hw-perf-max-dumps=*)
            HW_PERF_MAX_DUMPS="${1#*=}"
            shift
            ;;
        --check|--dry-run)
            check_only=1
            shift
            ;;
        --sudo)
            RUN_WITH_SUDO=1
            shift
            ;;
        --no-sudo)
            RUN_WITH_SUDO=0
            shift
            ;;
        --*) fatal "unknown option: $1" ;;
        *) fatal "unexpected positional argument: $1" ;;
    esac
done

[[ "$MAX_REQUESTS" =~ ^[0-9]+$ ]] \
    || fatal "--max-requests must be a non-negative integer: $MAX_REQUESTS"
[[ "$NOISE_SEED" =~ ^[0-9]+$ ]] \
    || fatal "--noise-seed must be a non-negative integer: $NOISE_SEED"
[[ "$ROBOT_ID" =~ ^[0-9]+$ ]] \
    || fatal "--robot-id must contain only digits: $ROBOT_ID"
[[ "$HW_PERF_MAX_DUMPS" =~ ^[1-9][0-9]*$ ]] \
    || fatal "--hw-perf-max-dumps must be a positive integer: $HW_PERF_MAX_DUMPS"
[[ "$LKN_RPU_FREQ_MHZ" =~ ^[1-9][0-9]*([.][0-9]+)?$ ]] \
    || fatal "LKN_RPU_FREQ_MHZ must be a positive MHz value: $LKN_RPU_FREQ_MHZ"
[[ "$NORM_KEY" == x2_normal ]] \
    || fatal "--norm-key is fixed to x2_normal for this open-loop profile: $NORM_KEY"
[[ "$NUM_INFERENCE_STEPS" == 10 ]] \
    || fatal "--inference-steps is fixed to 10 for this profile: $NUM_INFERENCE_STEPS"
normalize_bool use_sudo RUN_WITH_SUDO "$RUN_WITH_SUDO"
normalize_bool profile_enabled TORCH_PROFILE "$TORCH_PROFILE"
normalize_bool profile_record_shapes \
    TORCH_PROFILE_RECORD_SHAPES "$TORCH_PROFILE_RECORD_SHAPES"
normalize_bool profile_memory TORCH_PROFILE_MEMORY "$TORCH_PROFILE_MEMORY"
normalize_bool profile_with_stack \
    TORCH_PROFILE_WITH_STACK "$TORCH_PROFILE_WITH_STACK"
normalize_bool hw_perf_enabled HW_PERF "$HW_PERF"
[[ -z "$HW_PERF_OUTPUT" ]] || hw_perf_enabled=1
[[ -z "$TORCH_PROFILE_OUTPUT" ]] || profile_enabled=1
if [[ -n "$TORCH_PROFILE_DIR" ]] && ((profile_enabled)); then
    fatal "choose either --torch-profile/--torch-profile-output or --torch-profile-dir"
fi
if [[ -n "$TORCH_PROFILE_DIR" ]]; then
    profile_mode=per-request
    profile_enabled=1
    # Match the Harrix per-request profiler contract.
    profile_record_shapes=1
    profile_with_stack=1
elif ((profile_enabled || profile_record_shapes || profile_memory || profile_with_stack)); then
    profile_mode=ready-replay
    profile_enabled=1
else
    profile_mode=disabled
fi
if ((check_only && profile_enabled)); then
    fatal "Torch profiling requires real execution; remove --check/--dry-run"
fi
if ((check_only && hw_perf_enabled)); then
    fatal "hardware profiling requires real execution; remove --check/--dry-run"
fi

[[ -f "$ENV_SH" ]] || fatal "environment script is missing: $ENV_SH"
# shellcheck disable=SC1090
source "$ENV_SH"
[[ -n "${CONDA_PREFIX:-}" ]] || fatal "ENV_SH did not set CONDA_PREFIX"
PYTHON_BIN="${PYTHON_BIN:-$CONDA_PREFIX/bin/python}"
[[ -x "$PYTHON_BIN" ]] || fatal "python is not executable: $PYTHON_BIN"
[[ -d "$DATASET_DIR" ]] || fatal "dataset directory is missing: $DATASET_DIR"
[[ -d "$CHECKPOINT_PATH" ]] || fatal "checkpoint directory is missing: $CHECKPOINT_PATH"
[[ -f "$RPU_KERNEL_LIB_PATH" ]] \
    || fatal "operator reference is missing: $RPU_KERNEL_LIB_PATH"
[[ -d "$RHINO_LAUNCH_LIB_DIR" ]] \
    || fatal "Rhino Launch library directory is missing: $RHINO_LAUNCH_LIB_DIR"

DATASET_DIR="$(cd -- "$DATASET_DIR" && pwd -P)"
CHECKPOINT_PATH="$(cd -- "$CHECKPOINT_PATH" && pwd -P)"
if [[ -n "$FLOW_NOISE" ]]; then
    [[ -f "$FLOW_NOISE" ]] || fatal "flow noise file is missing: $FLOW_NOISE"
    FLOW_NOISE="$(cd -- "$(dirname -- "$FLOW_NOISE")" && pwd -P)/$(basename -- "$FLOW_NOISE")"
fi
if [[ "$OUTPUT_DIR" != /* ]]; then
    OUTPUT_DIR="$(pwd -P)/$OUTPUT_DIR"
fi
if [[ "$profile_mode" == ready-replay ]]; then
    if [[ -z "$TORCH_PROFILE_OUTPUT" ]]; then
        TORCH_PROFILE_OUTPUT="$OUTPUT_DIR"
    elif [[ "$TORCH_PROFILE_OUTPUT" != /* ]]; then
        TORCH_PROFILE_OUTPUT="$(pwd -P)/$TORCH_PROFILE_OUTPUT"
    fi
    [[ ! -e "$TORCH_PROFILE_OUTPUT" || -d "$TORCH_PROFILE_OUTPUT" ]] \
        || fatal "torch profile output is not a directory: $TORCH_PROFILE_OUTPUT"
elif [[ "$profile_mode" == per-request ]]; then
    if [[ "$TORCH_PROFILE_DIR" != /* ]]; then
        TORCH_PROFILE_DIR="$(pwd -P)/$TORCH_PROFILE_DIR"
    fi
    [[ ! -e "$TORCH_PROFILE_DIR" || -d "$TORCH_PROFILE_DIR" ]] \
        || fatal "torch profile path is not a directory: $TORCH_PROFILE_DIR"
fi
if ((hw_perf_enabled)); then
    if [[ -z "$HW_PERF_OUTPUT" ]]; then
        HW_PERF_OUTPUT="$OUTPUT_DIR/rpu_hwperf"
    elif [[ "$HW_PERF_OUTPUT" != /* ]]; then
        HW_PERF_OUTPUT="$(pwd -P)/$HW_PERF_OUTPUT"
    fi
    [[ ! -e "$HW_PERF_OUTPUT" || -d "$HW_PERF_OUTPUT" ]] \
        || fatal "hardware profile output is not a directory: $HW_PERF_OUTPUT"
fi

TORCH_LIB_DIR="$($PYTHON_BIN -c 'from pathlib import Path; import torch; print(Path(torch.__file__).resolve().parent / "lib")')"
RUNTIME_LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$TORCH_LIB_DIR:$RHINO_LAUNCH_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PYTHON_ARGS=(
    "$SCRIPT_DIR/examples/wall_qwen35_openloop.py"
    --dataset-dir "$DATASET_DIR"
    --checkpoint "$CHECKPOINT_PATH"
    --instruction-source "$INSTRUCTION_SOURCE"
    --robot-id "$ROBOT_ID"
    --dataset-key "$NORM_KEY"
    --inference-steps "$NUM_INFERENCE_STEPS"
    --max-requests "$MAX_REQUESTS"
    --noise-seed "$NOISE_SEED"
)
if [[ -n "$FLOW_NOISE" ]]; then
    PYTHON_ARGS+=(--flow-noise "$FLOW_NOISE")
fi
if [[ $check_only == 1 ]]; then
    PYTHON_ARGS+=(--check-config)
else
    PYTHON_ARGS+=(
        --output-dir "$OUTPUT_DIR"
        --allow-numeric-blocked-vision
    )
fi
if [[ "$profile_mode" == ready-replay ]]; then
    PYTHON_ARGS+=(
        --torch-profile
        --torch-profile-output "$TORCH_PROFILE_OUTPUT"
    )
elif [[ "$profile_mode" == per-request ]]; then
    PYTHON_ARGS+=(--torch-profile-dir "$TORCH_PROFILE_DIR")
fi
if ((profile_enabled)); then
    ((profile_record_shapes)) && PYTHON_ARGS+=(--torch-profile-record-shapes)
    ((profile_memory)) && PYTHON_ARGS+=(--torch-profile-memory)
    ((profile_with_stack)) && PYTHON_ARGS+=(--torch-profile-with-stack)
fi
if ((hw_perf_enabled)); then
    PYTHON_ARGS+=(
        --hw-perf-output "$HW_PERF_OUTPUT"
        --hw-perf-max-dumps "$HW_PERF_MAX_DUMPS"
    )
fi

printf '%s\n' '[wall-qwen35-openloop] configuration'
printf '  %-22s %s\n' 'environment:' "$ENV_SH"
printf '  %-22s %s\n' 'Python:' "$PYTHON_BIN"
printf '  %-22s %s\n' 'checkpoint:' "$CHECKPOINT_PATH"
printf '  %-22s %s\n' 'dataset:' "$DATASET_DIR"
printf '  %-22s %s\n' 'instruction source:' "$INSTRUCTION_SOURCE"
printf '  %-22s %s\n' 'robot id:' "$ROBOT_ID"
printf '  %-22s %s\n' 'normalizer:' "$NORM_KEY"
printf '  %-22s %s\n' 'inference steps:' "$NUM_INFERENCE_STEPS"
printf '  %-22s %s\n' 'max requests:' "$([[ $MAX_REQUESTS == 0 ]] && printf all || printf '%s' "$MAX_REQUESTS")"
if [[ -n "$FLOW_NOISE" ]]; then
    printf '  %-22s %s\n' 'noise seed base:' 'ignored (explicit flow noise)'
else
    printf '  %-22s %s\n' 'noise seed base:' "$NOISE_SEED"
fi
printf '  %-22s %s\n' 'flow noise:' "${FLOW_NOISE:-generated and exported}"
printf '  %-22s %s\n' 'sudo:' "$([[ $use_sudo == 1 ]] && printf enabled || printf disabled)"
printf '  %-22s %s\n' 'torch profile:' "$profile_mode"
if ((profile_enabled)); then
    if [[ "$profile_mode" == ready-replay ]]; then
        printf '  %-22s %s\n' 'profile scope:' 'one repeat after unprofiled warmup'
    else
        printf '  %-22s %s\n' 'profile scope:' 'one trace per inference request'
    fi
    if [[ "$profile_mode" == ready-replay ]]; then
        printf '  %-22s %s\n' 'profile directory:' "$TORCH_PROFILE_OUTPUT"
    else
        printf '  %-22s %s\n' 'profile directory:' "$TORCH_PROFILE_DIR"
    fi
    printf '  %-22s %s\n' 'profile shapes:' "$profile_record_shapes"
    printf '  %-22s %s\n' 'profile memory:' "$profile_memory"
    printf '  %-22s %s\n' 'profile stacks:' "$profile_with_stack"
fi
printf '  %-22s %s\n' 'RPU hardware trace:' "$([[ $hw_perf_enabled == 1 ]] && printf enabled || printf disabled)"
if ((hw_perf_enabled)); then
    printf '  %-22s %s\n' 'hwperf directory:' "$HW_PERF_OUTPUT"
    printf '  %-22s %s\n' 'hwperf max dumps:' "$HW_PERF_MAX_DUMPS"
    printf '  %-22s %s MHz\n' 'RPU trace frequency:' "$LKN_RPU_FREQ_MHZ"
fi
printf '  %-22s %s\n' 'mode:' "$([[ $check_only == 1 ]] && printf check || printf controlled-RPU-evaluation)"
printf '  %-22s %s\n' 'output:' "$([[ $check_only == 1 ]] && printf none || printf '%s' "$OUTPUT_DIR")"

ENV_ARGS=(
    "PATH=$CONDA_PREFIX/bin:$PATH"
    "LD_LIBRARY_PATH=$RUNTIME_LD_LIBRARY_PATH"
    "RPU_KERNEL_LIB_PATH=$RPU_KERNEL_LIB_PATH"
    "RHINO_LAUNCH_LIB_DIR=$RHINO_LAUNCH_LIB_DIR"
    "LKN_RPU_FREQ_MHZ=$LKN_RPU_FREQ_MHZ"
    "RPU_FUSED_COEXIST_KEEP_PERSISTENT_GEN=1"
    "PYTHONUNBUFFERED=1"
    "TOKENIZERS_PARALLELISM=false"
    "TRANSFORMERS_OFFLINE=1"
    "MPLBACKEND=Agg"
    "MPLCONFIGDIR=/tmp/rhinoforge-matplotlib"
)

if [[ $use_sudo == 1 ]]; then
    COMMAND=(sudo /usr/bin/env "${ENV_ARGS[@]}" "$PYTHON_BIN" "${PYTHON_ARGS[@]}")
else
    COMMAND=(/usr/bin/env "${ENV_ARGS[@]}" "$PYTHON_BIN" "${PYTHON_ARGS[@]}")
fi

if [[ $check_only == 1 ]]; then
    "${COMMAND[@]}"
    printf '%s\n' '[wall-qwen35-openloop] preflight OK'
    exit 0
fi

LOG_FILE="${OUTPUT_DIR}.run.log"
[[ ! -e "$OUTPUT_DIR" ]] \
    || fatal "OUTPUT_DIR already exists; choose a fresh path: $OUTPUT_DIR"
[[ ! -e "$LOG_FILE" ]] \
    || fatal "log file already exists; choose a fresh OUTPUT_DIR: $LOG_FILE"
mkdir -p "$(dirname -- "$OUTPUT_DIR")"

set +e
"${COMMAND[@]}" 2>&1 | tee "$LOG_FILE"
pipeline_status=("${PIPESTATUS[@]}")
set -e
status=${pipeline_status[0]}
((status == 0 && pipeline_status[1] != 0)) && status=${pipeline_status[1]}
relocate_log() {
    local destination="$OUTPUT_DIR/run.log"
    [[ -d "$OUTPUT_DIR" && -f "$LOG_FILE" && ! -e "$destination" ]] || return 0
    if [[ $use_sudo == 1 ]]; then
        if sudo mv -- "$LOG_FILE" "$destination"; then
            LOG_FILE="$destination"
        else
            printf '[wall-qwen35-openloop] warning: could not move log into output directory\n' >&2
        fi
    elif mv -- "$LOG_FILE" "$destination"; then
        LOG_FILE="$destination"
    else
        printf '[wall-qwen35-openloop] warning: could not move log into output directory\n' >&2
    fi
}

if ((status != 0)); then
    relocate_log
    printf '[wall-qwen35-openloop] failed (status=%d); log: %s\n' \
        "$status" "$LOG_FILE" >&2
    exit "$status"
fi
relocate_log
printf '[wall-qwen35-openloop] completed; output: %s; log: %s\n' \
    "$OUTPUT_DIR" "$LOG_FILE"
