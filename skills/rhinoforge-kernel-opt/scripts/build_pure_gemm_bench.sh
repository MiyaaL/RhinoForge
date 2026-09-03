#!/usr/bin/env bash
# Build the standalone Launch-SDK GEMM harness without configuring the full
# PyTorch extension.  Keep the output outside the repository: the binary is a
# board diagnostic, not a production runtime asset.
set -euo pipefail

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
sdk_dir=${RHINO_LAUNCH_DIR:-/home/hx/.local/opt/rhino-launch-kernel-v1.0.0-linux-aarch64}
out=${1:-/tmp/rhinoforge-pure-gemm-build/pure_gemm_bench}
cxx=${CXX:-g++}

mkdir -p "$(dirname -- "$out")"
exec "$cxx" -std=c++17 -O2 -Wall -Wextra -Wpedantic \
  "$script_dir/pure_gemm_bench.cpp" \
  -I"$sdk_dir/include" -L"$sdk_dir/lib" \
  -Wl,-rpath,"$sdk_dir/lib" \
  -lrhino_launch -lrpu_api -lHxil -lHxMem -o "$out"
