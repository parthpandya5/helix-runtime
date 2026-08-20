#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${project_root}/build-cuda}"

cmake -S "${project_root}" -B "${build_dir}" \
  -DHELIX_ENABLE_CUDA=ON \
  -DHELIX_BUILD_TESTS=ON \
  -DHELIX_BUILD_BENCHMARKS=ON
cmake --build "${build_dir}" -j

# These tests do not require a physical NVIDIA device.
ctest --test-dir "${build_dir}" -LE gpu --output-on-failure

# This group requires a visible NVIDIA GPU. On compile-only hosts, the tests skip
# and therefore provide no runtime-validation claim.
ctest --test-dir "${build_dir}" -L gpu --output-on-failure
