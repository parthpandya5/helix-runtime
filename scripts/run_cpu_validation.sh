#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${project_root}/build}"

cmake -S "${project_root}" -B "${build_dir}" \
  -DHELIX_ENABLE_CUDA=OFF \
  -DHELIX_BUILD_TESTS=ON \
  -DHELIX_BUILD_BENCHMARKS=ON
cmake --build "${build_dir}" -j
ctest --test-dir "${build_dir}" --output-on-failure
cmake --build "${build_dir}" --target format-check
