#!/usr/bin/env bash

set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: profile_cuda.sh <nsys|ncu|all> [options]

Collect a bounded profile from one CUDA Google Benchmark case. The default case
is the 256x256x256 tiled SGEMM benchmark, whose ten explicit warmup launches make
the default Nsight Compute launch skip select the first measured kernel.

Options:
  --build-dir PATH          CUDA build tree (default: <repo>/build-cuda)
  --benchmark-binary PATH   Benchmark executable override
  --output-dir PATH         Report directory (default: <repo>/profiles)
  --output-name NAME        Safe report stem (default: UTC timestamp)
  --benchmark-filter REGEX  Google Benchmark filter for one case
  --iterations N            Exact measured benchmark iterations (default: 5)
  --launch-skip N           Matching kernels NCU skips (default: 10)
  --launch-count N          Matching kernels NCU profiles (default: 1)
  --ncu-set NAME            Nsight Compute section set (default: full)
  --force                   Overwrite this stem's existing reports
  --dry-run                 Print commands without requiring CUDA or writing files
  --help                    Show this help

Profiler timing perturbs the target. Do not publish Google Benchmark timings from
an Nsight Compute run as ordinary latency measurements.
EOF
}

fail() {
  printf 'profile_cuda.sh: %s\n' "$*" >&2
  exit 2
}

require_option_value() {
  local option="$1"
  local remaining="$2"
  if [[ "${remaining}" -lt 2 ]]; then
    fail "${option} requires a value"
  fi
}

render_command() {
  local argument
  for argument in "$@"; do
    printf '%q ' "${argument}"
  done
  printf '\n'
}

record_optional_command() {
  local label="$1"
  shift

  printf '%s:\n' "${label}"
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '  <command unavailable: %s>\n' "$1"
    return
  fi
  if ! "$@" 2>&1; then
    printf '  <command failed>\n'
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

mode="${1:-}"
case "${mode}" in
  nsys | ncu | all) ;;
  "")
    usage >&2
    fail "a profiler mode is required"
    ;;
  *)
    usage >&2
    fail "unknown profiler mode '${mode}'"
    ;;
esac
shift

build_dir="${project_root}/build-cuda"
benchmark_binary=""
output_dir="${project_root}/profiles"
output_name="helix-$(date -u '+%Y%m%dT%H%M%SZ')"
benchmark_filter='^BM_CudaMatmulTiled/256/256/256/manual_time$'
iterations="5"
launch_skip="10"
launch_count="1"
ncu_set="full"
force="false"
dry_run="false"

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --build-dir)
      require_option_value "$1" "$#"
      build_dir="$2"
      shift 2
      ;;
    --benchmark-binary)
      require_option_value "$1" "$#"
      benchmark_binary="$2"
      shift 2
      ;;
    --output-dir)
      require_option_value "$1" "$#"
      output_dir="$2"
      shift 2
      ;;
    --output-name)
      require_option_value "$1" "$#"
      output_name="$2"
      shift 2
      ;;
    --benchmark-filter)
      require_option_value "$1" "$#"
      benchmark_filter="$2"
      shift 2
      ;;
    --iterations)
      require_option_value "$1" "$#"
      iterations="$2"
      shift 2
      ;;
    --launch-skip)
      require_option_value "$1" "$#"
      launch_skip="$2"
      shift 2
      ;;
    --launch-count)
      require_option_value "$1" "$#"
      launch_count="$2"
      shift 2
      ;;
    --ncu-set)
      require_option_value "$1" "$#"
      ncu_set="$2"
      shift 2
      ;;
    --force)
      force="true"
      shift
      ;;
    --dry-run)
      dry_run="true"
      shift
      ;;
    --help | -h)
      usage
      exit 0
      ;;
    *) fail "unknown option '$1'" ;;
  esac
done

[[ -n "${build_dir}" ]] || fail "--build-dir must not be empty"
[[ -n "${output_dir}" ]] || fail "--output-dir must not be empty"
[[ -n "${benchmark_filter}" ]] || fail "--benchmark-filter must not be empty"
[[ "${benchmark_filter}" != *$'\n'* && "${benchmark_filter}" != *$'\r'* ]] ||
  fail "--benchmark-filter must be a single line"
[[ "${output_name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] ||
  fail "--output-name must contain only letters, digits, '.', '_', or '-' and start alphanumeric"
[[ "${#output_name}" -le 96 ]] || fail "--output-name must be at most 96 characters"
[[ "${iterations}" =~ ^[1-9][0-9]*$ ]] || fail "--iterations must be a positive integer"
[[ "${launch_skip}" =~ ^[0-9]+$ ]] || fail "--launch-skip must be a non-negative integer"
[[ "${launch_count}" =~ ^[1-9][0-9]*$ ]] || fail "--launch-count must be a positive integer"
[[ "${ncu_set}" =~ ^[A-Za-z0-9_-]+$ ]] || fail "--ncu-set is not a valid section-set name"

if [[ -z "${benchmark_binary}" ]]; then
  benchmark_binary="${build_dir}/benchmarks/helix_benchmarks"
fi

run_nsys="false"
run_ncu="false"
if [[ "${mode}" == "nsys" || "${mode}" == "all" ]]; then
  run_nsys="true"
fi
if [[ "${mode}" == "ncu" || "${mode}" == "all" ]]; then
  run_ncu="true"
fi

report_prefix="${output_dir}/${output_name}"
nsys_prefix="${report_prefix}-nsys"
ncu_prefix="${report_prefix}-ncu"
manifest_path="${report_prefix}-environment.txt"

benchmark_arguments=(
  "--benchmark_filter=${benchmark_filter}"
  "--benchmark_min_time=${iterations}x"
  "--benchmark_repetitions=1"
  "--benchmark_enable_random_interleaving=false"
  "--benchmark_color=false"
)

preflight_command=(
  "${benchmark_binary}"
  "--benchmark_filter=${benchmark_filter}"
  "--benchmark_min_time=1x"
  "--benchmark_repetitions=1"
  "--benchmark_color=false"
)

nsys_command=(
  nsys profile
  "--trace=cuda,nvtx,osrt"
  "--sample=none"
  "--cpuctxsw=none"
  "--stats=true"
  "--force-overwrite=${force}"
  "--output=${nsys_prefix}"
  "${benchmark_binary}"
  "${benchmark_arguments[@]}"
)

ncu_command=(
  ncu
  "--set=${ncu_set}"
  "--target-processes=application-only"
  "--kernel-name-base=demangled"
  "--launch-skip=${launch_skip}"
  "--launch-count=${launch_count}"
  "--export=${ncu_prefix}"
)
if [[ "${force}" == "true" ]]; then
  ncu_command+=(--force-overwrite)
fi
ncu_command+=("${benchmark_binary}" "${benchmark_arguments[@]}")

if [[ "${dry_run}" == "true" ]]; then
  if [[ "${run_nsys}" == "true" ]]; then
    printf 'DRY-RUN Nsight Systems command:\n'
    render_command "${nsys_command[@]}"
  fi
  if [[ "${run_ncu}" == "true" ]]; then
    printf 'DRY-RUN Nsight Compute command:\n'
    render_command "${ncu_command[@]}"
  fi
  printf 'DRY-RUN CUDA preflight command:\n'
  render_command "${preflight_command[@]}"
  printf 'DRY-RUN environment manifest: %s\n' "${manifest_path}"
  exit 0
fi

[[ -x "${benchmark_binary}" ]] ||
  fail "CUDA benchmark executable is missing or not executable: ${benchmark_binary}"

if [[ "${run_nsys}" == "true" ]]; then
  command -v nsys >/dev/null 2>&1 || fail "Nsight Systems CLI 'nsys' is not in PATH"
fi
if [[ "${run_ncu}" == "true" ]]; then
  command -v ncu >/dev/null 2>&1 || fail "Nsight Compute CLI 'ncu' is not in PATH"
fi
command -v nvidia-smi >/dev/null 2>&1 ||
  fail "nvidia-smi is required to verify and record the GPU environment"
if ! gpu_inventory="$(nvidia-smi -L 2>&1)"; then
  fail "no usable NVIDIA driver/device was reported by nvidia-smi: ${gpu_inventory}"
fi
[[ -n "${gpu_inventory}" ]] || fail "nvidia-smi reported no visible NVIDIA GPU"

if ! selected_benchmarks="$(
  "${benchmark_binary}" "--benchmark_filter=${benchmark_filter}" --benchmark_list_tests=true 2>&1
)"; then
  fail "Google Benchmark rejected filter '${benchmark_filter}': ${selected_benchmarks}"
fi
[[ -n "${selected_benchmarks}" ]] ||
  fail "benchmark filter '${benchmark_filter}' selected no registered cases"

selected_count=0
while IFS= read -r selected_benchmark; do
  [[ -n "${selected_benchmark}" ]] || continue
  [[ "${selected_benchmark}" == BM_Cuda* ]] ||
    fail "profiling accepts only CUDA benchmark cases, selected: ${selected_benchmark}"
  selected_count=$((selected_count + 1))
done <<<"${selected_benchmarks}"
[[ "${selected_count}" -eq 1 ]] ||
  fail "benchmark filter must select exactly one case, selected ${selected_count}"

expected_outputs=("${manifest_path}")
if [[ "${run_nsys}" == "true" ]]; then
  expected_outputs+=("${nsys_prefix}.nsys-rep" "${nsys_prefix}.sqlite")
fi
if [[ "${run_ncu}" == "true" ]]; then
  expected_outputs+=("${ncu_prefix}.ncu-rep")
fi

for output in "${expected_outputs[@]}"; do
  if [[ -L "${output}" ]]; then
    fail "refusing to overwrite symbolic-link output: ${output}"
  fi
  if [[ "${force}" != "true" && -e "${output}" ]]; then
    fail "output already exists (use --force for this exact stem): ${output}"
  fi
done

if ! preflight_output="$("${preflight_command[@]}" 2>&1)"; then
  fail "CUDA benchmark preflight failed: ${preflight_output}"
fi
if [[ "${preflight_output}" == *"ERROR OCCURRED:"* || "${preflight_output}" == *"SKIPPED"* ]]; then
  fail "CUDA benchmark preflight did not execute on the selected device: ${preflight_output}"
fi

mkdir -p "${output_dir}"

cxx_compiler=""
if [[ -r "${build_dir}/CMakeCache.txt" ]]; then
  while IFS= read -r cache_entry; do
    case "${cache_entry}" in
      CMAKE_CXX_COMPILER:FILEPATH=* | CMAKE_CXX_COMPILER:STRING=*)
        cxx_compiler="${cache_entry#*=}"
        break
        ;;
    esac
  done <"${build_dir}/CMakeCache.txt"
fi

{
  printf 'Helix CUDA profiling environment\n'
  printf 'UTC timestamp: %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  printf 'mode: %s\n' "${mode}"
  printf 'benchmark executable: %s\n' "${benchmark_binary}"
  printf 'benchmark filter: %s\n' "${benchmark_filter}"
  printf 'measured iterations: %s\n' "${iterations}"
  printf 'dtype: Float32\n'
  printf 'NCU launch skip/count/set: %s/%s/%s\n' "${launch_skip}" "${launch_count}" "${ncu_set}"
  printf 'selected benchmark cases:\n%s\n' "${selected_benchmarks}"
  printf 'CUDA preflight output:\n%s\n' "${preflight_output}"
  if [[ "${run_nsys}" == "true" ]]; then
    printf 'Nsight Systems command:\n'
    render_command "${nsys_command[@]}"
  fi
  if [[ "${run_ncu}" == "true" ]]; then
    printf 'Nsight Compute command:\n'
    render_command "${ncu_command[@]}"
  fi
  printf 'GPU inventory:\n%s\n' "${gpu_inventory}"
  record_optional_command "GPU details" nvidia-smi \
    --query-gpu=name,uuid,driver_version,memory.total \
    --format=csv,noheader
  record_optional_command "OS" uname -a
  record_optional_command "CPU" lscpu
  record_optional_command "RAM" free -h
  record_optional_command "Git commit" git -C "${project_root}" rev-parse HEAD
  record_optional_command "Git worktree" git -C "${project_root}" status --short --branch
  record_optional_command "CMake" cmake --version
  if [[ -n "${cxx_compiler}" ]]; then
    record_optional_command "C++ compiler" "${cxx_compiler}" --version
  else
    printf 'C++ compiler:\n  <not found in CMake cache>\n'
  fi
  record_optional_command "CUDA compiler" nvcc --version
  if [[ "${run_nsys}" == "true" ]]; then
    record_optional_command "Nsight Systems" nsys --version
  fi
  if [[ "${run_ncu}" == "true" ]]; then
    record_optional_command "Nsight Compute" ncu --version
  fi
  if [[ -r "${build_dir}/CMakeCache.txt" ]]; then
    printf 'Relevant CMake cache entries:\n'
    while IFS= read -r cache_entry; do
      case "${cache_entry}" in
        CMAKE_BUILD_TYPE:* | CMAKE_CXX_COMPILER:* | CMAKE_CUDA_COMPILER:* | \
          CMAKE_CUDA_ARCHITECTURES:* | HELIX_ENABLE_CUDA:* | \
          HELIX_ENABLE_CUDA_LINEINFO:* | HELIX_BUILD_BENCHMARKS:*)
          printf '%s\n' "${cache_entry}"
          ;;
      esac
    done <"${build_dir}/CMakeCache.txt"
  fi
} >"${manifest_path}"

printf 'Selected benchmark case(s):\n%s\n' "${selected_benchmarks}"
printf 'Environment manifest: %s\n' "${manifest_path}"

if [[ "${run_nsys}" == "true" ]]; then
  printf 'Collecting Nsight Systems report: %s.nsys-rep\n' "${nsys_prefix}"
  if ! "${nsys_command[@]}"; then
    fail "Nsight Systems collection failed; inspect its diagnostics and ${manifest_path}"
  fi
fi

if [[ "${run_ncu}" == "true" ]]; then
  printf 'Collecting Nsight Compute report: %s.ncu-rep\n' "${ncu_prefix}"
  if ! "${ncu_command[@]}"; then
    fail "Nsight Compute collection failed; inspect its diagnostics and ${manifest_path}"
  fi
fi

printf 'Profiling complete. Reports are under %s\n' "${output_dir}"
