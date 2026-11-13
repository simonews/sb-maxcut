#!/usr/bin/env bash
# Writes an environment report to results/env/env_<hostname>_<YYYYmmdd-HHMMSS>.txt (the folder
# is created if needed) and prints the path of the file.
#
# Usage: scripts/env_report.sh [--preset cuda]
#
# The report is a list of "key: value" lines grouped in [sections]; multi-line values follow
# their key, indented by two spaces. A missing or failing tool never makes the script fail:
# its value is written as "n/a". --preset (default: cuda) selects the build tree whose
# tests/smoke/smoke_simt output is appended to the report.
set -euo pipefail

readonly NA="n/a"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"

usage() {
  echo "Usage: scripts/env_report.sh [--preset NAME]   (default preset: cuda)"
}

preset="cuda"
while (($# > 0)); do
  case "$1" in
    --preset)
      if (($# < 2)); then
        usage >&2
        exit 1
      fi
      preset="$2"
      shift 2
      ;;
    --preset=*)
      preset="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 1
      ;;
  esac
done

# Prints the output of a command, or n/a when the command is missing or fails.
run_or_na() {
  local output status
  if ! command -v "$1" >/dev/null 2>&1; then
    printf '%s' "${NA}"
    return 0
  fi
  if output="$("$@" 2>&1)"; then
    printf '%s' "${output}"
  else
    status=$?
    printf '%s (exit status %d)' "${NA}" "${status}"
  fi
}

# Prints the version line of a tool's --version output, or n/a.
version_line() {
  local output line
  output="$(run_or_na "$1" --version)"
  if [[ "${output}" == "${NA}"* ]]; then
    printf '%s' "${output}"
    return 0
  fi
  line="$(grep -iE 'version|release|V[0-9]+\.[0-9]+' <<<"${output}" | tail -n 1 || true)"
  [[ -n "${line}" ]] || line="$(head -n 1 <<<"${output}")"
  printf '%s' "${line:-${NA}}"
}

report=""
section() { report+=$'\n'"[$1]"$'\n'; }
kv() { report+="$1: ${2:-${NA}}"$'\n'; }
kv_block() {
  report+="$1:"$'\n'
  report+="$(printf '%s\n' "${2:-${NA}}" | sed 's/^/  /')"$'\n'
}
yes_no() { if "$@"; then printf 'yes'; else printf 'no'; fi; }

host="${HOSTNAME:-}"
[[ -n "${host}" ]] || host="$(uname -n 2>/dev/null || echo unknown-host)"
stamp="$(date +%Y%m%d-%H%M%S)"

report+="# sbmc environment report"$'\n'
section "report"
kv "date" "$(date -Iseconds)"
kv "host" "${host}"
kv "preset" "${preset}"

# Prints the output of a git command run in the repository; fails when git fails or prints nothing.
git_out() {
  local output
  output="$(git -C "${repo_root}" "$@" 2>/dev/null)" || return 1
  [[ -n "${output}" ]] || return 1
  printf '%s' "${output}"
}

section "git"
if git -C "${repo_root}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  kv "commit" "$(git_out rev-parse --verify -q HEAD || echo "${NA} (no commits yet)")"
  kv "branch" "$(git_out symbolic-ref --short -q HEAD || echo "detached HEAD")"
  kv "tag" "$(git_out describe --tags --exact-match || echo none)"
  if [[ -n "$(git -C "${repo_root}" status --porcelain 2>/dev/null || true)" ]]; then
    kv "dirty" "yes"
  else
    kv "dirty" "no"
  fi
else
  kv "commit" "${NA}"
  kv "branch" "${NA}"
  kv "tag" "${NA}"
  kv "dirty" "${NA}"
fi

section "system"
pretty_name="${NA}"
if [[ -r /etc/os-release ]]; then
  pretty_name="$(sed -n 's/^PRETTY_NAME="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' /etc/os-release)"
fi
kv "pretty_name" "${pretty_name:-${NA}}"
kv "kernel" "$(uname -r 2>/dev/null || echo "${NA}")"
kv "wsl_distro" "${WSL_DISTRO_NAME:-${NA}}"
wsl_version="${NA}"
if command -v wsl.exe >/dev/null 2>&1; then
  # wsl.exe prints UTF-16 text with CRLF line ends.
  if wsl_output="$(timeout 20 wsl.exe --version 2>/dev/null | tr -d '\000\r')" &&
    [[ -n "${wsl_output}" ]]; then
    wsl_version="${wsl_output}"
  fi
fi
kv_block "wsl_version" "${wsl_version}"

section "cpu"
lscpu_output="$(run_or_na lscpu)"
model="$(sed -n 's/^Model name:[[:space:]]*//p' <<<"${lscpu_output}" | head -n 1)"
kv "model" "${model:-${NA}}"
kv "nproc" "$(run_or_na nproc)"
topology="$(grep -E '^(CPU\(s\)|On-line CPU\(s\) list|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)):' <<<"${lscpu_output}" || true)"
kv_block "lscpu_topology" "${topology:-${NA}}"
cpu_flags="$(grep -m 1 '^flags' /proc/cpuinfo 2>/dev/null || true)"
for flag in avx2 avx_vnni avx512f; do
  if [[ -z "${cpu_flags}" ]]; then
    kv "flag_${flag}" "${NA}"
  else
    kv "flag_${flag}" "$(yes_no grep -qw -- "${flag}" <<<"${cpu_flags}")"
  fi
done
kv "sys_devices_cpu_core" "$(if [[ -e /sys/devices/cpu_core ]]; then echo present; else echo absent; fi)"
kv "sys_devices_cpu_atom" "$(if [[ -e /sys/devices/cpu_atom ]]; then echo present; else echo absent; fi)"

section "memory"
kv "mem_total" "$(awk '/^MemTotal:/ { print $2 " " $3 }' /proc/meminfo 2>/dev/null || true)"
kv "swap_total" "$(awk '/^SwapTotal:/ { print $2 " " $3 }' /proc/meminfo 2>/dev/null || true)"

section "tools"
for tool in gcc g++ cmake ninja python3 git; do
  kv "${tool}" "$(version_line "${tool}")"
done

section "cuda_toolkit"
presets_file="${repo_root}/CMakeUserPresets.json"
toolkit_root=""
if [[ -f "${presets_file}" ]]; then
  kv "presets_file" "${presets_file}"
  toolkit_root="$(sed -n 's/^[[:space:]]*"toolkit-root":[[:space:]]*"\(.*\)",\{0,1\}[[:space:]]*$/\1/p' \
    "${presets_file}" | head -n 1)"
else
  kv "presets_file" "${NA} (run scripts/setup_cuda_pypi.sh)"
fi
kv "toolkit_root" "${toolkit_root:-${NA}}"
for tool in nvcc tileiras compute-sanitizer cuobjdump; do
  if [[ -n "${toolkit_root}" ]]; then
    kv "${tool}" "$(version_line "${toolkit_root}/bin/${tool}")"
  else
    kv "${tool}" "${NA}"
  fi
done

section "gpu"
kv_block "nvidia_smi_query" "$(run_or_na nvidia-smi \
  --query-gpu=name,driver_version,memory.total,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,clocks.max.sm,clocks.max.mem,power.limit \
  --format=csv)"
kv_block "nvidia_smi_topo" "$(run_or_na nvidia-smi topo -m)"

section "smoke_simt"
smoke="${repo_root}/build/${preset}/tests/smoke/smoke_simt"
kv "binary" "${smoke}"
if [[ -x "${smoke}" ]]; then
  smoke_status=0
  smoke_output="$("${smoke}" 2>&1)" || smoke_status=$?
  kv "exit_status" "${smoke_status}"
  kv_block "output" "${smoke_output}"
else
  kv "exit_status" "${NA} (not built: cmake --build --preset ${preset})"
fi

out_dir="${repo_root}/results/env"
mkdir -p -- "${out_dir}"
out_file="${out_dir}/env_${host}_${stamp}.txt"
printf '%s' "${report}" >"${out_file}"
echo "${out_file}"
