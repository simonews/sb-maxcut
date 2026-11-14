#!/usr/bin/env bash
# Installs a pinned CUDA toolkit from PyPI into a private virtual environment (no sudo) and
# generates CMakeUserPresets.json with the machine-specific toolkit paths.
#
# Usage: scripts/setup_cuda_pypi.sh [--version 13.4.3] [--prefix DIR] [--no-presets]
#
# Idempotent: a second run reuses the virtual environment, re-checks the installation,
# recreates the library shim and regenerates CMakeUserPresets.json. nvidia-nvjitlink is pinned
# to the version of nvidia-cuda-nvcc, also in an environment that has another version.
set -euo pipefail

readonly DEFAULT_VERSION="13.4.3"
readonly EXTRAS="nvcc,tileiras,cudart,cublas,cusparse,curand,cccl,sanitizer,cuobjdump,nvdisasm"
readonly GENERATOR_ID="scripts/setup_cuda_pypi.sh"

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"

work_dir=""
cleanup() {
  if [[ -n "${work_dir}" ]]; then
    rm -rf -- "${work_dir}"
  fi
}
trap cleanup EXIT

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: scripts/setup_cuda_pypi.sh [--version X.Y.Z] [--prefix DIR] [--no-presets]

Installs "cuda-toolkit[${EXTRAS}]==<version>"
from PyPI into a virtual environment under DIR, creates a library shim for CMake and writes
the 'cuda' and 'cuda-strict' presets to CMakeUserPresets.json in the repository root.

Options:
  --version X.Y.Z  toolkit version (default: ${DEFAULT_VERSION}); only CUDA 13.x is supported
  --prefix DIR     installation prefix
                   (default: \${XDG_DATA_HOME:-\$HOME/.local/share}/sbmc/cuda-<version>)
  --no-presets     do not write CMakeUserPresets.json
  -h, --help       show this help
EOF
}

# Prints the installed version of a Python distribution of the virtual environment; fails if the
# distribution is not installed. Needs venv_python.
package_version() {
  "${venv_python}" -c 'import importlib.metadata as m, sys; print(m.version(sys.argv[1]))' "$1" \
    2>/dev/null
}

# Escapes a string for use inside a JSON string literal.
json_escape() {
  local s="$1"
  s="${s//\\/\\\\}"
  s="${s//\"/\\\"}"
  printf '%s' "${s}"
}

# ---------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------
version="${DEFAULT_VERSION}"
prefix=""
write_presets=1

while (($# > 0)); do
  case "$1" in
    --version)
      (($# >= 2)) || die "--version needs a value"
      version="$2"
      shift 2
      ;;
    --version=*)
      version="${1#*=}"
      shift
      ;;
    --prefix)
      (($# >= 2)) || die "--prefix needs a value"
      prefix="$2"
      shift 2
      ;;
    --prefix=*)
      prefix="${1#*=}"
      shift
      ;;
    --no-presets)
      write_presets=0
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

if [[ ! "${version}" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
  die "invalid version '${version}': expected MAJOR.MINOR.PATCH, for example ${DEFAULT_VERSION}"
fi
major="${BASH_REMATCH[1]}"
minor="${BASH_REMATCH[2]}"

if [[ "${major}" == "12" ]]; then
  die "CUDA ${version} is not supported by this script yet: the CUDA 12 PyPI packages use a" \
    "different layout and do not ship nvcc. The cuda12 presets need a CUDA 12 toolkit" \
    "installed by other means (planned for phase 02)."
fi
if [[ "${major}" != "13" ]]; then
  die "CUDA ${version} is not supported by this script: only CUDA 13.x is handled"
fi

if [[ -z "${prefix}" ]]; then
  prefix="${XDG_DATA_HOME:-${HOME}/.local/share}/sbmc/cuda-${version}"
fi
mkdir -p -- "${prefix}"
prefix="$(cd -- "${prefix}" && pwd -P)"

command -v python3 >/dev/null 2>&1 || die "python3 not found (on Ubuntu: sudo apt install python3 python3-venv)"

# ---------------------------------------------------------------------------------------------
# 1-2. Virtual environment and PyPI installation
# ---------------------------------------------------------------------------------------------
venv_dir="${prefix}/venv"
venv_python="${venv_dir}/bin/python"
requirement="cuda-toolkit[${EXTRAS}]==${version}"

echo "==> Prefix: ${prefix}"
if [[ ! -x "${venv_python}" ]]; then
  echo "==> Creating the virtual environment ${venv_dir}"
  python3 -m venv "${venv_dir}" ||
    die "python3 -m venv failed (on Ubuntu: sudo apt install python3-venv), then remove ${venv_dir} and retry"
fi
"${venv_python}" -m pip --version >/dev/null 2>&1 ||
  die "pip is not available in ${venv_dir}: remove that directory and rerun the script"

echo "==> Installing ${requirement}"
"${venv_python}" -m pip install --disable-pip-version-check --no-input "${requirement}" ||
  die "pip install failed for ${requirement}"

# The metapackage requires nvidia-nvjitlink only as ">=<nvcc version>,<next major>", so pip picks
# the newest release. It is pinned to the version of the nvidia-cuda-nvcc package installed by
# the metapackage: a second pip install resolves the pin together with the metapackage (every
# requirement stays consistent) and moves an existing environment to the pinned version.
nvcc_package_version="$(package_version nvidia-cuda-nvcc)" ||
  die "nvidia-cuda-nvcc is not installed in ${venv_dir}: cannot pin nvidia-nvjitlink"
nvjitlink_pin="nvidia-nvjitlink==${nvcc_package_version}"
echo "==> Pinning ${nvjitlink_pin} (the version of nvidia-cuda-nvcc)"
work_dir="$(mktemp -d)"
if ! "${venv_python}" -m pip install --disable-pip-version-check --no-input "${requirement}" \
  "${nvjitlink_pin}" 2>&1 | tee "${work_dir}/pip-pin.log"; then
  if grep -qF "No matching distribution found" "${work_dir}/pip-pin.log"; then
    die "${nvjitlink_pin} is not available on PyPI: nvidia-nvjitlink cannot be pinned to the" \
      "nvcc version ${nvcc_package_version}"
  fi
  die "pip install failed for ${requirement} with ${nvjitlink_pin}"
fi
rm -rf -- "${work_dir}"
work_dir=""

# ---------------------------------------------------------------------------------------------
# 3. Toolkit root, asked to the virtual environment's interpreter
# ---------------------------------------------------------------------------------------------
toolkit_root="$(
  "${venv_python}" - "${major}" <<'PY'
import os
import sys
import sysconfig

major = sys.argv[1]
paths = sysconfig.get_paths()
for key in ("purelib", "platlib"):
    candidate = os.path.join(paths[key], "nvidia", "cu" + major)
    if os.path.isdir(candidate):
        print(os.path.realpath(candidate))
        break
PY
)"
[[ -n "${toolkit_root}" ]] ||
  die "nvidia/cu${major} not found in the site-packages of ${venv_dir}: unexpected package layout"

for required in bin/nvcc bin/tileiras bin/compute-sanitizer bin/cuobjdump bin/nvdisasm \
  include/cuda_runtime.h include/cuda_tile.h lib; do
  [[ -e "${toolkit_root}/${required}" ]] ||
    die "unexpected toolkit layout: ${toolkit_root}/${required} is missing"
done

# ---------------------------------------------------------------------------------------------
# 4. Library shim: versioned and unversioned names for every lib*.so.* of the toolkit
# ---------------------------------------------------------------------------------------------
shim_dir="${prefix}/lib-shim"
rm -rf -- "${shim_dir}"
mkdir -p -- "${shim_dir}"

shopt -s nullglob
libraries=("${toolkit_root}"/lib/lib*.so.*)
shopt -u nullglob
((${#libraries[@]} > 0)) || die "no versioned shared libraries (lib*.so.*) in ${toolkit_root}/lib"

# Shortest names first: the unversioned link then points to the SONAME-like file
# (libfoo.so.13 rather than libfoo.so.13.1.2) when both exist.
shim_count=0
while IFS= read -r library; do
  name="$(basename -- "${library}")"
  ln -s -- "${library}" "${shim_dir}/${name}"
  unversioned="${name%%.so.*}.so"
  if [[ ! -e "${shim_dir}/${unversioned}" && ! -L "${shim_dir}/${unversioned}" ]]; then
    ln -s -- "${library}" "${shim_dir}/${unversioned}"
  fi
  shim_count=$((shim_count + 1))
done < <(printf '%s\n' "${libraries[@]}" | awk '{ print length($0) "\t" $0 }' | sort -n -k1,1 -k2 |
  cut -f2-)

# ---------------------------------------------------------------------------------------------
# 5. Checks
# ---------------------------------------------------------------------------------------------
echo "==> Checking the installation"
nvcc="${toolkit_root}/bin/nvcc"

if [[ ! "${nvcc_package_version}" =~ ^([0-9]+\.[0-9]+\.[0-9]+) ]]; then
  die "unexpected nvidia-cuda-nvcc package version '${nvcc_package_version}'"
fi
expected_release="release ${major}.${minor}"
expected_build="V${BASH_REMATCH[1]}"

nvcc_version_output="$("${nvcc}" --version 2>&1)" || die "nvcc --version failed: ${nvcc_version_output}"
if ! grep -qF -- "${expected_release}," <<<"${nvcc_version_output}" ||
  ! grep -qF -- "${expected_build}" <<<"${nvcc_version_output}"; then
  printf '%s\n' "${nvcc_version_output}" >&2
  die "nvcc does not report '${expected_release}' and '${expected_build}'"
fi
echo "    nvcc: ${expected_release}, ${expected_build} (ok)"

nvdisasm_version_line=""
for tool in tileiras compute-sanitizer cuobjdump nvdisasm; do
  if ! tool_output="$("${toolkit_root}/bin/${tool}" --version 2>&1)"; then
    printf '%s\n' "${tool_output}" >&2
    die "${tool} --version failed"
  fi
  tool_version_line="$(grep -iE 'version|release|V[0-9]+\.[0-9]+' <<<"${tool_output}" | tail -n 1)"
  echo "    ${tool}: ${tool_version_line:-responds} (ok)"
  if [[ "${tool}" == "nvdisasm" ]]; then
    nvdisasm_version_line="${tool_version_line:-responds}"
  fi
done

nvjitlink_version="$(package_version nvidia-nvjitlink)" ||
  die "nvidia-nvjitlink is not installed in ${venv_dir}"
if [[ "${nvjitlink_version}" != "${nvcc_package_version}" ]]; then
  die "nvidia-nvjitlink ${nvjitlink_version} differs from nvidia-cuda-nvcc ${nvcc_package_version}"
fi
echo "    nvidia-nvjitlink: ${nvjitlink_version}, same as nvidia-cuda-nvcc (ok)"

work_dir="$(mktemp -d)"
cat >"${work_dir}/tile_probe.cu" <<'CU'
#include <cuda_tile.h>

__tile_global__ void sbmc_setup_tile_probe() {}
CU
if ! compile_output="$("${nvcc}" -enable-tile -std=c++20 -arch=sm_89 -c \
  "${work_dir}/tile_probe.cu" -o "${work_dir}/tile_probe.o" 2>&1)"; then
  printf '%s\n' "${compile_output}" >&2
  die "nvcc cannot compile a minimal Tile kernel for sm_89 (is tileiras reachable from nvcc?)"
fi
echo "    nvcc -enable-tile -std=c++20 -arch=sm_89: minimal __tile_global__ kernel compiled (ok)"

# cuobjdump runs the nvdisasm found next to it in the toolkit's bin folder (no PATH needed).
if ! sass_output="$("${toolkit_root}/bin/cuobjdump" -sass "${work_dir}/tile_probe.o" 2>&1)" ||
  ! grep -qE 'Function : .*sbmc_setup_tile_probe' <<<"${sass_output}"; then
  printf '%s\n' "${sass_output}" >&2
  die "cuobjdump -sass cannot disassemble the Tile kernel of the probe (nvdisasm not found?)"
fi
rm -rf -- "${work_dir}"
work_dir=""
echo "    cuobjdump -sass: Tile kernel of the minimal probe disassembled (ok)"

# ---------------------------------------------------------------------------------------------
# 6. CMakeUserPresets.json
# ---------------------------------------------------------------------------------------------
presets_file="${repo_root}/CMakeUserPresets.json"
presets_status="not written (--no-presets)"
if ((write_presets)); then
  if [[ -e "${presets_file}" ]] &&
    ! grep -qF -- "\"generated-by\": \"${GENERATOR_ID}\"" "${presets_file}"; then
    die "${presets_file} exists and was not generated by this script: left untouched." \
      "Move it away (or merge the presets by hand) and rerun, or use --no-presets."
  fi

  j_root="$(json_escape "${toolkit_root}")"
  j_nvcc="$(json_escape "${nvcc}")"
  j_shim="$(json_escape "${shim_dir}")"
  j_venv="$(json_escape "${venv_dir}")"
  work_dir="$(mktemp -d)"
  cat >"${work_dir}/CMakeUserPresets.json" <<EOF
{
  "version": 6,
  "vendor": {
    "sbmc": {
      "generated-by": "${GENERATOR_ID}",
      "note": "Machine-specific file, ignored by git. Regenerate it with ${GENERATOR_ID}.",
      "cuda-version": "${version}",
      "toolkit-root": "${j_root}",
      "lib-shim": "${j_shim}",
      "venv": "${j_venv}"
    }
  },
  "configurePresets": [
    {
      "name": "cuda",
      "displayName": "CUDA ${version} (PyPI)",
      "inherits": "cuda-base",
      "cacheVariables": {
        "CMAKE_CUDA_COMPILER": "${j_nvcc}",
        "CUDAToolkit_ROOT": "${j_root}",
        "CMAKE_LIBRARY_PATH": "${j_shim}"
      }
    },
    {
      "name": "cuda-strict",
      "displayName": "CUDA ${version} (PyPI), no FMA contraction",
      "inherits": "cuda-strict-base",
      "cacheVariables": {
        "CMAKE_CUDA_COMPILER": "${j_nvcc}",
        "CUDAToolkit_ROOT": "${j_root}",
        "CMAKE_LIBRARY_PATH": "${j_shim}"
      }
    }
  ],
  "buildPresets": [
    {
      "name": "cuda",
      "configurePreset": "cuda",
      "jobs": 8
    },
    {
      "name": "cuda-strict",
      "configurePreset": "cuda-strict",
      "jobs": 8
    }
  ],
  "testPresets": [
    {
      "name": "cuda",
      "configurePreset": "cuda",
      "output": {
        "outputOnFailure": true
      }
    },
    {
      "name": "cuda-strict",
      "configurePreset": "cuda-strict",
      "output": {
        "outputOnFailure": true
      }
    }
  ]
}
EOF
  mv -f -- "${work_dir}/CMakeUserPresets.json" "${presets_file}"
  rm -rf -- "${work_dir}"
  work_dir=""
  presets_status="${presets_file} (configure/build/test presets: cuda, cuda-strict)"
fi

# ---------------------------------------------------------------------------------------------
# 7. Summary
# ---------------------------------------------------------------------------------------------
cat <<EOF

==> CUDA toolkit ready
    requested version : ${version} (${requirement})
    nvcc              : ${expected_release}, ${expected_build}
    nvdisasm          : ${nvdisasm_version_line}
    nvidia-nvjitlink  : ${nvjitlink_version} (pinned, same as nvidia-cuda-nvcc ${nvcc_package_version})
    virtual env       : ${venv_dir}
    toolkit root      : ${toolkit_root}
    library shim      : ${shim_dir} (${shim_count} versioned libraries)
    presets           : ${presets_status}

Next steps (from ${repo_root}):
    cmake --preset cuda-strict && cmake --build --preset cuda-strict && ctest --preset cuda-strict
    cmake --preset cuda        && cmake --build --preset cuda        && ctest --preset cuda
EOF
