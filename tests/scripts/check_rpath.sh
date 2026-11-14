#!/usr/bin/env bash
# Checks that executables find the CUDA libraries through RPATH, not RUNPATH.
#
# Usage: tests/scripts/check_rpath.sh <executable>...
#
# For every executable, readelf -d must show an RPATH entry with a folder that contains
# libcudart.so.13 (the lib-shim of the project's toolkit) and no RUNPATH entry: the dynamic loader
# searches RPATH before LD_LIBRARY_PATH, RUNPATH after it (see sbmc::cuda_rpath in
# cmake/SbmcFlags.cmake). $ORIGIN in RPATH stands for the folder of the executable.
#
# Prints one line per executable: PASS or FAIL, with the RPATH and RUNPATH entries found.
# Exit codes: 0 every executable passed; 1 at least one FAIL or missing file (also for a command
# line without executables); 77 readelf is not available (registered in CTest as
# SKIP_RETURN_CODE).
set -euo pipefail

readonly CUDART="libcudart.so.13"

if (($# == 0)); then
  echo "Usage: tests/scripts/check_rpath.sh <executable>..." >&2
  exit 1
fi
if ! command -v readelf >/dev/null 2>&1; then
  echo "SKIP: readelf not found (it comes with binutils)"
  exit 77
fi

# Prints the value of a dynamic-section entry ("RPATH" or "RUNPATH") of readelf -d output, with
# several entries joined by ':'; prints nothing when there is no such entry.
dynamic_entry() {
  local values
  values="$(sed -n "s/^.*($1)[[:space:]]*Library [a-z]*: \[\(.*\)\][[:space:]]*\$/\1/p" <<<"$2")"
  printf '%s' "${values//$'\n'/:}"
}

status=0
for exe in "$@"; do
  if [[ ! -f "${exe}" ]]; then
    echo "FAIL ${exe}: file not found"
    status=1
    continue
  fi
  if ! dynamic="$(readelf -d -- "${exe}" 2>&1)"; then
    echo "FAIL ${exe}: readelf -d failed: ${dynamic%%$'\n'*}"
    status=1
    continue
  fi
  rpath="$(dynamic_entry RPATH "${dynamic}")"
  runpath="$(dynamic_entry RUNPATH "${dynamic}")"
  entries="RPATH [${rpath:-none}], RUNPATH [${runpath:-none}]"

  origin="$(cd -- "$(dirname -- "${exe}")" && pwd -P)"
  cudart_dir=""
  IFS=':' read -r -a rpath_dirs <<<"${rpath}"
  for dir in "${rpath_dirs[@]}"; do
    dir="${dir//\$\{ORIGIN\}/${origin}}"
    dir="${dir//\$ORIGIN/${origin}}"
    if [[ -n "${dir}" && -e "${dir}/${CUDART}" ]]; then
      cudart_dir="${dir}"
      break
    fi
  done

  if [[ -n "${runpath}" ]]; then
    echo "FAIL ${exe}: RUNPATH present; ${entries}"
    status=1
  elif [[ -z "${cudart_dir}" ]]; then
    echo "FAIL ${exe}: no RPATH folder contains ${CUDART}; ${entries}"
    status=1
  else
    echo "PASS ${exe}: ${entries}; ${CUDART} in ${cudart_dir}"
  fi
done
exit "${status}"
