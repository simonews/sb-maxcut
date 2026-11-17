#!/usr/bin/env bash
# Creates or reuses the Python virtual environment of the project, <repository root>/.venv, and
# installs into it, from wheels only, the pinned packages of python/requirements.txt (NumPy).
# The Python tests (tests/python) use .venv/bin/python when CMake finds it at configure time.
#
# Usage: scripts/setup_python.sh [--python INTERPRETER]
#
# The base interpreter is python3 from the PATH, or the one given with --python; it must be
# Python 3.12 or newer (NumPy 2.5.4 requires it).
#
# Virtual environment, with the logic of scripts/setup_cuda_pypi.sh: an existing one whose pip
# works is reused ("existing"); otherwise "python -m venv" creates it ("ensurepip"). Where that
# fails because the Python has no ensurepip, or where an existing one has no working pip, the
# fallback recreates it with "python -m venv --without-pip" and installs pip into it with the
# system pip ("python -m pip --python ...", pip >= 22.3): "system pip <version>". The script
# removes only its own .venv, and only in those cases.
#
# Summary on stdout: "python: <version> (<path>)", "venv bootstrap: ...", "numpy: <version>",
# then "setup_python: OK". Exit status 0 on success, 1 on error. A second run reuses the virtual
# environment ("existing").
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
venv_dir="${repo_root}/.venv"
venv_python="${venv_dir}/bin/python"
requirements="${repo_root}/python/requirements.txt"

die() {
  printf 'error: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<EOF
Usage: scripts/setup_python.sh [--python INTERPRETER]

Creates or reuses the virtual environment ${venv_dir}
and installs python/requirements.txt into it from wheels only (pip --only-binary=:all:).

Options:
  --python INTERPRETER  base interpreter of a new virtual environment (default: python3 from the
                        PATH); Python 3.12 or newer
  -h, --help            show this help
EOF
}

# Succeeds when the virtual environment has an interpreter that runs "-m pip --version".
venv_pip_works() {
  [[ -x "${venv_python}" ]] && "${venv_python}" -m pip --version >/dev/null 2>&1
}

# Removes the virtual environment, but only the path computed by this script:
# <repository root>/.venv, with a repository root that is neither empty nor /.
remove_venv() {
  if [[ -z "${repo_root}" || "${repo_root}" == "/" || "${venv_dir}" != "${repo_root}/.venv" ]]; then
    die "refusing to remove '${venv_dir}': not <repository root>/.venv"
  fi
  rm -rf -- "${venv_dir}"
}

# Prints major.minor.micro of an interpreter; fails if it does not run.
python_version() {
  "$1" -I -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null
}

# Succeeds when the interpreter is Python 3.12 or newer.
python_recent_enough() {
  "$1" -I -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null
}

# ---------------------------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------------------------
base_python="python3"
while (($# > 0)); do
  case "$1" in
    --python)
      (($# >= 2)) || die "--python needs a value (see --help)"
      base_python="$2"
      shift 2
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument '$1' (see --help)"
      ;;
  esac
done

[[ -f "${requirements}" ]] || die "${requirements} not found"
numpy_pin="$(sed -n 's/^numpy==\([0-9][0-9.]*\)[[:space:]]*$/\1/p' "${requirements}")"
[[ -n "${numpy_pin}" ]] || die "${requirements} has no line 'numpy==<version>'"

base_path="$(command -v -- "${base_python}" || true)"
[[ -n "${base_path}" ]] ||
  die "Python interpreter '${base_python}' not found (on Ubuntu: sudo apt install python3 python3-venv)"
base_version="$(python_version "${base_path}")" || die "'${base_path}' does not run as a Python interpreter"
if ! python_recent_enough "${base_path}"; then
  die "Python ${base_version} (${base_path}) is too old: NumPy ${numpy_pin} requires Python 3.12" \
    "or newer; install it or pass a newer interpreter with --python"
fi

# ---------------------------------------------------------------------------------------------
# Virtual environment
# ---------------------------------------------------------------------------------------------
echo "==> Virtual environment: ${venv_dir}"
# venv_bootstrap: existing (reused), ensurepip (python -m venv) or "system pip <version>".
venv_bootstrap=""
venv_created_now=0
if venv_pip_works; then
  venv_bootstrap="existing"
elif [[ ! -e "${venv_dir}" ]]; then
  echo "==> Creating the virtual environment with ${base_path} (Python ${base_version})"
  venv_created_now=1
  if "${base_path}" -m venv "${venv_dir}" && venv_pip_works; then
    venv_bootstrap="ensurepip"
  else
    echo "==> python -m venv could not install pip into ${venv_dir} (ensurepip missing?)"
  fi
else
  echo "==> The virtual environment ${venv_dir} exists, but pip does not work in it"
fi

if [[ -z "${venv_bootstrap}" ]]; then
  # Fallback: needs a system pip >= 22.3, the first release with --python. It is checked before
  # anything is removed or installed.
  echo "==> Fallback: virtual environment without pip, then pip from the system pip"
  system_pip_status=0
  system_pip_output="$("${base_path}" -m pip --version 2>&1)" || system_pip_status=$?
  if ((system_pip_status == 0)) &&
    [[ "${system_pip_output}" =~ ^pip\ ([0-9]+)\.([0-9]+)([.0-9a-zA-Z]*) ]] &&
    ((BASH_REMATCH[1] > 22 || (BASH_REMATCH[1] == 22 && BASH_REMATCH[2] >= 3))); then
    system_pip_version="${BASH_REMATCH[1]}.${BASH_REMATCH[2]}${BASH_REMATCH[3]}"
  else
    # Nothing installed: the incomplete environment created by this run is removed, an older one
    # is left as it was.
    hint=""
    if ((venv_created_now)); then
      remove_venv
    else
      hint=" With python3-venv installed, removing ${venv_dir} lets the next run create it again."
    fi
    found="${system_pip_output%%$'\n'*}"
    if ((system_pip_status != 0)); then
      found="${found:+${found} }(exit status ${system_pip_status})"
    fi
    die "no working pip for the virtual environment ${venv_dir}: the fallback needs a system pip" \
      ">= 22.3 (the first with --python), but '${base_path} -m pip --version' gives:" \
      "${found:-no output}. Install the python3-venv package (on Ubuntu: sudo apt install" \
      "python3-venv) or a pip >= 22.3 for ${base_path}, then rerun.${hint}"
  fi
  remove_venv
  "${base_path}" -m venv --without-pip "${venv_dir}" ||
    die "${base_path} -m venv --without-pip ${venv_dir} failed"
  echo "==> Installing pip into ${venv_dir} with the system pip ${system_pip_version}"
  "${base_path}" -m pip --python "${venv_python}" install --disable-pip-version-check --no-input pip ||
    die "the system pip could not install pip into ${venv_dir}"
  venv_pip_works || die "pip does not work in ${venv_dir} after the fallback installation"
  venv_bootstrap="system pip ${system_pip_version}"
fi

# An existing environment keeps the Python it was created with: it must be recent enough too.
venv_version="$(python_version "${venv_python}")" ||
  die "${venv_python} does not run as a Python interpreter"
if ! python_recent_enough "${venv_python}"; then
  die "the virtual environment ${venv_dir} has Python ${venv_version}: NumPy ${numpy_pin} requires" \
    "Python 3.12 or newer; remove ${venv_dir} and rerun with a newer interpreter (--python)"
fi

# ---------------------------------------------------------------------------------------------
# Packages
# ---------------------------------------------------------------------------------------------
echo "==> Installing python/requirements.txt (wheels only)"
"${venv_python}" -m pip install --disable-pip-version-check --no-input --only-binary=:all: \
  -r "${requirements}" || die "pip install -r ${requirements} failed"

numpy_version="$("${venv_python}" -I -c 'import numpy; print(numpy.__version__)' 2>/dev/null)" ||
  die "numpy cannot be imported from ${venv_dir}"
[[ "${numpy_version}" == "${numpy_pin}" ]] ||
  die "numpy ${numpy_version} is installed in ${venv_dir}, but ${requirements} pins ${numpy_pin}"

echo "python: ${venv_version} (${venv_python})"
echo "venv bootstrap: ${venv_bootstrap}"
echo "numpy: ${numpy_version}"
echo "setup_python: OK"
