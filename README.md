# sb-maxcut: simulated bifurcation on GPU for Max-Cut

High Performance Computing project. Max-Cut is solved as the minimisation of an Ising energy
with simulated bifurcation (ballistic bSB and discrete dSB), an algorithm that updates all spins
at once and whose dominant cost is the product J·S between the coupling matrix and the signs of
many independent replicas. The project compares CPU (OpenMP) and GPU implementations of that
product and of the update step (cuBLAS FP32, cuBLASLt INT8, a fused CUDA Tile kernel, a fused
classic CUDA kernel, a bitwise variant, a sparse variant), with a multi-GPU extension on two
Kaggle T4 GPUs, and validates them with bit-for-bit identical dSB trajectories.

## Status

Phase 01, scaffolding: build system, CUDA toolkit setup, working tools and smoke tests.
Phase 02: builds and tests on Kaggle's two Tesla T4 with the same toolkit as the laptop.
No algorithm of the project is implemented yet.

## Requirements

- WSL2 with Ubuntu 24.04 (the CPU code uses only standard C++20 and OpenMP)
- GCC 13, CMake 3.28 or newer, Ninja
- Python 3 with `venv` (`sudo apt install python3-venv`): used only to install the CUDA toolkit
- An NVIDIA driver that supports CUDA 13.4 (`nvidia-smi` in WSL reports "CUDA Version" 13.4 or
  newer)
- For the tools and their test: git, rsync, unzip and zip (`sudo apt install rsync unzip zip`)

## CPU build and tests

```bash
cmake --preset cpu-strict
cmake --build --preset cpu-strict
ctest --preset cpu-strict
```

Configure presets (build trees in `build/<preset>/`, Release, Ninja):

| Preset        | CUDA                            | FMA contraction                           |
|---------------|---------------------------------|-------------------------------------------|
| `cpu`         | off                             | on (`-ffp-contract=fast`)                 |
| `cpu-strict`  | off                             | off (`-ffp-contract=off`)                 |
| `cuda`        | on: sm_75 and sm_89, Tile sm_89 | on (`-ffp-contract=fast`, `--fmad=true`)  |
| `cuda-strict` | on: sm_75 and sm_89, Tile sm_89 | off (`-ffp-contract=off`, `--fmad=false`) |
| `t4`          | on: sm_75 only, no Tile         | on (`-ffp-contract=fast`, `--fmad=true`)  |
| `t4-strict`   | on: sm_75 only, no Tile         | off (`-ffp-contract=off`, `--fmad=false`) |

`cpu` and `cpu-strict` are in `CMakePresets.json`. The CUDA presets are written by
`scripts/setup_cuda_pypi.sh` into `CMakeUserPresets.json` (next section): `cuda` and
`cuda-strict` for the laptop, `t4` and `t4-strict` for Kaggle.

The strict presets exist for the bit-for-bit equality tests of later phases. All compile options
come from the interface target `sbmc::flags` (`cmake/SbmcFlags.cmake`), never from
`CMAKE_CXX_FLAGS` or `CMAKE_CUDA_FLAGS`.

## CUDA toolkit and GPU build

The CUDA toolkit is installed from PyPI with a pinned version, without sudo:

```bash
scripts/setup_cuda_pypi.sh   # options: --version 13.4.3, --prefix DIR, --presets laptop|t4,
                             #          --no-presets
cmake --preset cuda-strict && cmake --build --preset cuda-strict && ctest --preset cuda-strict
cmake --preset cuda        && cmake --build --preset cuda        && ctest --preset cuda
```

The script installs `cuda-toolkit[...]==13.4.3`, nvdisasm included, into a virtual environment
under `~/.local/share/sbmc/cuda-13.4.3/`. It pins `nvidia-nvjitlink`, which the metapackage only
requires as a minimum version, to the version of nvcc (13.4.92 for 13.4.3), also when it reruns
on an existing environment. It creates a library shim (versioned and unversioned names of the
toolkit's shared libraries, which CMake needs to find them), checks nvcc, tileiras,
compute-sanitizer, cuobjdump, nvdisasm, the nvjitlink pin, a minimal Tile compilation and its
disassembly, and writes `CMakeUserPresets.json`.
That file holds the machine-specific paths, is ignored by git and defines visible presets on top
of the hidden ones of `CMakePresets.json`: with `--presets laptop` (the default) `cuda` and
`cuda-strict`, built with 8 jobs; with `--presets t4` `t4` and `t4-strict`, built with one job
per CPU. `--presets` and `--no-presets` cannot be combined. Rerun the script after moving the
toolkit. IDEs that support CMake presets (CLion included) read both files.

With the `cuda` presets (hidden bases `cuda-base` and `cuda-strict-base`), CUDA code is compiled
for sm_75 (Kaggle T4) and sm_89 (RTX 4050 Laptop); CUDA Tile targets only for sm_89, with
`-enable-tile -std=c++20`. The `t4` presets (hidden bases `t4-base` and `t4-strict-base`)
compile CUDA code for sm_75 only and have no Tile targets: they are the Kaggle builds.

CUDA executables find the toolkit's libraries through RPATH, not RUNPATH. The linker writes
RUNPATH by default, and the dynamic loader searches the folders of `LD_LIBRARY_PATH` before
RUNPATH: on Kaggle that variable lists `/usr/local/cuda/lib64`, the system toolkit (CUDA 13.0),
whose runtime and cuBLAS would replace those of the project's toolkit. RPATH comes before
`LD_LIBRARY_PATH`, so the library shim recorded at link time wins. The INTERFACE target
`sbmc::cuda_rpath` (`cmake/SbmcFlags.cmake`) adds only the linker option `--disable-new-dtags`.
Rule: every executable that loads CUDA libraries links `sbmc::cuda_rpath`, directly or through a
library that propagates it. The test `test_cuda_rpath` (label `tools`, CUDA builds only) runs
`tests/scripts/check_rpath.sh` on the CUDA executables of the build: each one needs an RPATH
folder with `libcudart.so.13` and no RUNPATH. As a second guard, `smoke_simt` fails when the CUDA
runtime or cuBLAS it loads differ from the headers it was compiled with.

Test labels: `cpu`, `tools`, `gpu`, `tile`, `probe` (for example `ctest --preset cuda -L tile`).
A test that cannot run on the current machine (no GPU, missing tool) exits with 77 and CTest
reports it as skipped. The GPU smoke tests, built in `build/<preset>/tests/smoke/`, are:

- `smoke_simt`: loaded CUDA runtime and cuBLAS against the headers of the build
  (`CUDA_RUNTIME_MATCH`, `CUBLAS_MATCH`; a mismatch fails with exit code 1, GPU or not), device
  report, P2P matrix (`cudaDeviceCanAccessPeer`, two devices or more), then on every visible
  device vector addition, exact cuBLAS SGEMM and cuBLASLt handle, with a `device<i>_verdict` line
  per device; `SIMT_VERDICT: PASS` only when every device passes;
- `smoke_tile`: CUDA Tile vector addition and INT8 `ct::mma` product checked against int64;
- `fma_probe_tile_fmad_on` / `fma_probe_tile_fmad_off`: whether FMA contraction happens in Tile
  code and whether `--fmad=false` prevents it.

### Reading the SASS of an executable

`cuobjdump -sass` prints the machine code (SASS) of every kernel of an executable: the SIMT
kernels and, in a separate section, the CUDA Tile kernels, each under a `Function : <name>`
line. The toolkit root comes from `CMakeUserPresets.json`; cuobjdump uses the nvdisasm installed
next to it, so no PATH change is needed. From the repository root:

```bash
TK=$(sed -n 's/^[[:space:]]*"toolkit-root":[[:space:]]*"\(.*\)",\{0,1\}[[:space:]]*$/\1/p' CMakeUserPresets.json)
"$TK/bin/cuobjdump" -sass build/cuda/tests/smoke/fma_probe_tile_fmad_on
```

For example, to count the FFMA, FMUL and FADD instructions of each kernel of an FMA probe:

```bash
"$TK/bin/cuobjdump" -sass build/cuda/tests/smoke/fma_probe_tile_fmad_on |
  awk '/Function :/ { fn = $NF; seen[fn] = 1 }
       fn != "" && /[[:space:]]FFMA[[:space:].]/ { ffma[fn]++ }
       fn != "" && /[[:space:]]FMUL[[:space:].]/ { fmul[fn]++ }
       fn != "" && /[[:space:]]FADD[[:space:].]/ { fadd[fn]++ }
       END { for (f in seen) printf "%s FFMA=%d FMUL=%d FADD=%d\n", f, ffma[f], fmul[f], fadd[f] }'
```

## Running on Kaggle (2 x T4)

`scripts/kaggle_run.sh` builds, tests and reports the project in one go on a Kaggle notebook
with two Tesla T4. The branch or tag to test must be published on GitHub first.

Notebook settings: accelerator **GPU T4 x2**, **Internet** on. Then paste this cell, with `REF`
set to the branch or tag to test, and run it:

```bash
%%bash
set -euo pipefail
REF=fase-02
mkdir -p /tmp/sbmc && cd /tmp/sbmc && rm -rf sb-maxcut
git clone -q https://github.com/simonews/sb-maxcut && cd sb-maxcut
git checkout -q "$REF"
scripts/kaggle_run.sh --ref "$REF"
```

The script installs the toolkit with `scripts/setup_cuda_pypi.sh --presets t4`, configures,
builds from scratch and tests `t4-strict` and then `t4`, runs `smoke_simt` on both GPUs and
writes the environment report. Everything goes to `/kaggle/working/sbmc_<REF>_<YYYYmmdd-HHMMSS>/`
(UTC time), with the archive `/kaggle/working/sbmc_<REF>_<YYYYmmdd-HHMMSS>.tar.gz` next to it:
both are in the notebook's output files (`/kaggle/working`), where the archive can be
downloaded. The folder holds:

- `summary.txt`, also printed at the end of the cell: commit, nvcc and nvidia-nvjitlink
  versions, outcome of configure, build and ctest for each preset with the names of the passed,
  failed and skipped tests, RPATH and RUNPATH of `smoke_simt`, its compiled and loaded runtime
  and cuBLAS versions and its verdict lines, P2P matrix, environment report;
- the log of every step: `1_context.log`, `2_setup.log`, `3_<preset>_configure.log`,
  `3_<preset>_build.log`, `3_<preset>_ctest.log`, `5_env_report.log`;
- the output of `smoke_simt` (`4_smoke_simt.txt`) and the environment report
  (`env_<host>_<timestamp>.txt`).

Exit codes: 0 everything passed and `smoke_simt` printed `SIMT_VERDICT: PASS`; 1 a step or a
test failed (the cell also prints the last lines of the failed logs); 2 no usable GPU
(`smoke_simt` exited with 77). With 1 or 2 the cell ends with an error, and the folder and the
archive are there all the same. A test skipped by ctest (`test_sync_from_zip` when rsync, unzip
or zip is missing) is not a failure and is listed in `summary.txt`. Options: `--ref LABEL`
(label in the file names, default the short hash of HEAD), `--out DIR` (default
`/kaggle/working`, or `./kaggle-out` where that folder does not exist). In the clone, the script
writes only `build/`, `results/env/` and `CMakeUserPresets.json` (and `kaggle-out/` when that is
the output folder).

Notes:

- A session consumes GPU quota from the moment it starts, whether a cell is running or not:
  start it when the branch is published, and stop it once the archive is downloaded.
- Kaggle's system toolkit (CUDA 13.0 in `/usr/local/cuda`) is not used: the presets point CMake
  to the toolkit installed from PyPI, CUDA 13.4.3 as on the laptop. Kaggle's driver (580,
  "CUDA Version: 13.0") runs CUDA 13.x executables thanks to CUDA's minor version
  compatibility, with a limited set of features; the executables carry machine code for sm_75,
  so no PTX JIT compilation is needed.
- No environment variable is printed or saved, apart from `KAGGLE_DOCKER_IMAGE` and
  `KAGGLE_KERNEL_RUN_TYPE` in the environment report. Before making the archive, the script
  searches the files of the run for the values of the variables whose name contains TOKEN,
  SECRET, KEY or PASSWORD (Kaggle's tokens among them), without printing them: if one is found,
  the run fails and no archive is made.

## Tools

- `scripts/sync_from_zip.sh <sb-maxcut_FNN_RN.zip>`: applies a delivery zip to a clean working
  tree. It checks the zip (single `sb-maxcut/` folder, sentinel files, no excluded path, no file
  over 5 MiB), requires the same `.sync-exclude` as the repository and its consistency with
  `.gitignore`, shows a dry run and asks for confirmation. Exit codes: 0 applied, 1 failed
  checks, 2 different `.sync-exclude`, 3 cancelled. Paths in `.sync-exclude` (build trees, IDE
  folders, `CMakeUserPresets.json`, `results/`, ...) are never copied nor deleted; this covers
  the local files that git ignores and that `.sync-exclude` lists (`.cache/`, `__pycache__/`,
  `*.nsys-rep`, `*.ncu-rep`), and `.git` also when it is a file (git worktrees). When a delivery
  changes `.sync-exclude`, the script stops with exit code 2 and shows the diff: copy the new
  `.sync-exclude` by hand, commit it, then run the script again.
- `scripts/env_report.sh [--preset NAME]` (`cuda` by default; also `cuda-strict`, `t4`,
  `t4-strict`): writes `results/env/env_<host>_<timestamp>.txt` (git state, OS and WSL, Kaggle
  image and run type, CPU, memory, tool and toolkit versions, GPU with the number of GPUs and the
  P2P status, MPI and NCCL versions, output of `build/<preset>/tests/smoke/smoke_simt`).
- `scripts/setup_cuda_pypi.sh`: see above.
- `scripts/kaggle_run.sh [--ref LABEL] [--out DIR]`: see "Running on Kaggle (2 x T4)".

Files larger than 5 MB never go into git. `.gitignore` cannot express this rule:
`scripts/sync_from_zip.sh` enforces it on every delivery.

## Repository layout (planned)

```
CMakeLists.txt, CMakePresets.json, cmake/   build system
include/sbmc/        public headers
src/core/            G-set reader, Ising model, int64 cut and energy, Philox, TTS statistics
src/cpu/             sequential reference, OpenMP SB, simulated annealing, brute force
src/gpu/             cuBLAS, cuBLASLt, bitwise and cuSPARSE products, update kernels
src/tile/            fused CUDA Tile kernel (sm_89 only)
src/mgpu/            MPI and NCCL multi-GPU drivers
apps/                command-line programs
tests/               unit, integration and smoke tests (ctest)
python/              NumPy reference, analysis, plots
bench/configs/       experiment configurations
data/                G-set and best known values with sources
results/             CSV files and metadata (raw data outside git)
scripts/             setup, tools, measurement campaigns
docs/                report sources and figures
third_party/         vendored dependencies
```

Only `cmake/`, `scripts/`, `tests/` and `third_party/` exist in phase 01.

## License and third-party code

MIT License, see `LICENSE`. Vendored dependency: doctest 2.5.3 (MIT), single header in
`third_party/doctest/`; versions, sources and checksums in `third_party/README.md`.
