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

| Preset        | CUDA | FMA contraction                                    |
|---------------|------|----------------------------------------------------|
| `cpu`         | off  | on (`-ffp-contract=fast`)                          |
| `cpu-strict`  | off  | off (`-ffp-contract=off`)                          |
| `cuda`        | on   | on (`-ffp-contract=fast`, `--fmad=true`)           |
| `cuda-strict` | on   | off (`-ffp-contract=off`, `--fmad=false`)          |

The strict presets exist for the bit-for-bit equality tests of later phases. All compile options
come from the interface target `sbmc::flags` (`cmake/SbmcFlags.cmake`), never from
`CMAKE_CXX_FLAGS` or `CMAKE_CUDA_FLAGS`.

## CUDA toolkit and GPU build

The CUDA toolkit is installed from PyPI with a pinned version, without sudo:

```bash
scripts/setup_cuda_pypi.sh          # options: --version 13.4.3, --prefix DIR, --no-presets
cmake --preset cuda-strict && cmake --build --preset cuda-strict && ctest --preset cuda-strict
cmake --preset cuda        && cmake --build --preset cuda        && ctest --preset cuda
```

The script installs `cuda-toolkit[...]==13.4.3` into a virtual environment under
`~/.local/share/sbmc/cuda-13.4.3/`, creates a library shim (versioned and unversioned names of
the toolkit's shared libraries, which CMake needs to find them), checks nvcc, tileiras,
compute-sanitizer, cuobjdump and a minimal Tile compilation, and writes `CMakeUserPresets.json`.
That file holds the machine-specific paths, is ignored by git and defines the visible presets
`cuda` and `cuda-strict` on top of the hidden ones of `CMakePresets.json`; rerun the script
after moving the toolkit. IDEs that support CMake presets (CLion included) read both files.

CUDA code is compiled for sm_75 (Kaggle T4) and sm_89 (RTX 4050 Laptop); CUDA Tile targets only
for sm_89, with `-enable-tile -std=c++20`. The hidden presets `cuda12-base` and
`cuda12-strict-base` (CUDA 12, Tile off, sm_75) are reserved for the Kaggle builds of a later
phase.

Test labels: `cpu`, `tools`, `gpu`, `tile`, `probe` (for example `ctest --preset cuda -L tile`).
A test that cannot run on the current machine (no GPU, missing tool) exits with 77 and CTest
reports it as skipped. The GPU smoke tests, built in `build/<preset>/tests/smoke/`, are:

- `smoke_simt`: device report, vector addition, exact cuBLAS SGEMM, cuBLASLt handle;
- `smoke_tile`: CUDA Tile vector addition and INT8 `ct::mma` product checked against int64;
- `fma_probe_tile_fmad_on` / `fma_probe_tile_fmad_off`: whether FMA contraction happens in Tile
  code and whether `--fmad=false` prevents it.

## Tools

- `scripts/sync_from_zip.sh <sb-maxcut_FNN_RN.zip>`: applies a delivery zip to a clean working
  tree. It checks the zip (single `sb-maxcut/` folder, sentinel files, no excluded path, no file
  over 5 MiB), requires the same `.sync-exclude` as the repository and its consistency with
  `.gitignore`, shows a dry run and asks for confirmation. Exit codes: 0 applied, 1 failed
  checks, 2 different `.sync-exclude`, 3 cancelled. Paths in `.sync-exclude` (build trees, IDE
  folders, `CMakeUserPresets.json`, `results/`, ...) are never copied nor deleted.
- `scripts/env_report.sh [--preset cuda]`: writes `results/env/env_<host>_<timestamp>.txt`
  (git state, OS and WSL, CPU, memory, tool and toolkit versions, GPU, `smoke_simt` output).
- `scripts/setup_cuda_pypi.sh`: see above.

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
