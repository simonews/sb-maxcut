# Changelog

All notable changes to this project, one section per delivery. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [F04-R1] - 2026-11-15

Phase 04, part 1 of 2: sequential simulated bifurcation (C0), the Python reference (P0) and T8.

### Added

- `include/sbmc/sb.hpp` and `src/core/sb.cpp` (in `sbmc_core`): `SbAlgorithm`, `SbParams`,
  `SbReplicaResult`, `SbResult`; `pressure_schedule` (p[m] in double, one rounding to float);
  `auto_c0` for `Graph`, `DenseSpinGlass` and `Instance` (int64 sums with overflow checks, then
  double in a fixed order); `sb_sign` and `sb_update` (`SBMC_HD constexpr`, symplectic Euler
  update and inelastic walls), shared by every variant.
- `include/sbmc/sb_sequential.hpp`, `src/cpu/sb_sequential.cpp` and `src/cpu/CMakeLists.txt`:
  static library `sbmc_cpu` (`sbmc::cpu`) with `run_sb_sequential`, the sequential reference C0
  of bSB and dSB in FP32 (compressed rows for a `Graph`, dense INT8 J for a `DenseSpinGlass`
  with n <= 32768; best-cut rule of F04-C1, section 5.3; final check with `cut_direct`; optional
  trace of the first steps); no OpenMP directive and no public function in the step loop.
- `tests/core/test_sb.cpp` and `tests/cpu/test_sb_sequential.cpp` (label `cpu`): reference bits
  of the pressure, of c0 and of the update; the short trajectory and the cases G1-G6 bit for bit
  in the strict presets; every step of G3-G6 recomputed from the specified operations; properties
  of every preset; replica split; independence from the number of threads; preconditions.
- `tests/tools/sbmc_trace.cpp`: tool of the tests (`<build>/tests/tools/sbmc_trace`) that runs C0
  and writes the binary trace `SBMCTRC1` read by T8.
- `tests/data/torus_4x6_unit.rudy` and `tests/data/torus_4x6_pm1.rudy`: 4 x 6 toroidal grids
  with unit and +-1 weights.
- `python/requirements.txt` (`numpy==2.5.4`) and the package `python/sbmc_ref` (P0, float64 and
  exact integers, NumPy only): `model.py` (rudy reader, couplings, W, E, C, `auto_c0`) and
  `sb.py` (`sign`, `pressure_schedule`, `sb_step`, `run_sb`, `best_from_states`).
- `tests/python/test_p0.py` and `tests/python/test_t8.py` (label `python`, skipped with 77 without
  NumPy): P0 alone, and T8 (coherence, one step at a time, free run, best cut on ten cases, plus
  the exit statuses of `sbmc_trace`).
- `scripts/setup_python.sh`: the virtual environment `.venv` of the project with the pinned NumPy
  (venv bootstrap as in `setup_cuda_pypi.sh`; Python 3.12 or newer).

### Changed

- `CMakeLists.txt`: `add_subdirectory(src/cpu)`; interpreter of the Python tests chosen at every
  configure (`SBMC_PYTHON`, then `.venv/bin/python`, then `python3` from the PATH); summary line
  "Python (tests)".
- `src/core/CMakeLists.txt`: `sb.cpp` in `sbmc_core`; `sb.hpp` and `sb_sequential.hpp` in the
  header check; header comment.
- `tests/CMakeLists.txt`: header comment (labels and folders); `test_sb` in the loop of the core
  tests; `test_sb_sequential`; the tool `sbmc_trace`; `test_p0` and `test_t8`. Tests per preset:
  10 (`cpu`, `cpu-strict`), 16 (`cuda`, `cuda-strict`), 13 (`t4`, `t4-strict`).
- `README.md`: status (phase 04, part 1 of 2), requirements, sections "Sequential simulated
  bifurcation" and "Python reference P0 and T8", tests, origin of the reference values,
  repository layout.

### Removed

- Nothing.

## [F03-R1] - 2026-11-15

Phase 03: the core library, the first library of the project (standard C++20 and OpenMP only,
no CUDA types in the public headers).

### Added

- `include/sbmc/error.hpp`: `sbmc::Error` (derived from `std::runtime_error`), `sbmc::IoError`,
  `sbmc::InvalidArgument`.
- `include/sbmc/philox.hpp`: macro `SBMC_HD`; `U32x4`, `U32x2` (with defaulted `operator==`);
  `philox4x32_10` (Philox4x32-10, header-only, `constexpr`, `noexcept`, host and device, no
  intrinsics); `RngDomain` (`couplings` 1, `sb_initial_positions` 2, `sa_initial_spins` 3,
  `sa_acceptance` 4); `stream_block`, `stream_word` (the stream convention, identical to the
  cuRAND sequence); `coupling_from_word`, `position_from_word`.
- `include/sbmc/instance.hpp` and `src/core/instance.cpp`: `Spin`, `Edge`, `Graph`,
  `DenseSpinGlass`, `Instance = std::variant<Graph, DenseSpinGlass>`; `num_spins`,
  `total_weight`, `ising_energy`, `cut_direct` (int64, cut computed directly) and
  `dense_rows_int8` (INT8 rows of J with padding) for `Graph`, `DenseSpinGlass` and `Instance`;
  `coupling` (implicit dense generator on `(min(i, j), max(i, j))`); full precondition checks
  with `InvalidArgument`; OpenMP over the rows for `DenseSpinGlass`, rows generated in groups of
  four.
- `include/sbmc/rudy.hpp` and `src/core/rudy.cpp`: `read_rudy` and `parse_rudy`, canonical
  graph, `IoError` messages `<source>:<line>: <reason>` and `<source>: <reason>`, the first
  error in input order; a file that cannot be opened, or a directory, gives an `IoError` with
  the path.
- `include/sbmc/initial_state.hpp` and `src/core/initial_state.cpp`: `initial_position`
  (`SBMC_HD constexpr`), `InitialState`, `make_initial_state` (replica-major x0, global replica
  indices, OpenMP).
- `src/core/CMakeLists.txt`: INTERFACE target `sbmc_headers` (`sbmc::headers`), static library
  `sbmc_core` (`sbmc::core`), header check `sbmc_header_check` (C++, every preset) and
  `sbmc_header_check_cuda` (nvcc, CUDA presets), generated units that include each public
  header twice.
- Tests: `test_philox` (T11, CPU part), `test_instance` (T1, T5 for dense INT8 rows, T12, int64
  sums, preconditions), `test_rudy` (T6 on example inputs), `test_initial_state` (x0), label
  `cpu`, in `build/<preset>/tests/core/`; `tests/data/four_nodes.rudy`;
  `test_curand_philox` (T11, GPU part: cuRAND, the project's Philox and x0 on the device, label
  `gpu`, in `build/<preset>/tests/gpu/`, listed in `test_cuda_rpath`).
- `README.md`: section "Core library" (instances, rudy reader rules, domains and stream rule,
  cuRAND equivalence, conversions, initial positions, origin of the reference values) and
  section "Tests" (labels, folders, tests per preset).

### Changed

- `CMakeLists.txt`: `add_subdirectory(src/core)` after `include(cmake/SbmcFlags.cmake)`.
- `tests/CMakeLists.txt`: header comment (labels and folders); the four core test executables;
  `test_curand_philox` in the CUDA block, before `test_cuda_rpath`. Tests per preset: 6 (`cpu`,
  `cpu-strict`), 12 (`cuda`, `cuda-strict`), 9 (`t4`, `t4-strict`).
- `README.md`: status (phase 03), test labels moved to the section "Tests", repository layout.

### Removed

- Nothing.

## [F02-R3] - 2026-11-14

Phase 02, correction after the first run on Kaggle: the image's Python has no ensurepip, so
`python3 -m venv` could not create the toolkit's virtual environment.

### Added

- `scripts/setup_cuda_pypi.sh`: fallback for the virtual environment. When `python3 -m venv`
  fails, or an existing environment has no working pip, the incomplete environment is removed
  (only `<prefix>/venv`), recreated with `python3 -m venv --without-pip` and given pip by the
  system pip (`python3 -m pip --python <venv>/bin/python install pip`, pip >= 22.3, checked
  before anything is installed). The summary gains the line "venv bootstrap" (`existing`,
  `ensurepip` or `system pip <version>`).
- `scripts/kaggle_run.sh`: `venv_bootstrap` in the `[toolkit]` section of `summary.txt`.

### Changed

- `scripts/setup_cuda_pypi.sh`: without ensurepip and without a system pip >= 22.3 the script
  stops with an error that names both remedies, without installing anything.
- `README.md`: the fallback, in the toolkit section.

### Removed

- Nothing.

## [F02-R2] - 2026-11-14

Phase 02, corrections after the review of F02-R1: CUDA libraries always from the project's
toolkit, whatever `LD_LIBRARY_PATH` says.

### Added

- `cmake/SbmcFlags.cmake`: INTERFACE target `sbmc_cuda_rpath` (alias `sbmc::cuda_rpath`) with
  the linker option `--disable-new-dtags`, so that executables record the lib-shim as RPATH,
  searched before `LD_LIBRARY_PATH`, instead of RUNPATH; rule: every executable that loads CUDA
  libraries links it.
- `tests/scripts/check_rpath.sh` and the test `test_cuda_rpath` (label `tools`, CUDA builds
  only): every CUDA executable has an RPATH folder with `libcudart.so.13` and no RUNPATH; exit
  code 77 without readelf.
- `tests/smoke/smoke_simt.cu`: compiled and loaded versions of the CUDA runtime and cuBLAS
  (`cuda_runtime_compiled`, `cuda_runtime_loaded`, `cublas_compiled`, `cublas_loaded`) with
  `CUDA_RUNTIME_MATCH` and `CUBLAS_MATCH`; a mismatch prints `SIMT_VERDICT: FAIL` and exits
  with 1, also without a GPU.
- `scripts/kaggle_run.sh`: RPATH and RUNPATH of `smoke_simt` and its compiled and loaded
  versions in `summary.txt`.

### Changed

- `tests/CMakeLists.txt`: `smoke_simt`, `smoke_tile`, `fma_probe_tile_fmad_on` and
  `fma_probe_tile_fmad_off` link `sbmc::cuda_rpath`.
- `README.md`: why CUDA executables use RPATH, the `sbmc::cuda_rpath` rule and
  `test_cuda_rpath`; `smoke_simt` and `summary.txt` descriptions.

### Removed

- Nothing.

## [F02-R1] - 2026-11-14

Phase 02: builds and tests on Kaggle's two Tesla T4 with the PyPI toolkit 13.4.3.

### Added

- `CMakePresets.json`: hidden bases `t4-base` (CUDA on, Tile off, sm_75 only, FMA contraction
  on) and `t4-strict-base` (FMA contraction off).
- `scripts/setup_cuda_pypi.sh`: option `--presets laptop|t4`. `laptop`, the default, writes the
  same `CMakeUserPresets.json` as before (`cuda`, `cuda-strict`); `t4` writes the presets `t4`
  and `t4-strict`, with one build job per CPU, and records the set in `vendor.sbmc.presets`.
- `scripts/kaggle_run.sh`: Kaggle runner. Context, toolkit setup, configure, build from scratch
  and ctest of `t4-strict` and `t4`, `smoke_simt`, environment report, `summary.txt`, secret
  scan and archive; exit codes 0 (passed), 1 (failed), 2 (no usable GPU).
- `scripts/env_report.sh`: `[platform]` section (`KAGGLE_DOCKER_IMAGE`,
  `KAGGLE_KERNEL_RUN_TYPE`), number of GPUs and `nvidia-smi topo -p2p r` / `-p2p w` in `[gpu]`,
  `[communication]` section (mpirun path and version, NCCL version from `nccl.h` or, without
  the header, the library file resolved by `ldconfig -p`).
- `README.md`: section "Running on Kaggle (2 x T4)" with the notebook cell.

### Changed

- `tests/smoke/smoke_simt.cu`: vector addition, cuBLAS SGEMM and cuBLASLt handle run on every
  visible device, with `device<i>_...` lines and a `device<i>_verdict` line per device. The
  existing lines keep their names and report the totals over all devices; `SIMT_VERDICT: PASS`
  only when every device passes.
- `tests/smoke/smoke_common.cuh`: optional verdict of the part being checked
  (`set_part_verdict_key`), printed as FAIL before the final verdict when a check fails.
- `scripts/setup_cuda_pypi.sh`: CUDA 12.x is refused as not supported by the project (decision
  of phase 02); `--presets` together with `--no-presets` is an error.
- `scripts/env_report.sh`: `--preset` documented for `cuda`, `cuda-strict`, `t4`, `t4-strict`.
- `README.md`: status, presets table (`t4`, `t4-strict`), toolkit section, `smoke_simt` and
  tools.

### Removed

- `CMakePresets.json`: hidden bases `cuda12-base` and `cuda12-strict-base` (building with CUDA 12
  is no longer a requirement).
- `README.md`: references to CUDA 12.

## [F01-R2] - 2026-11-13

Phase 01, corrections after the review of F01-R1.

### Added

- `scripts/setup_cuda_pypi.sh`: the `nvdisasm` extra of the metapackage, so `cuobjdump -sass`
  works without manual steps; new checks (`nvdisasm --version`, nvidia-nvjitlink equal to
  nvidia-cuda-nvcc, `cuobjdump -sass` on the Tile kernel of the minimal probe); summary lines
  for nvdisasm and nvidia-nvjitlink.
- `scripts/env_report.sh`: `nvdisasm` version and a `python_packages` block with the `cuda*` and
  `nvidia*` packages of the toolkit's virtual environment.
- `tests/scripts/test_sync_from_zip.sh`: scenario (i), sync in a git worktree (`.git` is a
  file); scenario (a) also checks that `.cache/`, `__pycache__/` and a `*.nsys-rep` report
  survive the sync.
- `README.md`: how to read the SASS of an executable with `cuobjdump -sass`.

### Changed

- `cmake/SbmcFlags.cmake`: no `-Wpedantic` for the host compiler of CUDA files (it reported
  every line directive generated by nvcc); C++ files keep it.
- `scripts/setup_cuda_pypi.sh`: nvidia-nvjitlink pinned to the version of nvidia-cuda-nvcc
  (13.4.92 for CUDA 13.4.3), also on an existing environment with another version.
- `.sync-exclude`: `.git` without trailing slash (a file in git worktrees); new entries
  `.cache/`, `__pycache__/`, `*.nsys-rep`, `*.ncu-rep`.
- `README.md`: toolkit installation (nvdisasm, nvjitlink pin) and synchronisation (ignored local
  files never deleted, manual update of `.sync-exclude` after exit code 2).

### Removed

- Nothing.

## [F01-R1] - 2026-11-13

Phase 01: repository scaffolding, CUDA toolkit setup from PyPI, tools and smoke tests.

### Added

- CMake project `sbmc` 0.1.0 (C++20, CUDA optional) with options `SBMC_ENABLE_CUDA`,
  `SBMC_ENABLE_TILE`, `SBMC_ENABLE_MPI`, `SBMC_ENABLE_NCCL` (both stop the configure: planned for
  phases 16/17) and `SBMC_STRICT_FP`, plus a configuration summary.
- `cmake/SbmcFlags.cmake`: interface target `sbmc::flags` with warnings, Release optimisation
  and explicit FMA contraction (`-ffp-contract` on the host, `--fmad` on the device).
- `CMakePresets.json`: presets `cpu` and `cpu-strict`; hidden bases `cuda-base`,
  `cuda-strict-base`, `cuda12-base`, `cuda12-strict-base`.
- `scripts/setup_cuda_pypi.sh`: installs `cuda-toolkit==13.4.3` from PyPI into a virtual
  environment, creates the library shim, checks the tools and a minimal Tile compilation, and
  generates `CMakeUserPresets.json` with the `cuda` and `cuda-strict` presets.
- `scripts/sync_from_zip.sh`: applies a delivery zip after structure checks, a dry run and a
  confirmation.
- `scripts/env_report.sh`: environment report in `results/env/`.
- Tests: `test_sanity` (doctest: OpenMP, build mode), `test_sync_from_zip` (scenarios a-h),
  GPU smoke tests `smoke_simt`, `smoke_tile`, `fma_probe_tile_fmad_on`, `fma_probe_tile_fmad_off`.
- `third_party/doctest` 2.5.3 (MIT).
- `.gitignore`, `.gitattributes`, `.clang-format`, `.sync-exclude`, `LICENSE` (MIT), `README.md`.

### Changed

- Nothing (first delivery).

### Removed

- Nothing (first delivery).
