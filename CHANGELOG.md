# Changelog

All notable changes to this project, one section per delivery. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

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
