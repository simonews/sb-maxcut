# Changelog

All notable changes to this project, one section per delivery. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [F08-R1] - 2026-11-28

Phase 08, part 1 of 3: the results of phase 07 in the README and the probes of the GPU
environment on the laptop. No change to the code, to CMake, to the presets, to the tests, to the
results schema or to the campaigns.

### Added

- `scripts/laptop/f08r1.sh`: `run "$NOTES"` checks the repository and the toolkit (nvcc
  V13.4.92), builds from scratch and tests `cpu-strict`, `cpu`, `cuda-strict` and `cuda` (33, 33,
  39 and 39 tests, none skipped, no warning), records the device from `smoke_simt`, probes every
  field of nvidia-smi of WSL and of Windows (`VALUE`, `NA`, `INVALID`, `NOT_FOUND`), samples both
  sources every 100 ms at rest and under `ctest -L gpu`, reads the registry value
  `EnableInterface` of the debugger interface (read only) and runs compute-sanitizer memcheck on
  `smoke_simt` and `test_curand_philox`; `sanitizer` repeats the last part. Outputs in
  `results/f08/env/`, logs in `~/sbmc_logs/f08r1/`, a summary of about 20 lines with
  `F08R1_VERDICT`.

### Changed

- `README.md`: Status (phase 07 complete, phase 08 part 1); "Results" subsections for E2 (the
  choices at level 1000 and the new edge b = 2 of SA on G2), E3 (the most severe common level,
  SA/SB) and E9 (the last point of every series); the external check opens with its result (SA
  2.3-2.7 times faster per proposal than dwave-samplers, z and KS PASS) and its values; the new
  subsection "GPU environment probes (phase 08)"; the script in "Tools".

## [F07-R4e] - 2026-11-27

Correction before E9: on the laptop `scripts/laptop/f07r4.sh run` stopped at the bitwise
comparison, since `sbmc_run.exe` with `--instance`, `--out` and `--replica-out` under
`\\wsl.localhost\Ubuntu-24.04\...` (from `wslpath -w`) exited with 1, "cannot compare the two
paths": `std::filesystem::absolute` and `weakly_canonical` of the libstdc++ of MinGW fail on a UNC
path. The campaign would have stopped the same way. No change to the solvers, to the results
schema, to the series and loads of E9 or to the Windows build.

### Changed

- `apps/sbmc_run.cpp`: `--out` and `--replica-out` are compared with `absolute` and
  `weakly_canonical` as before and, when either fails, in their lexically normal forms (of the
  absolute paths, or of the given ones), ignoring case on Windows; a path that cannot be resolved
  is no longer an error by itself (on Linux the only visible difference: a loop of symbolic links
  in the path is reported by the check of the folder, still exit status 1).
- `python/sbmc_campaign.py`: `WindowsCopy` replaces `WindowsPaths`, one mechanism for the series
  of platform `windows` and for `sbmc_e9.py bitwise`: `DEST` from the executable
  (`DEST/build-<preset>/...`), translated once with `wslpath -w` and required on a drive; inputs
  from the same relative path in `DEST/src` (an error before the run when missing); `--out` and
  `--replica-out` in `DEST/runs/<campaign>/`, deleted before each run and moved into the repository
  after exit status 0 (kept there otherwise); the process in `DEST/src`; before the first
  configuration of a Windows executable, `--version` and `DEST/src/.sbmc_build_info` checked
  against `git rev-parse HEAD`.
- `python/sbmc_e9.py`: `bitwise` through `WindowsCopy` (work folder `DEST/runs/bitwise`, both
  Windows builds checked against HEAD).
- Tests: `test_sbmc_campaign_series` (inputs from the copy, outputs in `runs/`, no UNC argument,
  stale work files deleted, a failed configuration and the resumed run, a missing file, another
  commit, a UNC `DEST`), `test_sbmc_e9` (`bitwise` through the copy, another commit),
  `test_sbmc_run` (the loop of symbolic links: the folder, not the comparison; the same file in it
  written in two ways: exit status 2).
- `README.md`: the fact and the correction in "E9 on Windows native", the comparison of `sbmc_run`,
  the runner, the tests.

## [F07-R4d] - 2026-11-23

Correction of F07-R4b and F07-R4c: on the laptop the build of `win-cpu-strict` failed once
(`FAILED: [code=1] apps/sbmc_bruteforce.exe`, `file COPY_FILE failed to copy`), since the copies
of the runtime DLLs of `sbmc_run`, `sbmc_bruteforce` and `sbmc_spectrum` write the same files in
`apps/` and Ninja ran two of them in parallel; on Windows a file being written by one copy cannot
be opened by another. No change to the binaries, to their options or to the list of the DLLs.

### Changed

- `cmake/SbmcWindowsRuntime.cmake`: the links of the targets that copy (their POST_BUILD command
  is part of the link step in Ninja) go to the job pool `sbmc_runtime_copy` of depth 1
  (`JOB_POOL_LINK`), so the copies never run in parallel.
- `cmake/sbmc_copy_runtime.cmake`: a copy that fails stops the build with the DLL, the source,
  the destination and the error (`file(COPY_FILE ... RESULT)`).
- `scripts/windows/build.ps1`: when a build fails, the first line `FAILED:` and the first line
  with `error` of the log, with the lines after it up to an empty line (at most 5), go to
  `build_summary.txt`.
- `README.md`: the race and its correction, in "E9 on Windows native".

## [F07-R4c] - 2026-11-23

Correction of F07-R4b: on the laptop the presets `win-cpu` and `win-cpu-strict` configured, but
the link failed (`GOMP_parallel`, `omp_get_thread_num` and other symbols of libgomp undefined),
since with the MinGW-w64 of CLion FindOpenMP finds no library (`OpenMP_CXX_LIB_NAMES` empty) and
passes `-fopenmp` only to the compilation.

### Changed

- `CMakePresets.json`: `win-base` sets `CMAKE_EXE_LINKER_FLAGS` to `-fopenmp`, as in the test
  made by hand on the laptop; g++ then links libgomp and its threads.
- `README.md`: the reason, in the build on Windows of "E9 on Windows native".

## [F07-R4b] - 2026-11-23

Correction of F07-R4 before the campaign E9 (decision W5): on the laptop the presets `win-cpu`
and `win-cpu-strict` did not configure, since with the MinGW-w64 of CLion (g++ 15.2.0) the static
link of OpenMP fails (`__intrinsic_setjmpex` and `__ms_vsnprintf` undefined) and FindOpenMP does
not find OpenMP. The Windows builds are now linked dynamically, with the runtime DLLs of the
toolchain next to the executables. No change to the C++ code, to Linux or to the campaigns.

### Added

- `cmake/SbmcWindowsRuntime.cmake` (Windows only): the four runtime DLLs of the toolchain
  (`libgomp-1.dll`, `libwinpthread-1.dll`, `libstdc++-6.dll`, `libgcc_s_seh-1.dll`) checked at
  configure time, next to the compiler or from `-print-file-name` (a missing DLL is an error that
  names it); `sbmc_copy_runtime_dlls()`.
- `cmake/sbmc_copy_runtime.cmake`: after every link, the DLLs imported by the executable
  (`objdump -p`) and held by the toolchain, followed through their own imports, copied next to
  it; used for `sbmc_run`, `sbmc_bruteforce`, `sbmc_spectrum` and `sbmc_sa_thresholds`.

### Changed

- `CMakePresets.json`: `win-base` without `-static` and without the cache variables of OpenMP
  (FindOpenMP finds `-fopenmp`); the display names without "static".
- `scripts/windows/build.ps1`: per preset, the line `runtime <preset> dlls=<DLL,...>
  standalone=PASS|FAIL` (the imported DLLs of `sbmc_run.exe` and `sbmc_sa_thresholds.exe` next
  to them or of the system, then `sbmc_run.exe --version` with a PATH without CLion), part of
  `WIN_BUILD`, details in `<preset>_runtime.log`; the empty lines of stderr written to the logs as
  empty lines (not as `System.Management.Automation.RemoteException`); configure with
  `--fresh`, since a reconfigure keeps the `-static` of F07-R4 in `CMakeCache.txt` (verified in
  the sandbox), so every run builds from scratch.
- `scripts/laptop/f07r4.sh`: prints the runtime lines of `build_summary.txt` and requires
  `standalone=PASS` in both presets for `WIN_BUILD`.
- `CMakeLists.txt`, `apps/CMakeLists.txt`, `tests/CMakeLists.txt`: the module and its calls, on
  Windows only.
- `README.md`: the build on Windows (dynamic link, the copy of the DLLs and why, the runtime
  check), Status, limits and layout.

## [F07-R4] - 2026-11-23

Phase 07, part 4 of 4: E9, the OpenMP scaling of C1 and C2 on Windows native (MinGW-w64 GCC of
CLion, built from a copy of the commit) and in WSL2 as the control, with process masks by
efficiency class, the runner extended to series of executables, masks and threads, the analysis
and the bitwise comparison of the two builds. On Linux the rows and the output of `sbmc_run` are
unchanged; no change to the library, to the results schema or to the replica file.

### Added

- `apps/cpu_topology.hpp` (portable): the rules `p1`, `p`, `e` and `all` of `--affinity` as masks
  of the logical processors (index, core, efficiency class), `cpulist` and `cpu_classes`.
- `apps/win_platform.hpp`, `apps/win_platform.cpp` (Windows only, the only code with Windows API):
  the logical processors from `GetLogicalProcessorInformationEx`, the process mask
  (`SetProcessAffinityMask`, read back), the columns `cpu_affinity`, `cpu_classes`, `host` and `os`
  on Windows, the lines of `--print-topology`, stdout in binary mode.
- `sbmc_run --affinity p1|p|e|all` and `--print-topology` (Windows only, elsewhere exit status 2)
  and `--version` (every platform).
- `CMakePresets.json`: the presets `win-cpu` and `win-cpu-strict` (configure, build, test): Ninja,
  g++ from the PATH, `-static` with libgomp linked statically, `binaryDir`
  `${sourceParentDir}/build-<preset>`; on Windows only the C++ tests of the label `cpu`.
- `scripts/windows/stage.sh` (the copy of HEAD with `.sbmc_build_info` and K2000),
  `scripts/windows/build.ps1` (PowerShell 5.1: the toolchain of CLion, the two presets built and
  tested, `build_summary.txt`, `WIN_BUILD`), `scripts/laptop/f07r4.sh` (`stage` and `run`: the
  whole procedure of the laptop with its verdicts).
- `bench/configs/f07_e9.toml`: G1 and K2000, SA and dSB with the parameters of E2, R 1024, five
  series (`win-p1`, `win-p`, `win-e`, `win-all`, `wsl`), 136 configurations.
- `python/sbmc_e9.py`: `analyze` (`e9_summary.csv`, the scaling plots, the checks of
  consistency) and `bitwise` (Windows build against WSL build, with the comparison of the SA
  thresholds on a difference).
- Tests: `test_cpu_topology` (`cpu`), `test_sbmc_e9` (`python`), `test_sbmc_campaign_series`
  (`apps`), the tool `tests/tools/sbmc_sa_thresholds`; 33 tests with `cpu`, 39 with `cuda`, 36
  with `t4`, 18 with the Windows presets.

### Changed

- `python/sbmc_campaign.py`: `[[series]]` (`id`, `sbmc_run`, `platform`, `affinity`, `threads`,
  `base`) with the ids `<id>_<series>_T<threads>`; the Windows series get `--affinity` and the
  paths translated with `wslpath -w`; `threads` of `[campaign]` and stages are not allowed with
  series. Configurations without series run as before.
- `cmake/sbmc_build_info.cmake`: `.sbmc_build_info` in the source folder when git gives no hash
  (the copy for Windows); a malformed file is an error. `.gitignore`: `.sbmc_build_info`.
- `apps/CMakeLists.txt`, `tests/CMakeLists.txt`: the Windows sources and the new tests and tool.
- `README.md`: Status, Requirements, the new options of `sbmc_run`, the runner with series, the
  columns of the environment on Windows, the campaign table, "E9 on Windows native" (procedure,
  series, analysis, bitwise comparison, limits), Tests, Tools and the layout.

## [F07-R3b] - 2026-11-22

Correction of phase 07, part 3 (decision N1): the grid of E2 extended to short S and M on G2 and
G12 for both methods, where the choices of F07-R3 fell on the shortest steps, and the steps of E3
to match. Configurations and documentation only: no executable line changes.

### Changed

- `bench/configs/f07_e2.toml`: S and M in {125, 250, 500, 1000, 2000, 4000, 8000} on G2 and in
  {64, 125, ..., 8000} on G12, the same for SA and dSB; 42 configurations per method on G2, 48 on
  G12, 24 on G15 and 24 on the dense instance, 138 per method and 276 in all, 278 with the stage
  `target`. The 194 configurations of F07-R3 keep their ids; the runner runs only the 84 new ones.
- `bench/configs/f07_e3.toml`: the steps of the grid of E2 of the family (7 on G1, 8 on G11, 4 on
  G14 and K2000): 46 configurations in all.
- `python/sbmc_e3.py`: the docstring of the module only ("the steps" of the grid of E2, not "the
  four steps").
- `README.md`: Status; the table of the campaigns (E2 2 + 276, E3 46); "E2: tuning" (the budget,
  the grids, the factor covered, the paragraph on the extension with the facts of the laptop and of
  the pilot run, and why the other edges are not extended); the external check and E3 after the
  extension. With the extension the index of a configuration in the order of the grid changes, and
  with it the seed of its bootstrap: the bootstrap intervals of the configurations already measured
  change a little, within the noise of the bootstrap (more when the successes are few); P_s,
  T_batch, TTS99 and the Wilson intervals do not.

## [F07-R3] - 2026-11-22

Phase 07, part 3 of 4: the resumable runner of the measurement campaigns with TOML configurations,
the tuning E2 of dSB and SA, the external check of SA with dwave-samplers and E3 on the CPU, with
their analysis tools and plots. Python tools, configurations and documentation only: no C++
code, no change to the results schema or to the replica file.

### Added

- `python/sbmc_campaign.py`: the runner (`run`, `status`, `manifest`): configurations in TOML
  validated in full (exit status 2), the scales h and w of the instances and the line of
  `sbmc_spectrum` in `instances.csv`, stages before the grid (the target best-found), one pair of
  files (results and replicas) per configuration, resuming (complete configurations skipped,
  incomplete ones removed and repeated), stop at the first failure of `sbmc_run`, SIGINT and
  SIGTERM handled, a lock, `campaign.log`, the SHA-256 manifest of `raw/` and its versioned copy,
  `--dry-run` and `--stage`.
- `python/sbmc_e2.py`: the analysis of E2 (`e2_summary.csv`, `selected.csv` with the choice per
  instance and method, the plots of sensitivity and of the chosen configurations).
- `python/sbmc_e3.py`: the analysis of E3 (`e3_summary.csv`, `e3_tts.csv` with the transferred
  configurations, the ratio SA/SB and its conservative interval, the common level;
  `quality_equal_time.csv`; plots 1 and 2 of section 14.5).
- `python/sbmc_dwave.py`: the external check of SA against the `SimulatedAnnealingSampler` of
  dwave-samplers (z of the means, two-sample Kolmogorov-Smirnov, nanoseconds per proposal);
  `DWAVE_VERDICT: SKIP` without dwave-samplers.
- `python/requirements-dwave.txt`: dwave-samplers 1.8.0, dimod 0.12.23, networkx 3.7 (optional).
- `bench/configs/f07_e2.toml`, `f07_dwave.toml`, `f07_e3.toml`: the campaigns of phase 07.
- Tests: `tests/python/test_sbmc_campaign.py` (label `apps`), `test_sbmc_e2.py`, `test_sbmc_e3.py`
  and `test_sbmc_dwave.py` (label `python`). 30 tests with the `cpu` presets, 36 with `cuda`, 33
  with `t4`.

### Changed

- `python/requirements.txt`: matplotlib 3.11.2 and its dependencies, pinned, with a header
  comment.
- `scripts/setup_python.sh`: option `--dwave` (also `python/requirements-dwave.txt`, from wheels);
  the summary shows matplotlib and, with `--dwave`, dwave-samplers.
- `tests/CMakeLists.txt`: the four tests, with their folders in the build tree.
- `README.md`: Status part 3 of 4; the section "Campaigns" (the runner, E2, the external check,
  E3); the measurement of the laptop of F07-R2 under "Layouts of the block"; the Python
  requirements; the tests; the repository layout.

## [F07-R2] - 2026-11-22

Phase 07, part 2 of 4: the estimate of the spectrum of J with the Lanczos method, the automatic dt
and c0 of simulated bifurcation, `sbmc_spectrum`, C2 on dense J with the update of the accepting
replicas only, the final check on dense rows in product form. No result of C0, C1, C2 or of the
reference of SA changes: every field of `SbResult` and `SaResult` but the times is that of F07-R1.

### Added

- `include/sbmc/spectrum.hpp`, `src/core/spectrum.cpp`: `SpectrumEstimate`,
  `kSpectrumIterations` (100), `estimate_spectrum` for `Graph`, `DenseSpinGlass` and `Instance`
  (Lanczos in double with full reorthogonalization, a start vector with continuous components from
  the Philox domain 5, restarts after a breakdown, extremes of the tridiagonal matrix by bisection
  with Sturm counts; the same bits for every number of threads), `spectral_dt` (with a factor) and
  `spectral_c0`.
- `apps/sbmc_spectrum.cpp`: `sbmc_spectrum`, one line `SPECTRUM ...` with lambda_min, lambda_max,
  the iterations, the restarts, dt_auto, c0_lambda, c0_sigma and the time.
- `apps/sbmc_run.cpp`: `--dt auto`, `--dt-factor F` (only with `--dt auto`) and `--c0 lambda`; the
  spectrum is estimated once per invocation, inside `t_setup_ms`, only when dt or c0 is derived
  from it; the columns `dt_rule` (`auto`), `c0_rule` (`lambda`), `lambda_min` and `lambda_max`
  filled; exit status 1, before any run, when J = 0.
- Tests: `tests/core/test_spectrum.cpp` (label `cpu`) and `tests/python/test_sbmc_spectrum.py`
  (label `apps`); the cases "detail: cut_dense_rows_unchecked, product form" of `test_instance`
  and "C2: dense J, equal to C0" of `test_sa_openmp`; check 14 of `test_sbmc_run`. 26 tests with
  the `cpu` presets, 32 with `cuda`, 29 with `t4`.

### Changed

- `include/sbmc/philox.hpp`: `RngDomain::spectrum_start` = 5, in the table of the domains.
- `src/cpu/sa_openmp.cpp`: on dense J (a dense `Graph` or a `DenseSpinGlass`) the state of a
  block in lane-major order, with ld = n rounded up to 64 bytes, and the update of the fields only
  in the lanes that accept a proposal; the initial fields as sums of rows of J; the path with
  compressed rows unchanged; both block functions on a 64-byte boundary; the header comment with
  the two layouts, the rule (always lane-major on dense rows) and the measurement of the sandbox.
  The results are those of F07-R1 bit for bit.
- `src/core/instance.cpp`: the inner loop of `detail::cut_dense_rows_unchecked` in product form,
  (S - R) / 2, which vectorizes with AVX2; signature, contract and values unchanged.
- `src/core/CMakeLists.txt`, `apps/CMakeLists.txt`, `tests/CMakeLists.txt`: the new files and
  targets.
- `tests/core/test_instance.cpp`, `tests/cpu/test_sa_openmp.cpp`, `tests/python/test_sbmc_run.py`:
  the new cases; in `test_sbmc_run` the check of every row accepts `dt_rule` `auto`, `c0_rule`
  `lambda` and the lambdas of the rows with an estimate.
- `include/sbmc/sb_sequential.hpp`, `sb_openmp.hpp`, `sa_sequential.hpp`, `sa_openmp.hpp`,
  `solve_times.hpp`, `src/cpu/sb_sequential.cpp`, `src/cpu/sb_openmp.cpp`: comments only (the
  final check and the times of F07-R1, the two layouts of C2; the instance is checked in full by
  `total_weight` and, with the automatic c0, by `auto_c0`).
- `README.md`: the sections "Spectrum of J, automatic dt and c0" and "Layouts of the block", the
  options of `sbmc_run`, the product form in "Times of the solvers", the domain 5, the tests.

## [F07-R1] - 2026-11-21

Phase 07, part 1 of 4 (measuring tools): results schema 2, the separate times of the solvers
(preparation, run, final check), the final check without a second validation of the instance,
`python/sbmc_tts.py` for schemas 1 and 2, the README at the closure of phase 06. No result of the
solvers changes: only the recorded times and the way the cuts are checked.

### Added

- `include/sbmc/solve_times.hpp`: `SolveTimes` (`prep_ms`, `run_ms`, `check_ms`), the three
  consecutive phases of a call of a solver measured with `std::chrono::steady_clock`.
- `include/sbmc/instance.hpp`, `src/core/instance.cpp`: `sbmc::detail::cut_edges_unchecked` and
  `sbmc::detail::cut_dense_rows_unchecked`, the cut of the final check of the solvers without any
  check, exception, allocation or OpenMP (the edge loop without branches; int32 partial sums per
  dense row up to n = 2^24).
- `tests/cpu/test_solve_times.cpp` (label `cpu`): the times start at zero; 16 calls of the four
  solvers with three finite, non-negative times whose sum is at most the duration of the call.
- `apps/sbmc_run.cpp`: `--tag TEXT` (column `run_tag`); the columns `t_prep_ms`, `t_run_ms`,
  `t_check_ms`, `dt_rule`, `c0_rule`, `cpu_affinity` (from `/proc/self/status`), `omp_places` and
  `omp_proc_bind` (read once at startup; a line break in their value gives exit status 1), and the
  empty `lambda_min`, `lambda_max`, `cpu_classes`.
- Tests: the cases `detail:` of `test_instance`; the new checks of `test_results`,
  `test_sbmc_run` and `test_sbmc_tts` (checks 8 to 10).

### Changed

- `include/sbmc/sb.hpp`, `include/sbmc/sa.hpp`: the field `times` (last field) of `SbResult` and
  `SaResult`.
- `src/cpu/sb_sequential.cpp`, `src/cpu/sb_openmp.cpp`, `src/cpu/sa_sequential.cpp`,
  `src/cpu/sa_openmp.cpp`: the times t0 to t3; the initial states of every replica computed in
  the preparation into `final_x` or `final_spins` (C1 and C2: a parallel loop with
  num_threads(min(T_req, B))); in C1 and C2 the final check in a second parallel region; the final
  check with the `detail` functions instead of `cut_direct` ("the direct cut gives" in the
  messages); header comments. The results are those of F06-R2.
- `include/sbmc/results.hpp`, `src/core/results.cpp`: results schema 2, 56 columns
  (`kResultsSchemaVersion` = 2); a file with the header of schema 1 is refused like any other
  header. The replica file does not change.
- `apps/sbmc_run.cpp`: writes schema 2; `t_step_us` from `t_run_ms`; the line on stdout ends with
  `t_run_ms` and `t_check_ms`; header comment, description and footer of the help.
- `python/sbmc_tts.py`: reads results schemas 1 and 2 (exact headers; any other header: exit
  status 1); with schema 2 the batch time is `t_run_ms`, the group has 31 columns and the summary
  the column `t_batch_column`; with schema 1 the summary and stdout of F06-R2, byte for byte.
- `src/core/CMakeLists.txt`: `solve_times.hpp` in the header check, after `initial_state.hpp`.
- `tests/CMakeLists.txt`: the test `test_solve_times`; `SBMC_TEST_DATA_DIR` and `SBMC_DATA_DIR` for
  `test_instance`; comments. 24 tests with the cpu presets, 30 with the cuda presets, 27 with the
  t4 presets.
- `tests/core/test_results.cpp`, `tests/python/test_sbmc_run.py`, `tests/python/test_sbmc_tts.py`:
  schema 2.
- `README.md`: Status; the measurement of phase 06 on the laptop in "C2: simulated annealing with
  OpenMP"; the new section "Times of the solvers"; the final check in the sections of the four
  solvers; "sbmc_run and the results CSV" and "Time to solution" for schema 2; "Tests".

## [F06-R2] - 2026-11-19

Phase 06, part 2 of 2: C2, simulated annealing with OpenMP over blocks of replicas, equal to the
reference bit for bit; `sbmc_run --variant c2`; T2 for C2; the decisions Q3 (`sbmc_tts.py`) and
Q4 (`sbmc_run`) of F06-R1.

### Added

- `include/sbmc/sa_openmp.hpp` and `src/cpu/sa_openmp.cpp` (in `sbmc_cpu`): `run_sa_openmp` and
  `kSaOpenmpReplicasPerBlock` = 32. Blocks of 32 replicas distributed with `schedule(dynamic, 1)`,
  team min(T_req, B), spin-major work areas per thread on 64-byte boundaries, SIMD over the lanes
  (Philox words, decisions, masked update of the local fields); the format of J and the
  accumulator of C1 (int16, int32, int64); thresholds from a table per block and sweep when
  L <= 4096, from a direct call of `sa_threshold` otherwise; best signs rebuilt by replaying the
  sweep of the last improvement; final check of the best and final cuts with `cut_direct`, in
  parallel. Equal to `run_sa_sequential` field by field (threads apart) in every preset.
- `tests/cpu/test_sa_openmp.cpp` (label `cpu`): the constants SA-1 to SA-7 and the new SA-8 and
  SA-9, C2 against C0 on 50 pairs of runs and on the int16 accumulator with direct thresholds,
  independence from the number of threads and from the split of the replicas, the last global
  indices, RB, every precondition.

### Changed

- `apps/sbmc_run.cpp`: `--variant c2` (with `--algo sa`; "--variant c2 requires --algo sa" with dsb
  and bsb), `--threads` with `c1` or `c2` ("--threads requires --variant c1 or c2"), and
  "--replica-out must differ from --out" when the two paths name the same file (exit status 2
  before any run; exit status 1 when the paths cannot be compared); header comment, help and
  footer.
- `python/sbmc_tts.py`: without `--targets` the fractions are 1000, 995 and 990 with `--replicas`
  and 1000 alone without it (decision Q3); docstring and help.
- `src/core/CMakeLists.txt`: `sa_openmp.hpp` in the header check, after `sa_sequential.hpp`.
- `src/cpu/CMakeLists.txt`: `sa_openmp.cpp` in `sbmc_cpu`; header comment.
- `tests/CMakeLists.txt`: the test `test_sa_openmp`; comments. 23 tests with the cpu presets, 29
  with the cuda presets, 26 with the t4 presets.
- `tests/cpu/test_t2.cpp`: the table SA-T2 also with C2 and three threads; header comment.
- `tests/python/test_sbmc_run.py`: the variant c2 against c0 (four_nodes.rudy with the replica
  file, G1 with R = 133 and three threads), the new messages (also `--replica-out` on the `--out`
  file written in four ways), exit status 1 for paths that cannot be compared; `--variant c3`.
- `tests/python/test_sbmc_tts.py`: the default targets without `--replicas` give the three groups
  at 1000 (exit status 0); exit status 2 also for `--targets 1000,995` without `--replicas`, with
  the message of the relaxed targets.
- `README.md`: status, the section "C2: simulated annealing with OpenMP" with "T2 for C2" (paths
  of the test instances, the measurements of the sandbox, a placeholder for the laptop), sbmc_run
  (`--variant c2`, `--threads`, `--replica-out`), the default targets of sbmc_tts, tests and
  layout.

## [F06-R1] - 2026-11-18

Phase 06, part 1 of 2: the sequential reference of simulated annealing and T2 for it,
`sbmc_run --algo sa`, the replica file, the tool of the time to solution, and an error when the row
of `--target best-known` has another n or m.

### Added

- `include/sbmc/sa.hpp` and `src/core/sa.cpp` (in `sbmc_core`): `SaParams`, `SaReplicaResult`,
  `SaResult`; `sa_beta` (geometric schedule of beta) and `sa_threshold` (acceptance threshold in
  [0, 2^32]), defined once with the formulas of F06-C1; `sa_initial_spin` and
  `sa_acceptance_word`, `constexpr` and `SBMC_HD`, from the domains 3 and 4 of Philox.
- `include/sbmc/sa_sequential.hpp` and `src/cpu/sa_sequential.cpp` (in `sbmc_cpu`):
  `run_sa_sequential`, single-spin Metropolis with sequential sweeps, local fields in int64,
  incremental energy, one acceptance word consumed per proposal, best state with strict
  improvement, final check of the best and of the final spins with `cut_direct`; compressed rows
  or dense INT8 J as in C0; no OpenMP.
- `tests/cpu/test_sa_sequential.cpp` (label `cpu`): the reference cases SA-1 to SA-7, properties,
  replica split, the helper functions, instances with one spin, the acceptance rule at its
  boundary (a word equal to the threshold rejects the uphill move), preconditions; exact values in
  every preset.
- `python/sbmc_tts.py`: TTS99 per group of runs and per fraction of the target, with the Wilson
  interval, the bootstrap, the relaxed targets from the replica file and the gaps; summary CSV.
- `tests/python/test_sbmc_tts.py` (label `python`): the fixture of F06-C1 and its expected values,
  reproducibility, every exit status of the tool, a group with skewed times (median, not mean),
  the bootstrap intervals of one group against the procedure written again in the test.

### Changed

- `include/sbmc/philox.hpp`: documentation of the domains 3 and 4 (word k of `sa_acceptance`:
  sweep * n + spin; their use). No change to the code.
- `include/sbmc/results.hpp`, `src/core/results.cpp`: the replica file, schema 1 (13 columns,
  `kReplicaSchemaVersion`, `replica_columns`, `append_replica_rows`); the file part common to
  `append_results_row` and `append_replica_rows` is an internal function, without change of
  behaviour for `append_results_row`.
- `src/core/CMakeLists.txt`: `sa.cpp` in `sbmc_core`; `sa.hpp` and `sa_sequential.hpp` in the
  header check (C++ and CUDA); header comment.
- `src/cpu/CMakeLists.txt`: `sa_sequential.cpp` in `sbmc_cpu`; header comment.
- `apps/sbmc_run.cpp`: `--algo sa` with `--sweeps`, `--beta0` and `--beta1` (required with `sa`,
  not used with `dsb` and `bsb`); `--steps`, `--dt`, `--c0`, `--a0`, `--eval-every` and
  `--variant c1` not used with `sa`; `--steps` and `--dt` checked after the parsing with the same
  messages; the rows of SA (steps = sweeps, beta0 and beta1, empty dt, c0, a0 and eval_every);
  `--replica-out FILE` (requires `--out`, checked before the runs); with `--target best-known`, a
  row with another n or m is an error (exit status 1) instead of a warning; header comment,
  description and footer of the help.
- `tests/CMakeLists.txt`: `test_sa_sequential` and `test_sbmc_tts`; header comment. Tests per
  preset: 22 (`cpu`, `cpu-strict`), 28 (`cuda`, `cuda-strict`), 25 (`t4`, `t4-strict`).
- `tests/core/test_results.cpp`: a case for `replica_columns` and `append_replica_rows`.
- `tests/cpu/test_t2.cpp`: T2 for the SA reference (table SA-T2); header comment.
- `tests/python/test_sbmc_run.py`: the rows of SA and their replica file, `--replica-out` with
  dSB, the new messages of the command line, a replica file with another header, the error on n
  and m (`--name G1` on `four_nodes.rudy`, a table of the test with a right and a wrong row).
- `README.md`: status, the domains 3 and 4, the sections "Simulated annealing (reference)" (with
  T2 for SA) and "Time to solution (python/sbmc_tts.py)", `sbmc_run` with `--algo sa`, the replica
  file and the error on n and m, tests, repository layout.

### Removed

- Nothing.

## [F05-R2] - 2026-11-18

Phase 05, part 2 of 2: C1, simulated bifurcation with OpenMP over blocks of replicas and SIMD over
the replicas of a block, equal to C0 bit for bit in the strict presets (T7) and part of T2;
`sbmc_run --variant c1 --threads T`; the corrections of the review of F05-R1.

### Added

- `include/sbmc/sb_openmp.hpp` and `src/cpu/sb_openmp.cpp` (in `sbmc_cpu`):
  `kSbOpenmpReplicasPerBlock` (RB = 32, the fastest of 8, 16 and 32 in the measurements) and
  `run_sb_openmp(inst, params, threads, trace_steps)`: blocks of RB consecutive replicas
  distributed by an OpenMP loop with `schedule(dynamic, 1)` in a region with
  `num_threads(min(T_req, B))`; one work area per thread on 64-byte boundaries; spin-major layout
  inside a block and `omp simd` loops over its RB lanes, padding lanes included; J dense INT8 or
  in compressed rows, and an int16, int32 or int64 accumulator from the row norm L; the operations
  of C0 in the same order in every lane; final check with `cut_direct` per replica inside the
  region, errors reported after it for the smallest replica. OpenMP 4.5 constructs only, no
  change of the global OpenMP state.
- `tests/cpu/test_sb_openmp.cpp` (label `cpu`): T7 for C1 (the reference cases G1-G6 and GS-G1;
  C1 equal to C0 field by field on 180 pairs of runs in the strict presets), the integer
  accumulators at the limits of int16 and int32, independence from the number of threads and from
  the replica split, other properties, preconditions.

### Changed

- `include/sbmc/sb.hpp`: `SbResult::threads`, the team of the step loop (1 for C0).
- `apps/sbmc_run.cpp`: `--variant c0|c1` and `--threads T` (0 to 1024, only with `c1`: otherwise
  exit status 2 with "--threads requires --variant c1"); the column `threads` from the result;
  header comment, description and footer of the help.
- `apps/sbmc_bruteforce.cpp`: `--threads` from 0 to 1024 (exit status 2 above); header comment.
- `cmake/sbmc_build_info.cmake`: comment of the generated header on `dirty` (uncommitted changes
  to tracked files outside `results/` of the repository root).
- `src/cpu/CMakeLists.txt`: `sb_openmp.cpp` in `sbmc_cpu`; header comment.
- `src/core/CMakeLists.txt`: `sb_openmp.hpp` in the header check (C++ and CUDA).
- `tests/CMakeLists.txt`: header comment; `test_sb_openmp`; `test_sbmc_run` receives
  `--strict-fp`. Tests per preset: 20 (`cpu`, `cpu-strict`), 26 (`cuda`, `cuda-strict`), 23
  (`t4`, `t4-strict`).
- `tests/cpu/test_t2.cpp`: T2 for C1 with three threads (strict presets: equal to C0 replica by
  replica); header comment.
- `tests/cpu/test_bruteforce.cpp`: the instance "Graph n = 10, triangle and 7 isolated nodes"
  (256 optimal partitions spread over many intervals) among the reference optima and in the test
  of the number of threads.
- `tests/python/test_sbmc_run.py`: argument `--strict-fp`; variant `c0` or `c1` and the column
  `threads` in the common checks; exit statuses of `--variant` and `--threads`; the checks of
  `c1` (the column `threads`, rows independent of the number of threads, the default number of
  threads and, in the strict presets, the same results as `c0`).
- `tests/python/test_sbmc_bruteforce.py`: exit status 2 for `--threads 1025` and for a default
  name taken from a file name with a space.
- `README.md`: status (phase 05 complete in the code), section "OpenMP simulated bifurcation
  (C1)" with the choice of RB and the preliminary scaling measurement, `sbmc_run` with
  `--variant c1` and `--threads`, the laptop times of C3 and the extrapolation to n = 40, the
  limit of `--threads` of `sbmc_bruteforce`, T2 for C1, tests, repository layout.

### Removed

- Nothing.

## [F05-R1] - 2026-11-17

Phase 05, part 1 of 2: the exact solution by Gray-code enumeration (C3), parallel with OpenMP,
`sbmc_bruteforce`, the test T2 for C0, and the flag `dirty` of the build information, which now
ignores `results/`.

### Added

- `include/sbmc/bruteforce.hpp` and `src/cpu/bruteforce.cpp` (in `sbmc_cpu`): `kBruteForceMaxSpins`
  (40), `BruteForceResult` and `solve_bruteforce(inst, threads)`: s_0 = +1, the 2^(n-1) codes in
  2^p contiguous intervals distributed by an OpenMP loop with a dynamic schedule, Gray-code order
  inside an interval with the energy and the fields updated in O(n) in int64, exact combination of
  the intervals (results independent of the number of threads), canonical optimum (smallest
  code), number of optimal partitions, team size; checks in the order threads, n, whole instance;
  final check with `cut_direct` and `ising_energy`. OpenMP 4.5 constructs only, no change of the
  global OpenMP state.
- `apps/sbmc_bruteforce.cpp`: runs C3 on one instance (`--instance` or `--dense` with
  `--instance-seed`, `--name`, `--threads`) and writes the line `BRUTEFORCE instance=... config=...`
  on stdout; exit statuses 0, 1 (library, reading) and 2 (command line).
- `tests/cpu/test_bruteforce.cpp` (label `cpu`): reference optima of 15 instances, independence
  from the number of threads, preconditions.
- `tests/cpu/test_t2.cpp` (label `cpu`): T2, C0 against the exact optimum of C3 on seven cases
  (counts of the replicas at the optimum and the best cuts of T2-3 in the strict presets).
- `tests/python/test_sbmc_bruteforce.py` (label `apps`): command line, exit statuses and output
  line of `sbmc_bruteforce`.
- `tests/scripts/test_build_info.sh` (label `tools`): `cmake/sbmc_build_info.cmake` in a temporary
  git repository, twelve cases.
- `tests/data/sparse_24_unit.rudy` (24 nodes, 48 unit edges, maximum cut 40) and
  `tests/data/weighted_isolated_13.rudy` (13 nodes, weights up to 2^31 - 1 of both signs, two
  isolated nodes).

### Changed

- `cmake/sbmc_build_info.cmake`: `dirty` from
  `git status --porcelain --untracked-files=no -- . ":(exclude)results"`: changes under `results/`
  of the repository root no longer mark the build as dirty; header comment.
- `src/cpu/CMakeLists.txt`: `bruteforce.cpp` in `sbmc_cpu`, which links `OpenMP::OpenMP_CXX`;
  header comment.
- `src/core/CMakeLists.txt`: `bruteforce.hpp` in the header check (C++ and CUDA).
- `apps/CMakeLists.txt`: the executable `sbmc_bruteforce`; header comment.
- `tests/CMakeLists.txt`: header comment; `test_bruteforce`, `test_t2`, `test_sbmc_bruteforce` and
  `test_build_info`. Tests per preset: 19 (`cpu`, `cpu-strict`), 25 (`cuda`, `cuda-strict`), 22
  (`t4`, `t4-strict`).
- `README.md`: status (phase 05, part 1 of 2), section "Exact solution by brute force (C3)" with
  `sbmc_bruteforce` and T2, the definition of `dirty`, tests and test data, origin of the reference
  values, repository layout.

### Removed

- Nothing.

## [F04-R2] - 2026-11-16

Phase 04, part 2 of 2: G-set and K2000, the table of best-known values, `sbmc_run` with the
results CSV (schema 1), T6 on the real files and the fingerprints of the trajectories for T7.

### Added

- `data/gset/`: 13 G-set instances (G1, G2, G11, G12, G14, G15, G22, G43, G48, G55, G63, G70, G81),
  byte for byte as published at Stanford, and `SHA256SUMS` with the SHA-256 of all 71 G-set files.
- `data/best_known.csv`: best-known cut values of the 13 instances and of K2000, with status,
  sources and the date of the check; `data/README.md`: origin, checks and format of the data.
- `scripts/fetch_data.sh`: checks the committed G-set files and downloads into `data/raw/` the
  other 58 G-set files and K2000, every file checked against its SHA-256.
- `third_party/CLI11/` (CLI11 2.7.2, BSD-3-Clause): `CLI11.hpp` and `LICENSE`.
- `include/sbmc/results.hpp` and `src/core/results.cpp` (in `sbmc_core`): results schema 1 (44
  columns), `csv_escape`, `csv_line`, `csv_split`, `append_results_row`, `read_best_known`,
  `find_best_known`.
- `cmake/sbmc_build_info.cmake`: build information (git hash, uncommitted changes, compiler,
  build type) written at every build into `<build>/generated/sbmc/build_info.hpp`, only when it
  changes.
- `apps/CMakeLists.txt` and `apps/sbmc_run.cpp`: interface target `sbmc::cli11`, target
  `sbmc_build_info`, and `sbmc_run`, which runs C0 on one instance and appends one row of schema 1
  per repetition (repetition k on the global replicas [kR, (k + 1)R), same seed).
- `tests/core/test_gset.cpp` (T6 on `data/gset/`), `tests/core/test_k2000.cpp` (skipped with 77
  until `scripts/fetch_data.sh --k2000`) and `tests/core/test_results.cpp`, label `cpu`.
- `tests/python/test_data.py` (label `python`): fingerprints of `data/gset/`, P0's reader on the
  13 files, `data/best_known.csv`; `tests/python/test_sbmc_run.py` (label `apps`): command line,
  exit statuses and CSV of `sbmc_run`.

### Changed

- `.gitattributes`: `data/gset/G* -text`, so that git never converts the line endings of the
  G-set files.
- `CMakeLists.txt`: `add_subdirectory(apps)`.
- `src/core/CMakeLists.txt`: `results.cpp` in `sbmc_core`, `results.hpp` in the header check
  (C++ and CUDA); header comment.
- `src/core/sb.cpp`: comments on the three checks of `auto_c0` that cannot fire with a valid
  instance (defensive checks without a test).
- `tests/CMakeLists.txt`: header comment (label `apps`); `test_gset`, `test_k2000` and
  `test_results` in the loop of the core tests; `SBMC_DATA_DIR` for `test_sb_sequential`; the
  function `sbmc_add_python_test` for the Python tests; `test_data` and `test_sbmc_run`. Tests per
  preset: 15 (`cpu`, `cpu-strict`), 21 (`cuda`, `cuda-strict`), 18 (`t4`, `t4-strict`).
- `tests/cpu/test_sb_sequential.cpp`: fingerprints of the whole trajectories of G1-G6 (strict) and
  the domain case GS-G1 on G1 of the G-set (dt = 0.5).
- `third_party/README.md`: section CLI11.
- `README.md`: status (phase 04 complete), requirements, sections "Data" and "sbmc_run and the
  results CSV", dt on graphs with positive weights, defensive checks of `auto_c0`, references of
  T7, tests, tools, repository layout, third-party code.

### Removed

- Nothing.

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
