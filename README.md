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
Phase 03, core library: rudy reader, Ising model with int64 sums, the project's Philox4x32-10
streams (checked against cuRAND), initial positions, implicit dense instances and dense INT8 rows
of J (see "Core library").
Phase 04 complete (C0, P0, G-set, sbmc_run): the sequential reference C0 of bSB and dSB in FP32
with a fixed order of operations, the parts common to every variant, the Python reference P0 in
float64 and the test T8 that compares them (see "Sequential simulated bifurcation" and "Python
reference P0 and T8"); the G-set instances of the experiments, K2000 and the table of the
best-known values (see "Data"); `sbmc_run`, which runs C0 and writes the results CSV (see
"sbmc_run and the results CSV"). Simulated annealing and the TTS statistics are not implemented
yet.

## Requirements

- WSL2 with Ubuntu 24.04 (the CPU code uses only standard C++20 and OpenMP)
- GCC 13, CMake 3.28 or newer, Ninja
- Python 3 with `venv` (`sudo apt install python3-venv`): used to install the CUDA toolkit and,
  with Python 3.12 or newer, the virtual environment of the Python tests (`scripts/setup_python.sh`)
- An NVIDIA driver that supports CUDA 13.4 (`nvidia-smi` in WSL reports "CUDA Version" 13.4 or
  newer)
- For the tools and their test: git, rsync, unzip and zip (`sudo apt install rsync unzip zip`)
- For `scripts/fetch_data.sh`: curl and sha256sum

## CPU build and tests

```bash
scripts/setup_python.sh          # once: .venv with NumPy for the Python tests
scripts/fetch_data.sh            # once: the data that is not in git (K2000 for test_k2000)
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

The strict presets exist for the bit-for-bit equality tests (the exact values of C0 and, from
phase 05, T7). All compile options come from the interface target `sbmc::flags`
(`cmake/SbmcFlags.cmake`), never from `CMAKE_CXX_FLAGS` or `CMAKE_CUDA_FLAGS`.

## CUDA toolkit and GPU build

The CUDA toolkit is installed from PyPI with a pinned version, without sudo:

```bash
scripts/setup_cuda_pypi.sh   # options: --version 13.4.3, --prefix DIR, --presets laptop|t4,
                             #          --no-presets
cmake --preset cuda-strict && cmake --build --preset cuda-strict && ctest --preset cuda-strict
cmake --preset cuda        && cmake --build --preset cuda        && ctest --preset cuda
```

The script installs `cuda-toolkit[...]==13.4.3`, nvdisasm included, into a virtual environment
under `~/.local/share/sbmc/cuda-13.4.3/`. Where `python3 -m venv` cannot install pip because
the Python has no ensurepip, as on Kaggle's image, it creates the environment without pip and
installs pip into it with the system pip (`python3 -m pip --python ...`, pip 22.3 or newer); the
summary line "venv bootstrap" says which way was taken. It pins `nvidia-nvjitlink`, which the
metapackage only requires as a minimum version, to the version of nvcc (13.4.92 for 13.4.3),
also when it reruns on an existing environment. It creates a library shim (versioned and
unversioned names of the toolkit's shared libraries, which CMake needs to find them), checks
nvcc, tileiras, compute-sanitizer, cuobjdump, nvdisasm, the nvjitlink pin, a minimal Tile
compilation and its disassembly, and writes `CMakeUserPresets.json`.
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

Test labels, folders and the number of tests per preset are in "Tests" below. The GPU smoke
tests, built in `build/<preset>/tests/smoke/`, are:

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

## Core library

The static library `sbmc::core` (`src/core/`) and its public headers (`include/sbmc/`, INTERFACE
target `sbmc::headers`) use only standard C++20 and OpenMP, so the CPU part also builds without
CUDA and, later, on Windows. The public headers contain no CUDA type and include no CUDA header:
the only CUDA-related element is the macro `SBMC_HD` of `sbmc/philox.hpp` (`__host__ __device__`
under `__CUDACC__`, empty otherwise). This is enforced by the build: for each public header CMake
generates a translation unit that includes it twice, compiled as C++ in the OBJECT library
`sbmc_header_check` in every preset and, in the CUDA presets, also by nvcc for the architectures
of the preset in `sbmc_header_check_cuda`.

Errors are exceptions derived from `sbmc::Error` (`sbmc/error.hpp`): `sbmc::IoError` for inputs
that cannot be read or break their format, `sbmc::InvalidArgument` for violated preconditions
(the function then writes nothing to its outputs). Preconditions are checked in every build,
never with `assert`: CMake adds `-DNDEBUG` to Release builds.

### Instances and Ising model (`sbmc/instance.hpp`)

- `Graph`: `n` and the canonical edge list, 0-based, `u < v`, `w != 0` and `w != INT32_MIN`,
  in strictly increasing `(u, v)` order (no duplicates), with `1 <= n <= 2^31 - 1`.
- `DenseSpinGlass`: `n` and a `seed`, the implicit complete graph with `J_ij = +-1` taken from the
  `couplings` stream of the seed (below) and never stored: any `n` up to `2^31 - 1` is valid, and
  the project uses it up to about 160,000 nodes.
- `Instance = std::variant<Graph, DenseSpinGlass>`, the input of the solvers.

Conventions: `J_ij = -w_ij`, `J_ii = 0` (for a `DenseSpinGlass`, `w_ij = -J_ij`);
`W = sum_{i<j} w_ij`, `E(s) = sum_{i<j} w_ij s_i s_j`, `C(s) = sum_{i<j} w_ij [s_i != s_j]`, so
`2C = W - E`. The functions `num_spins`, `total_weight`, `ising_energy`, `cut_direct` and
`dense_rows_int8` exist for `Graph`, `DenseSpinGlass` and `Instance`; `coupling(g, i, j)` gives
one `J_ij` of a `DenseSpinGlass`. Sums are in int64 and `cut_direct` computes `C` directly on
the edges or pairs, never from `W` and `E`. Every function that takes a `Graph` checks it in
full, except `num_spins`, which checks `n` only. `dense_rows_int8(inst, row_begin, row_end, ld,
out)` writes rows `[row_begin, row_end)` of J in INT8, row-major with stride `ld`, with a zero
diagonal and zero padding columns (`j >= n`) and rows (`i >= n`); for a `Graph` every edge needs
`1 <= |w| <= 127`. On a `DenseSpinGlass` the functions run with OpenMP over the rows (rows in
groups of four, so that one Philox block gives four elements) and their results do not depend
on the number of threads.

### Rudy reader (`sbmc/rudy.hpp`)

`read_rudy(path)` reads a G-set file; `parse_rudy(stream, source)` applies the same rules to a
stream. Format: a header line `N M`, then exactly `M` edge lines `i j w` with 1-based indices.

- Accepted: `\n` line endings, a last line without `\n`, CRLF (one trailing `\r` per line is
  removed), leading and trailing spaces and tabs, several spaces or tabs between tokens, blank
  lines anywhere (they still count for the line numbers), integer tokens with an optional `+` or
  `-`, edges with `i > j` and edges in any order.
- Rejected: no header; a header without exactly two integer tokens, or with `N` outside
  `[1, 2^31 - 1]` or `M` outside `[0, N(N - 1)/2]`; an edge line without exactly three integer
  tokens, with `i` or `j` outside `[1, N]`, with `i == j`, with `w == 0` or `|w| > 2^31 - 1`; a
  token that is not an optional sign followed by decimal digits only (`1.0`, `1.5`, `1e3`, `12a`,
  `+-1`); a repeated edge, in the same or in the opposite direction; fewer than `M` edge lines;
  a non-blank line after the `M` edge lines.
- Result: the canonical `Graph`, with `n = N`, `u = min(i, j) - 1`, `v = max(i, j) - 1` and the
  edges sorted by `(u, v)`.
- Messages of `IoError::what()`: `<source>:<line>: <reason>` for an error tied to a line, with
  the physical line number counted from 1 (for a repeated edge, the line of the second
  occurrence, and the reason gives `line <number>` of the first); `<source>: <reason>` for no
  header, fewer than `M` edge lines and read errors. The source of `read_rudy` is the path; a
  file that cannot be opened, or a directory, gives an `IoError` with the path. When the input
  has several errors, the first one in input order is reported. Nothing is preallocated from
  `M`.

### Random streams (`sbmc/philox.hpp`)

Philox4x32-10 of Salmon, Moraes, Dror and Shaw (SC 2011), identical to Random123 and to cuRAND's
`curand_Philox4x32_10`, written as header-only `constexpr noexcept` functions without
intrinsics: the same code compiles for the host and for the device (`SBMC_HD`). Every random
number of the project comes from a stream identified by a seed, a domain `d` and a stream index:

```
stream_block(seed, d, stream, b) = philox4x32_10({lo32(b), hi32(b), stream, d},
                                                 {lo32(seed), hi32(seed)})
stream_word(seed, d, stream, k)  = component k mod 4 of stream_block(seed, d, stream, k / 4)
                                   (0, 1, 2, 3 = x, y, z, w)
```

| Domain                 | Value | Seed (key)    | Stream index       | Word k            | Use                               |
|------------------------|-------|---------------|--------------------|-------------------|-----------------------------------|
| `couplings`            | 1     | instance seed | `min(i, j)`        | `max(i, j)`       | `J_ij = coupling_from_word(word)` |
| `sb_initial_positions` | 2     | run seed      | global replica `r` | spin index `i`    | `x0 = position_from_word(word)`   |
| `sa_initial_spins`     | 3     | run seed      | `r`                | `i`               | reserved for phase 06, not used   |
| `sa_acceptance`        | 4     | run seed      | `r`                | fixed in phase 06 | reserved for phase 06, not used   |

The values are fixed: 0 is never used, a new domain takes the next number and numbers are never
reassigned. Because a stream depends only on indices (instance, replica, spin), results do not
depend on threads, devices or on how the work is split.

Equivalence with cuRAND, checked on the GPU by `test_curand_philox`: after
`curand_init(seed, ((uint64)d << 32) | stream, k, &state)`, the t-th call to `curand(&state)`
returns `stream_word(seed, d, stream, k + t)`, and with `k` a multiple of 4 `curand4(&state)`
returns `stream_block(seed, d, stream, k / 4)`. The GPU code can therefore use either cuRAND or
the project's Philox and get the same bits.

Conversions:

- `coupling_from_word(w)`: `+1` when the top bit of `w` is 0, `-1` otherwise.
- `position_from_word(w)`: `t = (int32)(w >> 8) - 2^23`, an exact integer in `[-2^23, 2^23 - 1]`,
  converted to float and multiplied once by `S = 0.1f / 2^23` (`0x1.99999ap-27f`, bits
  `0x324CCCCD`): a value in `[-0.1f, 0.1f)`. With a single rounded multiplication no FMA
  contraction is possible, so the bits are the same in every preset, strict or not, on the host
  and on the device (`w = 0` gives `-0.1f`).

### Initial positions (`sbmc/initial_state.hpp`)

`initial_position(seed, r, i)` is the x0 of spin `i` in the replica with global index `r` (the
`sb_initial_positions` stream); `y0 = 0`. `make_initial_state(n, replicas, seed, first_replica)`
returns the x0 of replicas `first_replica, ..., first_replica + replicas - 1`, replica-major
(`x0[r * n + i]`), with OpenMP: splitting the replicas over several calls (multi-GPU, M1) gives
the same values. The initial conditions are generated once on the host and shared by every
variant, the condition of the bit-for-bit dSB test T7.

### Reference values of the tests

The reference values of `tests/core/` (Philox words and blocks, couplings, sums and rows of dense
instances, bits of x0) were computed by cldArchitect with an independent implementation (two
Python versions, vectorised NumPy and scalar, that agree with each other), checked on the
Philox4x32-10 known-answer vectors of Random123 (`tests/kat_vectors` of
github.com/DEShawResearch/random123, commit 9545ff6), which `test_philox` checks too. The values
of simulated bifurcation (`test_sb`, `test_sb_sequential`, `test_p0`) were computed by
cldArchitect with two independent implementations of the algorithm, NumPy in float32 and scalar
Python rounding to float32 after every operation, which agree bit for bit; the optima of the
reference cases come from an exact enumeration with Gray code, checked against an enumeration
with NumPy for n <= 20. The fingerprints of the trajectories (G1-G6, GS-G1) come from the same two
implementations, also compared at every step with `sbmc_trace` of the cpu-strict preset of
F04-R1; the bits of `auto_c0` for G1 and K2000 were obtained from the library of F04-R1 in the
cpu-strict preset. The values of the G-set files (`test_gset`, `test_data`) and of K2000
(`test_k2000`) are counts on the files, given by cldArchitect.

## Sequential simulated bifurcation

The static library `sbmc::cpu` (`src/cpu/`) contains the sequential reference C0,
`run_sb_sequential(inst, params, trace_steps)` (`sbmc/sb_sequential.hpp`); the parts common to
every variant are in `sbmc/sb.hpp` (`sbmc::core`). Every spin i of a replica has a position x_i
and a momentum y_i; the equations of motion of the reference document (section 6) are integrated
with the symplectic Euler scheme of Goto et al., in FP32 as on the GPU:

```
x = x0 (make_initial_state: uniform in [-0.1, 0.1), Philox domain 2), y = 0
for m = 0 .. M - 1:
    s = sgn(x)                       # s(m), before any update of the step; sgn(0) = +1
    dSB: F_i = sum_j J_ij s_j         # int64, exact; field_i = (float)F_i
    bSB: acc = +0.0 (double); acc += (double)J_ij * (double)x_j, j increasing; field_i = (float)acc
    evaluate s(m): dSB always, E = -(sum_i s_i F_i) / 2; bSB when eval_every > 0 and
                   m mod eval_every = 0, E = E(s(m)) in int64
    for every spin (sb_update):
        y = y + dt * (c0 * field - p[m] * x)
        x = x + dt_a0 * y            # dt_a0 = dt * a0, computed once in float
        if x > 1 or x < -1: x = (x < 0) ? -1 : 1; y = 0      # inelastic walls; x = +-1 stays
evaluate s(M), the signs after the last step (every algorithm, every eval_every)
```

- Pressure: `p[m] = a0 (1 - (m + 1)/M)`, computed as `(float)((double)a0 * ((double)(M - 1 - m)
  / (double)M))`: one division, one multiplication, one rounding and no addition, so the same
  bits in every preset; `p[M - 1] = 0`.
- Automatic c0 (`params.c0 = 0`): `c0 = 1 / (2 sigma sqrt(n))`, with sigma the standard deviation
  of the n(n - 1) off-diagonal J_ij, computed from int64 sums (`S1 = -2W`, `S2 = 2Q`,
  `P = n(n - 1)`, `Q = sum_{i<j} w_ij^2`) and then in double in a fixed order. It is meant for
  large random instances: on very small graphs it can be too large (on the 4-node example it is
  0.6708 and dSB oscillates, reaching the optimum in few replicas), so the tests on that graph use
  `c0 = 0.3` explicitly. It is undefined, and rejected, when the off-diagonal elements have zero
  variance (for example a complete graph with equal weights). Three of its checks cannot fire
  with a valid instance (overflow of `S1 = -2W`, a variance that is not finite, a c0 that is not
  finite): they stay as defensive checks, documented in `src/core/sb.cpp`, without a test.
- Time step: `dt = 1.25` (the default of `SbParams`) is the value of the literature for random
  matrices with zero mean, and it fails on graphs with positive weights. On G1, C0 with
  `dt = 1.25` keeps in every replica the cut of the random initial configuration (about 9,600, at
  step 0). For G1, lambda_min(J) = -48.79 and lambda_max(J) = 13.27, and the general formula of the
  reference document (section 6), `dt = 1.25 sqrt(2 / (1 - lambda_min / lambda_max))`, gives 0.818;
  in cldArchitect's emulation `dt = 1.0` fails, 0.83 reaches 11,539 and 0.5 with M = 2000 reaches
  11,624, the best-known value (the test GS-G1 below). The automatic choice of dt comes with the
  estimate of lambda in phase 07 (tuning): until then `sbmc_run` requires `--dt`.
- Best cut of a replica: the minimum E over the evaluated configurations; a configuration becomes
  the best when it is the first evaluated or strictly better (ties keep the smallest m);
  `best_cut = (W - best_energy)/2`, `final_cut` is the cut of s(M). bSB with `eval_every = 0`
  evaluates only s(M). After the run the best cut of every replica is checked with one call of
  `cut_direct` (`sbmc::Error` with "internal" on a mismatch).
- C0 checks the instance once at the entry (B5): the step loop works on internal copies of J
  (compressed rows for a `Graph`, dense INT8 for a `DenseSpinGlass`, at most n = 32768) and calls
  no public function of the instance. C0 itself has no OpenMP directive; it runs the replicas
  one after the other and records, on request, the states after the first `trace_steps` steps.
- Why C0 is the bit-for-bit reference: positions, momenta, fields and parameters are float, the
  only double operations are the ones written above, the dSB field is an exact integer, and every
  variant uses the same `sb_sign` and `sb_update` (`SBMC_HD`, also on the device). In the strict
  presets (no FMA contraction) the floating-point results are therefore fixed bit for bit: the
  tests compare C0 with exact reference values there, and from phase 05 every dSB variant must
  reproduce C0's trajectories exactly (T7). In the production presets FMA contraction may change
  the last bits, and the tests check only robust properties (cuts, energies, the final check).
- References of T7: the tests fix the fingerprints (FNV-1a 64 of `trace_x` then `trace_y`, order
  [k][r][i]) of the whole trajectories of G1-G6 and of the first 100 steps of GS-G1 (G1 of the
  G-set, dSB, R = 16, M = 2000, dt = 0.5, automatic c0, seed 1). The fingerprints of the final
  states are not enough: at the end every spin of every replica sits on a wall (|x| = 1, y = 0),
  so they see only the final signs. The trajectories are what T7 compares.

## Python reference P0 and T8

`python/sbmc_ref` is P0, an independent reference in float64 and exact integers, with NumPy and
the standard library only (no Philox, no C++ code): an independent rudy reader, `coupling_matrix`,
`total_weight`, `ising_energy`, `cut_direct`, `auto_c0` (written with `numpy.std`, differently
from the C++ formula on purpose), `sign`, `pressure_schedule`, `sb_step`, `run_sb` and
`best_from_states`.

The Python tests use the virtual environment of the project, `.venv/` (git-ignored), created by
`scripts/setup_python.sh` with NumPy 2.5.4 (`python/requirements.txt`, wheels only; Python 3.12 or
newer). A second run reuses it ("venv bootstrap: existing"). At every configure CMake picks the
interpreter of the tests: the cache variable `SBMC_PYTHON` when set, otherwise `.venv/bin/python`,
otherwise `python3` from the PATH; the summary shows the choice on the line "Python (tests)". The
tests run in isolated mode without bytecode (`python -I -B`), so nothing is written into the
source tree; without NumPy or without an interpreter they exit with 77 (skipped).

```bash
scripts/setup_python.sh
cmake --preset cpu-strict                      # picks .venv/bin/python
ctest --preset cpu-strict -L python -V         # test_p0 and test_t8 with their full output
```

`test_p0` checks P0 alone against exact values (T1, the two tori, the pressure, c0, the sign,
three SB runs, the errors of the reader). `test_t8` is T8: the test tool `sbmc_trace`
(`tests/tools/`) runs C0 and writes a binary trace (little-endian, magic `SBMCTRC1`: header with
n, R, M, K, algorithm, eval_every, dt, a0, the c0 used, seed, first_replica and W; then p[m], J as
INT8, x0, the states x and y after the first K steps, and per replica best_energy, best_cut,
best_step, final_cut); the script reads it, with a check of its exact size, into `<build>/tests/
python/`, and compares it with P0 on ten cases (the 4-node graph, the 4 x 6 torus with +-1
weights, a dense instance with n = 64; bSB and dSB):

| Check | What                                                                                  | Tolerance                             |
|-------|---------------------------------------------------------------------------------------|---------------------------------------|
| A     | header, J, W against P0; c0 against P0's `auto_c0` (or the bits of 0.3f); p; x0 range | c0: 1e-6 relative; p: 1e-7            |
| B     | one step at a time: P0 starts from the C++ state k, compared with state k + 1         | `1e-5 (1 + abs(P0))`                  |
| C     | free run of the first 20 steps of a 1000-step run, from x0 only                       | `1e-4 max(abs(P0), 0.01)`, same signs |
| D     | P0's best-cut rule on the recorded states: best_energy, best_step, final_cut          | exact                                 |

Each case prints `T8 <case>: one-step max=... [free-run max=... signs=identical] [best=equal]
PASS|FAIL`, "max" being the largest ratio between error and tolerance; a further line checks the
exit statuses of `sbmc_trace` on invalid command lines (2) and on errors of the library, of the
instance or of the output file (1). The last line is `T8_VERDICT: PASS|FAIL` (`P0_VERDICT` for
`test_p0`).

## Data

The instances and the target values are in `data/`; their origin, checks and format are described
in `data/README.md`.

- `data/gset/`: the 13 G-set instances of the experiments (evaluation: G1, G11, G14, G22, G43, G48,
  G55, G63, G70, G81; tuning: G2, G12, G15), byte for byte as published at Stanford, and
  `SHA256SUMS` with the SHA-256 of all 71 G-set files.
- `data/best_known.csv`: best-known cut values with status, sources and the date of the check.
- `data/raw/` (not in git): the other 58 G-set files and K2000 (`WK2000_1.rud`, 22.8 MB, above
  the 5 MB limit), downloaded by `scripts/fetch_data.sh`, which checks every file against its
  SHA-256 (`--gset`, `--k2000`, both by default; a second run reports "cached").
- `.gitattributes` marks `data/gset/G*` as `-text`: git never converts the line endings of the
  G-set files (the originals of G56, G60 and G61 have CRLF).

## sbmc_run and the results CSV

`sbmc_run` (`apps/`, built in `build/<preset>/apps/`) runs the sequential reference C0 on one
instance and appends one row per repetition to a CSV with the results schema 1
(`sbmc/results.hpp`). The command line is parsed with CLI11 (`third_party/CLI11`).

```
sbmc_run (--instance FILE | --dense N --instance-seed S) --algo dsb|bsb --replicas R --steps M
         --dt X [--name NAME] [--variant c0] [--a0 X] [--c0 auto|X] [--eval-every K] [--seed S]
         [--reps K] [--warmup K] [--target best-known|none|INTEGER] [--best-known FILE]
         [--out FILE] [--notes TEXT]
```

| Option | Meaning |
|--------|---------|
| `--instance FILE` | rudy file; or `--dense N --instance-seed S`, the implicit complete graph with J_ij = +-1 |
| `--name NAME` | name in the `instance` column and in the best-known table; default: the file name without folder and extension (`data/gset/G1` gives `G1`, `WK2000_1.rud` gives `WK2000_1`), or `dense_n<N>_s<S>` |
| `--algo`, `--replicas`, `--steps`, `--dt` | required: dSB or bSB, R, M and the time step (no default for dt, see "Sequential simulated bifurcation") |
| `--variant c0` | the implementation; phase 04 has only `c0` |
| `--a0 X`, `--c0 auto\|X` | a0 (default 1); c0 automatic (default) or a finite number > 0 (`--c0 0` is an error: in the library c0 = 0 means automatic, and the CSV would record another value) |
| `--eval-every K` | bSB: evaluate the signs every K steps (default 0: only the final configuration) |
| `--seed S` | seed of the initial positions (default 0) |
| `--reps K`, `--warmup K` | recorded repetitions (default 1) and warm-up runs, not recorded (default 1) |
| `--target` | `best-known` (the row `--name` of `--best-known`, default `data/best_known.csv` relative to the current folder), an integer, or `none` (default) |
| `--out FILE` | CSV of the results: header written when the file is new or empty, one row appended per repetition (no CSV without `--out`) |
| `--notes TEXT` | free text of the `notes` column (no line breaks) |

Integers are decimal only (CLI11 alone would read `010` as octal). Exit status: 0 success; 1
error of the library (for example `--replicas 0`: the preconditions of C0 stay with the library,
as in `sbmc_trace`), of reading or writing, or an instance that is not in the best-known table; 2
error of the command line (unknown, missing or repeated option, value that cannot be converted
or is not in the list, missing or double instance). On stdout every repetition prints
`rep <k>: best_cut <v> mean_cut <v> [n_success <a>/<R>] t_solve_ms <v>`.

Repetitions (decision B7): repetition k runs C0 with `first_replica = k R` and the same `--seed`,
that is on the global replicas [k R, (k + 1) R): different initial positions for every
repetition, each one reproducible on its own (the replicas of a repetition are the same whatever
the number of repetitions). The warm-up runs use `first_replica = 0` and are not recorded.
`t_solve_ms` is the duration of `run_sb_sequential`, including its preparation (check of the
instance, internal copy of J, automatic c0, initial positions); `t_setup_ms` is the time to read
the rudy file (or to set up the dense instance), the same in every row of a run.

Example: G1 with the parameters of GS-G1, five repetitions, target the best-known value:

```bash
mkdir -p results/f04
build/cpu/apps/sbmc_run --instance data/gset/G1 --algo dsb --replicas 16 --steps 2000 --dt 0.5 \
    --seed 1 --reps 5 --target best-known --out results/f04/sbmc_run_G1_c0.csv \
    --notes "laptop on AC power, Windows power mode: Best performance"
```

K2000: the file is `data/raw/k2000/WK2000_1.rud` and its row in the table is `K2000`, so a run with
`--target best-known` needs `--name K2000`.

Columns of schema 1 (44, in order: the columns of section 10.4 of the reference document,
preceded by `schema_version`; `a0`, `eval_every`, `first_replica`, `threads` and
`target_source` added before `notes`, which stays the last column):

| Column | Meaning | C0 (`sbmc_run`, phase 04) |
|--------|---------|---------------------------|
| `schema_version` | version of the schema | 1 |
| `timestamp` | start of the repetition, UTC, `YYYY-MM-DDThh:mm:ssZ` | |
| `git_hash` | commit of the source tree at build time | 40 hexadecimal digits, or `unknown` |
| `dirty` | uncommitted changes to tracked files at build time | 1, 0, or empty when unknown |
| `host` | host name (`/proc/sys/kernel/hostname`) | |
| `os` | `PRETTY_NAME` of `/etc/os-release`, " / ", kernel release (`/proc/sys/kernel/osrelease`, which names the WSL kernel on WSL) | `unknown` for a missing part |
| `gpu`, `driver`, `cuda`, `cublas` | GPU, driver, CUDA runtime and cuBLAS versions | empty (CPU) |
| `compiler` | C++ compiler, `<id> <version>` | for example `GNU 13.3.0` |
| `algo` | `dsb`, `bsb` (`sa` from phase 06) | |
| `variant` | implementation | `c0` |
| `instance` | `--name` | |
| `n`, `nnz` | spins; edges (a dense instance has n(n - 1)/2) | |
| `replicas`, `steps` | R and M | |
| `dt`, `c0`, `a0` | parameters as float, shortest form that reads back as the same float; `c0` is the value used (the automatic one with `--c0 auto`) | |
| `beta0`, `beta1` | temperature schedule of simulated annealing | empty (SB) |
| `seed`, `rep` | `--seed`; repetition k | |
| `t_setup_ms`, `t_solve_ms`, `t_step_us` | setup time, solve time of the repetition, `t_solve_ms` * 1000 / M; three decimals | |
| `best_cut`, `mean_cut` | maximum and mean (three decimals) over the replicas of their best cut | |
| `n_success`, `p_s`, `target` | replicas with best cut >= target, `n_success` / R, the target | empty without a target |
| `tts99_ms` | time to solution at 99% | empty: phase 06 (statistics over the repetitions) |
| `ranks`, `t_comm_ms` | MPI processes, communication time | 1; empty (multi-GPU only) |
| `sm_clock_avg_mhz`, `temp_max_c`, `power_avg_w` | GPU clock, temperature and power during the run | empty (CPU) |
| `eval_every`, `first_replica` | bSB evaluation period; global index of the first replica, k R | |
| `threads` | CPU threads | 1 (C0 is sequential) |
| `target_source` | origin of the target: `best_known`, `user`, `none` | |
| `notes` | `--notes` | |

Reproducibility (section 8.8 of the reference document): the build information (`git_hash`,
`dirty`, `compiler`) is written at every build by `cmake/sbmc_build_info.cmake` into
`build/<preset>/generated/sbmc/build_info.hpp`, rewritten only when it changes, so `dirty`
follows the state of the tracked files at the last build without a new configure. The version of
WSL appears in the `os` column (kernel release). The power profile of the laptop cannot be read
automatically: in the measurement campaigns it goes into `--notes`.

## Tests

| Label    | Tests                                                                                                             | Sources                               |
|----------|-------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `cpu`    | `test_sanity`, `test_philox`, `test_instance`, `test_rudy`, `test_initial_state`, `test_sb`, `test_gset`, `test_k2000`, `test_results`, `test_sb_sequential` | `tests/`, `tests/core/`, `tests/cpu/` |
| `python` | `test_p0`, `test_t8` (with the tool `sbmc_trace`), `test_data`                                                    | `tests/python/`, `tests/tools/`       |
| `apps`   | `test_sbmc_run` (with the tool `sbmc_trace`)                                                                      | `tests/python/`                       |
| `tools`  | `test_sync_from_zip`; `test_cuda_rpath` (CUDA presets)                                                            | `tests/scripts/`                      |
| `gpu`    | `smoke_simt`, `test_curand_philox`; `smoke_tile`, FMA probes (Tile only)                                          | `tests/smoke/`, `tests/gpu/`          |
| `tile`   | `smoke_tile`, `fma_probe_tile_fmad_on`, `fma_probe_tile_fmad_off`                                                 | `tests/smoke/`                        |
| `probe`  | `fma_probe_tile_fmad_on`, `fma_probe_tile_fmad_off`                                                               | `tests/smoke/`                        |

Executables are built in `build/<preset>/tests/` (`test_sanity`), `build/<preset>/tests/core/`,
`build/<preset>/tests/cpu/`, `build/<preset>/tests/tools/` (`sbmc_trace`, a tool of the tests, not
a test), `build/<preset>/tests/gpu/` and `build/<preset>/tests/smoke/`; the traces of T8 go to
`build/<preset>/tests/python/`, the files of `test_sbmc_run` to `build/<preset>/tests/apps/`.
`tests/data/` holds the input files of the tests: `four_nodes.rudy` (the 4-node example of the
reference document) and two 4 x 6 toroidal grids, `torus_4x6_unit.rudy` (unit weights, maximum
cut 48) and `torus_4x6_pm1.rudy` (+-1 weights, maximum cut 24); `test_gset`, `test_data`,
`test_results` and the case GS-G1 read `data/`. Number of tests: 15 with `cpu` and `cpu-strict`, 21
with `cuda` and `cuda-strict`, 18 with `t4` and `t4-strict`. A test that cannot run on the current
machine (no GPU, no NumPy, missing tool, missing data) exits with 77 and CTest reports it as
skipped (for example `ctest --preset cuda -L tile` runs only the Tile tests): `test_k2000` needs
`scripts/fetch_data.sh --k2000` first.

Tests in detail (the C++ ones use doctest):

- `test_philox` (T11, CPU part): the three Random123 vectors (the first also at compile time),
  reference words of the streams, a block against its four words, both conversions;
- `test_instance`: T1 (the 4-node example), int64 sums and the identity `2C = W - E` on random
  graphs, every precondition, T5 (dense INT8 rows of a `Graph`: values, symmetry, padding, rows
  in blocks, round trip) and T12 (implicit dense instances: reference couplings, sums and rows up
  to n = 160,000, statistics, independence from the number of threads);
- `test_rudy` (T6 on example inputs): the accepted variants and every rejection rule, with the
  line numbers of the messages;
- `test_initial_state`: reference bits of x0, layout, replica split, independence from the
  number of threads, range and mean;
- `test_sb`: the pressure schedule and the automatic c0 against reference bits, `sb_sign`,
  `sb_update` (walls, exact values), every precondition;
- `test_gset` (T6 on the files of `data/gset/`): n, m and W of the 13 instances, the first and
  last canonical edges of G1 and G81, E and C of G1 with every spin +1, `auto_c0` of G1;
- `test_k2000`: n, m, W and the counts of the weights of K2000, `auto_c0`; exits with 77 and
  "SKIP: run scripts/fetch_data.sh --k2000" when the file is missing;
- `test_results`: the 44 columns of schema 1, `csv_escape`, `csv_line` and `csv_split` (round
  trips, malformed lines), `append_results_row` in a temporary folder (new file, appended rows,
  file with another header left unchanged, wrong rows), `read_best_known` on
  `data/best_known.csv` and on malformed tables;
- `test_sb_sequential`: C0 on a short trajectory bit for bit, the reference cases G1-G6 (best cut,
  best step and final cut of every replica, fingerprint of the final state and of the whole
  trajectory), every step of G3-G6 recomputed from the specified operations, the domain case
  GS-G1 on G1 of the G-set (every preset: a replica reaches at least 11,600 and none exceeds the
  best-known 11,624; strict: every replica, the fingerprints of the first 100 steps and of the
  final state), properties of every preset (optimum reached, final check, final state equal to
  the last recorded one), replica split, independence from the number of threads, every
  precondition. Exact floating-point values only in the strict presets;
- `test_p0` and `test_t8` (label `python`): P0 alone and T8, C0 against P0 (section "Python
  reference P0 and T8"), with the lines `P0 ...` and `T8 ...` and the final `P0_VERDICT` and
  `T8_VERDICT`;
- `test_data` (label `python`): `data/gset/SHA256SUMS` (71 sorted lines, its SHA-256) and the
  SHA-256 of the 13 files, P0's reader on the 13 files (n, m, W), `data/best_known.csv` read with
  the csv module (header and 14 rows; best-known value at most the sum of the positive weights);
  final line `DATA_VERDICT`;
- `test_sbmc_run` (label `apps`): `sbmc_run` run from the repository root, its exit statuses (2 for
  the command line, 1 for the library and for reading and writing), the 44 columns and their
  values on the 4-node graph, a repetition against `sbmc_trace` on the same global replicas,
  appending to an existing CSV, `--target best-known` on G1 and with `--name`, a dense instance
  with a warm-up run, the lines on stdout; final line `RUN_VERDICT`;
- `test_curand_philox` (T11, GPU part, label `gpu`): cuRAND (`curand`, sequences of 16 calls,
  `curand4`), the project's Philox on the device and x0 on the device, all against the host, with
  the lines `T11_CURAND_WORD`, `T11_CURAND_SEQUENCE`, `T11_CURAND4`, `T11_OWN_PHILOX_DEVICE`,
  `T11_X0_DEVICE` (`PASS` or `FAIL` with `(<different>/<cases>)`) and the final
  `T11_GPU_VERDICT: PASS|FAIL`.

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
- `scripts/fetch_data.sh [--gset] [--k2000]`: see "Data" and `data/README.md`.
- `scripts/kaggle_run.sh [--ref LABEL] [--out DIR]`: see "Running on Kaggle (2 x T4)".

Files larger than 5 MB never go into git. `.gitignore` cannot express this rule:
`scripts/sync_from_zip.sh` enforces it on every delivery.

## Repository layout

```
CMakeLists.txt, CMakePresets.json, cmake/   build system (cmake/sbmc_build_info.cmake: build
                     information of the results rows)
include/sbmc/        public headers (no CUDA types): error, philox, instance, rudy, initial_state,
                     sb, sb_sequential, results
src/core/            rudy reader, Ising model, int64 cut and energy, dense INT8 rows, initial
                     positions, pressure schedule and automatic c0, results schema and CSV, table
                     of best-known values (TTS statistics later)
src/cpu/             sequential reference C0 (phase 04); OpenMP SB, simulated annealing and
                     brute force in later phases
src/gpu/             cuBLAS, cuBLASLt, bitwise and cuSPARSE products, update kernels
src/tile/            fused CUDA Tile kernel (sm_89 only)
src/mgpu/            MPI and NCCL multi-GPU drivers
apps/                command-line programs: sbmc_run (phase 04)
tests/               unit, integration and smoke tests (ctest)
tests/core/          unit tests of the core library (doctest)
tests/cpu/           unit tests of the CPU solvers (doctest)
tests/data/          input files of the tests
tests/gpu/           CUDA tests (test_curand_philox)
tests/python/        Python tests: P0 alone (test_p0), T8 (test_t8), the data (test_data), sbmc_run
                     (test_sbmc_run)
tests/smoke/         CUDA smoke tests and FMA probes
tests/scripts/       tool tests and checks
tests/tools/         tools of the tests (sbmc_trace)
python/              NumPy reference P0 (sbmc_ref), requirements.txt; analysis and plots later
bench/configs/       experiment configurations
data/                G-set instances (gset/), best-known values with sources (best_known.csv),
                     downloaded data outside git (raw/); see data/README.md
results/             CSV files and metadata (raw data outside git)
scripts/             setup, tools, measurement campaigns
docs/                report sources and figures
third_party/         vendored dependencies: doctest, CLI11
```

Existing in phase 04: `apps/`, `cmake/`, `data/`, `include/sbmc/`, `python/`, `scripts/`,
`src/core/`, `src/cpu/`, `tests/` (with `core/`, `cpu/`, `data/`, `gpu/`, `python/`, `scripts/`,
`smoke/`, `tools/`) and `third_party/`; the other folders come with later phases.

## License and third-party code

MIT License, see `LICENSE`. Vendored dependencies: doctest 2.5.3 (MIT), single header in
`third_party/doctest/`, and CLI11 2.7.2 (BSD-3-Clause), single header in `third_party/CLI11/`;
versions, sources and checksums in `third_party/README.md`. The G-set files in `data/gset/` are
third-party data, not covered by the license of the project (see `data/README.md`).
