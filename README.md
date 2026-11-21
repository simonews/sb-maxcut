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
"sbmc_run and the results CSV").
Phase 05 complete (C1, C3, T2, T7 for C1, preliminary scaling measurement on the laptop): C1,
simulated bifurcation with OpenMP over blocks of replicas and SIMD over the replicas of a block,
equal to C0 bit for bit in the strict presets (T7), and `sbmc_run --variant c1` (see "OpenMP
simulated bifurcation (C1)"); the exact enumeration C3 with Gray code, parallel with OpenMP, and
`sbmc_bruteforce`, which runs it on one instance; T2, C0 and C1 against the exact optimum on small
graphs (see "Exact solution by brute force (C3)"); the flag `dirty` of the build information
ignores `results/` (see "sbmc_run and the results CSV").
Phase 06 complete, with the measurement on the laptop. Part 1: the
sequential reference of simulated annealing, with T2 (see "Simulated annealing (reference)");
`sbmc_run --algo sa`, the replica file of `--replica-out` and an error when the row of
`--target best-known` has another n or m (see "sbmc_run and the results CSV"); the time to
solution with `python/sbmc_tts.py` (see "Time to solution (python/sbmc_tts.py)"). Part 2: C2,
simulated annealing with OpenMP over blocks of replicas and SIMD over the replicas of a block,
equal to the reference bit for bit in every preset, with T2, and `sbmc_run --variant c2` (see "C2:
simulated annealing with OpenMP"); `--replica-out` must name another file than `--out`; without
`--replicas` and `--targets`, `python/sbmc_tts.py` uses the target 1000 alone.
Phase 07, part 1 of 4 (measuring tools): results schema 2, with the times of the solvers, the
rules of dt and c0, columns reserved for the spectrum of J and for E9, and a tag of the run (see
"sbmc_run and the results CSV"); the separate times of the solvers, preparation, run and final
check, and the final check without a second validation of the instance (see "Times of the
solvers"); `python/sbmc_tts.py` for schemas 1 and 2 (see "Time to solution
(python/sbmc_tts.py)").

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
on the number of threads. Internal to the solvers, `sbmc::detail::cut_edges_unchecked` and
`sbmc::detail::cut_dense_rows_unchecked` compute the cut of their final check without any check
(see "Times of the solvers").

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
| `sa_initial_spins`     | 3     | run seed      | `r`                | `i`               | `s_i = coupling_from_word(word)`  |
| `sa_acceptance`        | 4     | run seed      | `r`                | `sweep * n + spin` | accepted when `word < sa_threshold(beta, delta_e)` |

The values are fixed: 0 is never used, a new domain takes the next number and numbers are never
reassigned. Simulated annealing (see "Simulated annealing (reference)") takes the initial spins
from domain 3 and one acceptance word per proposal from domain 4, consumed whether the move is
accepted or not. Because a stream depends only on indices (instance, replica, spin), results do not
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
(`test_k2000`) are counts on the files, given by cldArchitect. The exact optima of C3
(`test_bruteforce`, `test_t2`, `test_sbmc_bruteforce`: minimum energy, number of optimal
partitions, canonical optimum) come from cldArchitect's own Gray-code enumeration in C++,
independent of the project's code and checked against an exhaustive enumeration in NumPy on every
instance up to 24 nodes; the dense instances with 30 and 32 nodes come from the Gray-code
enumeration only. The numbers of replicas of C0 at the optimum in T2 come from cldArchitect's
emulation of C0 in float32, equal to `n_success` of `sbmc_run` in the cpu-strict preset of F04-R2
(an emulation in float64 gives the same counts).

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
  evaluates only s(M). After the run the best cut of every replica is checked against its cut
  computed directly, on the edges of a `Graph` or on the internal INT8 rows of a
  `DenseSpinGlass`, without a second validation of the instance (`sbmc::Error` with "internal" on
  a mismatch; see "Times of the solvers").
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

## OpenMP simulated bifurcation (C1)

`run_sb_openmp(inst, params, threads = 0, trace_steps = 0)` (`sbmc/sb_openmp.hpp`, in
`sbmc::cpu`) is C1, the first parallel variant (reference document, sections 7 and 9.7; decision
H2): bSB and dSB with OpenMP over blocks of replicas and SIMD over the replicas of a block. Its
parameters, preconditions and result are those of C0, `run_sb_sequential`, plus `threads` (checked
first: `threads >= 0`); the result has one more field, `SbResult::threads`, which C0 leaves at 1.

- Threads. The R replicas form B = ceil(R / RB) blocks of RB = `kSbOpenmpReplicasPerBlock` = 32
  consecutive replicas. T_req is `threads` when it is positive, otherwise `omp_get_max_threads()`
  read at the entry; the parallel region of the steps gets `num_threads(min(T_req, B))`, since
  threads beyond the blocks would have nothing to do, and `result.threads` is the size of its team
  (min(T_req, B) when `omp_get_dynamic()` is 0). The library never changes the global OpenMP state
  (no `omp_set_*`) and has no upper limit on `threads`; the applications accept at most 1024.
- Organization. A loop with `schedule(dynamic, 1)` gives the blocks to the threads, so faster
  cores take more blocks. A block runs all M steps on one thread, in the private work area of that
  thread (64-byte aligned, whole cache lines, allocated before the region). Inside a block the
  arrays are spin-major, element (spin i, lane l) at i * RB + l, so the RB replicas are the lanes
  of the vector registers: every loop over the lanes has exactly RB iterations and is an
  `omp simd` loop. The lanes of the last block beyond R are padding: x = 0 and y = 0 without
  `initial_position` (its replica index could overflow), computed like the others and discarded.
  The sums over the columns of a row add four columns per lane loop and take eight per iteration,
  in two lane loops: with one lane loop per iteration GCC -O3 fuses consecutive iterations
  (unroll-and-jam) and vectorizes the result badly (several times slower, measured). A block
  reads x0 of its replicas from `final_x`, where the preparation wrote it, and after its M steps
  writes its rows of the outputs. The final check runs in a second parallel region, after the
  region of the steps (see "Times of the solvers"): the best cut of every replica against its cut
  computed directly; a mismatch is recorded per replica, and after the region the one of the
  smallest replica is reported (`sbmc::Error` with "internal", as in C0). After the entry no
  public function of the instance is called (B5).
- Format of J and accumulator, fixed for the whole run. J is dense INT8 (rows on 64-byte
  boundaries, from `dense_rows_int8`) for a `DenseSpinGlass` (at most n = 32768) and for a `Graph`
  with n <= 32768, |w| <= 127 on every edge and 8m >= n(n - 1) (density at least 25%), compressed
  rows with increasing columns and int32 values otherwise. The integer fields (the field of dSB,
  the evaluations of bSB) use int16 when L = max_i sum_j |J_ij| <= 32767, int32 when
  L <= 2^31 - 1 and int64 otherwise (L = n - 1 for a `DenseSpinGlass`): every partial sum is
  bounded by L, so no order of the terms overflows. The paths of the test instances, checked in
  phase 05 with a temporary diagnostic print:

| Instance | n | m | J | L | Accumulator |
|----------|---|---|---|---|-------------|
| `four_nodes.rudy` | 4 | 5 | dense | 3 | int16 |
| `torus_4x6_pm1.rudy` | 24 | 48 | compressed | 4 | int16 |
| `data/gset/G1` | 800 | 19,176 | compressed | 67 | int16 |
| heavy_torus (`torus_4x6_unit.rudy`, weights x 10000) | 24 | 48 | compressed | 40,000 | int32 |
| dense_graph_40 (see `test_sb_openmp`) | 40 | 689 | dense | 117 | int16 |
| dense_graph_40_heavy (the edge (0, 1) with weight 200) | 40 | 689 | compressed | 268 | int16 |
| `weighted_isolated_13.rudy` | 13 | 19 | compressed | 4,294,969,294 | int64 |
| `DenseSpinGlass{200, 7}` | 200 | 19,900 | dense | 199 | int16 |
| `Graph{1, {}}`, `DenseSpinGlass{1, 3}` | 1 | 0 | dense | 0 | int16 |
| `Graph{3, {{0, 1, w}, {0, 2, 1}}}`, w = 32766, 32767 | 3 | 2 | compressed | w + 1 | int16, int32 |
| the same, w = 2^31 - 2, 2^31 - 1 | 3 | 2 | compressed | w + 1 | int32, int64 |
| K2000 (`data/raw/k2000/WK2000_1.rud`) | 2,000 | 1,999,000 | dense | 1,999 | int16 |

- Operations. Every lane runs the floating-point operations of C0 in the same order: s(m) before
  the updates of the step; the dSB field, (float) of the exact integer sum; the bSB field,
  (float) of the double sum that starts at +0.0 and adds (double)J_ij * (double)x_j for
  increasing j (a dense J adds its zero terms too: each one is +-0.0 and the sum is never -0.0, so
  the bits are those of C0); the evaluations, the best configuration with ties to the smallest
  step, s(M), and `sb_update`.
- Guarantees. In the strict presets every field of the result except `threads` is equal bit for
  bit to `run_sb_sequential(inst, params, trace_steps)`, in dSB and in bSB, for every number of
  threads (T7, decision H3). In every preset the result depends neither on `threads`, nor on the
  size of the team, nor on the order of the blocks, and the results of local replica r depend
  only on the instance, on the parameters other than `replicas` and `first_replica`, and on the
  global index `first_replica + r`: R replicas at once or split with `first_replica` give the same
  replicas. In the production presets FMA contraction may change the bits with respect to C0, the
  same way in every lane, block and thread.

Choice of RB among 8, 16 and 32: C1 with one and two threads on G1 (dSB, R = 256, M = 2000,
dt = 0.5) and on K2000 (dSB, R = 64, M = 200, dt = 1.25), `sbmc_run --variant c1 --reps 3
--warmup 1`, median of `t_solve_ms` in two rounds (first / second). Measurements of cldExecutor's
sandbox (Intel Xeon Processor @ 2.10GHz, 2 cores, preset `cpu`), not of the laptop:

| RB | G1, 1 thread | G1, 2 threads | K2000, 1 thread | K2000, 2 threads |
|----|--------------|---------------|-----------------|------------------|
| 8  | 2188 / 2031 ms | 1034 / 1266 ms | 3447 / 3675 ms | 1785 / 1813 ms |
| 16 | 1265 / 1289 ms | 695 / 668 ms | 2232 / 2238 ms | 1141 / 1120 ms |
| 32 | 1190 / 1172 ms | 606 / 645 ms | 1934 / 1895 ms | 1017 / 1011 ms |

RB = 32 is the fastest in every case, and so it is in bSB on K2000 (4.85 s against 5.43 s with
RB = 16, one thread). On the same two cases C0 (`--variant c0 --reps 1 --warmup 0`) takes
16,958 ms on G1 and 32,022 ms on K2000, about 14 and 17 times C1 with one thread. On K2000 about
a quarter of the time of C1 did not depend on M (the same run with M = 1 or 2 took about
480 ms): mostly the final check of phase 05, `cut_direct` of every replica, which validated the
instance again, O(m) with m = 1,999,000 (since phase 07 the final check is cheaper and measured
apart, see "Times of the solvers"). The
price of RB = 32 is the granularity: a run keeps at most ceil(R / 32) threads at work (R = 64
gives 2), and the dynamic schedule balances the cores with blocks of 32 replicas.

Preliminary scaling measurement on the laptop (decision H5). In WSL2, with
`build/cpu/apps/sbmc_run` and no `OMP_*` variable in its environment, C1 runs on G1 (dSB,
R = 1024, M = 2000, dt = 0.5) and on K2000 (dSB, R = 1024, M = 200, dt = 1.25) with T = 1, 2, 4,
6, 8, 12 and 16 threads (`results/f05/scaling_G1.csv`, `results/f05/scaling_K2000.csv`), and C0
against C1 with one thread on G1 with R = 256 and on K2000 with R = 32
(`results/f05/c0_vs_c1.csv`), every run with `--c0 auto --seed 1 --reps 3 --warmup 1
--target best-known` (the warm-up run also absorbs the start of the OpenMP runtime, tens of
milliseconds in WSL2). With R = 1024 there are 32 blocks, so the team has the requested size up to
16 threads. Acceptance: the same best cut, mean cut and successes for every T, the column
`threads` equal to T, and a speedup of at least 4 with 16 threads with respect to C1 with one
thread. This is not the experiment E9 of the reference document (section 14.2), which comes in
phase 07: C1 and C2 on the P cores only, on the E cores only and on both, with `OMP_PLACES` and
`OMP_PROC_BIND`, at least five repetitions with median and interquartile range, results schema 2
and separate times for the preparation, the steps and the final check. Here (results schema 1)
`t_solve_ms` is the whole call of the solver, its preparation and final check included; since
phase 07 they are measured apart (see "Times of the solvers").

## Simulated annealing (reference)

`run_sa_sequential(inst, params)` (`sbmc/sa_sequential.hpp`, in `sbmc::cpu`) is the sequential
reference of simulated annealing (reference document, section 7): the variant `c0` of
`sbmc_run --algo sa`, and the oracle of C2 (see "C2: simulated annealing with OpenMP"), which gives
the same results bit for bit. The parts common to every variant are in `sbmc/sa.hpp` (functions
in `sbmc::core`): `SaParams`, `SaReplicaResult`, `SaResult`, `sa_beta`, `sa_threshold`,
`sa_initial_spin` and `sa_acceptance_word`.

- Algorithm: single-spin Metropolis on E(s) = sum_{i<j} w_ij s_i s_j, with sequential sweeps.
  For every local replica r, with global index g = first_replica + r:
  `s_i = sa_initial_spin(seed, g, i)`; local fields h_i = sum_j J_ij s_j, exact in int64;
  E = -(sum_i s_i h_i) / 2. Sweep k = 0, ..., S - 1, at beta_k = `sa_beta(beta0, beta1, S, k)`,
  proposes the flip of spin 0, 1, ..., n - 1 in this order: proposal p = k n + i takes the word
  w = `sa_acceptance_word(seed, g, p)`, whether the move is accepted or not;
  dE = 2 s_i h_i; the flip is accepted when dE <= 0 or w < `sa_threshold(beta_k, dE)`, and
  then h_j -= 2 J_ji s_i for every neighbour j (with the old s_i), s_i = -s_i, E += dE and the
  count of accepted moves grows by one. A move with dE = 0 is accepted (isolated spins; with
  n = 1 every proposal).
- Formulas, exactly as written (they fix the bits):

```
sa_beta(beta0, beta1, S, k)      t = (S > 1) ? double(k) / double(S - 1) : 0.0
                                 beta0 * std::pow(beta1 / beta0, t)
sa_threshold(beta, dE)           p = std::exp(-(beta * static_cast<double>(dE)))
                                 static_cast<std::uint64_t>(p * 4294967296.0), in [0, 2^32]
sa_initial_spin(seed, r, i)      coupling_from_word(word i of (seed, sa_initial_spins, r))
sa_acceptance_word(seed, r, p)   word p of the stream (seed, sa_acceptance, r)
```

  beta is geometric from beta0 (first sweep) to beta1 (last sweep), beta0 alone when S = 1; a
  threshold of 2^32 accepts every word, 0 none. `sa_beta` and `sa_threshold` check their
  arguments (InvalidArgument: S >= 1, 0 <= k < S, beta0 finite > 0, beta1 finite >= beta0; beta
  finite > 0, dE > 0); the other two are `constexpr` and `SBMC_HD`, for the GPU too.
- Best cut during the run: a state becomes the best when its energy is strictly lower than the
  best so far (ties keep the earlier state). `best_sweep` is 0 for the initial state, k when the
  best state was found during sweep k (1..S); best_cut = (W - best_energy) / 2;
  final_cut = (W - E) / 2 of the state after the last sweep; `accepted` counts the accepted moves.
- Final check, for every replica: the cut of the best spins computed directly must give
  `best_cut`, and that of the final spins `final_cut`, on the edges of a `Graph` or on the
  internal INT8 rows of a `DenseSpinGlass`, without a second validation of the instance
  (`sbmc::Error` with "internal" otherwise; see "Times of the solvers").
- Internal copy of J as in C0: compressed rows for a `Graph`, dense INT8 for a `DenseSpinGlass`
  (at most n = 32768); no public function of the instance in the sweep loop (B5); no OpenMP in
  the reference itself.
- API: `SaParams` (replicas R >= 1, sweeps S >= 1, beta0 finite > 0, beta1 finite >= beta0 with
  no default, seed, first_replica >= 0 with first_replica + R <= 2^32); `SaResult` (n, replicas,
  sweeps, first_replica, beta0, beta1, total_weight, `per_replica` with best_energy, best_cut,
  best_sweep, final_cut and accepted, `best_spins` and `final_spins` in replica-major order,
  R * n, and threads, 1 for the reference). Preconditions in this order: the parameters, n for a
  `DenseSpinGlass`, `total_weight` (which checks the instance); InvalidArgument with
  "run_sa_sequential: ", except the errors of `total_weight`.
- Guarantees: a replica depends only on the instance, S, beta0, beta1, the seed and its global
  index, so splitting the replicas with `first_replica` gives the same replicas. The only
  floating-point operations are those of `sa_beta` and `sa_threshold`, without sums, defined once
  in `sbmc_core`: no FMA contraction is possible, and the results are the same in every preset,
  strict or not. The tests compare the exact values in every preset.

Cost, measured in cldExecutor's sandbox (Intel Xeon Processor @ 2.10GHz, preset `cpu`): G1 with
R = 8, S = 1000, beta0 = 0.1, beta1 = 3 takes about 125 ms (three runs: 123, 125 and 146 ms), that
is about 125 us per sweep of the 8 replicas and 20 ns per proposal; 2 of the 8 replicas reach
11624. The reference copies the best spins at every improvement and computes the threshold of
every uphill proposal: C2 is the fast version.

### T2 for SA

`test_t2` also checks that the SA reference reaches the exact optimum of C3 and never exceeds it,
with its own table: R = 16, beta0 = 0.1, beta1 = 3, first_replica = 0; the optimum from
`solve_bruteforce`; every best cut equal to `cut_direct` of its configuration. The counts at the
optimum, and the 16 best cuts of SA-T2-1, SA-T2-3 and SA-T2-5, were computed by cldArchitect with
two independent emulators of the reference and are compared in every preset.

| Case | Instance | S | seed | Optimum | Replicas at the optimum |
|------|----------|---|------|---------|-------------------------|
| SA-T2-1 | `sparse_24_unit.rudy` | 20 | 61 | 40 | 9 |
| SA-T2-2 | `sparse_24_unit.rudy` | 100 | 62 | 40 | 15 |
| SA-T2-3 | `DenseSpinGlass{16, 16}` | 20 | 63 | 29 | 11 |
| SA-T2-4 | `DenseSpinGlass{16, 16}` | 100 | 64 | 29 | 16 |
| SA-T2-5 | `DenseSpinGlass{22, 22}` | 20 | 65 | 34 | 15 |
| SA-T2-6 | `DenseSpinGlass{22, 22}` | 100 | 66 | 34 | 16 |
| SA-T2-7 | `torus_4x6_pm1.rudy` | 20 | 67 | 24 | 16 |

## C2: simulated annealing with OpenMP

`run_sa_openmp(inst, params, threads = 0)` (`sbmc/sa_openmp.hpp`, in `sbmc::cpu`) is C2, the
parallel variant of simulated annealing (reference document, section 7; decision S2 (b)): OpenMP
over blocks of replicas and SIMD over the replicas of a block, with the results of the reference
`run_sa_sequential` bit for bit. Its parameters, preconditions and result are those of the
reference, plus `threads`, checked first (`threads >= 0`); the messages start with
"run_sa_openmp: ", except the errors of `total_weight`, and `SaResult::threads` is the team.

- Threads, as C1. The R replicas form B = ceil(R / RB) blocks of RB = `kSaOpenmpReplicasPerBlock`
  = 32 consecutive replicas. T_req is `threads` when it is positive, otherwise
  `omp_get_max_threads()` read at the entry; the region of the sweeps gets
  `num_threads(min(T_req, B))`, and `result.threads` is the size of its team (min(T_req, B) when
  `omp_get_dynamic()` is 0). The library never changes the global OpenMP state.
- Organization, as C1. A loop with `schedule(dynamic, 1)` gives the blocks to the threads; a block
  runs all S sweeps on one thread, without barriers between blocks, in the work area of its thread
  (allocated before the region, 64-byte aligned, whole cache lines: the signs, the local fields,
  the signs at the start of the sweep and those kept for the best state, n * RB elements each, and
  the table of the thresholds). The arrays are spin-major, element (spin i, lane l) at i * RB + l;
  every loop over the lanes has exactly RB iterations. The lanes of the last block beyond R are
  padding: they follow the Philox streams of lane 0 of the block (no global index beyond
  2^32 - 1), take the same decisions and are discarded. A block reads the initial spins of its
  replicas from `final_spins`, where the preparation wrote them (the padding lanes take those of
  lane 0), and after its sweeps writes its rows of the outputs; an exception of the sweeps is
  recorded per block. The final check runs in a second parallel region (see "Times of the
  solvers"): the best and the final cut of every replica whose block ran, against their cuts
  computed directly; a mismatch is recorded per replica, and after the region the first error in
  the order of the replicas is reported (that of its block, then that of its check; `sbmc::Error`
  with "internal" for a mismatch). After the entry no public function of the instance is called
  (B5), and nothing is allocated inside the regions.
- One proposal of spin i for all the lanes of a block: the acceptance words, one Philox block for
  four consecutive proposals of a lane, in a vector loop over the lanes; the RB decisions in one
  lane loop, m = s_i h_i, accepted when m <= 0 or w < threshold(beta_k, 2m), every lane consuming
  the word of every proposal; then, only when at least one lane accepted, the masked update of the
  fields of the neighbours of i, h_c += J_ic d_l with d_l = -2 s_i in the lanes that accepted and 0
  in the others, in modular unsigned arithmetic (the exact result is in [-L, L] and fits the
  accumulator). Energy, best energy and accepted moves of every lane are int64; the improvement is
  strict, as in the reference. Each lane follows the trajectory of its replica in the reference
  whatever the other lanes do.
- Format of J and accumulator, the rule of C1 (see "OpenMP simulated bifurcation (C1)"): dense INT8
  for a `DenseSpinGlass` and for a `Graph` with n <= 32768, |w| <= 127 and 8m >= n(n - 1),
  compressed rows with int32 values otherwise; int16 when L = max_i sum_j |J_ij| <= 32767, int32
  when L <= 2^31 - 1, int64 otherwise.
- Thresholds. beta_k comes only from `sa_beta`, computed once for every sweep before the region,
  and every threshold only from `sa_threshold`. With L <= L_tab = 4096 = 2^12 a table of L + 1
  entries in the work area, filled at the start of every sweep of every block by calling
  `sa_threshold(beta_k, 2m)` for m = 1, ..., L, with entry 0 = 2^32 (accepted: every word is
  below it), read with the index max(m, 0) (a vector gather); with L > L_tab a call of
  `sa_threshold` for every uphill proposal of every lane (a scalar loop). A table per block and
  sweep rather than one S x L table shared before the region: it takes L + 1 entries per thread
  whatever S (a shared table takes 8 S (L + 1) bytes, 32 MB with L = 4096 and S = 1000, and grows
  with S), at the cost of B S L calls of 6.4 ns (measured): about 0.4% of the time of the G1 case
  below and 0.2% of the K2000 case. L_tab is the lowest value allowed: the table costs L calls per
  block and sweep whatever happens, the direct path at most one call per uphill proposal of a lane
  (n RB per block and sweep), so a large L with few spins is cheaper without the table.
- Best signs: no copy at every improvement. Every lane records the sweep and the spin of its last
  improvement; the block keeps the signs at the start of the current sweep and, for the lanes that
  improved during a sweep, copies them at its end into the kept states (one masked copy per sweep
  with an improvement). At the end of the block the fields of the kept states are computed for all
  the lanes at once, and every lane is replayed from its kept state, with the same words and the
  thresholds of `sa_threshold`, up to the proposal of its last improvement; `best_sweep = 0` gives
  the initial state, recomputed from its stream. The final check compares the cut of these signs,
  computed directly, with `best_cut`.
- Guarantees. In every preset, strict or not, every field of the result except `threads` is equal
  to `run_sa_sequential(inst, params)` (the bits of beta0 and beta1 included), for every number of
  threads: there is no floating-point operation outside `sa_beta` and `sa_threshold`. So the
  result depends neither on `threads`, nor on the size of the team, nor on the order of the
  blocks, nor on how the replicas are split with `first_replica`.

The paths of the test instances, checked in phase 06 with a temporary diagnostic print (removed):

| Instance | J | L | Accumulator | Thresholds |
|----------|---|---|-------------|------------|
| `data/gset/G1` | compressed | 67 | int16 | table |
| `torus_4x6_pm1.rudy` | compressed | 4 | int16 | table |
| heavy_torus (`torus_4x6_unit.rudy`, weights x 10000) | compressed | 40,000 | int32 | direct |
| huge_torus (`torus_4x6_unit.rudy`, weights x 500000) | compressed | 2,000,000 | int32 | direct |
| `weighted_isolated_13.rudy` | compressed | 4,294,969,294 | int64 | direct |
| dense_graph_40 (see `test_sb_openmp`) | dense | 117 | int16 | table |
| dense_graph_40_heavy (the edge (0, 1) with weight 200) | compressed | 268 | int16 | table |
| `DenseSpinGlass{200, 7}`, `DenseSpinGlass{300, 9}` | dense | 199, 299 | int16 | table |
| `Graph{1, {}}`, `DenseSpinGlass{1, 3}` | dense | 0 | int16 | table |
| mid_torus (`torus_4x6_unit.rudy`, weights x 2000) | compressed | 8,000 | int16 | direct |
| K2000 (`data/raw/k2000/WK2000_1.rud`) | dense | 1,999 | int16 | table |

Choice of RB and of the best signs: C2 with one and two threads on G1 (R = 256, S = 1000,
beta0 = 0.1, beta1 = 3) and on K2000 (R = 64, S = 100, beta0 = 0.01, beta1 = 3), `sbmc_run
--algo sa --variant c2 --reps 3 --warmup 1`, median of `t_solve_ms`, for RB = 8, 16, 32 and 64
and two ways of keeping the best signs: "copy", a masked copy of the whole block at every
improvement of a lane, and "replay", the one above. Measurements of cldExecutor's sandbox (Intel
Xeon Processor @ 2.30GHz, 2 cores, preset `cpu`), not of the laptop, with variations of about 10%
between rounds:

| RB | Best signs | G1, 1 thread | G1, 2 threads | K2000, 1 thread | K2000, 2 threads |
|----|------------|--------------|---------------|-----------------|------------------|
| 8  | copy   | 1289 ms | 659 ms | 1384 ms | 775 ms |
| 8  | replay | 1191 ms | 643 ms | 1613 ms | 747 ms |
| 16 | copy   | 875 ms | 447 ms | 1287 ms | 654 ms |
| 16 | replay | 841 ms | 435 ms | 1442 ms | 793 ms |
| 32 | copy   | 832 ms | 428 ms | 1210 ms | 852 ms |
| 32 | replay | 882 ms | 445 ms | 1150 ms | 836 ms |
| 64 | copy   | 852 ms | 434 ms | 1364 ms | 1784 ms (team 1) |
| 64 | replay | 827 ms | 419 ms | 1173 ms | 1527 ms (team 1) |

RB = 8 is slower on G1 (about 40%); 16, 32 and 64 tie on G1, and on K2000 with one thread 32 and
64 are the fastest; 64 leaves a run with R = 64 a single block, hence a single thread. RB = 32 it
is, as in C1. The two ways of keeping the best signs tie on G1; on K2000, three more rounds with
RB = 32 and one thread give 1380 to 1672 ms for "copy" and 1172 to 1197 ms for "replay" (median
1439 against 1196 ms, 17% less): on a dense instance a copy of the block costs n * RB elements
and improvements are frequent. With these choices, in the same sandbox, C2 takes 940 ms on G1 with
one thread (489 ms with two) and 1338 ms on K2000 (712 ms with two); C0 (`--variant c0 --reps 1
--warmup 0`) takes 2762 ms on G1 and 3760 ms on K2000, about 2.9 and 2.8 times C2 with one thread.
Per replica and proposal, C2 with one thread costs about 4.6 ns on G1 and 105 ns on K2000, where
every accepted move of any lane updates the fields of the whole block (n * RB elements).

Measurement on the laptop, closure of phase 06 (`results/f06/`, results schema 1, the fixed
power profile, R = 1024 for the scaling, 3 recorded repetitions after one warm-up run, medians of
`t_solve_ms`). In schema 1 `t_solve_ms` includes the preparation and the final check of the
solver, measured apart since phase 07 (see "Times of the solvers").

- Scaling of C2 (`results/f06/scaling_sa_G1.csv`, `results/f06/scaling_sa_K2000.csv`): on G1
  (S = 1000, beta0 = 0.1, beta1 = 3) 3665.2 ms with 1 thread and 488.2 ms with 16, speedup 7.51; on
  K2000 (S = 200, beta0 = 0.01, beta1 = 3) 20172.9 ms with 1 thread, 3287.8 ms with 12 and
  3291.1 ms with 16, speedup 6.13, no gain from 12 to 16 threads. The results are identical for
  every number of threads.
- C0 against C2 with one thread (`results/f06/c0_vs_c2.csv`): on G1 (R = 256) 2321.4 against
  908.5 ms, x2.56 (11.3 against 4.4 ns per replica and proposal); on K2000 (R = 32) 2108.1 against
  654.2 ms, x3.22 (164.7 against 51.1 ns per replica and proposal). The results are identical.

### T2 for C2

`test_t2` runs the table of "T2 for SA" with C2 and three threads: the same optima, counts at the
optimum and sequences of best cuts, in every preset.

## Exact solution by brute force (C3)

`solve_bruteforce(inst, threads)` (`sbmc/bruteforce.hpp`, in `sbmc::cpu`) computes the exact
Max-Cut of an instance with at most 40 spins (`kBruteForceMaxSpins`) by enumeration: the reference
optimum of the domain test T2 on small graphs (reference document, section 8.2). It returns n, W,
`min_energy`, `max_cut`, `optimal_count`, `best_spins` and `threads`.

- Code and canonical optimum. s and -s have the same energy, so s_0 = +1. A configuration with
  s_0 = +1 has the code `c = sum_{i=1}^{n-1} [s_i = -1] 2^(i-1)`, an integer in [0, 2^(n-1)): every
  partition of the nodes is exactly one code, and `optimal_count`, the number of codes with
  E = `min_energy`, is the number of optimal partitions. `best_spins` is the canonical optimum: the
  optimal code with the smallest value (`best_spins[0] = +1`).
- Algorithm. The 2^(n-1) codes are split into 2^p contiguous intervals, each with its p high bits
  fixed. An interval computes its first configuration from scratch (local fields
  `h_i = sum_j J_ij s_j` and `E = -(sum_i s_i h_i) / 2`, O(n^2)), then visits its other codes in
  Gray-code order: step g flips spin `a = countr_zero(g) + 1`, with `E += 2 s_a h_a` and
  `h_j -= 2 J_ja s_a` for every j (O(n)), the values taken before the flip. An OpenMP loop with
  `schedule(dynamic, 1)` distributes the intervals: p is the smallest value with 2^p >= 8 times the
  number of threads, at most n - 1 and at most 16. The update of the fields adds a row of -2J or
  +2J padded with zeros to a width fixed at compile time (8, 16, 24, 32 or 40), so the compiler
  vectorises it. Everything is in int64: with |w| <= 2^31 - 1 and n <= 40, |h_i| < 2^37 and
  |E| < 2^41.
- Independence from the number of threads. Every interval writes only its own minimum, count and
  smallest optimal code; after the parallel region they are combined with exact integer
  operations (the smallest minimum, the sum of the counts that reach it, the smallest of their
  codes), so `min_energy`, `max_cut`, `optimal_count` and `best_spins` depend neither on the number
  of threads nor on p. `threads = 0` means the default number of OpenMP threads, a positive value
  goes into the `num_threads` clause, and the global OpenMP state is never changed; `threads` in
  the result is the size of the team that ran the enumeration.
- Checks, in this order and before any large allocation: `threads >= 0`; n (`num_spins`, which
  checks n only) and n <= 40; the whole instance (`total_weight`). For a large `DenseSpinGlass`,
  `total_weight` costs O(n^2): it must not run before the check on n. Inside the parallel region no
  public function of the instance is called (B5). At the end `max_cut = (W - min_energy)/2` is
  checked against `cut_direct` and `min_energy` against `ising_energy` (`sbmc::Error` with
  "internal" on a mismatch).
- Cost: 2^(n-1) configurations, so the time doubles with every node. Times of
  `sbmc_bruteforce --dense <n> --instance-seed <n>`, preset `cpu`, one run each, on the laptop
  (Intel Core i7-13620H, WSL2; laptop log of F05-R1, 2026-11-17) and in cldExecutor's sandbox
  (Intel Xeon Processor @ 2.30GHz, 2 cores). With the laptop's 5.3 ns per configuration at
  n = 32, n = 40 (2^39 configurations) takes about 49 minutes with one thread and, with the
  speedup measured at n = 32, about 6 minutes with 16 threads: an extrapolation, not a
  measurement.

| n  | Configurations (2^(n-1)) | Laptop, 1 thread | Laptop, 16 threads | Sandbox, 1 thread | Sandbox, 2 threads | Per configuration, 1 thread: laptop, sandbox |
|----|---------------|----------|-----------------|--------|--------|----------------|
| 30 | 536,870,912   | 3.054 s  | 0.437 s (x 7.0) | 2.41 s | 1.17 s | 5.7 ns, 4.5 ns |
| 32 | 2,147,483,648 | 11.428 s | 1.373 s (x 8.3) | 8.43 s | 4.11 s | 5.3 ns, 3.9 ns |

### sbmc_bruteforce

`sbmc_bruteforce` (`apps/`, built in `build/<preset>/apps/`) runs C3 on one instance and writes one
line on stdout, nothing else (no CSV, no build information). The command line follows the
conventions of `sbmc_run` (CLI11, decimal integers only).

```
sbmc_bruteforce (--instance FILE | --dense N --instance-seed S) [--name NAME] [--threads T]
```

| Option | Meaning |
|--------|---------|
| `--instance FILE` | rudy file; or `--dense N --instance-seed S`, the implicit complete graph with J_ij = +-1 |
| `--name NAME` | name in the output line; default: the file name without folder and extension, or `dense_n<N>_s<S>`. Not empty, and without spaces, tabs, `\r` or `\n` (also when it is taken from the file name) |
| `--threads T` | OpenMP threads, an integer from 0 to 1024; default 0, the default number of OpenMP threads (the library, `solve_bruteforce`, has no upper limit) |

Output, one space between the fields:

```
BRUTEFORCE instance=<name> n=<n> m=<m> W=<W> max_cut=<C> min_energy=<E> optimal_count=<k> threads=<t> t_setup_ms=<x> t_solve_ms=<y> config=<s>
```

| Field | Meaning |
|-------|---------|
| `instance` | `--name` |
| `n`, `m` | spins; edges of the rudy graph, or n(n - 1)/2 for a dense instance |
| `W` | total weight |
| `max_cut`, `min_energy` | the exact optimum, `max_cut = (W - min_energy)/2` |
| `optimal_count` | number of optimal partitions (codes with s_0 = +1 and E = `min_energy`) |
| `threads` | size of the OpenMP team that ran the enumeration |
| `t_setup_ms`, `t_solve_ms` | time to read the file (or set up the dense instance) and time of `solve_bruteforce`, `std::chrono::steady_clock`, fixed notation with three decimals |
| `config` | the canonical optimum, n characters `+` or `-`, spin 0 first |

Exit status: 0 success; 1 error of the library (for example n > 40 or an invalid instance) or of
reading; 2 error of the command line (unknown option, missing or double instance, `--dense`
without `--instance-seed` or the opposite, integer that is not decimal or out of range, `--threads`
negative or above 1024, name not valid). Every message on stderr starts with `sbmc_bruteforce: `.

```bash
build/cpu/apps/sbmc_bruteforce --instance tests/data/four_nodes.rudy --threads 1
# BRUTEFORCE instance=four_nodes n=4 m=5 W=5 max_cut=4 min_energy=-3 optimal_count=1 threads=1 t_setup_ms=0.037 t_solve_ms=0.018 config=+-+-
```

### T2: C0 and C1 against the exact optimum

`test_t2` (label `cpu`) checks, on small graphs, that the sequential reference C0 and C1 reach the
exact optimum of C3 and never exceed it. Every case first computes the optimum with
`solve_bruteforce`, which must equal the expected one, then runs C0, and C1 with three threads,
with R = 16, a0 = 1, automatic c0 and first_replica = 0:

| Case | Instance | Algorithm | dt | M | seed | eval_every | Optimum | Replicas at the optimum (strict) |
|------|----------|-----------|----|---|------|------------|---------|----------------------------------|
| T2-1 | `sparse_24_unit.rudy` | dSB | 0.75 | 300 | 51 | 0 | 40 | 16 |
| T2-2 | `sparse_24_unit.rudy` | bSB | 0.75 | 300 | 51 | 1 | 40 | 16 |
| T2-3 | `DenseSpinGlass{16, 16}` | dSB | 1.25 | 200 | 54 | 0 | 29 | 14 |
| T2-4 | `DenseSpinGlass{16, 16}` | bSB | 1.25 | 200 | 54 | 1 | 29 | 16 |
| T2-5 | `DenseSpinGlass{22, 22}` | dSB | 1.25 | 300 | 54 | 0 | 34 | 16 |
| T2-6 | `DenseSpinGlass{22, 22}` | bSB | 1.25 | 300 | 54 | 1 | 34 | 16 |
| T2-7 | `torus_4x6_pm1.rudy` | bSB | 1.25 | 200 | 54 | 1 | 24 | 16 |

In every preset: at least one replica reaches the optimum, no best cut exceeds it, and the best cut
of every replica equals `cut_direct` of its best configuration. In the strict presets the number
of replicas at the optimum is compared exactly, and so are the 16 best cuts of T2-3 (two replicas
stop at 27), and C1 equals C0 replica by replica (best cut, best step, final cut, best
configuration); in the production presets FMA contraction may change them. On `sparse_24_unit.rudy`
dt = 0.75 because its weights are all positive: with lambda_min(J) = -4.359 and
lambda_max(J) = 3.121, the general formula of the reference document (section 6),
`dt = 1.25 sqrt(2 / (1 - lambda_min / lambda_max))`, gives 1.142, below the 1.25 of random
matrices with zero mean (the same issue as on G1, see "Sequential simulated bifurcation").
The reference of simulated annealing has its own table (see "T2 for SA").

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

## Times of the solvers

Every solver, `run_sb_sequential` (C0), `run_sb_openmp` (C1), `run_sa_sequential` (the reference
of SA) and `run_sa_openmp` (C2), measures inside the call three consecutive phases with
`std::chrono::steady_clock`, at four instants t0 <= t1 <= t2 <= t3, and returns them in
milliseconds in the field `times`, the last field of `SbResult` and `SaResult`
(`sbmc/solve_times.hpp`, decision L2 of phase 07):

```cpp
struct SolveTimes {
  double prep_ms = 0.0;   // preparation, t1 - t0
  double run_ms = 0.0;    // run, t2 - t1
  double check_ms = 0.0;  // final check, t3 - t2
};
```

- t0 is the first instruction of the call, before the preconditions. Preparation [t0, t1): the
  preconditions; `total_weight`, the only full check of the instance (B5); c0 and the pressure
  schedule (SB); beta_k of every sweep where it is computed before the run (C2); the internal copy
  of J; the plan of the accumulators and of the thresholds (C1, C2); the allocation of the
  outputs, of the trace and of the work areas; the initial states of every replica, with their
  usual values, written in replica-major order into the output that holds the final state at the
  end: x0 of SB in `final_x` (y = 0), the initial spins of SA in `final_spins`.
- Run [t1, t2): every replica from its initial state, read from `final_x` or `final_spins`: in SA
  the initial local fields and energy; all the M steps or S sweeps, the evaluations, the best
  state and the trace; the replay of the best signs (C2); `per_replica` and the outputs
  `best_spins`, `final_x`, `final_y`, `final_spins`.
- Check [t2, t3): the final check of every replica, and nothing else. A mismatch throws after t3
  (for the smallest replica); `times` is written only before a normal return.
- C1 and C2: the run and the check are two parallel regions, both with num_threads(min(T_req,
  B)); t1 is taken right before the first, t2 after its implicit barrier, t3 after the second;
  `threads` of the result stays the team of the run. The check distributes the replicas with
  schedule(static); in C2 the replicas of a block whose run threw are not checked. In C1 and C2 the
  initial states are computed in the preparation by a parallel loop with num_threads(min(T_req,
  B)), from whole Philox blocks of their streams (the values of `initial_position` and
  `sa_initial_spin`); C0 calls `make_initial_state` (the default number of OpenMP threads), the
  reference of SA calls `sa_initial_spin` in a sequential loop. The padding lanes of C1 start from
  x = 0, y = 0; those of C2 take the initial spins of lane 0 of their block.
- The intervals are consecutive, so prep_ms + run_ms + check_ms is at most the duration of the
  call measured outside it, `t_solve_ms` of `sbmc_run` (which also counts the time between the
  call and t0, and the return of the result).

Final check (decision L3 of phase 07). SB (C0, C1): the cut of the best signs of every replica
must equal its `best_cut`; SA (the reference, C2): the cut of its best spins must equal `best_cut`
and that of its final spins `final_cut`; a mismatch is `sbmc::Error` with the prefix of the solver,
"internal" and the values. The instance is checked once, at the entry, and the cuts are computed
directly by two functions of `sbmc::detail` (`sbmc/instance.hpp`), internal to the solvers: they
check nothing, never throw, never allocate and use no OpenMP, so that they run inside the parallel
region of the check, and they give the values of `cut_direct` for the same instance and spins
(the cases `detail:` of `test_instance`).

- `detail::cut_edges_unchecked(g, s)`: the sum over the edges of g of w_uv [s_u != s_v], in int64,
  without branches (w AND a mask: with a conditional expression GCC -O3 emits a branch whose
  outcome cannot be predicted, about 9 times slower on G1). Used for every `Graph`, also when the
  internal J of C1 or C2 is dense.
- `detail::cut_dense_rows_unchecked(rows, n, ld, s)`: -sum_{i<j} J_ij [s_i != s_j] from dense
  INT8 rows of J (element (i, j) at `rows[i * ld + j]`, the columns j >= n never read), with the
  sum of every row in int32 (exact up to n = 2^24, int64 beyond) and the total in int64. Used for a
  `DenseSpinGlass`, on the internal rows of the solver (ld = n in C0 and in the reference of SA,
  rows on 64-byte boundaries in C1 and C2).

What the check verifies: the energy accounting of every replica, the best (and final) energy that
the solver kept from its fields or incrementally, against a cut computed directly on the edges or
on the rows of J. What it does not verify: for a `DenseSpinGlass` it reads the same INT8 rows as
the solver, generated by `dense_rows_int8`, so an error in those rows would go unseen here;
`dense_rows_int8` has its own tests with independent values (T5, T12). Until phase 06 the solvers
called `cut_direct`, which checks the whole instance again at every call: on K2000 (m = 1,999,000
edges of 12 bytes) two reads of the edge list per call, one call per replica in SB and two in SA.

## sbmc_run and the results CSV

`sbmc_run` (`apps/`, built in `build/<preset>/apps/`) runs simulated bifurcation (dSB or bSB,
with the sequential reference C0 or with C1, `--variant c1`) or simulated annealing (`--algo sa`,
with the sequential reference or with C2, `--variant c2`) on one instance, and appends one row
per repetition to a CSV with the results schema 2 (`sbmc/results.hpp`) and, with
`--replica-out`, one row per replica to the replica file. The command line is parsed with CLI11
(`third_party/CLI11`).

```
sbmc_run (--instance FILE | --dense N --instance-seed S) --algo dsb|bsb|sa --replicas R
         [--steps M --dt X | --sweeps S --beta0 X --beta1 X] [--name NAME]
         [--variant c0|c1|c2] [--threads T] [--a0 X] [--c0 auto|X] [--eval-every K]
         [--seed S] [--reps K] [--warmup K] [--target best-known|none|INTEGER]
         [--best-known FILE] [--out FILE] [--replica-out FILE] [--notes TEXT] [--tag TEXT]
```

| Option | Meaning |
|--------|---------|
| `--instance FILE` | rudy file; or `--dense N --instance-seed S`, the implicit complete graph with J_ij = +-1 |
| `--name NAME` | name in the `instance` column and in the best-known table; default: the file name without folder and extension (`data/gset/G1` gives `G1`, `WK2000_1.rud` gives `WK2000_1`), or `dense_n<N>_s<S>` |
| `--algo dsb\|bsb\|sa`, `--replicas R` | required: dSB, bSB or simulated annealing; R |
| `--steps M`, `--dt X` | required with `dsb` and `bsb`, not used with `sa`: M and the time step (no default for dt, see "Sequential simulated bifurcation") |
| `--sweeps S`, `--beta0 X`, `--beta1 X` | required with `sa`, not used with `dsb` and `bsb`, no default: the sweeps (an integer >= 1) and the inverse temperatures of the first and of the last sweep, finite numbers > 0 with `--beta1` >= `--beta0` (see "Simulated annealing (reference)") |
| `--variant c0\|c1\|c2` | the implementation: `c0` (default), the sequential reference; `c1`, OpenMP over blocks of replicas, only with `dsb` and `bsb` (see "OpenMP simulated bifurcation (C1)"); `c2`, OpenMP over blocks of replicas, only with `sa` (see "C2: simulated annealing with OpenMP") |
| `--threads T` | OpenMP threads of `c1` and `c2`, an integer from 0 to 1024; default 0, the default number of OpenMP threads. Only with `--variant c1` or `c2` |
| `--a0 X`, `--c0 auto\|X` | `dsb` and `bsb`: a0 (default 1); c0 automatic (default) or a finite number > 0 (`--c0 0` is an error: in the library c0 = 0 means automatic, and the CSV would record another value) |
| `--eval-every K` | bSB: evaluate the signs every K steps (default 0: only the final configuration); not with `sa` |
| `--seed S` | seed of the initial positions (`dsb`, `bsb`) or of the initial spins and of the acceptance words (`sa`); default 0 |
| `--reps K`, `--warmup K` | recorded repetitions (default 1) and warm-up runs, not recorded (default 1) |
| `--target` | `best-known` (the row `--name` of `--best-known`, default `data/best_known.csv` relative to the current folder), an integer, or `none` (default) |
| `--out FILE` | CSV of the results: header written when the file is new or empty, one row appended per repetition (no CSV without `--out`) |
| `--replica-out FILE` | replica file: header written when the file is new or empty, R rows appended per repetition after its row of results (see "Replica file"); requires `--out`, and another file |
| `--notes TEXT` | free text of the `notes` column (no line breaks) |
| `--tag TEXT` | free text of the `run_tag` column, the identifier of the campaign or of the configuration (no line breaks; commas allowed, the field is then quoted); default empty |

Integers are decimal only (CLI11 alone would read `010` as octal). Exit status: 0 success; 1
error of the library (for example `--replicas 0`: the preconditions of C0, C1 and of the SA
reference stay with the library, as in `sbmc_trace`), of reading or writing, a `--out` or
`--replica-out` with another header or a partial last line (checked before the runs, the file
is not modified; a `--out` with the header of results schema 1 is refused like any other header,
with a message that names "results schema 2"), an instance that is not in the best-known table, a
row of the table whose n or m differs from the instance, paths of `--out` and `--replica-out`
that cannot be compared (the message names both options), or `OMP_PLACES` or `OMP_PROC_BIND`
with a line break in its value (before any run, with a message that names the variable); 2 error
of the command line (unknown, missing or repeated
option, value that cannot be converted or is not in the list or in its range, missing or double
instance, and the rules of the algorithms, checked after the parsing in this order: "--steps is
required" and "--dt is required" with `dsb` and `bsb`; with `sa`, "--sweeps is required with
--algo sa" (and so for `--beta0`, `--beta1`), "--steps is not used with --algo sa" (and so for
`--dt`, `--c0`, `--a0`, `--eval-every`), "--variant c1 is not available with --algo sa", "--beta1
must be >= --beta0"; with `dsb` and `bsb`, "--sweeps requires --algo sa" (and so for `--beta0`,
`--beta1`) and "--variant c2 requires --algo sa"; then a missing instance, "--threads requires
--variant c1 or c2" (`--threads` with `--variant c0`, explicit or by default), "--replica-out
requires --out" and "--replica-out must differ from --out", when the two paths name the same file
once made absolute and passed to `std::filesystem::weakly_canonical`: `x.csv`, `./x.csv` and
`/abs/x.csv` are the same file, before any run). Until phase 05
`--steps` and `--dt` were required options of CLI11; the messages and their place among the
checks of `sbmc_run` are unchanged, but an error that CLI11 finds during the parsing (a value
that cannot be converted, an unexpected argument) now comes before them; `--tag` with a line
break is an error of the command line too. On stdout every repetition prints `rep <k>: best_cut
<v> mean_cut <v> [n_success <a>/<R>] t_solve_ms <v> t_run_ms <v> t_check_ms <v>`, with the
values of its row.

With `--target best-known`, the row `--name` of the table must have the n and m of the instance
(m: edges of the rudy file, n(n - 1)/2 for a dense instance); otherwise exit status 1, before any
run, with "--target best-known: <name> in <table> has n <n> and m <m>, the instance has n <n> and
<m> edges" (until phase 05 a warning).

Repetitions (decision B7): repetition k runs the solver with `first_replica = k R` and the same
`--seed`, that is on the global replicas [k R, (k + 1) R): different initial positions for every
repetition, each one reproducible on its own (the replicas of a repetition are the same whatever
the number of repetitions). The warm-up runs use `first_replica = 0` and are not recorded.
`t_solve_ms` is the duration of the call of the solver, `run_sb_sequential` (dsb and bsb, c0),
`run_sb_openmp` (c1), `run_sa_sequential` (sa, c0) or `run_sa_openmp` (sa, c2), measured by
`sbmc_run` around the call; `t_prep_ms`, `t_run_ms` and `t_check_ms` are the preparation, the run
and the final check measured inside it (see "Times of the solvers"), whose sum is at most
`t_solve_ms`; `t_step_us` is `t_run_ms` per step or sweep. `t_setup_ms` is the time to read
the rudy file (or to set up the dense instance), the same in every row of a run. At startup,
before the warm-up runs, `sbmc_run` reads once the logical processors allowed to it (the line
`Cpus_allowed_list:` of `/proc/self/status`) and the variables `OMP_PLACES` and `OMP_PROC_BIND`,
recorded in every row.

Example: G1 with the parameters of GS-G1, five repetitions, target the best-known value:

```bash
mkdir -p results/f04
build/cpu/apps/sbmc_run --instance data/gset/G1 --algo dsb --replicas 16 --steps 2000 --dt 0.5 \
    --seed 1 --reps 5 --target best-known --out results/f04/sbmc_run_G1_c0.csv \
    --notes "laptop on AC power, Windows power mode: Best performance"
```

The same instance with C1, 1024 replicas and 8 threads (the column `threads` records the team, 8):

```bash
mkdir -p results/f05
build/cpu/apps/sbmc_run --instance data/gset/G1 --algo dsb --replicas 1024 --steps 2000 --dt 0.5 \
    --variant c1 --threads 8 --seed 1 --reps 3 --target best-known \
    --out results/f05/sbmc_run_G1_c1_example.csv --notes "laptop on AC power"
```

Simulated annealing on G1, the sequential reference, with the replica file:

```bash
mkdir -p results/f06
build/cpu/apps/sbmc_run --instance data/gset/G1 --algo sa --replicas 32 --sweeps 1000 \
    --beta0 0.1 --beta1 3 --seed 1 --reps 5 --target best-known \
    --out results/f06/sa_G1_example.csv --replica-out results/f06/sa_G1_example_replicas.csv
```

The same with C2 and 1024 replicas on 8 threads (one row per repetition, `threads` 8):

```bash
build/cpu/apps/sbmc_run --instance data/gset/G1 --algo sa --variant c2 --threads 8 \
    --replicas 1024 --sweeps 1000 --beta0 0.1 --beta1 3 --seed 1 --reps 3 --target best-known \
    --out results/f06/sa_G1_c2_example.csv
```

K2000: the file is `data/raw/k2000/WK2000_1.rud` and its row in the table is `K2000`, so a run with
`--target best-known` needs `--name K2000`.

Columns of schema 2 (56, in order: the columns of section 10.4 of the reference document,
preceded by `schema_version`; `a0`, `eval_every`, `first_replica`, `threads` and
`target_source`; the 12 columns of phase 07, from `t_prep_ms` to `run_tag`; `notes`, which stays
the last column). Schema 1 (phases 03 to 06) had the first 43 of them and `notes`; it is no longer
written (`python/sbmc_tts.py` reads both):

| Column | Meaning | `sbmc_run` |
|--------|---------|------------|
| `schema_version` | version of the schema | 2 |
| `timestamp` | start of the repetition, UTC, `YYYY-MM-DDThh:mm:ssZ` | |
| `git_hash` | commit of the source tree at build time | 40 hexadecimal digits, or `unknown` |
| `dirty` | uncommitted changes, at build time, to tracked files outside `results/` of the root | 1, 0, or empty when unknown |
| `host` | host name (`/proc/sys/kernel/hostname`) | |
| `os` | `PRETTY_NAME` of `/etc/os-release`, " / ", kernel release (`/proc/sys/kernel/osrelease`, which names the WSL kernel on WSL) | `unknown` for a missing part |
| `gpu`, `driver`, `cuda`, `cublas` | GPU, driver, CUDA runtime and cuBLAS versions | empty (CPU) |
| `compiler` | C++ compiler, `<id> <version>` | for example `GNU 13.3.0` |
| `algo` | `dsb`, `bsb`, `sa` | |
| `variant` | implementation | `c0` or `c1` (SB); `c0` or `c2` (SA) |
| `instance` | `--name` | |
| `n`, `nnz` | spins; edges (a dense instance has n(n - 1)/2) | |
| `replicas`, `steps` | R; M (SB) or S, the sweeps (SA) | |
| `dt`, `c0`, `a0` | parameters as float, shortest form that reads back as the same float; `c0` is the value used (the automatic one with `--c0 auto`) | empty (SA) |
| `beta0`, `beta1` | inverse temperatures of the first and of the last sweep of simulated annealing, shortest form that reads back as the same double | empty (SB) |
| `seed`, `rep` | `--seed`; repetition k | |
| `t_setup_ms`, `t_solve_ms`, `t_step_us` | setup time; duration of the call of the solver; `t_run_ms` * 1000 / M (SB) or / S (SA), from the unrounded time (until schema 1 from `t_solve_ms`); three decimals | |
| `best_cut`, `mean_cut` | maximum and mean (three decimals) over the replicas of their best cut | |
| `n_success`, `p_s`, `target` | replicas with best cut >= target, `n_success` / R, the target | empty without a target |
| `tts99_ms` | time to solution at 99% | empty: a statistic over the repetitions, computed by `python/sbmc_tts.py` |
| `ranks`, `t_comm_ms` | MPI processes, communication time | 1; empty (multi-GPU only) |
| `sm_clock_avg_mhz`, `temp_max_c`, `power_avg_w` | GPU clock, temperature and power during the run | empty (CPU) |
| `eval_every`, `first_replica` | bSB evaluation period (empty for SA); global index of the first replica, k R | |
| `threads` | CPU threads | 1 for `c0` (sequential, SB and SA); for `c1` the team of the steps, for `c2` the team of the sweeps, min(T_req, B) |
| `target_source` | origin of the target: `best_known`, `user`, `none` | |
| `t_prep_ms`, `t_run_ms`, `t_check_ms` | the times measured inside the solver: preparation, run, final check (see "Times of the solvers"); three decimals | |
| `dt_rule` | how dt was chosen: `value`, the value of `--dt` (phase 07, part 2 adds `auto`) | `value` (SB); empty (SA) |
| `c0_rule` | how c0 was chosen: `sigma`, automatic, 1/(2 sigma sqrt(n)); `value`, an explicit `--c0` (phase 07, part 2 adds `lambda`) | SB; empty (SA) |
| `lambda_min`, `lambda_max` | estimated extremes of the spectrum of J | empty (filled from phase 07, part 2) |
| `cpu_affinity` | logical processors allowed to the process at startup, cpulist format (for example `0-15` or `0-3,8`) | the text after `Cpus_allowed_list:` in `/proc/self/status`, empty when missing |
| `cpu_classes` | efficiency classes of the allowed logical processors, groups `<class>:<cpulist>` separated by a space in decreasing class order (for example `1:0-11 0:12-15`) | empty (filled on Windows in phase 07, part 4) |
| `omp_places`, `omp_proc_bind` | `OMP_PLACES` and `OMP_PROC_BIND` at startup | empty when not set |
| `run_tag` | `--tag` | |
| `notes` | `--notes` | |

Reproducibility (section 8.8 of the reference document): the build information (`git_hash`,
`dirty`, `compiler`) is written at every build by `cmake/sbmc_build_info.cmake` into
`build/<preset>/generated/sbmc/build_info.hpp`, rewritten only when it changes, so `dirty`
follows the state of the tracked files at the last build without a new configure. The version of
WSL appears in the `os` column (kernel release). The power profile of the laptop cannot be read
automatically: in the measurement campaigns it goes into `--notes`.

`dirty` is 1 when a tracked file outside the folder `results/` of the repository root has changes,
staged or not, 0 otherwise (-1, an empty column, when git is missing or the source tree is not a
repository): the command is `git status --porcelain --untracked-files=no -- . ":(exclude)results"`,
run in the root. `results/` holds the committed results of the measurements, and a row appended
to one of its CSV files does not change the sources: without the exclusion, the rows appended by a
measurement would mark the following builds as dirty until they are committed. A `results` folder
elsewhere (`sub/results/`) and the files of the root whose name starts with `results`
(`results_old.csv`) still count. The test `test_build_info` (label `tools`) checks these cases in
a temporary repository.

### Replica file

`--replica-out FILE` (with `--out`) writes one row per replica of every recorded repetition, after
the row of results of the repetition, for the relaxed targets and the bootstrap of
`python/sbmc_tts.py`; the results schema does not change. It must be another file than `--out`
(exit status 2 before any run). The file is checked before the runs
like `--out` (another header or a partial last line: exit status 1, file not modified) and gets
its header when it is new or empty. Schema 1 of the replica file (`replica_columns()` and
`append_replica_rows` of `sbmc/results.hpp`), 13 columns:

| Column | Meaning |
|--------|---------|
| `schema_version` | 1 (`kReplicaSchemaVersion`) |
| `timestamp`, `instance`, `algo`, `variant`, `replicas`, `steps`, `seed`, `rep` | the values of the row of results of the same repetition (`steps`: M for SB, S for SA) |
| `replica` | global index of the replica, first_replica + r |
| `best_cut`, `best_step`, `final_cut` | best cut of the replica; `best_step` of SB or `best_sweep` of SA; cut of the final configuration |

The rows of a repetition are in increasing replica order. A row of results and its replica rows
share the key (timestamp, instance, algo, variant, replicas, steps, seed, rep).

## Time to solution (python/sbmc_tts.py)

`python/sbmc_tts.py` computes the time to solution TTS99 (reference document, section 8.5) from a
CSV of results and, optionally, its replica file. It runs with the Python of the project (3.12 or
newer, the one of `.venv`), standard library and NumPy; its functions can also be imported.

```
python3 python/sbmc_tts.py --results FILE [--replicas FILE] [--targets 1000,995,990]
                           [--bootstrap B] [--seed S] [--out FILE]
```

| Option | Meaning |
|--------|---------|
| `--results FILE` | CSV of results schema 1 or 2 (exact header, which tells the schema) |
| `--replicas FILE` | replica file of the same runs (exact header) |
| `--targets` | fractions of the target in thousandths, distinct integers in [1, 1000]; default 1000,995,990 with `--replicas` and 1000 alone without it (decision Q3 of F06-R1); an explicit list without `--replicas` may only be 1000 ("relaxed targets need --replicas") |
| `--bootstrap B` | bootstrap resamples: 0 (none) or at least 1000; default 1000 |
| `--seed S` | seed of the bootstrap, an integer >= 0; default 1 |
| `--out FILE` | CSV of the summary, overwritten |

- Groups: the rows of results with the same git_hash, dirty, host, compiler, algo, variant,
  instance, n, nnz, replicas, steps, dt, c0, beta0, beta1, a0, eval_every, threads, target,
  target_source and notes (21 columns), and with schema 2 also the same os, dt_rule, c0_rule,
  lambda_min, lambda_max, cpu_affinity, cpu_classes, omp_places, omp_proc_bind and run_tag (31
  columns), in the order of first appearance. A group without a target is skipped
  (one line `SKIP` on stdout). In a group a pair (seed, rep) appears once: the same replicas are
  never counted twice.
- Join with the replica file, on the key (timestamp, instance, algo, variant, replicas, steps,
  seed, rep): every row of results considered has exactly R replica rows, with the replicas
  first_replica, ..., first_replica + R - 1; a replica row without its row of results, a repeated
  key or a repeated replica is an error (the replica rows of a skipped row are ignored). Every row
  is checked against its replicas: the maximum of the best cuts is `best_cut`, the number of best
  cuts >= target is `n_success`, and `format(sum / R, ".3f")` is `mean_cut`.
- For every group and fraction f, with T the target, R the replicas, N = rows * R:
  target_value = ceil(f T / 1000), and a replica succeeds when 1000 best_cut >= f T; the
  successes come from the replica rows (without `--replicas`, the sum of `n_success`, f = 1000
  only); p_s = successes / N with the Wilson interval at 95% (z = `NormalDist().inv_cdf(0.975)`;
  exactly 0 below for no success and 1 above for N successes); t_batch = median of `t_solve_ms`
  with schema 1 and of `t_run_ms` with schema 2 (decision L2 of phase 07), with the quartiles of
  `numpy.percentile`; TTS(p) = t_batch when p = 1, undefined when p = 0,
  otherwise t_batch * max(1, ln(0.01) / (R log1p(-p))) (a batch that already reaches 99% costs
  one batch); tts99 = TTS(p_s), its interval TTS(upper Wilson bound) and TTS(lower bound).
- Bootstrap (B > 0): `numpy.random.default_rng([seed, group index, fraction index])` (indices from
  0, skipped groups not counted); every resample draws the rows with replacement
  (`rng.integers(0, rows, size=rows)`), then for every drawn row j, in that order,
  `rng.binomial(R, s_j / R)`; TTS of the resample from those successes and the median of the drawn
  times (infinite when no success); the interval is made of the sorted values at the ranks
  ceil(0.025 B) and ceil(0.975 B), computed in integer arithmetic.
- Best and mean cut: maximum and mean over the replicas (without `--replicas`, the maximum of
  `best_cut` and the mean of `mean_cut`); gaps 100 (T - best) / T and 100 (T - mean) / T, empty
  when T = 0.
- Summary (`--out`), one row per group and fraction: the 21 columns of the group (schema 1), or
  the 31 columns of the group and `t_batch_column`, the column of the batch time, `t_run_ms`
  (schema 2), then
  target_permille, target_value, rows, replicas_total, successes, p_s, p_s_lo, p_s_hi, t_batch_ms,
  t_batch_q1_ms, t_batch_q3_ms, tts99_ms, tts99_lo_ms, tts99_hi_ms, bootstrap, bootstrap_seed,
  tts99_boot_lo_ms, tts99_boot_hi_ms, best_cut, mean_cut, gap_best_pct, gap_mean_pct. Floats are
  written with `repr()`, an undefined value is empty, infinity is `inf`; with `--bootstrap 0` the
  four bootstrap columns are empty. On stdout, one line `TTS ...` per group and fraction (instance,
  algo, variant, R, fraction, target value, p_s, tts99_ms), then
  `TTS_DONE groups=<g> lines=<l> skipped=<s>`.
- Exit status: 0 success; 1 error in the data (missing or unreadable file, header, join, checks
  between the files, duplicates; also NumPy missing); 2 error of the command line. Messages on
  stderr start with `sbmc_tts: `.

Limit, schema 1 only: its `t_solve_ms` is the whole call of the solver, its preparation and
final check included, so the TTS of these runs is slightly pessimistic. With schema 2 the batch
time is `t_run_ms`, the run alone. With schema 1 the summary and stdout are those of F06-R2 byte
for byte.

Example, on the runs of the example of "Replica file":

```bash
.venv/bin/python python/sbmc_tts.py --results results/f06/sa_G1_example.csv \
    --replicas results/f06/sa_G1_example_replicas.csv --out results/f06/sa_G1_example_tts.csv
```

## Tests

| Label    | Tests                                                                                                             | Sources                               |
|----------|-------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `cpu`    | `test_sanity`, `test_philox`, `test_instance`, `test_rudy`, `test_initial_state`, `test_sb`, `test_gset`, `test_k2000`, `test_results`, `test_sb_sequential`, `test_sa_sequential`, `test_sb_openmp`, `test_sa_openmp`, `test_bruteforce`, `test_t2`, `test_solve_times` | `tests/`, `tests/core/`, `tests/cpu/` |
| `python` | `test_p0`, `test_t8` (with the tool `sbmc_trace`), `test_data`, `test_sbmc_tts`                                   | `tests/python/`, `tests/tools/`       |
| `apps`   | `test_sbmc_run` (with the tool `sbmc_trace`), `test_sbmc_bruteforce`                                              | `tests/python/`                       |
| `tools`  | `test_sync_from_zip`, `test_build_info`; `test_cuda_rpath` (CUDA presets)                                         | `tests/scripts/`                      |
| `gpu`    | `smoke_simt`, `test_curand_philox`; `smoke_tile`, FMA probes (Tile only)                                          | `tests/smoke/`, `tests/gpu/`          |
| `tile`   | `smoke_tile`, `fma_probe_tile_fmad_on`, `fma_probe_tile_fmad_off`                                                 | `tests/smoke/`                        |
| `probe`  | `fma_probe_tile_fmad_on`, `fma_probe_tile_fmad_off`                                                               | `tests/smoke/`                        |

Executables are built in `build/<preset>/tests/` (`test_sanity`), `build/<preset>/tests/core/`,
`build/<preset>/tests/cpu/`, `build/<preset>/tests/tools/` (`sbmc_trace`, a tool of the tests, not
a test), `build/<preset>/tests/gpu/` and `build/<preset>/tests/smoke/`; the traces of T8 go to
`build/<preset>/tests/python/`, the files of `test_sbmc_run` to `build/<preset>/tests/apps/`, those
of `test_sbmc_bruteforce` to `build/<preset>/tests/bruteforce/`, those of `test_sbmc_tts` to
`build/<preset>/tests/tts/`.
`tests/data/` holds the input files of the tests: `four_nodes.rudy` (the 4-node example of the
reference document); two 4 x 6 toroidal grids, `torus_4x6_unit.rudy` (unit weights, maximum
cut 48) and `torus_4x6_pm1.rudy` (+-1 weights, maximum cut 24, 35 optimal partitions);
`sparse_24_unit.rudy`, a sparse random-like graph with 24 nodes, 48 edges of weight 1 and degrees
between 2 and 6, like the random graphs of the G-set (maximum cut 40, the instance of T2-1 and
T2-2); `weighted_isolated_13.rudy`, 13 nodes and 19 edges with weights of both signs up to
2^31 - 1 and two isolated nodes (maximum cut 6,442,452,014, 4 optimal partitions), which tests the
int64 sums of C3 (only C3: `auto_c0` rejects it, because Q overflows int64). `test_gset`,
`test_data`, `test_results`, `test_instance`, `test_solve_times`, the case GS-G1, the cases SA-5
and SA-9 read `data/`. Number of tests: 24 with `cpu` and `cpu-strict`, 30 with `cuda` and
`cuda-strict`, 27 with `t4` and `t4-strict`. A
test that cannot run on the current machine (no GPU, no NumPy, missing tool, missing data) exits
with 77 and CTest reports it as skipped (for example `ctest --preset cuda -L tile` runs only the
Tile tests): `test_k2000` needs `scripts/fetch_data.sh --k2000` first.

Tests in detail (the C++ ones use doctest):

- `test_philox` (T11, CPU part): the three Random123 vectors (the first also at compile time),
  reference words of the streams, a block against its four words, both conversions;
- `test_instance`: T1 (the 4-node example), int64 sums and the identity `2C = W - E` on random
  graphs, every precondition, T5 (dense INT8 rows of a `Graph`: values, symmetry, padding, rows
  in blocks, round trip) and T12 (implicit dense instances: reference couplings, sums and rows up
  to n = 160,000, statistics, independence from the number of threads); the cases `detail:`
  (phase 07): `detail::cut_edges_unchecked` equal to `cut_direct` on every spin vector of
  `four_nodes.rudy` (with the cuts 4 and 3 of the reference document), on 20 vectors of G1 and on
  64 of `weighted_isolated_13.rudy` (sums beyond 2^31), 0 on graphs without edges;
  `detail::cut_dense_rows_unchecked` equal to `cut_direct` on `DenseSpinGlass{200, 7}` with
  ld = 200 and ld = 256 (padding columns overwritten with 77, never read), on
  `torus_4x6_pm1.rudy` with ld = 24 and 32, and 0 for one spin;
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
- `test_results`: the 56 columns of schema 2 (the first 43 those of schema 1, written in the
  test; distinct names; `notes` last; version 2), `csv_escape`, `csv_line` and `csv_split` (round
  trips, malformed lines), `append_results_row` in a temporary folder (new file, appended rows,
  file with another header, the header of schema 1 included, left unchanged, wrong rows), the 13
  columns of the replica file and
  `append_replica_rows` (header and rows, a second append without header, another header or a
  partial last line left unchanged, wrong rows before the file is touched, no rows),
  `read_best_known` on `data/best_known.csv` and on malformed tables;
- `test_sb_sequential`: C0 on a short trajectory bit for bit, the reference cases G1-G6 (best cut,
  best step and final cut of every replica, fingerprint of the final state and of the whole
  trajectory), every step of G3-G6 recomputed from the specified operations, the domain case
  GS-G1 on G1 of the G-set (every preset: a replica reaches at least 11,600 and none exceeds the
  best-known 11,624; strict: every replica, the fingerprints of the first 100 steps and of the
  final state), properties of every preset (optimum reached, final check, final state equal to
  the last recorded one), replica split, independence from the number of threads, every
  precondition. Exact floating-point values only in the strict presets;
- `test_sa_sequential` (the SA reference): the reference cases SA-1 to SA-7 of F06-C1 (per replica
  best cut, best energy, best sweep, final cut and accepted moves, the fingerprint of the best and
  final spins, W, n, sweeps, first replica, threads), the properties of the same cases (best and
  final cut against `cut_direct`, best cut >= final cut), the replica split, the four functions of
  `sbmc/sa.hpp` against reference values (`sa_beta` bit for bit), instances with one spin (every
  proposal accepted), the acceptance rule at its boundary (an uphill move whose word equals the
  threshold is rejected), every precondition (also of `sa_beta` and `sa_threshold`), the last
  valid global replica index. Exact values in every preset;
- `test_sb_openmp` (C1, T7): the reference cases G1-G6 with one thread and with the default number
  of threads (strict: every constant of C0, the fingerprints of the final states and of the
  trajectories), GS-G1 with two threads (strict: best cuts, best steps, fingerprints of the first
  100 steps and of the final state), C1 equal to C0 field by field on 180 pairs of runs (strict;
  nine instances that cover both formats of J and the three accumulators, dSB and bSB with
  eval_every 0, 1 and 7, five configurations of replicas, first replica and threads), the integer
  accumulators at their limits (L = 32767, 32768, 2^31 - 1 and 2^31: no overflow, best and final
  cuts against `cut_direct`, strict: equal to C0), the same result with 1, 2, 3 and the default
  number of threads (the team size in `threads`, the OpenMP settings unchanged), the replica split
  with `first_replica`, the c0 actually used, instances with one spin, `trace_steps = 0`, every
  precondition (`threads = -1`, those of C0, a `DenseSpinGlass` with n = 32769 rejected before
  `total_weight`) and the last valid global replica index;
- `test_sa_openmp` (C2): the constants SA-1 to SA-7 with one thread and with the default number
  of threads, SA-8 and SA-9 of F06-C2 (sums over 70 replicas, maximum and minimum, fingerprint,
  four replicas; SA-9 on the last 70 global indices) with the reference and with C2; C2 equal to
  C0 field by field on 50 pairs of runs (ten instances that cover both formats of J, the three
  accumulators and both paths of the thresholds, five configurations of replicas, first replica
  and threads), plus the int16 accumulator with direct thresholds; the same result with 1, 2, 3
  and the default number of threads (the team in `threads`, the OpenMP settings unchanged); the
  replica split; the last global indices (2^32 - 69 replicas: equal to C0; one more: rejected);
  RB; every precondition (`threads = -1` first, those of the reference, a `DenseSpinGlass` with
  n = 32769 before `total_weight`, the errors of `total_weight`). Exact values in every preset;
- `test_bruteforce` (C3): n, W, minimum energy, maximum cut, number of optimal partitions and
  canonical optimum of 16 instances (the five files of `tests/data/`, dense instances with 1 to 24
  spins, graphs with 1 to 3 nodes, a triangle with 7 isolated nodes, whose 256 optimal partitions
  fall in many intervals), with `cut_direct` and `ising_energy` of the optimum; the same
  result with 1, 2, 3 and the default number of threads (`omp_set_dynamic(0)`, the team size in
  `threads`, the OpenMP settings unchanged by the library); every precondition, among them a
  `DenseSpinGlass` with n = 2^31 - 1, rejected at once because n is checked before `total_weight`;
- `test_solve_times` (phase 07): `SolveTimes{}` and the field `times` of `SbResult{}` and
  `SaResult{}` at zero; for 16 calls of the four solvers (G1, dense_graph_40 and
  `DenseSpinGlass{200, 7}`; dSB and bSB; one and three threads) the three times finite and >= 0,
  their sum at most the duration of the call measured outside it, and on G1 a run time > 0;
- `test_t2` (T2): the seven cases of "T2: C0 and C1 against the exact optimum", the optimum from
  C3, the replicas of C0 and of C1 at the optimum (counts and the best cuts of T2-3 in the strict
  presets, where C1 also equals C0 replica by replica); the seven cases of "T2 for SA" with the
  reference and with C2 (counts and three sequences of best cuts in every preset);
- `test_p0` and `test_t8` (label `python`): P0 alone and T8, C0 against P0 (section "Python
  reference P0 and T8"), with the lines `P0 ...` and `T8 ...` and the final `P0_VERDICT` and
  `T8_VERDICT`;
- `test_data` (label `python`): `data/gset/SHA256SUMS` (71 sorted lines, its SHA-256) and the
  SHA-256 of the 13 files, P0's reader on the 13 files (n, m, W), `data/best_known.csv` read with
  the csv module (header and 14 rows; best-known value at most the sum of the positive weights);
  final line `DATA_VERDICT`;
- `test_sbmc_tts` (label `python`): `python/sbmc_tts.py` on a fixture written to its folder of the
  build tree (three groups and a row without target): the values of every group and fraction
  computed by cldArchitect (successes, P_s and its Wilson interval, the batch time and its
  quartiles, TTS99 and its interval, best and mean cut, gaps), the SKIP line, the bootstrap
  interval (around TTS99, infinite, or inside the times), the same summary byte for byte on a
  second call, `--bootstrap 0`, the runs without `--replicas` (with `--targets 1000`, and
  without `--targets`: the three groups at 1000 alone), every exit status 1 (data) and 2
  (command line), a group with skewed times (the batch time is their median, not their mean), the
  bootstrap intervals of one group against the procedure of "Time to solution", written again in
  the test; since phase 07 the fixture in results schema 2 (the same P_s, intervals, cuts and
  gaps; the times 0.9 times those of schema 1 since its `t_run_ms` is 0.9 `t_solve_ms`;
  `t_batch_column`; the header of the summary), two groups when only `run_tag` or `os` differ, and
  a header of schema 2 with a column less or more (exit status 1); one line `TTS <check>:
  PASS|FAIL` per check and the final line `TTS_VERDICT`;
- `test_sbmc_run` (label `apps`): `sbmc_run` run from the repository root, its exit statuses (2 for
  the command line, among them `--threads` with `--variant c0` and out of [0, 1024]; 1 for the
  library and for reading and writing), the 56 columns and their values on the 4-node graph, a
  repetition against `sbmc_trace` on the same global replicas, appending to an existing CSV,
  `--target best-known` on G1, `--name G1` on another instance (exit status 1: another n and m)
  and a table of the test with a right and a wrong row, a dense instance with a warm-up run, the
  lines on stdout; with `--variant c1`: the column `threads` equal to the request, the same rows
  with 1 and 3 threads apart from the times, `--threads 0`, and in the strict presets (argument
  `--strict-fp 1`, passed by CMake) the same best cut, mean cut, successes and c0 as `c0` in dSB
  and bSB; with `--algo sa`: the rows and the replica file against the values of F06-C1, every
  new message of the command line (compared word for word), `--replica-out` with dSB against the
  rows of results and a second run on the same files, a replica file with another header left
  unchanged; with `--variant c2`: the run of `--algo sa` with two threads (one block: `threads`
  1, the rows and the replica file of `c0`), G1 with R = 133 and three threads against `c0`, the
  messages of `--variant c2` with dSB and bSB, of `--threads` with `c0`, of `--variant c3`, of
  `--replica-out` on the `--out` file written in four ways, and exit status 1 for a loop of
  symbolic links in the path of `--replica-out`; with results schema 2 (phase 07): in every row of
  every run the three times >= 0 with their sum at most `t_solve_ms` + 0.002, `t_step_us` from
  `t_run_ms`, `dt_rule` and `c0_rule`, `lambda_min`, `lambda_max` and `cpu_classes` empty,
  `cpu_affinity` equal to the `Cpus_allowed_list` of the test, `omp_places` and `omp_proc_bind`;
  a run under `taskset -c 0` (skipped, and said so, without `taskset`), `OMP_PLACES=cores` and
  `OMP_PROC_BIND=close` in the environment and then neither, `OMP_PLACES` with a line break (exit
  status 1), `--tag "E2 cfg,1"` and `--tag` with a line break (exit status 2), an `--out` with the
  header of schema 1 (exit status 1, file unchanged), the lines on stdout with `t_run_ms` and
  `t_check_ms`; final line `RUN_VERDICT`;
- `test_sbmc_bruteforce` (label `apps`, standard library only): `sbmc_bruteforce` with absolute
  paths, its exit statuses (2 for the command line, among them `--threads 1025` and a default
  name taken from a file name with a space; 1 for the library and for reading; always with an
  empty stdout and a message that starts with `sbmc_bruteforce: `), the output line (the
  word `BRUTEFORCE` and the 11 fields in order, the times with three decimals) and its values on
  `four_nodes.rudy`, `weighted_isolated_13.rudy` and dense instances with 1, 16 and 20 spins (the
  last one with 1 and 4 threads), `--name`; one line `BF <check>: PASS|FAIL` per check and the
  final line `BF_VERDICT`;
- `test_build_info` (label `tools`): `cmake/sbmc_build_info.cmake` run with `cmake -P` in a
  temporary git repository (git configuration isolated as in `test_sync_from_zip`): `dirty` stays 0
  for changes in `results/` (modified, untracked, deleted files) and becomes 1 for a modified or
  staged source, for `sub/results/` and for `results_old.csv`; git_hash equal to HEAD; a second
  run without changes writes nothing; -1 and `unknown` without git and outside a repository. One
  line `BUILD_INFO <case>: PASS|FAIL` per case and `BUILD_INFO_VERDICT`; exit 77 without git;
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
                     solve_times, sb, sb_sequential, sb_openmp, sa, sa_sequential, sa_openmp,
                     bruteforce, results
src/core/            rudy reader, Ising model, int64 cut and energy, dense INT8 rows, the
                     unchecked cut of the final check, initial positions, pressure schedule and
                     automatic c0, schedule of beta and acceptance threshold of SA, results
                     schema 2, replica file and CSV, table of best-known values
src/cpu/             sequential reference C0 (phase 04), OpenMP simulated bifurcation C1 and
                     exact enumeration C3 with OpenMP (phase 05), sequential reference of
                     simulated annealing and OpenMP simulated annealing C2 (phase 06)
src/gpu/             cuBLAS, cuBLASLt, bitwise and cuSPARSE products, update kernels
src/tile/            fused CUDA Tile kernel (sm_89 only)
src/mgpu/            MPI and NCCL multi-GPU drivers
apps/                command-line programs: sbmc_run (phase 04, C1 from phase 05, SA, C2 and
                     the replica file from phase 06), sbmc_bruteforce (phase 05)
tests/               unit, integration and smoke tests (ctest)
tests/core/          unit tests of the core library (doctest)
tests/cpu/           unit tests of the CPU solvers (C0, C1 with T7, the SA reference, C2, C3), T2
                     and the times of the solvers (doctest)
tests/data/          input files of the tests
tests/gpu/           CUDA tests (test_curand_philox)
tests/python/        Python tests: P0 alone (test_p0), T8 (test_t8), the data (test_data), sbmc_run
                     (test_sbmc_run), sbmc_bruteforce (test_sbmc_bruteforce), the time to
                     solution (test_sbmc_tts)
tests/smoke/         CUDA smoke tests and FMA probes
tests/scripts/       tool tests and checks (sync_from_zip.sh, build information, RPATH)
tests/tools/         tools of the tests (sbmc_trace)
python/              NumPy reference P0 (sbmc_ref), time to solution (sbmc_tts.py),
                     requirements.txt; analysis and plots later
bench/configs/       experiment configurations
data/                G-set instances (gset/), best-known values with sources (best_known.csv),
                     downloaded data outside git (raw/); see data/README.md
results/             CSV files and metadata (raw data outside git)
scripts/             setup, tools, measurement campaigns
docs/                report sources and figures
third_party/         vendored dependencies: doctest, CLI11
```

Existing in phase 07: `apps/`, `cmake/`, `data/`, `include/sbmc/`, `python/`, `scripts/`,
`src/core/`, `src/cpu/`, `tests/` (with `core/`, `cpu/`, `data/`, `gpu/`, `python/`, `scripts/`,
`smoke/`, `tools/`) and `third_party/`; the other folders come with later phases.

## License and third-party code

MIT License, see `LICENSE`. Vendored dependencies: doctest 2.5.3 (MIT), single header in
`third_party/doctest/`, and CLI11 2.7.2 (BSD-3-Clause), single header in `third_party/CLI11/`;
versions, sources and checksums in `third_party/README.md`. The G-set files in `data/gset/` are
third-party data, not covered by the license of the project (see `data/README.md`).
