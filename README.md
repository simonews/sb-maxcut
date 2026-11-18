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
Phase 05 complete in the code (C1, C3, T2, T7 for C1); the laptop measurement of the scaling
follows. C1, simulated bifurcation with OpenMP over blocks of replicas and SIMD over the replicas
of a block, equal to C0 bit for bit in the strict presets (T7), and `sbmc_run --variant c1` (see
"OpenMP simulated bifurcation (C1)"); the exact enumeration C3 with Gray code, parallel with
OpenMP, and `sbmc_bruteforce`, which runs it on one instance; T2, C0 and C1 against the exact
optimum on small graphs (see "Exact solution by brute force (C3)"); the flag `dirty` of the build
information ignores `results/` (see "sbmc_run and the results CSV"). Simulated annealing and the
TTS statistics are not implemented yet.

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
  (unroll-and-jam) and vectorizes the result badly (several times slower, measured). After its M
  steps a block writes its rows of the outputs and checks the best cut of each of its replicas
  with `cut_direct`, in the same region; a mismatch or an exception is recorded per replica, and
  after the region the one of the smallest replica is reported (`sbmc::Error` with "internal", as
  in C0). Inside the region no public function of the instance is called (B5).
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
a quarter of the time of C1 does not depend on M (the same run with M = 1 or 2 takes about
480 ms): mostly the final check, `cut_direct` of every replica, O(m) with m = 1,999,000. The price of
RB = 32 is the granularity: a run keeps at most ceil(R / 32) threads at work (R = 64 gives 2),
and the dynamic schedule balances the cores with blocks of 32 replicas.

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
and separate times for the preparation, the steps and the final check. Here `t_solve_ms` keeps
its meaning, the whole call of the solver.

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
Simulated annealing joins T2 in a later phase.

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

`sbmc_run` (`apps/`, built in `build/<preset>/apps/`) runs simulated bifurcation on one instance,
with the sequential reference C0 or with C1 (`--variant c1`), and appends one row per repetition
to a CSV with the results schema 1 (`sbmc/results.hpp`). The command line is parsed with CLI11
(`third_party/CLI11`).

```
sbmc_run (--instance FILE | --dense N --instance-seed S) --algo dsb|bsb --replicas R --steps M
         --dt X [--name NAME] [--variant c0|c1] [--threads T] [--a0 X] [--c0 auto|X]
         [--eval-every K] [--seed S] [--reps K] [--warmup K] [--target best-known|none|INTEGER]
         [--best-known FILE] [--out FILE] [--notes TEXT]
```

| Option | Meaning |
|--------|---------|
| `--instance FILE` | rudy file; or `--dense N --instance-seed S`, the implicit complete graph with J_ij = +-1 |
| `--name NAME` | name in the `instance` column and in the best-known table; default: the file name without folder and extension (`data/gset/G1` gives `G1`, `WK2000_1.rud` gives `WK2000_1`), or `dense_n<N>_s<S>` |
| `--algo`, `--replicas`, `--steps`, `--dt` | required: dSB or bSB, R, M and the time step (no default for dt, see "Sequential simulated bifurcation") |
| `--variant c0\|c1` | the implementation: `c0` (default), the sequential reference; `c1`, OpenMP over blocks of replicas (see "OpenMP simulated bifurcation (C1)") |
| `--threads T` | OpenMP threads of `c1`, an integer from 0 to 1024; default 0, the default number of OpenMP threads. Only with `--variant c1` |
| `--a0 X`, `--c0 auto\|X` | a0 (default 1); c0 automatic (default) or a finite number > 0 (`--c0 0` is an error: in the library c0 = 0 means automatic, and the CSV would record another value) |
| `--eval-every K` | bSB: evaluate the signs every K steps (default 0: only the final configuration) |
| `--seed S` | seed of the initial positions (default 0) |
| `--reps K`, `--warmup K` | recorded repetitions (default 1) and warm-up runs, not recorded (default 1) |
| `--target` | `best-known` (the row `--name` of `--best-known`, default `data/best_known.csv` relative to the current folder), an integer, or `none` (default) |
| `--out FILE` | CSV of the results: header written when the file is new or empty, one row appended per repetition (no CSV without `--out`) |
| `--notes TEXT` | free text of the `notes` column (no line breaks) |

Integers are decimal only (CLI11 alone would read `010` as octal). Exit status: 0 success; 1
error of the library (for example `--replicas 0`: the preconditions of C0 and C1 stay with the
library, as in `sbmc_trace`), of reading or writing, or an instance that is not in the best-known
table; 2 error of the command line (unknown, missing or repeated option, value that cannot be
converted or is not in the list or in its range, missing or double instance, `--threads` without
`--variant c1`, explicit or by default: "--threads requires --variant c1"). On stdout every
repetition prints `rep <k>: best_cut <v> mean_cut <v> [n_success <a>/<R>] t_solve_ms <v>`.

Repetitions (decision B7): repetition k runs the solver with `first_replica = k R` and the same
`--seed`, that is on the global replicas [k R, (k + 1) R): different initial positions for every
repetition, each one reproducible on its own (the replicas of a repetition are the same whatever
the number of repetitions). The warm-up runs use `first_replica = 0` and are not recorded.
`t_solve_ms` is the duration of the call of the solver, `run_sb_sequential` (c0) or
`run_sb_openmp` (c1), including its internal preparation (check of the instance, internal copy of
J, automatic c0, initial positions; for c1 also the work areas and the start of the parallel
region) and its final check; `t_setup_ms` is the time to read the rudy file (or to set up the
dense instance), the same in every row of a run.

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

K2000: the file is `data/raw/k2000/WK2000_1.rud` and its row in the table is `K2000`, so a run with
`--target best-known` needs `--name K2000`.

Columns of schema 1 (44, in order: the columns of section 10.4 of the reference document,
preceded by `schema_version`; `a0`, `eval_every`, `first_replica`, `threads` and
`target_source` added before `notes`, which stays the last column):

| Column | Meaning | `sbmc_run` (C0 and C1) |
|--------|---------|------------------------|
| `schema_version` | version of the schema | 1 |
| `timestamp` | start of the repetition, UTC, `YYYY-MM-DDThh:mm:ssZ` | |
| `git_hash` | commit of the source tree at build time | 40 hexadecimal digits, or `unknown` |
| `dirty` | uncommitted changes, at build time, to tracked files outside `results/` of the root | 1, 0, or empty when unknown |
| `host` | host name (`/proc/sys/kernel/hostname`) | |
| `os` | `PRETTY_NAME` of `/etc/os-release`, " / ", kernel release (`/proc/sys/kernel/osrelease`, which names the WSL kernel on WSL) | `unknown` for a missing part |
| `gpu`, `driver`, `cuda`, `cublas` | GPU, driver, CUDA runtime and cuBLAS versions | empty (CPU) |
| `compiler` | C++ compiler, `<id> <version>` | for example `GNU 13.3.0` |
| `algo` | `dsb`, `bsb` (`sa` from phase 06) | |
| `variant` | implementation | `c0` or `c1` |
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
| `threads` | CPU threads | 1 for `c0` (sequential); for `c1` the team of the steps, min(T_req, B) |
| `target_source` | origin of the target: `best_known`, `user`, `none` | |
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

## Tests

| Label    | Tests                                                                                                             | Sources                               |
|----------|-------------------------------------------------------------------------------------------------------------------|---------------------------------------|
| `cpu`    | `test_sanity`, `test_philox`, `test_instance`, `test_rudy`, `test_initial_state`, `test_sb`, `test_gset`, `test_k2000`, `test_results`, `test_sb_sequential`, `test_sb_openmp`, `test_bruteforce`, `test_t2` | `tests/`, `tests/core/`, `tests/cpu/` |
| `python` | `test_p0`, `test_t8` (with the tool `sbmc_trace`), `test_data`                                                    | `tests/python/`, `tests/tools/`       |
| `apps`   | `test_sbmc_run` (with the tool `sbmc_trace`), `test_sbmc_bruteforce`                                              | `tests/python/`                       |
| `tools`  | `test_sync_from_zip`, `test_build_info`; `test_cuda_rpath` (CUDA presets)                                         | `tests/scripts/`                      |
| `gpu`    | `smoke_simt`, `test_curand_philox`; `smoke_tile`, FMA probes (Tile only)                                          | `tests/smoke/`, `tests/gpu/`          |
| `tile`   | `smoke_tile`, `fma_probe_tile_fmad_on`, `fma_probe_tile_fmad_off`                                                 | `tests/smoke/`                        |
| `probe`  | `fma_probe_tile_fmad_on`, `fma_probe_tile_fmad_off`                                                               | `tests/smoke/`                        |

Executables are built in `build/<preset>/tests/` (`test_sanity`), `build/<preset>/tests/core/`,
`build/<preset>/tests/cpu/`, `build/<preset>/tests/tools/` (`sbmc_trace`, a tool of the tests, not
a test), `build/<preset>/tests/gpu/` and `build/<preset>/tests/smoke/`; the traces of T8 go to
`build/<preset>/tests/python/`, the files of `test_sbmc_run` to `build/<preset>/tests/apps/`, those
of `test_sbmc_bruteforce` to `build/<preset>/tests/bruteforce/`.
`tests/data/` holds the input files of the tests: `four_nodes.rudy` (the 4-node example of the
reference document); two 4 x 6 toroidal grids, `torus_4x6_unit.rudy` (unit weights, maximum
cut 48) and `torus_4x6_pm1.rudy` (+-1 weights, maximum cut 24, 35 optimal partitions);
`sparse_24_unit.rudy`, a sparse random-like graph with 24 nodes, 48 edges of weight 1 and degrees
between 2 and 6, like the random graphs of the G-set (maximum cut 40, the instance of T2-1 and
T2-2); `weighted_isolated_13.rudy`, 13 nodes and 19 edges with weights of both signs up to
2^31 - 1 and two isolated nodes (maximum cut 6,442,452,014, 4 optimal partitions), which tests the
int64 sums of C3 (only C3: `auto_c0` rejects it, because Q overflows int64). `test_gset`,
`test_data`, `test_results` and the case GS-G1 read `data/`. Number of tests: 20 with `cpu` and
`cpu-strict`, 26 with `cuda` and `cuda-strict`, 23 with `t4` and `t4-strict`. A test that cannot
run on the current machine (no GPU, no NumPy, missing tool, missing data) exits with 77 and CTest
reports it as skipped (for example `ctest --preset cuda -L tile` runs only the Tile tests):
`test_k2000` needs `scripts/fetch_data.sh --k2000` first.

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
- `test_bruteforce` (C3): n, W, minimum energy, maximum cut, number of optimal partitions and
  canonical optimum of 16 instances (the five files of `tests/data/`, dense instances with 1 to 24
  spins, graphs with 1 to 3 nodes, a triangle with 7 isolated nodes, whose 256 optimal partitions
  fall in many intervals), with `cut_direct` and `ising_energy` of the optimum; the same
  result with 1, 2, 3 and the default number of threads (`omp_set_dynamic(0)`, the team size in
  `threads`, the OpenMP settings unchanged by the library); every precondition, among them a
  `DenseSpinGlass` with n = 2^31 - 1, rejected at once because n is checked before `total_weight`;
- `test_t2` (T2): the seven cases of "T2: C0 and C1 against the exact optimum", the optimum from
  C3, the replicas of C0 and of C1 at the optimum (counts and the best cuts of T2-3 in the strict
  presets, where C1 also equals C0 replica by replica);
- `test_p0` and `test_t8` (label `python`): P0 alone and T8, C0 against P0 (section "Python
  reference P0 and T8"), with the lines `P0 ...` and `T8 ...` and the final `P0_VERDICT` and
  `T8_VERDICT`;
- `test_data` (label `python`): `data/gset/SHA256SUMS` (71 sorted lines, its SHA-256) and the
  SHA-256 of the 13 files, P0's reader on the 13 files (n, m, W), `data/best_known.csv` read with
  the csv module (header and 14 rows; best-known value at most the sum of the positive weights);
  final line `DATA_VERDICT`;
- `test_sbmc_run` (label `apps`): `sbmc_run` run from the repository root, its exit statuses (2 for
  the command line, among them `--threads` without `--variant c1` and out of [0, 1024]; 1 for the
  library and for reading and writing), the 44 columns and their values on the 4-node graph, a
  repetition against `sbmc_trace` on the same global replicas, appending to an existing CSV,
  `--target best-known` on G1 and with `--name`, a dense instance with a warm-up run, the lines on
  stdout; with `--variant c1`: the column `threads` equal to the request, the same rows with 1 and
  3 threads apart from the times, `--threads 0`, and in the strict presets (argument
  `--strict-fp 1`, passed by CMake) the same best cut, mean cut, successes and c0 as `c0` in dSB
  and bSB; final line `RUN_VERDICT`;
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
                     sb, sb_sequential, sb_openmp, bruteforce, results
src/core/            rudy reader, Ising model, int64 cut and energy, dense INT8 rows, initial
                     positions, pressure schedule and automatic c0, results schema and CSV, table
                     of best-known values (TTS statistics later)
src/cpu/             sequential reference C0 (phase 04), OpenMP simulated bifurcation C1 and
                     exact enumeration C3 with OpenMP (phase 05); simulated annealing later
src/gpu/             cuBLAS, cuBLASLt, bitwise and cuSPARSE products, update kernels
src/tile/            fused CUDA Tile kernel (sm_89 only)
src/mgpu/            MPI and NCCL multi-GPU drivers
apps/                command-line programs: sbmc_run (phase 04, C1 from phase 05), sbmc_bruteforce
                     (phase 05)
tests/               unit, integration and smoke tests (ctest)
tests/core/          unit tests of the core library (doctest)
tests/cpu/           unit tests of the CPU solvers (C0, C1 with T7, C3) and T2 (doctest)
tests/data/          input files of the tests
tests/gpu/           CUDA tests (test_curand_philox)
tests/python/        Python tests: P0 alone (test_p0), T8 (test_t8), the data (test_data), sbmc_run
                     (test_sbmc_run), sbmc_bruteforce (test_sbmc_bruteforce)
tests/smoke/         CUDA smoke tests and FMA probes
tests/scripts/       tool tests and checks (sync_from_zip.sh, build information, RPATH)
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

Existing in phase 05: `apps/`, `cmake/`, `data/`, `include/sbmc/`, `python/`, `scripts/`,
`src/core/`, `src/cpu/`, `tests/` (with `core/`, `cpu/`, `data/`, `gpu/`, `python/`, `scripts/`,
`smoke/`, `tools/`) and `third_party/`; the other folders come with later phases.

## License and third-party code

MIT License, see `LICENSE`. Vendored dependencies: doctest 2.5.3 (MIT), single header in
`third_party/doctest/`, and CLI11 2.7.2 (BSD-3-Clause), single header in `third_party/CLI11/`;
versions, sources and checksums in `third_party/README.md`. The G-set files in `data/gset/` are
third-party data, not covered by the license of the project (see `data/README.md`).
