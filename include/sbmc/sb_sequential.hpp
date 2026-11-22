// Sequential reference C0 of simulated bifurcation (bSB and dSB), in FP32 with a fixed order of
// the floating-point operations: the bit-for-bit reference of every other variant in the strict
// builds (test T7).
//
// Preparation: W = total_weight(inst), which checks the instance; an internal copy of J (compressed
// rows for a Graph, dense INT8 for a DenseSpinGlass); c0 = params.c0, or auto_c0(inst) when
// params.c0 is 0; p = pressure_schedule(a0, M); dt_a0 = dt * a0 in float; x = x0 of
// make_initial_state(n, R, seed, first_replica), y = 0. Step m = 0, ..., M - 1, for every replica:
// 1. s(m) = sb_sign(x) for every spin, before any update of the step;
// 2. field: dSB, F_i = sum_j J_ij s_j in int64 and field_i = (float)F_i; bSB, acc = +0.0 in
//    double, acc += (double)J_ij * (double)x_j for increasing j, field_i = (float)acc;
// 3. evaluation of s(m): dSB always, with E = -(sum_i s_i F_i) / 2; bSB only when
//    eval_every > 0 and m mod eval_every == 0, with E(s(m)) in int64. A configuration becomes the
//    best one when it is the first evaluated or when E < best_energy (ties: smallest m);
// 4. sb_update of every spin with p[m];
// 5. if m < trace_steps, x and y are recorded as the state after step m + 1.
// After the last step: s(M) is evaluated (every algorithm, every eval_every);
// final_cut = (W - E(s(M))) / 2 and best_cut = (W - best_energy) / 2. Final check (decision L3 of
// phase 07): the best cut of every replica is recomputed directly, without a second validation
// of the instance, on the edges of a Graph or on the internal INT8 rows of a DenseSpinGlass
// (sbmc::detail::cut_edges_unchecked, cut_dense_rows_unchecked), and must equal best_cut
// (sbmc::Error with "internal" on mismatch). No OpenMP in C0 itself, and no public function of the
// instance after the preparation. Times (sbmc/solve_times.hpp, result.times): preparation from the
// entry to the replica loop (the checks, the internal J, c0, p, the outputs and x0 of every
// replica), run (the replica loop), check (the final check).
#pragma once

#include <sbmc/instance.hpp>
#include <sbmc/sb.hpp>

namespace sbmc {

// Sequential reference C0. Preconditions, all checked before the large buffers are allocated
// (InvalidArgument): a valid instance; algorithm ballistic or discrete; replicas >= 1;
// steps >= 1; dt and a0 finite and > 0; c0 finite and >= 0 (0, also -0.0f, means automatic);
// eval_every >= 0; first_replica >= 0 and first_replica + replicas <= 2^32;
// 0 <= trace_steps <= steps; n <= 32768 for a DenseSpinGlass (C0 stores J as dense INT8, at most
// 1 GiB); with c0 = 0, auto_c0 must be defined. trace_x and trace_y are empty when
// trace_steps = 0.
SbResult run_sb_sequential(const Instance& inst, const SbParams& params, int trace_steps = 0);

}  // namespace sbmc
