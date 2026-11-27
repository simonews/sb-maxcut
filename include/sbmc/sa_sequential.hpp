// Sequential reference of simulated annealing (variant c0 of --algo sa): the oracle of the
// optimized variant C2, which must give the same results bit for bit.
//
// Preparation: W = total_weight(inst), which checks the instance; an internal copy of J
// (compressed rows for a Graph, dense INT8 for a DenseSpinGlass). For every local replica
// r = 0, ..., R - 1, with global index g = first_replica + r, one after the other:
//   s_i = sa_initial_spin(seed, g, i); h_i = sum_j J_ij s_j (int64, exact);
//   E = -(sum_i s_i h_i) / 2; best_E = E, best_sweep = 0, best spins = s, accepted = 0;
//   for k = 0 .. S - 1, with beta = sa_beta(beta0, beta1, S, k), for i = 0 .. n - 1:
//     w = sa_acceptance_word(seed, g, k * n + i), consumed by every proposal;
//     dE = 2 s_i h_i;
//     if dE <= 0 or w < sa_threshold(beta, dE): h_j -= 2 J_ji s_i for every j (old s_i), then
//       s_i = -s_i, E += dE, accepted += 1, and if E < best_E: best_E = E,
//       best_sweep = k + 1, best spins = s (ties keep the earlier state);
//   final_cut = (W - E) / 2, best_cut = (W - best_E) / 2.
// Final check of every replica (decision L3 of phase 07): the cuts of the best spins and of the
// final spins, recomputed directly without a second validation of the instance (on the edges of a
// Graph or on the internal INT8 rows of a DenseSpinGlass), must equal best_cut and final_cut
// (sbmc::Error with "internal" on mismatch). No OpenMP, and no public function of the instance
// after the preparation (B5). No floating-point sum: the results are the same in every preset.
// Times (result.times, sbmc/solve_times.hpp): preparation (the checks, the internal J, the outputs,
// the initial spins of every replica), run (the replica loop), check (the final check).
#pragma once

#include <sbmc/instance.hpp>
#include <sbmc/sa.hpp>

namespace sbmc {

// Sequential reference of simulated annealing (variant c0 of --algo sa): the oracle of C2.
// Preconditions, in this order, all checked before the large buffers are allocated
// (InvalidArgument, message "run_sa_sequential: ..." except the errors of total_weight):
// 1. replicas >= 1; sweeps >= 1; beta0 finite and > 0; beta1 finite and >= beta0;
//    first_replica >= 0; first_replica + replicas <= 2^32;
// 2. n <= 32768 for a DenseSpinGlass (J stored as dense INT8, at most 1 GiB);
// 3. a valid instance (total_weight).
// Result: per_replica, best_spins and final_spins in replica-major order; threads = 1.
SaResult run_sa_sequential(const Instance& inst, const SaParams& params);

}  // namespace sbmc
