// Times of the solvers (phase 07, decision L2): every solver of simulated bifurcation and of
// simulated annealing (run_sb_sequential, run_sb_openmp, run_sa_sequential, run_sa_openmp)
// measures inside the call three consecutive phases and returns them in the field times of its
// result (SbResult, SaResult). Preparation: preconditions, the only full check of the instance,
// the internal copy of J, the allocation of the outputs and of the work areas, the initial states
// of every replica. Run: the steps or sweeps of every replica from its initial state, the best
// states and the outputs. Check: the final check of the cuts of every replica. The times are
// written only before a normal return. Standard C++ only: no CUDA type or header.
#pragma once

namespace sbmc {

// Durations of the three phases of a call of a solver, in milliseconds, measured with
// std::chrono::steady_clock at four instants t0 <= t1 <= t2 <= t3 of the call:
// prep_ms = t1 - t0, run_ms = t2 - t1, check_ms = t3 - t2. The intervals are consecutive, so
// prep_ms + run_ms + check_ms <= the duration of the call measured outside it.
struct SolveTimes {
  double prep_ms = 0.0;   // preparation: preconditions, check of the instance, internal J, ...
  double run_ms = 0.0;    // run: the steps or sweeps of every replica, best states, outputs
  double check_ms = 0.0;  // final check of the cuts of every replica
};

}  // namespace sbmc
