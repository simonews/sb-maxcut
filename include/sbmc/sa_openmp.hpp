// C2: simulated annealing on the CPU with OpenMP over blocks of replicas and SIMD over the replicas
// of a block (reference document, section 7; decision S2 (b)).
//
// The R replicas are split into B = ceil(R / RB) blocks of RB consecutive replicas, RB =
// kSaOpenmpReplicasPerBlock. An OpenMP loop with schedule(dynamic, 1) distributes the blocks; a
// block runs all S sweeps on one thread, with its state (signs, local fields) in a private work
// area that starts on a 64-byte boundary. The lanes of the last block beyond R are padding: they
// follow the stream of lane 0 and are discarded. Two layouts of the work area (phase 07, decision
// L5): with compressed rows, spin-major, element (spin i, lane l) at i * RB + l, so that the RB
// replicas of a block are the lanes of the vector registers; with dense rows (J dense INT8),
// lane-major, element (lane l, spin c) at l * ld + c, ld = n rounded up to a multiple of 64 bytes
// in elements of the accumulator, every lane on a 64-byte boundary.
//
// Every lane runs the Metropolis sweeps of the reference (sbmc/sa_sequential.hpp) in the same
// order, with the acceptance word of every proposal at its fixed position of the stream: the
// trajectory of a replica is that of the reference. The decisions of the RB lanes for spin i are
// taken together; then, with compressed rows, the local fields of the neighbours are updated with a
// mask (the lanes that rejected add 0); with dense rows, only the lanes that accepted update their
// fields, each along its contiguous row (the lanes that rejected do not touch them). The internal
// copy of J follows the rule of C1: dense INT8 for a DenseSpinGlass and for
// a Graph with n <= 32768, |w| <= 127 and 8m >= n(n - 1), compressed rows otherwise; the
// accumulator of the local fields is chosen from L = max_i sum_j |J_ij|: int16 when L <= 32767,
// int32 when L <= 2^31 - 1, int64 otherwise. The thresholds of the uphill moves come only from
// sa_threshold: from a table of the values for delta_e = 2, 4, ..., 2L, filled for every sweep of
// every block, when L <= L_tab = 4096; from a direct call for every uphill proposal otherwise. The
// best signs of a lane are rebuilt at the end of the block by replaying the sweep of its last
// improvement from the signs at the start of that sweep.
//
// OpenMP: the function does not change the global OpenMP state (no omp_set_*): the number of
// threads goes into the num_threads clause of its parallel region.
#pragma once

#include <sbmc/instance.hpp>
#include <sbmc/sa.hpp>

namespace sbmc {

// Replicas per block of C2, the SIMD lanes of the sweep loop: 8, 16, 32 or 64. 32 is the fastest
// of the four on K2000 and ties with 16 and 64 on G1 in the measurements of phase 06 (README,
// section on C2).
inline constexpr int kSaOpenmpReplicasPerBlock = 32;

// C2: simulated annealing with OpenMP over blocks of replicas and SIMD over the replicas of a
// block; the same results as run_sa_sequential, bit for bit.
//
// Preconditions, checked in this order and before any large allocation (InvalidArgument, with a
// message that starts with "run_sa_openmp: ", except the errors of total_weight): threads >= 0;
// then those of run_sa_sequential, with the same conditions in the same order (the parameters;
// n <= 32768 for a DenseSpinGlass; total_weight, which checks the instance).
//
// Threads: T_req = threads when threads > 0, otherwise omp_get_max_threads() read at the entry;
// the sweep loop runs in a region with num_threads(min(T_req, B)), and result.threads is the size
// of its team (min(T_req, B) when omp_get_dynamic() = 0).
//
// Result: in every preset, strict or not, every field except threads is equal to
// run_sa_sequential(inst, params) (the bits of beta0 and beta1 included), for every value of
// threads; so the result does not depend on threads, on the size of the team, on the order of the
// blocks, on the layout of the work area, or on how the replicas are split between calls. Final
// check (decision L3 of phase 07): the best and the final cut of every replica are recomputed
// directly, without a second validation of the instance, on the edges of a Graph (whatever the
// format of its internal J) or on the internal INT8 rows of a DenseSpinGlass, in a second parallel
// region after the region of the sweeps, with num_threads(min(T_req, B)) (sbmc::Error with
// "internal" for the smallest replica with a mismatch). Times (result.times,
// sbmc/solve_times.hpp): preparation (the checks, beta_k, the internal J, the work areas, the
// initial spins of every replica), run (the region of the sweeps), check (the region of the final
// check).
SaResult run_sa_openmp(const Instance& inst, const SaParams& params, int threads = 0);

}  // namespace sbmc
