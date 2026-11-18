// C1: simulated bifurcation (bSB and dSB) on the CPU with OpenMP over blocks of replicas and SIMD
// over the replicas of a block (reference document, sections 6, 7 and 9.7; decision H2).
//
// The R replicas are split into B = ceil(R / RB) blocks of RB consecutive replicas, RB =
// kSbOpenmpReplicasPerBlock. An OpenMP loop with schedule(dynamic, 1) distributes the blocks; a
// block runs all M steps on one thread, with its state (x, y, signs, fields) in a private work
// area that starts on a 64-byte boundary, in spin-major order: element (spin i, lane l) at
// i * RB + l, so that the RB replicas of a block are the lanes of the vector registers. The lanes
// of the last block beyond R are padding: computed like the others and discarded.
//
// Every lane runs the floating-point operations of C0 (sbmc/sb_sequential.hpp) in the same order:
// s(m) = sb_sign(x) before the updates of the step; dSB field (float)(sum_j J_ij s_j), exact in an
// integer accumulator; bSB field (float)(acc), acc = +0.0 in double plus (double)J_ij *
// (double)x_j for increasing j; the evaluations, the best configuration (ties: smallest m), the
// final evaluation of s(M) and sb_update as in C0. The internal copy of J is dense INT8 for a
// DenseSpinGlass and for a Graph with n <= 32768, |w| <= 127 and 8m >= n(n - 1) (density at least
// 25%), compressed rows otherwise; the integer accumulator is chosen from
// L = max_i sum_j |J_ij|: int16 when L <= 32767, int32 when L <= 2^31 - 1, int64 otherwise.
//
// OpenMP: the function does not change the global OpenMP state (no omp_set_*): the number of
// threads goes into the num_threads clause of its parallel region.
#pragma once

#include <sbmc/instance.hpp>
#include <sbmc/sb.hpp>

namespace sbmc {

// Replicas per block of C1, the SIMD lanes of the step loop: 8, 16 or 32. 32 is the fastest of
// the three in the measurements of phase 05 (README, section on C1).
inline constexpr int kSbOpenmpReplicasPerBlock = 32;

// C1: simulated bifurcation (bSB and dSB) with OpenMP over blocks of replicas and SIMD over the
// replicas of a block.
//
// Preconditions, checked in this order and before any large allocation (InvalidArgument, with a
// message that starts with "run_sb_openmp: " or, for the public functions it calls, with theirs):
// threads >= 0; then those of run_sb_sequential, with the same conditions in the same order (the
// parameters and trace_steps; n <= 32768 for a DenseSpinGlass; total_weight, which checks the
// instance; auto_c0 when c0 = 0).
//
// Threads: T_req = threads when threads > 0, otherwise omp_get_max_threads() read at the entry;
// the step loop runs in a region with num_threads(min(T_req, B)), and result.threads is the size
// of its team (min(T_req, B) when omp_get_dynamic() = 0).
//
// Result: every field of SbResult with the meaning it has in C0. In the strict builds (no FMA
// contraction) every field except threads is equal bit for bit to run_sb_sequential(inst, params,
// trace_steps), for every value of threads (test T7). In every build the result does not depend
// on threads, on the size of the team or on the order of the blocks, and the results of local
// replica r depend only on the instance, on the parameters other than replicas and first_replica,
// and on its global index first_replica + r. The best cut of every replica is checked with
// cut_direct, in parallel after its block (sbmc::Error with "internal" for the smallest replica
// with a mismatch).
SbResult run_sb_openmp(const Instance& inst, const SbParams& params, int threads = 0,
                       int trace_steps = 0);

}  // namespace sbmc
