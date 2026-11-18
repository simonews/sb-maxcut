// Exact Max-Cut by enumeration with Gray code (C3), the reference optimum of the domain tests on
// small graphs (test T2, reference document, section 8.2).
//
// Definitions. A configuration with s_0 = +1 has the code c = sum_{i=1}^{n-1} [s_i = -1] 2^(i-1),
// an integer in [0, 2^(n-1)). The configurations with s_0 = -1 are the opposites of these and
// have the same energy, so every partition of the nodes corresponds to exactly one configuration
// with s_0 = +1, and optimal_count is the number of optimal partitions. The canonical optimum is,
// among the configurations with s_0 = +1 and energy min_energy, the one with the smallest code.
//
// Algorithm (src/cpu/bruteforce.cpp): the 2^(n-1) codes are split into 2^p contiguous intervals
// (their p high bits fixed), distributed by an OpenMP loop with a dynamic schedule; every interval
// computes its first configuration from scratch and then visits its other codes in Gray-code
// order, one spin flip per step, with the energy and the local fields h_i = sum_j J_ij s_j updated
// in O(n) and in int64. The intervals are combined with exact integer operations, so the result
// does not depend on the number of threads or on how the codes are split.
//
// OpenMP: the function does not change the global OpenMP state (no omp_set_*): the number of
// threads is passed with the num_threads clause of the parallel region.
#pragma once

#include <cstdint>
#include <vector>

#include <sbmc/instance.hpp>

namespace sbmc {

// Largest n accepted by solve_bruteforce: 2^39 configurations, minutes on the laptop.
inline constexpr std::int64_t kBruteForceMaxSpins = 40;

struct BruteForceResult {
  std::int64_t n = 0;
  std::int64_t total_weight = 0;    // W
  std::int64_t min_energy = 0;      // minimum of E(s) over all 2^n configurations
  std::int64_t max_cut = 0;         // (W - min_energy) / 2, checked with cut_direct
  std::uint64_t optimal_count = 0;  // configurations with s_0 = +1 and E(s) = min_energy
  std::vector<Spin> best_spins;     // canonical optimum (s_0 = +1, smallest code), n spins
  int threads = 0;                  // threads of the parallel region that ran the enumeration
};

// Exact Max-Cut of an instance by Gray-code enumeration (C3), parallel with OpenMP.
//
// threads = 0: the default number of threads of OpenMP (omp_get_max_threads()); threads > 0: the
// number requested, passed with the num_threads clause. Preconditions, checked in this order and
// before any large allocation (InvalidArgument): threads >= 0; n = num_spins(inst), which checks
// n only, and n <= kBruteForceMaxSpins; W = total_weight(inst), which checks the whole instance.
// The check on n comes first because total_weight costs O(n^2) on a large DenseSpinGlass.
// min_energy, max_cut, optimal_count and best_spins do not depend on the number of threads;
// threads is the size of the team that ran the enumeration (it can be smaller than requested
// when the OpenMP runtime adjusts it). Final check with the public functions: cut_direct and
// ising_energy of best_spins must give max_cut and min_energy (sbmc::Error with "internal"
// otherwise).
BruteForceResult solve_bruteforce(const Instance& inst, int threads = 0);

}  // namespace sbmc
