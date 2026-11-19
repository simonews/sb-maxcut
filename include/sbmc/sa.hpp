// Simulated annealing: the parts common to every variant (reference document, section 7).
//
// Single-spin Metropolis on the Ising energy E(s) = sum_{i<j} w_ij s_i s_j (sbmc/instance.hpp),
// with sequential sweeps: sweep k (0 <= k < S) proposes the flip of spin 0, 1, ..., n - 1 in this
// order at the inverse temperature beta_k = sa_beta(beta0, beta1, S, k), geometric from beta0 to
// beta1. The flip of spin i changes the energy by delta_e = 2 s_i h_i, with the local field
// h_i = sum_j J_ij s_j kept up to date in int64; it is accepted when delta_e <= 0, or when the
// acceptance word of the proposal is below sa_threshold(beta_k, delta_e).
//
// Random numbers (sbmc/philox.hpp): the initial spin i of the replica with global index r is
// word i of the stream (seed, RngDomain::sa_initial_spins, r); the acceptance word of proposal
// p = k * n + i is word p of the stream (seed, RngDomain::sa_acceptance, r), consumed by every
// proposal, accepted or not. A replica therefore depends only on (instance, S, beta0, beta1,
// seed, r), not on threads, devices or on how the replicas are split.
//
// The only floating-point operations are those of sa_beta and sa_threshold, without sums, so no
// FMA contraction is possible: they are defined once, in sbmc_core, and every variant calls
// them, so that the results are the same bit for bit in every preset.
#pragma once

#include <cstdint>
#include <vector>

#include <sbmc/instance.hpp>
#include <sbmc/philox.hpp>

namespace sbmc {

struct SaParams {
  int replicas = 1;                // R >= 1
  int sweeps = 1;                  // S >= 1
  double beta0 = 0.0;              // finite, > 0 (no default: the caller chooses it)
  double beta1 = 0.0;              // finite, >= beta0
  std::uint64_t seed = 0;          // key of the domains sa_initial_spins and sa_acceptance
  std::int64_t first_replica = 0;  // global index of local replica 0
};

struct SaReplicaResult {
  std::int64_t best_energy = 0;  // lowest energy visited (initial state included)
  std::int64_t best_cut = 0;     // (W - best_energy) / 2
  int best_sweep = 0;            // 0: the initial state; k: found during sweep k (1..S)
  std::int64_t final_cut = 0;    // cut of the state after the last sweep
  std::int64_t accepted = 0;     // accepted moves
};

struct SaResult {
  std::int64_t n = 0;
  int replicas = 0;
  int sweeps = 0;
  std::int64_t first_replica = 0;
  double beta0 = 0.0;
  double beta1 = 0.0;
  std::int64_t total_weight = 0;
  std::vector<SaReplicaResult> per_replica;  // R entries
  std::vector<Spin> best_spins;              // replica-major, R * n
  std::vector<Spin> final_spins;             // replica-major, R * n
  int threads = 1;                           // threads of the sweep loop (1 for the reference)
};

// beta of sweep k (0 <= k < sweeps): beta0 * pow(beta1 / beta0, k / (sweeps - 1)), beta0 when
// sweeps = 1.
// Exactly, in double: t = (sweeps > 1) ? double(k) / double(sweeps - 1) : 0.0, then
// beta0 * std::pow(beta1 / beta0, t). InvalidArgument (message "sa_beta: ...") unless
// sweeps >= 1, 0 <= k < sweeps, beta0 finite and > 0, beta1 finite and >= beta0.
double sa_beta(double beta0, double beta1, int sweeps, int k);

// Acceptance threshold of an uphill move (delta_e > 0): floor(exp(-(beta * delta_e)) * 2^32), in
// [0, 2^32]. The move is accepted when the acceptance word is below the threshold.
// Exactly, in double: p = std::exp(-(beta * static_cast<double>(delta_e))), then
// static_cast<std::uint64_t>(p * 4294967296.0). InvalidArgument (message "sa_threshold: ...")
// unless beta is finite and > 0 and delta_e > 0.
std::uint64_t sa_threshold(double beta, std::int64_t delta_e);

// Initial spin i of global replica r: word i of the stream (seed, sa_initial_spins, r), +1 when
// its top bit is 0 (coupling_from_word).
SBMC_HD constexpr Spin sa_initial_spin(std::uint64_t seed, std::uint32_t replica,
                                       std::uint32_t spin) noexcept {
  return coupling_from_word(stream_word(seed, RngDomain::sa_initial_spins, replica, spin));
}

// Acceptance word of proposal p = k * n + i (sweep k, spin i) of global replica r: word p of the
// stream (seed, sa_acceptance, r). Every proposal consumes its word, accepted or not.
SBMC_HD constexpr std::uint32_t sa_acceptance_word(std::uint64_t seed, std::uint32_t replica,
                                                   std::uint64_t proposal) noexcept {
  return stream_word(seed, RngDomain::sa_acceptance, replica, proposal);
}

}  // namespace sbmc
