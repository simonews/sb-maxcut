// Initial positions x0 of simulated bifurcation (reference document, sections 6 and 8.3).
//
// x0 is uniform in [-0.1f, 0.1f) and y0 = 0. The position of spin i in the replica with global
// index r comes from word i of the stream (run seed, RngDomain::sb_initial_positions, r): it
// depends only on (seed, r, i), not on threads, devices or on how the replicas are split, and it
// has the same bits in every preset, on the host and on the device. The initial conditions are
// generated once on the host and shared by every variant.
#pragma once

#include <cstdint>
#include <vector>

#include <sbmc/philox.hpp>

namespace sbmc {

// x0 of spin `spin` in the replica with global index `replica`.
SBMC_HD constexpr float initial_position(std::uint64_t seed, std::uint32_t replica,
                                         std::uint32_t spin) noexcept {
  return position_from_word(stream_word(seed, RngDomain::sb_initial_positions, replica, spin));
}

struct InitialState {
  std::int64_t n = 0;
  int replicas = 0;
  std::int64_t first_replica = 0;  // global index of local replica 0 (M1)
  std::vector<float> x0;           // replica-major: x0[r * n + i]; y0 = 0
};

// x0 of `replicas` replicas with global indices first_replica, ..., first_replica + replicas - 1:
// x0[r * n + i] = initial_position(seed, first_replica + r, i). Preconditions (InvalidArgument
// otherwise): 1 <= n <= 2^31 - 1; replicas >= 1; first_replica >= 0;
// first_replica + replicas <= 2^32; n * replicas representable as the size of a
// std::vector<float>. Splitting the replicas over several calls with first_replica gives the
// same values. Computed with OpenMP; the result does not depend on the number of threads.
InitialState make_initial_state(std::int64_t n, int replicas, std::uint64_t seed,
                                std::int64_t first_replica = 0);

}  // namespace sbmc
