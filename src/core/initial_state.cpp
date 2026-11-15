// Initial positions of simulated bifurcation (sbmc/initial_state.hpp).
#include <sbmc/initial_state.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/philox.hpp>

namespace sbmc {
namespace {

constexpr std::int64_t kMaxSpins = std::numeric_limits<std::int32_t>::max();  // 2^31 - 1
constexpr std::int64_t kReplicaIndexLimit = std::int64_t{1} << 32;            // 2^32

[[noreturn]] void invalid(const std::string& reason) {
  throw InvalidArgument("make_initial_state: " + reason);
}

}  // namespace

// Words 4b, ..., 4b + 3 of a replica's stream are one Philox block: the loop runs over the
// (replica, block) pairs, distributed with OpenMP. Every element depends only on
// (seed, global replica, spin), so the result does not depend on the number of threads.
InitialState make_initial_state(std::int64_t n, int replicas, std::uint64_t seed,
                                std::int64_t first_replica) {
  if (n < 1 || n > kMaxSpins) {
    invalid("number of spins n = " + std::to_string(n) + " is outside [1, 2147483647]");
  }
  if (replicas < 1) {
    invalid("number of replicas " + std::to_string(replicas) + " is not positive");
  }
  if (first_replica < 0) {
    invalid("first_replica = " + std::to_string(first_replica) + " is negative");
  }
  if (first_replica > kReplicaIndexLimit - replicas) {
    invalid("first_replica + replicas = " + std::to_string(first_replica) + " + " +
            std::to_string(replicas) + " exceeds 2^32: global replica indices are 32-bit");
  }
  const auto elements = static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(replicas);
  InitialState state;
  if (elements > static_cast<std::uint64_t>(state.x0.max_size())) {
    invalid("n * replicas = " + std::to_string(elements) +
            " elements do not fit in a std::vector<float>");
  }

  state.n = n;
  state.replicas = replicas;
  state.first_replica = first_replica;
  state.x0.resize(static_cast<std::size_t>(elements));

  float* const x0 = state.x0.data();
  const std::int64_t blocks_per_replica = (n + 3) / 4;
  const std::int64_t tasks = blocks_per_replica * replicas;
#pragma omp parallel for schedule(static)
  for (std::int64_t task = 0; task < tasks; ++task) {
    const std::int64_t r = task / blocks_per_replica;
    const std::int64_t b = task % blocks_per_replica;
    const auto replica = static_cast<std::uint32_t>(first_replica + r);
    const U32x4 block =
        stream_block(seed, RngDomain::sb_initial_positions, replica, static_cast<std::uint64_t>(b));
    const std::uint32_t words[4] = {block.x, block.y, block.z, block.w};
    const std::int64_t i0 = 4 * b;
    const std::int64_t count = std::min<std::int64_t>(4, n - i0);
    float* const out = x0 + r * n + i0;
    for (std::int64_t t = 0; t < count; ++t) {
      out[t] = position_from_word(words[t]);  // = initial_position(seed, replica, i0 + t)
    }
  }
  return state;
}

}  // namespace sbmc
