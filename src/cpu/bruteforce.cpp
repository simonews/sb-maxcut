// Exact Max-Cut by Gray-code enumeration (C3, sbmc/bruteforce.hpp).
//
// Preparation, outside the parallel region: the preconditions; J as a dense n x n int64 matrix
// with a zero diagonal; for the inner loop, the rows of -2 J and +2 J padded with zeros to a width
// K, the multiple of 8 between 8 and 40 that holds n, so that the update of the fields runs over a
// fixed number of elements and vectorises. Parallel region: one OpenMP loop over the 2^p
// intervals of codes, schedule(dynamic, 1); the interval t writes only element t of a vector of
// partial results, combined after the region in the order of t with exact integer operations.
// Inside the region no public function of the instance is called (B5) and nothing allocates; the
// global OpenMP state is never changed (the number of threads goes into num_threads).
#include <sbmc/bruteforce.hpp>

#include <omp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>

namespace sbmc {
namespace {

static_assert(kBruteForceMaxSpins >= 1 && kBruteForceMaxSpins <= 40,
              "the widths K of the flip tables cover n <= 40");

// The number of intervals is 2^p, with p the smallest value such that 2^p >= 8 * threads,
// at most q = n - 1 (one code per interval) and at most 2^16 intervals (a bound on the vector of
// partial results for absurd thread counts).
constexpr std::int64_t kIntervalsPerThread = 8;
constexpr int kMaxIntervalBits = 16;

[[noreturn]] void invalid(const std::string& reason) {
  throw InvalidArgument("solve_bruteforce: " + reason);
}

// J of a valid graph, dense n x n in int64: J_uv = J_vu = -w_uv, zero elsewhere.
std::vector<std::int64_t> dense_couplings(const Graph& g) {
  const auto n = static_cast<std::size_t>(g.n);
  std::vector<std::int64_t> j(n * n, 0);
  for (const Edge& e : g.edges) {
    const auto u = static_cast<std::size_t>(e.u);
    const auto v = static_cast<std::size_t>(e.v);
    j[u * n + v] = -std::int64_t{e.w};
    j[v * n + u] = -std::int64_t{e.w};
  }
  return j;
}

// J of a dense spin glass, dense n x n in int64: one call of dense_rows_int8, whose elements are
// J_ij = coupling(g, i, j) with a zero diagonal.
std::vector<std::int64_t> dense_couplings(const DenseSpinGlass& g) {
  const auto n = static_cast<std::size_t>(g.n);
  std::vector<std::int8_t> rows(n * n);
  dense_rows_int8(g, 0, g.n, g.n, rows);
  return {rows.begin(), rows.end()};
}

// Minimum of one interval of codes: its energy, how many of its codes reach it and the smallest of
// those codes.
struct IntervalResult {
  std::int64_t min_energy = 0;
  std::uint64_t count = 0;
  std::uint64_t best_code = 0;
};

// Rows of the field updates for the flip of spin a: minus_two_j[a][j] = -2 J_ja when s_a = +1
// before the flip, plus_two_j[a][j] = +2 J_ja when s_a = -1; zero for j >= n.
template <int K>
struct FlipTables {
  std::vector<std::array<std::int64_t, K>> minus_two_j;
  std::vector<std::array<std::int64_t, K>> plus_two_j;
};

template <int K>
FlipTables<K> make_flip_tables(std::span<const std::int64_t> j, int n) {
  FlipTables<K> tables;
  const auto un = static_cast<std::size_t>(n);
  tables.minus_two_j.assign(un, std::array<std::int64_t, K>{});
  tables.plus_two_j.assign(un, std::array<std::int64_t, K>{});
  for (std::size_t a = 0; a < un; ++a) {
    for (std::size_t c = 0; c < un; ++c) {
      tables.minus_two_j[a][c] = -2 * j[c * un + a];
      tables.plus_two_j[a][c] = 2 * j[c * un + a];
    }
  }
  return tables;
}

// Interval t: the codes [t 2^low_bits, (t + 1) 2^low_bits). The first configuration is computed
// from scratch (fields h_i = sum_j J_ij s_j and E = -(sum_i s_i h_i) / 2); then step g = 1, ...,
// 2^low_bits - 1 flips spin a = countr_zero(g) + 1, that is bit a - 1 of the code: with the values
// before the flip, E += 2 s_a h_a and h_j -= 2 J_ja s_a for every j, then s_a = -s_a. After step g
// the low bits of the code are the Gray code g ^ (g >> 1), so every code of the interval is
// visited once. Every visited configuration, the first included, is compared with the minimum.
template <int K>
IntervalResult enumerate_interval(std::span<const std::int64_t> j, int n,
                                  const FlipTables<K>& tables, int low_bits,
                                  std::uint64_t t) noexcept {
  const auto un = static_cast<std::size_t>(n);
  const std::uint64_t base = t << low_bits;
  std::array<std::int64_t, K> s{};  // s_i in {-1, +1} for i < n, 0 in the padding
  std::array<std::int64_t, K> h{};  // h_i for i < n, 0 in the padding
  s[0] = 1;
  for (std::size_t i = 1; i < un; ++i) {
    s[i] = ((base >> (i - 1)) & 1u) != 0 ? -1 : 1;
  }
  std::int64_t sum = 0;
  for (std::size_t i = 0; i < un; ++i) {
    std::int64_t field = 0;
    for (std::size_t c = 0; c < un; ++c) {
      field += j[i * un + c] * s[c];
    }
    h[i] = field;
    sum += s[i] * field;
  }
  std::int64_t energy = -sum / 2;  // sum = 2 sum_{i<j} J_ij s_i s_j is even
  IntervalResult result{energy, 1, base};

  const std::uint64_t codes = std::uint64_t{1} << low_bits;
  for (std::uint64_t g = 1; g < codes; ++g) {
    const auto a = static_cast<std::size_t>(std::countr_zero(g)) + 1;
    const std::int64_t s_a = s[a];
    energy += 2 * s_a * h[a];
    const std::array<std::int64_t, K>& delta =
        s_a > 0 ? tables.minus_two_j[a] : tables.plus_two_j[a];
    for (std::size_t c = 0; c < static_cast<std::size_t>(K); ++c) {
      h[c] += delta[c];
    }
    s[a] = -s_a;
    if (energy <= result.min_energy) {
      const std::uint64_t code = base | (g ^ (g >> 1));
      if (energy < result.min_energy) {
        result = {energy, 1, code};
      } else {
        ++result.count;
        result.best_code = std::min(result.best_code, code);
      }
    }
  }
  return result;
}

int interval_bits(int free_spins, int threads) {
  const std::int64_t target = kIntervalsPerThread * threads;
  int p = 0;
  while (p < free_spins && p < kMaxIntervalBits && (std::int64_t{1} << p) < target) {
    ++p;
  }
  return p;
}

// Runs the parallel enumeration and returns the partial results of the intervals, in the order of
// the intervals; team receives the size of the team of the parallel region.
template <int K>
std::vector<IntervalResult> enumerate(std::span<const std::int64_t> j, int n, int threads,
                                      int& team) {
  const FlipTables<K> tables = make_flip_tables<K>(j, n);
  const int free_spins = n - 1;
  const int p = interval_bits(free_spins, threads);
  const int low_bits = free_spins - p;
  const std::int64_t intervals = std::int64_t{1} << p;
  std::vector<IntervalResult> partial(static_cast<std::size_t>(intervals));

  int team_size = 0;
#pragma omp parallel num_threads(threads)
  {
#pragma omp single
    team_size = omp_get_num_threads();

#pragma omp for schedule(dynamic, 1)
    for (std::int64_t t = 0; t < intervals; ++t) {
      partial[static_cast<std::size_t>(t)] =
          enumerate_interval<K>(j, n, tables, low_bits, static_cast<std::uint64_t>(t));
    }
  }
  team = team_size;
  return partial;
}

std::vector<IntervalResult> enumerate_all(std::span<const std::int64_t> j, int n, int threads,
                                          int& team) {
  switch ((n + 7) / 8) {
    case 1:
      return enumerate<8>(j, n, threads, team);
    case 2:
      return enumerate<16>(j, n, threads, team);
    case 3:
      return enumerate<24>(j, n, threads, team);
    case 4:
      return enumerate<32>(j, n, threads, team);
    default:
      return enumerate<40>(j, n, threads, team);
  }
}

}  // namespace

BruteForceResult solve_bruteforce(const Instance& inst, int threads) {
  if (threads < 0) {
    invalid("threads = " + std::to_string(threads) +
            " must be >= 0 (0: the default number of OpenMP threads)");
  }
  const std::int64_t n = num_spins(inst);  // checks n only
  if (n > kBruteForceMaxSpins) {
    invalid("n = " + std::to_string(n) + " exceeds " + std::to_string(kBruteForceMaxSpins) +
            ", the largest number of spins of the exact enumeration");
  }
  const std::int64_t total = total_weight(inst);  // full check of the instance

  const std::vector<std::int64_t> j =
      std::visit([](const auto& g) { return dense_couplings(g); }, inst);
  const int requested = threads > 0 ? threads : omp_get_max_threads();
  int team = 0;
  const std::vector<IntervalResult> partial =
      enumerate_all(j, static_cast<int>(n), requested, team);

  // Exact combination: the smallest minimum, the sum of the counts of the intervals that reach it
  // and the smallest of their codes.
  IntervalResult best = partial.front();
  for (std::size_t t = 1; t < partial.size(); ++t) {
    const IntervalResult& r = partial[t];
    if (r.min_energy < best.min_energy) {
      best = r;
    } else if (r.min_energy == best.min_energy) {
      best.count += r.count;
      best.best_code = std::min(best.best_code, r.best_code);
    }
  }

  BruteForceResult result;
  result.n = n;
  result.total_weight = total;
  result.min_energy = best.min_energy;
  result.max_cut = (total - best.min_energy) / 2;
  result.optimal_count = best.count;
  result.threads = team;
  result.best_spins.assign(static_cast<std::size_t>(n), Spin{1});
  for (std::int64_t i = 1; i < n; ++i) {
    if (((best.best_code >> (i - 1)) & 1u) != 0) {
      result.best_spins[static_cast<std::size_t>(i)] = Spin{-1};
    }
  }

  // Final check with the public functions.
  const std::int64_t cut = cut_direct(inst, result.best_spins);
  const std::int64_t energy = ising_energy(inst, result.best_spins);
  if (cut != result.max_cut || energy != result.min_energy) {
    throw Error("solve_bruteforce: internal error: cut_direct gives " + std::to_string(cut) +
                " and ising_energy " + std::to_string(energy) + ", the enumeration gives max_cut " +
                std::to_string(result.max_cut) + " and min_energy " +
                std::to_string(result.min_energy));
  }
  return result;
}

}  // namespace sbmc
