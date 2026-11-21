// Ising model of Max-Cut on explicit graphs and on implicit dense spin glasses: validation, int64
// sums, dense INT8 rows of J and the unchecked cut of the final check of the solvers
// (sbmc/instance.hpp).
#include <sbmc/instance.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <variant>

#include <sbmc/error.hpp>
#include <sbmc/philox.hpp>

namespace sbmc {
namespace {

constexpr std::int64_t kMaxSpins = std::numeric_limits<std::int32_t>::max();  // 2^31 - 1
constexpr std::int32_t kMaxAbsInt8Weight = 127;

[[noreturn]] void invalid(const char* function, const std::string& reason) {
  throw InvalidArgument(std::string(function) + ": " + reason);
}

void check_spin_count(std::int64_t n, const char* function) {
  if (n < 1 || n > kMaxSpins) {
    invalid(function, "number of spins n = " + std::to_string(n) + " is outside [1, 2147483647]");
  }
}

std::string edge_text(std::size_t index, const Edge& e) {
  return "edge " + std::to_string(index) + " (u = " + std::to_string(e.u) +
         ", v = " + std::to_string(e.v) + ", w = " + std::to_string(e.w) + ")";
}

// Full check of a Graph, O(M): n in [1, 2^31 - 1]; 0 <= u < v < n, w != 0 and w != INT32_MIN on
// every edge; edges in strictly increasing (u, v) order.
void check_graph(const Graph& g, const char* function) {
  check_spin_count(g.n, function);
  for (std::size_t k = 0; k < g.edges.size(); ++k) {
    const Edge& e = g.edges[k];
    if (e.u < 0 || e.u >= e.v || e.v >= g.n) {
      invalid(function, edge_text(k, e) + ": expected 0 <= u < v < n = " + std::to_string(g.n));
    }
    if (e.w == 0) {
      invalid(function, edge_text(k, e) + ": the weight must not be 0");
    }
    if (e.w == std::numeric_limits<std::int32_t>::min()) {
      invalid(function, edge_text(k, e) + ": the weight must not be INT32_MIN");
    }
    if (k > 0) {
      const Edge& prev = g.edges[k - 1];
      if (prev.u > e.u || (prev.u == e.u && prev.v >= e.v)) {
        invalid(function, edge_text(k, e) + " does not follow " + edge_text(k - 1, prev) +
                              ": edges must be in strictly increasing (u, v) order");
      }
    }
  }
}

void check_dense(const DenseSpinGlass& g, const char* function) {
  check_spin_count(g.n, function);
}

void check_spins(std::span<const Spin> s, std::int64_t n, const char* function) {
  if (s.size() != static_cast<std::size_t>(n)) {
    invalid(function, "the spin vector has " + std::to_string(s.size()) +
                          " elements, expected n = " + std::to_string(n));
  }
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (s[i] != 1 && s[i] != -1) {
      invalid(function, "spin s[" + std::to_string(i) + "] = " +
                            std::to_string(static_cast<int>(s[i])) + " is neither -1 nor +1");
    }
  }
}

// Preconditions of dense_rows_int8 on the shape: n <= ld <= 2^31 - 1,
// 0 <= row_begin <= row_end <= ld, out_size >= (row_end - row_begin) * ld.
void check_rows(std::int64_t n, std::int64_t row_begin, std::int64_t row_end, std::int64_t ld,
                std::size_t out_size, const char* function) {
  if (ld < n || ld > kMaxSpins) {
    invalid(function, "the leading dimension ld = " + std::to_string(ld) +
                          " is outside [n = " + std::to_string(n) + ", 2147483647]");
  }
  if (row_begin < 0 || row_begin > row_end || row_end > ld) {
    invalid(function,
            "the row range [" + std::to_string(row_begin) + ", " + std::to_string(row_end) +
                ") does not satisfy 0 <= row_begin <= row_end <= ld = " + std::to_string(ld));
  }
  const auto needed =
      static_cast<std::uint64_t>(row_end - row_begin) * static_cast<std::uint64_t>(ld);
  if (static_cast<std::uint64_t>(out_size) < needed) {
    invalid(function, "the output has " + std::to_string(out_size) +
                          " elements, (row_end - row_begin) * ld = " + std::to_string(needed) +
                          " are needed");
  }
}

// J_ij of the dense spin glass with this seed, for 0 <= i, j < n (not checked).
std::int8_t dense_coupling(std::uint64_t seed, std::int64_t i, std::int64_t j) noexcept {
  if (i == j) {
    return 0;
  }
  const std::int64_t lo = std::min(i, j);
  const std::int64_t hi = std::max(i, j);
  return coupling_from_word(stream_word(seed, RngDomain::couplings, static_cast<std::uint32_t>(lo),
                                        static_cast<std::uint64_t>(hi)));
}

// Sum over the pairs i < j of J_ij * term(i, j), with term(i, j) an int64. The strict upper part
// of row i is the run of words i + 1, ..., n - 1 of stream i, four per Philox block. Rows are
// distributed with OpenMP; the sum is exact in int64, so the result does not depend on the
// number of threads or on the order of the partial sums.
template <typename Term>
std::int64_t sum_upper_pairs(const DenseSpinGlass& g, const Term& term) {
  const std::int64_t n = g.n;
  const std::uint64_t seed = g.seed;
  std::int64_t total = 0;
#pragma omp parallel for schedule(dynamic, 8) reduction(+ : total)
  for (std::int64_t i = 0; i < n - 1; ++i) {
    const auto stream = static_cast<std::uint32_t>(i);
    std::int64_t row = 0;
    for (std::int64_t b = (i + 1) / 4; 4 * b < n; ++b) {
      const U32x4 block =
          stream_block(seed, RngDomain::couplings, stream, static_cast<std::uint64_t>(b));
      const std::uint32_t words[4] = {block.x, block.y, block.z, block.w};
      for (std::int64_t t = 0; t < 4; ++t) {
        const std::int64_t j = 4 * b + t;
        if (j > i && j < n) {
          row += coupling_from_word(words[t]) * term(i, j);
        }
      }
    }
    total += row;
  }
  return total;
}

}  // namespace

// --- Graph -----------------------------------------------------------------------------------

std::int64_t num_spins(const Graph& g) {
  check_spin_count(g.n, "num_spins");
  return g.n;
}

std::int64_t total_weight(const Graph& g) {
  check_graph(g, "total_weight");
  std::int64_t sum = 0;
  for (const Edge& e : g.edges) {
    sum += e.w;
  }
  return sum;
}

std::int64_t ising_energy(const Graph& g, std::span<const Spin> s) {
  check_graph(g, "ising_energy");
  check_spins(s, g.n, "ising_energy");
  std::int64_t sum = 0;
  for (const Edge& e : g.edges) {
    const std::int64_t w = e.w;
    sum += s[static_cast<std::size_t>(e.u)] == s[static_cast<std::size_t>(e.v)] ? w : -w;
  }
  return sum;
}

std::int64_t cut_direct(const Graph& g, std::span<const Spin> s) {
  check_graph(g, "cut_direct");
  check_spins(s, g.n, "cut_direct");
  std::int64_t sum = 0;
  for (const Edge& e : g.edges) {
    if (s[static_cast<std::size_t>(e.u)] != s[static_cast<std::size_t>(e.v)]) {
      sum += e.w;
    }
  }
  return sum;
}

void dense_rows_int8(const Graph& g, std::int64_t row_begin, std::int64_t row_end, std::int64_t ld,
                     std::span<std::int8_t> out) {
  constexpr const char* kFunction = "dense_rows_int8";
  check_graph(g, kFunction);
  for (std::size_t k = 0; k < g.edges.size(); ++k) {
    const Edge& e = g.edges[k];
    if (e.w < -kMaxAbsInt8Weight || e.w > kMaxAbsInt8Weight) {
      invalid(kFunction, edge_text(k, e) + ": |w| must be at most 127, so that J = -w fits INT8");
    }
  }
  check_rows(g.n, row_begin, row_end, ld, out.size(), kFunction);

  std::int8_t* const base = out.data();
  std::fill(base, base + (row_end - row_begin) * ld, std::int8_t{0});
  for (const Edge& e : g.edges) {
    const auto j_uv = static_cast<std::int8_t>(-e.w);  // J_uv = J_vu = -w_uv
    if (e.u >= row_begin && e.u < row_end) {
      base[(e.u - row_begin) * ld + e.v] = j_uv;
    }
    if (e.v >= row_begin && e.v < row_end) {
      base[(e.v - row_begin) * ld + e.u] = j_uv;
    }
  }
}

// --- DenseSpinGlass --------------------------------------------------------------------------

std::int64_t num_spins(const DenseSpinGlass& g) {
  check_dense(g, "num_spins");
  return g.n;
}

std::int64_t total_weight(const DenseSpinGlass& g) {
  check_dense(g, "total_weight");
  // W = sum_{i<j} w_ij = -sum_{i<j} J_ij.
  return -sum_upper_pairs(g, [](std::int64_t, std::int64_t) { return std::int64_t{1}; });
}

std::int64_t ising_energy(const DenseSpinGlass& g, std::span<const Spin> s) {
  check_dense(g, "ising_energy");
  check_spins(s, g.n, "ising_energy");
  // E = sum_{i<j} w_ij s_i s_j = -sum_{i<j} J_ij s_i s_j.
  return -sum_upper_pairs(g, [s](std::int64_t i, std::int64_t j) {
    return std::int64_t{s[static_cast<std::size_t>(i)]} * s[static_cast<std::size_t>(j)];
  });
}

std::int64_t cut_direct(const DenseSpinGlass& g, std::span<const Spin> s) {
  check_dense(g, "cut_direct");
  check_spins(s, g.n, "cut_direct");
  // C = sum_{i<j} w_ij [s_i != s_j] = -sum_{i<j} J_ij [s_i != s_j].
  return -sum_upper_pairs(g, [s](std::int64_t i, std::int64_t j) {
    return std::int64_t{s[static_cast<std::size_t>(i)] != s[static_cast<std::size_t>(j)]};
  });
}

// Rows are generated in groups of four aligned rows i = 4q, ..., 4q + 3 (OpenMP over the groups,
// every element computed independently of the others):
// - columns j < 4q: J_ij is word i of stream j, and block q of stream j holds the words
//   4q, ..., 4q + 3, that is column j of all four rows: one block for four elements;
// - columns j >= 4q + 4: J_ij is word j of stream i: one block for four consecutive columns;
// - the columns of the group itself, diagonal included, element by element.
void dense_rows_int8(const DenseSpinGlass& g, std::int64_t row_begin, std::int64_t row_end,
                     std::int64_t ld, std::span<std::int8_t> out) {
  constexpr const char* kFunction = "dense_rows_int8";
  check_dense(g, kFunction);
  check_rows(g.n, row_begin, row_end, ld, out.size(), kFunction);

  const std::int64_t n = g.n;
  const std::uint64_t seed = g.seed;
  std::int8_t* const base = out.data();

  // Rows [row_begin, real_end) are rows of J; rows [real_end, row_end) are padding rows (i >= n).
  const std::int64_t real_end = std::max(row_begin, std::min(row_end, n));
  std::fill(base + (real_end - row_begin) * ld, base + (row_end - row_begin) * ld, std::int8_t{0});
  if (real_end == row_begin) {
    return;
  }

  const std::int64_t q_first = row_begin / 4;
  const std::int64_t q_last = (real_end - 1) / 4;
#pragma omp parallel for schedule(static)
  for (std::int64_t q = q_first; q <= q_last; ++q) {
    const std::int64_t i0 = 4 * q;
    const std::int64_t i_begin = std::max(i0, row_begin);  // rows of the group to write
    const std::int64_t i_end = std::min(i0 + 4, real_end);

    for (std::int64_t j = 0; j < i0; ++j) {
      const U32x4 block = stream_block(seed, RngDomain::couplings, static_cast<std::uint32_t>(j),
                                       static_cast<std::uint64_t>(q));
      const std::uint32_t words[4] = {block.x, block.y, block.z, block.w};
      for (std::int64_t i = i_begin; i < i_end; ++i) {
        base[(i - row_begin) * ld + j] = coupling_from_word(words[i - i0]);
      }
    }

    const std::int64_t group_end = std::min(i0 + 4, n);
    for (std::int64_t i = i_begin; i < i_end; ++i) {
      std::int8_t* const row = base + (i - row_begin) * ld;
      for (std::int64_t j = i0; j < group_end; ++j) {
        row[j] = dense_coupling(seed, i, j);
      }
      const auto stream = static_cast<std::uint32_t>(i);
      for (std::int64_t b = q + 1; 4 * b < n; ++b) {
        const U32x4 block =
            stream_block(seed, RngDomain::couplings, stream, static_cast<std::uint64_t>(b));
        const std::uint32_t words[4] = {block.x, block.y, block.z, block.w};
        const std::int64_t j0 = 4 * b;
        const std::int64_t count = std::min<std::int64_t>(4, n - j0);
        for (std::int64_t t = 0; t < count; ++t) {
          row[j0 + t] = coupling_from_word(words[t]);
        }
      }
      std::fill(row + n, row + ld, std::int8_t{0});
    }
  }
}

std::int8_t coupling(const DenseSpinGlass& g, std::int64_t i, std::int64_t j) {
  check_dense(g, "coupling");
  if (i < 0 || i >= g.n || j < 0 || j >= g.n) {
    invalid("coupling", "the indices (i, j) = (" + std::to_string(i) + ", " + std::to_string(j) +
                            ") are outside [0, n = " + std::to_string(g.n) + ")");
  }
  return dense_coupling(g.seed, i, j);
}

// --- Instance --------------------------------------------------------------------------------

std::int64_t num_spins(const Instance& inst) {
  return std::visit([](const auto& g) { return num_spins(g); }, inst);
}

std::int64_t total_weight(const Instance& inst) {
  return std::visit([](const auto& g) { return total_weight(g); }, inst);
}

std::int64_t ising_energy(const Instance& inst, std::span<const Spin> s) {
  return std::visit([s](const auto& g) { return ising_energy(g, s); }, inst);
}

std::int64_t cut_direct(const Instance& inst, std::span<const Spin> s) {
  return std::visit([s](const auto& g) { return cut_direct(g, s); }, inst);
}

void dense_rows_int8(const Instance& inst, std::int64_t row_begin, std::int64_t row_end,
                     std::int64_t ld, std::span<std::int8_t> out) {
  std::visit([&](const auto& g) { dense_rows_int8(g, row_begin, row_end, ld, out); }, inst);
}

// --- Internal cut of the final check of the solvers ------------------------------------------

namespace detail {
namespace {

// Largest n for which the sum of one row, at most 127 (n - 1) in absolute value, fits in int32:
// 127 * 2^24 = 2130706432 < 2^31 - 1.
constexpr std::int64_t kMaxSpinsInt32Row = std::int64_t{1} << 24;

// -sum_{i<j} J_ij [s_i != s_j], with the sum of each row j > i in Acc (exact by the caller's
// choice of Acc) and the total in int64. The select keeps the inner loop free of branches.
template <typename Acc>
std::int64_t dense_rows_cut(const std::int8_t* rows, std::int64_t n, std::int64_t ld,
                            const Spin* s) noexcept {
  std::int64_t total = 0;
  for (std::int64_t i = 0; i + 1 < n; ++i) {
    const std::int8_t* row = rows + i * ld;
    const Spin si = s[i];
    Acc acc = 0;
    for (std::int64_t j = i + 1; j < n; ++j) {
      acc += s[j] != si ? static_cast<Acc>(row[j]) : Acc{0};
    }
    total += static_cast<std::int64_t>(acc);
  }
  return -total;
}

}  // namespace

std::int64_t cut_edges_unchecked(const Graph& g, std::span<const Spin> s) noexcept {
  const Spin* spins = s.data();
  std::int64_t total = 0;
  for (const Edge& e : g.edges) {
    // w AND a mask of all ones when s_u != s_v, all zeros otherwise: no branch, whose outcome
    // would be unpredictable (with a conditional expression GCC -O3 emits a branch, about 9 times
    // slower on G1).
    const auto cut = static_cast<std::int64_t>(spins[e.u] != spins[e.v]);
    total += static_cast<std::int64_t>(e.w) & -cut;
  }
  return total;
}

std::int64_t cut_dense_rows_unchecked(std::span<const std::int8_t> rows, std::int64_t n,
                                      std::int64_t ld, std::span<const Spin> s) noexcept {
  if (n <= kMaxSpinsInt32Row) {
    return dense_rows_cut<std::int32_t>(rows.data(), n, ld, s.data());
  }
  return dense_rows_cut<std::int64_t>(rows.data(), n, ld, s.data());
}

}  // namespace detail

}  // namespace sbmc
