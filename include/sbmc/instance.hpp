// Problem instances and the Ising model of Max-Cut.
//
// Two kinds of instance enter the solvers, as sbmc::Instance:
// - Graph: an explicit weighted graph as a canonical edge list (0-based, u < v, w != 0 and
//   w != INT32_MIN, edges in strictly increasing (u, v) order, hence without duplicates);
// - DenseSpinGlass: an implicit complete graph with J_ij = +-1, generated on demand from the
//   couplings stream of its seed (sbmc/philox.hpp), never stored.
//
// Conventions (reference document, section 5): J_ij = -w_ij and J_ii = 0; for a DenseSpinGlass
// the J_ij come from the generator and w_ij = -J_ij. With sums in int64:
//   W = sum_{i<j} w_ij,  E(s) = sum_{i<j} w_ij s_i s_j,  C(s) = sum_{i<j} w_ij [s_i != s_j],
// so that 2C = W - E; C is computed directly on the edges or pairs, never from W and E.
//
// Every function checks its preconditions in every build and throws sbmc::InvalidArgument when
// one is violated, without writing anything to its outputs. A valid Graph has 1 <= n <= 2^31 - 1
// and canonical edges; it is checked in full (O(M)) by every function except num_spins, which
// checks n only. A valid DenseSpinGlass has 1 <= n <= 2^31 - 1 and any seed. A valid spin vector
// has n elements, each -1 or +1. The functions on a DenseSpinGlass run in parallel with OpenMP
// over the rows; their results do not depend on the number of threads.
#pragma once

#include <cstdint>
#include <span>
#include <variant>
#include <vector>

namespace sbmc {

using Spin = std::int8_t;  // -1 or +1

// Edge of a Graph.
struct Edge {
  std::int32_t u, v, w;  // 0-based, u < v, w != 0
};

// Explicit graph.
struct Graph {
  std::int64_t n = 0;
  std::vector<Edge> edges;  // canonical: strictly increasing (u, v)
};

// Implicit complete graph, J_ij = +-1.
struct DenseSpinGlass {
  std::int64_t n = 0;
  std::uint64_t seed = 0;
};

// Input of the solvers.
using Instance = std::variant<Graph, DenseSpinGlass>;

// Number of spins n.
std::int64_t num_spins(const Graph& g);
std::int64_t num_spins(const DenseSpinGlass& g);
std::int64_t num_spins(const Instance& inst);

// Total weight W.
std::int64_t total_weight(const Graph& g);
std::int64_t total_weight(const DenseSpinGlass& g);
std::int64_t total_weight(const Instance& inst);

// Ising energy E(s).
std::int64_t ising_energy(const Graph& g, std::span<const Spin> s);
std::int64_t ising_energy(const DenseSpinGlass& g, std::span<const Spin> s);
std::int64_t ising_energy(const Instance& inst, std::span<const Spin> s);

// Cut C(s), computed directly on the edges (Graph) or on the pairs i < j (DenseSpinGlass).
std::int64_t cut_direct(const Graph& g, std::span<const Spin> s);
std::int64_t cut_direct(const DenseSpinGlass& g, std::span<const Spin> s);
std::int64_t cut_direct(const Instance& inst, std::span<const Spin> s);

// Rows [row_begin, row_end) of J in INT8, row-major with stride ld:
// out[(i - row_begin) * ld + j] = J_ij for i < n, j < n and i != j, and 0 otherwise (diagonal,
// padding columns j >= n, padding rows i >= n), for every j in [0, ld). Preconditions, all
// checked before writing: valid instance; n <= ld <= 2^31 - 1; 0 <= row_begin <= row_end <= ld;
// out.size() >= (row_end - row_begin) * ld; for a Graph, 1 <= |w| <= 127 on every edge, so that
// -w fits in INT8. Elements of out beyond (row_end - row_begin) * ld are not touched.
void dense_rows_int8(const Graph& g, std::int64_t row_begin, std::int64_t row_end, std::int64_t ld,
                     std::span<std::int8_t> out);
void dense_rows_int8(const DenseSpinGlass& g, std::int64_t row_begin, std::int64_t row_end,
                     std::int64_t ld, std::span<std::int8_t> out);
void dense_rows_int8(const Instance& inst, std::int64_t row_begin, std::int64_t row_end,
                     std::int64_t ld, std::span<std::int8_t> out);

// J_ij of a DenseSpinGlass, with 0 <= i, j < n: 0 when i == j, otherwise
// coupling_from_word(stream_word(g.seed, RngDomain::couplings, min(i, j), max(i, j))).
// Symmetric by construction.
std::int8_t coupling(const DenseSpinGlass& g, std::int64_t i, std::int64_t j);

// Internal to the solvers (run_sb_sequential, run_sb_openmp, run_sa_sequential, run_sa_openmp):
// the cut of their final check, after the instance has been checked once at their entry
// (decision B5). Unlike every function above, these check nothing, never throw, never allocate
// and never use OpenMP, so that they can run inside the parallel regions of the solvers. Their
// results are equal to cut_direct for the same instance and the same spins.
namespace detail {

// Cut C = sum over the edges of g of w_uv [s_u != s_v], in int64, without any check: the caller
// has checked g (for example with total_weight) and guarantees s.size() == g.n and s_i in
// {-1, +1}. For the final check of the solvers (decision L3 of phase 07); elsewhere use
// cut_direct.
std::int64_t cut_edges_unchecked(const Graph& g, std::span<const Spin> s) noexcept;

// Cut C = -sum_{i<j} J_ij [s_i != s_j] from dense INT8 rows of J: element (i, j) at
// rows[i * ld + j] for 0 <= i, j < n, symmetric with a zero diagonal, as written by
// dense_rows_int8; the columns j >= n are never read. No check: the caller guarantees n >= 1,
// ld >= n, rows.size() >= n * ld, s.size() == n and s_i in {-1, +1}.
std::int64_t cut_dense_rows_unchecked(std::span<const std::int8_t> rows, std::int64_t n,
                                      std::int64_t ld, std::span<const Spin> s) noexcept;

}  // namespace detail

}  // namespace sbmc
