// C1: simulated bifurcation with OpenMP over blocks of replicas and SIMD over the replicas of a
// block (sbmc/sb_openmp.hpp).
//
// Preparation, after the preconditions and outside the parallel region: W, c0, the pressure
// schedule; the internal copy of J (dense INT8 with rows on 64-byte boundaries, or compressed
// rows with increasing columns, int32 values); the integer accumulator from L = max_i sum_j
// |J_ij|; the outputs; one work area per thread of the region, each on a 64-byte boundary and made
// of whole cache lines. Parallel region: num_threads(min(T_req, B)); a single construct reads the
// team size; a for loop with schedule(dynamic, 1) gives the blocks to the threads. A block runs
// all M steps in the work area of its thread, then writes only its rows of the outputs and
// checks the best cuts of its replicas with cut_direct; a mismatch or an exception is recorded per
// replica and reported after the region for the smallest replica. Nothing allocates and no
// exception leaves the region; in the step loop no public function of the instance is called
// (B5); the global OpenMP state is never changed.
//
// Lanes. Every loop that computes over the lanes of a block runs over all
// kSbOpenmpReplicasPerBlock lanes, a constant, and every floating-point one is a vector loop (omp
// simd): every lane, padding lanes included, runs the same instructions, so a replica gives the
// same bits in every lane, block and thread, also in the production builds where FMA contraction
// may change the bits with respect to C0. The double sum of bSB stays sequential in j for every
// lane; the integer sums are exact in any order.
#include <sbmc/sb_openmp.hpp>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/initial_state.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/sb.hpp>

namespace sbmc {
namespace {

constexpr int kLanes = kSbOpenmpReplicasPerBlock;
static_assert(kLanes == 8 || kLanes == 16 || kLanes == 32, "RB must be 8, 16 or 32");
constexpr auto kLaneCount = static_cast<std::size_t>(kLanes);

constexpr std::int64_t kMaxDenseSpins = 32768;                      // dense INT8 J: at most 1 GiB
constexpr std::int64_t kReplicaIndexLimit = std::int64_t{1} << 32;  // global replica indices
constexpr std::int32_t kMaxDenseWeight = 127;                       // |w| of a dense INT8 J
constexpr std::size_t kCacheLine = 64;

[[noreturn]] void invalid(const std::string& reason) {
  throw InvalidArgument("run_sb_openmp: " + reason);
}

bool finite_positive(float v) {
  return std::isfinite(v) && v > 0.0f;
}

// The checks of run_sb_sequential on the parameters, same conditions and same order.
void check_params(const SbParams& p, int trace_steps) {
  if (p.algorithm != SbAlgorithm::ballistic && p.algorithm != SbAlgorithm::discrete) {
    invalid("algorithm " + std::to_string(static_cast<int>(p.algorithm)) +
            " is neither ballistic (0) nor discrete (1)");
  }
  if (p.replicas < 1) {
    invalid("replicas = " + std::to_string(p.replicas) + " must be at least 1");
  }
  if (p.steps < 1) {
    invalid("steps = " + std::to_string(p.steps) + " must be at least 1");
  }
  if (!finite_positive(p.dt)) {
    invalid("dt = " + std::to_string(p.dt) + " must be finite and > 0");
  }
  if (!finite_positive(p.a0)) {
    invalid("a0 = " + std::to_string(p.a0) + " must be finite and > 0");
  }
  if (!std::isfinite(p.c0) || p.c0 < 0.0f) {
    invalid("c0 = " + std::to_string(p.c0) + " must be finite and >= 0 (0: automatic)");
  }
  if (p.eval_every < 0) {
    invalid("eval_every = " + std::to_string(p.eval_every) + " must be >= 0");
  }
  if (p.first_replica < 0) {
    invalid("first_replica = " + std::to_string(p.first_replica) + " is negative");
  }
  if (p.first_replica > kReplicaIndexLimit - p.replicas) {
    invalid("first_replica + replicas = " + std::to_string(p.first_replica) + " + " +
            std::to_string(p.replicas) + " exceeds 2^32");
  }
  if (trace_steps < 0 || trace_steps > p.steps) {
    invalid("trace_steps = " + std::to_string(trace_steps) +
            " is outside [0, steps = " + std::to_string(p.steps) + "]");
  }
}

std::size_t round_up(std::size_t value, std::size_t multiple) {
  return (value + multiple - 1) / multiple * multiple;
}

// Memory on a 64-byte boundary, released by its deleter (RAII).
struct AlignedDelete {
  void operator()(std::byte* p) const noexcept {
    ::operator delete(p, std::align_val_t{kCacheLine});
  }
};
using AlignedBytes = std::unique_ptr<std::byte, AlignedDelete>;

AlignedBytes allocate_aligned(std::size_t bytes) {
  return AlignedBytes(static_cast<std::byte*>(
      ::operator new(std::max(bytes, kCacheLine), std::align_val_t{kCacheLine})));
}

// J dense in INT8, J_ij at data[i * ld + j], zero diagonal; every row starts on a 64-byte boundary.
struct DenseJ {
  std::size_t n = 0;
  std::size_t ld = 0;
  AlignedBytes bytes;
  [[nodiscard]] const std::int8_t* row(std::size_t i) const noexcept {
    return reinterpret_cast<const std::int8_t*>(bytes.get()) + i * ld;
  }
};

// J in compressed rows: J_ij = -w_ij in both directions, the columns of every row increasing.
struct SparseJ {
  std::size_t n = 0;
  std::vector<std::int64_t> row_begin;  // n + 1 offsets
  std::vector<std::int32_t> column;
  std::vector<std::int32_t> value;
};

// Dense INT8 J of an instance through dense_rows_int8 (J_ij = -w_ij, or coupling(g, i, j)).
template <typename G>
DenseJ make_dense(const G& g) {
  DenseJ j;
  j.n = static_cast<std::size_t>(g.n);
  j.ld = round_up(j.n, kCacheLine);
  j.bytes = allocate_aligned(j.n * j.ld);
  dense_rows_int8(
      g, 0, g.n, static_cast<std::int64_t>(j.ld),
      std::span<std::int8_t>(reinterpret_cast<std::int8_t*>(j.bytes.get()), j.n * j.ld));
  return j;
}

// Compressed rows of a valid graph, as in C0: the edges are canonical, so filling the rows in edge
// order gives every row in increasing column order.
SparseJ make_sparse(const Graph& g) {
  SparseJ c;
  c.n = static_cast<std::size_t>(g.n);
  c.row_begin.assign(c.n + 1, 0);
  for (const Edge& e : g.edges) {
    ++c.row_begin[static_cast<std::size_t>(e.u) + 1];
    ++c.row_begin[static_cast<std::size_t>(e.v) + 1];
  }
  for (std::size_t i = 0; i < c.n; ++i) {
    c.row_begin[i + 1] += c.row_begin[i];
  }
  c.column.resize(2 * g.edges.size());
  c.value.resize(2 * g.edges.size());
  std::vector<std::int64_t> next(c.row_begin.begin(), c.row_begin.end() - 1);
  for (const Edge& e : g.edges) {
    const auto u = static_cast<std::size_t>(e.u);
    const auto v = static_cast<std::size_t>(e.v);
    const std::int32_t j_uv = -e.w;  // w != INT32_MIN: -w fits
    c.column[static_cast<std::size_t>(next[u])] = e.v;
    c.value[static_cast<std::size_t>(next[u]++)] = j_uv;
    c.column[static_cast<std::size_t>(next[v])] = e.u;
    c.value[static_cast<std::size_t>(next[v]++)] = j_uv;
  }
  return c;
}

// L = max_i sum_j |J_ij| of a valid graph, in int64.
std::int64_t max_row_norm(const Graph& g) {
  std::vector<std::int64_t> norm(static_cast<std::size_t>(g.n), 0);
  for (const Edge& e : g.edges) {
    norm[static_cast<std::size_t>(e.u)] += std::abs(std::int64_t{e.w});
    norm[static_cast<std::size_t>(e.v)] += std::abs(std::int64_t{e.w});
  }
  return *std::max_element(norm.begin(), norm.end());
}

// The rule of the format of J: a Graph is dense when n <= 32768, |w| <= 127 on every edge and
// 8m >= n(n - 1) (density at least 25%).
bool dense_graph(const Graph& g) {
  if (g.n > kMaxDenseSpins) {
    return false;
  }
  const bool small_weights = std::all_of(g.edges.begin(), g.edges.end(), [](const Edge& e) {
    return e.w >= -kMaxDenseWeight && e.w <= kMaxDenseWeight;
  });
  return small_weights && 8 * static_cast<std::int64_t>(g.edges.size()) >= g.n * (g.n - 1);
}

// --- Kernels on one block: arrays of n * kLanes elements in spin-major order ---------------------
//
// A sum over the columns of a row adds four columns per lane loop, the four terms from left to
// right, then the remaining columns one at a time: every lane adds its terms in the order of the
// columns, as C0 does. The main loop takes eight columns in two lane loops: when its body is a
// single lane loop, GCC -O3 fuses consecutive iterations (unroll-and-jam) once the accumulator is
// a local array, and vectorizes the fused loop with shorter vectors or not at all (several times
// slower, measured in phase 05); a body with two loops is never fused.

// acc[l] = acc[l] + v0 * u0[l] + v1 * u1[l] + v2 * u2[l] + v3 * u3[l], from left to right.
template <typename T>
void add_four(T* acc, T v0, const T* u0, T v1, const T* u1, T v2, const T* u2, T v3,
              const T* u3) noexcept {
#pragma omp simd
  for (int l = 0; l < kLanes; ++l) {
    acc[l] = static_cast<T>(acc[l] + v0 * u0[l] + v1 * u1[l] + v2 * u2[l] + v3 * u3[l]);
  }
}

// acc[l] = acc[l] + v * u[l].
template <typename T>
void add_one(T* acc, T v, const T* u) noexcept {
#pragma omp simd
  for (int l = 0; l < kLanes; ++l) {
    acc[l] = static_cast<T>(acc[l] + v * u[l]);
  }
}

// acc[l] += sum of value(k) * u_{column(k)}[l] over k in [begin, end), in the order of k, where
// u_c = u + c * kLanes.
template <typename T, typename Value, typename Column>
void row_sum(std::size_t begin, std::size_t end, Value value, Column column, const T* u,
             T* acc) noexcept {
  const auto add_four_at = [&](std::size_t k) {
    add_four(acc, value(k), u + column(k) * kLaneCount, value(k + 1),
             u + column(k + 1) * kLaneCount, value(k + 2), u + column(k + 2) * kLaneCount,
             value(k + 3), u + column(k + 3) * kLaneCount);
  };
  std::size_t k = begin;
  for (; end - k >= 8; k += 8) {
    add_four_at(k);
    add_four_at(k + 4);
  }
  if (end - k >= 4) {
    add_four_at(k);
    k += 4;
  }
  for (; k < end; ++k) {
    add_one(acc, value(k), u + column(k) * kLaneCount);
  }
}

// acc[l] = sum_c row[c] * u_c[l] over the n columns of a dense row, zero terms included.
template <typename T>
void dense_row_sum(const std::int8_t* row, std::size_t n, const T* u, T* acc) noexcept {
  row_sum(
      0, n, [row](std::size_t c) { return static_cast<T>(row[c]); },
      [](std::size_t c) { return c; }, u, acc);
}

// acc[l] = sum_k value[k] * u_column[k][l] over the entries of the row i of compressed rows.
template <typename T>
void sparse_row_sum(const SparseJ& j, std::size_t i, const T* u, T* acc) noexcept {
  const std::int32_t* const value = j.value.data();
  const std::int32_t* const column = j.column.data();
  row_sum(
      static_cast<std::size_t>(j.row_begin[i]), static_cast<std::size_t>(j.row_begin[i + 1]),
      [value](std::size_t k) { return static_cast<T>(value[k]); },
      [column](std::size_t k) { return static_cast<std::size_t>(column[k]); }, u, acc);
}

// e[l] += s_i[l] * F_i[l] for the row i of the integer fields, and field_i = (float)F_i when field
// is not null.
template <typename Acc>
void finish_spin_row(const Acc* acc, const Acc* s_i, float* field_i, std::int64_t* e) noexcept {
  if (field_i != nullptr) {
#pragma omp simd
    for (int l = 0; l < kLanes; ++l) {
      field_i[l] = static_cast<float>(acc[l]);
    }
  }
#pragma omp simd
  for (int l = 0; l < kLanes; ++l) {
    const auto f = static_cast<std::int64_t>(acc[l]);
    e[l] += s_i[l] > 0 ? f : -f;
  }
}

// Integer fields F_i = sum_j J_ij s_j of every lane (dense J, zero terms included). Every partial
// sum is bounded by L, so the accumulator never overflows.
template <typename Acc>
void spin_fields(const DenseJ& j, const Acc* s, float* field, std::int64_t* e) noexcept {
  for (std::size_t i = 0; i < j.n; ++i) {
    Acc acc[kLanes] = {};
    dense_row_sum(j.row(i), j.n, s, acc);
    finish_spin_row(acc, s + i * kLaneCount, field == nullptr ? nullptr : field + i * kLaneCount,
                    e);
  }
}

// Integer fields F_i = sum_j J_ij s_j of every lane (compressed rows).
template <typename Acc>
void spin_fields(const SparseJ& j, const Acc* s, float* field, std::int64_t* e) noexcept {
  for (std::size_t i = 0; i < j.n; ++i) {
    Acc acc[kLanes] = {};
    sparse_row_sum(j, i, s, acc);
    finish_spin_row(acc, s + i * kLaneCount, field == nullptr ? nullptr : field + i * kLaneCount,
                    e);
  }
}

// field_i = (float)acc for every lane.
void store_position_row(const double* acc, float* field_i) noexcept {
#pragma omp simd
  for (int l = 0; l < kLanes; ++l) {
    field_i[l] = static_cast<float>(acc[l]);
  }
}

// bSB fields: acc = +0.0, acc += (double)J_ij * (double)x_j for increasing j, field = (float)acc.
// Dense J: the zero terms are included, which gives the bits of C0 (each term is +-0.0 and acc is
// never -0.0).
void position_fields(const DenseJ& j, const double* xd, float* field) noexcept {
  for (std::size_t i = 0; i < j.n; ++i) {
    double acc[kLanes] = {};
    dense_row_sum(j.row(i), j.n, xd, acc);
    store_position_row(acc, field + i * kLaneCount);
  }
}

void position_fields(const SparseJ& j, const double* xd, float* field) noexcept {
  for (std::size_t i = 0; i < j.n; ++i) {
    double acc[kLanes] = {};
    sparse_row_sum(j, i, xd, acc);
    store_position_row(acc, field + i * kLaneCount);
  }
}

// --- One block ------------------------------------------------------------------------------

// What every block needs: sizes, parameters and the outputs.
struct Plan {
  std::size_t n = 0;
  int replicas = 0;
  int steps = 0;
  bool discrete = true;
  int eval_every = 0;
  const float* p = nullptr;
  float dt = 0.0f;
  float c0 = 0.0f;
  float dt_a0 = 0.0f;
  std::uint64_t seed = 0;
  std::int64_t first_replica = 0;
  int trace_steps = 0;
  std::int64_t total = 0;
  SbResult* out = nullptr;
};

// The work area of one thread, n * kLanes elements per array, each array on a 64-byte boundary.
template <typename Acc>
struct WorkArea {
  float* x = nullptr;
  float* y = nullptr;
  float* field = nullptr;
  double* xd = nullptr;  // bSB only
  Acc* s = nullptr;
  Spin* best = nullptr;
};

template <typename Acc>
std::size_t work_area_bytes(std::size_t n, bool discrete) {
  const std::size_t elements = n * kLaneCount;
  std::size_t bytes = 3 * round_up(elements * sizeof(float), kCacheLine);
  bytes += discrete ? 0 : round_up(elements * sizeof(double), kCacheLine);
  bytes += round_up(elements * sizeof(Acc), kCacheLine);
  bytes += round_up(elements * sizeof(Spin), kCacheLine);
  return bytes;
}

template <typename Acc>
WorkArea<Acc> work_area_at(std::byte* base, std::size_t n, bool discrete) {
  const std::size_t elements = n * kLaneCount;
  WorkArea<Acc> a;
  std::byte* next = base;
  const auto take = [&next](std::size_t bytes) {
    std::byte* const at = next;
    next += round_up(bytes, kCacheLine);
    return at;
  };
  a.x = reinterpret_cast<float*>(take(elements * sizeof(float)));
  a.y = reinterpret_cast<float*>(take(elements * sizeof(float)));
  a.field = reinterpret_cast<float*>(take(elements * sizeof(float)));
  if (!discrete) {
    a.xd = reinterpret_cast<double*>(take(elements * sizeof(double)));
  }
  a.s = reinterpret_cast<Acc*>(take(elements * sizeof(Acc)));
  a.best = reinterpret_cast<Spin*>(take(elements * sizeof(Spin)));
  return a;
}

// The best configuration of every lane, as in C0: the first evaluated one, then only a strictly
// smaller energy (ties keep the smallest m).
struct LaneBest {
  std::int64_t energy[kLanes] = {};
  int step[kLanes] = {};
  bool evaluated[kLanes] = {};
};

template <typename Acc>
void consider(LaneBest& best, const std::int64_t* e, int m, std::size_t n,
              const WorkArea<Acc>& a) noexcept {
  bool improved[kLanes] = {};
  bool any = false;
  for (int l = 0; l < kLanes; ++l) {
    const std::int64_t energy = -e[l] / 2;  // e = sum_i s_i F_i is even
    if (!best.evaluated[l] || energy < best.energy[l]) {
      best.evaluated[l] = true;
      best.energy[l] = energy;
      best.step[l] = m;
      improved[l] = true;
      any = true;
    }
  }
  if (!any) {
    return;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const Acc* const s_i = a.s + i * kLaneCount;
    Spin* const best_i = a.best + i * kLaneCount;
#pragma omp simd
    for (int l = 0; l < kLanes; ++l) {
      const auto now = static_cast<Spin>(s_i[l]);  // both values loaded: a blend, no masked load
      const Spin kept = best_i[l];
      best_i[l] = improved[l] ? now : kept;
    }
  }
}

template <typename Acc>
void signs(std::size_t n, const WorkArea<Acc>& a) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const float* const x_i = a.x + i * kLaneCount;
    Acc* const s_i = a.s + i * kLaneCount;
#pragma omp simd
    for (int l = 0; l < kLanes; ++l) {
      s_i[l] = static_cast<Acc>(sb_sign(x_i[l]));
    }
  }
}

// All M steps of block b, then its rows of the outputs. No allocation, no exception.
template <typename Acc, typename Couplings>
void run_block(const Plan& plan, const Couplings& j, const WorkArea<Acc>& a, int b) noexcept {
  const std::size_t n = plan.n;
  const int r0 = b * kLanes;
  const int real = std::min(kLanes, plan.replicas - r0);  // lanes l < real are replicas
  SbResult& out = *plan.out;

  // Initial state: x0 of the real lanes from their global index, 0 in the padding lanes
  // (their global index could exceed 2^32 - 1); y = 0.
  for (std::size_t i = 0; i < n; ++i) {
    for (int l = 0; l < kLanes; ++l) {
      const std::size_t at = i * kLaneCount + static_cast<std::size_t>(l);
      a.x[at] = l < real ? initial_position(plan.seed,
                                            static_cast<std::uint32_t>(plan.first_replica + r0 + l),
                                            static_cast<std::uint32_t>(i))
                         : 0.0f;
      a.y[at] = 0.0f;
    }
  }

  LaneBest best;
  std::int64_t e[kLanes] = {};
  for (int m = 0; m < plan.steps; ++m) {
    const bool evaluate = plan.discrete || (plan.eval_every > 0 && m % plan.eval_every == 0);
    if (evaluate) {
      signs(n, a);  // s(m), before any update of the step
      std::fill(e, e + kLanes, std::int64_t{0});
    }
    if (plan.discrete) {
      spin_fields(j, a.s, a.field, e);
    } else {
      for (std::size_t i = 0; i < n; ++i) {
        const float* const x_i = a.x + i * kLaneCount;
        double* const xd_i = a.xd + i * kLaneCount;
#pragma omp simd
        for (int l = 0; l < kLanes; ++l) {
          xd_i[l] = static_cast<double>(x_i[l]);
        }
      }
      position_fields(j, a.xd, a.field);
      if (evaluate) {
        spin_fields(j, a.s, static_cast<float*>(nullptr), e);
      }
    }
    if (evaluate) {
      consider(best, e, m, n, a);
    }

    const float p_m = plan.p[m];
    for (std::size_t i = 0; i < n; ++i) {
      float* const x_i = a.x + i * kLaneCount;
      float* const y_i = a.y + i * kLaneCount;
      const float* const field_i = a.field + i * kLaneCount;
#pragma omp simd
      for (int l = 0; l < kLanes; ++l) {
        sb_update(x_i[l], y_i[l], field_i[l], p_m, plan.dt, plan.c0, plan.dt_a0);
      }
    }

    if (m < plan.trace_steps) {
      for (int l = 0; l < real; ++l) {
        const std::size_t at =
            (static_cast<std::size_t>(m) * static_cast<std::size_t>(plan.replicas) +
             static_cast<std::size_t>(r0 + l)) *
            n;
        for (std::size_t i = 0; i < n; ++i) {
          out.trace_x[at + i] = a.x[i * kLaneCount + static_cast<std::size_t>(l)];
          out.trace_y[at + i] = a.y[i * kLaneCount + static_cast<std::size_t>(l)];
        }
      }
    }
  }

  // s(M), evaluated with every algorithm.
  signs(n, a);
  std::fill(e, e + kLanes, std::int64_t{0});
  spin_fields(j, a.s, static_cast<float*>(nullptr), e);
  consider(best, e, plan.steps, n, a);

  for (int l = 0; l < real; ++l) {
    const auto r = static_cast<std::size_t>(r0 + l);
    SbReplicaResult& rep = out.per_replica[r];
    rep.best_energy = best.energy[l];
    rep.best_cut = (plan.total - best.energy[l]) / 2;
    rep.best_step = best.step[l];
    rep.final_cut = (plan.total + e[l] / 2) / 2;  // E(s(M)) = -e / 2
    for (std::size_t i = 0; i < n; ++i) {
      const std::size_t from = i * kLaneCount + static_cast<std::size_t>(l);
      out.best_spins[r * n + i] = a.best[from];
      out.final_x[r * n + i] = a.x[from];
      out.final_y[r * n + i] = a.y[from];
    }
  }
}

// Outcome of the final check of one replica.
struct CheckStatus {
  bool mismatch = false;
  std::int64_t cut = 0;
  std::exception_ptr error;
};

// cut_direct of the best configuration of every replica of block b against its best_cut.
void check_block(const Instance& inst, const Plan& plan, int b,
                 std::vector<CheckStatus>& status) noexcept {
  const SbResult& out = *plan.out;
  const int r0 = b * kLanes;
  const int real = std::min(kLanes, plan.replicas - r0);
  for (int l = 0; l < real; ++l) {
    const auto r = static_cast<std::size_t>(r0 + l);
    try {
      const std::int64_t cut =
          cut_direct(inst, std::span<const Spin>(out.best_spins.data() + r * plan.n, plan.n));
      if (cut != out.per_replica[r].best_cut) {
        status[r].mismatch = true;
        status[r].cut = cut;
      }
    } catch (...) {
      status[r].error = std::current_exception();
    }
  }
}

template <typename Acc, typename Couplings>
void run_blocks(const Instance& inst, const Couplings& j, const Plan& plan, int threads,
                int& team) {
  const int blocks = (plan.replicas + kLanes - 1) / kLanes;
  const int team_request = std::min(threads, blocks);
  const std::size_t area_bytes = work_area_bytes<Acc>(plan.n, plan.discrete);
  const AlignedBytes areas = allocate_aligned(area_bytes * static_cast<std::size_t>(team_request));
  std::vector<CheckStatus> status(static_cast<std::size_t>(plan.replicas));

  int team_size = 0;
#pragma omp parallel num_threads(team_request)
  {
#pragma omp single
    team_size = omp_get_num_threads();

    const WorkArea<Acc> area =
        work_area_at<Acc>(areas.get() + static_cast<std::size_t>(omp_get_thread_num()) * area_bytes,
                          plan.n, plan.discrete);
#pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < blocks; ++b) {
      run_block(plan, j, area, b);
      check_block(inst, plan, b, status);
    }
  }
  team = team_size;

  for (std::size_t r = 0; r < status.size(); ++r) {
    if (status[r].error) {
      std::rethrow_exception(status[r].error);
    }
    if (status[r].mismatch) {
      throw Error("run_sb_openmp: internal error: replica " + std::to_string(r) +
                  ": cut_direct gives " + std::to_string(status[r].cut) +
                  ", (W - best_energy) / 2 gives " +
                  std::to_string(plan.out->per_replica[r].best_cut));
    }
  }
}

template <typename Couplings>
void run_with_accumulator(const Instance& inst, const Couplings& j, std::int64_t norm,
                          const Plan& plan, int threads, int& team) {
  if (norm <= 32767) {
    run_blocks<std::int16_t>(inst, j, plan, threads, team);
  } else if (norm <= 2147483647) {
    run_blocks<std::int32_t>(inst, j, plan, threads, team);
  } else {
    run_blocks<std::int64_t>(inst, j, plan, threads, team);
  }
}

}  // namespace

SbResult run_sb_openmp(const Instance& inst, const SbParams& params, int threads, int trace_steps) {
  if (threads < 0) {
    invalid("threads = " + std::to_string(threads) +
            " must be >= 0 (0: the default number of OpenMP threads)");
  }
  const int requested = threads > 0 ? threads : omp_get_max_threads();
  check_params(params, trace_steps);
  if (const auto* dense = std::get_if<DenseSpinGlass>(&inst);
      dense != nullptr && dense->n > kMaxDenseSpins) {
    invalid("n = " + std::to_string(dense->n) +
            ": C1 stores the J of a DenseSpinGlass as dense INT8 and accepts n <= 32768");
  }
  const std::int64_t total = total_weight(inst);  // full check of the instance
  const float c0 = params.c0 > 0.0f ? params.c0 : auto_c0(inst);
  const std::vector<float> p = pressure_schedule(params.a0, params.steps);

  const std::int64_t n = num_spins(inst);
  const auto un = static_cast<std::size_t>(n);
  const auto state_size = un * static_cast<std::size_t>(params.replicas);
  SbResult result;
  result.n = n;
  result.replicas = params.replicas;
  result.steps = params.steps;
  result.first_replica = params.first_replica;
  result.c0 = c0;
  result.total_weight = total;
  result.trace_steps = trace_steps;
  result.per_replica.resize(static_cast<std::size_t>(params.replicas));
  result.best_spins.resize(state_size);
  result.final_x.resize(state_size);
  result.final_y.resize(state_size);
  result.trace_x.resize(static_cast<std::size_t>(trace_steps) * state_size);
  result.trace_y.resize(static_cast<std::size_t>(trace_steps) * state_size);

  Plan plan;
  plan.n = un;
  plan.replicas = params.replicas;
  plan.steps = params.steps;
  plan.discrete = params.algorithm == SbAlgorithm::discrete;
  plan.eval_every = params.eval_every;
  plan.p = p.data();
  plan.dt = params.dt;
  plan.c0 = c0;
  plan.dt_a0 = params.dt * params.a0;
  plan.seed = params.seed;
  plan.first_replica = params.first_replica;
  plan.trace_steps = trace_steps;
  plan.total = total;
  plan.out = &result;

  int team = 0;
  if (const auto* g = std::get_if<Graph>(&inst)) {
    const std::int64_t norm = g->edges.empty() ? 0 : max_row_norm(*g);
    if (dense_graph(*g)) {
      run_with_accumulator(inst, make_dense(*g), norm, plan, requested, team);
    } else {
      run_with_accumulator(inst, make_sparse(*g), norm, plan, requested, team);
    }
  } else {
    const auto& dense = std::get<DenseSpinGlass>(inst);
    run_with_accumulator(inst, make_dense(dense), dense.n - 1, plan, requested, team);
  }
  result.threads = team;
  return result;
}

}  // namespace sbmc
