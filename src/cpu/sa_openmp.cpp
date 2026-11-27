// C2: simulated annealing with OpenMP over blocks of replicas and SIMD over the replicas of a
// block (sbmc/sa_openmp.hpp).
//
// Preparation [t0, t1), outside the parallel regions of the run and of the check: the
// preconditions; W (total_weight, the only full check of the instance, B5); beta_k for every
// sweep (sa_beta); the internal copy of J (dense INT8 with rows on 64-byte boundaries, or
// compressed rows with increasing columns, int32 values); the accumulator of the local fields from
// L = max_i sum_j |J_ij|; the path of the thresholds (a table when L <= L_tab, direct calls
// otherwise); the outputs; one work area per thread of the run, each on a 64-byte boundary and
// made of whole cache lines; the initial spins of every replica in final_spins, computed in a
// parallel loop with num_threads(min(T_req, B)) from the stream of sa_initial_spin.
// Run [t1, t2): a parallel region with num_threads(min(T_req, B)); a single construct reads the
// team size; a for loop with schedule(dynamic, 1) gives the blocks to the threads. A block reads
// the initial spins of its replicas from final_spins (the padding lanes take those of lane 0),
// runs all S sweeps in the work area of its thread, then writes only its rows of the outputs; an
// exception of the sweeps is recorded per block.
// Check [t2, t3): a second parallel region with num_threads(min(T_req, B)), schedule(static)
// over the replicas: the best and the final cut of every replica recomputed directly, without any
// check of the instance (decision L3 of phase 07), on the edges for a Graph (whatever the format
// of the internal J) and on the internal INT8 rows for a DenseSpinGlass; the replicas of a block
// whose run threw are not checked; a mismatch is recorded per replica. After t3 the replicas are
// scanned in order and the first error (of its block, then of its check) is reported. Nothing
// allocates and no exception leaves the regions; after the entry no public function of the
// instance is called (B5); the global OpenMP state is never changed.
//
// One proposal of spin i for all the lanes of a block: the acceptance words (one Philox block
// gives the words of four consecutive proposals of a lane), then the RB decisions in one lane
// loop (m = s_i h_i; accepted when m <= 0 or the word is below the threshold of delta_e = 2m),
// then, when at least one lane accepted, the update of the local fields. Every lane consumes the
// word of every proposal, so a lane follows the trajectory of its replica in the reference
// whatever the other lanes do. There is no floating-point operation outside sa_beta and
// sa_threshold, so the results are the same bit for bit in every preset.
//
// Two layouts of the state of a block (spins s, fields h, and the copies start and best_start),
// chosen by the format of J (F07-R2):
// - compressed rows (a Graph that is not dense): spin-major, element (i, l) at i * RB + l, the RB
//   lanes of a spin contiguous; the update is masked: for every neighbour c of i one SIMD loop
//   over the RB lanes adds J_ic d_l, with d_l = 0 for the lanes that rejected;
// - dense INT8 rows (a dense Graph or a DenseSpinGlass): lane-major, element (l, c) at l * ld + c
//   with ld = n rounded up to a multiple of 64 bytes / sizeof(Acc), the n spins of a lane
//   contiguous and every lane row on a 64-byte boundary; s_i and h_i of the RB lanes are gathered
//   before the decisions, and only the lanes that accepted update their row, h[l * ld + c] +=
//   J_ic d_l over the n columns in one SIMD loop; the initial fields of a lane, and those before
//   its replay, are sums of rows of J (one such loop per spin), not dot products.
// Rule: the lane-major layout always with dense rows, no adaptive rule. Until F07-R1 the dense
// rows used the spin-major layout with the masked update, n * RB additions per proposal with an
// acceptance whatever the number of accepting lanes; the lane-major layout does n per accepting
// lane, and makes the replay (one lane) contiguous. Measured in the sandbox (2 cores of an Intel
// Xeon @ 2.30 GHz with AVX-512, GCC 13.3, preset cpu with -march=native, medians of t_run_ms over
// 3 repetitions, C2 with R 64, one thread unless said otherwise), F07-R1 against F07-R2 with the
// same results: DenseSpinGlass{2000, 1} with S 100 and beta 0.01 -> 3 473 -> 199 ms (two threads
// 234 -> 99), 0.001 -> 0.01 1079 -> 953, 1 -> 3 152 -> 91, 3 -> 10 134 -> 67; with beta 0.01 -> 3
// and S 1, 5, 20: 228 -> 39, 81 -> 42, 132 -> 64; K2000, S 100, beta 0.01 -> 3: 478 -> 202 (two
// threads 243 -> 115); with -march=raptorlake (AVX2, the laptop) DenseSpinGlass{2000, 1} 492 ->
// 222. No regime was slower, so there is no rule to choose the old layout.
//
// Thresholds: with L <= L_tab = 4096 a table of L + 1 entries in the work area, filled at the
// start of every sweep of the block by calling sa_threshold (entry 0 = 2^32 accepts m <= 0), and
// read with the index max(m, 0); with L > L_tab a call of sa_threshold for every uphill proposal
// of every lane. Best signs: no copy at every improvement; every lane records the sweep and the
// spin of its last improvement, the block keeps the signs at the start of the current sweep and,
// for the lanes that improved during a sweep, the signs at its start (one masked copy per sweep
// with an improvement); at the end every lane is replayed from that state, with the same words
// and thresholds, up to the proposal of its last improvement (measured in phase 06: 17% faster
// than the copy at every improvement on K2000, equal on G1).
#include <sbmc/sa_openmp.hpp>

#include <omp.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <system_error>
#include <type_traits>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/philox.hpp>
#include <sbmc/sa.hpp>

namespace sbmc {
namespace {

constexpr int kLanes = kSaOpenmpReplicasPerBlock;
static_assert(kLanes == 8 || kLanes == 16 || kLanes == 32 || kLanes == 64,
              "RB must be 8, 16, 32 or 64");
constexpr auto kLaneCount = static_cast<std::size_t>(kLanes);

constexpr std::int64_t kMaxDenseSpins = 32768;                      // dense INT8 J: at most 1 GiB
constexpr std::int64_t kReplicaIndexLimit = std::int64_t{1} << 32;  // global replica indices
constexpr std::int32_t kMaxDenseWeight = 127;                       // |w| of a dense INT8 J
constexpr std::size_t kCacheLine = 64;
constexpr std::int64_t kTableLimit = 4096;                         // L_tab = 2^12
constexpr std::uint64_t kAlwaysAccepted = std::uint64_t{1} << 32;  // above every word

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

[[noreturn]] void invalid(const std::string& reason) {
  throw InvalidArgument("run_sa_openmp: " + reason);
}

// Shortest form that reads back as the same double ("inf", "nan" included).
std::string number(double v) {
  std::array<char, 32> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), v);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("?");
}

// The checks of run_sa_sequential on the parameters, same conditions and same order.
void check_params(const SaParams& p) {
  if (p.replicas < 1) {
    invalid("replicas = " + std::to_string(p.replicas) + " must be at least 1");
  }
  if (p.sweeps < 1) {
    invalid("sweeps = " + std::to_string(p.sweeps) + " must be at least 1");
  }
  if (!std::isfinite(p.beta0) || !(p.beta0 > 0.0)) {
    invalid("beta0 = " + number(p.beta0) + " must be finite and > 0");
  }
  if (!std::isfinite(p.beta1) || !(p.beta1 >= p.beta0)) {
    invalid("beta1 = " + number(p.beta1) + " must be finite and >= beta0 = " + number(p.beta0));
  }
  if (p.first_replica < 0) {
    invalid("first_replica = " + std::to_string(p.first_replica) + " is negative");
  }
  if (p.first_replica > kReplicaIndexLimit - p.replicas) {
    invalid("first_replica + replicas = " + std::to_string(p.first_replica) + " + " +
            std::to_string(p.replicas) + " exceeds 2^32");
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

// Compressed rows of a valid graph: the edges are canonical, so filling the rows in edge order
// gives every row in increasing column order.
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

// The rule of the format of J (that of C1): a Graph is dense when n <= 32768, |w| <= 127 on every
// edge and 8m >= n(n - 1) (density at least 25%).
bool dense_graph(const Graph& g) {
  if (g.n > kMaxDenseSpins) {
    return false;
  }
  const bool small_weights = std::all_of(g.edges.begin(), g.edges.end(), [](const Edge& e) {
    return e.w >= -kMaxDenseWeight && e.w <= kMaxDenseWeight;
  });
  return small_weights && 8 * static_cast<std::int64_t>(g.edges.size()) >= g.n * (g.n - 1);
}

// The cut of the final check, computed directly and without any check (decision L3 of phase 07):
// on the edges of a Graph, whatever the format of its internal J; on the internal dense rows of a
// DenseSpinGlass, with their ld.
struct DirectCut {
  const Graph* graph = nullptr;  // the Graph, or nullptr for a DenseSpinGlass
  const DenseJ* rows = nullptr;  // the internal J of a DenseSpinGlass

  std::int64_t operator()(std::span<const Spin> s) const noexcept {
    if (graph != nullptr) {
      return detail::cut_edges_unchecked(*graph, s);
    }
    return detail::cut_dense_rows_unchecked(
        std::span<const std::int8_t>(reinterpret_cast<const std::int8_t*>(rows->bytes.get()),
                                     rows->n * rows->ld),
        static_cast<std::int64_t>(rows->n), static_cast<std::int64_t>(rows->ld), s);
  }
};

// --- The path with compressed rows: arrays of n * kLanes elements in spin-major order ----------

// Unsigned type of the masked update: the exact result is in [-L, L] and fits Acc, the modular
// arithmetic gives it without signed overflow.
template <typename Acc>
using Wide = std::conditional_t<sizeof(Acc) == 8, std::uint64_t, std::uint32_t>;

// h[l] = h[l] + v * d[l] for every lane.
template <typename Acc>
void add_scaled(Acc* h, Acc v, const Acc* d) noexcept {
  using U = Wide<Acc>;
#pragma omp simd
  for (int l = 0; l < kLanes; ++l) {
    h[l] = static_cast<Acc>(static_cast<U>(h[l]) + static_cast<U>(v) * static_cast<U>(d[l]));
  }
}

// Initial local fields h_i = sum_j J_ij s_j of every lane, exact in Acc: every partial sum of a
// row is bounded by L. The columns of a row go one at a time, in increasing order.
template <typename Acc>
void initial_fields(const SparseJ& j, const Acc* s, Acc* h) noexcept {
  for (std::size_t i = 0; i < j.n; ++i) {
    Acc* const h_i = h + i * kLaneCount;
    std::fill(h_i, h_i + kLanes, Acc{0});
    for (auto k = static_cast<std::size_t>(j.row_begin[i]);
         k < static_cast<std::size_t>(j.row_begin[i + 1]); ++k) {
      add_scaled(h_i, static_cast<Acc>(j.value[k]),
                 s + static_cast<std::size_t>(j.column[k]) * kLaneCount);
    }
  }
}

// After the decisions of spin i: h_c += J_ic * d for every neighbour c (d = -2 s_i in the lanes
// that accepted, 0 in the others; J is symmetric, so row i holds the J_ci).
template <typename Acc>
void update_fields(const SparseJ& j, std::size_t i, const Acc* d, Acc* h) noexcept {
  for (auto k = static_cast<std::size_t>(j.row_begin[i]);
       k < static_cast<std::size_t>(j.row_begin[i + 1]); ++k) {
    add_scaled(h + static_cast<std::size_t>(j.column[k]) * kLaneCount, static_cast<Acc>(j.value[k]),
               d);
  }
}

// Words 4q .. 4q + 3 of the streams (seed, d, g[l]) of every lane: block q of each stream.
struct Words {
  alignas(kCacheLine) std::uint32_t word[4][kLanes];
};

void fill_words(std::uint64_t seed, RngDomain domain, const std::uint32_t* g, std::uint64_t q,
                Words& w) noexcept {
#pragma omp simd
  for (int l = 0; l < kLanes; ++l) {
    const U32x4 b = stream_block(seed, domain, g[l], q);
    w.word[0][l] = b.x;
    w.word[1][l] = b.y;
    w.word[2][l] = b.z;
    w.word[3][l] = b.w;
  }
}

// The per-lane state of a block that is not spin-major.
template <typename Acc>
struct Lanes {
  alignas(kCacheLine) std::uint32_t global[kLanes];  // global replica index (lane 0 for padding)
  alignas(kCacheLine) std::int64_t energy[kLanes];
  alignas(kCacheLine) std::int64_t best_energy[kLanes];
  alignas(kCacheLine) std::int64_t accepted[kLanes];
  alignas(kCacheLine) Acc d[kLanes];  // -2 s_i in the lanes that accepted, 0 in the others
  int best_sweep[kLanes];          // k + 1 for an improvement in sweep k, 0 for the initial state
  int best_spin[kLanes];           // spin i of the proposal of the last improvement, in its sweep
  bool improved_in_sweep[kLanes];  // an improvement in the current sweep
};

// The decisions of spin i with the table of the sweep (table[0] = 2^32 accepts m <= 0, table[m] =
// sa_threshold(beta_k, 2m)). Returns whether a lane accepted.
template <typename Acc>
bool decide(const std::uint64_t* table, const std::uint32_t* w, Acc* s_i, const Acc* h_i,
            Lanes<Acc>& lane) noexcept {
  int any = 0;
#pragma omp simd reduction(| : any)
  for (int l = 0; l < kLanes; ++l) {
    const std::int32_t m = static_cast<std::int32_t>(s_i[l]) * static_cast<std::int32_t>(h_i[l]);
    const bool accept = std::uint64_t{w[l]} < table[m > 0 ? m : 0];
    lane.d[l] = accept ? static_cast<Acc>(-2 * s_i[l]) : Acc{0};
    lane.energy[l] += accept ? 2 * std::int64_t{m} : std::int64_t{0};
    lane.accepted[l] += accept ? 1 : 0;
    s_i[l] = accept ? static_cast<Acc>(-s_i[l]) : s_i[l];
    any |= accept ? 1 : 0;
  }
  return any != 0;
}

// The decisions of spin i with a direct call of sa_threshold for every uphill proposal.
template <typename Acc>
bool decide(double beta, const std::uint32_t* w, Acc* s_i, const Acc* h_i, Lanes<Acc>& lane) {
  bool any = false;
  for (int l = 0; l < kLanes; ++l) {
    const std::int64_t m = std::int64_t{s_i[l]} * std::int64_t{h_i[l]};
    const bool accept = m <= 0 || std::uint64_t{w[l]} < sa_threshold(beta, 2 * m);
    lane.d[l] = accept ? static_cast<Acc>(-2 * s_i[l]) : Acc{0};
    lane.energy[l] += accept ? 2 * m : std::int64_t{0};
    lane.accepted[l] += accept ? 1 : 0;
    s_i[l] = accept ? static_cast<Acc>(-s_i[l]) : s_i[l];
    any = any || accept;
  }
  return any;
}

// What every block needs: sizes, parameters, the beta of every sweep and the outputs.
struct Plan {
  std::size_t n = 0;
  int replicas = 0;
  int sweeps = 0;
  std::uint64_t seed = 0;
  std::int64_t first_replica = 0;
  std::int64_t total = 0;
  std::int64_t norm = 0;  // L
  const double* beta = nullptr;
  SaResult* out = nullptr;
  Clock::time_point start;  // t0, the entry of run_sa_openmp
};

// The work area of one thread: four arrays of `elements` elements (spin-major, n * kLanes, with
// compressed rows; lane-major, kLanes * ld, with dense rows), the table of the sweep; each array on
// a 64-byte boundary.
template <typename Acc>
struct WorkArea {
  Acc* s = nullptr;      // signs
  Acc* h = nullptr;      // local fields
  Acc* start = nullptr;  // the signs at the start of the current sweep
  Acc* best_start =
      nullptr;  // per lane, the signs at the start of the sweep of its last improvement
  std::uint64_t* table = nullptr;
};

template <typename Acc>
std::size_t work_area_bytes(std::size_t elements, std::size_t table_size) {
  return 4 * round_up(elements * sizeof(Acc), kCacheLine) +
         round_up(table_size * sizeof(std::uint64_t), kCacheLine);
}

template <typename Acc>
WorkArea<Acc> work_area_at(std::byte* base, std::size_t elements, std::size_t table_size) {
  WorkArea<Acc> a;
  std::byte* next = base;
  const auto take = [&next](std::size_t bytes) {
    std::byte* const at = next;
    next += round_up(bytes, kCacheLine);
    return at;
  };
  a.s = reinterpret_cast<Acc*>(take(elements * sizeof(Acc)));
  a.h = reinterpret_cast<Acc*>(take(elements * sizeof(Acc)));
  a.start = reinterpret_cast<Acc*>(take(elements * sizeof(Acc)));
  a.best_start = reinterpret_cast<Acc*>(take(elements * sizeof(Acc)));
  if (table_size > 0) {
    a.table = reinterpret_cast<std::uint64_t*>(take(table_size * sizeof(std::uint64_t)));
  }
  return a;
}

// to[i][l] = from[i][l] in the lanes where mask[l], for every spin.
template <typename Acc>
void blend(std::size_t n, const bool* mask, const Acc* from, Acc* to) noexcept {
  for (std::size_t i = 0; i < n; ++i) {
    const Acc* const from_i = from + i * kLaneCount;
    Acc* const to_i = to + i * kLaneCount;
#pragma omp simd
    for (int l = 0; l < kLanes; ++l) {
      const Acc now = from_i[l];  // both values loaded: a blend, no masked load
      const Acc kept = to_i[l];
      to_i[l] = mask[l] ? now : kept;
    }
  }
}

// h_c += J_ic * d for the lane l only (the replay of the best state).
template <typename Acc>
void update_lane(const SparseJ& j, std::size_t i, Acc d, Acc* h, int l) noexcept {
  using U = Wide<Acc>;
  for (auto k = static_cast<std::size_t>(j.row_begin[i]);
       k < static_cast<std::size_t>(j.row_begin[i + 1]); ++k) {
    Acc& h_c = h[static_cast<std::size_t>(j.column[k]) * kLaneCount + static_cast<std::size_t>(l)];
    h_c = static_cast<Acc>(static_cast<U>(h_c) +
                           static_cast<U>(static_cast<Acc>(j.value[k])) * static_cast<U>(d));
  }
}

// Replay of lane l from the state at the start of sweep k (column l of s, fields h) up to and
// including the proposal of spin last of the sweep, with the words of the sweep and the thresholds
// of sa_threshold (the values of the table): the state of the lane right after that proposal, the
// improvement that gave its best energy.
template <typename Acc, typename Couplings>
void replay_lane(const Plan& plan, const Couplings& j, Acc* s, Acc* h, int l, std::uint32_t global,
                 int k, int last) {
  const double beta = plan.beta[k];
  const std::uint64_t first = static_cast<std::uint64_t>(k) * plan.n;
  for (int i = 0; i <= last; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    Acc& s_i = s[ui * kLaneCount + static_cast<std::size_t>(l)];
    const std::int64_t m =
        std::int64_t{s_i} * std::int64_t{h[ui * kLaneCount + static_cast<std::size_t>(l)]};
    const std::uint32_t w = sa_acceptance_word(plan.seed, global, first + ui);
    if (m <= 0 || std::uint64_t{w} < sa_threshold(beta, 2 * m)) {
      update_lane(j, ui, static_cast<Acc>(-2 * s_i), h, l);
      s_i = static_cast<Acc>(-s_i);
    }
  }
}

// All S sweeps of block b, then its rows of the outputs. No allocation; sa_threshold may throw
// (never with valid arguments): the caller catches. The two block functions start on a 64-byte
// boundary: the code of run_block is that of F07-R1, but the removal of the dense overloads moved
// it from the offset 48 modulo 64 to 16, and with 16 C2 on G1 was 7% slower in the sandbox
// (Xeon with AVX-512, -march=native); with 0 or 48 the times of F07-R1 (F07-R2).
template <typename Acc, bool Table, typename Couplings>
[[gnu::aligned(64)]] void run_block(const Plan& plan, const Couplings& j, const WorkArea<Acc>& a,
                                    int b) {
  const std::size_t n = plan.n;
  const int r0 = b * kLanes;
  const int real = std::min(kLanes, plan.replicas - r0);  // lanes l < real are replicas
  SaResult& out = *plan.out;
  Acc* const s = a.s;
  Acc* const h = a.h;

  Lanes<Acc> lane{};
  for (int l = 0; l < kLanes; ++l) {  // padding lanes follow the stream of lane 0
    lane.global[l] = static_cast<std::uint32_t>(plan.first_replica + r0 + (l < real ? l : 0));
  }

  // Initial state: s_i = sa_initial_spin(seed, g, i), read from final_spins, where the
  // preparation wrote it (the block reads only its rows, before writing the final spins there);
  // the padding lanes take the spins of lane 0, the values of the stream they follow.
  for (std::size_t i = 0; i < n; ++i) {
    Acc* const s_i = s + i * kLaneCount;
    for (int l = 0; l < kLanes; ++l) {
      const auto r = static_cast<std::size_t>(r0 + (l < real ? l : 0));
      s_i[l] = static_cast<Acc>(out.final_spins[r * n + i]);
    }
  }
  initial_fields(j, s, h);
  for (std::size_t i = 0; i < n; ++i) {
    const Acc* const s_i = s + i * kLaneCount;
    const Acc* const h_i = h + i * kLaneCount;
#pragma omp simd
    for (int l = 0; l < kLanes; ++l) {
      lane.energy[l] += std::int64_t{s_i[l]} * std::int64_t{h_i[l]};
    }
  }
  for (int l = 0; l < kLanes; ++l) {
    lane.energy[l] = -lane.energy[l] / 2;  // the sum is even
    lane.best_energy[l] = lane.energy[l];
  }
  std::copy(s, s + n * kLaneCount, a.start);
  std::copy(s, s + n * kLaneCount, a.best_start);  // read by the replay only after an improvement

  Words w{};
  std::uint64_t p = 0;  // proposal index k * n + i
  for (int k = 0; k < plan.sweeps; ++k) {
    const double beta = plan.beta[k];
    if constexpr (Table) {
      a.table[0] = kAlwaysAccepted;
      for (std::int64_t m = 1; m <= plan.norm; ++m) {
        a.table[m] = sa_threshold(beta, 2 * m);
      }
    }
    bool sweep_improved = false;
    for (std::size_t i = 0; i < n; ++i, ++p) {
      if ((p & 3u) == 0) {
        fill_words(plan.seed, RngDomain::sa_acceptance, lane.global, p >> 2, w);
      }
      const std::uint32_t* const word = w.word[p & 3u];
      Acc* const s_i = s + i * kLaneCount;
      const Acc* const h_i = h + i * kLaneCount;
      bool any = false;
      if constexpr (Table) {
        any = decide(a.table, word, s_i, h_i, lane);
      } else {
        any = decide(beta, word, s_i, h_i, lane);
      }
      if (!any) {
        continue;
      }
      update_fields(j, i, lane.d, h);
      for (int l = 0; l < kLanes; ++l) {
        if (lane.energy[l] < lane.best_energy[l]) {  // strict: ties keep the earlier state
          lane.best_energy[l] = lane.energy[l];
          lane.best_sweep[l] = k + 1;
          lane.best_spin[l] = static_cast<int>(i);
          lane.improved_in_sweep[l] = true;
          sweep_improved = true;
        }
      }
    }
    // Best signs: the lanes that improved during the sweep keep its starting state (one copy per
    // sweep with an improvement); the state of the improvement is replayed from it at the end.
    if (sweep_improved) {
      blend(n, lane.improved_in_sweep, a.start, a.best_start);
      std::fill(lane.improved_in_sweep, lane.improved_in_sweep + kLanes, false);
    }
    std::copy(s, s + n * kLaneCount, a.start);
  }

  for (int l = 0; l < real; ++l) {
    const auto r = static_cast<std::size_t>(r0 + l);
    SaReplicaResult& rep = out.per_replica[r];
    rep.best_energy = lane.best_energy[l];
    rep.best_cut = (plan.total - lane.best_energy[l]) / 2;
    rep.best_sweep = lane.best_sweep[l];
    rep.final_cut = (plan.total - lane.energy[l]) / 2;
    rep.accepted = lane.accepted[l];
    for (std::size_t i = 0; i < n; ++i) {
      out.final_spins[r * n + i] =
          static_cast<Spin>(s[i * kLaneCount + static_cast<std::size_t>(l)]);
    }
  }

  // Best signs: the fields of the kept states (all lanes at once), then every lane replayed from
  // the start of the sweep of its last improvement; best_sweep 0 is the initial state.
  initial_fields(j, a.best_start, h);
  for (int l = 0; l < real; ++l) {
    const auto r = static_cast<std::size_t>(r0 + l);
    if (lane.best_sweep[l] == 0) {
      for (std::size_t i = 0; i < n; ++i) {
        out.best_spins[r * n + i] =
            sa_initial_spin(plan.seed, lane.global[l], static_cast<std::uint32_t>(i));
      }
      continue;
    }
    replay_lane(plan, j, a.best_start, h, l, lane.global[l], lane.best_sweep[l] - 1,
                lane.best_spin[l]);
    for (std::size_t i = 0; i < n; ++i) {
      out.best_spins[r * n + i] =
          static_cast<Spin>(a.best_start[i * kLaneCount + static_cast<std::size_t>(l)]);
    }
  }
}

// --- The path with dense rows: arrays of kLanes * ld elements in lane-major order --------------
//
// Element (lane l, spin c) of s, h, start and best_start at l * ld + c, with ld = n rounded up to a
// multiple of 64 bytes and divided by sizeof(Acc): every lane starts on a 64-byte boundary. The
// columns n .. ld - 1 of a lane are padding, set to 0 at the start of the block and never read by
// the loops over the spins. For the proposal of spin i the s_i and h_i of the RB lanes are
// gathered, decided with the same decide as the other path, the new signs scattered back, then only
// the lanes that accepted update their fields, h[l * ld + c] += J_ic d_l for c = 0 .. n - 1: a
// contiguous row of the lane and a contiguous row of J, vectorized over c. The lanes that rejected
// do not touch their fields (with the betas of K2000 a lane accepts about 11% of the proposals).

// The stride of a lane: n rounded up to a multiple of 64 bytes, in elements of Acc.
template <typename Acc>
std::size_t lane_stride(std::size_t n) noexcept {
  return round_up(n * sizeof(Acc), kCacheLine) / sizeof(Acc);
}

// Elements of one array of the work area: lane-major with dense rows, spin-major otherwise.
template <typename Acc, typename Couplings>
std::size_t area_elements(std::size_t n) noexcept {
  if constexpr (std::is_same_v<Couplings, DenseJ>) {
    return kLaneCount * lane_stride<Acc>(n);
  } else {
    return n * kLaneCount;
  }
}

// h_c += J_ic * d for every spin c of one lane (h: the row of the lane), d = -2 s_i of the lane.
template <typename Acc>
void lane_update(const DenseJ& j, std::size_t i, Acc d, Acc* h) noexcept {
  using U = Wide<Acc>;
  const std::int8_t* const row = j.row(i);
#pragma omp simd
  for (std::size_t c = 0; c < j.n; ++c) {
    h[c] = static_cast<Acc>(static_cast<U>(h[c]) +
                            static_cast<U>(static_cast<Acc>(row[c])) * static_cast<U>(d));
  }
}

// h_i = sum_c J_ic s_c for every spin i of one lane (s and h: the rows of the lane), as a sum of
// the rows c of J times s_c (J is symmetric): contiguous loops over i, without horizontal sums.
// Exact: every partial sum is bounded by L, and the modular arithmetic of Wide<Acc> gives it.
template <typename Acc>
void lane_fields(const DenseJ& j, const Acc* s, Acc* h) noexcept {
  std::fill(h, h + j.n, Acc{0});
  for (std::size_t c = 0; c < j.n; ++c) {
    lane_update(j, c, s[c], h);
  }
}

// Replay of one lane (s and h: its rows) from the state at the start of sweep k up to and
// including the proposal of spin last of the sweep, as replay_lane of the other path.
template <typename Acc>
void replay_dense_lane(const Plan& plan, const DenseJ& j, Acc* s, Acc* h, std::uint32_t global,
                       int k, int last) {
  const double beta = plan.beta[k];
  const std::uint64_t first = static_cast<std::uint64_t>(k) * plan.n;
  for (int i = 0; i <= last; ++i) {
    const auto ui = static_cast<std::size_t>(i);
    const std::int64_t m = std::int64_t{s[ui]} * std::int64_t{h[ui]};
    const std::uint32_t w = sa_acceptance_word(plan.seed, global, first + ui);
    if (m <= 0 || std::uint64_t{w} < sa_threshold(beta, 2 * m)) {
      lane_update(j, ui, static_cast<Acc>(-2 * s[ui]), h);
      s[ui] = static_cast<Acc>(-s[ui]);
    }
  }
}

// All S sweeps of block b with dense rows, then its rows of the outputs: the same decisions, in
// the same order, as run_block, in the lane-major layout. No allocation; sa_threshold may throw
// (never with valid arguments): the caller catches.
template <typename Acc, bool Table>
[[gnu::aligned(64)]] void run_dense_block(const Plan& plan, const DenseJ& j, const WorkArea<Acc>& a,
                                          int b) {
  const std::size_t n = plan.n;
  const std::size_t ld = lane_stride<Acc>(n);
  const std::size_t elements = kLaneCount * ld;
  const int r0 = b * kLanes;
  const int real = std::min(kLanes, plan.replicas - r0);  // lanes l < real are replicas
  SaResult& out = *plan.out;
  Acc* const s = a.s;
  Acc* const h = a.h;
  const auto lane_row = [ld](Acc* base, int l) { return base + static_cast<std::size_t>(l) * ld; };

  Lanes<Acc> lane{};
  for (int l = 0; l < kLanes; ++l) {  // padding lanes follow the stream of lane 0
    lane.global[l] = static_cast<std::uint32_t>(plan.first_replica + r0 + (l < real ? l : 0));
  }

  // Initial state: s_i = sa_initial_spin(seed, g, i), read from final_spins, where the
  // preparation wrote it (the block reads only its rows, before writing the final spins there);
  // the padding lanes take the spins of lane 0. The padding columns are 0.
  std::fill(s, s + elements, Acc{0});
  std::fill(h, h + elements, Acc{0});
  for (int l = 0; l < kLanes; ++l) {
    const auto r = static_cast<std::size_t>(r0 + (l < real ? l : 0));
    Acc* const s_l = lane_row(s, l);
    for (std::size_t i = 0; i < n; ++i) {
      s_l[i] = static_cast<Acc>(out.final_spins[r * n + i]);
    }
    lane_fields(j, s_l, lane_row(h, l));
    std::int64_t sum = 0;
    const Acc* const h_l = lane_row(h, l);
    for (std::size_t i = 0; i < n; ++i) {
      sum += std::int64_t{s_l[i]} * std::int64_t{h_l[i]};
    }
    lane.energy[l] = -sum / 2;  // the sum is even
    lane.best_energy[l] = lane.energy[l];
  }
  std::copy(s, s + elements, a.start);
  std::copy(s, s + elements, a.best_start);  // read by the replay only after an improvement

  Words w{};
  alignas(kCacheLine) Acc s_i[kLanes];
  alignas(kCacheLine) Acc h_i[kLanes];
  std::uint64_t p = 0;  // proposal index k * n + i
  for (int k = 0; k < plan.sweeps; ++k) {
    const double beta = plan.beta[k];
    if constexpr (Table) {
      a.table[0] = kAlwaysAccepted;
      for (std::int64_t m = 1; m <= plan.norm; ++m) {
        a.table[m] = sa_threshold(beta, 2 * m);
      }
    }
    bool sweep_improved = false;
    for (std::size_t i = 0; i < n; ++i, ++p) {
      if ((p & 3u) == 0) {
        fill_words(plan.seed, RngDomain::sa_acceptance, lane.global, p >> 2, w);
      }
      const std::uint32_t* const word = w.word[p & 3u];
      for (int l = 0; l < kLanes; ++l) {
        const std::size_t at = static_cast<std::size_t>(l) * ld + i;
        s_i[l] = s[at];
        h_i[l] = h[at];
      }
      bool any = false;
      if constexpr (Table) {
        any = decide(a.table, word, s_i, h_i, lane);
      } else {
        any = decide(beta, word, s_i, h_i, lane);
      }
      if (!any) {
        continue;
      }
      for (int l = 0; l < kLanes; ++l) {
        if (lane.d[l] != 0) {  // only the lanes that accepted: new sign and fields
          s[static_cast<std::size_t>(l) * ld + i] = s_i[l];
          lane_update(j, i, lane.d[l], lane_row(h, l));
        }
      }
      for (int l = 0; l < kLanes; ++l) {
        if (lane.energy[l] < lane.best_energy[l]) {  // strict: ties keep the earlier state
          lane.best_energy[l] = lane.energy[l];
          lane.best_sweep[l] = k + 1;
          lane.best_spin[l] = static_cast<int>(i);
          lane.improved_in_sweep[l] = true;
          sweep_improved = true;
        }
      }
    }
    // Best signs: the lanes that improved during the sweep keep its starting state (one copy per
    // lane that improved); the state of the improvement is replayed from it at the end.
    if (sweep_improved) {
      for (int l = 0; l < kLanes; ++l) {
        if (lane.improved_in_sweep[l]) {
          std::copy(lane_row(a.start, l), lane_row(a.start, l) + n, lane_row(a.best_start, l));
          lane.improved_in_sweep[l] = false;
        }
      }
    }
    std::copy(s, s + elements, a.start);
  }

  for (int l = 0; l < real; ++l) {
    const auto r = static_cast<std::size_t>(r0 + l);
    SaReplicaResult& rep = out.per_replica[r];
    rep.best_energy = lane.best_energy[l];
    rep.best_cut = (plan.total - lane.best_energy[l]) / 2;
    rep.best_sweep = lane.best_sweep[l];
    rep.final_cut = (plan.total - lane.energy[l]) / 2;
    rep.accepted = lane.accepted[l];
    const Acc* const s_l = lane_row(s, l);
    for (std::size_t i = 0; i < n; ++i) {
      out.final_spins[r * n + i] = static_cast<Spin>(s_l[i]);
    }
  }

  // Best signs: every lane replayed from the start of the sweep of its last improvement, with the
  // fields of that state (computed for the lanes that replay only); best_sweep 0 is the initial
  // state.
  for (int l = 0; l < real; ++l) {
    const auto r = static_cast<std::size_t>(r0 + l);
    if (lane.best_sweep[l] == 0) {
      for (std::size_t i = 0; i < n; ++i) {
        out.best_spins[r * n + i] =
            sa_initial_spin(plan.seed, lane.global[l], static_cast<std::uint32_t>(i));
      }
      continue;
    }
    Acc* const s_l = lane_row(a.best_start, l);
    Acc* const h_l = lane_row(h, l);
    lane_fields(j, s_l, h_l);
    replay_dense_lane(plan, j, s_l, h_l, lane.global[l], lane.best_sweep[l] - 1, lane.best_spin[l]);
    for (std::size_t i = 0; i < n; ++i) {
      out.best_spins[r * n + i] = static_cast<Spin>(s_l[i]);
    }
  }
}

// Outcome of the final check of one replica.
struct CheckStatus {
  bool mismatch = false;
  std::int64_t best_cut = 0;
  std::int64_t final_cut = 0;
};

// The initial spins of every replica in final_spins, replica-major: final_spins[r * n + i] =
// sa_initial_spin(seed, first_replica + r, i), computed from whole Philox blocks of the stream (4
// words each), with a team of team_request threads. Every element is written once; no
// allocation, no exception.
void initial_spins(const Plan& plan, int team_request) noexcept {
  const auto n = static_cast<std::int64_t>(plan.n);
  const std::int64_t blocks_per_replica = (n + 3) / 4;
  const std::int64_t tasks = blocks_per_replica * plan.replicas;
  Spin* const s0 = plan.out->final_spins.data();
#pragma omp parallel for num_threads(team_request) schedule(static)
  for (std::int64_t task = 0; task < tasks; ++task) {
    const std::int64_t r = task / blocks_per_replica;
    const std::int64_t b = task % blocks_per_replica;
    const auto replica = static_cast<std::uint32_t>(plan.first_replica + r);
    const U32x4 block = stream_block(plan.seed, RngDomain::sa_initial_spins, replica,
                                     static_cast<std::uint64_t>(b));
    const std::uint32_t words[4] = {block.x, block.y, block.z, block.w};
    const std::int64_t i0 = 4 * b;
    const std::int64_t count = std::min<std::int64_t>(4, n - i0);
    for (std::int64_t t = 0; t < count; ++t) {
      s0[r * n + i0 + t] = coupling_from_word(words[t]);  // = sa_initial_spin(seed, replica, i)
    }
  }
}

template <typename Acc, bool Table, typename Couplings>
void run_blocks(const DirectCut& direct_cut, const Couplings& j, const Plan& plan, int threads,
                int& team) {
  const int blocks = (plan.replicas + kLanes - 1) / kLanes;
  const int team_request = std::min(threads, blocks);
  const std::size_t table_size = Table ? static_cast<std::size_t>(plan.norm) + 1 : 0;
  const std::size_t elements = area_elements<Acc, Couplings>(plan.n);
  const std::size_t area_bytes = work_area_bytes<Acc>(elements, table_size);
  const AlignedBytes areas = allocate_aligned(area_bytes * static_cast<std::size_t>(team_request));
  std::vector<CheckStatus> status(static_cast<std::size_t>(plan.replicas));
  std::vector<std::exception_ptr> block_error(static_cast<std::size_t>(blocks));
  initial_spins(plan, team_request);

  // Run: all S sweeps of every block.
  const Clock::time_point t1 = Clock::now();
  int team_size = 0;
#pragma omp parallel num_threads(team_request)
  {
#pragma omp single
    team_size = omp_get_num_threads();

    const WorkArea<Acc> area =
        work_area_at<Acc>(areas.get() + static_cast<std::size_t>(omp_get_thread_num()) * area_bytes,
                          elements, table_size);
#pragma omp for schedule(dynamic, 1)
    for (int b = 0; b < blocks; ++b) {
      try {
        if constexpr (std::is_same_v<Couplings, DenseJ>) {
          run_dense_block<Acc, Table>(plan, j, area, b);
        } else {
          run_block<Acc, Table>(plan, j, area, b);
        }
      } catch (...) {
        block_error[static_cast<std::size_t>(b)] = std::current_exception();
      }
    }
  }
  const Clock::time_point t2 = Clock::now();

  // Check: the direct cuts of the best and of the final spins of every replica whose block ran.
  const SaResult& out = *plan.out;
  const int replicas = plan.replicas;
#pragma omp parallel num_threads(team_request)
  {
#pragma omp for schedule(static)
    for (int r = 0; r < replicas; ++r) {
      const auto ur = static_cast<std::size_t>(r);
      if (block_error[ur / kLaneCount]) {
        continue;
      }
      const std::int64_t best_cut =
          direct_cut(std::span<const Spin>(out.best_spins.data() + ur * plan.n, plan.n));
      const std::int64_t final_cut =
          direct_cut(std::span<const Spin>(out.final_spins.data() + ur * plan.n, plan.n));
      if (best_cut != out.per_replica[ur].best_cut || final_cut != out.per_replica[ur].final_cut) {
        status[ur].mismatch = true;
        status[ur].best_cut = best_cut;
        status[ur].final_cut = final_cut;
      }
    }
  }
  const Clock::time_point t3 = Clock::now();
  team = team_size;

  for (std::size_t r = 0; r < status.size(); ++r) {
    if (r % kLaneCount == 0 && block_error[r / kLaneCount]) {
      std::rethrow_exception(block_error[r / kLaneCount]);
    }
    if (status[r].mismatch) {
      const SaReplicaResult& rep = out.per_replica[r];
      throw Error("run_sa_openmp: internal error: replica " + std::to_string(r) +
                  ": the direct cut gives best " + std::to_string(status[r].best_cut) +
                  " and final " + std::to_string(status[r].final_cut) +
                  ", the incremental energy gives best " + std::to_string(rep.best_cut) +
                  " and final " + std::to_string(rep.final_cut));
    }
  }
  plan.out->times.prep_ms = milliseconds(plan.start, t1);
  plan.out->times.run_ms = milliseconds(t1, t2);
  plan.out->times.check_ms = milliseconds(t2, t3);
}

// The accumulator and the path of the thresholds from L: the table (L <= L_tab, so int16) or
// direct calls with int16, int32 or int64.
template <typename Couplings>
void run_with_accumulator(const DirectCut& direct_cut, const Couplings& j, const Plan& plan,
                          int threads, int& team) {
  if (plan.norm <= kTableLimit) {
    run_blocks<std::int16_t, true>(direct_cut, j, plan, threads, team);
  } else if (plan.norm <= 32767) {
    run_blocks<std::int16_t, false>(direct_cut, j, plan, threads, team);
  } else if (plan.norm <= 2147483647) {
    run_blocks<std::int32_t, false>(direct_cut, j, plan, threads, team);
  } else {
    run_blocks<std::int64_t, false>(direct_cut, j, plan, threads, team);
  }
}

}  // namespace

SaResult run_sa_openmp(const Instance& inst, const SaParams& params, int threads) {
  const Clock::time_point t0 = Clock::now();
  if (threads < 0) {
    invalid("threads = " + std::to_string(threads) +
            " must be >= 0 (0: the default number of OpenMP threads)");
  }
  const int requested = threads > 0 ? threads : omp_get_max_threads();
  check_params(params);
  if (const auto* dense = std::get_if<DenseSpinGlass>(&inst);
      dense != nullptr && dense->n > kMaxDenseSpins) {
    invalid("n = " + std::to_string(dense->n) +
            ": C2 stores the J of a DenseSpinGlass as dense INT8 and accepts n <= 32768");
  }
  const std::int64_t total = total_weight(inst);  // full check of the instance

  std::vector<double> beta(static_cast<std::size_t>(params.sweeps));
  for (int k = 0; k < params.sweeps; ++k) {
    beta[static_cast<std::size_t>(k)] = sa_beta(params.beta0, params.beta1, params.sweeps, k);
  }

  const std::int64_t n = num_spins(inst);
  const auto un = static_cast<std::size_t>(n);
  const auto state_size = un * static_cast<std::size_t>(params.replicas);
  SaResult result;
  result.n = n;
  result.replicas = params.replicas;
  result.sweeps = params.sweeps;
  result.first_replica = params.first_replica;
  result.beta0 = params.beta0;
  result.beta1 = params.beta1;
  result.total_weight = total;
  result.per_replica.resize(static_cast<std::size_t>(params.replicas));
  result.best_spins.resize(state_size);
  result.final_spins.resize(state_size);

  Plan plan;
  plan.n = un;
  plan.replicas = params.replicas;
  plan.sweeps = params.sweeps;
  plan.seed = params.seed;
  plan.first_replica = params.first_replica;
  plan.total = total;
  plan.beta = beta.data();
  plan.out = &result;
  plan.start = t0;

  int team = 0;
  if (const auto* g = std::get_if<Graph>(&inst)) {
    plan.norm = g->edges.empty() ? 0 : max_row_norm(*g);
    const DirectCut direct_cut{g, nullptr};
    if (dense_graph(*g)) {
      run_with_accumulator(direct_cut, make_dense(*g), plan, requested, team);
    } else {
      run_with_accumulator(direct_cut, make_sparse(*g), plan, requested, team);
    }
  } else {
    const auto& dense = std::get<DenseSpinGlass>(inst);
    plan.norm = dense.n - 1;
    const DenseJ j = make_dense(dense);
    const DirectCut direct_cut{nullptr, &j};
    run_with_accumulator(direct_cut, j, plan, requested, team);
  }
  result.threads = team;
  return result;
}

}  // namespace sbmc
