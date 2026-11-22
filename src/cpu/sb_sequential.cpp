// Sequential reference C0 of simulated bifurcation (sbmc/sb_sequential.hpp).
//
// The replicas are independent and run one after the other, each for all M steps. The step loop
// works only on internal copies (couplings, positions, momenta, signs): the instance is checked
// in full in the preparation, by total_weight and, with the automatic c0, also by auto_c0, and
// never again (B5). The final check recomputes the best
// cut of every replica directly, without any check of the instance (decision L3 of phase 07):
// on the edges for a Graph (detail::cut_edges_unchecked), on the internal INT8 rows of J with
// ld = n for a DenseSpinGlass (detail::cut_dense_rows_unchecked); no public function of the
// instance is called after the entry. No OpenMP here: the library functions called in the
// preparation (total_weight, auto_c0, dense_rows_int8, make_initial_state) may use it, with
// results that do not depend on the number of threads.
//
// Times (sbmc/solve_times.hpp): t0 is the first instruction of run_sb_sequential; the
// preparation [t0, t1) covers the preconditions, total_weight, c0, the internal J, the pressure
// schedule, the outputs, the trace, the work areas and x0 of every replica in final_x
// (make_initial_state); the run [t1, t2) is the replica loop; the check [t2, t3) is the final
// check loop.
#include <sbmc/sb_sequential.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

constexpr std::int64_t kMaxDenseSpins = 32768;                      // dense INT8 J: at most 1 GiB
constexpr std::int64_t kReplicaIndexLimit = std::int64_t{1} << 32;  // global replica indices

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point from, Clock::time_point to) {
  return std::chrono::duration<double, std::milli>(to - from).count();
}

[[noreturn]] void invalid(const std::string& reason) {
  throw InvalidArgument("run_sb_sequential: " + reason);
}

bool finite_positive(float v) {
  return std::isfinite(v) && v > 0.0f;
}

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

// J of an explicit graph in compressed rows: J_ij = -w_ij in both directions, the neighbours of
// every row in increasing j.
struct SparseCouplings {
  std::int64_t n = 0;
  std::vector<std::int64_t> row_begin;  // n + 1 offsets
  std::vector<std::int32_t> column;
  std::vector<std::int32_t> value;

  // sum_j J_ij s_j, exact.
  std::int64_t spin_field(std::int64_t i, const Spin* s) const noexcept {
    std::int64_t sum = 0;
    for (std::int64_t k = row_begin[static_cast<std::size_t>(i)];
         k < row_begin[static_cast<std::size_t>(i) + 1]; ++k) {
      sum +=
          std::int64_t{value[static_cast<std::size_t>(k)]} * s[column[static_cast<std::size_t>(k)]];
    }
    return sum;
  }

  // acc = +0.0; acc += (double)J_ij * (double)x_j for increasing j (zero terms skipped).
  double position_field(std::int64_t i, const float* x) const noexcept {
    double acc = 0.0;
    for (std::int64_t k = row_begin[static_cast<std::size_t>(i)];
         k < row_begin[static_cast<std::size_t>(i) + 1]; ++k) {
      acc += static_cast<double>(value[static_cast<std::size_t>(k)]) *
             static_cast<double>(x[column[static_cast<std::size_t>(k)]]);
    }
    return acc;
  }
};

// J of a dense spin glass, INT8, row-major n x n.
struct DenseCouplings {
  std::int64_t n = 0;
  std::vector<std::int8_t> j;

  std::int64_t spin_field(std::int64_t i, const Spin* s) const noexcept {
    const std::int8_t* const row = j.data() + i * n;
    std::int64_t sum = 0;
    for (std::int64_t c = 0; c < n; ++c) {
      sum += std::int64_t{row[c]} * s[c];
    }
    return sum;
  }

  // Same operations as SparseCouplings::position_field: the zero terms (the diagonal) are
  // skipped, in increasing j.
  double position_field(std::int64_t i, const float* x) const noexcept {
    const std::int8_t* const row = j.data() + i * n;
    double acc = 0.0;
    for (std::int64_t c = 0; c < n; ++c) {
      if (row[c] != 0) {
        acc += static_cast<double>(row[c]) * static_cast<double>(x[c]);
      }
    }
    return acc;
  }
};

// Compressed rows of a valid graph. The edges are canonical (sorted by (u, v)), so filling the
// rows in edge order gives every row in increasing j: row i first receives the u < i of the
// edges (u, i), in increasing u, then the v > i of the edges (i, v), in increasing v.
SparseCouplings make_couplings(const Graph& g) {
  SparseCouplings c;
  c.n = g.n;
  const auto n = static_cast<std::size_t>(g.n);
  c.row_begin.assign(n + 1, 0);
  for (const Edge& e : g.edges) {
    ++c.row_begin[static_cast<std::size_t>(e.u) + 1];
    ++c.row_begin[static_cast<std::size_t>(e.v) + 1];
  }
  for (std::size_t i = 0; i < n; ++i) {
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

DenseCouplings make_couplings(const DenseSpinGlass& g) {
  DenseCouplings c;
  c.n = g.n;
  c.j.resize(static_cast<std::size_t>(g.n * g.n));
  dense_rows_int8(g, 0, g.n, g.n, c.j);
  return c;
}

// The cut of the final check, computed directly and without any check (decision L3 of phase 07):
// on the edges of a Graph, on the internal dense rows (ld = n) of a DenseSpinGlass.
std::int64_t direct_cut(const Graph& g, const SparseCouplings& /*c*/,
                        std::span<const Spin> s) noexcept {
  return detail::cut_edges_unchecked(g, s);
}

std::int64_t direct_cut(const DenseSpinGlass& /*g*/, const DenseCouplings& c,
                        std::span<const Spin> s) noexcept {
  return detail::cut_dense_rows_unchecked(c.j, c.n, c.n, s);
}

// E(s) = -(sum_i s_i F_i) / 2 with F = J s, exact in int64 (the sum is even).
template <typename Couplings>
std::int64_t energy(const Couplings& c, const Spin* s) {
  std::int64_t sum = 0;
  for (std::int64_t i = 0; i < c.n; ++i) {
    sum += s[i] * c.spin_field(i, s);
  }
  return -sum / 2;
}

template <typename G, typename Couplings>
SbResult run(const G& g, const Couplings& c, std::int64_t total, float c0, const SbParams& params,
             int trace_steps, Clock::time_point t0) {
  const std::int64_t n = c.n;
  const int replicas = params.replicas;
  const int steps = params.steps;
  const bool discrete = params.algorithm == SbAlgorithm::discrete;
  const std::vector<float> p = pressure_schedule(params.a0, steps);
  const float dt = params.dt;
  const float dt_a0 = params.dt * params.a0;
  const auto un = static_cast<std::size_t>(n);
  const auto state_size = un * static_cast<std::size_t>(replicas);

  SbResult result;
  result.n = n;
  result.replicas = replicas;
  result.steps = steps;
  result.first_replica = params.first_replica;
  result.c0 = c0;
  result.total_weight = total;
  result.trace_steps = trace_steps;
  result.per_replica.resize(static_cast<std::size_t>(replicas));
  result.best_spins.resize(state_size);
  result.final_x = make_initial_state(n, replicas, params.seed, params.first_replica).x0;
  result.final_y.assign(state_size, 0.0f);
  result.trace_x.resize(static_cast<std::size_t>(trace_steps) * state_size);
  result.trace_y.resize(static_cast<std::size_t>(trace_steps) * state_size);

  std::vector<Spin> s(un);
  std::vector<float> field(un);

  const Clock::time_point t1 = Clock::now();
  for (int r = 0; r < replicas; ++r) {
    const std::size_t offset = static_cast<std::size_t>(r) * un;
    float* const x = result.final_x.data() + offset;
    float* const y = result.final_y.data() + offset;
    Spin* const best = result.best_spins.data() + offset;
    bool evaluated = false;
    std::int64_t best_energy = 0;
    int best_step = 0;
    // The configuration s of step m becomes the best one if it is the first evaluated or
    // strictly better: ties keep the smallest m.
    const auto consider = [&](std::int64_t e, int m) {
      if (!evaluated || e < best_energy) {
        evaluated = true;
        best_energy = e;
        best_step = m;
        std::copy(s.begin(), s.end(), best);
      }
    };

    for (int m = 0; m < steps; ++m) {
      for (std::size_t i = 0; i < un; ++i) {
        s[i] = sb_sign(x[i]);
      }
      if (discrete) {
        std::int64_t sum = 0;
        for (std::size_t i = 0; i < un; ++i) {
          const std::int64_t f = c.spin_field(static_cast<std::int64_t>(i), s.data());
          field[i] = static_cast<float>(f);
          sum += s[i] * f;
        }
        consider(-sum / 2, m);
      } else {
        for (std::size_t i = 0; i < un; ++i) {
          field[i] = static_cast<float>(c.position_field(static_cast<std::int64_t>(i), x));
        }
        if (params.eval_every > 0 && m % params.eval_every == 0) {
          consider(energy(c, s.data()), m);
        }
      }
      const float pm = p[static_cast<std::size_t>(m)];
      for (std::size_t i = 0; i < un; ++i) {
        sb_update(x[i], y[i], field[i], pm, dt, c0, dt_a0);
      }
      if (m < trace_steps) {
        const std::size_t at = (static_cast<std::size_t>(m) * replicas + r) * un;
        std::copy(x, x + n, result.trace_x.begin() + static_cast<std::ptrdiff_t>(at));
        std::copy(y, y + n, result.trace_y.begin() + static_cast<std::ptrdiff_t>(at));
      }
    }

    for (std::size_t i = 0; i < un; ++i) {
      s[i] = sb_sign(x[i]);
    }
    const std::int64_t final_energy = energy(c, s.data());
    consider(final_energy, steps);
    SbReplicaResult& out = result.per_replica[static_cast<std::size_t>(r)];
    out.best_energy = best_energy;
    out.best_cut = (total - best_energy) / 2;
    out.best_step = best_step;
    out.final_cut = (total - final_energy) / 2;
  }

  const Clock::time_point t2 = Clock::now();

  // Final check: the direct cut of the best configuration of every replica against its best_cut.
  // The first mismatch (the smallest replica) is thrown after t3.
  int mismatch = -1;
  std::int64_t mismatch_cut = 0;
  for (int r = 0; r < replicas; ++r) {
    const std::span<const Spin> spins(result.best_spins.data() + static_cast<std::size_t>(r) * un,
                                      un);
    const std::int64_t cut = direct_cut(g, c, spins);
    if (cut != result.per_replica[static_cast<std::size_t>(r)].best_cut) {
      mismatch = r;
      mismatch_cut = cut;
      break;
    }
  }
  const Clock::time_point t3 = Clock::now();
  if (mismatch >= 0) {
    throw Error("run_sb_sequential: internal error: replica " + std::to_string(mismatch) +
                ": the direct cut gives " + std::to_string(mismatch_cut) +
                ", (W - best_energy) / 2 gives " +
                std::to_string(result.per_replica[static_cast<std::size_t>(mismatch)].best_cut));
  }
  result.times.prep_ms = milliseconds(t0, t1);
  result.times.run_ms = milliseconds(t1, t2);
  result.times.check_ms = milliseconds(t2, t3);
  return result;
}

}  // namespace

SbResult run_sb_sequential(const Instance& inst, const SbParams& params, int trace_steps) {
  const Clock::time_point t0 = Clock::now();
  check_params(params, trace_steps);
  if (const auto* dense = std::get_if<DenseSpinGlass>(&inst);
      dense != nullptr && dense->n > kMaxDenseSpins) {
    invalid("n = " + std::to_string(dense->n) +
            ": C0 stores the J of a DenseSpinGlass as dense INT8 and accepts n <= 32768");
  }
  const std::int64_t total = total_weight(inst);  // full check of the instance
  const float c0 = params.c0 > 0.0f ? params.c0 : auto_c0(inst);
  return std::visit(
      [&](const auto& g) { return run(g, make_couplings(g), total, c0, params, trace_steps, t0); },
      inst);
}

}  // namespace sbmc
