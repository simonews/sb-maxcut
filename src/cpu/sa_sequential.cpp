// Sequential reference of simulated annealing (sbmc/sa_sequential.hpp).
//
// The replicas are independent and run one after the other, each for all S sweeps. The sweep
// loop works only on internal copies (couplings, spins, local fields): the instance is checked
// once at the entry, and the public cut_direct is called twice per replica at the end (B5). No
// OpenMP here: the library functions called at the entry (total_weight, dense_rows_int8) may use
// it, with results that do not depend on the number of threads.
#include <sbmc/sa_sequential.hpp>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <system_error>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/philox.hpp>
#include <sbmc/sa.hpp>

namespace sbmc {
namespace {

constexpr std::int64_t kMaxDenseSpins = 32768;                      // dense INT8 J: at most 1 GiB
constexpr std::int64_t kReplicaIndexLimit = std::int64_t{1} << 32;  // global replica indices

[[noreturn]] void invalid(const std::string& reason) {
  throw InvalidArgument("run_sa_sequential: " + reason);
}

// Shortest form that reads back as the same double ("inf", "nan" included).
std::string number(double v) {
  std::array<char, 32> text{};
  const auto [end, error] = std::to_chars(text.data(), text.data() + text.size(), v);
  return error == std::errc{} ? std::string(text.data(), end) : std::string("?");
}

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

// J of an explicit graph in compressed rows: J_ij = -w_ij in both directions, the neighbours of
// every row in increasing j (filled in the canonical edge order, as in C0).
struct SparseCouplings {
  std::int64_t n = 0;
  std::vector<std::int64_t> row_begin;  // n + 1 offsets
  std::vector<std::int32_t> column;
  std::vector<std::int32_t> value;

  // h_i = sum_j J_ij s_j, exact.
  std::int64_t field(std::int64_t i, const Spin* s) const noexcept {
    std::int64_t sum = 0;
    for (std::int64_t k = row_begin[static_cast<std::size_t>(i)];
         k < row_begin[static_cast<std::size_t>(i) + 1]; ++k) {
      sum +=
          std::int64_t{value[static_cast<std::size_t>(k)]} * s[column[static_cast<std::size_t>(k)]];
    }
    return sum;
  }

  // Before the flip of spin i, whose sign is s_i: h_j -= 2 J_ji s_i for every neighbour j.
  void flip(std::int64_t i, Spin s_i, std::int64_t* h) const noexcept {
    const std::int64_t twice = 2 * std::int64_t{s_i};
    for (std::int64_t k = row_begin[static_cast<std::size_t>(i)];
         k < row_begin[static_cast<std::size_t>(i) + 1]; ++k) {
      h[column[static_cast<std::size_t>(k)]] -= twice * value[static_cast<std::size_t>(k)];
    }
  }
};

// J of a dense spin glass, INT8, row-major n x n (J_ii = 0: its terms change nothing).
struct DenseCouplings {
  std::int64_t n = 0;
  std::vector<std::int8_t> j;

  std::int64_t field(std::int64_t i, const Spin* s) const noexcept {
    const std::int8_t* const row = j.data() + i * n;
    std::int64_t sum = 0;
    for (std::int64_t c = 0; c < n; ++c) {
      sum += std::int64_t{row[c]} * s[c];
    }
    return sum;
  }

  // J is symmetric, so row i holds the J_ji.
  void flip(std::int64_t i, Spin s_i, std::int64_t* h) const noexcept {
    const std::int8_t* const row = j.data() + i * n;
    const std::int64_t twice = 2 * std::int64_t{s_i};
    for (std::int64_t c = 0; c < n; ++c) {
      h[c] -= twice * row[c];
    }
  }
};

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

// The acceptance words of one replica in proposal order p = 0, 1, 2, ...: word p of the stream
// (seed, sa_acceptance, replica), that is sa_acceptance_word(seed, replica, p). One Philox block
// gives four consecutive words.
class AcceptanceWords {
 public:
  AcceptanceWords(std::uint64_t seed, std::uint32_t replica) noexcept
      : seed_(seed), replica_(replica) {}

  std::uint32_t next() noexcept {
    if ((proposal_ & 3u) == 0) {
      block_ = stream_block(seed_, RngDomain::sa_acceptance, replica_, proposal_ >> 2);
    }
    const std::array<std::uint32_t, 4> words{block_.x, block_.y, block_.z, block_.w};
    return words[static_cast<std::size_t>(proposal_++ & 3u)];
  }

 private:
  std::uint64_t seed_;
  std::uint32_t replica_;
  std::uint64_t proposal_ = 0;
  U32x4 block_{};
};

template <typename Couplings>
SaResult run(const Instance& inst, const Couplings& c, std::int64_t total, const SaParams& params) {
  const std::int64_t n = c.n;
  const auto un = static_cast<std::size_t>(n);
  const int replicas = params.replicas;
  const int sweeps = params.sweeps;

  SaResult result;
  result.n = n;
  result.replicas = replicas;
  result.sweeps = sweeps;
  result.first_replica = params.first_replica;
  result.beta0 = params.beta0;
  result.beta1 = params.beta1;
  result.total_weight = total;
  result.per_replica.resize(static_cast<std::size_t>(replicas));
  result.best_spins.resize(un * static_cast<std::size_t>(replicas));
  result.final_spins.resize(un * static_cast<std::size_t>(replicas));
  result.threads = 1;

  std::vector<std::int64_t> h(un);
  for (int r = 0; r < replicas; ++r) {
    const auto global = static_cast<std::uint32_t>(params.first_replica + r);  // < 2^32
    const std::size_t offset = static_cast<std::size_t>(r) * un;
    Spin* const s = result.final_spins.data() + offset;  // the state of the replica
    Spin* const best = result.best_spins.data() + offset;

    for (std::size_t i = 0; i < un; ++i) {
      s[i] = sa_initial_spin(params.seed, global, static_cast<std::uint32_t>(i));
    }
    std::int64_t sum = 0;
    for (std::size_t i = 0; i < un; ++i) {
      h[i] = c.field(static_cast<std::int64_t>(i), s);
      sum += s[i] * h[i];
    }
    std::int64_t energy = -sum / 2;  // the sum is even
    std::int64_t best_energy = energy;
    int best_sweep = 0;
    std::int64_t accepted = 0;
    std::copy(s, s + n, best);

    AcceptanceWords words(params.seed, global);
    for (int k = 0; k < sweeps; ++k) {
      const double beta = sa_beta(params.beta0, params.beta1, sweeps, k);
      for (std::size_t i = 0; i < un; ++i) {
        const std::uint32_t w = words.next();  // consumed by every proposal
        const std::int64_t delta = 2 * s[i] * h[i];
        if (delta <= 0 || w < sa_threshold(beta, delta)) {
          c.flip(static_cast<std::int64_t>(i), s[i], h.data());
          s[i] = static_cast<Spin>(-s[i]);
          energy += delta;
          ++accepted;
          if (energy < best_energy) {  // strict: ties keep the earlier state
            best_energy = energy;
            best_sweep = k + 1;
            std::copy(s, s + n, best);
          }
        }
      }
    }

    SaReplicaResult& out = result.per_replica[static_cast<std::size_t>(r)];
    out.best_energy = best_energy;
    out.best_cut = (total - best_energy) / 2;
    out.best_sweep = best_sweep;
    out.final_cut = (total - energy) / 2;
    out.accepted = accepted;
  }

  // Final check with the public functions: the best and the final spins of every replica.
  for (int r = 0; r < replicas; ++r) {
    const std::size_t offset = static_cast<std::size_t>(r) * un;
    const SaReplicaResult& out = result.per_replica[static_cast<std::size_t>(r)];
    const std::int64_t best_cut =
        cut_direct(inst, std::span<const Spin>(result.best_spins.data() + offset, un));
    const std::int64_t final_cut =
        cut_direct(inst, std::span<const Spin>(result.final_spins.data() + offset, un));
    if (best_cut != out.best_cut || final_cut != out.final_cut) {
      throw Error("run_sa_sequential: internal error: replica " + std::to_string(r) +
                  ": cut_direct gives best " + std::to_string(best_cut) + " and final " +
                  std::to_string(final_cut) + ", the incremental energy gives best " +
                  std::to_string(out.best_cut) + " and final " + std::to_string(out.final_cut));
    }
  }
  return result;
}

}  // namespace

SaResult run_sa_sequential(const Instance& inst, const SaParams& params) {
  check_params(params);
  if (const auto* dense = std::get_if<DenseSpinGlass>(&inst);
      dense != nullptr && dense->n > kMaxDenseSpins) {
    invalid("n = " + std::to_string(dense->n) +
            ": the SA reference stores the J of a DenseSpinGlass as dense INT8 and accepts "
            "n <= 32768");
  }
  const std::int64_t total = total_weight(inst);  // full check of the instance
  return std::visit([&](const auto& g) { return run(inst, make_couplings(g), total, params); },
                    inst);
}

}  // namespace sbmc
