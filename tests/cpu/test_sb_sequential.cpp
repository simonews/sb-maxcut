// Sequential reference C0 (sbmc/sb_sequential.hpp): a short trajectory bit for bit, the reference
// cases G1-G6, properties that hold in every preset, and the preconditions.
//
// The reference values were computed by cldArchitect with two independent implementations of
// the algorithm (NumPy in float32 and scalar Python rounding to float32 after every operation);
// the optima come from an exact enumeration with Gray code. "Strict" means
// SBMC_STRICT_FP_BUILD = 1 (no FMA contraction): there the floating-point states, the
// fingerprints of the final states and the integer results of G1-G6 are compared exactly. In
// every preset dt = 1.25f and a0 = 1; "auto" means c0 = 0.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>
#include <omp.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/initial_state.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sb.hpp>
#include <sbmc/sb_sequential.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif
#ifndef SBMC_STRICT_FP_BUILD
#error "SBMC_STRICT_FP_BUILD must be defined by CMake (1 in the strict presets, 0 otherwise)"
#endif

namespace {

using sbmc::DenseSpinGlass;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::InvalidArgument;
using sbmc::SbAlgorithm;
using sbmc::SbParams;
using sbmc::SbResult;
using sbmc::Spin;

constexpr bool kStrict = SBMC_STRICT_FP_BUILD != 0;

std::uint32_t bits(float value) {
  return std::bit_cast<std::uint32_t>(value);
}

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
  return text;
}

std::string hex64(std::uint64_t value) {
  char text[24];
  std::snprintf(text, sizeof(text), "0x%016llX", static_cast<unsigned long long>(value));
  return text;
}

Graph rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

SbParams params(SbAlgorithm algorithm, int replicas, int steps, float c0, std::uint64_t seed,
                int eval_every) {
  SbParams p;
  p.algorithm = algorithm;
  p.replicas = replicas;
  p.steps = steps;
  p.dt = 1.25f;
  p.a0 = 1.0f;
  p.c0 = c0;
  p.eval_every = eval_every;
  p.seed = seed;
  p.first_replica = 0;
  return p;
}

// FNV-1a 64 of the bytes of final_x (index r * n + i, 4 little-endian bytes per float) followed
// by those of final_y.
std::uint64_t fingerprint(const SbResult& result) {
  std::uint64_t h = 0xCBF29CE484222325ull;
  for (const std::vector<float>* state : {&result.final_x, &result.final_y}) {
    for (const float value : *state) {
      const std::uint32_t word = bits(value);
      for (int byte = 0; byte < 4; ++byte) {
        h ^= (word >> (8 * byte)) & 0xFFu;
        h *= 0x100000001B3ull;
      }
    }
  }
  return h;
}

std::span<const Spin> replica_spins(const SbResult& result, int r) {
  const auto n = static_cast<std::size_t>(result.n);
  return {result.best_spins.data() + static_cast<std::size_t>(r) * n, n};
}

std::vector<Spin> signs_of(std::span<const float> x) {
  std::vector<Spin> s(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    s[i] = sbmc::sb_sign(x[i]);
  }
  return s;
}

std::size_t bit_differences(const std::vector<float>& a, const std::vector<float>& b) {
  std::size_t count = a.size() == b.size() ? 0 : 1;
  for (std::size_t k = 0; k < a.size() && k < b.size(); ++k) {
    count += bits(a[k]) != bits(b[k]) ? 1 : 0;
  }
  return count;
}

bool same_replica_results(const SbResult& a, const SbResult& b) {
  if (a.per_replica.size() != b.per_replica.size()) {
    return false;
  }
  for (std::size_t r = 0; r < a.per_replica.size(); ++r) {
    const auto& x = a.per_replica[r];
    const auto& y = b.per_replica[r];
    if (x.best_energy != y.best_energy || x.best_cut != y.best_cut || x.best_step != y.best_step ||
        x.final_cut != y.final_cut) {
      return false;
    }
  }
  return a.best_spins == b.best_spins;
}

// Restores the OpenMP settings changed by the thread-count test, also when a check throws.
class OmpSettingsGuard {
 public:
  OmpSettingsGuard() : max_threads_(omp_get_max_threads()), dynamic_(omp_get_dynamic()) {}
  ~OmpSettingsGuard() {
    omp_set_dynamic(dynamic_);
    omp_set_num_threads(max_threads_);
  }
  OmpSettingsGuard(const OmpSettingsGuard&) = delete;
  OmpSettingsGuard& operator=(const OmpSettingsGuard&) = delete;

  [[nodiscard]] int initial_max_threads() const noexcept { return max_threads_; }
  [[nodiscard]] int initial_dynamic() const noexcept { return dynamic_; }

 private:
  int max_threads_;
  int dynamic_;
};

// The reference cases G1-G6 (R = 8).
struct ReferenceCase {
  std::string name;
  Instance instance;
  SbAlgorithm algorithm;
  float c0;  // 0: auto
  int steps;
  std::uint64_t seed;
  int eval_every;
  std::int64_t optimum;
  // Strict values.
  std::vector<int> best_step;
  std::vector<std::int64_t> final_cut;
  std::uint64_t fingerprint;
  std::uint32_t c0_bits;
  std::int64_t total_weight;
};

std::vector<ReferenceCase> reference_cases() {
  return {
      {.name = "G1",
       .instance = rudy("four_nodes.rudy"),
       .algorithm = SbAlgorithm::discrete,
       .c0 = 0.3f,
       .steps = 100,
       .seed = 2,
       .eval_every = 0,
       .optimum = 4,
       .best_step = {4, 3, 4, 7, 4, 3, 3, 7},
       .final_cut = {4, 4, 4, 4, 4, 4, 4, 4},
       .fingerprint = 0x01AA86FA0D1DC425ull,
       .c0_bits = 0x3E99999Au,
       .total_weight = 5},
      {.name = "G2",
       .instance = rudy("four_nodes.rudy"),
       .algorithm = SbAlgorithm::ballistic,
       .c0 = 0.3f,
       .steps = 100,
       .seed = 2,
       .eval_every = 10,
       .optimum = 4,
       .best_step = {70, 60, 10, 70, 70, 70, 70, 10},
       .final_cut = {4, 4, 4, 4, 4, 4, 4, 4},
       .fingerprint = 0xABD93D3593EDC425ull,
       .c0_bits = 0x3E99999Au,
       .total_weight = 5},
      {.name = "G3",
       .instance = rudy("torus_4x6_pm1.rudy"),
       .algorithm = SbAlgorithm::discrete,
       .c0 = 0.0f,
       .steps = 100,
       .seed = 3,
       .eval_every = 0,
       .optimum = 24,
       .best_step = {40, 35, 21, 22, 19, 17, 37, 28},
       .final_cut = {24, 24, 24, 24, 24, 24, 24, 24},
       .fingerprint = 0x263996E44423941Bull,
       .c0_bits = 0x3E7D1123u,
       .total_weight = 16},
      {.name = "G4",
       .instance = rudy("torus_4x6_unit.rudy"),
       .algorithm = SbAlgorithm::ballistic,
       .c0 = 0.0f,
       .steps = 100,
       .seed = 3,
       .eval_every = 0,
       .optimum = 48,
       .best_step = {100, 100, 100, 100, 100, 100, 100, 100},
       .final_cut = {48, 48, 48, 48, 48, 48, 48, 48},
       .fingerprint = 0x59888B4D1012BD25ull,
       .c0_bits = 0x3E89DD8Du,
       .total_weight = 48},
      {.name = "G5",
       .instance = DenseSpinGlass{20, 20},
       .algorithm = SbAlgorithm::discrete,
       .c0 = 0.0f,
       .steps = 200,
       .seed = 4,
       .eval_every = 0,
       .optimum = 35,
       .best_step = {53, 35, 77, 47, 90, 84, 115, 75},
       .final_cut = {35, 34, 34, 34, 35, 35, 35, 35},
       .fingerprint = 0x17BF891DFB82BBA5ull,
       .c0_bits = 0x3DE54A89u,
       .total_weight = 10},
      {.name = "G6",
       .instance = DenseSpinGlass{24, 24},
       .algorithm = SbAlgorithm::ballistic,
       .c0 = 0.0f,
       .steps = 200,
       .seed = 5,
       .eval_every = 1,
       .optimum = 47,
       .best_step = {64, 76, 30, 95, 35, 103, 64, 70},
       .final_cut = {47, 47, 47, 47, 47, 47, 47, 47},
       .fingerprint = 0xA021B0A5174B4D25ull,
       .c0_bits = 0x3DD12914u,
       .total_weight = 10},
  };
}

SbResult run_reference(const ReferenceCase& c, int trace_steps) {
  return sbmc::run_sb_sequential(
      c.instance, params(c.algorithm, 8, c.steps, c.c0, c.seed, c.eval_every), trace_steps);
}

}  // namespace

TEST_CASE("C0: short trajectory on four_nodes.rudy, bit for bit (strict)") {
  const Instance inst = rudy("four_nodes.rudy");
  const std::vector<std::uint32_t> x0_bits = {0xBD802C93u, 0xBC8416FAu, 0xBD392CC3u, 0xBB16C39Au};
  const std::vector<float> x0 = sbmc::make_initial_state(4, 1, 1).x0;
  for (std::size_t i = 0; i < 4; ++i) {
    CHECK(bits(x0[i]) == x0_bits[i]);
  }

  struct Trajectory {
    SbAlgorithm algorithm;
    std::vector<std::vector<std::uint32_t>> x;  // after steps 1, 2, 3
    std::vector<std::vector<std::uint32_t>> y;
    std::int64_t best_cut;
    int best_step;
    std::int64_t final_cut;
  };
  const std::vector<Trajectory> trajectories = {
      {SbAlgorithm::discrete,
       {{0x3F800000u, 0x3F702C08u, 0x3F800000u, 0x3F700648u},
        {0xBF6D5558u, 0x3EEEC3B0u, 0xBF6D5558u, 0x3EE7401Cu},
        {0xBF800000u, 0x3F6E97A8u, 0xBF800000u, 0x3F6739D4u}},
       {{0x00000000u, 0x3F43709Au, 0x00000000u, 0x3F407DA3u},
        {0xBFC55556u, 0xBEC14380u, 0xBFC55556u, 0xBEC709F6u},
        {0x00000000u, 0x3EBEBC80u, 0x00000000u, 0x3EB8F60Au}},
       4,
       2,
       4},
      {SbAlgorithm::ballistic,
       {{0x3D04DB7Cu, 0x3D51B73Fu, 0x3D234119u, 0x3D4F5B47u},
        {0x3D34D69Au, 0x3D6D727Au, 0x3D28E029u, 0x3D3155D8u},
        {0xBC331508u, 0x3CCA8010u, 0xBCD1668Eu, 0xBB84EA20u}},
       {{0x3D9BAEA7u, 0x3D5C9BCAu, 0x3D8B5F25u, 0x3D2D6C67u},
        {0x3C1989F8u, 0x3BB17B10u, 0x3A8FE800u, 0xBBC022C8u},
        {0xBD347CB0u, 0xBCD9EA50u, 0xBD5ADC5Au, 0xBD1B28E3u}},
       2,
       3,
       2},
  };
  for (const Trajectory& t : trajectories) {
    const bool discrete = t.algorithm == SbAlgorithm::discrete;
    INFO(std::string(discrete ? "dSB" : "bSB"));
    const SbResult result = sbmc::run_sb_sequential(inst, params(t.algorithm, 1, 3, 0.3f, 1, 0), 3);
    REQUIRE(result.trace_x.size() == 12u);
    REQUIRE(result.trace_y.size() == 12u);
    if (!kStrict) {
      MESSAGE("production preset: the bit-for-bit trajectory is checked only in the strict ones");
      continue;
    }
    for (std::size_t k = 0; k < 3; ++k) {
      for (std::size_t i = 0; i < 4; ++i) {
        const float x = result.trace_x[k * 4 + i];
        const float y = result.trace_y[k * 4 + i];
        INFO("step " << k + 1 << ", spin " << i << ": x expected " << hex32(t.x[k][i]) << " got "
                     << hex32(bits(x)) << ", y expected " << hex32(t.y[k][i]) << " got "
                     << hex32(bits(y)));
        CHECK(bits(x) == t.x[k][i]);
        CHECK(bits(y) == t.y[k][i]);
      }
    }
    REQUIRE(result.per_replica.size() == 1u);
    CHECK(result.per_replica[0].best_cut == t.best_cut);
    CHECK(result.per_replica[0].best_step == t.best_step);
    CHECK(result.per_replica[0].final_cut == t.final_cut);
  }
}

TEST_CASE("C0: reference cases G1-G6") {
  for (const ReferenceCase& c : reference_cases()) {
    INFO(c.name);
    const SbResult result = run_reference(c, c.steps);
    const auto n = static_cast<std::size_t>(sbmc::num_spins(c.instance));
    REQUIRE(result.per_replica.size() == 8u);
    REQUIRE(result.best_spins.size() == 8 * n);
    REQUIRE(result.final_x.size() == 8 * n);
    REQUIRE(result.final_y.size() == 8 * n);
    REQUIRE(result.trace_steps == c.steps);
    CHECK(result.n == static_cast<std::int64_t>(n));
    CHECK(result.replicas == 8);
    CHECK(result.steps == c.steps);
    CHECK(result.first_replica == 0);
    CHECK(result.total_weight == c.total_weight);

    // Strict values: c0, fingerprint of the final state and the integer results, exactly.
    if (kStrict) {
      INFO("fingerprint expected " << hex64(c.fingerprint) << ", got " << hex64(fingerprint(result))
                                   << "; c0 expected " << hex32(c.c0_bits) << ", got "
                                   << hex32(bits(result.c0)));
      CHECK(bits(result.c0) == c.c0_bits);
      CHECK(fingerprint(result) == c.fingerprint);
      for (int r = 0; r < 8; ++r) {
        const auto& rep = result.per_replica[static_cast<std::size_t>(r)];
        const auto k = static_cast<std::size_t>(r);
        INFO("replica " << r << ": best_cut " << rep.best_cut << ", best_step " << rep.best_step
                        << ", final_cut " << rep.final_cut);
        CHECK(rep.best_cut == c.optimum);
        CHECK(rep.best_step == c.best_step[k]);
        CHECK(rep.final_cut == c.final_cut[k]);
      }
    } else {
      const double reference = std::bit_cast<float>(c.c0_bits);
      CHECK(std::abs(static_cast<double>(result.c0) - reference) <= 1e-6 * reference);
    }

    // Properties of every preset.
    bool optimum_reached = false;
    for (int r = 0; r < 8; ++r) {
      const auto& rep = result.per_replica[static_cast<std::size_t>(r)];
      INFO("replica " << r);
      const std::span<const Spin> best = replica_spins(result, r);
      const std::span<const float> final_x(result.final_x.data() + static_cast<std::size_t>(r) * n,
                                           n);
      optimum_reached = optimum_reached || rep.best_cut == c.optimum;
      CHECK(rep.best_cut <= c.optimum);
      CHECK(rep.best_cut == sbmc::cut_direct(c.instance, best));
      CHECK(rep.best_energy == sbmc::ising_energy(c.instance, best));
      CHECK(2 * rep.best_cut == result.total_weight - rep.best_energy);
      CHECK(rep.best_cut >= rep.final_cut);
      CHECK(rep.best_step >= 0);
      CHECK(rep.best_step <= c.steps);
      CHECK(rep.final_cut == sbmc::cut_direct(c.instance, signs_of(final_x)));
      if (c.algorithm == SbAlgorithm::ballistic && c.eval_every == 0) {
        CHECK(rep.best_step == c.steps);
      }
    }
    CHECK(optimum_reached);
    // With trace_steps = M the last recorded state is the final state.
    const std::size_t last = static_cast<std::size_t>(c.steps - 1) * 8 * n;
    const std::vector<float> last_x(result.trace_x.begin() + static_cast<std::ptrdiff_t>(last),
                                    result.trace_x.end());
    const std::vector<float> last_y(result.trace_y.begin() + static_cast<std::ptrdiff_t>(last),
                                    result.trace_y.end());
    CHECK(bit_differences(last_x, result.final_x) == 0u);
    CHECK(bit_differences(last_y, result.final_y) == 0u);
  }
}

// Every recorded step of G3-G6 (compressed rows and dense J, dSB and bSB) recomputed in the test
// from the previous recorded state with the operations of F04-C1, section 5.3: dSB field
// (float)(sum_j J_ij s_j) in int64; bSB field (float)(acc), acc = +0.0 in double plus
// (double)J_ij * (double)x_j for every j in increasing order, zero terms included (C0 skips
// them: the bits must be the same); then sb_update. Bit for bit in the strict presets: with FMA
// contraction the test and the library may contract differently.
TEST_CASE("C0: every step recomputed from the specified operations, bit for bit (strict)") {
  if (!kStrict) {
    MESSAGE("production preset: checked only in the strict ones");
    return;
  }
  const std::vector<ReferenceCase> cases = reference_cases();
  for (std::size_t c = 2; c < cases.size(); ++c) {  // G3-G6
    const ReferenceCase& rc = cases[c];
    INFO(rc.name);
    const SbResult result = run_reference(rc, rc.steps);
    const auto n = static_cast<std::size_t>(result.n);
    const std::size_t state = 8 * n;
    std::vector<std::int64_t> j(n * n, 0);
    if (const auto* g = std::get_if<Graph>(&rc.instance)) {
      for (const auto& e : g->edges) {
        j[static_cast<std::size_t>(e.u) * n + static_cast<std::size_t>(e.v)] = -e.w;
        j[static_cast<std::size_t>(e.v) * n + static_cast<std::size_t>(e.u)] = -e.w;
      }
    } else {
      const auto& dense = std::get<DenseSpinGlass>(rc.instance);
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t col = 0; col < n; ++col) {
          j[r * n + col] =
              sbmc::coupling(dense, static_cast<std::int64_t>(r), static_cast<std::int64_t>(col));
        }
      }
    }
    const std::vector<float> p = sbmc::pressure_schedule(1.0f, rc.steps);
    const float dt_a0 = 1.25f * 1.0f;
    const std::vector<float> x0 = sbmc::make_initial_state(result.n, 8, rc.seed).x0;
    const std::vector<float> y0(state, 0.0f);
    const bool discrete = rc.algorithm == SbAlgorithm::discrete;
    std::size_t different = 0;
    for (std::size_t k = 0; k < static_cast<std::size_t>(rc.steps); ++k) {
      const float* const x_prev = k == 0 ? x0.data() : result.trace_x.data() + (k - 1) * state;
      const float* const y_prev = k == 0 ? y0.data() : result.trace_y.data() + (k - 1) * state;
      for (std::size_t r = 0; r < 8; ++r) {
        const float* const xr = x_prev + r * n;
        for (std::size_t i = 0; i < n; ++i) {
          float field = 0.0f;
          if (discrete) {
            std::int64_t f = 0;
            for (std::size_t col = 0; col < n; ++col) {
              f += j[i * n + col] * sbmc::sb_sign(xr[col]);
            }
            field = static_cast<float>(f);
          } else {
            double acc = 0.0;
            for (std::size_t col = 0; col < n; ++col) {
              acc += static_cast<double>(j[i * n + col]) * static_cast<double>(xr[col]);
            }
            field = static_cast<float>(acc);
          }
          float x = xr[i];
          float y = y_prev[r * n + i];
          sbmc::sb_update(x, y, field, p[k], 1.25f, result.c0, dt_a0);
          const std::size_t at = k * state + r * n + i;
          different += bits(x) != bits(result.trace_x[at]) ? 1 : 0;
          different += bits(y) != bits(result.trace_y[at]) ? 1 : 0;
        }
      }
    }
    CHECK(different == 0u);
  }
}

TEST_CASE("C0: splitting the replicas with first_replica gives the same replicas") {
  const ReferenceCase g5 = reference_cases()[4];
  REQUIRE(g5.name == "G5");
  const SbResult all = run_reference(g5, 0);
  SbParams first = params(g5.algorithm, 3, g5.steps, g5.c0, g5.seed, g5.eval_every);
  SbParams second = params(g5.algorithm, 5, g5.steps, g5.c0, g5.seed, g5.eval_every);
  second.first_replica = 3;
  const SbResult a = sbmc::run_sb_sequential(g5.instance, first);
  const SbResult b = sbmc::run_sb_sequential(g5.instance, second);
  CHECK(b.first_replica == 3);
  SbResult joined = a;
  joined.per_replica.insert(joined.per_replica.end(), b.per_replica.begin(), b.per_replica.end());
  joined.best_spins.insert(joined.best_spins.end(), b.best_spins.begin(), b.best_spins.end());
  joined.final_x.insert(joined.final_x.end(), b.final_x.begin(), b.final_x.end());
  joined.final_y.insert(joined.final_y.end(), b.final_y.begin(), b.final_y.end());
  CHECK(same_replica_results(joined, all));
  CHECK(bit_differences(joined.final_x, all.final_x) == 0u);
  CHECK(bit_differences(joined.final_y, all.final_y) == 0u);
}

TEST_CASE("C0: the result does not depend on the number of OpenMP threads") {
  const ReferenceCase g5 = reference_cases()[4];
  const OmpSettingsGuard guard;
  omp_set_dynamic(0);
  const int initial_threads = guard.initial_max_threads();
  std::vector<SbResult> results;
  for (const int threads : {1, 3, initial_threads}) {
    omp_set_num_threads(threads);
    REQUIRE(omp_get_max_threads() == threads);
    results.push_back(run_reference(g5, 0));
  }
  for (std::size_t k = 1; k < results.size(); ++k) {
    INFO("configuration " << k << " against the single-thread one");
    CHECK(same_replica_results(results[k], results[0]));
    CHECK(bit_differences(results[k].final_x, results[0].final_x) == 0u);
    CHECK(bit_differences(results[k].final_y, results[0].final_y) == 0u);
    CHECK(bits(results[k].c0) == bits(results[0].c0));
  }
  omp_set_dynamic(guard.initial_dynamic());
  omp_set_num_threads(initial_threads);
  CHECK(omp_get_max_threads() == initial_threads);
}

TEST_CASE("C0: the c0 actually used") {
  const Instance inst = rudy("torus_4x6_pm1.rudy");
  const float automatic = sbmc::auto_c0(inst);
  const SbResult explicit_c0 =
      sbmc::run_sb_sequential(inst, params(SbAlgorithm::discrete, 2, 10, 0.4f, 1, 0));
  CHECK(bits(explicit_c0.c0) == bits(0.4f));
  for (const float zero : {0.0f, -0.0f}) {
    const SbResult result =
        sbmc::run_sb_sequential(inst, params(SbAlgorithm::ballistic, 2, 10, zero, 1, 0));
    INFO("c0 = " << std::string(std::signbit(zero) ? "-0.0f" : "+0.0f"));
    CHECK(bits(result.c0) == bits(automatic));
  }
}

TEST_CASE("C0: instances with a single spin") {
  const std::vector<Instance> instances = {Graph{1, {}}, DenseSpinGlass{1, 3}};
  for (const Instance& inst : instances) {
    for (const SbAlgorithm algorithm : {SbAlgorithm::discrete, SbAlgorithm::ballistic}) {
      const bool discrete = algorithm == SbAlgorithm::discrete;
      INFO(std::string(inst.index() == 0 ? "Graph" : "DenseSpinGlass")
           << ", " << std::string(discrete ? "dSB" : "bSB"));
      const SbResult result = sbmc::run_sb_sequential(inst, params(algorithm, 2, 10, 0.5f, 9, 0));
      REQUIRE(result.per_replica.size() == 2u);
      for (const auto& rep : result.per_replica) {
        CHECK(rep.best_energy == 0);
        CHECK(rep.best_cut == 0);
        CHECK(rep.final_cut == 0);
        CHECK(rep.best_step == (discrete ? 0 : 10));
      }
    }
  }
}

TEST_CASE("C0: trace_steps = 0 records nothing") {
  const SbResult result = sbmc::run_sb_sequential(
      rudy("four_nodes.rudy"), params(SbAlgorithm::discrete, 3, 20, 0.3f, 1, 0), 0);
  CHECK(result.trace_steps == 0);
  CHECK(result.trace_x.empty());
  CHECK(result.trace_y.empty());
  CHECK(result.final_x.size() == 12u);
}

TEST_CASE("C0: invalid arguments throw InvalidArgument") {
  const Instance four = rudy("four_nodes.rudy");
  const SbParams good = params(SbAlgorithm::discrete, 2, 10, 0.3f, 1, 0);
  CHECK_NOTHROW(sbmc::run_sb_sequential(four, good, 10));

  const float nan = std::numeric_limits<float>::quiet_NaN();
  const float inf = std::numeric_limits<float>::infinity();
  struct Case {
    std::string name;
    SbParams p;
    int trace_steps;
  };
  std::vector<Case> cases;
  const auto with = [&](const std::string& name, auto change, int trace_steps = 0) {
    SbParams p = good;
    change(p);
    cases.push_back({name, p, trace_steps});
  };
  with("replicas 0", [](SbParams& p) { p.replicas = 0; });
  with("steps 0", [](SbParams& p) { p.steps = 0; });
  with("dt 0", [](SbParams& p) { p.dt = 0.0f; });
  with("dt -1", [](SbParams& p) { p.dt = -1.0f; });
  with("dt NaN", [&](SbParams& p) { p.dt = nan; });
  with("dt +inf", [&](SbParams& p) { p.dt = inf; });
  with("a0 0", [](SbParams& p) { p.a0 = 0.0f; });
  with("c0 -1", [](SbParams& p) { p.c0 = -1.0f; });
  with("c0 NaN", [&](SbParams& p) { p.c0 = nan; });
  with("c0 +inf", [&](SbParams& p) { p.c0 = inf; });
  with("eval_every -1", [](SbParams& p) { p.eval_every = -1; });
  with("first_replica -1", [](SbParams& p) { p.first_replica = -1; });
  with("first_replica 2^32 - 2 with 3 replicas", [](SbParams& p) {
    p.first_replica = (std::int64_t{1} << 32) - 2;
    p.replicas = 3;
  });
  with("trace_steps -1", [](SbParams&) {}, -1);
  with("trace_steps M + 1", [](SbParams&) {}, 11);
  with("algorithm 2", [](SbParams& p) { p.algorithm = static_cast<SbAlgorithm>(2); });
  for (const Case& c : cases) {
    INFO(c.name);
    CHECK_THROWS_AS(sbmc::run_sb_sequential(four, c.p, c.trace_steps), InvalidArgument);
  }

  CHECK_THROWS_AS(sbmc::run_sb_sequential(Graph{4, {{0, 2, 1}, {0, 1, 1}}}, good),
                  InvalidArgument);  // edges out of order
  CHECK_THROWS_AS(sbmc::run_sb_sequential(DenseSpinGlass{32769, 1}, good), InvalidArgument);
  SbParams automatic = good;
  automatic.c0 = 0.0f;
  CHECK_THROWS_AS(sbmc::run_sb_sequential(Graph{2, {{0, 1, 1}}}, automatic), InvalidArgument);
  // The last valid global replica index.
  SbParams last = good;
  last.first_replica = (std::int64_t{1} << 32) - 2;
  CHECK_NOTHROW(sbmc::run_sb_sequential(four, last));
}
