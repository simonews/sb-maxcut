// Sequential reference of simulated annealing (sbmc/sa_sequential.hpp, sbmc/sa.hpp): the
// reference cases SA-1 to SA-7, properties of the results, the split of the replicas, the helper
// functions, instances with one spin, the acceptance rule at its boundary and the
// preconditions.
//
// The reference values were computed by cldArchitect with two emulators independent of the code
// of the project (Python with a NumPy Philox, math.exp and math.pow; C++ with glibc exp and pow),
// which agree in every case. SA has no floating-point sum (only sa_beta and sa_threshold, without
// FMA contraction), so the values are compared exactly in every preset, strict or not.
// Fingerprint of a result: FNV-1a 64 over the bytes (uint8_t) of best_spins, then of final_spins,
// in the order of the vectors (+1 is 0x01, -1 is 0xFF).
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sa.hpp>
#include <sbmc/sa_sequential.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif
#ifndef SBMC_STRICT_FP_BUILD
#error "SBMC_STRICT_FP_BUILD must be defined by CMake (1 in the strict presets, 0 otherwise)"
#endif
#ifndef SBMC_DATA_DIR
#error "SBMC_DATA_DIR must be defined by CMake (absolute path of data/)"
#endif

namespace {

using sbmc::DenseSpinGlass;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::InvalidArgument;
using sbmc::SaParams;
using sbmc::SaResult;
using sbmc::Spin;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

Graph rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

Graph g1_of_gset() {
  return sbmc::read_rudy(std::string(SBMC_DATA_DIR) + "/gset/G1");
}

SaParams params(int replicas, int sweeps, double beta0, double beta1, std::uint64_t seed,
                std::int64_t first_replica = 0) {
  SaParams p;
  p.replicas = replicas;
  p.sweeps = sweeps;
  p.beta0 = beta0;
  p.beta1 = beta1;
  p.seed = seed;
  p.first_replica = first_replica;
  return p;
}

std::uint64_t fingerprint(const SaResult& result) {
  std::uint64_t hash = 0xCBF29CE484222325ULL;
  for (const std::vector<Spin>* spins : {&result.best_spins, &result.final_spins}) {
    for (const Spin s : *spins) {
      hash ^= static_cast<std::uint8_t>(s);
      hash *= 0x100000001B3ULL;
    }
  }
  return hash;
}

std::span<const Spin> replica_spins(const std::vector<Spin>& spins, std::int64_t n, int r) {
  return {spins.data() + static_cast<std::size_t>(r) * static_cast<std::size_t>(n),
          static_cast<std::size_t>(n)};
}

struct ReferenceCase {
  const char* name;
  Instance instance;
  SaParams params;
  std::int64_t total_weight;
  std::vector<std::int64_t> best_cut;
  std::vector<int> best_sweep;
  std::vector<std::int64_t> final_cut;
  std::vector<std::int64_t> accepted;
  std::uint64_t fingerprint;
};

// Section 5.8 of F06-C1.
std::vector<ReferenceCase> reference_cases() {
  std::vector<ReferenceCase> cases;
  cases.push_back({"SA-1 four_nodes.rudy",
                   rudy("four_nodes.rudy"),
                   params(8, 10, 0.1, 3.0, 2),
                   5,
                   {4, 4, 4, 4, 4, 4, 4, 4},
                   {6, 0, 2, 1, 0, 3, 0, 1},
                   {4, 4, 4, 4, 4, 4, 4, 4},
                   {24, 20, 17, 15, 18, 24, 20, 22},
                   0x981cc2c5c4d485e5ULL});
  cases.push_back({"SA-2 torus_4x6_pm1.rudy",
                   rudy("torus_4x6_pm1.rudy"),
                   params(8, 50, 0.1, 3.0, 3),
                   16,
                   {24, 24, 24, 24, 24, 24, 24, 24},
                   {31, 23, 30, 20, 21, 25, 26, 25},
                   {24, 24, 24, 24, 24, 24, 24, 24},
                   {520, 489, 505, 514, 487, 473, 471, 519},
                   0xbc8f2be192279e69ULL});
  cases.push_back({"SA-3 DenseSpinGlass{20, 20}",
                   DenseSpinGlass{20, 20},
                   params(8, 100, 0.1, 3.0, 4),
                   10,
                   {35, 35, 35, 35, 35, 35, 35, 35},
                   {18, 38, 27, 33, 25, 18, 20, 38},
                   {35, 35, 35, 35, 34, 35, 34, 35},
                   {369, 361, 357, 355, 399, 387, 378, 387},
                   0xebb72b8d3bacd7b1ULL});
  cases.push_back({"SA-4 weighted_isolated_13.rudy",
                   rudy("weighted_isolated_13.rudy"),
                   params(8, 20, 1e-10, 1e-9, 5),
                   2147483669,
                   {6442452014, 6442452013, 6442452014, 6442452014, 6442452014, 6442452014,
                    6442451974, 6442452013},
                   {3, 9, 15, 15, 7, 7, 15, 2},
                   {6442451973, 6442450953, 4294967294, 6442450975, 6442451973, 6442451904,
                    6442450965, 6442451916},
                   {152, 167, 178, 167, 166, 172, 175, 160},
                   0xc675acc36d13b2c7ULL});
  cases.push_back({"SA-5 data/gset/G1",
                   g1_of_gset(),
                   params(4, 20, 0.1, 3.0, 1),
                   19176,
                   {11532, 11465, 11498, 11517},
                   {18, 18, 20, 18},
                   {11532, 11465, 11498, 11517},
                   {2285, 2348, 2373, 2422},
                   0xeea4b13999139ab9ULL});
  cases.push_back({"SA-6 four_nodes.rudy, one sweep",
                   rudy("four_nodes.rudy"),
                   params(4, 1, 0.5, 0.5, 7),
                   5,
                   {4, 3, 3, 3},
                   {1, 0, 0, 0},
                   {3, 2, 3, 3},
                   {4, 2, 2, 2},
                   0x245c54e6779d5ff9ULL});
  cases.push_back({"SA-7 sparse_24_unit.rudy, first_replica 5",
                   rudy("sparse_24_unit.rudy"),
                   params(3, 30, 0.2, 2.0, 9, 5),
                   48,
                   {40, 40, 39},
                   {21, 23, 14},
                   {40, 40, 36},
                   {226, 228, 245},
                   0x231789d5863d723dULL});
  return cases;
}

template <typename Function>
std::string message_of(Function&& function) {
  try {
    function();
  } catch (const InvalidArgument& e) {
    return e.what();
  }
  return "(no exception)";
}

}  // namespace

TEST_CASE("SA: reference cases SA-1 to SA-7") {
  for (const ReferenceCase& c : reference_cases()) {
    INFO(c.name);
    const SaResult result = sbmc::run_sa_sequential(c.instance, c.params);
    const auto replicas = static_cast<std::size_t>(c.params.replicas);
    CHECK(result.n == sbmc::num_spins(c.instance));
    CHECK(result.replicas == c.params.replicas);
    CHECK(result.sweeps == c.params.sweeps);
    CHECK(result.first_replica == c.params.first_replica);
    CHECK(std::bit_cast<std::uint64_t>(result.beta0) ==
          std::bit_cast<std::uint64_t>(c.params.beta0));
    CHECK(std::bit_cast<std::uint64_t>(result.beta1) ==
          std::bit_cast<std::uint64_t>(c.params.beta1));
    CHECK(result.total_weight == c.total_weight);
    CHECK(result.threads == 1);
    REQUIRE(result.per_replica.size() == replicas);
    REQUIRE(result.best_spins.size() == replicas * static_cast<std::size_t>(result.n));
    REQUIRE(result.final_spins.size() == replicas * static_cast<std::size_t>(result.n));
    for (std::size_t r = 0; r < replicas; ++r) {
      INFO("replica " << r);
      const sbmc::SaReplicaResult& out = result.per_replica[r];
      CHECK(out.best_cut == c.best_cut[r]);
      CHECK(out.best_energy == c.total_weight - 2 * c.best_cut[r]);
      CHECK(out.best_sweep == c.best_sweep[r]);
      CHECK(out.final_cut == c.final_cut[r]);
      CHECK(out.accepted == c.accepted[r]);
    }
    CHECK(fingerprint(result) == c.fingerprint);
  }
}

TEST_CASE("SA: properties of the reference cases") {
  for (const ReferenceCase& c : reference_cases()) {
    INFO(c.name);
    const SaResult result = sbmc::run_sa_sequential(c.instance, c.params);
    for (int r = 0; r < c.params.replicas; ++r) {
      INFO("replica " << r);
      const sbmc::SaReplicaResult& out = result.per_replica[static_cast<std::size_t>(r)];
      CHECK(sbmc::cut_direct(c.instance, replica_spins(result.best_spins, result.n, r)) ==
            out.best_cut);
      CHECK(sbmc::cut_direct(c.instance, replica_spins(result.final_spins, result.n, r)) ==
            out.final_cut);
      CHECK(out.best_cut >= out.final_cut);
      CHECK(out.best_sweep >= 0);
      CHECK(out.best_sweep <= c.params.sweeps);
    }
  }
}

TEST_CASE("SA: splitting the replicas with first_replica gives the same replicas") {
  const Graph torus = rudy("torus_4x6_pm1.rudy");
  const SaResult whole = sbmc::run_sa_sequential(torus, params(8, 50, 0.1, 3.0, 3, 0));
  const SaResult head = sbmc::run_sa_sequential(torus, params(3, 50, 0.1, 3.0, 3, 0));
  const SaResult tail = sbmc::run_sa_sequential(torus, params(5, 50, 0.1, 3.0, 3, 3));
  std::vector<sbmc::SaReplicaResult> per_replica = head.per_replica;
  per_replica.insert(per_replica.end(), tail.per_replica.begin(), tail.per_replica.end());
  REQUIRE(per_replica.size() == whole.per_replica.size());
  for (std::size_t r = 0; r < per_replica.size(); ++r) {
    INFO("replica " << r);
    CHECK(per_replica[r].best_energy == whole.per_replica[r].best_energy);
    CHECK(per_replica[r].best_cut == whole.per_replica[r].best_cut);
    CHECK(per_replica[r].best_sweep == whole.per_replica[r].best_sweep);
    CHECK(per_replica[r].final_cut == whole.per_replica[r].final_cut);
    CHECK(per_replica[r].accepted == whole.per_replica[r].accepted);
  }
  std::vector<Spin> best = head.best_spins;
  best.insert(best.end(), tail.best_spins.begin(), tail.best_spins.end());
  std::vector<Spin> final_spins = head.final_spins;
  final_spins.insert(final_spins.end(), tail.final_spins.begin(), tail.final_spins.end());
  CHECK(best == whole.best_spins);
  CHECK(final_spins == whole.final_spins);
  CHECK(tail.first_replica == 3);
}

TEST_CASE("SA: helper functions") {
  const std::vector<std::uint64_t> beta_bits = {0x3fb999999999999aULL, 0x3fcdf4d99fe3603dULL,
                                                0x3fe186f174f88473ULL, 0x3ff48280b336ec5cULL,
                                                0x4008000000000000ULL};
  for (int k = 0; k < 5; ++k) {
    INFO("sa_beta(0.1, 3.0, 5, " << k << ")");
    CHECK(std::bit_cast<std::uint64_t>(sbmc::sa_beta(0.1, 3.0, 5, k)) ==
          beta_bits[static_cast<std::size_t>(k)]);
  }
  CHECK(sbmc::sa_beta(0.25, 0.75, 1, 0) == 0.25);

  CHECK(sbmc::sa_threshold(0.5, 2) == 1580030168U);
  CHECK(sbmc::sa_threshold(0.1, 4) == 2879002675U);
  CHECK(sbmc::sa_threshold(3.0, 1) == 213833830U);
  CHECK(sbmc::sa_threshold(2.0, 1) == 581260615U);
  CHECK(sbmc::sa_threshold(1e-300, 1) == 4294967296ULL);
  CHECK(sbmc::sa_threshold(3.0, 1000) == 0U);
  CHECK(sbmc::sa_threshold(1e-10, 4294967336) == 2795321725U);

  const std::vector<Spin> spins = {1, 1, -1, -1, -1, -1, -1, 1};
  for (std::uint32_t i = 0; i < 8; ++i) {
    INFO("sa_initial_spin(2, 0, " << i << ")");
    CHECK(sbmc::sa_initial_spin(2, 0, i) == spins[i]);
  }
  static_assert(sbmc::sa_initial_spin(2, 0, 0) == 1);  // constexpr

  const std::vector<std::uint32_t> words = {0xd0ac737eU, 0x3f53460eU, 0xa6818ea7U,
                                            0x0fffd16aU, 0xa5f2a38fU, 0x6c2482aeU};
  for (std::uint64_t p = 0; p < 6; ++p) {
    INFO("sa_acceptance_word(2, 0, " << p << ")");
    CHECK(sbmc::sa_acceptance_word(2, 0, p) == words[p]);
  }
  CHECK(sbmc::sa_acceptance_word(2, 7, 43) == 0xafb0de1cU);
  CHECK(sbmc::sa_acceptance_word(0xFFFFFFFF00000001ULL, 4294967295U,
                                 (std::uint64_t{1} << 40) + 1) == 0x878f3239U);
  static_assert(sbmc::sa_acceptance_word(2, 0, 0) == 0xd0ac737eU);  // constexpr
}

TEST_CASE("SA: instances with a single spin") {
  // delta_e = 0 for every proposal: every proposal is accepted, the best state stays the initial
  // one (strict improvement).
  for (const Instance& inst : {Instance{Graph{1, {}}}, Instance{DenseSpinGlass{1, 3}}}) {
    const SaResult result = sbmc::run_sa_sequential(inst, params(2, 5, 0.5, 1.0, 11));
    REQUIRE(result.per_replica.size() == 2);
    CHECK(result.total_weight == 0);
    for (const sbmc::SaReplicaResult& out : result.per_replica) {
      CHECK(out.best_energy == 0);
      CHECK(out.best_cut == 0);
      CHECK(out.final_cut == 0);
      CHECK(out.best_sweep == 0);
      CHECK(out.accepted == 5);
    }
  }
}

// The acceptance rule at its boundary (added in F06-R1, to close the mutation "w <= threshold"):
// on the graph 0 - 1 with weight 1, with a seed whose two initial spins differ (cut 1, the
// optimum), beta is chosen so that sa_threshold(beta, 2) equals the word of proposal 0. Proposal
// 0 (uphill, delta_e = 2) must then be rejected; proposal 1 (uphill as well) is accepted only
// when its word is below the threshold. With "<=" proposal 0 would be accepted, and proposal 1
// would follow downhill: two accepted moves.
TEST_CASE("SA: an uphill move needs a word strictly below the threshold") {
  const Graph edge{2, {{0, 1, 1}}};
  std::uint64_t seed = 1;
  while (sbmc::sa_initial_spin(seed, 0, 0) == sbmc::sa_initial_spin(seed, 0, 1)) {
    ++seed;
  }
  const std::uint32_t w0 = sbmc::sa_acceptance_word(seed, 0, 0);
  const std::uint32_t w1 = sbmc::sa_acceptance_word(seed, 0, 1);
  const double beta = -std::log((static_cast<double>(w0) + 0.5) / 4294967296.0) / 2.0;
  INFO("seed " << seed << ", words " << w0 << " and " << w1 << ", beta " << beta);
  REQUIRE(sbmc::sa_threshold(beta, 2) == w0);

  const SaResult result = sbmc::run_sa_sequential(edge, params(1, 1, beta, beta, seed));
  const bool second_accepted = w1 < w0;
  CHECK(result.per_replica[0].accepted == (second_accepted ? 1 : 0));
  CHECK(result.per_replica[0].best_cut == 1);
  CHECK(result.per_replica[0].best_sweep == 0);
  CHECK(result.per_replica[0].final_cut == (second_accepted ? 0 : 1));
}

TEST_CASE("SA: invalid arguments throw InvalidArgument") {
  const Graph four = rudy("four_nodes.rudy");
  const SaParams good = params(2, 3, 0.1, 3.0, 1);
  const auto with = [&good](auto change) {
    SaParams p = good;
    change(p);
    return p;
  };
  const std::vector<SaParams> invalid = {
      with([](SaParams& p) { p.replicas = 0; }),
      with([](SaParams& p) { p.sweeps = 0; }),
      with([](SaParams& p) { p.beta0 = 0.0; }),
      with([](SaParams& p) { p.beta0 = -1.0; }),
      with([](SaParams& p) { p.beta0 = kNaN; }),
      with([](SaParams& p) { p.beta0 = kInf; }),
      with([](SaParams& p) { p.beta1 = kNaN; }),
      with([](SaParams& p) { p.beta1 = 0.05; }),  // beta1 < beta0
      with([](SaParams& p) { p.first_replica = -1; }),
      with([](SaParams& p) {
        p.sweeps = 1;
        p.replicas = 3;
        p.first_replica = (std::int64_t{1} << 32) - 2;  // the last index would be 2^32
      }),
  };
  for (std::size_t k = 0; k < invalid.size(); ++k) {
    INFO("invalid parameters, case " << k);
    const std::string message = message_of([&] { sbmc::run_sa_sequential(four, invalid[k]); });
    CHECK(message.starts_with("run_sa_sequential: "));
  }
  CHECK(message_of([&] { sbmc::run_sa_sequential(four, invalid[0]); }) ==
        "run_sa_sequential: replicas = 0 must be at least 1");

  // n > 32768 for a DenseSpinGlass: rejected at once, before total_weight (O(n^2)).
  CHECK(message_of([] {
          sbmc::run_sa_sequential(DenseSpinGlass{32769, 1}, params(1, 1, 0.1, 3.0, 1));
        }).starts_with("run_sa_sequential: n = 32769"));
  // An invalid instance: the error of total_weight.
  const Graph repeated{3, {{0, 1, 1}, {0, 1, 1}}};
  const std::string repeated_message = message_of([&] { sbmc::run_sa_sequential(repeated, good); });
  CHECK(repeated_message.starts_with("total_weight"));

  // The last valid global replica index: no exception.
  SaParams last = good;
  last.replicas = 2;
  last.first_replica = (std::int64_t{1} << 32) - 2;
  CHECK_NOTHROW(sbmc::run_sa_sequential(four, last));

  // sa_beta and sa_threshold.
  CHECK_THROWS_AS(sbmc::sa_beta(0.1, 3.0, 0, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(0.1, 3.0, 5, -1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(0.1, 3.0, 5, 5), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(0.0, 3.0, 5, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(-0.1, 3.0, 5, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(kNaN, 3.0, 5, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(kInf, kInf, 5, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(0.1, kNaN, 5, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(0.1, kInf, 5, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_beta(0.1, 0.05, 5, 0), InvalidArgument);
  CHECK(message_of([] { sbmc::sa_beta(0.1, 3.0, 0, 0); }).starts_with("sa_beta: "));
  CHECK_THROWS_AS(sbmc::sa_threshold(0.0, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_threshold(-1.0, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_threshold(kNaN, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_threshold(kInf, 1), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_threshold(1.0, 0), InvalidArgument);
  CHECK_THROWS_AS(sbmc::sa_threshold(1.0, -2), InvalidArgument);
  CHECK(message_of([] { sbmc::sa_threshold(1.0, 0); }).starts_with("sa_threshold: "));
}
