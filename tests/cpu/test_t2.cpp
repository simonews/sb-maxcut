// T2 (reference document, section 8.2): on small graphs the heuristic solvers reach the exact
// optimum of the enumeration C3 and never exceed it. Phase 05: the sequential reference C0
// (sbmc/sb_sequential.hpp); C1 and simulated annealing join in the later phases.
//
// Every case first computes the optimum with solve_bruteforce, which must equal the value of the
// table, then runs C0 with R = 16, a0 = 1, automatic c0 (c0 = 0), first_replica = 0 and no trace.
// The numbers of replicas that reach the optimum were computed by cldArchitect with an emulation
// of C0 in float32 (the one that produced the values of phase 04), equal to n_success of sbmc_run
// in the cpu-strict preset of F04-R2; an emulation in float64 gives the same counts. "Strict"
// means SBMC_STRICT_FP_BUILD = 1 (no FMA contraction): there the counts and the best cuts of T2-3
// are compared exactly; in the production presets FMA contraction may change them, and only the
// properties of every preset are checked.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include <sbmc/bruteforce.hpp>
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
using sbmc::Instance;
using sbmc::SbAlgorithm;
using sbmc::SbParams;
using sbmc::SbResult;
using sbmc::Spin;

constexpr bool kStrict = SBMC_STRICT_FP_BUILD != 0;
constexpr int kReplicas = 16;

Instance rudy(const char* name) {
  return sbmc::read_rudy(std::string(SBMC_TEST_DATA_DIR) + "/" + name);
}

struct T2Case {
  std::string name;  // case and instance, for the messages
  Instance instance;
  SbAlgorithm algorithm;
  float dt;
  int steps;
  std::uint64_t seed;
  int eval_every;
  std::int64_t optimum;
  int at_optimum_strict;  // replicas with best_cut = optimum in the strict presets
};

// dt = 0.75 on sparse_24_unit.rudy: all its weights are positive, and with lambda_min = -4.359
// and lambda_max = 3.121 the general formula of the reference document (section 6) gives 1.142.
std::vector<T2Case> t2_cases() {
  const SbAlgorithm dsb = SbAlgorithm::discrete;
  const SbAlgorithm bsb = SbAlgorithm::ballistic;
  return {
      {"T2-1 sparse_24_unit.rudy dSB", rudy("sparse_24_unit.rudy"), dsb, 0.75f, 300, 51, 0, 40, 16},
      {"T2-2 sparse_24_unit.rudy bSB", rudy("sparse_24_unit.rudy"), bsb, 0.75f, 300, 51, 1, 40, 16},
      {"T2-3 DenseSpinGlass{16, 16} dSB", DenseSpinGlass{16, 16}, dsb, 1.25f, 200, 54, 0, 29, 14},
      {"T2-4 DenseSpinGlass{16, 16} bSB", DenseSpinGlass{16, 16}, bsb, 1.25f, 200, 54, 1, 29, 16},
      {"T2-5 DenseSpinGlass{22, 22} dSB", DenseSpinGlass{22, 22}, dsb, 1.25f, 300, 54, 0, 34, 16},
      {"T2-6 DenseSpinGlass{22, 22} bSB", DenseSpinGlass{22, 22}, bsb, 1.25f, 300, 54, 1, 34, 16},
      {"T2-7 torus_4x6_pm1.rudy bSB", rudy("torus_4x6_pm1.rudy"), bsb, 1.25f, 200, 54, 1, 24, 16},
  };
}

SbParams c0_params(const T2Case& c) {
  SbParams p;
  p.algorithm = c.algorithm;
  p.replicas = kReplicas;
  p.steps = c.steps;
  p.dt = c.dt;
  p.a0 = 1.0f;
  p.c0 = 0.0f;  // automatic
  p.eval_every = c.eval_every;
  p.seed = c.seed;
  p.first_replica = 0;
  return p;
}

std::string cuts_text(const std::vector<std::int64_t>& cuts) {
  std::string text;
  for (const std::int64_t cut : cuts) {
    text += (text.empty() ? "" : ", ") + std::to_string(cut);
  }
  return text;
}

// The properties of T2 for the best cuts and best configurations of a heuristic solver
// (replica-major spins, R * n); returns the number of replicas at the optimum.
int check_against_optimum(const Instance& inst, std::int64_t optimum,
                          const std::vector<std::int64_t>& best_cuts,
                          const std::vector<Spin>& best_spins) {
  const auto n = static_cast<std::size_t>(sbmc::num_spins(inst));
  REQUIRE(best_spins.size() == best_cuts.size() * n);
  int at_optimum = 0;
  for (std::size_t r = 0; r < best_cuts.size(); ++r) {
    INFO("replica " << r << ": best_cut " << best_cuts[r]);
    const std::span<const Spin> spins(best_spins.data() + r * n, n);
    CHECK(best_cuts[r] <= optimum);
    CHECK(best_cuts[r] == sbmc::cut_direct(inst, spins));
    at_optimum += best_cuts[r] == optimum ? 1 : 0;
  }
  CHECK(at_optimum >= 1);
  return at_optimum;
}

}  // namespace

TEST_CASE("T2: C0 reaches the exact optimum and never exceeds it") {
  for (const T2Case& c : t2_cases()) {
    INFO(c.name);
    const sbmc::BruteForceResult exact = sbmc::solve_bruteforce(c.instance);
    CHECK(exact.max_cut == c.optimum);

    const SbResult result = sbmc::run_sb_sequential(c.instance, c0_params(c));
    REQUIRE(result.per_replica.size() == static_cast<std::size_t>(kReplicas));
    std::vector<std::int64_t> best_cuts;
    for (const sbmc::SbReplicaResult& replica : result.per_replica) {
      best_cuts.push_back(replica.best_cut);
    }
    INFO("best cuts: " << cuts_text(best_cuts));
    const int at_optimum =
        check_against_optimum(c.instance, c.optimum, best_cuts, result.best_spins);

    if (!kStrict) {
      MESSAGE(c.name << ": " << at_optimum << "/" << kReplicas
                     << " replicas at the optimum (production preset: not compared)");
      continue;
    }
    CHECK(at_optimum == c.at_optimum_strict);
    if (c.name.starts_with("T2-3 ")) {
      const std::vector<std::int64_t> expected = {29, 29, 29, 29, 29, 29, 29, 29,
                                                  27, 29, 27, 29, 29, 29, 29, 29};
      CHECK(best_cuts == expected);
    }
  }
}
