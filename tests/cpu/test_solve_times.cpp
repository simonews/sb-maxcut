// The times of the solvers (sbmc/solve_times.hpp, phase 07): SolveTimes and the field times of
// SbResult and SaResult start at zero; every call of run_sb_sequential, run_sb_openmp,
// run_sa_sequential and run_sa_openmp returns three finite, non-negative phases (preparation,
// run, final check) whose sum is at most the duration of the call measured outside it with
// std::chrono::steady_clock; on data/gset/G1 the run takes a measurable time. Valid in every
// preset: the times do not depend on the floating-point build. dense_graph_40 is a copy of the
// function of test_sb_openmp.cpp.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>

#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>
#include <sbmc/sa.hpp>
#include <sbmc/sa_openmp.hpp>
#include <sbmc/sa_sequential.hpp>
#include <sbmc/sb.hpp>
#include <sbmc/sb_openmp.hpp>
#include <sbmc/sb_sequential.hpp>
#include <sbmc/solve_times.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif
#ifndef SBMC_DATA_DIR
#error "SBMC_DATA_DIR must be defined by CMake (absolute path of data/)"
#endif

namespace {

using sbmc::DenseSpinGlass;
using sbmc::Graph;
using sbmc::Instance;
using sbmc::SaParams;
using sbmc::SbAlgorithm;
using sbmc::SbParams;
using sbmc::SolveTimes;

Graph g1_of_gset() {
  return sbmc::read_rudy(std::string(SBMC_DATA_DIR) + "/gset/G1");
}

// dense_graph_40: n = 40 and, for 0 <= u < v < 40 in canonical order, the edge (u, v, w) with
// w = ((3u + 5v + uv) mod 7) - 3 when w != 0: 689 edges, |w| <= 3.
Graph dense_graph_40() {
  Graph g{40, {}};
  for (std::int32_t u = 0; u < 40; ++u) {
    for (std::int32_t v = u + 1; v < 40; ++v) {
      const std::int32_t w = ((3 * u + 5 * v + u * v) % 7) - 3;
      if (w != 0) {
        g.edges.push_back({u, v, w});
      }
    }
  }
  return g;
}

SbParams sb_params(SbAlgorithm algorithm, int replicas, int steps, float dt, float c0,
                   int eval_every, std::uint64_t seed) {
  SbParams p;
  p.algorithm = algorithm;
  p.replicas = replicas;
  p.steps = steps;
  p.dt = dt;
  p.c0 = c0;  // 0: automatic
  p.eval_every = eval_every;
  p.seed = seed;
  return p;
}

SaParams sa_params(int replicas, int sweeps, double beta0, double beta1, std::uint64_t seed) {
  SaParams p;
  p.replicas = replicas;
  p.sweeps = sweeps;
  p.beta0 = beta0;
  p.beta1 = beta1;
  p.seed = seed;
  return p;
}

// The times of a call and its duration D measured outside it, in milliseconds.
struct Timed {
  SolveTimes times;
  double outside_ms = 0.0;
};

template <typename Call>
Timed timed(Call&& call) {
  const auto start = std::chrono::steady_clock::now();
  const auto result = std::forward<Call>(call)();
  const auto stop = std::chrono::steady_clock::now();
  return {result.times, std::chrono::duration<double, std::milli>(stop - start).count()};
}

// The three phases finite and >= 0, their sum <= D + 1e-6; run_ms > 0 when run_positive.
void check_times(const std::string& label, const Timed& t, bool run_positive) {
  INFO(label << ": prep_ms " << t.times.prep_ms << ", run_ms " << t.times.run_ms << ", check_ms "
             << t.times.check_ms << ", D " << t.outside_ms);
  for (const double phase : {t.times.prep_ms, t.times.run_ms, t.times.check_ms}) {
    CHECK(std::isfinite(phase));
    CHECK(phase >= 0.0);
  }
  CHECK(t.times.prep_ms + t.times.run_ms + t.times.check_ms <= t.outside_ms + 1e-6);
  if (run_positive) {
    CHECK(t.times.run_ms > 0.0);
  }
}

// run_sb_sequential (threads 0) or run_sb_openmp with `threads` threads.
void check_sb(const std::string& label, const Instance& inst, const SbParams& params, int threads,
              bool run_positive) {
  if (threads == 0) {
    check_times(label + ", run_sb_sequential",
                timed([&] { return sbmc::run_sb_sequential(inst, params); }), run_positive);
  } else {
    check_times(label + ", run_sb_openmp, threads " + std::to_string(threads),
                timed([&] { return sbmc::run_sb_openmp(inst, params, threads); }), run_positive);
  }
}

// run_sa_sequential (threads 0) or run_sa_openmp with `threads` threads.
void check_sa(const std::string& label, const Instance& inst, const SaParams& params, int threads,
              bool run_positive) {
  if (threads == 0) {
    check_times(label + ", run_sa_sequential",
                timed([&] { return sbmc::run_sa_sequential(inst, params); }), run_positive);
  } else {
    check_times(label + ", run_sa_openmp, threads " + std::to_string(threads),
                timed([&] { return sbmc::run_sa_openmp(inst, params, threads); }), run_positive);
  }
}

}  // namespace

TEST_CASE("TIMES: SolveTimes and the field times start at zero") {
  const SolveTimes times{};
  CHECK(times.prep_ms == 0.0);
  CHECK(times.run_ms == 0.0);
  CHECK(times.check_ms == 0.0);
  const sbmc::SbResult sb{};
  CHECK(sb.times.prep_ms == 0.0);
  CHECK(sb.times.run_ms == 0.0);
  CHECK(sb.times.check_ms == 0.0);
  const sbmc::SaResult sa{};
  CHECK(sa.times.prep_ms == 0.0);
  CHECK(sa.times.run_ms == 0.0);
  CHECK(sa.times.check_ms == 0.0);
}

TEST_CASE("TIMES: simulated bifurcation, C0 and C1") {
  const Instance g1 = g1_of_gset();
  const SbParams dsb = sb_params(SbAlgorithm::discrete, 40, 30, 0.5f, 0.0f, 0, 1);
  const SbParams bsb = sb_params(SbAlgorithm::ballistic, 40, 30, 0.5f, 0.0f, 7, 1);
  check_sb("G1 dSB", g1, dsb, 0, true);
  check_sb("G1 bSB eval_every 7", g1, bsb, 0, true);
  check_sb("G1 dSB", g1, dsb, 1, true);
  check_sb("G1 dSB", g1, dsb, 3, true);
  check_sb("G1 bSB eval_every 7", g1, bsb, 3, true);

  const Instance dense40 = dense_graph_40();
  const SbParams dense40_params = sb_params(SbAlgorithm::discrete, 33, 20, 0.5f, 0.3f, 0, 2);
  check_sb("dense_graph_40 dSB", dense40, dense40_params, 0, false);
  check_sb("dense_graph_40 dSB", dense40, dense40_params, 3, false);

  const Instance glass = DenseSpinGlass{200, 7};
  const SbParams glass_params = sb_params(SbAlgorithm::discrete, 33, 20, 1.25f, 0.0f, 0, 3);
  check_sb("DenseSpinGlass{200, 7} dSB", glass, glass_params, 0, false);
  check_sb("DenseSpinGlass{200, 7} dSB", glass, glass_params, 3, false);
}

TEST_CASE("TIMES: simulated annealing, the reference and C2") {
  const Instance g1 = g1_of_gset();
  const SaParams g1_params = sa_params(40, 20, 0.1, 3.0, 1);
  check_sa("G1", g1, g1_params, 0, true);
  check_sa("G1", g1, g1_params, 1, true);
  check_sa("G1", g1, g1_params, 3, true);

  const Instance dense40 = dense_graph_40();
  const SaParams dense40_params = sa_params(33, 20, 0.05, 3.0, 2);
  check_sa("dense_graph_40", dense40, dense40_params, 0, false);
  check_sa("dense_graph_40", dense40, dense40_params, 3, false);

  const Instance glass = DenseSpinGlass{200, 7};
  const SaParams glass_params = sa_params(33, 10, 0.01, 3.0, 3);
  check_sa("DenseSpinGlass{200, 7}", glass, glass_params, 0, false);
  check_sa("DenseSpinGlass{200, 7}", glass, glass_params, 3, false);
}
