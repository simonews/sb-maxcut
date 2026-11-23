// The rules of the process masks of sbmc_run --affinity (apps/cpu_topology.hpp, phase 07, part 4),
// on lists of logical processors written here: the i7-13620H of the laptop (6 P cores with two
// logical processors each, 0-11, efficiency class 1; 4 E cores, 12-15, class 0), a processor with
// one class, and the errors. The same test runs on Windows (the presets win-cpu and win-cpu-strict)
// and on Linux, where --affinity itself is not available.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpu_topology.hpp"

namespace {

using sbmc_app::Affinity;
using sbmc_app::LogicalProcessor;

// i7-13620H as Windows lists it: P core c has the logical processors 2c and 2c + 1 (class 1); the
// E cores 6..9 have 12..15 (class 0).
std::vector<LogicalProcessor> laptop() {
  std::vector<LogicalProcessor> processors;
  for (int core = 0; core < 6; ++core) {
    processors.push_back({2 * core, core, 1});
    processors.push_back({2 * core + 1, core, 1});
  }
  for (int k = 0; k < 4; ++k) {
    processors.push_back({12 + k, 6 + k, 0});
  }
  return processors;
}

// 8 cores with SMT, one class (class 0, as on a processor that is not hybrid).
std::vector<LogicalProcessor> one_class() {
  std::vector<LogicalProcessor> processors;
  for (int index = 0; index < 16; ++index) {
    processors.push_back({index, index / 2, 0});
  }
  return processors;
}

}  // namespace

TEST_CASE("cpulist: runs of two or more as a-b, the format of Cpus_allowed_list") {
  CHECK(sbmc_app::cpulist(0) == "");
  CHECK(sbmc_app::cpulist(0b1) == "0");
  CHECK(sbmc_app::cpulist(0b11) == "0-1");
  CHECK(sbmc_app::cpulist(0xFFF) == "0-11");
  CHECK(sbmc_app::cpulist(0x555) == "0,2,4,6,8,10");
  CHECK(sbmc_app::cpulist(0x10F) == "0-3,8");
  CHECK(sbmc_app::cpulist(~std::uint64_t{0}) == "0-63");
  CHECK(sbmc_app::cpulist(std::uint64_t{1} << 63) == "63");
}

TEST_CASE("the masks of the laptop: p1 6, p 12, e 4, all 16") {
  const std::vector<LogicalProcessor> processors = laptop();
  CHECK(sbmc_app::class_count(processors) == 2);
  const std::uint64_t p1 = sbmc_app::affinity_mask(processors, Affinity::p1);
  const std::uint64_t p = sbmc_app::affinity_mask(processors, Affinity::p);
  const std::uint64_t e = sbmc_app::affinity_mask(processors, Affinity::e);
  const std::uint64_t all = sbmc_app::affinity_mask(processors, Affinity::all);
  CHECK(sbmc_app::cpulist(p1) == "0,2,4,6,8,10");
  CHECK(sbmc_app::cpulist(p) == "0-11");
  CHECK(sbmc_app::cpulist(e) == "12-15");
  CHECK(sbmc_app::cpulist(all) == "0-15");
  CHECK(sbmc_app::mask_count(p1) == 6);
  CHECK(sbmc_app::mask_count(p) == 12);
  CHECK(sbmc_app::mask_count(e) == 4);
  CHECK(sbmc_app::mask_count(all) == 16);
  CHECK(sbmc_app::cpu_classes(processors, all) == "1:0-11 0:12-15");
  CHECK(sbmc_app::cpu_classes(processors, p1) == "1:0,2,4,6,8,10");
  CHECK(sbmc_app::cpu_classes(processors, e) == "0:12-15");
  CHECK(sbmc_app::cpu_classes(processors, 0) == "");
}

TEST_CASE("p1 takes the smallest index of every P core, whatever the order of the list") {
  std::vector<LogicalProcessor> processors = laptop();
  std::vector<LogicalProcessor> reversed(processors.rbegin(), processors.rend());
  CHECK(sbmc_app::affinity_mask(reversed, Affinity::p1) ==
        sbmc_app::affinity_mask(processors, Affinity::p1));
  // A P core whose logical processors are 7 and 3: p1 takes 3.
  const std::vector<LogicalProcessor> odd = {{7, 0, 2}, {3, 0, 2}, {5, 1, 2}, {0, 2, 1}};
  CHECK(sbmc_app::cpulist(sbmc_app::affinity_mask(odd, Affinity::p1)) == "3,5");
  CHECK(sbmc_app::cpulist(sbmc_app::affinity_mask(odd, Affinity::p)) == "3,5,7");
  CHECK(sbmc_app::cpulist(sbmc_app::affinity_mask(odd, Affinity::e)) == "0");
  CHECK(sbmc_app::cpu_classes(odd, 0xFF) == "2:3,5,7 1:0");
}

TEST_CASE("one class: p and p1 are all, e is an error") {
  const std::vector<LogicalProcessor> processors = one_class();
  CHECK(sbmc_app::class_count(processors) == 1);
  CHECK(sbmc_app::cpulist(sbmc_app::affinity_mask(processors, Affinity::p)) == "0-15");
  CHECK(sbmc_app::cpulist(sbmc_app::affinity_mask(processors, Affinity::p1)) == "0-15");
  CHECK(sbmc_app::cpulist(sbmc_app::affinity_mask(processors, Affinity::all)) == "0-15");
  CHECK_THROWS_AS(sbmc_app::affinity_mask(processors, Affinity::e), std::invalid_argument);
}

TEST_CASE("the errors of a list: empty, an index outside 0..63, an index twice, more than 64") {
  const std::vector<LogicalProcessor> none;
  CHECK_THROWS_AS(sbmc_app::affinity_mask(none, Affinity::all), std::runtime_error);
  const std::vector<LogicalProcessor> outside = {{0, 0, 0}, {64, 1, 0}};
  CHECK_THROWS_AS(sbmc_app::affinity_mask(outside, Affinity::all), std::runtime_error);
  const std::vector<LogicalProcessor> twice = {{0, 0, 0}, {0, 1, 0}};
  CHECK_THROWS_AS(sbmc_app::affinity_mask(twice, Affinity::all), std::runtime_error);
  std::vector<LogicalProcessor> many;
  for (int index = 0; index < 65; ++index) {
    many.push_back({index % 64, index, 0});
  }
  CHECK_THROWS_AS(sbmc_app::check_processors(many), std::runtime_error);
  std::vector<LogicalProcessor> full;
  for (int index = 0; index < 64; ++index) {
    full.push_back({index, index, 0});
  }
  CHECK(sbmc_app::affinity_mask(full, Affinity::all) == ~std::uint64_t{0});
}

TEST_CASE("the names of the rules") {
  for (const char* name : {"p1", "p", "e", "all"}) {
    const auto rule = sbmc_app::parse_affinity(name);
    REQUIRE(rule.has_value());
    CHECK(sbmc_app::affinity_name(*rule) == name);
  }
  CHECK_FALSE(sbmc_app::parse_affinity("P").has_value());
  CHECK_FALSE(sbmc_app::parse_affinity("").has_value());
  CHECK_FALSE(sbmc_app::parse_affinity("p2").has_value());
}
