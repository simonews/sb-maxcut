// T6 on example inputs: the rudy reader (sbmc/rudy.hpp). Valid inputs give the canonical graph;
// every error is an sbmc::IoError whose what() starts with "<source>:<line>: " for an error tied
// to a line and with "<source>: " otherwise. The check on the real G-set files belongs to
// phase 04.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>
#include <sbmc/rudy.hpp>

#ifndef SBMC_TEST_DATA_DIR
#error "SBMC_TEST_DATA_DIR must be defined by CMake (absolute path of tests/data)"
#endif

namespace {

using sbmc::Edge;
using sbmc::Graph;
using sbmc::Spin;

// Canonical edges of the 4-node example (tests/data/four_nodes.rudy).
const std::vector<Edge> kFourNodeEdges = {{0, 1, 1}, {0, 2, 1}, {0, 3, 1}, {1, 2, 1}, {2, 3, 1}};

Graph parse(std::string_view text) {
  std::istringstream in{std::string(text)};
  return sbmc::parse_rudy(in, "case");
}

// what() of the IoError thrown by parse_rudy on text with source "case", or "(no exception)".
// Any other exception type escapes and fails the test case.
std::string io_error(std::string_view text) {
  std::istringstream in{std::string(text)};
  try {
    (void)sbmc::parse_rudy(in, "case");
  } catch (const sbmc::IoError& e) {
    return e.what();
  }
  return "(no exception)";
}

// The input with '\n', '\r' and '\t' made visible, for the messages of the checks.
std::string visible(std::string_view text) {
  std::string out;
  for (const char c : text) {
    if (c == '\n') {
      out += "\\n";
    } else if (c == '\r') {
      out += "\\r";
    } else if (c == '\t') {
      out += "\\t";
    } else {
      out += c;
    }
  }
  return out;
}

bool same_edges(const std::vector<Edge>& a, const std::vector<Edge>& b) {
  return std::equal(a.begin(), a.end(), b.begin(), b.end(), [](const Edge& x, const Edge& y) {
    return x.u == y.u && x.v == y.v && x.w == y.w;
  });
}

std::string edges_text(const std::vector<Edge>& edges) {
  std::string text;
  for (const Edge& e : edges) {
    text += "(" + std::to_string(e.u) + "," + std::to_string(e.v) + "," + std::to_string(e.w) + ")";
  }
  return text;
}

}  // namespace

TEST_CASE("T6: four_nodes.rudy gives the canonical 4-node graph and the T1 values") {
  const Graph g = sbmc::read_rudy(SBMC_TEST_DATA_DIR "/four_nodes.rudy");
  CHECK(g.n == 4);
  INFO("edges " << edges_text(g.edges));
  CHECK(same_edges(g.edges, kFourNodeEdges));
  const std::vector<Spin> ac_bd = {1, -1, 1, -1};
  const std::vector<Spin> ab_cd = {1, 1, -1, -1};
  CHECK(sbmc::total_weight(g) == 5);
  CHECK(sbmc::ising_energy(g, ac_bd) == -3);
  CHECK(sbmc::cut_direct(g, ac_bd) == 4);
  CHECK(sbmc::ising_energy(g, ab_cd) == -1);
  CHECK(sbmc::cut_direct(g, ab_cd) == 3);
}

TEST_CASE("T6: CRLF, tabs, extra spaces, blank lines and a missing final newline") {
  const std::vector<std::string> inputs = {
      "4 5\r\n1 2 1\r\n2 3 1\r\n3 4 1\r\n4 1 1\r\n1 3 1\r\n",
      "  4 \t 5\t\n\t1   2\t1  \n 2 3 1\n3\t\t4 1 \n  4  1  1\n1 3   1\t\t\n",
      "\n4 5\n\n1 2 1\n   \n2 3 1\n\t\n3 4 1\n4 1 1\n \t \n1 3 1\n\n\n",
      "\r\n4 5\r\n \r\n1 2 1\r\n2 3 1\r\n3 4 1\r\n4 1 1\r\n1 3 1",
      "4 5\n1 2 1\n2 3 1\n3 4 1\n4 1 1\n1 3 1",
  };
  for (const std::string& text : inputs) {
    INFO("input \"" << visible(text) << "\"");
    const Graph g = parse(text);
    CHECK(g.n == 4);
    CHECK(same_edges(g.edges, kFourNodeEdges));
  }
}

TEST_CASE("T6: signed weights, edges with i > j and edges out of order give the canonical graph") {
  const Graph g = parse("5 4\n5 4 +3\n3 1 -1\n2 1 +1\n4 2 -2147483647\n");
  CHECK(g.n == 5);
  const std::vector<Edge> expected = {{0, 1, 1}, {0, 2, -1}, {1, 3, -2147483647}, {3, 4, 3}};
  INFO("edges " << edges_text(g.edges));
  CHECK(same_edges(g.edges, expected));

  const Graph extremes = parse("+2147483647 1\n2147483647 1 2147483647\n");
  CHECK(extremes.n == 2147483647);
  CHECK(same_edges(extremes.edges, std::vector<Edge>{{0, 2147483646, 2147483647}}));
}

TEST_CASE("T6: graphs without edges") {
  const Graph five = parse("5 0");
  CHECK(five.n == 5);
  CHECK(five.edges.empty());
  const Graph one = parse("1 0\n");
  CHECK(one.n == 1);
  CHECK(one.edges.empty());
}

TEST_CASE("T6: errors without a line number: no header, fewer than M edge lines") {
  const std::vector<std::string> inputs = {"", "  \n\n", "\t\r\n", "4 3\n1 2 1\n2 3 1"};
  for (const std::string& text : inputs) {
    INFO("input \"" << visible(text) << "\"");
    const std::string message = io_error(text);
    INFO("what() = " << message);
    CHECK(message.starts_with("case: "));
  }
}

TEST_CASE("T6: errors tied to a line report its physical number") {
  struct Case {
    std::string text;
    int line;
  };
  const std::vector<Case> cases = {
      // Header (line 1).
      {"4", 1},
      {"4 5 6", 1},
      {"four 5", 1},
      {"0 0", 1},
      {"4 -1", 1},
      {"4 7", 1},
      {"2147483648 0", 1},
      {"4 5.0", 1},
      {"99999999999999999999999 1", 1},
      // Edge lines (line 2).
      {"4 1\n1 2", 2},
      {"4 1\n1 2 1 1", 2},
      {"4 1\n0 2 1", 2},
      {"4 1\n1 5 1", 2},
      {"4 1\n3 3 1", 2},
      {"4 1\n1 2 0", 2},
      {"4 1\n1 2 1.5", 2},
      {"4 1\n1 2 1.0", 2},
      {"4 1\n1 2 1e3", 2},
      {"4 1\n1 2 2147483648", 2},
      {"4 1\n1 2 -2147483648", 2},
      {"4 1\n1 2a 1", 2},
      {"4 1\n1 2 +-1", 2},
      {"4 1\n1 2 -", 2},
      {"4 1\n1 2 1\r\r", 2},
      // A non-blank line after the M edge lines.
      {"4 1\n1 2 1\n2 3 1", 3},
      {"5 0\n1 2 1", 2},
      // Physical line numbers after blank lines.
      {"\n\n4 1\n\n1 1 1", 5},
      {"\r\n4 1\r\n\r\n\r\n1 2 x\r\n", 5},
  };
  for (const Case& c : cases) {
    INFO("input \"" << visible(c.text) << "\"");
    const std::string message = io_error(c.text);
    INFO("what() = " << message);
    CHECK(message.starts_with("case:" + std::to_string(c.line) + ": "));
  }
}

TEST_CASE("T6: a repeated edge is reported at its second occurrence, citing the first") {
  struct Case {
    std::string text;
    int line;
    std::string first;
  };
  const std::vector<Case> cases = {
      {"4 2\n1 2 1\n1 2 -1", 3, "line 2"},
      {"4 2\n1 2 1\n\n2 1 1", 4, "line 2"},
      // The first error in input order wins: the repeated edge (line 4) comes before the
      // invalid token (line 5) and before the missing edge lines.
      {"4 4\n1 2 1\n3 4 1\n4 3 1\n1 x 1", 4, "line 3"},
      {"4 5\n1 2 1\n2 3 1\n3 2 1", 4, "line 3"},
      // Two repeated edges: the one whose second occurrence comes first.
      {"4 4\n3 4 1\n1 2 1\n2 1 1\n4 3 1", 4, "line 3"},
  };
  for (const Case& c : cases) {
    INFO("input \"" << visible(c.text) << "\"");
    const std::string message = io_error(c.text);
    INFO("what() = " << message);
    CHECK(message.starts_with("case:" + std::to_string(c.line) + ": "));
    CHECK(message.find(c.first) != std::string::npos);
  }
}

TEST_CASE("T6: read_rudy on a missing file or on a directory throws IoError with the path") {
  const std::filesystem::path missing = SBMC_TEST_DATA_DIR "/does_not_exist.rudy";
  REQUIRE_FALSE(std::filesystem::exists(missing));
  std::string message;
  try {
    (void)sbmc::read_rudy(missing);
  } catch (const sbmc::IoError& e) {
    message = e.what();
  }
  INFO("what() = " << message);
  CHECK(!message.empty());
  CHECK(message.find(missing.string()) != std::string::npos);

  const std::filesystem::path directory = SBMC_TEST_DATA_DIR;
  std::string directory_message;
  try {
    (void)sbmc::read_rudy(directory);
  } catch (const sbmc::IoError& e) {
    directory_message = e.what();
  }
  INFO("what() for a directory = " << directory_message);
  CHECK(directory_message.starts_with(directory.string() + ": "));
}
