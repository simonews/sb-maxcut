// Results schema 1, CSV helpers and the table of best-known values (sbmc/results.hpp).
//
// The files written by the tests go to a fresh folder under the temporary directory of the
// system, removed at the end; data/best_known.csv is only read.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/results.hpp>

#ifndef SBMC_DATA_DIR
#error "SBMC_DATA_DIR must be defined by CMake (absolute path of data/)"
#endif

namespace {

using sbmc::InvalidArgument;
using sbmc::IoError;

// A new folder under the temporary directory, removed with its content by the destructor.
class TempDir {
 public:
  TempDir() {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    for (int attempt = 0;; ++attempt) {
      path_ = std::filesystem::temp_directory_path() /
              ("sbmc_test_results_" + std::to_string(stamp) + "_" + std::to_string(attempt));
      if (std::filesystem::create_directory(path_)) {
        return;
      }
    }
  }
  ~TempDir() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  TempDir(const TempDir&) = delete;
  TempDir& operator=(const TempDir&) = delete;

  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

 private:
  std::filesystem::path path_;
};

std::string read_file(const std::filesystem::path& file) {
  std::ifstream in(file, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

void write_file(const std::filesystem::path& file, std::string_view text) {
  std::ofstream out(file, std::ios::binary);
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
}

std::string header_line() {
  const std::span<const std::string_view> columns = sbmc::results_columns();
  std::string line;
  for (std::size_t k = 0; k < columns.size(); ++k) {
    line += (k > 0 ? "," : "") + std::string(columns[k]);
  }
  return line;
}

// A row of 44 values: "v0", "v1", ..., with a comma and a quote in some of them.
std::vector<std::string> sample_row(int tag) {
  std::vector<std::string> row;
  for (std::size_t k = 0; k < sbmc::results_columns().size(); ++k) {
    row.push_back("v" + std::to_string(k) + "_" + std::to_string(tag));
  }
  row[3] = "";
  row[43] = "notes, with \"quotes\"";
  return row;
}

std::filesystem::path best_known_csv() {
  return std::filesystem::path(SBMC_DATA_DIR) / "best_known.csv";
}

constexpr std::string_view kBestKnownHeader =
    "instance,n,m,best_known,status,source_1,source_2,checked,notes\n";

}  // namespace

TEST_CASE("RES: results_columns has the 44 names of schema 1, in order") {
  const std::vector<std::string_view> expected = {"schema_version",
                                                  "timestamp",
                                                  "git_hash",
                                                  "dirty",
                                                  "host",
                                                  "os",
                                                  "gpu",
                                                  "driver",
                                                  "cuda",
                                                  "cublas",
                                                  "compiler",
                                                  "algo",
                                                  "variant",
                                                  "instance",
                                                  "n",
                                                  "nnz",
                                                  "replicas",
                                                  "steps",
                                                  "dt",
                                                  "c0",
                                                  "beta0",
                                                  "beta1",
                                                  "seed",
                                                  "rep",
                                                  "t_setup_ms",
                                                  "t_solve_ms",
                                                  "t_step_us",
                                                  "best_cut",
                                                  "mean_cut",
                                                  "n_success",
                                                  "p_s",
                                                  "target",
                                                  "tts99_ms",
                                                  "ranks",
                                                  "t_comm_ms",
                                                  "sm_clock_avg_mhz",
                                                  "temp_max_c",
                                                  "power_avg_w",
                                                  "a0",
                                                  "eval_every",
                                                  "first_replica",
                                                  "threads",
                                                  "target_source",
                                                  "notes"};
  REQUIRE(expected.size() == 44u);
  const std::span<const std::string_view> columns = sbmc::results_columns();
  REQUIRE(columns.size() == 44u);
  for (std::size_t k = 0; k < expected.size(); ++k) {
    INFO("column " << k);
    CHECK(columns[k] == expected[k]);
  }
  CHECK(sbmc::kResultsSchemaVersion == 1);
}

TEST_CASE("RES: csv_escape quotes fields with commas or quotes and rejects line breaks") {
  CHECK(sbmc::csv_escape("abc") == "abc");
  CHECK(sbmc::csv_escape("a,b") == "\"a,b\"");
  CHECK(sbmc::csv_escape("say \"hi\"") == "\"say \"\"hi\"\"\"");
  CHECK(sbmc::csv_escape("") == "");
  CHECK(sbmc::csv_escape(" spaces ") == " spaces ");
  CHECK_THROWS_AS(sbmc::csv_escape("a\nb"), InvalidArgument);
  CHECK_THROWS_AS(sbmc::csv_escape("a\rb"), InvalidArgument);
  CHECK_THROWS_AS(sbmc::csv_escape("\n"), InvalidArgument);
}

TEST_CASE("RES: csv_split inverts csv_line and rejects malformed lines") {
  const std::vector<std::vector<std::string>> rows = {
      {"x"},
      {""},
      {"", ""},
      {"", "", ""},
      {"a", "", "b"},
      {"a,b", "c\"d", ""},
      {"\"", ",", "\"\"", ",,"},
      {"say \"hi\", twice", "end"},
      {"", "last"},
      {"first", ""},
  };
  for (const std::vector<std::string>& row : rows) {
    const std::string line = sbmc::csv_line(row);
    INFO("line: " << line);
    CHECK(sbmc::csv_split(line) == row);
  }
  CHECK(sbmc::csv_line(std::vector<std::string>{"a", "b,c", ""}) == "a,\"b,c\",");
  CHECK(sbmc::csv_split("") == std::vector<std::string>{""});

  CHECK_THROWS_AS(sbmc::csv_split("\"abc"), InvalidArgument);     // quote not closed
  CHECK_THROWS_AS(sbmc::csv_split("a,\"b"), InvalidArgument);     // quote not closed
  CHECK_THROWS_AS(sbmc::csv_split("\"a\"b"), InvalidArgument);    // text after the closing quote
  CHECK_THROWS_AS(sbmc::csv_split("\"a\" ,b"), InvalidArgument);  // text after the closing quote
  CHECK_THROWS_AS(sbmc::csv_split("a\"b,c"), InvalidArgument);    // quote in an unquoted field
  CHECK_THROWS_AS(sbmc::csv_split("a,b\nc"), InvalidArgument);    // line break
  CHECK_THROWS_AS(sbmc::csv_line(std::vector<std::string>{"a\rb"}), InvalidArgument);
}

TEST_CASE("RES: append_results_row writes the header once and appends the rows") {
  const TempDir dir;
  const std::filesystem::path file = dir.path() / "runs.csv";
  const std::string header = header_line() + "\n";
  const std::vector<std::string> first = sample_row(1);
  const std::vector<std::string> second = sample_row(2);

  SUBCASE("new file, then a second row") {
    sbmc::append_results_row(file, first);
    CHECK(read_file(file) == header + sbmc::csv_line(first) + "\n");
    sbmc::append_results_row(file, second);
    CHECK(read_file(file) == header + sbmc::csv_line(first) + "\n" + sbmc::csv_line(second) + "\n");
    const std::string text = read_file(file);
    const std::size_t first_end = text.find('\n');
    const std::size_t second_end = text.find('\n', first_end + 1);
    CHECK(sbmc::csv_split(
              std::string_view(text).substr(first_end + 1, second_end - first_end - 1)) == first);
  }
  SUBCASE("an existing empty file gets the header first") {
    write_file(file, "");
    sbmc::append_results_row(file, first);
    CHECK(read_file(file) == header + sbmc::csv_line(first) + "\n");
  }
  SUBCASE("a file with another header: IoError, file unchanged") {
    const std::string other = "a,b,c\n1,2,3\n";
    write_file(file, other);
    CHECK_THROWS_AS(sbmc::append_results_row(file, first), IoError);
    CHECK(read_file(file) == other);
  }
  SUBCASE("a file with the header but a partial last line: IoError, file unchanged") {
    const std::string partial = header + "1,2,3";
    write_file(file, partial);
    CHECK_THROWS_AS(sbmc::append_results_row(file, first), IoError);
    CHECK(read_file(file) == partial);
  }
  SUBCASE("a row without 44 values, or with a line break: InvalidArgument, file untouched") {
    std::vector<std::string> short_row = first;
    short_row.pop_back();
    CHECK_THROWS_AS(sbmc::append_results_row(file, short_row), InvalidArgument);
    std::vector<std::string> long_row = first;
    long_row.emplace_back("extra");
    CHECK_THROWS_AS(sbmc::append_results_row(file, long_row), InvalidArgument);
    std::vector<std::string> broken = first;
    broken[10] = "two\nlines";
    CHECK_THROWS_AS(sbmc::append_results_row(file, broken), InvalidArgument);
    CHECK_FALSE(std::filesystem::exists(file));
  }
  SUBCASE("a folder that does not exist: IoError") {
    CHECK_THROWS_AS(sbmc::append_results_row(dir.path() / "missing" / "runs.csv", first), IoError);
  }
}

TEST_CASE("RES: read_best_known reads data/best_known.csv") {
  const std::vector<sbmc::BestKnownEntry> table = sbmc::read_best_known(best_known_csv());
  const std::vector<std::string> order = {"G1",  "G2",  "G11", "G12", "G14", "G15", "G22",
                                          "G43", "G48", "G55", "G63", "G70", "G81", "K2000"};
  REQUIRE(table.size() == order.size());
  for (std::size_t k = 0; k < order.size(); ++k) {
    INFO("row " << k);
    CHECK(table[k].instance == order[k]);
    CHECK(table[k].checked == "2026-11-16");
  }
  CHECK(table[0].n == 800);
  CHECK(table[0].m == 19176);
  CHECK(table[0].best_known == 11624);
  CHECK(table[0].status == "best_known");
  CHECK(table[0].source_1 == "F. Ma, J.-K. Hao 2017, arXiv:1510.09156, Table 5");
  CHECK(table[0].source_2 == "Z.-S. Shen et al. 2024, arXiv:2412.09285, Table S1");
  CHECK(table[0].notes.empty());
  CHECK(table[8].status == "optimal");
  CHECK(table[8].notes == "equals the number of edges, all of weight +1: every edge is cut");
  CHECK(table[10].best_known == 27052);
  CHECK(table[10].source_2.empty());
  CHECK(table[12].status == "reported_optimal");
  CHECK(table[13].best_known == 33337);
  CHECK(table[13].m == 1999000);

  const sbmc::BestKnownEntry* g2 = sbmc::find_best_known(table, "G2");
  REQUIRE(g2 != nullptr);
  CHECK(g2->best_known == 11620);
  CHECK(sbmc::find_best_known(table, "G3") == nullptr);
  CHECK(sbmc::find_best_known(table, "") == nullptr);
}

TEST_CASE("RES: read_best_known rejects malformed tables") {
  const TempDir dir;
  const std::filesystem::path file = dir.path() / "best_known.csv";
  const std::string row = "G1,800,19176,11624,best_known,a,b,2026-11-16,\n";

  // InvalidArgument whose message names the line.
  const auto rejects = [&](const std::string& text, const std::string& line) {
    write_file(file, text);
    std::string message;
    try {
      static_cast<void>(sbmc::read_best_known(file));
    } catch (const InvalidArgument& e) {
      message = e.what();
    }
    INFO("table:\n" << text << "message: " << message);
    CHECK(message.find(":" + line + ":") != std::string::npos);
  };

  rejects("instance,n,m,best_known,status,source_1,source_2,checked\n" + row, "1");
  rejects("", "1");
  rejects(std::string(kBestKnownHeader) + "G1,800,19176,11624,best_known,a,b,2026-11-16\n", "2");
  rejects(std::string(kBestKnownHeader) + "G1,800,19176,11624.5,best_known,a,b,2026-11-16,\n", "2");
  rejects(std::string(kBestKnownHeader) + "G1,800,x,11624,best_known,a,b,2026-11-16,\n", "2");
  rejects(std::string(kBestKnownHeader) + ",800,19176,11624,best_known,a,b,2026-11-16,\n", "2");
  rejects(std::string(kBestKnownHeader) + row + row, "3");
  rejects(std::string(kBestKnownHeader) + "\"G1,800\n", "2");

  write_file(file, std::string(kBestKnownHeader) + row + "\n");
  CHECK(sbmc::read_best_known(file).size() == 1u);  // empty lines are skipped

  CHECK_THROWS_AS(sbmc::read_best_known(dir.path() / "missing.csv"), IoError);
}
