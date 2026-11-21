// Results schema 2, the replica file and the table of best-known values (sbmc/results.hpp).
#include <sbmc/results.hpp>

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <ios>
#include <iterator>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#include <sbmc/error.hpp>

namespace sbmc {
namespace {

// Schema 2: columns 1-43 are the first 43 of schema 1 (schema_version, the columns of section 10.4
// of the reference document in the same order, then a0, eval_every, first_replica, threads and
// target_source); columns 44-55 are the additions of phase 07 (the times of the solvers, the
// rules of dt and c0, the extremes of the spectrum of J, the affinity and the OpenMP settings of
// the process, the tag of the run); notes stays the last column.
constexpr auto kColumns = std::to_array<std::string_view>({
    "schema_version",
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
    "t_prep_ms",
    "t_run_ms",
    "t_check_ms",
    "dt_rule",
    "c0_rule",
    "lambda_min",
    "lambda_max",
    "cpu_affinity",
    "cpu_classes",
    "omp_places",
    "omp_proc_bind",
    "run_tag",
    "notes",
});
static_assert(kColumns.size() == 56);

// Replica file, schema 1.
constexpr auto kReplicaColumns = std::to_array<std::string_view>({
    "schema_version",
    "timestamp",
    "instance",
    "algo",
    "variant",
    "replicas",
    "steps",
    "seed",
    "rep",
    "replica",
    "best_cut",
    "best_step",
    "final_cut",
});
static_assert(kReplicaColumns.size() == 13);

constexpr std::string_view kBestKnownHeader =
    "instance,n,m,best_known,status,source_1,source_2,checked,notes";
constexpr std::size_t kBestKnownFields = 9;

std::string header_of(std::span<const std::string_view> columns) {
  std::string header;
  for (std::size_t k = 0; k < columns.size(); ++k) {
    if (k > 0) {
      header.push_back(',');
    }
    header += columns[k];
  }
  return header;
}

// The file part common to append_results_row and append_replica_rows: `lines` (CSV lines, each
// terminated by '\n', already checked) are appended to `file`, after `header` when the file is
// missing or empty. An existing file must start with `header` and end with '\n', otherwise
// IoError and the file is not modified. `function` and `what` (the name of the file format) only
// enter the messages.
void append_lines(std::string_view function, std::string_view what,
                  const std::filesystem::path& file, const std::string& header,
                  const std::string& lines) {
  const std::string prefix = std::string(function) + ": ";
  std::error_code error;
  const bool exists = std::filesystem::exists(file, error);
  if (error) {
    throw IoError(prefix + "cannot access " + file.string() + ": " + error.message());
  }
  std::uintmax_t size = 0;
  if (exists) {
    if (std::filesystem::is_directory(file, error)) {
      throw IoError(prefix + file.string() + " is a directory");
    }
    size = std::filesystem::file_size(file, error);
    if (error) {
      throw IoError(prefix + "cannot access " + file.string() + ": " + error.message());
    }
  }
  std::string text;
  if (size == 0) {
    text = header + '\n';
  } else {
    std::ifstream in(file, std::ios::binary);
    std::string first;
    if (!in || !std::getline(in, first)) {
      throw IoError(prefix + "cannot read " + file.string());
    }
    if (first != header) {
      throw IoError(prefix + file.string() + ": the first line is not the header of " +
                    std::string(what) + "; the file was not modified");
    }
    in.clear();
    in.seekg(-1, std::ios::end);
    char last = '\0';
    if (!in.get(last)) {
      throw IoError(prefix + "cannot read " + file.string());
    }
    if (last != '\n') {
      throw IoError(prefix + file.string() +
                    " does not end with a line break (a partial last line?); the file was not "
                    "modified");
    }
  }
  text += lines;

  std::ofstream out(file, std::ios::binary | std::ios::app);
  if (!out) {
    throw IoError(prefix + "cannot open " + file.string() + " for writing");
  }
  out.write(text.data(), static_cast<std::streamsize>(text.size()));
  out.flush();
  if (!out) {
    throw IoError(prefix + "write error on " + file.string());
  }
}

std::string location(const std::filesystem::path& file, std::size_t line) {
  return file.string() + ":" + std::to_string(line) + ": ";
}

// Decimal int64 over the whole field, with an optional '-'.
bool parse_int64(std::string_view text, std::int64_t& value) {
  const char* const first = text.data();
  const char* const last = text.data() + text.size();
  const auto [end, error] = std::from_chars(first, last, value);
  return !text.empty() && error == std::errc{} && end == last;
}

}  // namespace

std::span<const std::string_view> results_columns() noexcept {
  return kColumns;
}

std::string csv_escape(std::string_view field) {
  if (field.find_first_of("\n\r") != std::string_view::npos) {
    throw InvalidArgument("csv_escape: a CSV field cannot contain a line break (one row per line)");
  }
  if (field.find_first_of(",\"") == std::string_view::npos) {
    return std::string(field);
  }
  std::string quoted;
  quoted.reserve(field.size() + 2);
  quoted.push_back('"');
  for (const char c : field) {
    if (c == '"') {
      quoted.push_back('"');
    }
    quoted.push_back(c);
  }
  quoted.push_back('"');
  return quoted;
}

std::string csv_line(std::span<const std::string> fields) {
  std::string line;
  for (std::size_t k = 0; k < fields.size(); ++k) {
    if (k > 0) {
      line.push_back(',');
    }
    line += csv_escape(fields[k]);
  }
  return line;
}

std::vector<std::string> csv_split(std::string_view line) {
  if (line.find_first_of("\n\r") != std::string_view::npos) {
    throw InvalidArgument("csv_split: the line contains a line break");
  }
  std::vector<std::string> fields;
  std::size_t pos = 0;
  for (;;) {
    std::string field;
    const std::string number = std::to_string(fields.size() + 1);
    if (pos < line.size() && line[pos] == '"') {
      ++pos;  // opening quote
      bool closed = false;
      while (pos < line.size()) {
        const char c = line[pos++];
        if (c != '"') {
          field.push_back(c);
        } else if (pos < line.size() && line[pos] == '"') {
          field.push_back('"');  // doubled quote
          ++pos;
        } else {
          closed = true;
          break;
        }
      }
      if (!closed) {
        throw InvalidArgument("csv_split: field " + number + ": the quote is not closed");
      }
      if (pos < line.size() && line[pos] != ',') {
        throw InvalidArgument("csv_split: field " + number + ": text after the closing quote");
      }
    } else {
      std::size_t end = line.find(',', pos);
      if (end == std::string_view::npos) {
        end = line.size();
      }
      const std::string_view raw = line.substr(pos, end - pos);
      if (raw.find('"') != std::string_view::npos) {
        throw InvalidArgument("csv_split: field " + number +
                              ": '\"' inside a field that is not enclosed in quotes");
      }
      field.assign(raw);
      pos = end;
    }
    fields.push_back(std::move(field));
    if (pos == line.size()) {
      return fields;
    }
    ++pos;  // line[pos] was ','
  }
}

void append_results_row(const std::filesystem::path& file, std::span<const std::string> row) {
  if (row.size() != kColumns.size()) {
    throw InvalidArgument("append_results_row: " + std::to_string(row.size()) + " values, " +
                          std::to_string(kColumns.size()) +
                          " expected (one per column of results schema 2)");
  }
  // Everything that can be checked without the file is checked before touching it.
  const std::string header = header_of(kColumns);
  const std::string line = csv_line(row) + '\n';
  append_lines("append_results_row", "results schema 2", file, header, line);
}

std::span<const std::string_view> replica_columns() noexcept {
  return kReplicaColumns;
}

void append_replica_rows(const std::filesystem::path& file,
                         std::span<const std::vector<std::string>> rows) {
  // Every row is checked and converted before the file is touched.
  std::string lines;
  for (std::size_t k = 0; k < rows.size(); ++k) {
    if (rows[k].size() != kReplicaColumns.size()) {
      throw InvalidArgument("append_replica_rows: row " + std::to_string(k) + ": " +
                            std::to_string(rows[k].size()) + " values, " +
                            std::to_string(kReplicaColumns.size()) +
                            " expected (one per column of the replica file, schema 1)");
    }
    lines += csv_line(rows[k]);
    lines += '\n';
  }
  append_lines("append_replica_rows", "the replica file (schema 1)", file,
               header_of(kReplicaColumns), lines);
}

std::vector<BestKnownEntry> read_best_known(const std::filesystem::path& file) {
  std::error_code error;
  if (std::filesystem::is_directory(file, error)) {
    throw IoError("read_best_known: " + file.string() + " is a directory");
  }
  std::ifstream in(file, std::ios::binary);
  if (!in) {
    throw IoError("read_best_known: cannot open " + file.string());
  }
  const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (in.bad()) {
    throw IoError("read_best_known: read error on " + file.string());
  }

  std::vector<BestKnownEntry> table;
  std::set<std::string, std::less<>> instances;
  std::size_t line_number = 0;
  std::size_t begin = 0;
  bool header_seen = false;
  while (begin <= text.size()) {
    std::size_t end = text.find('\n', begin);
    const bool last_line = end == std::string::npos;
    if (last_line) {
      end = text.size();
    }
    std::string_view line(text.data() + begin, end - begin);
    begin = end + 1;
    ++line_number;
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    if (!header_seen) {
      if (line != kBestKnownHeader) {
        throw InvalidArgument(location(file, line_number) + "the header must be '" +
                              std::string(kBestKnownHeader) + "'");
      }
      header_seen = true;
    } else if (!line.empty()) {
      std::vector<std::string> fields;
      try {
        fields = csv_split(line);
      } catch (const InvalidArgument& e) {
        throw InvalidArgument(location(file, line_number) + e.what());
      }
      if (fields.size() != kBestKnownFields) {
        throw InvalidArgument(location(file, line_number) + std::to_string(fields.size()) +
                              " fields, " + std::to_string(kBestKnownFields) + " expected");
      }
      BestKnownEntry entry;
      entry.instance = fields[0];
      if (entry.instance.empty()) {
        throw InvalidArgument(location(file, line_number) + "empty instance name");
      }
      const std::array<std::pair<const char*, std::int64_t*>, 3> numbers = {
          {{"n", &entry.n}, {"m", &entry.m}, {"best_known", &entry.best_known}}};
      for (std::size_t k = 0; k < numbers.size(); ++k) {
        if (!parse_int64(fields[k + 1], *numbers[k].second)) {
          throw InvalidArgument(location(file, line_number) + numbers[k].first + " = '" +
                                fields[k + 1] + "' is not an integer");
        }
      }
      entry.status = std::move(fields[4]);
      entry.source_1 = std::move(fields[5]);
      entry.source_2 = std::move(fields[6]);
      entry.checked = std::move(fields[7]);
      entry.notes = std::move(fields[8]);
      if (!instances.insert(entry.instance).second) {
        throw InvalidArgument(location(file, line_number) + "instance " + entry.instance +
                              " is repeated");
      }
      table.push_back(std::move(entry));
    }
    if (last_line) {
      break;
    }
  }
  return table;
}

const BestKnownEntry* find_best_known(std::span<const BestKnownEntry> table,
                                      std::string_view instance) noexcept {
  for (const BestKnownEntry& entry : table) {
    if (entry.instance == instance) {
      return &entry;
    }
  }
  return nullptr;
}

}  // namespace sbmc
