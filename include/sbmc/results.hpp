// Results of the project: the CSV schema (version 1) written by the applications, one row per run,
// and the table of best-known cut values (data/best_known.csv).
//
// CSV format (RFC 4180): fields separated by ',', one row per line, lines terminated by '\n'. A
// field that contains ',' or '"' is enclosed in double quotes, with every inner '"' doubled; a
// field never contains a line break. An empty field is written as nothing between two commas.
//
// Schema 1 has 44 columns (results_columns()): the columns of section 10.4 of the reference
// document, in the same order, preceded by schema_version; a0, eval_every, first_replica, threads
// and target_source are added before notes, which stays the last column.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sbmc {

inline constexpr int kResultsSchemaVersion = 1;

// The 44 column names of schema 1, in order.
std::span<const std::string_view> results_columns() noexcept;

// RFC 4180 field: quoted when it contains ',' or '"' (inner quotes doubled).
// '\n' or '\r' in the field: InvalidArgument (one row per line).
std::string csv_escape(std::string_view field);
// Fields joined with ',' after csv_escape, without line terminator.
std::string csv_line(std::span<const std::string> fields);
// Inverse of csv_line for one line without terminator; malformed quoting: InvalidArgument.
// The result has at least one field (an empty line is one empty field). '\n' or '\r' in the line,
// a '"' inside an unquoted field, an unclosed quote or text after a closing quote:
// InvalidArgument.
std::vector<std::string> csv_split(std::string_view line);

// Appends one row (one value per column of results_columns(), "" = empty field).
// A missing or empty file gets the header first. An existing file whose first line differs
// from the header: IoError, and the file is not modified. Write errors: IoError.
// A row without exactly one value per column, or with a line break in a value: InvalidArgument,
// before the file is touched. A non-empty file that does not end with '\n' (a partial last line):
// IoError, and the file is not modified.
void append_results_row(const std::filesystem::path& file, std::span<const std::string> row);

// One row of data/best_known.csv.
struct BestKnownEntry {
  std::string instance;
  std::int64_t n = 0;
  std::int64_t m = 0;
  std::int64_t best_known = 0;
  std::string status, source_1, source_2, checked, notes;
};

// Reads a table with the header of data/best_known.csv (exact). Unreadable file: IoError.
// Wrong header, wrong number of fields, non-integer n, m or best_known, empty or repeated
// instance: InvalidArgument with the line number.
// Header: instance,n,m,best_known,status,source_1,source_2,checked,notes. Lines are separated by
// '\n' (one trailing '\r' per line is removed); empty lines are skipped; integers are decimal,
// with an optional '-'. Messages are "<file>:<line>: <reason>".
std::vector<BestKnownEntry> read_best_known(const std::filesystem::path& file);
// nullptr when the instance is not in the table.
const BestKnownEntry* find_best_known(std::span<const BestKnownEntry> table,
                                      std::string_view instance) noexcept;

}  // namespace sbmc
