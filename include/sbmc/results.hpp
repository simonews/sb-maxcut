// Results of the project: the CSV schema (version 2) written by the applications, one row per run,
// and the table of best-known cut values (data/best_known.csv).
//
// CSV format (RFC 4180): fields separated by ',', one row per line, lines terminated by '\n'. A
// field that contains ',' or '"' is enclosed in double quotes, with every inner '"' doubled; a
// field never contains a line break. An empty field is written as nothing between two commas.
//
// Schema 2 (phase 07) has 56 columns (results_columns()):
// - 1-43: the first 43 columns of schema 1, with the same names and in the same order: the columns
//   of section 10.4 of the reference document, in the same order, preceded by schema_version,
//   then a0, eval_every, first_replica, threads and target_source;
// - 44-55: t_prep_ms, t_run_ms, t_check_ms (the times of the solver, sbmc/solve_times.hpp);
//   dt_rule, c0_rule (how dt and c0 of SB were chosen); lambda_min, lambda_max (estimated
//   extremes of the spectrum of J); cpu_affinity, cpu_classes (the logical processors allowed to
//   the process and their efficiency classes); omp_places, omp_proc_bind (the OpenMP environment
//   variables); run_tag (a free label of the run);
// - 56: notes, which stays the last column.
// Schema 1 (44 columns, the 43 above and notes) is no longer written: a file with its header is
// rejected by append_results_row like a file with any other header.
//
// The replica file (schema 1, 13 columns, replica_columns()) has one row per replica of a recorded
// repetition, for the relaxed targets and the bootstrap of the time to solution: schema_version
// (1); timestamp, instance, algo, variant, replicas, steps, seed and rep, equal to the values of
// the row of results of the same repetition (steps: M for SB, S for SA); replica, the global index
// first_replica + r; best_cut, best_step (best_step of SB, best_sweep of SA) and final_cut of the
// replica. The rows of a repetition are in increasing replica order.
#pragma once

#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace sbmc {

inline constexpr int kResultsSchemaVersion = 2;

// The 56 column names of schema 2, in order.
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

inline constexpr int kReplicaSchemaVersion = 1;

// The 13 column names of the replica file, schema 1, in order:
// schema_version, timestamp, instance, algo, variant, replicas, steps, seed, rep, replica,
// best_cut, best_step, final_cut
std::span<const std::string_view> replica_columns() noexcept;

// Appends rows (one value per column of replica_columns() each) with the same rules as
// append_results_row: header first in a missing or empty file; another header or a partial last
// line: IoError and the file is not modified; a wrong number of values or a line break in a
// value: InvalidArgument before the file is touched; write errors: IoError. With no rows the file
// is checked, and gets the header when it is missing or empty.
void append_replica_rows(const std::filesystem::path& file,
                         std::span<const std::vector<std::string>> rows);

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
