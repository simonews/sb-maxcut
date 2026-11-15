// Reader of graphs in the rudy format of the G-set.
//
// Format: a header line "N M", then exactly M edge lines "i j w" with 1-based node indices.
// Rules:
// - Lines are separated by '\n'; the last line may lack it; one trailing '\r' is removed.
//   Leading and trailing spaces and tabs are ignored, and tokens are separated by spaces or
//   tabs. Blank lines (empty or made only of spaces and tabs) are ignored everywhere but count
//   for the line numbers, which start at 1 and are the physical lines of the input.
// - The header is the first non-blank line: exactly two integer tokens N and M, with
//   1 <= N <= 2^31 - 1 and 0 <= M <= N(N - 1)/2 (computed in int64).
// - Then exactly M non-blank lines with exactly three integer tokens i, j, w: 1 <= i, j <= N,
//   i != j, w != 0 and |w| <= 2^31 - 1.
// - An integer token is an optional '+' or '-' followed by decimal digits only: "1.0", "1.5",
//   "1e3", "12a" and "+-1" are errors, and so is a value outside the range of its field.
// - A repeated edge, in the same or in the opposite direction, is an error; a non-blank line after
//   the M edge lines is an error; fewer than M edge lines is an error.
// Result: the canonical Graph (sbmc/instance.hpp): n = N, u = min(i, j) - 1, v = max(i, j) - 1,
// edges sorted by (u, v). The returned graph is always valid.
//
// Errors throw sbmc::IoError. what() is "<source>:<line>: <reason>" for an error tied to a line
// (for a repeated edge, the line of its second occurrence, with "line <number>" of the first one
// in the reason), and "<source>: <reason>" otherwise (no header, fewer than M edge lines, read
// error). When the input has several errors, the first one in input order is reported.
#pragma once

#include <filesystem>
#include <istream>
#include <string_view>

#include <sbmc/instance.hpp>

namespace sbmc {

// Reads a rudy file; the source in the messages is file.string(). Throws IoError, with the path
// in the message, when the file cannot be opened or is a directory.
Graph read_rudy(const std::filesystem::path& file);

// Same rules on a stream; source names the input in the messages.
Graph parse_rudy(std::istream& in, std::string_view source);

}  // namespace sbmc
