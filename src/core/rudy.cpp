// Reader of the rudy format of the G-set (sbmc/rudy.hpp).
//
// The input is read line by line and checked as it goes. Repeated edges are found by sorting the
// edges read so far by (u, v, line): at the end of the input, and before reporting any error
// found later in the input, so that the first error in input order is the one reported. Nothing
// is preallocated from the M of the header.
#include <sbmc/rudy.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <sbmc/error.hpp>
#include <sbmc/instance.hpp>

namespace sbmc {
namespace {

constexpr std::int64_t kMaxNodes = 2147483647;      // 2^31 - 1
constexpr std::int64_t kMaxAbsWeight = 2147483647;  // 2^31 - 1
// Tokens with a larger magnitude are out of range for every field: the largest valid value is
// M = N(N - 1)/2 < 2^61, with N = 2^31 - 1.
constexpr std::uint64_t kTokenMagnitudeCap = std::uint64_t{1} << 62;
// Characters of a token shown in a message.
constexpr std::size_t kTokenDisplayLimit = 32;

enum class TokenStatus { ok, not_integer, out_of_range };

struct IntegerToken {
  TokenStatus status = TokenStatus::not_integer;
  std::int64_t value = 0;
};

// An integer token: an optional '+' or '-' followed by decimal digits only.
IntegerToken parse_integer(std::string_view token) {
  std::size_t pos = 0;
  bool negative = false;
  if (!token.empty() && (token.front() == '+' || token.front() == '-')) {
    negative = token.front() == '-';
    pos = 1;
  }
  if (pos == token.size()) {
    return {TokenStatus::not_integer, 0};
  }
  std::uint64_t magnitude = 0;
  bool too_large = false;
  for (std::size_t k = pos; k < token.size(); ++k) {
    const char c = token[k];
    if (c < '0' || c > '9') {
      return {TokenStatus::not_integer, 0};
    }
    const auto digit = static_cast<std::uint64_t>(c - '0');
    if (too_large || magnitude > (kTokenMagnitudeCap - digit) / 10) {
      too_large = true;
    } else {
      magnitude = magnitude * 10 + digit;
    }
  }
  if (too_large) {
    return {TokenStatus::out_of_range, 0};
  }
  const auto value = static_cast<std::int64_t>(magnitude);
  return {TokenStatus::ok, negative ? -value : value};
}

// A token quoted for a message: at most kTokenDisplayLimit characters, non-printable bytes as
// \xNN.
std::string display(std::string_view token) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string text = "'";
  std::size_t shown = 0;
  for (const char c : token) {
    if (shown == kTokenDisplayLimit) {
      text += "...";
      break;
    }
    const auto byte = static_cast<unsigned char>(c);
    if (byte >= 0x20 && byte < 0x7F) {
      text += c;
    } else {
      text += "\\x";
      text += kHex[byte >> 4];
      text += kHex[byte & 0x0F];
    }
    ++shown;
  }
  text += "'";
  return text;
}

bool is_blank(char c) {
  return c == ' ' || c == '\t';
}

// Tokens of a line (without '\n' and the trailing '\r'), separated by spaces or tabs. No token
// means a blank line.
void split_tokens(std::string_view line, std::vector<std::string_view>& tokens) {
  tokens.clear();
  std::size_t pos = 0;
  while (true) {
    while (pos < line.size() && is_blank(line[pos])) {
      ++pos;
    }
    if (pos == line.size()) {
      return;
    }
    const std::size_t start = pos;
    while (pos < line.size() && !is_blank(line[pos])) {
      ++pos;
    }
    tokens.push_back(line.substr(start, pos - start));
  }
}

// An edge as read, canonical (u < v, 0-based), with its physical line.
struct RawEdge {
  std::int32_t u;
  std::int32_t v;
  std::int32_t w;
  std::int64_t line;
};

class RudyParser {
 public:
  explicit RudyParser(std::string_view source) : source_(source) {}

  Graph parse(std::istream& in);

 private:
  void parse_header(const std::vector<std::string_view>& tokens);
  void parse_edge(const std::vector<std::string_view>& tokens);
  // Error tied to the current line. A repeated edge among the edges read so far comes earlier
  // in the input, so it is reported instead.
  [[noreturn]] void fail_line(const std::string& reason);
  // Sorts the edges read so far by (u, v, line) and throws for the repeated edge whose second
  // occurrence comes first in the input, if any.
  void throw_if_repeated_edge();

  std::string source_;
  std::int64_t line_ = 0;
  std::int64_t n_ = 0;
  std::int64_t m_ = 0;
  std::vector<RawEdge> edges_;
};

Graph RudyParser::parse(std::istream& in) {
  std::string text;
  std::vector<std::string_view> tokens;
  bool have_header = false;
  while (std::getline(in, text)) {
    ++line_;
    std::string_view line(text);
    if (!line.empty() && line.back() == '\r') {
      line.remove_suffix(1);
    }
    split_tokens(line, tokens);
    if (tokens.empty()) {
      continue;
    }
    if (!have_header) {
      parse_header(tokens);
      have_header = true;
      continue;
    }
    if (static_cast<std::int64_t>(edges_.size()) == m_) {
      fail_line("unexpected non-blank line after the M = " + std::to_string(m_) + " edge lines");
    }
    parse_edge(tokens);
  }
  if (in.bad()) {
    throw IoError(source_ + ": read error");
  }
  if (!have_header) {
    throw IoError(source_ + ": no header line \"N M\" (the input has no non-blank line)");
  }
  throw_if_repeated_edge();
  if (static_cast<std::int64_t>(edges_.size()) < m_) {
    throw IoError(source_ + ": expected M = " + std::to_string(m_) + " edge lines, found " +
                  std::to_string(edges_.size()));
  }

  Graph g;
  g.n = n_;
  g.edges.reserve(edges_.size());
  for (const RawEdge& e : edges_) {  // sorted by (u, v) by throw_if_repeated_edge
    g.edges.push_back(Edge{e.u, e.v, e.w});
  }
  return g;
}

void RudyParser::parse_header(const std::vector<std::string_view>& tokens) {
  if (tokens.size() != 2) {
    fail_line("the header must have exactly 2 integer tokens \"N M\", found " +
              std::to_string(tokens.size()));
  }
  const IntegerToken n = parse_integer(tokens[0]);
  const IntegerToken m = parse_integer(tokens[1]);
  if (n.status == TokenStatus::not_integer) {
    fail_line("N: " + display(tokens[0]) + " is not an integer");
  }
  if (m.status == TokenStatus::not_integer) {
    fail_line("M: " + display(tokens[1]) + " is not an integer");
  }
  if (n.status == TokenStatus::out_of_range || n.value < 1 || n.value > kMaxNodes) {
    fail_line("N = " + display(tokens[0]) + " is outside [1, 2147483647]");
  }
  const std::int64_t max_edges = n.value * (n.value - 1) / 2;
  if (m.status == TokenStatus::out_of_range || m.value < 0 || m.value > max_edges) {
    fail_line("M = " + display(tokens[1]) +
              " is outside [0, N(N - 1)/2 = " + std::to_string(max_edges) + "]");
  }
  n_ = n.value;
  m_ = m.value;
}

void RudyParser::parse_edge(const std::vector<std::string_view>& tokens) {
  if (tokens.size() != 3) {
    fail_line("an edge line must have exactly 3 integer tokens \"i j w\", found " +
              std::to_string(tokens.size()));
  }
  static constexpr const char* kNames[3] = {"i", "j", "w"};
  IntegerToken values[3];
  for (std::size_t k = 0; k < 3; ++k) {
    values[k] = parse_integer(tokens[k]);
    if (values[k].status == TokenStatus::not_integer) {
      fail_line(std::string(kNames[k]) + ": " + display(tokens[k]) + " is not an integer");
    }
  }
  for (std::size_t k = 0; k < 2; ++k) {
    if (values[k].status == TokenStatus::out_of_range || values[k].value < 1 ||
        values[k].value > n_) {
      fail_line(std::string(kNames[k]) + " = " + display(tokens[k]) +
                " is outside [1, N = " + std::to_string(n_) + "]");
    }
  }
  const std::int64_t i = values[0].value;
  const std::int64_t j = values[1].value;
  if (i == j) {
    fail_line("self-loop: i = j = " + std::to_string(i));
  }
  const IntegerToken& w = values[2];
  if (w.status == TokenStatus::out_of_range || w.value < -kMaxAbsWeight ||
      w.value > kMaxAbsWeight) {
    fail_line("w = " + display(tokens[2]) + " is outside [-2147483647, 2147483647]");
  }
  if (w.value == 0) {
    fail_line("w = " + display(tokens[2]) + ": zero weights are not allowed");
  }
  edges_.push_back(RawEdge{static_cast<std::int32_t>(std::min(i, j) - 1),
                           static_cast<std::int32_t>(std::max(i, j) - 1),
                           static_cast<std::int32_t>(w.value), line_});
}

void RudyParser::fail_line(const std::string& reason) {
  throw_if_repeated_edge();
  throw IoError(source_ + ":" + std::to_string(line_) + ": " + reason);
}

void RudyParser::throw_if_repeated_edge() {
  std::sort(edges_.begin(), edges_.end(), [](const RawEdge& a, const RawEdge& b) {
    if (a.u != b.u) {
      return a.u < b.u;
    }
    if (a.v != b.v) {
      return a.v < b.v;
    }
    return a.line < b.line;
  });
  const RawEdge* first = nullptr;
  const RawEdge* second = nullptr;
  for (std::size_t start = 0; start < edges_.size();) {
    std::size_t end = start + 1;
    while (end < edges_.size() && edges_[end].u == edges_[start].u &&
           edges_[end].v == edges_[start].v) {
      ++end;
    }
    if (end - start >= 2 && (second == nullptr || edges_[start + 1].line < second->line)) {
      first = &edges_[start];
      second = &edges_[start + 1];
    }
    start = end;
  }
  if (second != nullptr) {
    throw IoError(source_ + ":" + std::to_string(second->line) + ": the edge {" +
                  std::to_string(second->u + 1) + ", " + std::to_string(second->v + 1) +
                  "} repeats the edge given at line " + std::to_string(first->line));
  }
}

}  // namespace

Graph read_rudy(const std::filesystem::path& file) {
  // Binary mode: line endings are handled by the parser, the same way on every platform. A
  // directory can be opened as a stream on some platforms, so it is rejected explicitly.
  std::error_code status;
  if (std::filesystem::is_directory(file, status)) {
    throw IoError(file.string() + ": is a directory, not a rudy file");
  }
  std::ifstream in(file, std::ios::in | std::ios::binary);
  if (!in.is_open()) {
    throw IoError(file.string() + ": cannot open the file for reading");
  }
  return parse_rudy(in, file.string());
}

Graph parse_rudy(std::istream& in, std::string_view source) {
  RudyParser parser(source);
  return parser.parse(in);
}

}  // namespace sbmc
