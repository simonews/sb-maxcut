// T11, CPU part: Philox4x32-10 against the known-answer vectors of Random123, the streams of
// sbmc/philox.hpp against reference values, and the two conversions of the words.
//
// The reference values were computed by cldArchitect with two independent Python
// implementations (vectorised NumPy and scalar Python), which agree with each other and
// reproduce the Random123 vectors (tests/kat_vectors of github.com/DEShawResearch/random123,
// commit 9545ff6). The equivalence with cuRAND is checked on the GPU by
// tests/gpu/test_curand_philox.cu.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <string>

#include <sbmc/philox.hpp>

namespace {

using sbmc::RngDomain;
using sbmc::U32x2;
using sbmc::U32x4;

std::string hex32(std::uint32_t value) {
  char text[16];
  std::snprintf(text, sizeof(text), "0x%08X", static_cast<unsigned>(value));
  return text;
}

std::string hex64(std::uint64_t value) {
  char text[24];
  std::snprintf(text, sizeof(text), "0x%016llX", static_cast<unsigned long long>(value));
  return text;
}

std::string text_of(const U32x4& v) {
  return "{" + hex32(v.x) + ", " + hex32(v.y) + ", " + hex32(v.z) + ", " + hex32(v.w) + "}";
}

struct KatVector {
  U32x4 ctr;
  U32x2 key;
  U32x4 expected;
};

// Philox4x32-10 vectors of Random123 (tests/kat_vectors): counter, key, expected result.
constexpr KatVector kKatVectors[] = {
    {{0x00000000u, 0x00000000u, 0x00000000u, 0x00000000u},
     {0x00000000u, 0x00000000u},
     {0x6627e8d5u, 0xe169c58du, 0xbc57ac4cu, 0x9b00dbd8u}},
    {{0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu},
     {0xffffffffu, 0xffffffffu},
     {0x408f276du, 0x41c83b0eu, 0xa20bc7c6u, 0x6d5451fdu}},
    {{0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u},
     {0xa4093822u, 0x299f31d0u},
     {0xd16cfe09u, 0x94fdccebu, 0x5001e420u, 0x24126ea1u}},
};

// Compile-time evaluation: philox4x32_10 and the stream functions are constexpr.
static_assert(sbmc::philox4x32_10(kKatVectors[0].ctr, kKatVectors[0].key) ==
                  kKatVectors[0].expected,
              "philox4x32_10 must reproduce the first Random123 vector at compile time");
static_assert(sbmc::stream_word(0x0, RngDomain::couplings, 0, 0) == 0x2DCE73E5u,
              "stream_word must be usable in constant expressions");
static_assert(std::bit_cast<std::uint32_t>(sbmc::position_from_word(0)) == 0xBDCCCCCDu,
              "position_from_word(0) must be -0.1f at compile time");

struct WordCase {
  std::uint64_t seed;
  std::uint32_t domain;
  std::uint32_t stream;
  std::uint64_t k;
  std::uint32_t expected;
};

// seed, domain, stream, k -> word.
constexpr WordCase kWordCases[] = {
    {0x0u, 1, 0, 0, 0x2DCE73E5u},
    {0x0u, 1, 0, 1, 0x1348E23Fu},
    {0x0u, 1, 0, 2, 0xFCF8E0ECu},
    {0x0u, 1, 0, 3, 0xA287AADBu},
    {0x0u, 1, 0, 4, 0xE9535BE3u},
    {0x1u, 2, 0, 0, 0x2FE424F7u},
    {0x1u, 2, 7, 5, 0xAF2D0AFFu},
    {0x2Au, 1, 3, 17179869180u, 0x7B3B583Au},  // k = 2^34 - 4
    {0x2Au, 1, 3, 17179869183u, 0x5529100Bu},  // k = 2^34 - 1
    {0x2Au, 1, 3, 17179869184u, 0xD16D44D0u},  // k = 2^34
    {0xFFFFFFFFFFFFFFFFu, 4, 4294967295u, 18446744073709551615u, 0x48045D74u},
    // stream 0x13198A2E, k 0x0370734412345678
    {0x243F6A8885A308D3u, 3, 320440878u, 247824715705767544u, 0x869E3728u},
    {0x5u, 1, 9, 11, 0x32BD9E61u},
    {0x5u, 2, 9, 11, 0x78480257u},
    {0x5u, 3, 9, 11, 0x231F2CD9u},
    {0x5u, 4, 9, 11, 0x57BAA68Fu},
    {0x100000000u, 1, 0, 0, 0x96B27E8Cu},
};

}  // namespace

TEST_CASE("T11: philox4x32_10 reproduces the known-answer vectors of Random123") {
  for (const KatVector& v : kKatVectors) {
    const U32x4 result = sbmc::philox4x32_10(v.ctr, v.key);
    const std::string description = "ctr " + text_of(v.ctr) + ", key {" + hex32(v.key.x) + ", " +
                                    hex32(v.key.y) + "}: expected " + text_of(v.expected) +
                                    ", got " + text_of(result);
    INFO(description);
    CHECK(result.x == v.expected.x);
    CHECK(result.y == v.expected.y);
    CHECK(result.z == v.expected.z);
    CHECK(result.w == v.expected.w);
    CHECK(result == v.expected);
  }
}

TEST_CASE("T11: stream_word reproduces the reference values") {
  for (const WordCase& c : kWordCases) {
    const std::uint32_t word =
        sbmc::stream_word(c.seed, static_cast<RngDomain>(c.domain), c.stream, c.k);
    const std::string description = "seed " + hex64(c.seed) + ", domain " +
                                    std::to_string(c.domain) + ", stream " +
                                    std::to_string(c.stream) + ", k " + std::to_string(c.k) +
                                    ": expected " + hex32(c.expected) + ", got " + hex32(word);
    INFO(description);
    CHECK(word == c.expected);
  }
}

TEST_CASE("T11: stream_block holds the words 4b, ..., 4b + 3 of the stream") {
  constexpr std::uint64_t kSeed = 0x2A;
  constexpr std::uint32_t kStream = 3;
  constexpr std::uint64_t kBlock = 0xFFFFFFFFu;
  const U32x4 block = sbmc::stream_block(kSeed, RngDomain::couplings, kStream, kBlock);
  const U32x4 expected{0x7B3B583Au, 0xDBFEE0F3u, 0x7322066Cu, 0x5529100Bu};
  const std::string description = "expected " + text_of(expected) + ", got " + text_of(block);
  INFO(description);
  CHECK(block == expected);

  const std::uint64_t k0 = 4 * kBlock;
  CHECK(sbmc::stream_word(kSeed, RngDomain::couplings, kStream, k0 + 0) == block.x);
  CHECK(sbmc::stream_word(kSeed, RngDomain::couplings, kStream, k0 + 1) == block.y);
  CHECK(sbmc::stream_word(kSeed, RngDomain::couplings, kStream, k0 + 2) == block.z);
  CHECK(sbmc::stream_word(kSeed, RngDomain::couplings, kStream, k0 + 3) == block.w);
}

TEST_CASE("T11: coupling_from_word is +1 for a top bit 0 and -1 for a top bit 1") {
  CHECK(static_cast<int>(sbmc::coupling_from_word(0x00000000u)) == 1);
  CHECK(static_cast<int>(sbmc::coupling_from_word(0x7FFFFFFFu)) == 1);
  CHECK(static_cast<int>(sbmc::coupling_from_word(0x80000000u)) == -1);
  CHECK(static_cast<int>(sbmc::coupling_from_word(0xFFFFFFFFu)) == -1);
}

TEST_CASE("T11: position_from_word gives the reference bits") {
  static_assert(0x1.99999ap-27f == 0.1f / 8388608.0f, "S = 0.1f / 2^23");
  static_assert(std::bit_cast<std::uint32_t>(0x1.99999ap-27f) == 0x324CCCCDu, "bits of S");

  struct PositionCase {
    std::uint32_t word;
    std::uint32_t bits;
  };
  constexpr PositionCase kCases[] = {
      {0x00000000u, 0xBDCCCCCDu},                                                          // -0.1f
      {0x00000100u, 0xBDCCCCCBu}, {0x7FFFFFFFu, 0xB24CCCCDu}, {0x80000000u, 0x00000000u},  // +0.0f
      {0x800000FFu, 0x00000000u}, {0xFFFFFF00u, 0x3DCCCCCBu}, {0xFFFFFFFFu, 0x3DCCCCCBu},
  };
  for (const PositionCase& c : kCases) {
    const std::uint32_t bits = std::bit_cast<std::uint32_t>(sbmc::position_from_word(c.word));
    const std::string description =
        "word " + hex32(c.word) + ": expected bits " + hex32(c.bits) + ", got " + hex32(bits);
    INFO(description);
    CHECK(bits == c.bits);
  }
  CHECK(sbmc::position_from_word(0x00000000u) == -0.1f);
}
