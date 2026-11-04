#ifndef TLNO_CORE_MATH_BINARY64_HPP
#define TLNO_CORE_MATH_BINARY64_HPP

// Production builds use -ffast-math.  Under that contract GCC may fold
// std::isfinite(value) to true even when value came from strtod, a file, or a
// caller that supplied an IEEE-754 infinity/NaN.  Validation therefore has to
// inspect the binary64 representation without a floating-point classification
// operation.  Keep this helper small and header-only so every engine entry
// point uses exactly the same predicate.

#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

namespace tlno::core::math {

inline bool is_finite_binary64(double value) noexcept {
    static_assert(sizeof(double) == sizeof(std::uint64_t),
                  "binary64 validation requires a 64-bit double");
    static_assert(std::numeric_limits<double>::is_iec559,
                  "binary64 validation requires IEC 60559/IEEE-754 double");
    static_assert(std::numeric_limits<double>::radix == 2,
                  "binary64 validation requires radix-2 double");
    static_assert(std::numeric_limits<double>::digits == 53,
                  "binary64 validation requires a 53-bit significand");
    static_assert(std::numeric_limits<double>::max_exponent == 1024,
                  "binary64 validation requires an 11-bit exponent");
    static_assert(std::numeric_limits<std::uint64_t>::digits == 64,
                  "binary64 validation requires a 64-bit unsigned integer");
    static_assert(std::is_trivially_copyable<double>::value,
                  "binary64 validation requires bit-copyable double");

    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    constexpr std::uint64_t exponent_mask = UINT64_C(0x7ff0000000000000);
    return (bits & exponent_mask) != exponent_mask;
}

} // namespace tlno::core::math

#endif
