#ifndef TLNO_INTEGRALS_TWO_ELECTRON_SIMD_POLICY_HPP
#define TLNO_INTEGRALS_TWO_ELECTRON_SIMD_POLICY_HPP

#include "core/simd/packet.hpp"

// The Rys engine has two quartet routes: a legacy route specialized for four
// lanes (AVX2) and a width-general route used at every other width. Building
// with TLNO_TWOE_PORTABLE=1 forces the general route at four lanes too.
#ifndef TLNO_TWOE_PORTABLE
#define TLNO_TWOE_PORTABLE 0
#endif

namespace tlno::integrals::twoe {

inline constexpr bool portable_kernel =
    tlno::core::simd::lanes != 4 || TLNO_TWOE_PORTABLE;

} // namespace tlno::integrals::twoe

#endif
