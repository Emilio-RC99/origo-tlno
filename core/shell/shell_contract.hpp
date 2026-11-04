#ifndef TLNO_CORE_BASIS_SHELL_CONTRACT_HPP
#define TLNO_CORE_BASIS_SHELL_CONTRACT_HPP

#include <array>
#include <vector>

#include "core/basis/types.hpp"

namespace tlno::core::shell {

std::vector<basis::Primitive> normalize_contracted_primitives(
    const std::vector<double>& exponents,
    const std::vector<double>& coefficients,
    const std::array<int, 3>& ang);

basis::Shell make_shell(
    const basis::RawShell& raw_shell, const std::array<double, 3>& center);

} // namespace tlno::core::shell

#endif
