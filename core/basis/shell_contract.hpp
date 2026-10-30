#ifndef TLNO_CORE_BASIS_SHELL_CONTRACT_HPP
#define TLNO_CORE_BASIS_SHELL_CONTRACT_HPP

#include <array>
#include <vector>

#include "types.hpp"

namespace tlno::core::basis {

std::vector<Primitive> normalize_contracted_primitives(
    const std::vector<double>& exponents,
    const std::vector<double>& coefficients,
    const std::array<int, 3>& ang);

Shell make_shell(const RawShell& raw_shell, const std::array<double, 3>& center);

} // namespace tlno::core::basis

#endif
