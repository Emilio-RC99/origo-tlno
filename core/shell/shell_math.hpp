#ifndef TLNO_CORE_SHELL_SHELL_MATH_HPP
#define TLNO_CORE_SHELL_SHELL_MATH_HPP

#include <array>

namespace tlno::core::shell {

double dist2(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs);
double primitive_normalization(double exponent, const std::array<int, 3>& ang);
double contraction_overlap_same_center_metric(
    const std::array<int, 3>& ang, double alpha, double beta);

} // namespace tlno::core::shell

#endif
