#include "core/shell/shell_math.hpp"
#include "core/basis/types.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tlno::core::shell {
namespace {

double double_factorial(int value) {
    double result = 1.0;
    for (int current = value; current > 0; current -= 2) {
        result *= static_cast<double>(current);
    }
    return result;
}

void build_shape_table_1d(int la_max, int lb_max, double PA, double AB, double p, 
                           double* out) {
    const int Ltot_max = la_max + lb_max;
    const int ncol = lb_max + 1;
    out[0] = 1.0;
    if (Ltot_max >= 1) out[static_cast<std::size_t>(ncol)] = PA;
    for (int n = 1; n < Ltot_max; ++n) {
        out[static_cast<std::size_t>(n + 1) * static_cast<std::size_t>(ncol)] =
            PA * out[static_cast<std::size_t>(n) * static_cast<std::size_t>(ncol)] +
            (static_cast<double>(n) / (2.0 * p)) *
                out[static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(ncol)];
    }
    for (int ib = 0; ib < lb_max; ++ib) {
        const int imax = Ltot_max - ib - 1;
        for (int ia = 0; ia <= imax; ++ia) {
            out[static_cast<std::size_t>(ia) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(ib + 1)] =
                out[static_cast<std::size_t>(ia + 1) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(ib)] +
                AB * out[static_cast<std::size_t>(ia) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(ib)];
        }
    }
}

} // namespace

double dist2(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
    const double dx = lhs[0] - rhs[0];
    const double dy = lhs[1] - rhs[1];
    const double dz = lhs[2] - rhs[2];
    return dx * dx + dy * dy + dz * dz;
}

double primitive_normalization(double exponent, const std::array<int, 3>& ang) {
    const int lx = ang[0];
    const int ly = ang[1];
    const int lz = ang[2];
    const int L = lx + ly + lz;
    const double df = double_factorial(2 * lx - 1) * 
                      double_factorial(2 * ly - 1) *
                      double_factorial(2 * lz - 1);
    const double num = std::pow(2.0, 2.0 * static_cast<double>(L) + 1.5) * 
                       std::pow(exponent, static_cast<double>(L) + 1.5);
    const double den = std::pow(basis::kPi, 1.5) * df;
    return std::sqrt(num / den);
}

double contraction_overlap_same_center_metric(
    const std::array<int, 3>& ang, double alpha, double beta) {
    const double p = alpha + beta;
    double factors[3] = {1.0, 1.0, 1.0};
    for (int axis = 0; axis < 3; ++axis) {
        const int l = ang[static_cast<std::size_t>(axis)];
        std::vector<double> table(
            static_cast<std::size_t>(2 * l + 1) *
            static_cast<std::size_t>(l + 1));
        build_shape_table_1d(l, l, 0.0, 0.0, p, table.data());
        factors[axis] = table[static_cast<std::size_t>(l * (l + 1) + l)];
    }
    return std::pow(basis::kPi / p, 1.5) * factors[0] * factors[1] * factors[2];
}

} // namespace tlno::core::shell
