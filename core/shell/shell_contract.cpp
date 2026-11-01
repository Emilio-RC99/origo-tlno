#include "shell_contract.hpp"
#include "shell_math.hpp"

#include <cmath>
#include <stdexcept>

namespace tlno::core::shell {

std::vector<basis::Primitive> normalize_contracted_primitives(
    const std::vector<double>& exponents,
    const std::vector<double>& coefficients, 
    const std::array<int, 3>& ang) {
    if (exponents.size() != coefficients.size()) {
        throw std::runtime_error("Basis exponents and coefficients size mismatch.");
    }

    if (ang[0] < 0 || ang[1] < 0 || ang[2] < 0) {
        throw std::runtime_error("Shell angular momenta must be nonnegative.");
    }

    std::vector<basis::Primitive> primitives;
    primitives.reserve(exponents.size());
    for (std::size_t idx = 0; idx < exponents.size(); ++idx) {
        primitives.push_back({exponents[idx], coefficients[idx] * primitive_normalization(exponents[idx], ang)});
    }

    double norm2 = 0.0;
    for (const auto& lhs : primitives) {
        for (const auto& rhs : primitives) {
            norm2 += lhs.coefficient * rhs.coefficient * 
                     contraction_overlap_same_center_metric(ang, lhs.exponent, rhs.exponent);
        }
    }
    const double contraction_scale = 1.0 / std::sqrt(norm2);
    for (auto& primitive : primitives) {
        primitive.coefficient *= contraction_scale;
    }
    return primitives;
}

basis::Shell make_shell(const basis::RawShell& raw_shell, const std::array<double, 3>& center) {
    basis::Shell shell;
    shell.label = raw_shell.label;
    shell.center = center;
    shell.ang = raw_shell.ang;
    shell.primitives = normalize_contracted_primitives(raw_shell.exponents, raw_shell.coefficients, raw_shell.ang);
    return shell;
}

} // namespace tlno::core::shell
