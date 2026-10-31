#include "shell_contract.hpp"

#include <cmath>
#include <stdexcept>

namespace tlno::core::basis {
namespace {

double square(double value) {
    return value * value;
}

double distance_squared(const std::array<double, 3>& lhs, 
                        const std::array<double, 3>& rhs) {
    return square(lhs[0] - rhs[0]) + square(lhs[1] - rhs[1]) +
           square(lhs[2] - rhs[2]);
}

double double_factorial(int value) {
    double result = 1.0;
    for (int current = value; current > 0; current -= 2) {
        result *= static_cast<double>(current);
    }
    return result;
}

double primitive_normalization(double exponent,
                               const std::array<int, 3>& ang) {
    const int lx = ang[0];
    const int ly = ang[1];
    const int lz = ang[2];
    const int total_l = lx + ly + lz;
    const double angular_denominator = 
        double_factorial(2 * lx - 1) * 
        double_factorial(2 * ly - 1) * 
        double_factorial(2 * lz - 1);
    const double numerator = 
        std::pow(2.0, 2.0 * static_cast<double>(total_l) + 1.5) * 
        std::pow(exponent, static_cast<double>(total_l) + 1.5);
    const double denominator = std::pow(kPi, 1.5) * angular_denominator;
    return std::sqrt(numerator / denominator);
}

double overlap_1d(int left_l,
                  int right_l, 
                  double left_center, 
                  double right_center, 
                  double left_exponent, 
                  double right_exponent) {
    if (left_l < 0 || right_l < 0) {
        throw std::runtime_error(
            "Shell angular momenta must be nonnegative.");
    }
    const int total_l = left_l + right_l;
    if (total_l == 0) return 1.0;

    const double exponent_sum = left_exponent + right_exponent;
    const double product_minus_left = 
        (left_exponent * left_center + right_exponent * right_center) / 
            exponent_sum - 
        left_center;
    const double center_difference = left_center - right_center;

    std::vector<double> bra(static_cast<std::size_t>(total_l + 1), 0.0);
    bra[0] = 1.0;
    if (total_l >= 1) bra[1] = product_minus_left;
    for (int n = 1; n < total_l; ++n) {
        bra[n + 1] = 
            product_minus_left * bra[n] + 
            (static_cast<double>(n) / (2.0 * exponent_sum)) * bra[n - 1];
    }
    if (right_l == 0) return bra[left_l];

    const int column_count = right_l + 1;
    std::vector<double> overlap(
        static_cast<std::size_t>(total_l + 1) * 
            static_cast<std::size_t>(column_count), 
        0.0);
    for (int i = 0; i <= total_l; ++i) {
        overlap[static_cast<std::size_t>(i * column_count)] = bra[i];
    }
    for (int j = 0; j < right_l; ++j) {
        const int max_i = total_l - j - 1;
        for (int i = 0; i <= max_i; ++i) {
            overlap[static_cast<std::size_t>(i * column_count + j + 1)] = 
                overlap[static_cast<std::size_t>((i + 1) * column_count + j)] + 
                center_difference * 
                    overlap[static_cast<std::size_t>(i * column_count + j)];
        }
    }
    return overlap[
        static_cast<std::size_t>(left_l * column_count + right_l)];
}

double contraction_overlap_same_center_metric(
    const std::array<int, 3>& ang, double alpha, double beta) {
    static const std::array<double, 3> origin = {0.0, 0.0, 0.0};
    const double p = alpha + beta;
    const double chi = alpha * beta / p;
    const double sss = 
        std::pow(kPi / p, 1.5) * 
        std::exp(- chi * distance_squared(origin, origin));
    const double fx = 
        overlap_1d(ang[0], ang[0], origin[0], origin[0], alpha, beta);
    const double fy = 
        overlap_1d(ang[1], ang[1], origin[1], origin[1], alpha, beta);
    const double fz = 
        overlap_1d(ang[2], ang[2], origin[2], origin[2], alpha, beta);
    return sss * fx * fy * fz;
}

} // namespace

std::vector<Primitive> normalize_contracted_primitives(
    const std::vector<double>& exponents,
    const std::vector<double>& coefficients, 
    const std::array<int, 3>& ang) {
    if (exponents.size() != coefficients.size()) {
        throw std::runtime_error("Basis exponents and coefficients size mismatch.");
    }

    std::vector<Primitive> primitives;
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

Shell make_shell(const RawShell& raw_shell, const std::array<double, 3>& center) {
    Shell shell;
    shell.label = raw_shell.label;
    shell.center = center;
    shell.ang = raw_shell.ang;
    shell.primitives = normalize_contracted_primitives(raw_shell.exponents, raw_shell.coefficients, raw_shell.ang);
    return shell;
}

} // namespace tlno::core::basis
