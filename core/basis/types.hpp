#ifndef TLNO_CORE_BASIS_TYPES_HPP
#define TLNO_CORE_BASIS_TYPES_HPP

#include <array>
#include <cstddef>
#include <string>
#include <vector>

namespace tlno::core::basis {

constexpr double kPi = 3.141592653589793238462643383279502884;

struct Primitive {
    double exponent;
    double coefficient;
};

struct RawShell {
    std::string label;
    std::array<int, 3> ang{};
    std::vector<double> exponents;
    std::vector<double> coefficients;
};

struct RawAtomBasis {
    int charge = 0;
    std::vector<RawShell> shells;
};

struct Shell {
    std::string label;
    std::array<double, 3> center{};
    std::array<int, 3> ang{};
    std::vector<Primitive> primitives;
};

struct BasisBuildResult {
    std::vector<Shell> basis;
    std::vector<int> nuclear_charges;
    std::vector<std::array<double, 3>> nuclear_centers;
    int electron_count = 0;
};

struct DenseMatrix {
    int size = 0;
    std::vector<double> values;

    explicit DenseMatrix(int n = 0)
        : size(n),
          values(static_cast<std::size_t>(n) * static_cast<std::size_t>(n), 0.0) {}
    
    double& operator()(int row, int col) {
        return values[static_cast<std::size_t>(row) * static_cast<std::size_t>(size) + static_cast<std::size_t>(col)];
    }

    double operator()(int row, int col) const {
        return values[static_cast<std::size_t>(row) * static_cast<std::size_t>(size) + static_cast<std::size_t>(col)];
    }
};

} // namespace tlno::core::basis

#endif
