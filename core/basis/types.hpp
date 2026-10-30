#ifndef TLNO_CORE_BASIS_TYPES_HPP
#define TLNO_CORE_BASIS_TYPES_HPP

#include <array>
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
    int electron_count = 0;
};

} // namespace tlno::core::basis

#endif
