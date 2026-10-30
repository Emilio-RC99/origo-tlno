#include "get_6_31g_dp_params.hpp"
#include "get_6_31g_d_params.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace tlno::core::basis {
namespace {

std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

} // namespace

// BSE 6-31G** (6-31G(d,p)) version 1; tabulated parameters for H-Ar.
// Source: https://www.basissetexchange.org/api/basis/6-31g%2A%2A/format/orca/?elements=1-18&version=1
RawAtomBasis get_6_31g_dp_params(const std::string& symbol) {
    const std::string upper = uppercase(symbol);

    if (upper != "H" && upper != "HE") {
        return get_6_31g_d_params(symbol);
    }

    RawAtomBasis atom;
    auto add_s = [&](const std::string& label, const std::vector<double>& exps, const std::vector<double>& coeffs) {
        atom.shells.push_back({label, {0, 0, 0}, exps, coeffs});
    };
    auto add_p_block = [&](double exponent) {
        const std::vector<double> exps = {exponent};
        const std::vector<double> coeffs = {1.000000000};
        atom.shells.push_back({"p_x", {1, 0, 0}, exps, coeffs});
        atom.shells.push_back({"p_y", {0, 1, 0}, exps, coeffs});
        atom.shells.push_back({"p_z", {0, 0, 1}, exps, coeffs});
    };

    if (upper == "H") {
        atom.charge = 1;
        add_s("1s", {18.73113696, 2.825394365, 0.6401216923},
              {0.03349460434, 0.2347269535, 0.8137573261});
        add_s("1s", {0.1612777588}, {1.0});
        add_p_block(1.1);
        return atom;
    }

    atom.charge = 2;
    add_s("1s", {38.42163400, 5.778030000, 1.241774000}, {0.04013973935, 0.2612460970, 0.7931846246});
    add_s("1s", {0.2979640000}, {1.0});
    add_p_block(1.1);
    return atom;
}

} // namespace tlno::core::basis
