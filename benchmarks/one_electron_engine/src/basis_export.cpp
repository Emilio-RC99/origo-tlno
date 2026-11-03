// Benchmark-only access to the parameters actually compiled into this library.
// The Python oracle must not substitute a different revision of a named basis.
#include "core/basis/basis_general.hpp"
#include "core/basis/get_sto3g_params.hpp"
#include "core/basis/get_4_31g_params.hpp"
#include "core/basis/get_6_31g_params.hpp"
#include "core/basis/get_6_31g_d_params.hpp"
#include "core/basis/get_6_31g_dp_params.hpp"
#include "core/basis/get_def2_svp_params.hpp"
#include "core/basis/get_def2_tzvp_params.hpp"

#include <array>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

namespace basis = tlno::core::basis;

basis::RawAtomBasis raw_basis(basis::BasisKind kind, const std::string& symbol) {
    switch (kind) {
        case basis::BasisKind::Sto3g: return basis::get_sto3g_params(symbol);
        case basis::BasisKind::G431: return basis::get_4_31g_params(symbol);
        case basis::BasisKind::G631: return basis::get_6_31g_params(symbol);
        case basis::BasisKind::G631d: return basis::get_6_31g_d_params(symbol);
        case basis::BasisKind::G631dp: return basis::get_6_31g_dp_params(symbol);
        case basis::BasisKind::Def2SVP: return basis::get_def2_svp_params(symbol);
        case basis::BasisKind::Def2TZVP: return basis::get_def2_tzvp_params(symbol);
    }
    throw std::runtime_error("Unknown basis kind.");
}

} // namespace

extern "C" {

// A returned string belongs to this thread and remains valid until the next
// call on the same thread. Unsupported inputs return nullptr. Shell order and
// repeated contractions are preserved; only each complete Cartesian component
// chunk is collapsed into its shared radial exponent/coefficient list.
const char* tlno_onee_basis_json(const char* basis_name, const char* symbol) {
    thread_local std::string result;
    try {
        if (!basis_name || !symbol) return nullptr;
        const auto atom = raw_basis(basis::parse_basis_kind(basis_name), symbol);
        std::ostringstream json;
        json << std::setprecision(17) << "{\"charge\":" << atom.charge
             << ",\"shells\":[";
        bool first = true;
        for (std::size_t i = 0; i < atom.shells.size();) {
            const auto& radial = atom.shells[i];
            const int L = radial.ang[0] + radial.ang[1] + radial.ang[2];
            if (L < 0 || radial.exponents.empty() ||
                radial.exponents.size() != radial.coefficients.size()) {
                throw std::runtime_error("Invalid raw radial shell.");
            }
            const std::size_t count = static_cast<std::size_t>((L + 1) * (L + 2) / 2);
            if (count > atom.shells.size() - i) {
                throw std::runtime_error("Incomplete Cartesian shell.");
            }
            std::size_t component = 0;
            for (int x = L; x >= 0; --x) {
                for (int y = L - x; y >= 0; --y, ++component) {
                    const auto& shell = atom.shells[i + component];
                    if (shell.ang != std::array<int, 3>{x, y, L - x - y} ||
                        shell.exponents != radial.exponents ||
                        shell.coefficients != radial.coefficients) {
                        throw std::runtime_error("Noncanonical Cartesian shell.");
                    }
                }
            }
            if (!first) json << ',';
            first = false;
            json << '[' << L;
            for (std::size_t primitive = 0; primitive < radial.exponents.size(); ++primitive) {
                json << ",[" << radial.exponents[primitive] << ','
                     << radial.coefficients[primitive] << ']';
            }
            json << ']';
            i += count;
        }
        json << "]}";
        result = json.str();
        return result.c_str();
    } catch (...) {
        result.clear();
        return nullptr;
    }
}

} // extern "C"
