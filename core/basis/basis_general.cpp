#include "basis_general.hpp"

#include <algorithm>
#include <cctype> 
#include <stdexcept>
#include <string>

#include "core/shell/shell_contract.hpp"
#include "core/basis/types.hpp"
#include "core/basis/get_sto3g_params.hpp"
#include "core/basis/get_4_31g_params.hpp"
#include "core/basis/get_6_31g_params.hpp"
#include "core/basis/get_6_31g_d_params.hpp"
#include "core/basis/get_6_31g_dp_params.hpp"
#include "core/basis/get_def2_svp_params.hpp"
#include "core/basis/get_def2_tzvp_params.hpp"

namespace tlno::core::basis {
namespace {

std::string uppercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c)); 
    });
    return value;
}

RawAtomBasis build_atom_basis(const std::string& symbol,
                                          BasisKind basis_kind) {
    switch (basis_kind) {
        case BasisKind::Sto3g:
            return get_sto3g_params(symbol);
        case BasisKind::G431:
            return get_4_31g_params(symbol);
        case BasisKind::G631:
            return get_6_31g_params(symbol);
        case BasisKind::G631d:
            return get_6_31g_d_params(symbol);
        case BasisKind::G631dp:
            return get_6_31g_dp_params(symbol);
        case BasisKind::Def2SVP:
            return get_def2_svp_params(symbol);
        case BasisKind::Def2TZVP:
            return get_def2_tzvp_params(symbol);
    }
    throw std::runtime_error("Unsupported basis kind.");
}

} // namespace

BasisKind parse_basis_kind(const std::string& basis_name) {
    const std::string upper = uppercase(basis_name);
    if (upper == "STO-3G" || upper == "STO3G") {
        return BasisKind::Sto3g;
    }
    if (upper == "4-31G" || upper == "431G") {
        return BasisKind::G431;
    }
    if (upper == "6-31G" || upper == "631G") {
        return BasisKind::G631;
    }
    if (upper == "6-31G(D)" || upper == "631G(D)" || upper == "631GD" || 
        upper == "6-31G*" || upper == "631G*") {
        return BasisKind::G631d;
    }
    if (upper == "6-31G(D,P)" || upper == "631G(D,P)" || upper == "631GDP" || 
        upper == "6-31G(P,D)" || upper == "631G(P,D)" || upper == "631GPD" || 
        upper == "6-31G**" || upper == "631G**") {
        return BasisKind::G631dp;
    }
    if (upper == "DEF2-SVP" || upper == "DEF2SVP" || 
        upper == "DEF2-DZVP" || upper == "DEF2DZVP") {
        return BasisKind::Def2SVP;
    }
    if (upper == "DEF2-TZVP" || upper == "DEF2TZVP") {
        return BasisKind::Def2TZVP;
    }
    throw std::runtime_error(
        "Unsupported basis set: " + basis_name + 
        ". Valid: sto-3g, 4-31g, 6-31g, 6-31g(d), 6-31g*, " 
        "6-31g(d,p), 6-31g**, def2-svp, def2-tzvp.");
}

BasisBuildResult build_basis_general(
    const tlno::core::molecule::MoleculeSpec& molecule, 
    BasisKind basis_kind) {
    if (molecule.symbols.size() != molecule.coords.size()) {
        throw std::runtime_error("Atom and coordinate count mismatch.");
    }

    BasisBuildResult result;
    result.nuclear_centers = molecule.coords;
    result.nuclear_charges.resize(molecule.symbols.size());
    for (std::size_t atom_index = 0; atom_index < molecule.symbols.size();
         ++atom_index) {
        const RawAtomBasis atom = 
            build_atom_basis(molecule.symbols[atom_index], basis_kind);
        result.electron_count += atom.charge;    
        for (const auto& raw_shell : atom.shells) {
            result.basis.push_back(
                tlno::core::shell::make_shell(
                    raw_shell, molecule.coords[atom_index]));
        }
        result.nuclear_charges[atom_index] = atom.charge;
    }

    result.electron_count -= molecule.charge;
    return result;
}

} // namespace tlno::core::basis
