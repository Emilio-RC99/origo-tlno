#ifndef TLNO_CORE_BASIS_GENERAL_HPP
#define TLNO_CORE_BASIS_GENERAL_HPP

#include <string>

#include "types.hpp"
#include "../molecule/molecule_spec.hpp"

namespace tlno::core::basis {

enum class BasisKind {
    Sto3g,
    G431,
    G631,
    G631d,
    G631dp,
    Def2SVP,
    Def2TZVP,
};

BasisKind parse_basis_kind(const std::string& basis_name);

BasisBuildResult build_basis_general(
    const tlno::core::molecule::MoleculeSpec& molecule,
    BasisKind basis_kind);

} // namespace tlno::core::basis

#endif
