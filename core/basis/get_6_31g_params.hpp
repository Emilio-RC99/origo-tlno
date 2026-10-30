#ifndef TLNO_CORE_BASIS_GET_6_31G_PARAMS_HPP
#define TLNO_CORE_BASIS_GET_6_31G_PARAMS_HPP

#include <string>

#include "types.hpp"

namespace tlno::core::basis {

/** BSE 6-31G version 1 for H-Ar (no polarization).
 *  Tabulated exponents and contraction coefficients; make_shell normalizes them.
 *  https://www.basissetexchange.org/
 *  https://www.basissetexchange.org/api/basis/6-31g/format/orca/?elements=1-18&version=1
 */
RawAtomBasis get_6_31g_params(const std::string& symbol);

} // namespace tlno::core::basis

#endif
