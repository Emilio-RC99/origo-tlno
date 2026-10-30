#ifndef TLNO_CORE_BASIS_GET_6_31G_D_PARAMS_HPP
#define TLNO_CORE_BASIS_GET_6_31G_D_PARAMS_HPP

#include <string>

#include "types.hpp"

namespace tlno::core::basis {

/** BSE 6-31G* (6-31G(d)) version 1 for H-Ar.
 *  Li-Ar include six Cartesian d functions; H/He use unpolarized 6-31G.
 *  Tabulated coefficients; make_shell applies primitive/contraction normalization.
 *  https://www.basissetexchange.org/
 *  https://www.basissetexchange.org/api/basis/6-31g%2A/format/orca/?elements=1-18&version=1
 */
RawAtomBasis get_6_31g_d_params(const std::string& symbol);

} // namespace tlno::core::basis

#endif
