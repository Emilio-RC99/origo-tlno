#ifndef TLNO_CORE_BASIS_GET_6_31G_DP_PARAMS_HPP
#define TLNO_CORE_BASIS_GET_6_31G_DP_PARAMS_HPP

#include <string>

#include "types.hpp"

namespace tlno::core::basis {

/** BSE 6-31G** (6-31G(d,p)) version 1 for H-Ar.
 *  Reuse 6-31G(d) for Li-Ar and include H/He p polarization.
 *  Tabulated coefficients; make_shell applies primitive/contraction normalization.
 *  https://www.basissetexchange.org/
 *  https://www.basissetexchange.org/api/basis/6-31g%2A%2A/format/orca/?elements=1-18&version=1
 */
RawAtomBasis get_6_31g_dp_params(const std::string& symbol);

} // namespace tlno::core::basis

#endif
