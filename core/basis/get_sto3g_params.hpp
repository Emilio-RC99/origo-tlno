#ifndef TLNO_CORE_BASIS_GET_STO3G_PARAMS_HPP
#define TLNO_CORE_BASIS_GET_STO3G_PARAMS_HPP

#include <string>

#include "types.hpp"

namespace tlno::core::basis {

/** BSE STO-3G version 1 for H-Ar; tabulated exponents and contraction coefficients.
 *  Basis Set Exchange: https://www.basissetexchange.org/
 *  Source (ORCA export, version 1, H-Ar):
 *  https://www.basissetexchange.org/api/basis/sto-3g/format/orca/?elements=1-18&version=1
 *  make_shell applies primitive and contraction normalization.
 */
RawAtomBasis get_sto3g_params(const std::string& symbol);

} // namespace tlno::core::basis

#endif
