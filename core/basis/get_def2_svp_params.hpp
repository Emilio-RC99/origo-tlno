#ifndef TLNO_CORE_BASIS_GET_DEF2_SVP_PARAMS_HPP
#define TLNO_CORE_BASIS_GET_DEF2_SVP_PARAMS_HPP

#include <string>

#include "types.hpp"

namespace tlno::core::basis {

/** BSE def2-SVP version 1 for H-Ar, expanded into Cartesian components.
 *  Tabulated coefficients; make_shell applies primitive/contraction normalization.
 *  https://www.basissetexchange.org/
 *  https://www.basissetexchange.org/api/basis/def2-svp/format/orca/?elements=1-18&version=1
 */
RawAtomBasis get_def2_svp_params(const std::string& symbol);

} // namespace tlno::core::basis

#endif
