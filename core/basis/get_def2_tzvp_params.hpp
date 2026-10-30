#ifndef TLNO_CORE_BASIS_GET_DEF2_TZVP_PARAMS_HPP
#define TLNO_CORE_BASIS_GET_DEF2_TZVP_PARAMS_HPP

#include <string>

#include "types.hpp"

namespace tlno::core::basis {

/** BSE def2-TZVP version 1 for H-Ar, with Cartesian 6d/10f components.
 *  https://www.basissetexchange.org/api/basis/def2-tzvp/format/orca/?elements=1-18&version=1
 *  make_shell applies primitive and contraction normalization.
 */
RawAtomBasis get_def2_tzvp_params(const std::string& symbol);

} // namespace tlno::core::basis

#endif
