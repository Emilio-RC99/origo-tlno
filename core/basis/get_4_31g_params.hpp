#ifndef TLNO_CORE_BASIS_GET_4_31G_PARAMS_HPP
#define TLNO_CORE_BASIS_GET_4_31G_PARAMS_HPP

#include <string>

#include "types.hpp"

namespace tlno::core::basis {

/** BSE 4-31G version 1 for H, He, B, C, N, O, F, Ne, P, S, Cl.
 *  Basis Set Exchange: https://www.basissetexchange.org/
 *  https://www.basissetexchange.org/api/basis/4-31g/format/orca/?version=1
 *  Other elements raise std::runtime_error. make_shell applies normalization.
 */
RawAtomBasis get_4_31g_params(const std::string& symbol);

} // namespace tlno::core::basis

#endif
