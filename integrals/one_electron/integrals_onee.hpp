#ifndef TLNO_ONE_ELECTRON_INTEGRALS_ONEE_HPP
#define TLNO_ONE_ELECTRON_INTEGRALS_ONEE_HPP

#include "../../core/basis/types.hpp"
#include "../../core/shell/shell_math.hpp"

namespace tlno::integrals::oee {

// Shared data types and math live in tlno::core; the one-electron engine 
// is built on top of them.


struct ShellGroup {
    std::array<double, 3> cneter{};
    int L = 0;
    std::vector<int> basis_indices;
    std::vector<std::array<int, 3>> ang;
    std::vector<double> exponents;
    std::vector<std::vector<double>> coeffs;
};

// Struct-of-arrays primitive-pair data for one group pair, precomputed once in
// the reusable plan. Per-primitive-pair transcendentals and divisions live here
// so repeated evaluation is primarily arithmetic and memory operations.
// Index convention: k = ia_prim * n_prim_b + ib_prim.

struct ShellPairPrim {
    int n_prim_a = 0;
    int n_prim_b = 0;
    // Group-level geometry (same for every primitive pair).
    double ABx = 0.0, ABy = 0.0, ABz = 0.0, AB2 = 0.0;
      
};
} // namespace tlno::inegrlas

#endif