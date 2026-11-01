#ifndef TLNO_ONE_ELECTRON_ONEE_INTEGRALS_HPP
#define TLNO_ONE_ELECTRON_ONEE_INTEGRALS_HPP

#include "../../core/basis/types.hpp"
#include "../../core/shell/shell_math.hpp"

namespace tlno::integrals::onee {

// Shared data types and math live in tlno::core; the one-electron engine is
// built on top of them.


struct ShellGroup {
    std::array<double, 3> center{};
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
    // Per-primitive-pair SoA sections (length n_prim_a * n_prim_b each), all
    // carved out of ONE arena allocation: 18 separate vectors caused malloc
    // contention when plans are built in parallel and dominated plan time.
    // Each section is padded to a multiple of 8 doubles (64-byte lanes).
    // The named pointers stay valid under move because moving `storage`
    // transfers the same heap block.  Copying is deleted (would dangle).
    std::vector<double> storage;
    double* p = nullptr;             // alpha + beta
    double* inv2p = nullptr;         // 1 / (2 p)
    double* chi = nullptr;           // alpha * beta / p
    double* sss = nullptr;           // (pi/p)^1.5 * exp(-chi AB2)
    double* expmChiAB2 = nullptr;    // exp(-chi AB2)
    double* alpha = nullptr;         // bra exponent
    double* beta = nullptr;          // ket exponent
    double* PAx = nullptr;           // product centre minus A (overlap VRR)
    double* PAy = nullptr;
    double* PAz = nullptr;
    double* Px = nullptr;            // product centre
    double* Py = nullptr;
    double* Pz = nullptr;
    double* e0x = nullptr;           // exp(-chi q_axis^2) Hermite seeds
    double* e0y = nullptr;
    double* e0z = nullptr;
    double* neg2pi_over_p = nullptr; // -2 pi / p (nuclear prefactor)
    // Per-primitive-pair nuclear screening bound precomputed in the plan:
    // screen[k] = |neg2pi_over_p[k]| * exp(-chi AB^2) * max_a|coeff_a| * max_b|coeff_b|.
    // For s-s components, screen[k] * Z_sum bounds the monopole contribution
    // because F_0 <= 1. Higher-L components share the exp(-chi AB^2) seed, but
    // this scalar is not by itself a rigorous bound for every angular component;
    // the production threshold is therefore validated against no-screen matrices.
    double* screen = nullptr;

    // Optional immutable plan-owned Hermite projectors, in primitive/axis/
    // (bra,ket,t) order. Empty means compute into thread-local scratch.
    // Only pair geometry/exponents enter these coefficients, never charges or
    // screening decisions. Moving a plan preserves the cache's ownership.
    std::vector<double> nuclear_hermite;

    ShellPairPrim() = default;
    ShellPairPrim(ShellPairPrim&&) = default;
    ShellPairPrim& operator = (ShellPairPrim&&) = default;
    ShellPairPrim(const ShellPairPrim&) = delete;
    ShellPairPrim& operator = (const ShellPairPrim&) = delete;
};

ShellPairPrim build_shell_pair_prim(const ShellGroup& group_a, const ShellGroup& group_b);
void cache_shell_pair_nucleus_hermite(
    const ShellGroup& group_a, const ShellGroup& group_b, ShellPairPrim& spp);

void contracted_shell_block_overlap_kinetic(
    const ShellGroup& group_a, 
    const ShellGroup& group_b, 
    const ShellPairPrim& spp, 
    double* overlap_block, 
    double* kinetic_block, 
    double min_abs_overlap);

void contracted_shell_block_nuclear(
    const ShellGroup& group_a, 
    const ShellGroup& group_b, 
    const ShellPairPrim& spp, 
    double* nuclear_block, 
    const std::vector<int>& nuclear_charges, 
    const std::vector<std::array<double, 3>>& nuclear_centers);

// Compiled default for the nuclear primitive-pair screening tolerance: the
// single source of truth referenced both by the runtime variable's initializer
// and by the set_onee_screening_enabled reset.
constexpr double kNuclearScreenEps = 1.0e-15;

// Runtime control of the nuclear primitive-pair screening tolerance (default
// kNuclearScreenEps).  Set to 0 to disable nuclear screening (benchmark /
// diagnostic use).
void set_onee_nuclear_screen_eps(double eps);
double get_onee_nuclear_screen_eps();

} // namespace tlno::integrals::onee

#endif
