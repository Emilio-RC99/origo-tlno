#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_DENSITY_CONTEXT_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_DENSITY_CONTEXT_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

// Density-screening context. It travels through TLS because the J/K endpoint
// produces it (it is the one that knows P) and the sweep consumes it, several
// layers below, without the kernel signatures mentioning it. Null = off, and
// then the sweep generates exactly the previous code.
// The bound is the one used by CVHF's incremental Fock screening:
//
//   cap = max( 4*max(rho_ab, rho_cd) , max(rho_ac, rho_ad, rho_bc, rho_bd) )
//
// The Coulomb term uses density at the SHELL-PAIR level (rho_ab = max|P|
// over the AO pairs of the group pair), which is tight; only the four crossed
// exchange terms use per-GROUP-pair maxima. The 4.0 belongs to the Coulomb
// term and there is NO additional safety factor: the criterion is
// q * cap < thresh_den.
struct RysDensityScreenContext {
    const double* shell_rho = nullptr;             // per pair group
    // WHAT/WHY: O(N_pair_groups) metadata to carry to the scheduler the
    // factorizable upper bound computed once for each P/dP.
    const double* pair_group_density_upper = nullptr;
    const double* group_pair_density_max = nullptr; // ng x ng
    int ng = 0;
    const std::uint32_t* pair_group_left = nullptr;
    const std::uint32_t* pair_group_right = nullptr;
    double density_break = 0.0;
    bool j_only = false;
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    bool k_only = false;
#endif
};
thread_local const RysDensityScreenContext* g_rys_density_screen = nullptr;

class RysDensityScreenGuard {
public:
    explicit RysDensityScreenGuard(
        const RysDensityScreenContext* context) noexcept
        : previous_(g_rys_density_screen) {
        g_rys_density_screen = context;
    }
    ~RysDensityScreenGuard() noexcept { g_rys_density_screen = previous_; }
    RysDensityScreenGuard(const RysDensityScreenGuard&) = delete;
    RysDensityScreenGuard& operator = (const RysDensityScreenGuard&) = delete;

private:
    const RysDensityScreenContext* previous_;
};
#endif
