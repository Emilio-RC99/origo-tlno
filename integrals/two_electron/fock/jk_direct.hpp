#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_JK_DIRECT_HPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_JK_DIRECT_HPP

#include "eri_direct_jk.hpp"

#include <memory>
#include <vector>

namespace tlno::integrals::twoe::fock {

// WHAT: give J/K its own screening vocabulary instead of exposing an ERI
// builder option at the new API boundary.
// WHY: density is a contraction property and decay is not yet a rigorous ERI
// bound; keeping them in EriPackedBuildOptions entangles two products whose
// correctness and performance contracts are different.
// PURPOSE: the QUARANTINE backend can evolve density tasks/family masks now,
// then replace the transitional Rys-plan bridge without changing callers.
struct ScreeningOptions {
    double integral_threshold =
        tlno::integrals::twoe::kEriPackedDefaultSchwarzThreshold;
    double primitive_fastschwarz_threshold = 0.0;
    double decay_threshold = 0.0;
    double density_threshold = 0.0;
};

ScreeningOptions screening_from_eri_options(
    const tlno::integrals::twoe::EriPackedBuildOptions& options) noexcept;
tlno::integrals::twoe::EriPackedBuildOptions screening_to_eri_options(
    const ScreeningOptions& options) noexcept;

struct DirectPlanBackend;

// WHAT: own a J/K-facing plan handle and profile independently of the ERI API.
// WHY: production will schedule J/K tasks and layouts separately; callers must
// stop depending on the historical fact that J/K borrowed EriPackedPlan.
// PURPOSE: migration can proceed class by class while the opaque backend
// temporarily shares the already-validated Rys geometry/primitive plan.
struct DirectPlan {
    tlno::integrals::twoe::EriPackedPlanProfile profile;

private:
    std::shared_ptr<const DirectPlanBackend> backend_;
    friend struct DirectPlanBackendAccess;
};

DirectPlan build_direct_plan(
    const std::vector<tlno::core::basis::Shell>& basis,
    const ScreeningOptions& options = {});

// Transitional zero-rebuild constructor used by the benchmark/oracles while
// they still need the ERI plan for checksum and materialized validation.
// The shared backend is immutable; this copies only its owning handle/profile.
DirectPlan adopt_eri_plan_for_transition(
    const tlno::integrals::twoe::EriPackedPlan& plan);

struct DirectResult {
    std::vector<double> g;
    int M = 0;
#ifdef TLNO_JK_COVERAGE_CHECKSUM
    double coverage_checksum = 0.0;
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    double indexed_moment_1 = 0.0;
    double indexed_moment_2 = 0.0;
    double indexed_moment_3 = 0.0;
    unsigned long long canonical_count = 0;
#endif
};

// Separate direct endpoints share the immutable integral plan.  Neither
// materializes the ERI tensor or evaluates the other matrix as an intermediate.
struct CoulombJResult {
    std::vector<double> j;
    int M = 0;
};

struct ExchangeKResult {
    std::vector<double> k;
    int M = 0;
};

CoulombJResult build_coulomb_j(
    const DirectPlan& plan,
    const std::vector<double>& density,
    const ScreeningOptions& options);

// K[a,c] = sum_(b,d) P[b,d] (ab|cd), for a real symmetric density.
ExchangeKResult build_exchange_k(
    const DirectPlan& plan,
    const std::vector<double>& density,
    const ScreeningOptions& options);

DirectResult build_rhf_g(
    const DirectPlan& plan,
    const std::vector<double>& density,
    const ScreeningOptions& options);
}  // namespace tlno::integrals::twoe::fock
#endif
