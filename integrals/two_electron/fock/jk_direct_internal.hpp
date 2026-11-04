#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_JK_DIRECT_INTERNAL_HPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_JK_DIRECT_INTERNAL_HPP

#include "eri_direct_jk.hpp"
#include "jk_direct.hpp"

#include <cstddef>
#include <cstdint>

namespace tlno::integrals::twoe::fock::detail {

// WHAT: keep the canonical AO-pair metadata in the J/K subsystem instead of
// making the ERI stream own contraction records.
// WHY: J/K needs row offsets, symmetry scales and AO endpoints, while the ERI
// checksum stream does not.
// PURPOSE: the exact scalar consumer uses one compact immutable sidecar
// without materializing an ERI tensor.
struct PairInfo {
    std::uint32_t first_row = 0;
    std::uint32_t second_row = 0;
    std::uint16_t first = 0;
    std::uint16_t second = 0;
    float scale = 1.0f;
};
static_assert(sizeof(PairInfo) == 16,
              "J/K pair metadata must remain one compact 16-byte record.");

// WHAT: describe one projected spherical component and its canonical s8 pair.
// WHY: the tuned producers index values by component, whereas the scalar
// correctness path and validation gates index them by canonical AO pair.
// PURPOSE: the scalar consumer and validation record the same canonical
// stream without rebuilding its mapping.
struct PairComponent {
    std::uint32_t pair_index = 0;
    std::uint16_t component_index = 0;
    double norm = 1.0;
};

// WHAT: make the density/output state explicit at the producer boundary.
// WHY: the old output-stationary prototype handed an already-materialized
// block to a late callback and reread it six times, after losing the producer's
// register topology.
// PURPOSE: the exact scalar path keeps a neutral context that a future
// certified producer architecture can replace without changing public data.
struct BuildContext {
    double* g = nullptr;
    const double* p = nullptr;
    const PairInfo* pair_info = nullptr;
    // Canonical pair-space accumulator used by Coulomb-only consumers.  The
    // ordinary combined J/K policy leaves these null and retains its current
    // matrix-space writes.
    double* j_pair = nullptr;
    const double* j_pair_density = nullptr;
    std::size_t M = 0;
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    long double* indexed_moment_1 = nullptr;
    long double* indexed_moment_2 = nullptr;
    long double* indexed_moment_3 = nullptr;
    unsigned long long* canonical_count = nullptr;
#endif
#ifdef TLNO_JK_COVERAGE_CHECKSUM
    double* coverage_checksum = nullptr;
#endif
};

}  // namespace tlno::integrals::twoe::fock::detail

namespace tlno::integrals::twoe::fock::detail {

// Narrow internal transition seam used by physically separate experimental
// facades.  It exposes immutable producer plan data, not a backend selector.
const tlno::integrals::twoe::EriPackedPlan& transitional_eri_plan(
    const DirectPlan& plan);

}  // namespace tlno::integrals::twoe::fock::detail

namespace tlno::integrals::twoe::jk_bridge {

// WHAT: expose the scalar bridge while the J/K plan still borrows the validated
// Rys geometry and screening plan.
// WHY: rebuilding that immutable plan for control and candidate would
// contaminate the comparison, but a packet callback recreates the failed late
// receiver architecture.
// PURPOSE: the independent J/K API remains usable while a replacement
// scheduler is designed; this bridge disappears when J/K owns that scheduler.
EriDirectJKResult build_direct_scalar(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options);

EriDirectJKResult build_direct_fused(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options);

EriDirectJKResult build_exchange_k(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options);

// Productive J-only bridge.  Production and opt-in mathematical Rys variants
// instantiate the same symbol and public semantics; only their private HGP
// producer/reducer tunables may differ.  It remains deliberately distinct
// from the scalar combined-G bridge.
std::vector<double> build_coulomb_j(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options);

}  // namespace tlno::integrals::twoe::jk_bridge
#endif
