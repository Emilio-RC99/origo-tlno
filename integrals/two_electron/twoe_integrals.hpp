#ifndef TLNO_INTEGRALS_TWO_ELECTRON_TWOE_INTEGRALS_HPP
#define TLNO_INTEGRALS_TWO_ELECTRON_TWOE_INTEGRALS_HPP

#include "core/basis/types.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace tlno::integrals::twoe {

struct EriPackedGAccessor {
    // AO indices are non-negative and strictly below the plan limit M <= 65535.
    // The resulting pair id fits in uint32_t, but the undivided triangular
    // product does not fit in signed int near that limit.
    static constexpr std::uint32_t pair_index(int a, int b) noexcept {
        if (a < b) {
            const int tmp = a;
            a = b;
            b = tmp;
        }
        const std::uint64_t hi = static_cast<std::uint64_t>(a);
        const std::uint64_t lo = static_cast<std::uint64_t>(b);
        return static_cast<std::uint32_t>(hi * (hi + 1u) / 2u + lo);
    }

    // Quartet ids require 64 bits well before the AO limit (from M=431).
    static constexpr std::uint64_t quartet_index(
        std::uint32_t ab, std::uint32_t cd) noexcept {
        const std::uint64_t hi = (ab >= cd) ? ab : cd;
        const std::uint64_t lo = (ab >= cd) ? cd : ab;
        return hi * (hi + 1u) / 2u + lo;
    }
};

struct EriPackedGData {
    std::vector<double> values;
};

inline constexpr double kEriPackedDefaultSchwarzThreshold = 1.0e-12;

struct EriPackedBuildOptions {
    double schwarz_threshold = kEriPackedDefaultSchwarzThreshold;

    // Simint-compatible FASTSCHWARZ primitive-pair estimator for the complete
    // AVX2/FMA shell4 portfolio from s through f.  It rejects only a SIMD
    // primitive packet with no surviving logical lane.  Higher-L and
    // non-AVX2 fallbacks do not consume this option.  Zero disables it.
    double primitive_fastschwarz_threshold = 0.0;

    // Heuristic Gaussian shell-pair decay indicator threshold. This is not a
    // rigorous ERI bound; calibrate it with materialized equal-error tests.
    // Native W8/SSE2 supports Schwarz AND decay, followed by primitive
    // FASTSCHWARZ on surviving packets. The legacy specialized W4 route
    // still requires exactly one shell-level metric.
    double decay_threshold = 0.0;

    // Density-coupled screening threshold, in the same units as
    // schwarz_threshold. The CVHF bound multiplies B_ab*B_cd by the maximum of
    // 4*max(rho_ab, rho_cd) and the four cross density blocks. Zero disables
    // it. It only affects the direct G route; the checksum stream ignores it
    // because there is no density P to bound against. The heuristic decay
    // indicator is a different metric and is not combined with this rigorous bound.
    double density_threshold = 0.0;
};

struct EriPackedPlanClassStats {
    int la = 0;
    int lb = 0;
    int lc = 0;
    int ld = 0;
    std::uint64_t pair_group_quartets = 0;
    std::uint64_t primitive_quartets = 0;
    std::uint64_t lhs_component_refs = 0;
    std::uint64_t rhs_component_refs = 0;
};

struct EriPackedPlanProfile {
    int M = 0;
    std::uint64_t shell_groups = 0;
    std::uint64_t pair_groups = 0;
    std::uint64_t pair_group_quartets = 0;
    std::uint64_t packed_pair_count = 0;
    std::uint64_t packed_value_count = 0;
    std::uint64_t primitive_pairs = 0;
    std::uint64_t max_primitive_pairs_per_pair_group = 0;
    std::vector<EriPackedPlanClassStats> class_stats;
};

// Implementation-only plan state. The pointee stays incomplete and its 
// owning handle stays private so callers cannot manufacture, replace, or 
// alias an unrelated object into a production plan.
struct EriPackedPlanBackend;

struct EriPackedPlan {
    EriPackedPlanProfile profile;

private:
    std::shared_ptr<const EriPackedPlanBackend> backend_;
    friend struct EriPackedPlanBackendAccess;
};

EriPackedPlan build_eri_packed_plan_for_G_rys(
    const std::vector<tlno::core::basis::Shell>& basis,
    const EriPackedBuildOptions& options);

// Sole production endpoint: execute a cached plan and emit the canonical
// pure-spherical stream. There is intentionally no basis-vector overload that
// can hide plan reconstruction or select Cartesian materialization.
EriPackedGData build_eri_packed_for_G_rys(
    const EriPackedPlan& plan,
    const EriPackedBuildOptions& options);

} // namespace tlno::integrals::twoe

#endif
