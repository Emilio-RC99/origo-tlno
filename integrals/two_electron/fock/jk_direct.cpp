#include "jk_direct.hpp"
#include "jk_direct_internal.hpp"

#include <memory>
#include <stdexcept>
#include <utility>

namespace tlno::integrals::twoe::fock {

struct DirectPlanBackend {
    explicit DirectPlanBackend(tlno::integrals::twoe::EriPackedPlan plan_in)
        : transitional_rys_plan(std::move(plan_in)) {}

    // WHAT: share the immutable producer plan only during QUARANTINE
    // migration.
    // WHY: rebuilding geometry/primitives for the control and candidate would
    // contaminate A/B and double setup without making J/K more independent.
    // PURPOSE: the public J/K plan is already neutral while its backend can be
    // replaced by the future standalone scheduler without changing callers.
    tlno::integrals::twoe::EriPackedPlan transitional_rys_plan;
};

struct DirectPlanBackendAccess {
    static const std::shared_ptr<const DirectPlanBackend>& get(
        const DirectPlan& plan) noexcept {
        return plan.backend_;
    }

    static void set(
        DirectPlan& plan,
        std::shared_ptr<const DirectPlanBackend> backend) noexcept {
        plan.backend_ = std::move(backend);
    }
};

namespace detail {

const tlno::integrals::twoe::EriPackedPlan& transitional_eri_plan(
    const DirectPlan& plan) {
    const auto& owner = DirectPlanBackendAccess::get(plan);
    if (!owner) {
        throw std::runtime_error("jk_direct: null transitional plan backend.");
    }
    if (plan.profile.M != owner->transitional_rys_plan.profile.M) {
        throw std::runtime_error(
            "jk_direct: transitional plan/profile mismatch.");
    }
    return owner->transitional_rys_plan;
}

}  // namespace detail

ScreeningOptions screening_from_eri_options(
    const tlno::integrals::twoe::EriPackedBuildOptions& options) noexcept {
    ScreeningOptions converted;
    converted.integral_threshold = options.schwarz_threshold;
    converted.primitive_fastschwarz_threshold =
        options.primitive_fastschwarz_threshold;
    converted.decay_threshold = options.decay_threshold;
    converted.density_threshold = options.density_threshold;
    return converted;
}

tlno::integrals::twoe::EriPackedBuildOptions screening_to_eri_options(
    const ScreeningOptions& options) noexcept {
    tlno::integrals::twoe::EriPackedBuildOptions converted;
    converted.schwarz_threshold = options.integral_threshold;
    converted.primitive_fastschwarz_threshold =
        options.primitive_fastschwarz_threshold;
    converted.decay_threshold = options.decay_threshold;
    converted.density_threshold = options.density_threshold;
    return converted;
}

DirectPlan build_direct_plan(
    const std::vector<tlno::core::basis::Shell>& basis,
    const ScreeningOptions& options) {
    tlno::integrals::twoe::EriPackedPlan producer_plan =
        tlno::integrals::twoe::build_eri_packed_plan_for_G_rys(
            basis, screening_to_eri_options(options));
    DirectPlan plan;
    plan.profile = producer_plan.profile;
    DirectPlanBackendAccess::set(
        plan,
        std::make_shared<const DirectPlanBackend>(
            std::move(producer_plan)));
    return plan;
}

DirectPlan adopt_eri_plan_for_transition(
    const tlno::integrals::twoe::EriPackedPlan& producer_plan) {
    DirectPlan plan;
    plan.profile = producer_plan.profile;
    DirectPlanBackendAccess::set(
        plan,
        std::make_shared<const DirectPlanBackend>(producer_plan));
    return plan;
}

CoulombJResult build_coulomb_j(
    const DirectPlan& plan,
    const std::vector<double>& density,
    const ScreeningOptions& options) {
    const auto& owner = DirectPlanBackendAccess::get(plan);
    if (!owner) {
        throw std::runtime_error("jk_direct: null Coulomb-J plan backend.");
    }
    if (plan.profile.M != owner->transitional_rys_plan.profile.M) {
        throw std::runtime_error("jk_direct: Coulomb-J plan/profile mismatch.");
    }

    CoulombJResult result;
    result.j = tlno::integrals::twoe::jk_bridge::build_coulomb_j(
        owner->transitional_rys_plan, density,
        screening_to_eri_options(options));
    result.M = plan.profile.M;
    return result;
}

ExchangeKResult build_exchange_k(
    const DirectPlan& plan,
    const std::vector<double>& density,
    const ScreeningOptions& options) {
    const auto& producer = detail::transitional_eri_plan(plan);
    auto built = tlno::integrals::twoe::jk_bridge::build_exchange_k(
        producer, density, screening_to_eri_options(options));
    ExchangeKResult result;
    result.k = std::move(built.g);
    result.M = built.M;
    return result;
}

DirectResult build_rhf_g(
    const DirectPlan& plan,
    const std::vector<double>& density,
    const ScreeningOptions& options) {
    const auto& owner = DirectPlanBackendAccess::get(plan);
    if (!owner) {
        throw std::runtime_error("jk_direct: null J/K plan backend.");
    }
    if (plan.profile.M != owner->transitional_rys_plan.profile.M) {
        throw std::runtime_error("jk_direct: J/K plan/profile mismatch.");
    }

    // One ERI traversal constructs G = J - K/2.  The fused consumer uses the
    // same fast s/p producers as ERI0 and closes ss|ss at its terminal.
    tlno::integrals::twoe::EriDirectJKResult transitional =
        tlno::integrals::twoe::jk_bridge::build_direct_fused(
            owner->transitional_rys_plan, density,
            screening_to_eri_options(options));
    DirectResult result;
    result.g = std::move(transitional.g);
    result.M = transitional.M;
#ifdef TLNO_JK_COVERAGE_CHECKSUM
    result.coverage_checksum = transitional.coverage_checksum;
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    result.indexed_moment_1 = transitional.indexed_moment_1;
    result.indexed_moment_2 = transitional.indexed_moment_2;
    result.indexed_moment_3 = transitional.indexed_moment_3;
    result.canonical_count = transitional.canonical_count;
#endif
    return result;
}

}  // namespace tlno::integrals::twoe::fock
