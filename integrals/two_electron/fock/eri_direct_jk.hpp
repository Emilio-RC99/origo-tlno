#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_ERI_DIRECT_JK_HPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_ERI_DIRECT_JK_HPP

// Cached-plan RHF and incremental matrix builders.
// The existing namespace and signatures retain their API/ABI identity.
#include "integrals/two_electron/twoe_integrals.hpp"

namespace tlno::integrals::twoe {

// Direct construction of the RHF G = J - 0.5*K over the same pure-spherical
// stream, never materializing the packed O(M^4) tensor.
// `density` and the returned `g` are M x M row-major in the spherical AO basis,
// with M = profile.M (spherical_function_count). `density` must be finite and
// symmetric; the s8 path and its density bound depend on that contract.
// Memory is O(M^2). This API builds the RHF combination; it does not expose
// J and K separately.
struct EriDirectJKResult {
    std::vector<double> g;
    int M = 0;
#ifdef TLNO_JK_COVERAGE_CHECKSUM
    // Raw sum of the digested integrals; comparable with the stream checksum
    // to show that no class bypasses J/K.
    double coverage_checksum = 0.0;
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    // Moments weighted by the canonical quartet index. Unlike the checksum,
    // they change when a correct value lands in the wrong slot, which is the
    // failure the d/f classes could introduce without anything complaining.
    double indexed_moment_1 = 0.0;
    double indexed_moment_2 = 0.0;
    double indexed_moment_3 = 0.0;
    unsigned long long canonical_count = 0;
#endif
};

EriDirectJKResult build_jk_direct_for_G_rys(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options);

// Stateful linear-update path for SCF iterations.  The initial density and
// every forced/periodic refresh rebuild G from the full density with
// base_options.density_threshold forced to zero.  Between refreshes the state
// contracts dP, optionally density-screens that smaller matrix with
// delta_density_threshold, and accumulates dG into the cached G.  The state is
// permanently bound to the plan copied by its constructor and is not safe for
// concurrent update() calls.
struct EriDirectJKIncrementalOptions {
    EriPackedBuildOptions base_options;
    // An interval N permits at most N-1 non-skipped delta updates between
    // exact rebuilds.  N=1 therefore requests an exact rebuild for every
    // changed density.
    std::uint64_t exact_rebuild_interval = 8;
    double delta_density_threshold = 0.0;
    // Optionally use a distinct density screen for the first changed delta
    // update after every exact rebuild.  Unchanged updates do not consume
    // this first-update slot.  When disabled, delta_density_threshold applies
    // to every incremental update, preserving the original behavior.
    bool use_first_delta_density_threshold = false;
    double first_delta_density_threshold = 0.0;
};

enum class EriDirectJKUpdateStatus : std::uint8_t {
    uninitialized = 0,
    exact_rebuild,
    incremental_update,
    skipped_unchanged,
};

struct EriDirectJKIncrementalCounters {
    // Counts successful update() calls only.  It is always the sum of the
    // three outcome counters below.
    std::uint64_t updates = 0;
    std::uint64_t exact_rebuilds = 0;
    std::uint64_t incremental_updates = 0;
    std::uint64_t skipped_updates = 0;
};

class EriDirectJKIncrementalState {
public:
    explicit EriDirectJKIncrementalState(
        const EriPackedPlan& plan,
        const EriDirectJKIncrementalOptions& options = {});

    // Strong state guarantee: validation or builder failures leave the prior
    // density, G, cadence, counters, and last-update diagnostics unchanged.
    EriDirectJKUpdateStatus update(
        const std::vector<double>& density,
        bool force_exact_rebuild = false);

    // Retains the bound plan and effective configuration, but returns the
    // cache and every diagnostic/counter to the just-constructed state.
    void reset() noexcept;

    bool initialized() const noexcept { return initialized_; }
    int M() const noexcept { return M_; }
    const EriPackedPlan& plan() const noexcept { return plan_; }
    const EriDirectJKIncrementalOptions& options() const noexcept {
        return options_;
    }
    const std::vector<double>& density() const noexcept { return density_; }
    const std::vector<double>& g() const noexcept { return g_; }
    const EriDirectJKIncrementalCounters& counters() const noexcept {
        return counters_;
    }
    EriDirectJKUpdateStatus last_status() const noexcept {
        return last_status_;
    }
    double last_max_abs_density_delta() const noexcept {
        return last_max_abs_density_delta_;
    }
    // Effective density threshold used by the last successful build.  Exact
    // rebuilds and unchanged skips report zero because neither applies a dP
    // density screen.
    double last_effective_density_threshold() const noexcept {
        return last_effective_density_threshold_;
    }
    std::uint64_t updates_since_exact_rebuild() const noexcept {
        return updates_since_exact_rebuild_;
    }

private:
    EriPackedPlan plan_;
    EriDirectJKIncrementalOptions options_;
    int M_ = 0;
    std::vector<double> density_;
    std::vector<double> g_;
    EriDirectJKIncrementalCounters counters_;
    EriDirectJKUpdateStatus last_status_ =
        EriDirectJKUpdateStatus::uninitialized;
    double last_max_abs_density_delta_ = 0.0;
    double last_effective_density_threshold_ = 0.0;
    std::uint64_t updates_since_exact_rebuild_ = 0;
    bool initialized_ = false;
};

}  // namespace tlno::integrals::twoe
#endif
