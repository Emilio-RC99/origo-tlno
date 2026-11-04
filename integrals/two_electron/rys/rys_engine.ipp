// Packed pure-spherical Rys ERI stream engine.
//
// Contracted shells are grouped by center, angular momentum, and primitive
// data. A cached plan feeds specialized low-L SIMD kernels or the residual
// Rys/HGP block path. Projection precedes the final canonical stream emission;
// the production endpoint does not materialize a Cartesian O(N^4) tensor.

#include "integrals/two_electron/twoe_integrals.hpp"
#include "integrals/two_electron/simd_policy.hpp"
#ifdef TLNO_BUILD_RYS_VALIDATION
#include "../../../../validation/rys_validation.hpp"
#endif
#include "integrals/two_electron/fock/jk_direct_internal.hpp"
#include "core/math/binary64.hpp"
#include "core/simd/packet.hpp"
#include "core/basis/types.hpp"
#include <map>
#include <tuple>

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <type_traits>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#endif
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(TLNO_BUILD_RYS_VALIDATION) && \
    defined(TLNO_BUILD_RYS_AUDIT)
#error "Rys validation and experiment roles are mutually exclusive"
#endif


#if defined(TLNO_VALIDATE_RYS_SPHERICAL_PACKED) && \
    !defined(TLNO_BUILD_RYS_VALIDATION)
#error "Materialized spherical validation requires the validation Rys object"
#endif

#include "integrals/two_electron/fock/detail/consumer_config.ipp"

namespace tlno::integrals::twoe {
namespace {

// ── Constants ────────────────────────────────────────────────────────────────
static constexpr double kPi = 3.141592653589793238462643383279502884;
// 2 * pi^(5/2), the ERI prefactor constant. Computed once at static-init time
// (not per primitive quartet) from kPi — no hardcoded literal.
static const double kTwoPi52 = 2.0 * kPi * kPi * std::sqrt(kPi);
// Simint's FASTSCHWARZ primitive-pair value is the squared MEST estimate
// sqrt(2)*pi^(5/2) * p^(-5/2) * exp(-2*mu*R^2) * (cA*cB)^2.
// Keep the same binary64 literal used by Simint's generated constants.h.
static constexpr double kSimintFastSchwarzPrefactor =
    24.739429451193148050228357586821944304008810699990039583;

// Highest per-shell L we support; static Cartesian-triple table covers
// |a| up to 2 * kMaxLSingle (e.g. (dd)-pair contribution).
static constexpr int kMaxLSingle = 4;
static constexpr int kMaxLSum    = 2 * kMaxLSingle;            // up to 8
static constexpr int kRysSoaBatch = 4;
#if !defined(__AVX2__) || !defined(__FMA__)
inline constexpr bool kRysJSpeedProbeExactRoots = true;
#endif

// Seven-component indexed stream fingerprint, bit-compatible with the audit's
// cross-engine consumer (same splitmix64 weight + long-double accumulation).
// Accumulated on the fast batched-emit path so the fair benchmark keeps the
// fast kernels instead of the scalar fallback.
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
struct EriStreamFP {
    long double checksum = 0.0L, absolute_sum = 0.0L, square_sum = 0.0L;
    long double indexed_moment_1 = 0.0L, indexed_moment_2 = 0.0L, indexed_moment_3 = 0.0L;
    long double max_abs = 0.0L;
    unsigned long long canonical_count = 0;
};
inline std::uint64_t eri_fp_mix(std::uint64_t v) noexcept {
    v += 0x9e3779b97f4a7c15ull;
    v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
    v = (v ^ (v >> 27)) * 0x94d049bb133111ebull;
    return v ^ (v >> 31);
}
inline long double eri_fp_weight(std::uint64_t index, std::uint64_t seed) noexcept {
    const std::uint64_t m = eri_fp_mix(index ^ seed);
    const long long centered = static_cast<long long>((m >> 40) & 0xffffffull) - 0x800000ll;
    return static_cast<long double>(centered) / 8388608.0L;
}
#endif

// Streaming (checksum-only, no-store) mode. Each build owns its accumulators,
// and the emitter reaches the active partition through a thread-local pointer.
// This keeps the hot emitter signature compact while allowing independent
// OpenMP partitions and concurrent host calls to use disjoint state.
struct alignas(64) EriStreamAccumulator {
    double checksum = 0.0;
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    EriStreamFP fingerprint;
#endif
};

thread_local EriStreamAccumulator* g_eri_stream_accumulator = nullptr;

#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
struct EriMaterializedSphericalAudit {
    std::size_t expected_count = 0;
    std::vector<std::uint64_t> seen;
    std::vector<std::uint64_t> allowed_missing;
    std::size_t unique_writes = 0;
    std::size_t first_duplicate = std::numeric_limits<std::size_t>::max();
    std::size_t first_out_of_range = std::numeric_limits<std::size_t>::max();

    void reset(std::size_t count) {
        expected_count = count;
        seen.assign((count + 63u) / 64u, 0u);
        allowed_missing.assign((count + 63u) / 64u, 0u);
        unique_writes = 0;
        first_duplicate = std::numeric_limits<std::size_t>::max();
        first_out_of_range = std::numeric_limits<std::size_t>::max();
    }

    void record(std::size_t index) noexcept {
        if (index >= expected_count) {
            if (first_out_of_range ==
                std::numeric_limits<std::size_t>::max()) {
                first_out_of_range = index;
            }
            return;
        }
        const std::size_t word = index >> 6u;
        const std::uint64_t bit =
            std::uint64_t{1} << static_cast<unsigned>(index & 63u);
        if ((seen[word] & bit) != 0u) {
            if (first_duplicate ==
                std::numeric_limits<std::size_t>::max()) {
                first_duplicate = index;
            }
            return;
        }
        seen[word] |= bit;
        ++unique_writes;
    }

    void allow_missing(std::size_t index) noexcept {
        if (index >= expected_count) {
            return;
        }
        const std::size_t word = index >> 6u;
        const std::uint64_t bit =
            std::uint64_t{1} << static_cast<unsigned>(index & 63u);
        allowed_missing[word] |= bit;
    }

    void validate(bool require_complete,
                  bool require_missing_allowed = false) const {
        if (first_out_of_range !=
            std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error(
                "twoe_integrals: materialized spherical validation "
                "emitted an out-of-range canonical index " +
                std::to_string(first_out_of_range) + " of " +
                std::to_string(expected_count));
        }
        if (first_duplicate !=
            std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error(
                "twoe_integrals: materialized spherical validation "
                "emitted canonical index " +
                std::to_string(first_duplicate) + " more than once.");
        }
        if (require_complete && unique_writes != expected_count) {
            throw std::runtime_error(
                "twoe_integrals: incomplete unscreened materialized "
                "spherical output: wrote " +
                std::to_string(unique_writes) + " unique indices, expected " +
                std::to_string(expected_count));
        }
        if (require_missing_allowed) {
            for (std::size_t word = 0; word < seen.size(); ++word) {
                const std::uint64_t disallowed_missing =
                    ~seen[word] & ~allowed_missing[word];
                if (disallowed_missing == 0u) {
                    continue;
                }
                const unsigned bit = static_cast<unsigned>(
                    __builtin_ctzll(disallowed_missing));
                const std::size_t index = (word << 6u) + bit;
                if (index < expected_count) {
                    throw std::runtime_error(
                        "twoe_integrals: materialized decay screening "
                        "omitted canonical index " +
                        std::to_string(index) +
                        " although its shell-pair decay-indicator product "
                        "passes the requested threshold.");
                }
            }
        }
    }
};

thread_local EriMaterializedSphericalAudit*
    g_eri_materialized_spherical_audit = nullptr;

class EriMaterializedSphericalAuditGuard {
public:
    explicit EriMaterializedSphericalAuditGuard(
        EriMaterializedSphericalAudit* audit) noexcept
        : previous_(g_eri_materialized_spherical_audit) {
        g_eri_materialized_spherical_audit = audit;
    }

    ~EriMaterializedSphericalAuditGuard() {
        g_eri_materialized_spherical_audit = previous_;
    }

    EriMaterializedSphericalAuditGuard(
        const EriMaterializedSphericalAuditGuard&) = delete;
    EriMaterializedSphericalAuditGuard& operator=(
        const EriMaterializedSphericalAuditGuard&) = delete;

private:
    EriMaterializedSphericalAudit* previous_;
};
#endif

inline bool eri_streaming() noexcept {
    return g_eri_stream_accumulator != nullptr;
}

inline EriStreamAccumulator& eri_stream_accumulator() noexcept {
    return *g_eri_stream_accumulator;
}

inline void reset_eri_stream_accumulator(EriStreamAccumulator& accumulator) noexcept {
    accumulator = EriStreamAccumulator{};
}

inline void merge_eri_stream_accumulator(EriStreamAccumulator& dst,
                                         const EriStreamAccumulator& src) noexcept {
    dst.checksum += src.checksum;
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    dst.fingerprint.checksum += src.fingerprint.checksum;
    dst.fingerprint.absolute_sum += src.fingerprint.absolute_sum;
    dst.fingerprint.square_sum += src.fingerprint.square_sum;
    dst.fingerprint.indexed_moment_1 += src.fingerprint.indexed_moment_1;
    dst.fingerprint.indexed_moment_2 += src.fingerprint.indexed_moment_2;
    dst.fingerprint.indexed_moment_3 += src.fingerprint.indexed_moment_3;
    dst.fingerprint.max_abs =
        std::max(dst.fingerprint.max_abs, src.fingerprint.max_abs);
    dst.fingerprint.canonical_count += src.fingerprint.canonical_count;
#endif
}

inline void eri_emit(std::vector<double>& vals, std::size_t idx, double val) noexcept {
    if (eri_streaming()) {
        EriStreamAccumulator& accumulator = eri_stream_accumulator();
        accumulator.checksum += val;
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
        EriStreamFP& f = accumulator.fingerprint;
        const long double x = static_cast<long double>(val);
        const long double a = x < 0.0L ? -x : x;
        f.checksum += x;
        f.absolute_sum += a;
        f.square_sum += x * x;
        f.indexed_moment_1 += x * eri_fp_weight(static_cast<std::uint64_t>(idx), 0x243f6a8885a308d3ull);
        f.indexed_moment_2 += x * eri_fp_weight(static_cast<std::uint64_t>(idx), 0x13198a2e03707344ull);
        f.indexed_moment_3 += x * eri_fp_weight(static_cast<std::uint64_t>(idx), 0xa4093822299f31d0ull);
        if (a > f.max_abs) f.max_abs = a;
        ++f.canonical_count;
#else
        (void)idx;
#endif
    } else {
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
        if (g_eri_materialized_spherical_audit != nullptr) {
            g_eri_materialized_spherical_audit->record(idx);
            if (idx >= vals.size()) {
                return;
            }
        }
#endif
        vals[idx] = val;
    }
}

// ---------------------------------------------------------------------------
// Emission destination
//
// WHAT: every integral the engine produces is owned by one compile-time emit
// policy. Non-packed consumers receive each canonical value through
// rys_emit_pairpair(); the packed policy may use equivalent block/lane checksum
// folds where no per-value consumer exists.
//
// WHY NOW: the roadmap adds consumers that contract the same integrals in
// different ways -- direct J/K assembly (which needs the density matrix in
// scope, something the packed-tensor route never has) and later the B^P
// intermediates of DF/RI.  Introducing the seam while there is exactly one
// consumer is cheap; introducing it once four consumers exist means editing
// every kernel that emits.
//
// WHY COMPILE TIME AND NOT A RUNTIME FLAG: this is the hottest function in the
// engine -- it runs once per emitted value, inside the innermost scatter loops.
// A runtime "which destination?" test would be paid on every integral, which
// is the kind of cost this route exists to avoid. Selecting the policy at the
// driver instantiation keeps each generated hot path branchless:
// static dispatch through a struct with a static member inlines completely.
//
// HOW TO ADD A CONSUMER: define a policy struct exposing
//     static void emit(std::vector<double>& vals, std::uint32_t lhs_pair,
//                      std::uint32_t rhs_pair, double val) noexcept;
// then instantiate the internal driver with that policy.  Public packed-tensor
// and benchmark entry points name RysPackedTensorEmit explicitly; a later J/K
// entry point will name its own policy explicitly.  No runtime destination
// branch is introduced in a kernel or scatter loop.
// ---------------------------------------------------------------------------

// PARAMETER ORDER -- easy to get wrong without reading this: Emit goes LAST
// when every earlier parameter is explicit, and FIRST when one of them is
// deduced from an argument (the for_each_spherical_screened_* case, whose
// Callback comes from the lambda). Explicit arguments fill from the left, so
// with Emit last it would take the deduced parameter's slot. Note also that
// Screened is always written explicitly before Emit.
//
// Packed policy: fold into the canonical packed ERI tensor, indexed by unique
// quartet id.  This body is the previous inline content of rys_emit_pairpair()
// verbatim, so the packed endpoint is bit-for-bit what it was before the seam.
struct RysPackedTensorEmit {
    static inline void emit(std::vector<double>& vals,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double val) noexcept {
        const std::size_t index =
            EriPackedGAccessor::quartet_index(lhs_pair, rhs_pair);
        eri_emit(vals, index, val);
    }
};

// Only the packed/checksum destination may use the hand-reduced checksum fast
// paths below.  All other policies require every canonical value together with
// its pair ids, so they must reach Emit::emit even in normal (non-fingerprint)
// builds.  `if constexpr` removes this distinction completely from each
// instantiation and preserves the packed endpoint's branchless inner loops.
#include "integrals/two_electron/fock/detail/consumer_policies.ipp"

template <class Emit>
inline constexpr bool rys_emit_uses_packed_checksum_fold_v =
    std::is_same_v<Emit, RysPackedTensorEmit>;

// No default is intentional: a forgotten policy at any converted layer is a
// compile error instead of a silent return to packed-tensor emission.
template <class Emit>
inline void rys_emit_pairpair(std::vector<double>& vals,
                              std::uint32_t lhs_pair,
                              std::uint32_t rhs_pair,
                              double val) noexcept {
    Emit::emit(vals, lhs_pair, rhs_pair, val);
}

constexpr int n_cart(int L) noexcept { return (L + 1) * (L + 2) / 2; }
constexpr int n_cart_cum(int L) noexcept {
    return (L + 1) * (L + 2) * (L + 3) / 6;
}

// Offset of the |a| = L slice in the cumulative flat list.
constexpr int offset_for_L(int L) noexcept {
    return (L == 0) ? 0 : L * (L + 1) * (L + 2) / 6;
}

// (2n-1)!!  with (2*0-1)!! = 1, (2*1-1)!! = 1, (2*2-1)!! = 3, etc.
inline double double_factorial_2nm1(int n) noexcept {
    double v = 1.0;
    for (int k = 2 * n - 1; k > 0; k -= 2) v *= static_cast<double>(k);
    return v;
}

// Angular normalization factor N_A(lx, ly, lz) = 1 / sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!).
// Angular component of the Cartesian primitive normalization.
inline double angular_norm(int lx, int ly, int lz) noexcept {
    const double df = double_factorial_2nm1(lx) * double_factorial_2nm1(ly)
                      * double_factorial_2nm1(lz);
    return 1.0 / std::sqrt(df);
}

inline int n_sph(int L) noexcept { return 2 * L + 1; }

struct SphC2STriple {
    int sph = 0;
    int cart = 0;
    double coeff = 0.0;
};

struct SphC2SBlock {
    const SphC2STriple* data = nullptr;
    int count = 0;
};

// TLNO/PySCF-compatible sparse cartesian->real-spherical transform.  The
// coefficients match qc_core_cpp/shell/cart2sph_tables.hpp and assume the
// Cartesian integral already includes the TLNO unit-diagonal angular
// normalization.
inline constexpr SphC2STriple kSphC2S_L0[] = {
    {0, 0, 1.0},
};
inline constexpr SphC2STriple kSphC2S_L1[] = {
    {0, 0, 1.0},
    {1, 1, 1.0},
    {2, 2, 1.0},
};
inline constexpr SphC2STriple kSphC2S_L2[] = {
    {2, 0, -0.49999999999999994},
    {4, 0, 0.8660254037844387},
    {0, 1, 1.0000000000000002},
    {3, 2, 1.0000000000000002},
    {2, 3, -0.49999999999999994},
    {4, 3, -0.8660254037844387},
    {1, 4, 1.0000000000000002},
    {2, 5, 0.9999999999999999},
};
inline constexpr SphC2STriple kSphC2S_L3[] = {
    {4, 0, -0.6123724356957946},
    {6, 0, 0.7905694150420949},
    {0, 1, 1.0606601717798212},
    {2, 1, -0.27386127875258304},
    {3, 2, -0.6708203932499369},
    {5, 2, 0.8660254037844387},
    {4, 3, -0.27386127875258304},
    {6, 3, -1.0606601717798212},
    {1, 4, 1.0000000000000002},
    {4, 5, 1.0954451150103321},
    {0, 6, -0.7905694150420949},
    {2, 6, -0.6123724356957946},
    {3, 7, -0.6708203932499369},
    {5, 7, -0.8660254037844387},
    {2, 8, 1.0954451150103321},
    {3, 9, 1.0},
};
inline constexpr SphC2STriple kSphC2S_L4[] = {
    {4, 0, 0.37500000000000006},
    {6, 0, -0.5590169943749475},
    {8, 0, 0.739509972887452},
    {0, 1, 1.1180339887498951},
    {2, 1, -0.42257712736425834},
    {5, 2, -0.8964214570007955},
    {7, 2, 0.790569415042095},
    {4, 3, 0.21957751641342002},
    {8, 3, -1.2990381056766582},
    {1, 4, 1.0606601717798214},
    {3, 4, -0.40089186286863665},
    {4, 5, -0.87831006565368},
    {6, 5, 0.9819805060619659},
    {0, 6, -1.1180339887498951},
    {2, 6, -0.42257712736425834},
    {5, 7, -0.40089186286863665},
    {7, 7, -1.0606601717798214},
    {2, 8, 1.133893419027682},
    {5, 9, 1.195228609334394},
    {4, 10, 0.37500000000000006},
    {6, 10, 0.5590169943749475},
    {8, 10, 0.739509972887452},
    {1, 11, -0.790569415042095},
    {3, 11, -0.8964214570007955},
    {4, 12, -0.87831006565368},
    {6, 12, -0.9819805060619659},
    {3, 13, 1.195228609334394},
    {4, 14, 1.0},
};

inline constexpr SphC2SBlock kSphC2SBlocks[] = {
    {kSphC2S_L0, static_cast<int>(sizeof(kSphC2S_L0) / sizeof(kSphC2S_L0[0]))},
    {kSphC2S_L1, static_cast<int>(sizeof(kSphC2S_L1) / sizeof(kSphC2S_L1[0]))},
    {kSphC2S_L2, static_cast<int>(sizeof(kSphC2S_L2) / sizeof(kSphC2S_L2[0]))},
    {kSphC2S_L3, static_cast<int>(sizeof(kSphC2S_L3) / sizeof(kSphC2S_L3[0]))},
    {kSphC2S_L4, static_cast<int>(sizeof(kSphC2S_L4) / sizeof(kSphC2S_L4[0]))},
};

// ── Boys table (independent copy; matches gemm engine) ──────────────────────
static constexpr int BOYS_N_HGP   = 4000;
static constexpr double BOYS_DX_HGP = 40.0 / static_cast<double>(BOYS_N_HGP);

struct BoysGaussLegendreRuleHGP {
    static constexpr int N = 96;
    double x[N]{};
    double w[N]{};

    BoysGaussLegendreRuleHGP() noexcept {
        const int half = (N + 1) / 2;
        for (int i = 0; i < half; ++i) {
            double z = std::cos(kPi * (static_cast<double>(i) + 0.75) /
                                (static_cast<double>(N) + 0.5));
            double pp = 0.0;
            for (int iter = 0; iter < 32; ++iter) {
                double p1 = 1.0;
                double p2 = 0.0;
                for (int j = 1; j <= N; ++j) {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * static_cast<double>(j) - 1.0) * z * p2
                          - (static_cast<double>(j) - 1.0) * p3) / static_cast<double>(j);
                }
                pp = static_cast<double>(N) * (z * p1 - p2) / (z * z - 1.0);
                const double z_next = z - p1 / pp;
                if (std::fabs(z_next - z) < 1.0e-15) {
                    z = z_next;
                    break;
                }
                z = z_next;
            }
            const double wi = 1.0 / ((1.0 - z * z) * pp * pp);
            x[i] = 0.5 * (1.0 - z);
            x[N - 1 - i] = 0.5 * (1.0 + z);
            w[i] = wi;
            w[N - 1 - i] = wi;
        }
    }
};

struct BoysTableHGP {
    static constexpr int MAX_ORDER = 28;
    double data[MAX_ORDER + 1][BOYS_N_HGP + 1]{};
    BoysTableHGP() noexcept {
        BoysGaussLegendreRuleHGP q;
        for (int k = 0; k <= BOYS_N_HGP; ++k) {
            const double x = k * BOYS_DX_HGP;
            for (int iq = 0; iq < BoysGaussLegendreRuleHGP::N; ++iq) {
                const double t2 = q.x[iq] * q.x[iq];
                double p = 1.0;
                const double ww = q.w[iq] * std::exp(-x * t2);
                for (int m = 0; m <= MAX_ORDER; ++m) {
                    data[m][k] += ww * p;
                    p *= t2;
                }
            }
        }
    }
};
static const BoysTableHGP g_boys_hgp;

inline double boys_eval_hgp(int m, double x) noexcept {
    if (x >= 40.0) {
        double v = 1.0;
        for (int i = 1; i <= m; ++i) v *= (2.0 * i - 1.0) / (2.0 * x);
        return v * 0.5 * std::sqrt(kPi / x);
    }
    const double fi = x / BOYS_DX_HGP;
    const int ki = static_cast<int>(fi);
    const double t = fi - ki;
    if (ki + 4 <= BOYS_N_HGP) {
        const double f0 = g_boys_hgp.data[m][ki];
        const double f1 = g_boys_hgp.data[m][ki + 1];
        const double f2 = g_boys_hgp.data[m][ki + 2];
        const double f3 = g_boys_hgp.data[m][ki + 3];
        const double f4 = g_boys_hgp.data[m][ki + 4];
        const double d1 = f1 - f0;
        const double d2 = (f2 - 2.0 * f1 + f0) * 0.5;
        const double d3 = (f3 - 3.0 * f2 + 3.0 * f1 - f0) * (1.0 / 6.0);
        const double d4 = (f4 - 4.0 * f3 + 6.0 * f2 - 4.0 * f1 + f0) * (1.0 / 24.0);
        return f0 + t * (d1 + (t - 1.0) * (d2 + (t - 2.0) * (d3 + (t - 3.0) * d4)));
    }
    if (ki + 2 <= BOYS_N_HGP) {
        const double f0 = g_boys_hgp.data[m][ki];
        const double f1 = g_boys_hgp.data[m][ki + 1];
        const double f2 = g_boys_hgp.data[m][ki + 2];
        return f0 + t * ((-3.0 * f0 + 4.0 * f1 - f2) * 0.5
                        + t * (f0 - 2.0 * f1 + f2) * 0.5);
    }
    return g_boys_hgp.data[m][ki] + t * (g_boys_hgp.data[m][ki + 1] - g_boys_hgp.data[m][ki]);
}

void boys_fill_hgp(int mmax, double x, double* out) noexcept {
    const int table_max = std::min(mmax, BoysTableHGP::MAX_ORDER);
    for (int m = 0; m <= table_max; ++m) out[m] = boys_eval_hgp(m, x);
    if (mmax > BoysTableHGP::MAX_ORDER) {
        if (x < 1e-10) {
            for (int m = BoysTableHGP::MAX_ORDER + 1; m <= mmax; ++m)
                out[m] = 1.0 / static_cast<double>(2 * m + 1);
        } else {
            const double ex = std::exp(-x);
            for (int m = BoysTableHGP::MAX_ORDER; m < mmax; ++m)
                out[m + 1] = ((2 * m + 1) * out[m] - ex) / (2.0 * x);
        }
    }
}

// ── Cartesian-triple enumeration ─────────────────────────────────────────────
// Order: for lx = L..0: for ly = L-lx..0: lz = L-lx-ly.
struct CartTriplesTable {
    static constexpr int kMaxStored = 165;  // n_cart_cum(8) = 165
    std::array<std::array<int, 3>, kMaxStored> triples{};
    CartTriplesTable() noexcept {
        int idx = 0;
        for (int L = 0; L <= kMaxLSum; ++L) {
            for (int lx = L; lx >= 0; --lx) {
                for (int ly = L - lx; ly >= 0; --ly) {
                    const int lz = L - lx - ly;
                    triples[static_cast<std::size_t>(idx++)] = {lx, ly, lz};
                }
            }
        }
    }
    inline const std::array<int, 3>& at(int flat_idx) const noexcept {
        return triples[static_cast<std::size_t>(flat_idx)];
    }
};
static const CartTriplesTable g_cart;

inline int triple_idx_within(int L, int lx, int ly) noexcept {
    const int dx = L - lx;
    return dx * (dx + 1) / 2 + (dx - ly);
}

inline int cart_flat_idx(int lx, int ly, int lz) noexcept {
    const int L = lx + ly + lz;
    return offset_for_L(L) + triple_idx_within(L, lx, ly);
}

// ── Shell grouping ───────────────────────────────────────────────────────────
struct ContractedShellGroupH {
    int L = 0;
    std::array<double, 3> center{};
    std::vector<int> basis_indices;                // [count]
    std::vector<std::array<int, 3>> angular;       // [count]
    std::vector<double> exponents;                 // [nprim]
    std::vector<std::vector<double>> coefficients; // [count][nprim]  (stored_c)
    std::vector<double> c_radial;                  // [nprim] = stored_c(:, 0) / N_A(angular[0])
    std::vector<double> n_ang;                     // [count] = N_A(angular[i])
};

struct RysPrimitivePair {
    double s = 0.0;           // exponent sum (p or q)
    double invs = 0.0;        // 1/s
    double Wx = 0.0, Wy = 0.0, Wz = 0.0;  // Gaussian product centre (P or Q)
    double w = 0.0;           // K * contraction-coeff * (1/s)
    double fast_schwarz = 0.0;  // Simint FASTSCHWARZ squared estimate
};

struct RysPrimitivePairSlab {
    std::vector<double> s;
    std::vector<double> invs;
    std::vector<double> Wx;
    std::vector<double> Wy;
    std::vector<double> Wz;
    std::vector<double> w;
    std::vector<double> fast_schwarz;

    int size() const noexcept { return static_cast<int>(s.size()); }
};

// WHAT: share the projected-component POD with J/K's exact scalar consumer and
// stream-identity audits.
// WHY/PURPOSE: immutable canonical pair maps keep projection, scalar J/K and
// validation on one representation without constructing a packet ABI.
using PackedPairComponentH = tlno::integrals::twoe::fock::detail::PairComponent;

#include "integrals/two_electron/fock/detail/block_receivers.ipp"

// Per-lane variant for the direct spherical microkernels. Its production
// packed endpoint first reduces all the values of one lane and adds them to
// the checksum once; keeping that association avoids bringing the per-integral
// TLS access of eri_emit back into the hot loop. The per-value consumers --
// direct J/K and any future policy -- use the normal block and therefore
// resolve their context only once. The validation fingerprint
// still sees every (index, value), as before.
template <class Emit, class ValueFn>
inline void rys_emit_spherical_lane_block(
    std::vector<double>& vals,
    const std::vector<PackedPairComponentH>& lhs_refs,
    const std::vector<PackedPairComponentH>& rhs_refs,
    bool same_pair_group,
    ValueFn&& value_of) noexcept {
#ifndef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    if constexpr (rys_emit_uses_packed_checksum_fold_v<Emit>) {
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
        // A materialized validation call deliberately runs without an active
        // stream accumulator.  In that mode Emit::emit writes the packed
        // tensor, so the production-only lane checksum fold must not run.
        if (!eri_streaming()) {
            rys_emit_block<Emit>(
                vals, lhs_refs, rhs_refs, same_pair_group,
                value_of);
            return;
        }
#endif
        double lane_checksum = 0.0;
        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (same_pair_group &&
                    rhs_ref.pair_index > lhs_ref.pair_index) continue;
                lane_checksum += value_of(lhs_ref, rhs_ref);
            }
        }
        eri_stream_accumulator().checksum += lane_checksum;
        return;
    }
#endif
    rys_emit_block<Emit>(
        vals, lhs_refs, rhs_refs, same_pair_group, value_of);
}

// Pretransposed shell-pair data for the low-L shell-batch kernels.
// Bins are keyed first by pair angular sum (0=ss, 1=ps, 2=pp), then by primitive-pair
// count. Within a bin, primitive ip for four independent shell pairs is
// contiguous, so the hot kernel can load an AVX2 vector directly instead of
// gathering four unrelated RysPrimitivePairSlabs. Lx/Ly/Lz are P-A for the
// angular-carrying p shell (zero and unused for ss pairs).
struct RysLowPairBinH {
    int primitive_count = 0;
    std::size_t pair_count = 0;
    std::size_t stride = 0;  // pair_count rounded up to four lanes
    std::vector<std::uint32_t> pair_group_ids;
    std::vector<double> s;
    std::vector<double> invs;
    std::vector<double> Wx;
    std::vector<double> Wy;
    std::vector<double> Wz;
    std::vector<double> w;
    std::vector<double> primitive_fast_schwarz;
    std::vector<double> Lx;
    std::vector<double> Ly;
    std::vector<double> Lz;
    std::vector<double> Rx;  // A-B for pp horizontal transfer (per pair group)
    std::vector<double> Ry;
    std::vector<double> Rz;
    // Lazily attached to a metric-specific pure-spherical screened view.  The
    // layout follows pair_group_ids (including its padded four-lane tail), so
    // a partial bin can test four shell quartets with one contiguous AVX2 load
    // without splitting the tuned bin into chunks.  A view contains exactly
    // one metric family; Schwarz and decay caches never alias.
    mutable std::vector<double> shell_screen_bound;
    mutable double shell_screen_bound_min = 0.0;
    mutable double shell_screen_bound_max = 0.0;
};

// One full, metric-descending view of every original s/p contraction-length
// bin. It does not fragment a tuned kernel into tasks: it only permutes the
// lanes once so a screened kernel can stop as soon as the monotone metric
// product falls below the active threshold.
struct RysScreenedSpFullBins {
    std::array<std::vector<RysLowPairBinH>, 3> bins;
    // Maximum over every real pair in the six s/p x s/p task families.  d/f
    // owns an independent chunk scheduler and never consumes this prefix.
    double sp_pair_group_metric_max = 0.0;
};

// Metric-sorted chunks for every oriented pure-spherical shell-pair class
// through f. The underlying RysLowPairBinH layout is angular-class agnostic:
// it stores only pair-group ids, primitive-pair slabs, and the two centres'
// displacement data.  Consequently the same tuned specialized and generic
// kernels can consume these chunks without changing their arithmetic.
//
// Per-original-bin extrema support a hybrid scheduler:
//   * max(A)*max(B) < threshold: reject the whole bin pair;
//   * min(A)*min(B) >= threshold: run the original bins once;
//   * otherwise: enumerate the bound-sorted chunks.
// This protects dense/near-field work from needless task fragmentation.
struct RysScreenedSphericalChunks {
    // [pair_class][primitive_count] -> chunks of that original bin
    std::array<std::vector<std::vector<RysLowPairBinH>>, 10> chunks;
    // [pair_class][primitive_count][chunk]
    std::array<std::vector<std::vector<double>>, 10> chunk_max_bound;
    // [pair_class][primitive_count]
    std::array<std::vector<double>, 10> bin_min_bound;
    std::array<std::vector<double>, 10> bin_max_bound;
};

struct RysSphericalSchwarzBounds {
    // Exact contracted pure-spherical AO-pair bounds, indexed in the
    // pure-spherical packed-pair space.
    std::vector<double> pair_bounds;
    // Maximum AO-pair bound in each contracted shell-pair group.
    std::vector<double> pair_group_max;
};

// Chunk width in pairs. A multiple of sixteen keeps the 1x16 (ss|ss) ket loop
// and the four-lane bra loops full, while staying small enough that a rejected
// region is not dragged along by one significant neighbour.
#ifndef TLNO_RYS_SPH_SCREEN_CHUNK_PAIRS
#define TLNO_RYS_SPH_SCREEN_CHUNK_PAIRS 16
#endif
inline constexpr std::size_t kRysScreenedChunkPairs =
    TLNO_RYS_SPH_SCREEN_CHUNK_PAIRS;

inline std::size_t rys_screened_spherical_low_l_chunk_pairs() {
    static const std::size_t chunk_pairs = []() {
        const char* const value =
            std::getenv("TLNO_RYS_SPH_SCREEN_CHUNK_PAIRS");
        if (value == nullptr || value[0] == '\0') {
            return kRysScreenedChunkPairs;
        }
        char* end = nullptr;
        const unsigned long long parsed =
            std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0u ||
            parsed > (1u << 20)) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: TLNO_RYS_SPH_SCREEN_CHUNK_PAIRS must "
                "be an integer in [1, 1048576].");
        }
        return static_cast<std::size_t>(parsed);
    }();
    return chunk_pairs;
}

struct RysSphericalScreeningDiagnostics {
    std::uint64_t whole_bin_pairs_accepted = 0;
    std::uint64_t whole_bin_pairs_rejected = 0;
    std::uint64_t partial_bin_pairs = 0;
    std::uint64_t chunk_pairs_accepted = 0;
    std::uint64_t chunk_pairs_rejected = 0;
    std::uint64_t scheduled_pair_group_quartets = 0;
    std::uint64_t accepted_pair_group_quartets = 0;
    std::uint64_t rejected_pair_group_quartets = 0;
    std::uint64_t generic_accepted_tasks = 0;
    std::uint64_t generic_accepted_pair_group_quartets = 0;
    // The generic shell4 q_floor is an adaptive two-axis stop inside the exact
    // mask: these counters record how often the maximum ket proves the current
    // and every later Q-descending bra packet dead.
    std::uint64_t generic_q_floor_tail_stops = 0;
    std::uint64_t generic_q_floor_pruned_bra_pairs = 0;
    // Prefix q_floor is applied to partial specialized cross-class chunks.
    // candidate is the complete accepted chunk rectangle, visited is the
    // monotone prefix handed to the kernel, and pruned is its proven-dead tail.
    // The exact in-kernel masks therefore see only visited, never pruned.
    std::uint64_t df_q_floor_tasks = 0;
    std::uint64_t df_q_floor_candidate_pair_group_quartets = 0;
    std::uint64_t df_q_floor_visited_pair_group_quartets = 0;
    std::uint64_t df_q_floor_pruned_pair_group_quartets = 0;
    std::uint64_t sp_masked_tasks = 0;
    std::uint64_t sp_masked_tiles = 0;
    std::uint64_t sp_masked_tiles_rejected = 0;
    std::uint64_t sp_masked_shell_lanes = 0;
    std::uint64_t sp_masked_shell_lanes_accepted = 0;
    std::uint64_t sp_masked_shell_lanes_rejected = 0;
    std::uint64_t primitive_fast_vectors_tested = 0;
    std::uint64_t primitive_fast_vectors_rejected = 0;
    std::uint64_t primitive_fast_lanes_tested = 0;
    std::uint64_t primitive_fast_lanes_rejected_simd = 0;
    // Audit only: lanes that a strict per-lane mask would reject.  Headline
    // timings deliberately retain Simint's whole-vector survival policy.
    std::uint64_t primitive_fast_lanes_rejected_strict = 0;
    std::array<std::uint64_t, 6> primitive_fast_class_vectors_tested{};
    std::array<std::uint64_t, 6> primitive_fast_class_vectors_rejected{};
    std::array<std::uint64_t, 6> primitive_fast_class_lanes_tested{};
    std::array<std::uint64_t, 6> primitive_fast_class_lanes_rejected_simd{};
    std::array<std::uint64_t, 6> primitive_fast_class_lanes_rejected_strict{};
    // Aggregate for every d/f packet, including the residual shell4
    // portfolio.  The six named counters above remain stable for the s/p
    // class study; this bucket proves that the generalized selector actually
    // reaches work outside that original domain.
    std::uint64_t primitive_fast_df_vectors_tested = 0;
    std::uint64_t primitive_fast_df_vectors_rejected = 0;
    std::uint64_t primitive_fast_df_lanes_tested = 0;
    std::uint64_t primitive_fast_df_lanes_rejected_simd = 0;
    std::uint64_t primitive_fast_df_lanes_rejected_strict = 0;
};

thread_local RysSphericalScreeningDiagnostics*
    g_rys_spherical_screening_diagnostics = nullptr;

class RysSphericalScreeningDiagnosticsGuard {
public:
    explicit RysSphericalScreeningDiagnosticsGuard(
        RysSphericalScreeningDiagnostics* diagnostics) noexcept
        : previous_(g_rys_spherical_screening_diagnostics) {
        g_rys_spherical_screening_diagnostics = diagnostics;
    }

    ~RysSphericalScreeningDiagnosticsGuard() {
        g_rys_spherical_screening_diagnostics = previous_;
    }

    RysSphericalScreeningDiagnosticsGuard(
        const RysSphericalScreeningDiagnosticsGuard&) = delete;
    RysSphericalScreeningDiagnosticsGuard& operator = (
        const RysSphericalScreeningDiagnosticsGuard&) = delete;

private:
    RysSphericalScreeningDiagnostics* previous_;
};

inline bool rys_spherical_screening_diagnostics_requested() noexcept {
    const char* const value =
        std::getenv("TLNO_RYS_SPH_SCREEN_DIAG");
    return value != nullptr && value[0] != '\0' &&
           !(value[0] == '0' && value[1] == '\0');
}

#include "integrals/two_electron/fock/detail/density_context.ipp"

#if defined(__AVX2__) && defined(__FMA__)
struct RysSpShellScreenContext {
    double threshold = 0.0;
    bool bounds_sorted_descending = false;
    // QUE: indica si este task mixto necesita refinar el mask Q con densidad.
    // POR QUE: un task con cuartetos Q-live pero resultado de densidad mixto
    // necesita P por paquete; density-live deja este campo falso y conserva
    // solamente el mask Q que todavia corresponda.
    bool density_mask = false;
    double primitive_fastschwarz_threshold_squared = 0.0;
    int primitive_fast_class = -1;
    // Per-call logical ends for immutable metric-sorted bins.  Keeping the
    // override here avoids mutating pair_count in a cache shared by builds and
    // worker threads; stride and every primitive-major offset remain full.
    const RysLowPairBinH* bra_bin = nullptr;
    std::size_t bra_pair_count = 0;
    const RysLowPairBinH* ket_bin = nullptr;
    std::size_t ket_pair_count = 0;
};

// Exact shell-quartet screening context shared by the pure-spherical d/f
// batches.  The metric is indexed by shell-pair group and satisfies the same
// product predicate used by the outer scheduler.  Keeping this context
// separate from RysSpShellScreenContext is intentional: s/p tasks still carry
// their class id and logical bin ends there, while this context supplies one
// class-agnostic primitive FASTSCHWARZ threshold to every d/f kernel.
struct RysDfQFloorPairWindow {
    // Role-agnostic identities: screened callbacks may swap the two bins so
    // the longer live prefix occupies the AVX bra lanes.  Matching by address
    // preserves the correct logical end across that orientation change.
    const RysLowPairBinH* first_bin = nullptr;
    std::size_t first_live = 0;
    const RysLowPairBinH* second_bin = nullptr;
    std::size_t second_live = 0;
};

struct RysBatchScreen {
    double q_break = 0.0;
    // Already squared at the public screened API boundary.  A zero value
    // disables primitive rejection; its small runtime branch in the shared
    // d/f bodies is measured explicitly against the unscreened control.
    double primitive_fastschwarz_threshold_squared = 0.0;
    const double* pair_group_metric = nullptr;
    // Density screening at the level of shell-group pairs.
    //
    // WHY here and not inside the digestion: rejecting in Emit would pay the
    // test per integral without saving the quartet's computation.  This is the
    // only point where the quartet has not been computed yet, and it is shared
    // by the eight specialized families AND the generic route (which calls
    // rys_batch_screen_lane_mask with exact_shell_mask), so a single hook
    // covers the whole engine.
    //
    // pair_group_density[pg] = max(drow[left], drow[right]), with
    // drow[g] = max_h D[g][h] over group blocks.  WHY this row bound and not
    // the exact maximum over the 10 pairs of {ga,gb,gc,gd}: that maximum does
    // NOT factorize as f(bra)*g(ket), so it cannot be folded into
    // pair_group_metric.  The multiplicative alternatives either degenerate
    // into the global constant (erasing all density structure and leaving a
    // rescaled Schwarz threshold) or UNDERESTIMATE the maximum -- the
    // geometric mean, for example -- and would then reject too much, which is
    // an error in the result, not in speed.  The row bound dominates every
    // crossed term D[ga][gc] through the row of either of its two indices,
    // so it is a valid upper bound: it rejects less, never too much.
    const double* shell_rho = nullptr;
    // WHAT: per-pair-group bound on the largest P block that can appear in
    // any of its four crossed contributions with another pair group.
    // WHY: it lets WholeDead be proved in O(1) without rereading four P blocks
    // per quartet in an exhaustive sweep of the scheduler.
    const double* pair_group_density_upper = nullptr;
    const double* group_pair_density_max = nullptr;
    int density_ng = 0;
    const std::uint32_t* pair_group_left = nullptr;
    const std::uint32_t* pair_group_right = nullptr;
    double density_break = 0.0;
    // WHAT: identify the J-only experiment inside the existing density plan.
    // WHY: J owns only the two Coulomb terms; retaining a quartet solely because
    // one of four exchange blocks is large performs K screening for work that
    // this endpoint never computes.
    // PURPOSE: the probe is compared fairly with CVHF's vj-only prescreen while
    // the combined G and packed ERI policies retain their original predicate.
    bool j_only = false;
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    // K-only uses exactly the four crossed density blocks and must not retain
    // work merely because either Coulomb density is large.
    bool k_only = false;
#endif
    // Per-callback q_floor window.  It is mutable only so a nested RAII guard
    // can update the already-installed thread-local screening context without
    // introducing another TLS variable or mutating cached bins.
    mutable const RysDfQFloorPairWindow* df_q_floor_window = nullptr;
};



thread_local const RysBatchScreen* g_rys_batch_screen = nullptr;

class RysBatchScreenGuard {
public:
    explicit RysBatchScreenGuard(const RysBatchScreen* screen) noexcept
        : previous_(g_rys_batch_screen) {
        g_rys_batch_screen = screen;
    }

    ~RysBatchScreenGuard() {
        g_rys_batch_screen = previous_;
    }

    RysBatchScreenGuard(const RysBatchScreenGuard&) = delete;
    RysBatchScreenGuard& operator = (const RysBatchScreenGuard&) = delete;

private:
    const RysBatchScreen* previous_;
};

class RysDfQFloorPairWindowGuard {
public:
    RysDfQFloorPairWindowGuard(
        const RysBatchScreen* screen,
        const RysDfQFloorPairWindow* window) noexcept
        : screen_(screen),
          previous_(screen == nullptr
                        ? nullptr
                        : screen->df_q_floor_window) {
        if (screen_ != nullptr) screen_->df_q_floor_window = window;
    }

    ~RysDfQFloorPairWindowGuard() {
        if (screen_ != nullptr) screen_->df_q_floor_window = previous_;
    }

    RysDfQFloorPairWindowGuard(
        const RysDfQFloorPairWindowGuard&) = delete;
    RysDfQFloorPairWindowGuard& operator = (
        const RysDfQFloorPairWindowGuard&) = delete;

private:
    const RysBatchScreen* screen_;
    const RysDfQFloorPairWindow* previous_;
};

thread_local const RysSpShellScreenContext*
    g_rys_sp_shell_screen_context = nullptr;

class RysSpShellScreenContextGuard {
public:
    explicit RysSpShellScreenContextGuard(
        const RysSpShellScreenContext* context) noexcept
        : previous_(g_rys_sp_shell_screen_context) {
        g_rys_sp_shell_screen_context = context;
    }

    ~RysSpShellScreenContextGuard() {
        g_rys_sp_shell_screen_context = previous_;
    }

    RysSpShellScreenContextGuard(
        const RysSpShellScreenContextGuard&) = delete;
    RysSpShellScreenContextGuard& operator = (
        const RysSpShellScreenContextGuard&) = delete;

private:
    const RysSpShellScreenContext* previous_;
};

template <bool Screened>
inline std::size_t rys_sp_active_pair_count(
    const RysLowPairBinH& bin) noexcept {
    if constexpr (!Screened) {
        // Preserve the validated unscreened machine path: no TLS load or
        // pointer branches survive this compile-time specialization.
        return bin.pair_count;
    } else {
        const RysSpShellScreenContext* const screen =
            g_rys_sp_shell_screen_context;
        if (screen != nullptr) {
            if (screen->bra_bin == &bin) return screen->bra_pair_count;
            if (screen->ket_bin == &bin) return screen->ket_pair_count;
        }
        return bin.pair_count;
    }
}

// q_floor d/f keeps immutable logical ends in the existing screened context so
// the seven specialized kernels retain their original hot ABI.  Screened=false
// is literally the validated bin.pair_count path: no TLS load or pointer branch.
template <bool Screened>
inline std::size_t rys_df_q_floor_active_pair_count(
    const RysLowPairBinH& bin) noexcept {
    if constexpr (!Screened) {
        return bin.pair_count;
    } else {
        const RysBatchScreen* const screen = g_rys_batch_screen;
        const RysDfQFloorPairWindow* const window =
            screen == nullptr ? nullptr : screen->df_q_floor_window;
        if (window == nullptr) return bin.pair_count;
        std::size_t live = bin.pair_count;
        if (window->first_bin == &bin) {
            live = window->first_live;
        } else if (window->second_bin == &bin) {
            live = window->second_live;
        }
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
        assert(live <= bin.pair_count);
#endif
        return live;
    }
}

inline unsigned rys_low_bits(std::size_t count) noexcept {
    return count >= 4u ? 0xFu : ((1u << count) - 1u);
}

// WHAT: evaluate a quartet of group pairs with the same combined predicate,
// Q*cap with the density cap, that the d/f kernels use before computing roots.
//
// WHY: the classification of s/p tasks and their partial mask must decide with
// exactly the same boundary.  Previously, enabling density sent every s/p task
// to the masked kernel, but that mask only repeated Schwarz and never applied
// the local P bound; the overhead was paid without rejecting those lanes.
inline bool rys_density_shell_quartet_live(
    const RysBatchScreen& screen,
    std::uint32_t bra_group,
    std::uint32_t ket_group) noexcept {
    const double quad_est =
        screen.pair_group_metric[bra_group] *
        screen.pair_group_metric[ket_group];
    if (quad_est < screen.q_break) return false;
    if (screen.shell_rho == nullptr) return true;

#ifdef TLNO_RYS_DIRECT_EXCHANGE
    if (screen.k_only) {
        const std::size_t ng = static_cast<std::size_t>(screen.density_ng);
        const double* const D = screen.group_pair_density_max;
        const std::size_t ga = screen.pair_group_left[bra_group];
        const std::size_t gb = screen.pair_group_right[bra_group];
        const std::size_t gc = screen.pair_group_left[ket_group];
        const std::size_t gd = screen.pair_group_right[ket_group];
        const double exch = std::max(
            std::max(D[gc * ng + ga], D[gd * ng + ga]),
            std::max(D[gc * ng + gb], D[gd * ng + gb]));
        return quad_est * exch >= screen.density_break;
    }
#endif
    const double coul = 4.0 * std::max(
        screen.shell_rho[bra_group], screen.shell_rho[ket_group]);
    if (screen.j_only) {
        return quad_est * coul >= screen.density_break;
    }
    // WHAT: resolve exchange-only group metadata after the J-only return.
    // WHY: merely loading these endpoints kept K's cache traffic in every J
    // Partial lane even though the pointer was never dereferenced there.
    // PURPOSE: the J predicate touches only Q and the two Coulomb density caps.
    const std::size_t ng = static_cast<std::size_t>(screen.density_ng);
    const double* const D = screen.group_pair_density_max;
    const std::size_t ga = screen.pair_group_left[bra_group];
    const std::size_t gb = screen.pair_group_right[bra_group];
    const std::size_t gc = screen.pair_group_left[ket_group];
    const std::size_t gd = screen.pair_group_right[ket_group];
    const double exch = std::max(
        std::max(D[gc * ng + ga], D[gd * ng + ga]),
        std::max(D[gc * ng + gb], D[gd * ng + gb]));
    return quad_est * std::max(coul, exch) >= screen.density_break;
}

// Exact shell-quartet mask for one four-lane packet, evaluated before roots,
// recurrence and spherical projection.  monotone_mask is deliberately kept
// separate from active_mask: callers may break only along the dimension they
// are traversing in descending Q order (bra packets in the specialized
// kernels, ket pairs in shell4).  For a non-monotone traversal it remains
// non-empty so an all-rejected packet can only be skipped, never terminate a
// tail.
inline unsigned rys_batch_screen_lane_mask(
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    std::size_t valid_lanes,
    std::uint32_t ket_group,
    bool tail_q_descending,
    unsigned& monotone_mask) noexcept {
    const RysBatchScreen* const screen = g_rys_batch_screen;
    const unsigned valid_mask = rys_low_bits(valid_lanes);
    if (screen == nullptr || screen->pair_group_metric == nullptr ||
        screen->q_break <= 0.0) {
        monotone_mask = valid_mask;
        return valid_mask;
    }

    unsigned active_mask = 0u;
    monotone_mask = tail_q_descending ? 0u : valid_mask;
    const double ket_q = screen->pair_group_metric[ket_group];

    // The density branch is HOISTED out of the lane loop on purpose.  With
    // density screening off, the same loop as always runs, so the code
    // generated for the fast paths is identical to the earlier code by
    // construction, not by trust.  This function is shared by the hottest
    // s/p families: adding a per-lane test here that is not always needed
    // is exactly how this engine has already lost fast paths on two
    // occasions.
    if (screen->shell_rho == nullptr) {
        for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
            const std::uint32_t bra_group =
                bra_bin.pair_group_ids[bra_pos + lane];
            const unsigned bit = 1u << static_cast<unsigned>(lane);
            if (screen->pair_group_metric[bra_group] * ket_q >=
                screen->q_break) {
                active_mask |= bit;
                if (tail_q_descending) monotone_mask |= bit;
            }
        }
        return active_mask;
    }

    // The bound uses 4*rho for the two Coulomb terms and the four crossed
    // maxima with no additional safety factor; it is the same CVHF scheme
    // documented in RysDensityScreenContext, not a factor of 6 for the number
    // of writes in the digestion.
    // The ket side is fixed for the call: its group indices and rho are hoisted.
    const double rho_cd = screen->shell_rho[ket_group];
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    if (screen->k_only) {
        const std::size_t ng = static_cast<std::size_t>(screen->density_ng);
        const double* const D = screen->group_pair_density_max;
        const std::size_t gc = screen->pair_group_left[ket_group];
        const std::size_t gd = screen->pair_group_right[ket_group];
        const double* const Dc = D + gc * ng;
        const double* const Dd = D + gd * ng;
        for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
            const std::uint32_t bra_group =
                bra_bin.pair_group_ids[bra_pos + lane];
            const unsigned bit = 1u << static_cast<unsigned>(lane);
            const double quad_est =
                screen->pair_group_metric[bra_group] * ket_q;
            if (quad_est < screen->q_break) continue;
            const std::size_t ga = screen->pair_group_left[bra_group];
            const std::size_t gb = screen->pair_group_right[bra_group];
            const double exch = std::max(
                std::max(Dc[ga], Dd[ga]),
                std::max(Dc[gb], Dd[gb]));
            if (quad_est * exch < screen->density_break) {
                if (tail_q_descending) monotone_mask |= bit;
                continue;
            }
            active_mask |= bit;
            if (tail_q_descending) monotone_mask |= bit;
        }
        return active_mask;
    }
#endif
    if (screen->j_only) {
        // WHAT: execute the vj-only local predicate in a branch hoisted outside
        // the four lanes.
        // WHY: even computing the exchange indices/loads here would preserve the
        // hot-path tax of K after its four contractions were removed.
        // PURPOSE: Partial J tasks test exactly q*4*max(rho_ab,rho_cd) and no K
        // metadata is touched.
        for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
            const std::uint32_t bra_group =
                bra_bin.pair_group_ids[bra_pos + lane];
            const unsigned bit = 1u << static_cast<unsigned>(lane);
            const double quad_est =
                screen->pair_group_metric[bra_group] * ket_q;
            if (quad_est < screen->q_break) continue;
            const double coul = 4.0 * std::max(
                screen->shell_rho[bra_group], rho_cd);
            if (quad_est * coul < screen->density_break) {
                if (tail_q_descending) monotone_mask |= bit;
                continue;
            }
            active_mask |= bit;
            if (tail_q_descending) monotone_mask |= bit;
        }
        return active_mask;
    }

    const std::size_t ng = static_cast<std::size_t>(screen->density_ng);
    const double* const D = screen->group_pair_density_max;
    const std::size_t gc = screen->pair_group_left[ket_group];
    const std::size_t gd = screen->pair_group_right[ket_group];
    const double* const Dc = D + gc * ng;
    const double* const Dd = D + gd * ng;
    for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
        const std::uint32_t bra_group =
            bra_bin.pair_group_ids[bra_pos + lane];
        const unsigned bit = 1u << static_cast<unsigned>(lane);
        const double quad_est =
            screen->pair_group_metric[bra_group] * ket_q;
        if (quad_est < screen->q_break) continue;
        const std::size_t ga = screen->pair_group_left[bra_group];
        const std::size_t gb = screen->pair_group_right[bra_group];
        const double coul =
            4.0 * std::max(screen->shell_rho[bra_group], rho_cd);
        const double exch = std::max(std::max(Dc[ga], Dd[ga]),
                                     std::max(Dc[gb], Dd[gb]));
        const double density_cap = std::max(coul, exch);
        if (quad_est * density_cap < screen->density_break) {
            // Rejected by density, but the quartet DOES pass the integral
            // bound: the monotonicity of the sweep in q says nothing about
            // the density, so this lane cannot cut off the tail.
            if (tail_q_descending) monotone_mask |= bit;
            continue;
        }
        active_mask |= bit;
        if (tail_q_descending) monotone_mask |= bit;
    }
    return active_mask;
}

inline unsigned rys_sp_shell_logical_bra4(
    std::size_t bra_pos,
    std::size_t valid_lanes,
    std::size_t ket_pos,
    bool same_bin) noexcept {
    unsigned logical_bits = rys_low_bits(valid_lanes);
    if (same_bin && ket_pos > bra_pos) {
        const std::size_t below_diagonal = ket_pos - bra_pos;
        if (below_diagonal >= 4u) {
            return 0u;
        }
        logical_bits &= ~rys_low_bits(below_diagonal);
    }
    return logical_bits;
}

inline unsigned rys_primitive_fast_survival_mask_at_threshold(
    __m256d bra_screen,
    __m256d ket_screen,
    unsigned logical_bits,
    double threshold_squared,
    int primitive_class) noexcept {
    const __m256d threshold = _mm256_set1_pd(threshold_squared);
    const unsigned live_bits = logical_bits &
        static_cast<unsigned>(_mm256_movemask_pd(_mm256_cmp_pd(
            _mm256_mul_pd(bra_screen, ket_screen),
            threshold, _CMP_GE_OQ)));

    RysSphericalScreeningDiagnostics* const diag =
        g_rys_spherical_screening_diagnostics;
    if (diag != nullptr) {
        const auto bit_count = [](unsigned bits) noexcept {
            return static_cast<std::uint64_t>(
                __builtin_popcount(bits));
        };
        const std::uint64_t logical = bit_count(logical_bits);
        const std::uint64_t live = bit_count(live_bits);
        ++diag->primitive_fast_vectors_tested;
        diag->primitive_fast_lanes_tested += logical;
        diag->primitive_fast_lanes_rejected_strict +=
            logical - live;
        if (live_bits == 0u) {
            ++diag->primitive_fast_vectors_rejected;
            diag->primitive_fast_lanes_rejected_simd +=
                logical;
        }
        if (primitive_class >= 0 && primitive_class < 6) {
            const std::size_t index =
                static_cast<std::size_t>(primitive_class);
            ++diag->primitive_fast_class_vectors_tested[index];
            diag->primitive_fast_class_lanes_tested[index] += logical;
            diag->primitive_fast_class_lanes_rejected_strict[index] +=
                logical - live;
            if (live_bits == 0u) {
                ++diag->primitive_fast_class_vectors_rejected[index];
                diag->primitive_fast_class_lanes_rejected_simd[index] +=
                    logical;
            }
        } else {
            ++diag->primitive_fast_df_vectors_tested;
            diag->primitive_fast_df_lanes_tested += logical;
            diag->primitive_fast_df_lanes_rejected_strict += logical - live;
            if (live_bits == 0u) {
                ++diag->primitive_fast_df_vectors_rejected;
                diag->primitive_fast_df_lanes_rejected_simd += logical;
            }
        }
    }
    return live_bits;
}

inline unsigned rys_primitive_fast_survival_mask(
    __m256d bra_screen,
    __m256d ket_screen,
    unsigned logical_bits) noexcept {
    const RysSpShellScreenContext* const screen =
        g_rys_sp_shell_screen_context;
    return rys_primitive_fast_survival_mask_at_threshold(
        bra_screen, ket_screen, logical_bits,
        screen->primitive_fastschwarz_threshold_squared,
        screen->primitive_fast_class);
}

inline double rys_df_primitive_fast_threshold_squared() noexcept {
    const RysBatchScreen* const screen = g_rys_batch_screen;
    return screen == nullptr
        ? 0.0
        : screen->primitive_fastschwarz_threshold_squared;
}

inline bool rys_df_primitive_fast_packet_dead(
    __m256d bra_screen,
    const RysLowPairBinH& ket_bin,
    std::size_t ket_primitive_offset,
    unsigned logical_bits,
    double threshold_squared) noexcept {
    const __m256d ket_screen = _mm256_set1_pd(
        ket_bin.primitive_fast_schwarz[ket_primitive_offset]);
    return rys_primitive_fast_survival_mask_at_threshold(
               bra_screen, ket_screen, logical_bits,
               threshold_squared, -1) == 0u;
}

inline void record_rys_sp_shell_mask(unsigned logical_bits,
                                     unsigned live_bits) noexcept {
    RysSphericalScreeningDiagnostics* const diag =
        g_rys_spherical_screening_diagnostics;
    if (diag == nullptr ||
        g_rys_sp_shell_screen_context == nullptr) {
        return;
    }
    ++diag->sp_masked_tiles;
    if (live_bits == 0u) {
        ++diag->sp_masked_tiles_rejected;
    }
    const auto bit_count = [](unsigned bits) noexcept {
        return static_cast<std::uint64_t>(
            __builtin_popcount(bits));
    };
    const std::uint64_t logical = bit_count(logical_bits);
    const std::uint64_t live = bit_count(live_bits);
    diag->sp_masked_shell_lanes += logical;
    diag->sp_masked_shell_lanes_accepted += live;
    diag->sp_masked_shell_lanes_rejected += logical - live;
}

// Four shell-pair lanes on the bra side and one scalar ket shell pair.
// logical_bits excludes padded lanes and, for a triangular same-bin tile,
// the lanes above the canonical pair-pair diagonal.
inline unsigned rys_sp_shell_mask_bra4(
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    std::size_t valid_lanes,
    const RysLowPairBinH& ket_bin,
    std::size_t ket_pos,
    bool same_bin,
    unsigned& q_live_bits) noexcept {
    const unsigned logical_bits = rys_sp_shell_logical_bra4(
        bra_pos, valid_lanes, ket_pos, same_bin);
    const RysSpShellScreenContext* const screen =
        g_rys_sp_shell_screen_context;
    if (screen == nullptr || logical_bits == 0u) {
        q_live_bits = logical_bits;
        return logical_bits;
    }
    q_live_bits = logical_bits;
    if (screen->threshold > 0.0) {
        // WHAT: compute the Q mask only when Schwarz left this task partial.
        // WHY/PURPOSE: a Q-WholeLive but density-Partial task already proved
        // all its lanes for Schwarz; repeating the SIMD load/mul/cmp here
        // before the scalar q*cap that density still needs was duplicate work.
        const __m256d bra_bound = _mm256_loadu_pd(
            bra_bin.shell_screen_bound.data() + bra_pos);
        const __m256d ket_bound = _mm256_set1_pd(
            ket_bin.shell_screen_bound[ket_pos]);
        const __m256d threshold =
            _mm256_set1_pd(screen->threshold);
        const unsigned significant = static_cast<unsigned>(
            _mm256_movemask_pd(_mm256_cmp_pd(
                _mm256_mul_pd(bra_bound, ket_bound),
                threshold, _CMP_GE_OQ)));
        q_live_bits &= significant;
    }
    unsigned live_bits = q_live_bits;
    if (screen->density_mask) {
        // WHAT: refine only the Q-live lanes with the local density bound.
        // WHY: q_live_bits is returned separately so that callers cut a tail
        // ordered only by Q; a zero caused by P may only skip (continue),
        // because density is not monotone in that order.
        const RysBatchScreen* const batch_screen = g_rys_batch_screen;
        const std::uint32_t ket_group =
            ket_bin.pair_group_ids[ket_pos];
        for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
            const unsigned bit = 1u << static_cast<unsigned>(lane);
            if ((live_bits & bit) != 0u &&
                !rys_density_shell_quartet_live(
                    *batch_screen,
                    bra_bin.pair_group_ids[bra_pos + lane],
                    ket_group)) {
                live_bits &= ~bit;
            }
        }
    }
    record_rys_sp_shell_mask(logical_bits, live_bits);
    return live_bits;
}

// WHAT: specialize and force-inline the screened four-bra mask used only by
// the J speed probe; its ss|ps tasks are cross-class, so no triangular test is
// needed and `logical_bits` is exactly the valid tail mask.
// WHY: the shared mask stayed outlined in all four Screened J instantiations,
// adding one call per packet even though its result is consumed before the
// three ERI accumulators become live.
// PURPOSE: the complete screened producer packet is a call-free leaf from
// mask through recurrence and final J writeback, without changing the shared
// ERI/G kernels or inflating each of their instantiations.
__attribute__((always_inline)) inline unsigned
rys_j_sp_shell_mask_bra4(
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    std::size_t valid_lanes,
    const RysLowPairBinH& ket_bin,
    std::size_t ket_pos,
    unsigned& q_live_bits) noexcept {
    const unsigned logical_bits = rys_low_bits(valid_lanes);
    const RysSpShellScreenContext* const screen =
        g_rys_sp_shell_screen_context;
    if (screen == nullptr) {
        q_live_bits = logical_bits;
        return logical_bits;
    }
    q_live_bits = logical_bits;
    if (screen->threshold > 0.0) {
        const __m256d bra_bound = _mm256_loadu_pd(
            bra_bin.shell_screen_bound.data() + bra_pos);
        const __m256d ket_bound = _mm256_set1_pd(
            ket_bin.shell_screen_bound[ket_pos]);
        const __m256d threshold = _mm256_set1_pd(screen->threshold);
        q_live_bits &= static_cast<unsigned>(_mm256_movemask_pd(
            _mm256_cmp_pd(_mm256_mul_pd(bra_bound, ket_bound), threshold,
                            _CMP_GE_OQ)));
    }
    unsigned live_bits = q_live_bits;
    if (screen->density_mask) {
        const RysBatchScreen* const batch_screen = g_rys_batch_screen;
        const std::uint32_t ket_group =
            ket_bin.pair_group_ids[ket_pos];
        const double ket_q =
            batch_screen->pair_group_metric[ket_group];
        const double* const shell_rho = batch_screen->shell_rho;
        const double rho_cd = shell_rho != nullptr
            ? shell_rho[ket_group]
            : 0.0;
        for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
            const unsigned bit = 1u << static_cast<unsigned>(lane);
            if ((live_bits & bit) == 0u) continue;
            const std::uint32_t bra_group =
                bra_bin.pair_group_ids[bra_pos + lane];
            const double quad_est =
                batch_screen->pair_group_metric[bra_group] * ket_q;
            if (quad_est < batch_screen->q_break) {
                live_bits &= ~bit;
                continue;
            }
            // Match the shared predicate's safe no-density behavior after
            // applying the integral screen; an incrementally constructed
            // context may not have shell_rho yet.
            if (shell_rho == nullptr) continue;
            const double coulomb_cap = 4.0 * std::max(
                shell_rho[bra_group], rho_cd);
            // WHAT: spell the J-only density bound directly in this
            // force-inlined packet mask.
            // WHY: calling the shared J/K predicate left four outlined calls
            // per Screened packet after the outer mask itself was inlined.
            // PURPOSE: the speed probe remains a true leaf and touches no K
            // endpoints or cross-density table while preserving exactly the
            // same q*4*max(rho_ab,rho_cd) decision.
            if (!(quad_est * coulomb_cap >=
                  batch_screen->density_break)) {
                live_bits &= ~bit;
            }
        }
    }
    record_rys_sp_shell_mask(logical_bits, live_bits);
    return live_bits;
}

// One scalar bra shell pair and four ket shell-pair lanes.  The arbitrary-depth
// ss|ss kernel uses this orientation for its tuned 1x16 path.
inline unsigned rys_sp_shell_mask_ket4(
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    const RysLowPairBinH& ket_bin,
    std::size_t ket_pos,
    std::size_t valid_lanes,
    unsigned& q_live_bits) noexcept {
    const unsigned logical_bits = rys_low_bits(valid_lanes);
    const RysSpShellScreenContext* const screen =
        g_rys_sp_shell_screen_context;
    if (screen == nullptr) {
        q_live_bits = logical_bits;
        return logical_bits;
    }
    q_live_bits = logical_bits;
    if (screen->threshold > 0.0) {
        // WHAT/WHY/PURPOSE: replicate the Q/P separation of bra4 in the ket4
        // orientation; density-only does not repeat the Schwarz SIMD mask,
        // although it keeps the scalar q product its own cap requires.
        const __m256d bra_bound = _mm256_set1_pd(
            bra_bin.shell_screen_bound[bra_pos]);
        const __m256d ket_bound = _mm256_loadu_pd(
            ket_bin.shell_screen_bound.data() + ket_pos);
        const __m256d threshold =
            _mm256_set1_pd(screen->threshold);
        const unsigned significant = static_cast<unsigned>(
            _mm256_movemask_pd(_mm256_cmp_pd(
                _mm256_mul_pd(bra_bound, ket_bound),
                threshold, _CMP_GE_OQ)));
        q_live_bits &= significant;
    }
    unsigned live_bits = q_live_bits;
    if (screen->density_mask) {
        // WHAT: apply the same refinement to the 1x4 orientation of ssss.
        // WHY: keeping q_live_bits separate prevents a packet dead only by
        // density from wrongly terminating the Q-descending tail.
        const RysBatchScreen* const batch_screen = g_rys_batch_screen;
        const std::uint32_t bra_group =
            bra_bin.pair_group_ids[bra_pos];
        for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
            const unsigned bit = 1u << static_cast<unsigned>(lane);
            if ((live_bits & bit) != 0u &&
                !rys_density_shell_quartet_live(
                    *batch_screen,
                    ket_bin.pair_group_ids[ket_pos + lane],
                    bra_group)) {
                live_bits &= ~bit;
            }
        }
    }
    record_rys_sp_shell_mask(logical_bits, live_bits);
    return live_bits;
}
#endif

inline std::uint64_t rys_spherical_bin_pair_work(
    const RysLowPairBinH& lhs,
    const RysLowPairBinH& rhs,
    bool triangular) noexcept {
    const std::uint64_t nl =
        static_cast<std::uint64_t>(lhs.pair_count);
    if (triangular) {
        return nl * (nl + 1u) / 2u;
    }
    return nl * static_cast<std::uint64_t>(rhs.pair_count);
}

// High-L shell quartets are much more expensive than the s/p kernels and need
// finer rejection resolution.  This remains a multiple of the four AVX2 bra
// lanes; it is intentionally independent of the validated low-L width above.
#ifndef TLNO_RYS_SPH_SCREEN_HIGH_L_CHUNK_PAIRS
#define TLNO_RYS_SPH_SCREEN_HIGH_L_CHUNK_PAIRS 8
#endif
inline constexpr std::size_t kRysScreenedSphericalHighLChunkPairs =
    TLNO_RYS_SPH_SCREEN_HIGH_L_CHUNK_PAIRS;

inline std::size_t rys_screened_spherical_high_l_chunk_pairs() {
    static const std::size_t chunk_pairs = []() {
        const char* const value =
            std::getenv("TLNO_RYS_SPH_SCREEN_HIGH_L_CHUNK_PAIRS");
        if (value == nullptr || value[0] == '\0') {
            return kRysScreenedSphericalHighLChunkPairs;
        }
        char* end = nullptr;
        const unsigned long long parsed =
            std::strtoull(value, &end, 10);
        if (end == value || *end != '\0' || parsed == 0u ||
            parsed > (1u << 20)) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: "
                "TLNO_RYS_SPH_SCREEN_HIGH_L_CHUNK_PAIRS must be an "
                "integer in [1, 1048576].");
        }
        return static_cast<std::size_t>(parsed);
    }();
    return chunk_pairs;
}

inline int rys_sp_pair_type(int left_L, int right_L) noexcept {
    if (left_L == 0 && right_L == 0) return 0;  // ss
    if (left_L == 1 && right_L == 0) return 1;  // ps (higher-L shell is left)
    if (left_L == 1 && right_L == 1) return 2;  // pp
    return -1;
}

// Triangular code for the ten oriented shell-pair classes through f:
// ss, ps, pp, ds, dp, dd, fs, fp, fd, ff.
inline int rys_spherical_pair_class_code(int left_L, int right_L) noexcept {
    if (left_L < right_L || left_L < 0 || left_L > 3 ||
        right_L < 0 || right_L > 3) {
        return -1;
    }
    return left_L * (left_L + 1) / 2 + right_L;
}

// (rys_sp_shell_batch_covers_pairpair removed 2026-08-04: nothing called it.
// It decided whether the s/p batch covered a pair of types from a
// `pp_shell_batch_mode` that came from the TLNO_RYS_PP_SHELL_BATCH environment
// knob.  In this tree the eight families are instantiated at compile time and
// that knob is read nowhere, so the function answered a question that no
// longer exists.  If deciding coverage per family is ever needed again, do it
// on the pair types, not on an environment mask.)

constexpr int kRysStatsLCount = kMaxLSingle + 1;
constexpr int kRysStatsBinCount =
    kRysStatsLCount * kRysStatsLCount * kRysStatsLCount * kRysStatsLCount;

inline int rys_stats_bin_index(int La, int Lb, int Lc, int Ld) noexcept {
    if (La < 0 || La > kMaxLSingle || Lb < 0 || Lb > kMaxLSingle ||
        Lc < 0 || Lc > kMaxLSingle || Ld < 0 || Ld > kMaxLSingle) {
        return -1;
    }
    return (((La * kRysStatsLCount + Lb) * kRysStatsLCount + Lc) *
                kRysStatsLCount +
            Ld);
}

inline void rys_stats_decode_bin(int idx, int& La, int& Lb, int& Lc, int& Ld) noexcept {
    Ld = idx % kRysStatsLCount;
    idx /= kRysStatsLCount;
    Lc = idx % kRysStatsLCount;
    idx /= kRysStatsLCount;
    Lb = idx % kRysStatsLCount;
    La = idx / kRysStatsLCount;
}

struct RysGeneratedScatterTerm {
    std::uint16_t t_index = 0;
    std::int8_t scale = 1;
    std::uint8_t ab_x = 0;
    std::uint8_t ab_y = 0;
    std::uint8_t ab_z = 0;
    std::uint8_t cd_x = 0;
    std::uint8_t cd_y = 0;
    std::uint8_t cd_z = 0;
};

struct RysGeneratedScatterEntry {
    std::uint8_t term_count = 0;
    RysGeneratedScatterTerm terms[2]{};
};

template <int La, int Lb, int Lc, int Ld>
struct RysRight01GeneratedMap {
    static constexpr bool available = false;
};

#include "integrals/two_electron/rys/rys_right01_maps.hpp"

inline void rys_generated_fill_sp_monomials(const double* v,
                                            double* monomials) noexcept {
    monomials[0] = 1.0;
    monomials[1] = v[2];
    monomials[4] = v[1];
    monomials[5] = v[1] * v[2];
    monomials[16] = v[0];
    monomials[17] = v[0] * v[2];
    monomials[20] = v[0] * v[1];
    monomials[21] = v[0] * v[1] * v[2];
}

inline double rys_generated_coeff_value(const RysGeneratedScatterTerm& term,
                                        const double* ab_monomials,
                                        const double* cd_monomials) noexcept {
    const int ab_idx = (static_cast<int>(term.ab_x) << 4) |
                       (static_cast<int>(term.ab_y) << 2) |
                       static_cast<int>(term.ab_z);
    const int cd_idx = (static_cast<int>(term.cd_x) << 4) |
                       (static_cast<int>(term.cd_y) << 2) |
                       static_cast<int>(term.cd_z);
    return static_cast<double>(term.scale) * ab_monomials[ab_idx] * cd_monomials[cd_idx];
}

inline int rys_small_binom(int n, int k) noexcept {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    return (n == 2 && k == 1) ? 2 : 1;
}

inline double rys_small_pow(double x, int n) noexcept {
    return (n == 0) ? 1.0 : ((n == 1) ? x : x * x);
}

inline int rys_small_binom_any(int n, int k) noexcept {
    if (k < 0 || k > n) return 0;
    if (k == 0 || k == n) return 1;
    if (n == 2) return 2;
    if (n == 3) return (k == 1 || k == 2) ? 3 : 1;
    int out = 1;
    for (int i = 1; i <= k; ++i) {
        out = out * (n - k + i) / i;
    }
    return out;
}

inline double rys_small_pow_any(double x, int n) noexcept {
    double out = 1.0;
    for (int i = 0; i < n; ++i) out *= x;
    return out;
}

struct SphProjectorTerm {
    int index = 0;
    double coeff = 0.0;
};

// Sparse transform for one final Cartesian shell pair (A,B) into the
// corresponding real-spherical pair. Unlike the radial-space projectors used
// by the shell4 backend, this acts after the Rys/HRR block is complete and
// remains the compact fallback for molecule-wide pure output.
struct FinalSphPairProjector {
    int cart_cols = 0;
    int sph_rows = 0;
    std::vector<std::vector<SphProjectorTerm>> rows;
};

FinalSphPairProjector build_final_sph_pair_projector(int left_L, int right_L) {
    FinalSphPairProjector out;
    const int right_cart = n_cart(right_L);
    const int right_sph = n_sph(right_L);
    out.cart_cols = n_cart(left_L) * right_cart;
    out.sph_rows = n_sph(left_L) * right_sph;
    out.rows.resize(static_cast<std::size_t>(out.sph_rows));

    const SphC2SBlock& left = kSphC2SBlocks[left_L];
    const SphC2SBlock& right = kSphC2SBlocks[right_L];
    for (int li = 0; li < left.count; ++li) {
        const SphC2STriple& le = left.data[li];
        const auto& lang = g_cart.at(offset_for_L(left_L) + le.cart);
        const double lc = le.coeff * angular_norm(lang[0], lang[1], lang[2]);
        for (int ri = 0; ri < right.count; ++ri) {
            const SphC2STriple& re = right.data[ri];
            const auto& rang = g_cart.at(offset_for_L(right_L) + re.cart);
            const int row = le.sph * right_sph + re.sph;
            out.rows[static_cast<std::size_t>(row)].push_back({
                le.cart * right_cart + re.cart,
                lc * re.coeff * angular_norm(rang[0], rang[1], rang[2]),
            });
        }
    }
    return out;
}

const FinalSphPairProjector& get_final_sph_pair_projector(int left_L,
                                                          int right_L) {
    static const std::array < FinalSphPairProjector,
                            (kMaxLSingle + 1) * (kMaxLSingle + 1) > cache = []() {
        std::array < FinalSphPairProjector,
                   (kMaxLSingle + 1) * (kMaxLSingle + 1) > out;
        for (int left = 0; left <= kMaxLSingle; ++left) {
            for (int right = 0; right <= kMaxLSingle; ++right) {
                out[static_cast<std::size_t>(left * (kMaxLSingle + 1) + right)] =
                    build_final_sph_pair_projector(left, right);
            }
        }
        return out;
    }();
    return cache[static_cast<std::size_t>(
        left_L * (kMaxLSingle + 1) + right_L)];
}

// Native production needs only its own pair layouts. The full representation
// remains available to private validation/experiment routes and memory audits.
#ifndef TLNO_NATIVE_LEAN_BASIS_PLAN
#if defined(TLNO_BUILD_RYS_VALIDATION) || defined(TLNO_BUILD_RYS_AUDIT)
#define TLNO_NATIVE_LEAN_BASIS_PLAN 0
#else
#define TLNO_NATIVE_LEAN_BASIS_PLAN 1
#endif
#endif
#if TLNO_NATIVE_LEAN_BASIS_PLAN != 0 && TLNO_NATIVE_LEAN_BASIS_PLAN != 1
#error "TLNO_NATIVE_LEAN_BASIS_PLAN must be 0 or 1"
#endif
#ifndef TLNO_NATIVE_SHARE_RIGHT_S_PROJECTORS
#define TLNO_NATIVE_SHARE_RIGHT_S_PROJECTORS 1
#endif
#if TLNO_NATIVE_SHARE_RIGHT_S_PROJECTORS != 0 && TLNO_NATIVE_SHARE_RIGHT_S_PROJECTORS != 1
#error "TLNO_NATIVE_SHARE_RIGHT_S_PROJECTORS must be 0 or 1"
#endif
#ifndef TLNO_NATIVE_PRUNE_DIRECT_BRA_PROJECTORS
#define TLNO_NATIVE_PRUNE_DIRECT_BRA_PROJECTORS 1
#endif
#if TLNO_NATIVE_PRUNE_DIRECT_BRA_PROJECTORS != 0 && TLNO_NATIVE_PRUNE_DIRECT_BRA_PROJECTORS != 1
#error "TLNO_NATIVE_PRUNE_DIRECT_BRA_PROJECTORS must be 0 or 1"
#endif

using RysRadialProjectorRowsH =
    std::vector<std::vector<SphProjectorTerm>>;

// A right-s ket projector contains angular coefficients only: every transfer
// exponent on that shell is zero, so the centres never enter its values.
// Read-only shared ownership avoids one nested allocation tree per shell pair
// while retaining the producer's original row and term iteration order.
class RysSharedRadialProjectorRowsH {
public:
    RysSharedRadialProjectorRowsH() = default;
    explicit RysSharedRadialProjectorRowsH(RysRadialProjectorRowsH&& rows)
        : rows_(std::make_shared<const RysRadialProjectorRowsH>(std::move(rows))) {}
    const std::vector<SphProjectorTerm>& operator[](std::size_t index) const noexcept {
        return (*rows_)[index];
    }
    std::size_t size() const noexcept { return rows_ ? rows_->size() : 0; }
    bool empty() const noexcept { return size() == 0; }
private:
    std::shared_ptr<const RysRadialProjectorRowsH> rows_;
};

using RysRadialKetProjectorRowsH = std::conditional_t <
    portable_kernel && TLNO_NATIVE_LEAN_BASIS_PLAN &&
        TLNO_NATIVE_SHARE_RIGHT_S_PROJECTORS,
    RysSharedRadialProjectorRowsH, RysRadialProjectorRowsH > ;

struct RysRadialPairProjectorH {
    // The bra form acts on radial rows with total degree La..La+Lb.
    // The ket form acts on the cumulative 0..Lc+Ld radial columns.
    std::vector<std::vector<SphProjectorTerm>> bra_rows;
    RysRadialKetProjectorRowsH ket_rows;
};

void build_pair_sph_projector_terms_runtime(
    const ContractedShellGroupH& gL,
    const ContractedShellGroupH& gR,
    const std::array<double, 3>& LR,
    bool bra_side,
    std::vector<std::vector<SphProjectorTerm>>& terms);

static bool same_exponents_h(const ContractedShellGroupH& g,
                              const tlno::core::basis::Shell& s) noexcept {
    if (g.exponents.size() != s.primitives.size()) return false;
    for (std::size_t i = 0; i < g.exponents.size(); ++i)
        if (g.exponents[i] != s.primitives[i].exponent) return false;
    return true;
}

inline bool same_radial_coefficient_h(double left, double right) noexcept {
    if (left == right) return true;
    const double scale = std::max(std::abs(left), std::abs(right));
    if (scale == 0.0) return true;
    constexpr double relative_tolerance =
        32.0 * std::numeric_limits<double>::epsilon();
    return std::abs(left - right) <= relative_tolerance * scale;
}

static bool same_radial_contraction_h(const ContractedShellGroupH& g,
                                      const tlno::core::basis::Shell& s) noexcept {
    if (!same_exponents_h(g, s) || g.angular.empty() ||
        g.coefficients.empty() ||
        g.coefficients.front().size() != s.primitives.size()) {
        return false;
    }
    const auto& rep_ang = g.angular.front();
    const double rep_norm =
        angular_norm(rep_ang[0], rep_ang[1], rep_ang[2]);
    const double shell_norm =
        angular_norm(s.ang[0], s.ang[1], s.ang[2]);
    const auto& rep_coefficients = g.coefficients.front();
    for (std::size_t i = 0; i < s.primitives.size(); ++i) {
        const double rep_radial = rep_coefficients[i] / rep_norm;
        const double shell_radial = s.primitives[i].coefficient / shell_norm;
        if (!same_radial_coefficient_h(rep_radial, shell_radial)) {
            return false;
        }
    }
    return true;
}

std::vector<ContractedShellGroupH>
build_groups_h(const std::vector<tlno::core::basis::Shell>& basis) {
    std::vector<ContractedShellGroupH> groups;
    for (int bi = 0; bi < static_cast<int>(basis.size()); ++bi) {
        const auto& s = basis[static_cast<std::size_t>(bi)];
        for (double coordinate : s.center) {
            if (!tlno::core::math::is_finite_binary64(coordinate)) {
                throw std::runtime_error(
                    "eri_ao_tensor_hgp: shell centers must be finite.");
            }
        }
        if (s.ang[0] < 0 || s.ang[1] < 0 || s.ang[2] < 0) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: Cartesian angular powers must be "
                "nonnegative.");
        }
        if (s.primitives.empty()) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: contracted shells need at least one "
                "primitive.");
        }
        for (const auto& primitive : s.primitives) {
            if (!tlno::core::math::is_finite_binary64(primitive.exponent) ||
                primitive.exponent <= 0.0 ||
                !tlno::core::math::is_finite_binary64(
                    primitive.coefficient)) {
                throw std::runtime_error(
                    "eri_ao_tensor_hgp: primitive exponents must be finite "
                    "and positive and coefficients must be finite.");
            }
        }
        const int L = s.ang[0] + s.ang[1] + s.ang[2];
        int gi = -1;
        for (int c = 0; c < static_cast<int>(groups.size()); ++c) {
            const auto& g = groups[static_cast<std::size_t>(c)];
            if (g.L == L && g.center == s.center &&
                same_radial_contraction_h(g, s)) {
                gi = c; break;
            }
        }
        if (gi < 0) {
            ContractedShellGroupH g;
            g.L = L;
            g.center = s.center;
            for (const auto& p : s.primitives) g.exponents.push_back(p.exponent);
            gi = static_cast<int>(groups.size());
            groups.push_back(std::move(g));
        }
        auto& g = groups[static_cast<std::size_t>(gi)];
        g.basis_indices.push_back(bi);
        g.angular.push_back(s.ang);
        std::vector<double> c;
        c.reserve(s.primitives.size());
        for (const auto& p : s.primitives) c.push_back(p.coefficient);
        g.coefficients.push_back(std::move(c));
    }

    // Derive c_radial and the per-component N_A factors.
    for (auto& g : groups) {
        const int count = static_cast<int>(g.angular.size());
        const int expected_count = n_cart(g.L);
        if (count != expected_count) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: each radial contraction must contain "
                "exactly one complete Cartesian shell.");
        }
        std::vector<bool> angular_seen(
            static_cast<std::size_t>(expected_count), false);
        g.n_ang.resize(static_cast<std::size_t>(count));
        for (int ca = 0; ca < count; ++ca) {
            const auto& ang = g.angular[static_cast<std::size_t>(ca)];
            if (ang[0] + ang[1] + ang[2] != g.L) {
                throw std::runtime_error(
                    "eri_ao_tensor_hgp: inconsistent Cartesian component "
                    "inside a contracted shell.");
            }
            const int local = triple_idx_within(g.L, ang[0], ang[1]);
            if (local < 0 || local >= expected_count ||
                angular_seen[static_cast<std::size_t>(local)]) {
                throw std::runtime_error(
                    "eri_ao_tensor_hgp: duplicate or invalid Cartesian "
                    "component inside a contracted shell.");
            }
            angular_seen[static_cast<std::size_t>(local)] = true;
            g.n_ang[static_cast<std::size_t>(ca)] = angular_norm(ang[0], ang[1], ang[2]);
        }
        // c_radial[ip] = coefficients[0][ip] / n_ang[0].
        // All components of a group share the same radial coefficient.
        const std::size_t np = g.exponents.size();
        g.c_radial.resize(np);
        const double inv_na = 1.0 / g.n_ang[0];
        for (std::size_t ip = 0; ip < np; ++ip) {
            g.c_radial[ip] = g.coefficients[0][ip] * inv_na;
        }
    }
    return groups;
}

struct NativeQuartetSchedule;

struct BasisDataHGP {
    mutable std::mutex native_schedule_mutex;
    mutable std::shared_ptr<const NativeQuartetSchedule> native_schedule;
    int basis_function_count = 0;
    int spherical_function_count = 0;
    std::vector<ContractedShellGroupH> groups;
    std::vector<int> spherical_group_bases;
    std::vector<std::pair<int, int>> pair_groups;
    // Density-screen metadata follows pair_groups exactly and is reused by
    // every SCF update.  Keeping the hot screen's structure-of-arrays view in
    // the immutable plan avoids rebuilding these two index vectors for each
    // new P or dP matrix.
    std::vector<std::uint32_t> pair_group_left;
    std::vector<std::uint32_t> pair_group_right;
    std::vector<std::vector<RysPrimitivePair>> rys_pair_primitives;
    std::vector<RysPrimitivePairSlab> rys_pair_slabs;
    std::array<std::vector<RysLowPairBinH>, 3> rys_low_pair_bins;
    std::vector<RysLowPairBinH> rys_ds_pair_bins;
    std::vector<RysLowPairBinH> rys_fs_pair_bins;
    std::vector<RysLowPairBinH> rys_dp_pair_bins;
    std::vector<RysLowPairBinH> rys_dd_pair_bins;
    std::array<std::vector<RysLowPairBinH>, 10> rys_all_pair_bins;
    // Cartesian benchmark ownership: low/low pair-group quartets belong to
    // the tuned s/p batches; each residual d-or-higher pair group owns all
    // remaining quartets that contain it. Both arrays are O(N_pair_groups)
    // plan metadata, never an O(N_shell^4) quartet list.
    std::vector<std::uint8_t> cartesian_pair_is_sp_batch;
    std::vector<std::uint32_t> cartesian_residual_pair_groups;
    std::vector<double> pair_decay_bounds;
    std::vector<std::vector<PackedPairComponentH>> packed_pair_components;
    std::vector<std::vector<PackedPairComponentH>> packed_spherical_pair_components;
    std::vector<RysJkPairInfo> jk_pair_info;
    std::vector<RysRadialPairProjectorH> rys_radial_pair_projectors;
    mutable std::mutex rys_spherical_schwarz_bounds_mutex;
    mutable std::shared_ptr<const RysSphericalSchwarzBounds>
        rys_spherical_schwarz_bounds;
    mutable std::mutex rys_spherical_sp_full_bins_mutex;
    mutable std::shared_ptr<const RysScreenedSpFullBins>
        rys_spherical_sp_full_bins;
    mutable std::mutex rys_spherical_decay_sp_full_bins_mutex;
    mutable std::shared_ptr<const RysScreenedSpFullBins>
        rys_spherical_decay_sp_full_bins;
    mutable std::mutex rys_spherical_screened_chunks_mutex;
    mutable std::shared_ptr<const RysScreenedSphericalChunks>
        rys_spherical_screened_chunks;
    mutable std::mutex rys_spherical_decay_screened_chunks_mutex;
    mutable std::shared_ptr<const RysScreenedSphericalChunks>
        rys_spherical_decay_screened_chunks;
};

// The low-L (s/p) batch kernels are shared between the cartesian dense builder
// and the pure-spherical stream.  Their packed pair components decide which
// canonical AO index each emitted value receives.  In pure-spherical mode the
// d shells renumber the AO space (5 vs 6 functions), so every pair_index after
// a d shell differs between the cartesian and spherical component tables; the
// values are identical (s/p cart == sph, angular norm 1), only the index space
// shifts.  This selector lets the spherical caller point the low-L kernels at
// the spherical table without disturbing the cartesian builder.  It is only
// ever set on the single thread that runs the spherical stream.
thread_local const std::vector<std::vector<PackedPairComponentH>>*
    g_low_batch_pair_components = nullptr;

inline const std::vector<std::vector<PackedPairComponentH>>&
low_batch_pair_components(const BasisDataHGP& bd) noexcept {
    return g_low_batch_pair_components != nullptr ? *g_low_batch_pair_components
                                                  : bd.packed_pair_components;
}

class LowBatchPairComponentsGuard {
public:
    LowBatchPairComponentsGuard(
        bool enabled,
        const std::vector<std::vector<PackedPairComponentH>>*
            pair_components) noexcept
        : active_(enabled),
          previous_(g_low_batch_pair_components) {
        if (active_) {
            g_low_batch_pair_components = pair_components;
        }
    }

    ~LowBatchPairComponentsGuard() {
        if (active_) {
            g_low_batch_pair_components = previous_;
        }
    }

    LowBatchPairComponentsGuard(
        const LowBatchPairComponentsGuard&) = delete;
    LowBatchPairComponentsGuard& operator = (
        const LowBatchPairComponentsGuard&) = delete;

private:
    bool active_;
    const std::vector<std::vector<PackedPairComponentH>>* previous_;
};

inline std::size_t spherical_packed_value_count(
    const BasisDataHGP& bd) noexcept {
    const std::size_t n =
        static_cast<std::size_t>(bd.spherical_function_count);
    const std::size_t pair_count = n * (n + 1u) / 2u;
    return pair_count * (pair_count + 1u) / 2u;
}



class EriStreamModeGuard {
public:
    EriStreamModeGuard(
        bool enabled,
        const std::vector<std::vector<PackedPairComponentH>>* pair_components,
        EriStreamAccumulator* accumulator) noexcept
        : active_(enabled),
          previous_accumulator_(g_eri_stream_accumulator),
          previous_pair_components_(g_low_batch_pair_components) {
        if (active_) {
            g_eri_stream_accumulator = accumulator;
            g_low_batch_pair_components = pair_components;
        }
    }

    ~EriStreamModeGuard() {
        if (active_) {
            g_eri_stream_accumulator = previous_accumulator_;
            g_low_batch_pair_components = previous_pair_components_;
        }
    }

    EriStreamModeGuard(const EriStreamModeGuard&) = delete;
    EriStreamModeGuard& operator = (const EriStreamModeGuard&) = delete;

private:
    bool active_;
    EriStreamAccumulator* previous_accumulator_;
    const std::vector<std::vector<PackedPairComponentH>>*
        previous_pair_components_;
};

struct GroupDecayScaleH {
    double alpha_min = 0.0;
    double K = 0.0;
};

GroupDecayScaleH build_group_decay_scale_h(const ContractedShellGroupH& g) {
    GroupDecayScaleH out;
    if (g.exponents.empty()) {
        return out;
    }
    out.alpha_min = g.exponents[0];
    for (std::size_t i = 0; i < g.exponents.size(); ++i) {
        const double alpha = g.exponents[i];
        out.alpha_min = std::min(out.alpha_min, alpha);
        out.K += std::abs(g.c_radial[i]) * std::pow(kPi / alpha, 0.75);
    }
    return out;
}

// Heuristic Gaussian shell-pair decay indicator.  It is intentionally kept
// separate from the rigorous contracted AO-pair Cauchy-Schwarz bounds: angular
// polynomial growth is not bounded here, so thresholds require empirical
// equal-error calibration against an exact materialized tensor.
double build_pair_decay_bound_h(const ContractedShellGroupH& gA,
                                const ContractedShellGroupH& gB,
                                const GroupDecayScaleH& a,
                                const GroupDecayScaleH& b) {
    if (!(a.alpha_min > 0.0) || !(b.alpha_min > 0.0)) {
        return 0.0;
    }
    const double dx = gA.center[0] - gB.center[0];
    const double dy = gA.center[1] - gB.center[1];
    const double dz = gA.center[2] - gB.center[2];
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double mu = a.alpha_min * b.alpha_min / (a.alpha_min + b.alpha_min);
    return a.K * b.K * std::exp(-mu * r2);
}

std::vector<RysPrimitivePair>
build_rys_primitive_pairs_h(const ContractedShellGroupH& gA,
                            const ContractedShellGroupH& gB) {
    const auto& A = gA.center;
    const auto& B = gB.center;
    const std::array<double, 3> AB = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
    const double rab2 = AB[0] * AB[0] + AB[1] * AB[1] + AB[2] * AB[2];
    const int npA = static_cast<int>(gA.exponents.size());
    const int npB = static_cast<int>(gB.exponents.size());

    std::vector<RysPrimitivePair> out;
    out.reserve(static_cast<std::size_t>(npA) * static_cast<std::size_t>(npB));
    for (int ipa = 0; ipa < npA; ++ipa) {
        const double alpha = gA.exponents[static_cast<std::size_t>(ipa)];
        const double cA_r  = gA.c_radial[static_cast<std::size_t>(ipa)];
        for (int ipb = 0; ipb < npB; ++ipb) {
            const double beta = gB.exponents[static_cast<std::size_t>(ipb)];
            const double p = alpha + beta;
            const double invp = 1.0 / p;
            const double cAB = cA_r * gB.c_radial[static_cast<std::size_t>(ipb)];
            if (cAB == 0.0) continue;
            const double K_AB = std::exp(-(alpha * beta * invp) * rab2);
            const double weight = K_AB * cAB * invp;
            const double fast_schwarz =
                kSimintFastSchwarzPrefactor *
                weight * weight * std::sqrt(invp);
            out.push_back({p, invp,
                           (alpha * A[0] + beta * B[0]) * invp,
                           (alpha * A[1] + beta * B[1]) * invp,
                           (alpha * A[2] + beta * B[2]) * invp,
                           weight, fast_schwarz});
        }
    }
    return out;
}

RysPrimitivePairSlab
build_rys_primitive_pair_slab_h(const std::vector<RysPrimitivePair>& pairs) {
    RysPrimitivePairSlab slab;
    const std::size_t n = pairs.size();
    slab.s.resize(n);
    slab.invs.resize(n);
    slab.Wx.resize(n);
    slab.Wy.resize(n);
    slab.Wz.resize(n);
    slab.w.resize(n);
    slab.fast_schwarz.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const RysPrimitivePair& p = pairs[i];
        slab.s[i] = p.s;
        slab.invs[i] = p.invs;
        slab.Wx[i] = p.Wx;
        slab.Wy[i] = p.Wy;
        slab.Wz[i] = p.Wz;
        slab.w[i] = p.w;
        slab.fast_schwarz[i] = p.fast_schwarz;
    }
    return slab;
}

std::vector<PackedPairComponentH>
build_packed_pair_components_h(const ContractedShellGroupH& left,
                               const ContractedShellGroupH& right) {
    const int count_left = static_cast<int>(left.basis_indices.size());
    const int count_right = static_cast<int>(right.basis_indices.size());
    const int right_block_count = n_cart(right.L);
    std::vector<PackedPairComponentH> refs;
    refs.reserve(static_cast<std::size_t>(count_left) * static_cast<std::size_t>(count_right));
    for (int ca = 0; ca < count_left; ++ca) {
        const int a = left.basis_indices[static_cast<std::size_t>(ca)];
        const auto& la = left.angular[static_cast<std::size_t>(ca)];
        const int la_local = triple_idx_within(left.L, la[0], la[1]);
        const double na_ca = left.n_ang[static_cast<std::size_t>(ca)];
        for (int cb = 0; cb < count_right; ++cb) {
            if (&left == &right && cb > ca) continue;
            const int b = right.basis_indices[static_cast<std::size_t>(cb)];
            const auto& lb = right.angular[static_cast<std::size_t>(cb)];
            const int lb_local = triple_idx_within(right.L, lb[0], lb[1]);
            const double norm = na_ca * right.n_ang[static_cast<std::size_t>(cb)];
            refs.push_back({
                EriPackedGAccessor::pair_index(a, b),
                static_cast<std::uint16_t>(
                    static_cast<std::size_t>(la_local) * static_cast<std::size_t>(right_block_count) +
                    static_cast<std::size_t>(lb_local)),
                norm,
            });
        }
    }
    return refs;
}

std::vector<PackedPairComponentH>
build_packed_spherical_pair_components_h(const ContractedShellGroupH& left,
                                         const ContractedShellGroupH& right,
                                         int left_base,
                                         int right_base) {
    const int count_left = n_sph(left.L);
    const int count_right = n_sph(right.L);
    std::vector<PackedPairComponentH> refs;
    refs.reserve(static_cast<std::size_t>(count_left) *
                 static_cast<std::size_t>(count_right));
    for (int sa = 0; sa < count_left; ++sa) {
        const int a = left_base + sa;
        for (int sb = 0; sb < count_right; ++sb) {
            if (&left == &right && sb > sa) continue;
            const int b = right_base + sb;
            refs.push_back({
                EriPackedGAccessor::pair_index(a, b),
                static_cast<std::uint16_t>(
                    static_cast<std::size_t>(sa) * static_cast<std::size_t>(count_right) +
                    static_cast<std::size_t>(sb)),
                1.0,
            });
        }
    }
    return refs;
}

std::shared_ptr<BasisDataHGP>
build_basis_data_h(const std::vector<tlno::core::basis::Shell>& basis) {
    constexpr bool native_only_plan =
        portable_kernel && TLNO_NATIVE_LEAN_BASIS_PLAN;
    auto d = std::make_shared<BasisDataHGP>();
    d->basis_function_count = static_cast<int>(basis.size());
    d->groups = build_groups_h(basis);
    const int ng = static_cast<int>(d->groups.size());
    d->spherical_group_bases.resize(static_cast<std::size_t>(ng));
    for (int gi = 0; gi < ng; ++gi) {
        const int L = d->groups[static_cast<std::size_t>(gi)].L;
        if (L < 0 || L > kMaxLSingle) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: pure-spherical stream supports shell L=0..4.");
        }
        d->spherical_group_bases[static_cast<std::size_t>(gi)] =
            d->spherical_function_count;
        d->spherical_function_count += n_sph(L);
    }
    const int M = d->spherical_function_count;
    if (M < 0 || static_cast<std::uint64_t>(M) > 65535u) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: direct J/K pair tables assume M <= 65535.");
    }
    const std::size_t m = static_cast<std::size_t>(M);
    const std::size_t pair_count = m * (m + 1u) / 2u;
    d->jk_pair_info.resize(pair_count);
    std::size_t pair = 0;
    for (int a = 0; a < M; ++a) {
        const auto a_row = static_cast<std::uint64_t>(a) *
                           static_cast<std::uint64_t>(M);
        for (int b = 0; b <= a; ++b, ++pair) {
            const auto b_row = static_cast<std::uint64_t>(b) *
                               static_cast<std::uint64_t>(M);
            RysJkPairInfo& info = d->jk_pair_info[pair];
            info.first_row = static_cast<std::uint32_t>(a_row);
            info.second_row = static_cast<std::uint32_t>(b_row);
            info.first = static_cast<std::uint16_t>(a);
            info.second = static_cast<std::uint16_t>(b);
            info.scale = (a == b) ? 0.5f : 1.0f;
        }
    }
    std::vector<GroupDecayScaleH> decay_scales;
    decay_scales.reserve(d->groups.size());
    for (const auto& group : d->groups) {
        decay_scales.push_back(build_group_decay_scale_h(group));
    }
    constexpr bool share_right_s_projectors =
        native_only_plan && TLNO_NATIVE_SHARE_RIGHT_S_PROJECTORS;
    std::vector<RysRadialKetProjectorRowsH> right_s_ket_by_group;
    if constexpr (share_right_s_projectors) {
        const auto right_s = std::find_if(
            d->groups.begin(), d->groups.end(),
            [](const ContractedShellGroupH& group) { return group.L == 0; });
        if (right_s != d->groups.end()) {
            right_s_ket_by_group.resize(d->groups.size());
            // Include the component normalization vector in the identity;
            // arbitrary accepted Cartesian component ordering stays exact.
            std::map < std::pair<int, std::vector<double>>,
                     RysRadialKetProjectorRowsH > templates;
            for (std::size_t gi = 0; gi < d->groups.size(); ++gi) {
                const auto& left = d->groups[gi];
                if (left.L > 3) continue;
                const auto key = std::make_pair(left.L, left.n_ang);
                auto found = templates.find(key);
                if (found == templates.end()) {
                    RysRadialProjectorRowsH rows;
                    const std::array<double, 3> unused_displacement{};
                    build_pair_sph_projector_terms_runtime(
                        left, *right_s, unused_displacement, false, rows);
                    found = templates.emplace(
                        key, RysRadialKetProjectorRowsH(std::move(rows))).first;
                }
                right_s_ket_by_group[gi] = found->second;
            }
        }
    }
    d->pair_groups.reserve(static_cast<std::size_t>(ng * (ng + 1) / 2));
    d->pair_group_left.reserve(
        static_cast<std::size_t>(ng * (ng + 1) / 2));
    d->pair_group_right.reserve(
        static_cast<std::size_t>(ng * (ng + 1) / 2));
    for (int lg = 0; lg < ng; ++lg) {
        for (int rg = 0; rg <= lg; ++rg) {
            int left = lg;
            int right = rg;
            // HGP transfers angular momentum from the right shell to the left
            // shell.  For each unordered shell pair, orient the higher-L shell
            // on the left to reduce HRR work without changing the integral.
            if (d->groups[static_cast<std::size_t>(right)].L >
                d->groups[static_cast<std::size_t>(left)].L) {
                std::swap(left, right);
            }
            d->pair_groups.emplace_back(left, right);
            d->pair_group_left.push_back(
                static_cast<std::uint32_t>(left));
            d->pair_group_right.push_back(
                static_cast<std::uint32_t>(right));
        }
    }
    d->rys_pair_primitives.reserve(d->pair_groups.size());
    if constexpr (!native_only_plan)
        d->rys_pair_slabs.reserve(d->pair_groups.size());
    d->pair_decay_bounds.reserve(d->pair_groups.size());
    if constexpr (!native_only_plan)
        d->packed_pair_components.reserve(d->pair_groups.size());
    d->packed_spherical_pair_components.reserve(d->pair_groups.size());
    d->rys_radial_pair_projectors.reserve(d->pair_groups.size());
    for (const auto& pg : d->pair_groups) {
        const auto& left = d->groups[static_cast<std::size_t>(pg.first)];
        const auto& right = d->groups[static_cast<std::size_t>(pg.second)];
        auto pairs = build_rys_primitive_pairs_h(left, right);
        if constexpr (!native_only_plan)
            d->rys_pair_slabs.push_back(build_rys_primitive_pair_slab_h(pairs));
        d->rys_pair_primitives.push_back(std::move(pairs));
        d->pair_decay_bounds.push_back(build_pair_decay_bound_h(
            left, right,
            decay_scales[static_cast<std::size_t>(pg.first)],
            decay_scales[static_cast<std::size_t>(pg.second)]));
        if constexpr (!native_only_plan)
            d->packed_pair_components.push_back(
                build_packed_pair_components_h(left, right));
        d->packed_spherical_pair_components.push_back(
            build_packed_spherical_pair_components_h(
                left, right,
                d->spherical_group_bases[static_cast<std::size_t>(pg.first)],
                d->spherical_group_bases[static_cast<std::size_t>(pg.second)]));

        RysRadialPairProjectorH radial_projector;
        if (left.L <= 3 && right.L <= 3) {
            const std::array<double, 3> LR = {
                left.center[0] - right.center[0],
                left.center[1] - right.center[1],
                left.center[2] - right.center[2],
            };
            // Native radial specialization places the larger angular class
            // on the bra. An s/p bra therefore always uses its closed kernel,
            // while s/p ket projectors remain needed by higher-L bra kernels.
            const bool direct_ds_bra = native_only_plan &&
                TLNO_NATIVE_PRUNE_DIRECT_BRA_PROJECTORS &&
                tlno::core::simd::lanes == 8 && left.L == 2 && right.L == 0;
            if (!native_only_plan || (left.L >= 2 && !direct_ds_bra))
                build_pair_sph_projector_terms_runtime(
                    left, right, LR, true, radial_projector.bra_rows);
            if (share_right_s_projectors && right.L == 0) {
                radial_projector.ket_rows =
                    right_s_ket_by_group[static_cast<std::size_t>(pg.first)];
            } else {
                RysRadialProjectorRowsH rows;
                build_pair_sph_projector_terms_runtime(
                    left, right, LR, false, rows);
                radial_projector.ket_rows =
                    RysRadialKetProjectorRowsH(std::move(rows));
            }
        }
        d->rys_radial_pair_projectors.push_back(
            std::move(radial_projector));
    }

    // NativePairBin::pack owns the only transposed view consumed by the native
    // producer. Avoid constructing every legacy specialized and universal
    // AoSoA view alongside it; none is consulted on the native production path.
    if constexpr (native_only_plan) return d;

    // Build AoSoA views of the ss, ps, and pp pair groups, grouped by
    // contraction length. Cache the complementary residual ownership at the
    // same O(N_pair_groups) plan-time pass, avoiding a dead low/low triangle in
    // the Cartesian molecular benchmark. Empty bins are deliberately retained
    // so the primitive count itself is the second-level bin index.
    std::array<std::size_t, 3> max_low_primitives = {0, 0, 0};
    d->cartesian_pair_is_sp_batch.assign(d->pair_groups.size(), 0u);
    d->cartesian_residual_pair_groups.reserve(d->pair_groups.size());
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        const int pair_type = rys_sp_pair_type(
            d->groups[static_cast<std::size_t>(left)].L,
            d->groups[static_cast<std::size_t>(right)].L);
        if (pair_type < 0) {
            d->cartesian_residual_pair_groups.push_back(
                static_cast<std::uint32_t>(pi));
            continue;
        }
        d->cartesian_pair_is_sp_batch[pi] = 1u;
        max_low_primitives[static_cast<std::size_t>(pair_type)] = std::max(
            max_low_primitives[static_cast<std::size_t>(pair_type)],
            d->rys_pair_slabs[pi].s.size());
    }
    for (int pair_type = 0; pair_type <= 2; ++pair_type) {
        d->rys_low_pair_bins[static_cast<std::size_t>(pair_type)].resize(
            max_low_primitives[static_cast<std::size_t>(pair_type)] + 1);
    }
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        const int pair_type = rys_sp_pair_type(
            d->groups[static_cast<std::size_t>(left)].L,
            d->groups[static_cast<std::size_t>(right)].L);
        if (pair_type < 0) continue;
        const std::size_t primitive_count = d->rys_pair_slabs[pi].s.size();
        if (primitive_count == 0) continue;
        d->rys_low_pair_bins[static_cast<std::size_t>(pair_type)][primitive_count]
            .pair_group_ids.push_back(static_cast<std::uint32_t>(pi));
    }
    for (int pair_type = 0; pair_type <= 2; ++pair_type) {
        auto& bins = d->rys_low_pair_bins[static_cast<std::size_t>(pair_type)];
        for (std::size_t primitive_count = 1;
             primitive_count < bins.size(); ++primitive_count) {
            auto& bin = bins[primitive_count];
            if (bin.pair_group_ids.empty()) continue;
            bin.primitive_count = static_cast<int>(primitive_count);
            bin.pair_count = bin.pair_group_ids.size();
            bin.stride = (bin.pair_count + 3u) & ~std::size_t{3u};
            bin.pair_group_ids.resize(bin.stride, bin.pair_group_ids.back());
            const std::size_t storage = primitive_count * bin.stride;
            bin.s.resize(storage);
            bin.invs.resize(storage);
            bin.Wx.resize(storage);
            bin.Wy.resize(storage);
            bin.Wz.resize(storage);
            bin.w.resize(storage);
            bin.primitive_fast_schwarz.resize(storage);
            bin.Lx.resize(storage);
            bin.Ly.resize(storage);
            bin.Lz.resize(storage);
            bin.Rx.resize(bin.stride);
            bin.Ry.resize(bin.stride);
            bin.Rz.resize(bin.stride);
            for (std::size_t lane_group = 0; lane_group < bin.stride;
                 ++lane_group) {
                const std::size_t pi =
                    static_cast<std::size_t>(bin.pair_group_ids[lane_group]);
                const auto& slab = d->rys_pair_slabs[pi];
                const int left_group = d->pair_groups[pi].first;
                const auto& left_center =
                    d->groups[static_cast<std::size_t>(left_group)].center;
                const int right_group = d->pair_groups[pi].second;
                const auto& right_center =
                    d->groups[static_cast<std::size_t>(right_group)].center;
                bin.Rx[lane_group] = left_center[0] - right_center[0];
                bin.Ry[lane_group] = left_center[1] - right_center[1];
                bin.Rz[lane_group] = left_center[2] - right_center[2];
                for (std::size_t ip = 0; ip < primitive_count; ++ip) {
                    const std::size_t dst = ip * bin.stride + lane_group;
                    bin.s[dst] = slab.s[ip];
                    bin.invs[dst] = slab.invs[ip];
                    bin.Wx[dst] = slab.Wx[ip];
                    bin.Wy[dst] = slab.Wy[ip];
                    bin.Wz[dst] = slab.Wz[ip];
                    bin.w[dst] = slab.w[ip];
                    bin.primitive_fast_schwarz[dst] =
                        slab.fast_schwarz[ip];
                    bin.Lx[dst] = slab.Wx[ip] - left_center[0];
                    bin.Ly[dst] = slab.Wy[ip] - left_center[1];
                    bin.Lz[dst] = slab.Wz[ip] - left_center[2];
                }
            }
        }
    }

    // Separate AoSoA bins for (d,s) pairs.  Keeping these out of the original
    // three low-L bins makes the established s/p traversal byte-for-byte
    // unchanged while enabling the pure-spherical ss|ds shellset kernel.
    std::size_t max_ds_primitives = 0;
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L == 2 &&
            d->groups[static_cast<std::size_t>(right)].L == 0) {
            max_ds_primitives = std::max(
                max_ds_primitives, d->rys_pair_slabs[pi].s.size());
        }
    }
    d->rys_ds_pair_bins.resize(max_ds_primitives + 1);
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L != 2 ||
            d->groups[static_cast<std::size_t>(right)].L != 0) {
            continue;
        }
        const std::size_t primitive_count = d->rys_pair_slabs[pi].s.size();
        if (primitive_count == 0) continue;
        d->rys_ds_pair_bins[primitive_count].pair_group_ids.push_back(
            static_cast<std::uint32_t>(pi));
    }
    for (std::size_t primitive_count = 1;
         primitive_count < d->rys_ds_pair_bins.size(); ++primitive_count) {
        auto& bin = d->rys_ds_pair_bins[primitive_count];
        if (bin.pair_group_ids.empty()) continue;
        bin.primitive_count = static_cast<int>(primitive_count);
        bin.pair_count = bin.pair_group_ids.size();
        bin.stride = (bin.pair_count + 3u) & ~std::size_t{3u};
        bin.pair_group_ids.resize(bin.stride, bin.pair_group_ids.back());
        const std::size_t storage = primitive_count * bin.stride;
        bin.s.resize(storage);
        bin.invs.resize(storage);
        bin.Wx.resize(storage);
        bin.Wy.resize(storage);
        bin.Wz.resize(storage);
        bin.w.resize(storage);
        bin.primitive_fast_schwarz.resize(storage);
        bin.Lx.resize(storage);
        bin.Ly.resize(storage);
        bin.Lz.resize(storage);
        bin.Rx.resize(bin.stride);
        bin.Ry.resize(bin.stride);
        bin.Rz.resize(bin.stride);
        for (std::size_t lane_group = 0; lane_group < bin.stride;
             ++lane_group) {
            const std::size_t pi =
                static_cast<std::size_t>(bin.pair_group_ids[lane_group]);
            const auto& slab = d->rys_pair_slabs[pi];
            const int left_group = d->pair_groups[pi].first;
            const auto& left_center =
                d->groups[static_cast<std::size_t>(left_group)].center;
            const int right_group = d->pair_groups[pi].second;
            const auto& right_center =
                d->groups[static_cast<std::size_t>(right_group)].center;
            bin.Rx[lane_group] = left_center[0] - right_center[0];
            bin.Ry[lane_group] = left_center[1] - right_center[1];
            bin.Rz[lane_group] = left_center[2] - right_center[2];
            for (std::size_t ip = 0; ip < primitive_count; ++ip) {
                const std::size_t dst = ip * bin.stride + lane_group;
                bin.s[dst] = slab.s[ip];
                bin.invs[dst] = slab.invs[ip];
                bin.Wx[dst] = slab.Wx[ip];
                bin.Wy[dst] = slab.Wy[ip];
                bin.Wz[dst] = slab.Wz[ip];
                bin.w[dst] = slab.w[ip];
                bin.primitive_fast_schwarz[dst] =
                    slab.fast_schwarz[ip];
                bin.Lx[dst] = slab.Wx[ip] - left_center[0];
                bin.Ly[dst] = slab.Wy[ip] - left_center[1];
                bin.Lz[dst] = slab.Wz[ip] - left_center[2];
            }
        }
    }

    // AoSoA bins for (f,s) pairs.  As with the d,s bins, equal primitive-pair
    // counts are grouped so each primitive index is a contiguous four-lane
    // vector across independent shellsets.
    std::size_t max_fs_primitives = 0;
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L == 3 &&
            d->groups[static_cast<std::size_t>(right)].L == 0) {
            max_fs_primitives = std::max(
                max_fs_primitives, d->rys_pair_slabs[pi].s.size());
        }
    }
    d->rys_fs_pair_bins.resize(max_fs_primitives + 1);
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L != 3 ||
            d->groups[static_cast<std::size_t>(right)].L != 0) {
            continue;
        }
        const std::size_t primitive_count = d->rys_pair_slabs[pi].s.size();
        if (primitive_count == 0) continue;
        d->rys_fs_pair_bins[primitive_count].pair_group_ids.push_back(
            static_cast<std::uint32_t>(pi));
    }
    for (std::size_t primitive_count = 1;
         primitive_count < d->rys_fs_pair_bins.size(); ++primitive_count) {
        auto& bin = d->rys_fs_pair_bins[primitive_count];
        if (bin.pair_group_ids.empty()) continue;
        bin.primitive_count = static_cast<int>(primitive_count);
        bin.pair_count = bin.pair_group_ids.size();
        bin.stride = (bin.pair_count + 3u) & ~std::size_t{3u};
        bin.pair_group_ids.resize(bin.stride, bin.pair_group_ids.back());
        const std::size_t storage = primitive_count * bin.stride;
        bin.s.resize(storage);
        bin.invs.resize(storage);
        bin.Wx.resize(storage);
        bin.Wy.resize(storage);
        bin.Wz.resize(storage);
        bin.w.resize(storage);
        bin.primitive_fast_schwarz.resize(storage);
        bin.Lx.resize(storage);
        bin.Ly.resize(storage);
        bin.Lz.resize(storage);
        bin.Rx.resize(bin.stride);
        bin.Ry.resize(bin.stride);
        bin.Rz.resize(bin.stride);
        for (std::size_t lane_group = 0; lane_group < bin.stride;
             ++lane_group) {
            const std::size_t pi =
                static_cast<std::size_t>(bin.pair_group_ids[lane_group]);
            const auto& slab = d->rys_pair_slabs[pi];
            const int left_group = d->pair_groups[pi].first;
            const auto& left_center =
                d->groups[static_cast<std::size_t>(left_group)].center;
            const int right_group = d->pair_groups[pi].second;
            const auto& right_center =
                d->groups[static_cast<std::size_t>(right_group)].center;
            bin.Rx[lane_group] = left_center[0] - right_center[0];
            bin.Ry[lane_group] = left_center[1] - right_center[1];
            bin.Rz[lane_group] = left_center[2] - right_center[2];
            for (std::size_t ip = 0; ip < primitive_count; ++ip) {
                const std::size_t dst = ip * bin.stride + lane_group;
                bin.s[dst] = slab.s[ip];
                bin.invs[dst] = slab.invs[ip];
                bin.Wx[dst] = slab.Wx[ip];
                bin.Wy[dst] = slab.Wy[ip];
                bin.Wz[dst] = slab.Wz[ip];
                bin.w[dst] = slab.w[ip];
                bin.primitive_fast_schwarz[dst] =
                    slab.fast_schwarz[ip];
                bin.Lx[dst] = slab.Wx[ip] - left_center[0];
                bin.Ly[dst] = slab.Wy[ip] - left_center[1];
                bin.Lz[dst] = slab.Wz[ip] - left_center[2];
            }
        }
    }

    // AoSoA bins for (d,p) pairs, again keyed by primitive-pair count so a
    // primitive index can be loaded directly across four shellsets.
    std::size_t max_dp_primitives = 0;
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L == 2 &&
            d->groups[static_cast<std::size_t>(right)].L == 1) {
            max_dp_primitives = std::max(
                max_dp_primitives, d->rys_pair_slabs[pi].s.size());
        }
    }
    d->rys_dp_pair_bins.resize(max_dp_primitives + 1);
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L != 2 ||
            d->groups[static_cast<std::size_t>(right)].L != 1) {
            continue;
        }
        const std::size_t primitive_count = d->rys_pair_slabs[pi].s.size();
        if (primitive_count == 0) continue;
        d->rys_dp_pair_bins[primitive_count].pair_group_ids.push_back(
            static_cast<std::uint32_t>(pi));
    }
    for (std::size_t primitive_count = 1;
         primitive_count < d->rys_dp_pair_bins.size(); ++primitive_count) {
        auto& bin = d->rys_dp_pair_bins[primitive_count];
        if (bin.pair_group_ids.empty()) continue;
        bin.primitive_count = static_cast<int>(primitive_count);
        bin.pair_count = bin.pair_group_ids.size();
        bin.stride = (bin.pair_count + 3u) & ~std::size_t{3u};
        bin.pair_group_ids.resize(bin.stride, bin.pair_group_ids.back());
        const std::size_t storage = primitive_count * bin.stride;
        bin.s.resize(storage);
        bin.invs.resize(storage);
        bin.Wx.resize(storage);
        bin.Wy.resize(storage);
        bin.Wz.resize(storage);
        bin.w.resize(storage);
        bin.primitive_fast_schwarz.resize(storage);
        bin.Lx.resize(storage);
        bin.Ly.resize(storage);
        bin.Lz.resize(storage);
        bin.Rx.resize(bin.stride);
        bin.Ry.resize(bin.stride);
        bin.Rz.resize(bin.stride);
        for (std::size_t lane_group = 0; lane_group < bin.stride;
             ++lane_group) {
            const std::size_t pi =
                static_cast<std::size_t>(bin.pair_group_ids[lane_group]);
            const auto& slab = d->rys_pair_slabs[pi];
            const int left_group = d->pair_groups[pi].first;
            const auto& left_center =
                d->groups[static_cast<std::size_t>(left_group)].center;
            const int right_group = d->pair_groups[pi].second;
            const auto& right_center =
                d->groups[static_cast<std::size_t>(right_group)].center;
            bin.Rx[lane_group] = left_center[0] - right_center[0];
            bin.Ry[lane_group] = left_center[1] - right_center[1];
            bin.Rz[lane_group] = left_center[2] - right_center[2];
            for (std::size_t ip = 0; ip < primitive_count; ++ip) {
                const std::size_t dst = ip * bin.stride + lane_group;
                bin.s[dst] = slab.s[ip];
                bin.invs[dst] = slab.invs[ip];
                bin.Wx[dst] = slab.Wx[ip];
                bin.Wy[dst] = slab.Wy[ip];
                bin.Wz[dst] = slab.Wz[ip];
                bin.w[dst] = slab.w[ip];
                bin.primitive_fast_schwarz[dst] =
                    slab.fast_schwarz[ip];
                bin.Lx[dst] = slab.Wx[ip] - left_center[0];
                bin.Ly[dst] = slab.Wy[ip] - left_center[1];
                bin.Lz[dst] = slab.Wz[ip] - left_center[2];
            }
        }
    }

    // AoSoA bins for (d,d) shell pairs.  These are the dominant pair type
    // left in the generic protein profile after the ds/dp shellset kernels.
    std::size_t max_dd_primitives = 0;
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L == 2 &&
            d->groups[static_cast<std::size_t>(right)].L == 2) {
            max_dd_primitives = std::max(
                max_dd_primitives, d->rys_pair_slabs[pi].s.size());
        }
    }
    d->rys_dd_pair_bins.resize(max_dd_primitives + 1);
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        if (d->groups[static_cast<std::size_t>(left)].L != 2 ||
            d->groups[static_cast<std::size_t>(right)].L != 2) {
            continue;
        }
        const std::size_t primitive_count = d->rys_pair_slabs[pi].s.size();
        if (primitive_count == 0) continue;
        d->rys_dd_pair_bins[primitive_count].pair_group_ids.push_back(
            static_cast<std::uint32_t>(pi));
    }
    for (std::size_t primitive_count = 1;
         primitive_count < d->rys_dd_pair_bins.size(); ++primitive_count) {
        auto& bin = d->rys_dd_pair_bins[primitive_count];
        if (bin.pair_group_ids.empty()) continue;
        bin.primitive_count = static_cast<int>(primitive_count);
        bin.pair_count = bin.pair_group_ids.size();
        bin.stride = (bin.pair_count + 3u) & ~std::size_t{3u};
        bin.pair_group_ids.resize(bin.stride, bin.pair_group_ids.back());
        const std::size_t storage = primitive_count * bin.stride;
        bin.s.resize(storage);
        bin.invs.resize(storage);
        bin.Wx.resize(storage);
        bin.Wy.resize(storage);
        bin.Wz.resize(storage);
        bin.w.resize(storage);
        bin.primitive_fast_schwarz.resize(storage);
        bin.Lx.resize(storage);
        bin.Ly.resize(storage);
        bin.Lz.resize(storage);
        bin.Rx.resize(bin.stride);
        bin.Ry.resize(bin.stride);
        bin.Rz.resize(bin.stride);
        for (std::size_t lane_group = 0; lane_group < bin.stride;
             ++lane_group) {
            const std::size_t pi =
                static_cast<std::size_t>(bin.pair_group_ids[lane_group]);
            const auto& slab = d->rys_pair_slabs[pi];
            const int left_group = d->pair_groups[pi].first;
            const auto& left_center =
                d->groups[static_cast<std::size_t>(left_group)].center;
            const int right_group = d->pair_groups[pi].second;
            const auto& right_center =
                d->groups[static_cast<std::size_t>(right_group)].center;
            bin.Rx[lane_group] = left_center[0] - right_center[0];
            bin.Ry[lane_group] = left_center[1] - right_center[1];
            bin.Rz[lane_group] = left_center[2] - right_center[2];
            for (std::size_t ip = 0; ip < primitive_count; ++ip) {
                const std::size_t dst = ip * bin.stride + lane_group;
                bin.s[dst] = slab.s[ip];
                bin.invs[dst] = slab.invs[ip];
                bin.Wx[dst] = slab.Wx[ip];
                bin.Wy[dst] = slab.Wy[ip];
                bin.Wz[dst] = slab.Wz[ip];
                bin.w[dst] = slab.w[ip];
                bin.primitive_fast_schwarz[dst] =
                    slab.fast_schwarz[ip];
                bin.Lx[dst] = slab.Wx[ip] - left_center[0];
                bin.Ly[dst] = slab.Wy[ip] - left_center[1];
                bin.Lz[dst] = slab.Wz[ip] - left_center[2];
            }
        }
    }

    // Universal s-through-f AoSoA view.  The specialized kernels above keep
    // their established layouts, while the generic backend uses this complete
    // ten-class index.  Storage remains O(N_shell^2): each primitive pair is
    // copied once into a class/contraction-length bin, never per quartet.
    std::array<std::size_t, 10> max_all_primitives{};
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        const int code = rys_spherical_pair_class_code(
            d->groups[static_cast<std::size_t>(left)].L,
            d->groups[static_cast<std::size_t>(right)].L);
        if (code < 0) continue;
        max_all_primitives[static_cast<std::size_t>(code)] =
            std::max(
                max_all_primitives[static_cast<std::size_t>(code)],
                d->rys_pair_slabs[pi].s.size());
    }
    for (int code = 0; code < 10; ++code) {
        d->rys_all_pair_bins[static_cast<std::size_t>(code)].resize(
            max_all_primitives[static_cast<std::size_t>(code)] + 1u);
    }
    for (std::size_t pi = 0; pi < d->pair_groups.size(); ++pi) {
        const auto [left, right] = d->pair_groups[pi];
        const int code = rys_spherical_pair_class_code(
            d->groups[static_cast<std::size_t>(left)].L,
            d->groups[static_cast<std::size_t>(right)].L);
        if (code < 0) continue;
        const std::size_t primitive_count = d->rys_pair_slabs[pi].s.size();
        if (primitive_count == 0) continue;
        d->rys_all_pair_bins[static_cast<std::size_t>(code)]
                            [primitive_count]
            .pair_group_ids.push_back(static_cast<std::uint32_t>(pi));
    }
    for (int code = 0; code < 10; ++code) {
        auto& bins =
            d->rys_all_pair_bins[static_cast<std::size_t>(code)];
        for (std::size_t primitive_count = 1;
             primitive_count < bins.size(); ++primitive_count) {
            auto& bin = bins[primitive_count];
            if (bin.pair_group_ids.empty()) continue;
            bin.primitive_count = static_cast<int>(primitive_count);
            bin.pair_count = bin.pair_group_ids.size();
            bin.stride = (bin.pair_count + 3u) & ~std::size_t{3u};
            bin.pair_group_ids.resize(
                bin.stride, bin.pair_group_ids.back());
            const std::size_t storage = primitive_count * bin.stride;
            bin.s.resize(storage);
            bin.invs.resize(storage);
            bin.Wx.resize(storage);
            bin.Wy.resize(storage);
            bin.Wz.resize(storage);
            bin.w.resize(storage);
            bin.primitive_fast_schwarz.resize(storage);
            bin.Lx.resize(storage);
            bin.Ly.resize(storage);
            bin.Lz.resize(storage);
            bin.Rx.resize(bin.stride);
            bin.Ry.resize(bin.stride);
            bin.Rz.resize(bin.stride);
            for (std::size_t lane_group = 0;
                 lane_group < bin.stride; ++lane_group) {
                const std::size_t pi =
                    static_cast<std::size_t>(
                        bin.pair_group_ids[lane_group]);
                const auto& slab = d->rys_pair_slabs[pi];
                const auto [left, right] = d->pair_groups[pi];
                const auto& left_center =
                    d->groups[static_cast<std::size_t>(left)].center;
                const auto& right_center =
                    d->groups[static_cast<std::size_t>(right)].center;
                bin.Rx[lane_group] =
                    left_center[0] - right_center[0];
                bin.Ry[lane_group] =
                    left_center[1] - right_center[1];
                bin.Rz[lane_group] =
                    left_center[2] - right_center[2];
                for (std::size_t ip = 0;
                     ip < primitive_count; ++ip) {
                    const std::size_t dst =
                        ip * bin.stride + lane_group;
                    bin.s[dst] = slab.s[ip];
                    bin.invs[dst] = slab.invs[ip];
                    bin.Wx[dst] = slab.Wx[ip];
                    bin.Wy[dst] = slab.Wy[ip];
                    bin.Wz[dst] = slab.Wz[ip];
                    bin.w[dst] = slab.w[ip];
                    bin.primitive_fast_schwarz[dst] =
                        slab.fast_schwarz[ip];
                    bin.Lx[dst] =
                        slab.Wx[ip] - left_center[0];
                    bin.Ly[dst] =
                        slab.Wy[ip] - left_center[1];
                    bin.Lz[dst] =
                        slab.Wz[ip] - left_center[2];
                }
            }
        }
    }

    return d;
}

std::shared_ptr<const BasisDataHGP> get_basis_data_h_shared(
    const std::vector<tlno::core::basis::Shell>& basis) {
    static std::mutex mtx;
    static std::unordered_map<std::string, std::shared_ptr<BasisDataHGP>> cache;

    std::string key;
    key.reserve(sizeof(std::uint64_t) + basis.size() * 80);
    const auto append_count = [&key](std::size_t count) {
        const std::uint64_t value = static_cast<std::uint64_t>(count);
        key.append(reinterpret_cast<const char*>(&value), sizeof(value));
    };
    append_count(basis.size());
    for (const auto& s : basis) {
        const char* pc = reinterpret_cast<const char*>(s.center.data());
        key.append(pc, 3 * sizeof(double));
        const char* pa = reinterpret_cast<const char*>(s.ang.data());
        key.append(pa, 3 * sizeof(int));
        append_count(s.primitives.size());
        for (const auto& pr : s.primitives) {
            const char* pe = reinterpret_cast<const char*>(&pr.exponent);
            const char* pf = reinterpret_cast<const char*>(&pr.coefficient);
            key.append(pe, sizeof(double));
            key.append(pf, sizeof(double));
        }
    }
    std::lock_guard<std::mutex> lk(mtx);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    auto d = build_basis_data_h(basis);
    auto out = std::shared_ptr<const BasisDataHGP>(d);
    cache.emplace(std::move(key), std::move(d));
    return out;
}

EriPackedPlanProfile build_packed_plan_profile_h(const BasisDataHGP& bd) {
    EriPackedPlanProfile profile;
    profile.M = bd.spherical_function_count;
    profile.shell_groups = static_cast<std::uint64_t>(bd.groups.size());
    profile.pair_groups = static_cast<std::uint64_t>(bd.pair_groups.size());
    profile.pair_group_quartets =
        profile.pair_groups * (profile.pair_groups + 1u) / 2u;
    profile.packed_pair_count =
        static_cast<std::uint64_t>(profile.M) * static_cast<std::uint64_t>(profile.M + 1) / 2;
    profile.packed_value_count =
        profile.packed_pair_count * (profile.packed_pair_count + 1u) / 2u;

    std::vector<EriPackedPlanClassStats> bins(static_cast<std::size_t>(kRysStatsBinCount));
    for (std::size_t pi = 0; pi < bd.rys_pair_primitives.size(); ++pi) {
        const std::uint64_t nprim = static_cast<std::uint64_t>(bd.rys_pair_primitives[pi].size());
        profile.primitive_pairs += nprim;
        profile.max_primitive_pairs_per_pair_group =
            std::max(profile.max_primitive_pairs_per_pair_group, nprim);
    }
    // Exact triangular class totals without materializing or walking the
    // O(N_shell^4) pair-pair list.  At pair-group ab, the prefix aggregates
    // contain precisely cd=0..ab, so each update below reproduces the former
    // nested traversal in O(N_pair_groups * Lmax^2) work and O(Lmax^2) memory.
    constexpr int kPairAngularClassCount =
        (kMaxLSingle + 1) * (kMaxLSingle + 1);
    std::array<std::uint64_t, kPairAngularClassCount> prefix_count{};
    std::array<std::uint64_t, kPairAngularClassCount> prefix_primitives{};
    std::array<std::uint64_t, kPairAngularClassCount> prefix_sph_refs{};
    const auto pair_angular_class = [&](std::size_t pair_group) noexcept {
        const auto [left, right] = bd.pair_groups[pair_group];
        const int left_l = bd.groups[static_cast<std::size_t>(left)].L;
        const int right_l = bd.groups[static_cast<std::size_t>(right)].L;
        return left_l * (kMaxLSingle + 1) + right_l;
    };
    for (std::size_t ab = 0; ab < bd.pair_groups.size(); ++ab) {
        const auto [lg, rg] = bd.pair_groups[ab];
        const auto& gA = bd.groups[static_cast<std::size_t>(lg)];
        const auto& gB = bd.groups[static_cast<std::size_t>(rg)];
        const int ab_class = pair_angular_class(ab);
        prefix_count[static_cast<std::size_t>(ab_class)] += 1u;
        prefix_primitives[static_cast<std::size_t>(ab_class)] +=
            static_cast<std::uint64_t>(bd.rys_pair_primitives[ab].size());
        prefix_sph_refs[static_cast<std::size_t>(ab_class)] +=
            static_cast<std::uint64_t>(
                bd.packed_spherical_pair_components[ab].size());

        const std::uint64_t ab_primitives =
            static_cast<std::uint64_t>(bd.rys_pair_primitives[ab].size());
        const std::uint64_t ab_refs =
            static_cast<std::uint64_t>(
                bd.packed_spherical_pair_components[ab].size());
        for (int cd_class = 0;
             cd_class < kPairAngularClassCount; ++cd_class) {
            const std::uint64_t count =
                prefix_count[static_cast<std::size_t>(cd_class)];
            if (count == 0u) continue;
            const int Lc = cd_class / (kMaxLSingle + 1);
            const int Ld = cd_class % (kMaxLSingle + 1);
            const int bin = rys_stats_bin_index(gA.L, gB.L, Lc, Ld);
            if (bin < 0) continue;
            auto& row = bins[static_cast<std::size_t>(bin)];
            row.pair_group_quartets += count;
            row.primitive_quartets +=
                ab_primitives *
                prefix_primitives[static_cast<std::size_t>(cd_class)];
            row.lhs_component_refs += ab_refs * count;
            row.rhs_component_refs +=
                prefix_sph_refs[static_cast<std::size_t>(cd_class)];
        }
    }
    for (int bin = 0; bin < kRysStatsBinCount; ++bin) {
        auto row = bins[static_cast<std::size_t>(bin)];
        if (row.pair_group_quartets == 0) {
            continue;
        }
        rys_stats_decode_bin(bin, row.la, row.lb, row.lc, row.ld);
        profile.class_stats.push_back(row);
    }
    return profile;
}

// ── HRR ket --------------------------------------------------------------
// Walks d from 0 to Ld on the contracted radial intermediate.
// Recurrence: (ab | c, d+1_i) = (ab | c+1_i, d) + (C_i - D_i)(ab | c, d).
//
// Input:  T_in[a_local * Nc_full + c_flat]   (single component pair, the
//         radial-only contracted [a 0 | c 0]).
// Output: K_out[a_local][c_local][d_local]
//   with |a| in [La, La+Lb] (a_local indexes the slice; total
//        Na_block_slice = sum_{l=La..La+Lb} n_cart(l)),
//   |c| = Lc (c_local in [0, n_cart(Lc))),
//   |d| = Ld (d_local in [0, n_cart(Ld))).
//
// Working buffer W[a_local][c_flat][d_flat] with d_flat covering
// |d| in [0, Ld] (cumulative).
void hrr_ket(int Lc, int Ld,
             const std::array<double, 3>& CmD,
             const double* T_in,
             int Nc_full,
             int Na_block_slice,
             std::vector<double>& K_out) {
    if (Ld == 0) {
        const int Nc_block = n_cart(Lc);
        const int c_offset = offset_for_L(Lc);
        K_out.assign(static_cast<std::size_t>(Na_block_slice) *
                         static_cast<std::size_t>(Nc_block),
                     0.0);
        for (int al = 0; al < Na_block_slice; ++al) {
            const double* src = T_in + static_cast<std::size_t>(al) *
                                           static_cast<std::size_t>(Nc_full) +
                                static_cast<std::size_t>(c_offset);
            double* dst = K_out.data() + static_cast<std::size_t>(al) *
                                             static_cast<std::size_t>(Nc_block);
            std::memcpy(dst, src, static_cast<std::size_t>(Nc_block) * sizeof(double));
        }
        return;
    }

    const int Lc_sum = Lc + Ld;
    const int Nc_cum = n_cart_cum(Lc_sum);
    const int Nd_cum = n_cart_cum(Ld);
    const std::size_t stride_a = static_cast<std::size_t>(Nc_cum) * static_cast<std::size_t>(Nd_cum);
    const std::size_t stride_c = static_cast<std::size_t>(Nd_cum);

    thread_local std::vector<double> W;
    W.assign(static_cast<std::size_t>(Na_block_slice) * stride_a, 0.0);

    // Seed: d_flat = 0 plane gets T_in.
    for (int al = 0; al < Na_block_slice; ++al) {
        const double* src = T_in + static_cast<std::size_t>(al) * static_cast<std::size_t>(Nc_full);
        double* dst_d0 = W.data() + static_cast<std::size_t>(al) * stride_a;
        for (int cf = 0; cf < Nc_cum; ++cf) {
            dst_d0[static_cast<std::size_t>(cf) * stride_c + 0] = src[cf];
        }
    }

    for (int Ld_cur = 0; Ld_cur < Ld; ++Ld_cur) {
        const int Ld_next = Ld_cur + 1;
        const int d_offset_next = offset_for_L(Ld_next);
        const int d_count_next = n_cart(Ld_next);
        const int c_max_next   = Lc_sum - Ld_next;

        for (int di = 0; di < d_count_next; ++di) {
            const int d_flat = d_offset_next + di;
            const auto& dtgt = g_cart.at(d_flat);
            int dir;
            if (dtgt[0] >= 1) dir = 0;
            else if (dtgt[1] >= 1) dir = 1;
            else dir = 2;
            const int dsrc_lx = dtgt[0] - (dir == 0 ? 1 : 0);
            const int dsrc_ly = dtgt[1] - (dir == 1 ? 1 : 0);
            const int dsrc_lz = dtgt[2] - (dir == 2 ? 1 : 0);
            const int dsrc_flat = cart_flat_idx(dsrc_lx, dsrc_ly, dsrc_lz);
            const double cmd_i = CmD[dir];

            for (int al = 0; al < Na_block_slice; ++al) {
                double* base_a = W.data() + static_cast<std::size_t>(al) * stride_a;
                for (int Lc_t = Lc; Lc_t <= c_max_next; ++Lc_t) {
                    const int ct_offset = offset_for_L(Lc_t);
                    const int ct_count  = n_cart(Lc_t);
                    for (int ci = 0; ci < ct_count; ++ci) {
                        const int ct_flat = ct_offset + ci;
                        const auto& ctgt = g_cart.at(ct_flat);
                        const int cup_lx = ctgt[0] + (dir == 0 ? 1 : 0);
                        const int cup_ly = ctgt[1] + (dir == 1 ? 1 : 0);
                        const int cup_lz = ctgt[2] + (dir == 2 ? 1 : 0);
                        const int cup_flat = cart_flat_idx(cup_lx, cup_ly, cup_lz);
                        const double v_up = base_a[static_cast<std::size_t>(cup_flat) * stride_c
                                                    + static_cast<std::size_t>(dsrc_flat)];
                        const double v_at = base_a[static_cast<std::size_t>(ct_flat) * stride_c
                                                    + static_cast<std::size_t>(dsrc_flat)];
                        base_a[static_cast<std::size_t>(ct_flat) * stride_c
                                + static_cast<std::size_t>(d_flat)] = v_up + cmd_i * v_at;
                    }
                }
            }
        }
    }

    // Extract |c| = Lc, |d| = Ld slice.
    const int Nc_block = n_cart(Lc);
    const int Nd_block = n_cart(Ld);
    const int c_offset = offset_for_L(Lc);
    const int d_offset = offset_for_L(Ld);
    K_out.assign(static_cast<std::size_t>(Na_block_slice) * static_cast<std::size_t>(Nc_block)
                     * static_cast<std::size_t>(Nd_block), 0.0);
    const std::size_t K_stride_a = static_cast<std::size_t>(Nc_block) * static_cast<std::size_t>(Nd_block);
    const std::size_t K_stride_c = static_cast<std::size_t>(Nd_block);
    for (int al = 0; al < Na_block_slice; ++al) {
        const double* Wal = W.data() + static_cast<std::size_t>(al) * stride_a;
        double* Kal = K_out.data() + static_cast<std::size_t>(al) * K_stride_a;
        for (int ci = 0; ci < Nc_block; ++ci) {
            for (int di = 0; di < Nd_block; ++di) {
                Kal[static_cast<std::size_t>(ci) * K_stride_c + static_cast<std::size_t>(di)] =
                    Wal[static_cast<std::size_t>(c_offset + ci) * stride_c
                        + static_cast<std::size_t>(d_offset + di)];
            }
        }
    }
}

// ── HRR bra --------------------------------------------------------------
// Walks b from 0 to Lb on the HRR-ket output.
// Recurrence: (a, b+1_i | cd) = (a+1_i, b | cd) + (A_i - B_i)(a, b | cd).
//
// Input:  K_in[a_local][cd_pair]  with a_local indexing
//         |a| in [La, La+Lb] (Na_block_slice entries),
//         cd_pair = c_local * Nd_block + d_local  (size Nc_block * Nd_block).
// Output: B_out[la_local][lb_local][c_local][d_local]
//         shape [Na_block, Nb_block, Nc_block, Nd_block].
void hrr_bra(int La, int Lb,
             const std::array<double, 3>& AmB,
             const std::vector<double>& K_in,
             int Nc_block, int Nd_block,
             std::vector<double>& out) {
    if (Lb == 0) {
        out = K_in;
        return;
    }

    const int La_sum = La + Lb;
    const int Na_cum = n_cart_cum(La_sum);
    const int Nb_cum = n_cart_cum(Lb);
    const int cd_size = Nc_block * Nd_block;
    const std::size_t stride_a = static_cast<std::size_t>(Nb_cum) * static_cast<std::size_t>(cd_size);
    const std::size_t stride_b = static_cast<std::size_t>(cd_size);

    thread_local std::vector<double> W;
    W.assign(static_cast<std::size_t>(Na_cum) * stride_a, 0.0);

    // Seed: K_in -> W[a_full][b=0][cd].
    {
        const int a_lo_offset = offset_for_L(La);
        const int Na_block_total = Na_cum - a_lo_offset;
        for (int al = 0; al < Na_block_total; ++al) {
            const int a_full = a_lo_offset + al;
            const double* src = K_in.data() + static_cast<std::size_t>(al) * static_cast<std::size_t>(cd_size);
            double* dst = W.data() + static_cast<std::size_t>(a_full) * stride_a;
            std::memcpy(dst, src, static_cast<std::size_t>(cd_size) * sizeof(double));
        }
    }

    for (int Lb_cur = 0; Lb_cur < Lb; ++Lb_cur) {
        const int Lb_next = Lb_cur + 1;
        const int b_offset_next = offset_for_L(Lb_next);
        const int b_count_next  = n_cart(Lb_next);
        const int a_max_next    = La_sum - Lb_next;

        for (int bi = 0; bi < b_count_next; ++bi) {
            const int b_flat = b_offset_next + bi;
            const auto& btgt = g_cart.at(b_flat);
            int dir;
            if (btgt[0] >= 1) dir = 0;
            else if (btgt[1] >= 1) dir = 1;
            else dir = 2;
            const int bsrc_lx = btgt[0] - (dir == 0 ? 1 : 0);
            const int bsrc_ly = btgt[1] - (dir == 1 ? 1 : 0);
            const int bsrc_lz = btgt[2] - (dir == 2 ? 1 : 0);
            const int bsrc_flat = cart_flat_idx(bsrc_lx, bsrc_ly, bsrc_lz);
            const double amb_i = AmB[dir];

            for (int La_t = La; La_t <= a_max_next; ++La_t) {
                const int at_offset = offset_for_L(La_t);
                const int at_count  = n_cart(La_t);
                for (int ai = 0; ai < at_count; ++ai) {
                    const int at_flat = at_offset + ai;
                    const auto& atgt = g_cart.at(at_flat);
                    const int aup_lx = atgt[0] + (dir == 0 ? 1 : 0);
                    const int aup_ly = atgt[1] + (dir == 1 ? 1 : 0);
                    const int aup_lz = atgt[2] + (dir == 2 ? 1 : 0);
                    const int aup_flat = cart_flat_idx(aup_lx, aup_ly, aup_lz);

                    double* dst = W.data() + static_cast<std::size_t>(at_flat) * stride_a
                                  + static_cast<std::size_t>(b_flat) * stride_b;
                    const double* up = W.data() + static_cast<std::size_t>(aup_flat) * stride_a
                                       + static_cast<std::size_t>(bsrc_flat) * stride_b;
                    const double* atv = W.data() + static_cast<std::size_t>(at_flat) * stride_a
                                        + static_cast<std::size_t>(bsrc_flat) * stride_b;
                    for (int cd = 0; cd < cd_size; ++cd) {
                        dst[cd] = up[cd] + amb_i * atv[cd];
                    }
                }
            }
        }
    }

    const int Na_block = n_cart(La);
    const int Nb_block = n_cart(Lb);
    const int a_offset = offset_for_L(La);
    const int b_offset = offset_for_L(Lb);
    out.assign(static_cast<std::size_t>(Na_block) * static_cast<std::size_t>(Nb_block)
                   * static_cast<std::size_t>(cd_size), 0.0);
    const std::size_t out_stride_a = static_cast<std::size_t>(Nb_block) * static_cast<std::size_t>(cd_size);
    const std::size_t out_stride_b = static_cast<std::size_t>(cd_size);
    for (int ai = 0; ai < Na_block; ++ai) {
        for (int bi = 0; bi < Nb_block; ++bi) {
            const double* src = W.data() + static_cast<std::size_t>(a_offset + ai) * stride_a
                                + static_cast<std::size_t>(b_offset + bi) * stride_b;
            double* dst = out.data() + static_cast<std::size_t>(ai) * out_stride_a
                          + static_cast<std::size_t>(bi) * out_stride_b;
            std::memcpy(dst, src, static_cast<std::size_t>(cd_size) * sizeof(double));
        }
    }
}

// ── Rys roots & weights -------------------------------------------------------
constexpr int kRysMaxRoots = 16;

void tridiag_gauss_rule(int N, const double* alpha, const double* beta,
                        double* roots, double* weights) noexcept {
    constexpr int kMaxDim = 2 * kRysMaxRoots;
    double A[kMaxDim * kMaxDim]{};
    double V[kMaxDim * kMaxDim]{};
    auto mat = [](double* M, int n, int r, int c) -> double& {
        return M[static_cast<std::size_t>(r) * static_cast<std::size_t>(n)
                 + static_cast<std::size_t>(c)];
    };

    for (int i = 0; i < N; ++i) {
        mat(A, N, i, i) = alpha[i];
        mat(V, N, i, i) = 1.0;
    }
    for (int i = 1; i < N; ++i) {
        const double off = std::sqrt(beta[i] > 0.0 ? beta[i] : 0.0);
        mat(A, N, i - 1, i) = off;
        mat(A, N, i, i - 1) = off;
    }

    constexpr double tol = 1.0e-15;
    for (int iter = 0; iter < 12 * N * N; ++iter) {
        int p = 0;
        int q = 1;
        double max_off = 0.0;
        for (int i = 0; i < N; ++i) {
            for (int j = i + 1; j < N; ++j) {
                const double aij = std::fabs(mat(A, N, i, j));
                if (aij > max_off) {
                    max_off = aij;
                    p = i;
                    q = j;
                }
            }
        }
        if (max_off <= tol) break;

        const double app = mat(A, N, p, p);
        const double aqq = mat(A, N, q, q);
        const double apq = mat(A, N, p, q);
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = std::copysign(1.0 / (std::fabs(tau) + std::hypot(tau, 1.0)), tau);
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        mat(A, N, p, p) = app - t * apq;
        mat(A, N, q, q) = aqq + t * apq;
        mat(A, N, p, q) = 0.0;
        mat(A, N, q, p) = 0.0;

        for (int k = 0; k < N; ++k) {
            if (k == p || k == q) continue;
            const double akp = mat(A, N, k, p);
            const double akq = mat(A, N, k, q);
            const double new_kp = c * akp - s * akq;
            const double new_kq = s * akp + c * akq;
            mat(A, N, k, p) = new_kp;
            mat(A, N, p, k) = new_kp;
            mat(A, N, k, q) = new_kq;
            mat(A, N, q, k) = new_kq;
        }
        for (int k = 0; k < N; ++k) {
            const double vkp = mat(V, N, k, p);
            const double vkq = mat(V, N, k, q);
            mat(V, N, k, p) = c * vkp - s * vkq;
            mat(V, N, k, q) = s * vkp + c * vkq;
        }
    }

    for (int i = 0; i < N; ++i) {
        roots[i] = mat(A, N, i, i);
        weights[i] = beta[0] * mat(V, N, 0, i) * mat(V, N, 0, i);
    }
    for (int i = 0; i < N - 1; ++i) {
        int k = i;
        double r = roots[i];
        for (int j = i + 1; j < N; ++j) {
            if (roots[j] < r) {
                k = j;
                r = roots[j];
            }
        }
        if (k != i) {
            std::swap(roots[k], roots[i]);
            std::swap(weights[k], weights[i]);
        }
    }
}

void rys_hermite_asymptotic_slow(int N, double T, double* roots, double* weights) noexcept {
    const int M = 2 * N;
    double alpha[2 * kRysMaxRoots]{};
    double beta[2 * kRysMaxRoots]{};
    double hroots[2 * kRysMaxRoots];
    double hweights[2 * kRysMaxRoots];
    beta[0] = std::sqrt(kPi);
    for (int i = 1; i < M; ++i) beta[i] = 0.5 * static_cast<double>(i);

    tridiag_gauss_rule(M, alpha, beta, hroots, hweights);

    const double invT = 1.0 / T;
    const double invSqrtT = std::sqrt(invT);
    int out = 0;
    for (int i = 0; i < M && out < N; ++i) {
        if (hroots[i] <= 0.0) continue;
        roots[out] = hroots[i] * hroots[i] * invT;
        weights[out] = hweights[i] * invSqrtT;
        ++out;
    }
}

struct RysHermiteAsymptoticTable {
    static constexpr int NMAX = 9;
    double roots_sq[NMAX + 1][NMAX]{};  // positive Hermite roots squared
    double weights[NMAX + 1][NMAX]{};   // corresponding half-line Hermite weights
    RysHermiteAsymptoticTable() noexcept {
        for (int N = 1; N <= NMAX; ++N) {
            const int M = 2 * N;
            double alpha[2 * kRysMaxRoots]{};
            double beta[2 * kRysMaxRoots]{};
            double hroots[2 * kRysMaxRoots]{};
            double hweights[2 * kRysMaxRoots]{};
            beta[0] = std::sqrt(kPi);
            for (int i = 1; i < M; ++i) beta[i] = 0.5 * static_cast<double>(i);
            tridiag_gauss_rule(M, alpha, beta, hroots, hweights);
            int out = 0;
            for (int i = 0; i < M && out < N; ++i) {
                if (hroots[i] <= 0.0) continue;
                roots_sq[N][out] = hroots[i] * hroots[i];
                weights[N][out] = hweights[i];
                ++out;
            }
        }
    }
};

static const RysHermiteAsymptoticTable g_rys_hermite_asymp;

template <int N>
inline void rys_hermite_asymptotic_table(double T, double* roots, double* weights) noexcept {
    static_assert(N >= 1 && N <= RysHermiteAsymptoticTable::NMAX,
                  "Rys asymptotic table order out of range");
    const double invT = 1.0 / T;
    const double invSqrtT = std::sqrt(invT);
    #pragma omp simd
    for (int i = 0; i < N; ++i) {
        roots[i] = g_rys_hermite_asymp.roots_sq[N][i] * invT;
        weights[i] = g_rys_hermite_asymp.weights[N][i] * invSqrtT;
    }
}

inline void rys_hermite_asymptotic(int N, double T, double* roots, double* weights) noexcept {
    switch (N) {
        case 1: rys_hermite_asymptotic_table<1>(T, roots, weights); return;
        case 2: rys_hermite_asymptotic_table<2>(T, roots, weights); return;
        case 3: rys_hermite_asymptotic_table<3>(T, roots, weights); return;
        case 4: rys_hermite_asymptotic_table<4>(T, roots, weights); return;
        case 5: rys_hermite_asymptotic_table<5>(T, roots, weights); return;
        case 6: rys_hermite_asymptotic_table<6>(T, roots, weights); return;
        case 7: rys_hermite_asymptotic_table<7>(T, roots, weights); return;
        case 8: rys_hermite_asymptotic_table<8>(T, roots, weights); return;
        case 9: rys_hermite_asymptotic_table<9>(T, roots, weights); return;
        default: rys_hermite_asymptotic_slow(N, T, roots, weights); return;
    }
}

// Auxiliary Fejér grid size for the discretized-Stieltjes construction.  This
// only runs while the root/weight interpolation table is being seeded at
// startup, so we can afford a generous grid for accuracy/stability.
constexpr int kStielMaxAux = 400;
int rys_aux_grid_size(int N) noexcept {
    const int n = 24 * N + 40;
    return n < kStielMaxAux ? n : kStielMaxAux;
}

double rys_asymptotic_threshold(int N) noexcept {
    static constexpr double xasymp[13] = {29.0, 37.0, 43.0, 49.0, 55.0, 60.0, 65.0,
                                          71.0, 76.0, 81.0, 86.0, 91.0, 96.0};
    return (N >= 1 && N <= 13) ? xasymp[N - 1] : (35.0 + 5.0 * static_cast<double>(N));
}

void fejer_positive_half(int nh, double* x, double* w) noexcept {
    const int n = 2 * nh;
    const double fn = static_cast<double>(n);
    for (int k = 1; k <= nh; ++k) {
        x[nh - k] = std::cos(0.5 * static_cast<double>(2 * k - 1) * kPi / fn);
    }
    for (int k = 0; k < nh; ++k) {
        double c1 = 1.0;
        double c0 = 2.0 * x[k] * x[k] - 1.0;
        const double t = 2.0 * c0;
        double sum = c0 / 3.0;
        for (int m = 2; m <= nh; ++m) {
            const double c2 = c1;
            c1 = c0;
            c0 = t * c1 - c2;
            sum += c0 / static_cast<double>(4 * m * m - 1);
        }
        w[k] = 2.0 * (1.0 - 2.0 * sum) / fn;
    }
}

bool stieltjes_recurrence(int N, int ncap, const double* x, const double* w,
                          double* alpha, double* beta) noexcept {
    double sum0 = 0.0;
    double sum1 = 0.0;
    for (int m = 0; m < ncap; ++m) {
        sum0 += w[m];
        sum1 += w[m] * x[m];
    }
    if (!(sum0 > 0.0) ||
        !tlno::core::math::is_finite_binary64(sum0)) return false;
    alpha[0] = sum1 / sum0;
    beta[0] = sum0;
    if (N == 1) return true;

    double p0[kStielMaxAux]{};
    double p1[kStielMaxAux]{};
    double p2[kStielMaxAux]{};
    for (int m = 0; m < ncap; ++m) p2[m] = 1.0;

    for (int k = 0; k < N - 1; ++k) {
        double next_sum0 = 0.0;
        double next_sum1 = 0.0;
        for (int m = 0; m < ncap; ++m) {
            if (w[m] == 0.0) continue;
            p0[m] = p1[m];
            p1[m] = p2[m];
            p2[m] = (x[m] - alpha[k]) * p1[m] - beta[k] * p0[m];
            const double t = w[m] * p2[m] * p2[m];
            next_sum0 += t;
            next_sum1 += t * x[m];
        }
        if (!(next_sum0 > 0.0) ||
            !tlno::core::math::is_finite_binary64(next_sum0)) return false;
        alpha[k + 1] = next_sum1 / next_sum0;
        beta[k + 1] = next_sum0 / sum0;
        sum0 = next_sum0;
    }
    return true;
}

bool rys_rule_basic_ok(int N, const double* roots, const double* weights) noexcept {
    double wsum = 0.0;
    double prev = -1.0;
    for (int i = 0; i < N; ++i) {
        const double r = roots[i];
        const double w = weights[i];
        if (!tlno::core::math::is_finite_binary64(r) ||
            !tlno::core::math::is_finite_binary64(w)) return false;
        if (r < -1.0e-12 || r > 1.0 + 1.0e-12) return false;
        if (i > 0 && r + 1.0e-12 < prev) return false;
        if (w < -1.0e-14) return false;
        prev = r;
        wsum += w;
    }
    return tlno::core::math::is_finite_binary64(wsum) &&
           wsum > 0.0 && wsum <= 1.0 + 1.0e-10;
}

bool rys_rule_moments_ok(int N, double T, const double* roots,
                         const double* weights, double rel_tol) noexcept {
    if (!rys_rule_basic_ok(N, roots, weights)) return false;

    double mu[2 * kRysMaxRoots]{};
    boys_fill_hgp(2 * N - 1, T, mu);
    for (int k = 0; k <= 2 * N - 1; ++k) {
        double q = 0.0;
        for (int i = 0; i < N; ++i) {
            double p = 1.0;
            for (int j = 0; j < k; ++j) p *= roots[i];
            q += weights[i] * p;
        }
        const double scale = std::max(1.0e-300, std::fabs(mu[k]));
        const double abs_tol = std::max(1.0e-13, rel_tol * scale);
        if (std::fabs(q - mu[k]) > abs_tol) return false;
    }
    return true;
}

bool rys_roots_weights_stieltjes(int N, double T, double* roots, double* weights) noexcept {
    if (T >= rys_asymptotic_threshold(N)) {
        rys_hermite_asymptotic(N, T, roots, weights);
        return rys_rule_moments_ok(N, T, roots, weights, 2.0e-11);
    }

    const int naux = rys_aux_grid_size(N);
    double tx[kStielMaxAux], tw[kStielMaxAux], xgrid[kStielMaxAux], wgrid[kStielMaxAux];
    fejer_positive_half(naux, tx, tw);
    for (int i = 0; i < naux; ++i) {
        const double t2 = tx[i] * tx[i];
        xgrid[i] = t2;
        wgrid[i] = tw[i] * std::exp(-T * t2);
    }

    double alpha[kRysMaxRoots]{};
    double beta[kRysMaxRoots]{};
    if (!stieltjes_recurrence(N, naux, xgrid, wgrid, alpha, beta)) return false;
    tridiag_gauss_rule(N, alpha, beta, roots, weights);
    return rys_rule_moments_ok(N, T, roots, weights, 2.0e-11);
}

// For argument T, returns N roots t_n^2 in [0,1) and weights w_n satisfying
//   ∫_0^1 f(t^2) e^{-T t^2} dt  =  Σ_n w_n f(t_n^2)   (exact for deg f ≤ 2N-1).
// The moments μ_k = ∫_0^1 t^{2k} e^{-T t^2} dt are exactly the Boys functions
// F_k(T).  Low orders use a fast moment path; higher orders switch to the
// stable discretized-Stieltjes construction used in Rys/Gautschi references.
void rys_roots_weights_moment(int N, double T, double* roots, double* weights) noexcept {
    double mu[2 * kRysMaxRoots];
    boys_fill_hgp(2 * N - 1, T, mu);
    if (N == 1) {
        roots[0] = mu[1] / mu[0];
        weights[0] = mu[0];
        return;
    }
    if (N == 2) {
        // Closed form for the 2x2 Jacobi matrix built from the first 4 moments.
        const double a0 = mu[1] / mu[0];
        const double s1 = mu[2] - a0 * mu[1];          // σ_1[1]
        const double s2 = mu[3] - a0 * mu[2];          // σ_1[2]
        const double a1 = s2 / s1 - mu[1] / mu[0];
        const double b1 = s1 / mu[0];
        const double half = 0.5 * (a0 + a1);
        const double diff = 0.5 * (a0 - a1);
        const double disc = std::sqrt(diff * diff + b1);
        const double r0 = half - disc;
        const double r1 = half + disc;
        // Eigenvector first component^2 for eigenvalue r: v0^2 = b1/(b1+(r-a0)^2).
        const double d0 = r0 - a0, d1 = r1 - a0;
        roots[0] = r0;
        roots[1] = r1;
        weights[0] = mu[0] * b1 / (b1 + d0 * d0);
        weights[1] = mu[0] * b1 / (b1 + d1 * d1);
        return;
    }
    const int nm = 2 * N;

    // Chebyshev algorithm: monic recurrence coefficients a_k (diag), b_k (>0).
    double a[kRysMaxRoots], b[kRysMaxRoots];
    double sig_pp[2 * kRysMaxRoots];  // σ_{k-2, l}
    double sig_p[2 * kRysMaxRoots];   // σ_{k-1, l}
    double sig_c[2 * kRysMaxRoots];   // σ_{k,   l}
    for (int l = 0; l < nm; ++l) { sig_pp[l] = 0.0; sig_p[l] = mu[l]; }
    a[0] = mu[1] / mu[0];
    b[0] = mu[0];
    for (int k = 1; k < N; ++k) {
        for (int l = k; l < nm - k; ++l) {
            sig_c[l] = sig_p[l + 1] - a[k - 1] * sig_p[l] - b[k - 1] * sig_pp[l];
        }
        a[k] = sig_c[k + 1] / sig_c[k] - sig_p[k] / sig_p[k - 1];
        b[k] = sig_c[k] / sig_p[k - 1];
        for (int l = 0; l < nm; ++l) { sig_pp[l] = sig_p[l]; sig_p[l] = sig_c[l]; }
    }

    tridiag_gauss_rule(N, a, b, roots, weights);
}

// High-accuracy generator: closed form (N<=2), else stable discretized
// Stieltjes (Chebyshev-moment fallback if that fails).  Used to *seed* the
// interpolation table below and as the fallback for orders beyond the
// tabulated range.  The Chebyshev-moment path is NOT used for N in [3,5]
// directly: it is numerically unstable there (root error O(10^2) at some T),
// which would poison the interpolation table.
void rys_roots_weights_accurate(int N, double T, double* roots, double* weights) noexcept {
    if (N <= 2) {
        rys_roots_weights_moment(N, T, roots, weights);
        return;
    }
    if (!rys_roots_weights_stieltjes(N, T, roots, weights)) {
        rys_roots_weights_moment(N, T, roots, weights);
    }
}

// ── Runtime Rys root/weight interpolation table ──────────────────────────────
// Rys roots t_n^2(T) and weights w_n(T) are smooth analytic functions of T.
// Below an N-dependent threshold we sample them on a uniform T grid at startup
// (via the accurate generator above) and evaluate per primitive quartet with a
// 5-point Newton-forward stencil — the same scheme used for the Boys table.
// Above the threshold the closed-form Hermite asymptotic is exact to working
// precision.  This replaces the per-quartet eigensolve, which profiling showed
// to be ~40% of the Rys kernel.  Nothing is hardcoded: every sample is computed
// at program startup from the orthogonal-polynomial recurrences.
// Interpolation table for orders N in [1, NMAX].  Roots and weights are stored
// k-major so each grid row is a contiguous root vector. Accuracy is validated
// offline; no per-quartet re-validation is needed on the hot path.
struct RysRootWeightTable {
    static constexpr int NMIN = 1;
    static constexpr int NMAX = 9;        // covers up to (gg|gg): nroots = 9
    static constexpr double DX = 0.01;
    static constexpr double INV_DX = 1.0 / DX;
    double tmax[NMAX + 1] = {0.0};
    std::vector<double> roots_tbl[NMAX + 1];   // [k*N + n]
    std::vector<double> wts_tbl[NMAX + 1];
    RysRootWeightTable() noexcept {
        for (int N = NMIN; N <= NMAX; ++N) {
            const double Tasymp = rys_asymptotic_threshold(N);
            tmax[N] = Tasymp;
            const int np = static_cast<int>(Tasymp * INV_DX) + 1;
            const int pad = np + 6;       // guard rows for the 5-point stencil
            roots_tbl[N].assign(static_cast<std::size_t>(pad) * N, 0.0);
            wts_tbl[N].assign(static_cast<std::size_t>(pad) * N, 0.0);
            double r[kRysMaxRoots], w[kRysMaxRoots];
            for (int k = 0; k <= np + 5; ++k) {
                rys_roots_weights_accurate(N, k * DX, r, w);
                // k-major layout [k*N + n]: all N roots at a grid point are
                // contiguous, so the 5-point interpolation vectorizes over roots.
                for (int n = 0; n < N; ++n) {
                    roots_tbl[N][static_cast<std::size_t>(k) * N + n] = r[n];
                    wts_tbl[N][static_cast<std::size_t>(k) * N + n]   = w[n];
                }
            }
        }
    }
};
static const RysRootWeightTable g_rys_tbl;

template <int N>
inline bool rys_roots_weights_table(double T, double* roots, double* weights) noexcept {
    static_assert(N >= RysRootWeightTable::NMIN && N <= RysRootWeightTable::NMAX,
                  "Rys table order out of range");
    if (T >= g_rys_tbl.tmax[N]) return false;
    const double fi = T * RysRootWeightTable::INV_DX;
    const int ki = static_cast<int>(fi);
    const double t = fi - ki;
    // k-major: grid rows ki..ki+4 are each a contiguous N-vector, so the whole
    // 4th-order Newton stencil runs SIMD across the N roots (and weights) at once.
    const double* rb = g_rys_tbl.roots_tbl[N].data() + static_cast<std::size_t>(ki) * N;
    const double* wb = g_rys_tbl.wts_tbl[N].data() + static_cast<std::size_t>(ki) * N;
    const double tm1 = t - 1.0, tm2 = t - 2.0, tm3 = t - 3.0;
#pragma omp simd
    for (int n = 0; n < N; ++n) {
        const double f0 = rb[n], f1 = rb[N + n], f2 = rb[2 * N + n], f3 = rb[3 * N + n], f4 = rb[4 * N + n];
        const double d1 = f1 - f0;
        const double d2 = (f2 - 2.0 * f1 + f0) * 0.5;
        const double d3 = (f3 - 3.0 * f2 + 3.0 * f1 - f0) * (1.0 / 6.0);
        const double d4 = (f4 - 4.0 * f3 + 6.0 * f2 - 4.0 * f1 + f0) * (1.0 / 24.0);
        roots[n] = f0 + t * (d1 + tm1 * (d2 + tm2 * (d3 + tm3 * d4)));
        const double g0 = wb[n], g1 = wb[N + n], g2 = wb[2 * N + n], g3 = wb[3 * N + n], g4 = wb[4 * N + n];
        const double e1 = g1 - g0;
        const double e2 = (g2 - 2.0 * g1 + g0) * 0.5;
        const double e3 = (g3 - 3.0 * g2 + 3.0 * g1 - g0) * (1.0 / 6.0);
        const double e4 = (g4 - 4.0 * g3 + 6.0 * g2 - 4.0 * g1 + g0) * (1.0 / 24.0);
        weights[n] = g0 + t * (e1 + tm1 * (e2 + tm2 * (e3 + tm3 * e4)));
    }
    return true;
}

template <int N>
inline void rys_roots_weights_fixed(double T, double* roots, double* weights) noexcept {
    static_assert(N >= RysRootWeightTable::NMIN && N <= RysRootWeightTable::NMAX,
                  "Rys fixed root order out of range");
    if (rys_roots_weights_table<N>(T, roots, weights)) return;
    rys_hermite_asymptotic_table<N>(T, roots, weights);
}

inline void rys_roots_weights(int N, double T, double* roots, double* weights) noexcept {
    switch (N) {
        case 1: rys_roots_weights_fixed<1>(T, roots, weights); return;
        case 2: rys_roots_weights_fixed<2>(T, roots, weights); return;
        case 3: rys_roots_weights_fixed<3>(T, roots, weights); return;
        case 4: rys_roots_weights_fixed<4>(T, roots, weights); return;
        case 5: rys_roots_weights_fixed<5>(T, roots, weights); return;
        case 6: rys_roots_weights_fixed<6>(T, roots, weights); return;
        case 7: rys_roots_weights_fixed<7>(T, roots, weights); return;
        case 8: rys_roots_weights_fixed<8>(T, roots, weights); return;
        case 9: rys_roots_weights_fixed<9>(T, roots, weights); return;
        default: break;
    }
    rys_roots_weights_accurate(N, T, roots, weights);
}

// Rebuild one AoSoA chunk from an explicit metric-ordered pair-group list.
// This plan/cache helper is portable; only its hot consumers require AVX2.
// The validated unscreened bins remain untouched.
inline void fill_rys_low_chunk(const BasisDataHGP& bd,
                               const std::uint32_t* pair_ids,
                               std::size_t count,
                               int primitive_count,
                               RysLowPairBinH& bin) {
    if (count == 0) return;
    bin.primitive_count = primitive_count;
    bin.pair_count = count;
    bin.stride = (count + 3u) & ~std::size_t{3u};
    bin.pair_group_ids.assign(pair_ids, pair_ids + count);
    // Pad the tail lanes by repeating the last pair, exactly as the plan bins
    // do; every kernel already guards padded lanes with pair_count.
    bin.pair_group_ids.resize(bin.stride, pair_ids[count - 1u]);

    const std::size_t nprim = static_cast<std::size_t>(primitive_count);
    const std::size_t storage = nprim * bin.stride;
    bin.s.assign(storage, 0.0);
    bin.invs.assign(storage, 0.0);
    bin.Wx.assign(storage, 0.0);
    bin.Wy.assign(storage, 0.0);
    bin.Wz.assign(storage, 0.0);
    bin.w.assign(storage, 0.0);
    bin.primitive_fast_schwarz.assign(storage, 0.0);
    bin.Lx.assign(storage, 0.0);
    bin.Ly.assign(storage, 0.0);
    bin.Lz.assign(storage, 0.0);
    bin.Rx.assign(bin.stride, 0.0);
    bin.Ry.assign(bin.stride, 0.0);
    bin.Rz.assign(bin.stride, 0.0);

    for (std::size_t lane = 0; lane < bin.stride; ++lane) {
        const std::size_t pi =
            static_cast<std::size_t>(bin.pair_group_ids[lane]);
        const auto& slab = bd.rys_pair_slabs[pi];
        const int left_group = bd.pair_groups[pi].first;
        const auto& left_center =
            bd.groups[static_cast<std::size_t>(left_group)].center;
        const int right_group = bd.pair_groups[pi].second;
        const auto& right_center =
            bd.groups[static_cast<std::size_t>(right_group)].center;
        bin.Rx[lane] = left_center[0] - right_center[0];
        bin.Ry[lane] = left_center[1] - right_center[1];
        bin.Rz[lane] = left_center[2] - right_center[2];
        for (std::size_t ip = 0; ip < nprim; ++ip) {
            const std::size_t dst = ip * bin.stride + lane;
            bin.s[dst] = slab.s[ip];
            bin.invs[dst] = slab.invs[ip];
            bin.Wx[dst] = slab.Wx[ip];
            bin.Wy[dst] = slab.Wy[ip];
            bin.Wz[dst] = slab.Wz[ip];
            bin.w[dst] = slab.w[ip];
            bin.primitive_fast_schwarz[dst] =
                slab.fast_schwarz[ip];
            bin.Lx[dst] = slab.Wx[ip] - left_center[0];
            bin.Ly[dst] = slab.Wy[ip] - left_center[1];
            bin.Lz[dst] = slab.Wz[ip] - left_center[2];
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
// 4x4 double transpose: rows a..d of 4 lanes -> per-column vectors r0..r3.
inline void rys_tp4(__m256d a, __m256d b, __m256d c, __m256d d,
                    __m256d& r0, __m256d& r1, __m256d& r2, __m256d& r3) noexcept {
    const __m256d t0 = _mm256_unpacklo_pd(a, b);
    const __m256d t1 = _mm256_unpackhi_pd(a, b);
    const __m256d t2 = _mm256_unpacklo_pd(c, d);
    const __m256d t3 = _mm256_unpackhi_pd(c, d);
    r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}

// Lane-parallel 5-point Newton-forward stencil.
inline __m256d rys_newton5v(__m256d f0, __m256d f1, __m256d f2, __m256d f3,
                            __m256d f4, __m256d t, __m256d tm1, __m256d tm2,
                            __m256d tm3) noexcept {
    const __m256d three = _mm256_set1_pd(3.0);
    const __m256d d1 = _mm256_sub_pd(f1, f0);
    const __m256d d2 = _mm256_mul_pd(
        _mm256_add_pd(_mm256_sub_pd(f2, _mm256_add_pd(f1, f1)), f0),
        _mm256_set1_pd(0.5));
    const __m256d d3 = _mm256_mul_pd(
        _mm256_sub_pd(_mm256_fmadd_pd(three, f1, f3), _mm256_fmadd_pd(three, f2, f0)),
        _mm256_set1_pd(1.0 / 6.0));
    const __m256d d4 = _mm256_mul_pd(
        _mm256_add_pd(_mm256_fmadd_pd(_mm256_set1_pd(6.0), f2, _mm256_add_pd(f4, f0)),
                      _mm256_mul_pd(_mm256_set1_pd(-4.0), _mm256_add_pd(f3, f1))),
        _mm256_set1_pd(1.0 / 24.0));
    return _mm256_fmadd_pd(
        t, _mm256_fmadd_pd(tm1, _mm256_fmadd_pd(tm2, _mm256_fmadd_pd(tm3, d4, d3), d2), d1),
        f0);
}

// Batched Rys table lookup: 4 quartets' roots/weights per call, lane-parallel
// Newton stencils over the k-major table (each lane's 5*N stencil values are
// contiguous). All-asymptotic packets use the same Hermite constants in SIMD;
// mixed packets retain the scalar repair of their asymptotic lanes. True means
// all outputs are ready, so callers need no scalar fallback. Output layout
// matches the SOA kernel: roots[n*4 + lane], weights[n*4 + lane].
template <int N>
inline bool rys_roots_weights_batch4(const double* __restrict targ,
                                     double* __restrict roots,
                                     double* __restrict weights) noexcept {
    static_assert(N >= 1 && N <= RysRootWeightTable::NMAX,
                  "batch4 table lookup order out of range");
    const double tm = g_rys_tbl.tmax[N];
    const __m256d target = _mm256_loadu_pd(targ);
    const __m256d outside = _mm256_cmp_pd(
        target, _mm256_set1_pd(tm), _CMP_GE_OQ);
    const int mask = _mm256_movemask_pd(outside);
    if (mask == 15) {
        // Lane-wise rys_hermite_asymptotic_table<N>, with the same division,
        // square root and root-major outputs as the former scalar fallback.
        const __m256d invT = _mm256_div_pd(_mm256_set1_pd(1.0), target);
        const __m256d invSqrtT = _mm256_sqrt_pd(invT);
        for (int n = 0; n < N; ++n) {
            _mm256_storeu_pd(roots + n * 4, _mm256_mul_pd(
                _mm256_set1_pd(g_rys_hermite_asymp.roots_sq[N][n]), invT));
            _mm256_storeu_pd(weights + n * 4, _mm256_mul_pd(
                _mm256_set1_pd(g_rys_hermite_asymp.weights[N][n]), invSqrtT));
        }
        return true;
    }
    const double* __restrict rtbl = g_rys_tbl.roots_tbl[N].data();
    const double* __restrict wtbl = g_rys_tbl.wts_tbl[N].data();
    // Preserve the same clamped target, truncated grid cell and fraction as
    // the scalar preparation; only their four-lane evaluation changes.
    const __m256d safe_target = _mm256_blendv_pd(
        target, _mm256_set1_pd(tm - 0.1), outside);
    const __m256d fi = _mm256_mul_pd(
        safe_target, _mm256_set1_pd(RysRootWeightTable::INV_DX));
    const __m128i ki = _mm256_cvttpd_epi32(fi);
    const __m256d t = _mm256_sub_pd(fi, _mm256_cvtepi32_pd(ki));
    int base[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(base),
        _mm_mullo_epi32(ki, _mm_set1_epi32(N)));
    const __m256d tm1 = _mm256_sub_pd(t, _mm256_set1_pd(1.0));
    const __m256d tm2 = _mm256_sub_pd(t, _mm256_set1_pd(2.0));
    const __m256d tm3 = _mm256_sub_pd(t, _mm256_set1_pd(3.0));
    // Each lane's 5-point stencil spans 5*N contiguous doubles (k-major table,
    // flat index j = row*N + root). Transpose 4-wide chunks across lanes into
    // per-j lane vectors, fill the remainder with set_pd, then run one
    // lane-parallel Newton stencil per root. All loops unroll (N constexpr).
    constexpr int total = 5 * N;
    constexpr int nchunk = total / 4;
    __m256d R[total], W[total];
    for (int c = 0; c < nchunk; ++c) {
        rys_tp4(_mm256_loadu_pd(rtbl + base[0] + 4 * c),
                _mm256_loadu_pd(rtbl + base[1] + 4 * c),
                _mm256_loadu_pd(rtbl + base[2] + 4 * c),
                _mm256_loadu_pd(rtbl + base[3] + 4 * c),
                R[4 * c], R[4 * c + 1], R[4 * c + 2], R[4 * c + 3]);
        rys_tp4(_mm256_loadu_pd(wtbl + base[0] + 4 * c),
                _mm256_loadu_pd(wtbl + base[1] + 4 * c),
                _mm256_loadu_pd(wtbl + base[2] + 4 * c),
                _mm256_loadu_pd(wtbl + base[3] + 4 * c),
                W[4 * c], W[4 * c + 1], W[4 * c + 2], W[4 * c + 3]);
    }
    for (int j = nchunk * 4; j < total; ++j) {
        R[j] = _mm256_set_pd(rtbl[base[3] + j], rtbl[base[2] + j],
                             rtbl[base[1] + j], rtbl[base[0] + j]);
        W[j] = _mm256_set_pd(wtbl[base[3] + j], wtbl[base[2] + j],
                             wtbl[base[1] + j], wtbl[base[0] + j]);
    }
    for (int n = 0; n < N; ++n) {
        _mm256_storeu_pd(roots + n * 4,
            rys_newton5v(R[n], R[N + n], R[2 * N + n],
                         R[3 * N + n], R[4 * N + n],
                         t, tm1, tm2, tm3));
        _mm256_storeu_pd(weights + n * 4,
            rys_newton5v(W[n], W[N + n], W[2 * N + n],
                         W[3 * N + n], W[4 * N + n],
                         t, tm1, tm2, tm3));
    }
    if (mask != 0) {
        for (int l = 0; l < 4; ++l) {
            if (mask & (1 << l)) {
                double lane_roots[N];
                double lane_weights[N];
                rys_roots_weights_fixed<N>(targ[l], lane_roots, lane_weights);
                for (int n = 0; n < N; ++n) {
                    roots[n * 4 + l] = lane_roots[n];
                    weights[n * 4 + l] = lane_weights[n];
                }
            }
        }
    }
    return true;
}

#if defined(TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE) && \
    !defined(TLNO_J_SPEED_PROBE_EXACT_ROOTS)
#error "TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE requires TLNO_J_SPEED_PROBE_EXACT_ROOTS"
#endif

#ifdef TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE
struct RysJExactN1LagrangeBasis {
    __m256d l0;
    __m256d l1;
    __m256d l2;
    __m256d l3;
    __m256d l4;
};

__attribute__((always_inline)) inline RysJExactN1LagrangeBasis
rys_j_exact_n1_lagrange_basis(__m256d t) noexcept {
    const __m256d b2 = _mm256_mul_pd(
        _mm256_mul_pd(t, _mm256_sub_pd(t, _mm256_set1_pd(1.0))),
        _mm256_set1_pd(0.5));
    const __m256d b3 = _mm256_mul_pd(
        _mm256_mul_pd(b2, _mm256_sub_pd(t, _mm256_set1_pd(2.0))),
        _mm256_set1_pd(1.0 / 3.0));
    const __m256d b4 = _mm256_mul_pd(
        _mm256_mul_pd(b3, _mm256_sub_pd(t, _mm256_set1_pd(3.0))),
        _mm256_set1_pd(0.25));
    const __m256d b4x2 = _mm256_add_pd(b4, b4);
    const __m256d b4x4 = _mm256_add_pd(b4x2, b4x2);
    RysJExactN1LagrangeBasis basis;
    basis.l4 = b4;
    basis.l3 = _mm256_sub_pd(b3, b4x4);
    basis.l2 = _mm256_add_pd(
        _mm256_add_pd(b4x4, b4x2),
        _mm256_fnmadd_pd(_mm256_set1_pd(3.0), b3, b2));
    basis.l1 = _mm256_fnmadd_pd(
        _mm256_set1_pd(4.0), b4,
        _mm256_fmadd_pd(
            _mm256_set1_pd(3.0), b3,
            _mm256_fnmadd_pd(_mm256_set1_pd(2.0), b2, t)));
    basis.l0 = _mm256_add_pd(
        _mm256_sub_pd(
            _mm256_add_pd(
                _mm256_sub_pd(_mm256_set1_pd(1.0), t), b2),
            b3),
        b4);
    return basis;
}

__attribute__((always_inline)) inline __m256d
rys_j_exact_n1_lagrange_table_value(
    const double* __restrict table,
    const int (&base)[4],
    const RysJExactN1LagrangeBasis& basis) noexcept {
    __m256d f0, f1, f2, f3;
    rys_tp4(
        _mm256_loadu_pd(table + base[0]),
        _mm256_loadu_pd(table + base[1]),
        _mm256_loadu_pd(table + base[2]),
        _mm256_loadu_pd(table + base[3]),
        f0, f1, f2, f3);
    const __m256d f4 = _mm256_set_pd(
        table[base[3] + 4], table[base[2] + 4],
        table[base[1] + 4], table[base[0] + 4]);
    return _mm256_fmadd_pd(
        basis.l4, f4,
        _mm256_fmadd_pd(
            basis.l3, f3,
            _mm256_fmadd_pd(
                basis.l2, f2,
                _mm256_fmadd_pd(
                    basis.l1, f1, _mm256_mul_pd(basis.l0, f0)))));
}

// WHAT: evaluate both N=1 outputs from the established five table samples
// with direct cardinal Lagrange coefficients.
// WHY: the generic Newton body constructs two five-vector scratch families;
// the J-only exact experiment needs one root and one weight and can stream
// each family through the same five coefficients.
// PURPOSE: preserve the frozen N=1 tmax=29, safe mixed-lane target and
// fixed<1> asymptotic repair without changing any table or non-J policy.
__attribute__((noinline, hot)) void
rys_j_exact_n1_batch4_lagrange(
    const double* __restrict targ,
    double* __restrict roots,
    double* __restrict weights) noexcept {
    constexpr int N = 1;
    const double tm = g_rys_tbl.tmax[N];
    int mask = 0;
    if (!(targ[0] < tm && targ[1] < tm &&
          targ[2] < tm && targ[3] < tm)) {
        mask = (targ[0] >= tm ? 1 : 0) |
               (targ[1] >= tm ? 2 : 0) |
               (targ[2] >= tm ? 4 : 0) |
               (targ[3] >= tm ? 8 : 0);
        if (mask == 15) {
            for (int lane = 0; lane < 4; ++lane) {
                rys_roots_weights_fixed<N>(
                    targ[lane], roots + lane, weights + lane);
            }
            return;
        }
    }

    int base[4];
    alignas(32) double tf[4];
    for (int lane = 0; lane < 4; ++lane) {
        const double safe_targ =
            targ[lane] >= tm ? tm - 0.1 : targ[lane];
        const double fi = safe_targ * RysRootWeightTable::INV_DX;
        const int ki = static_cast<int>(fi);
        tf[lane] = fi - ki;
        base[lane] = ki;
    }
    const RysJExactN1LagrangeBasis basis =
        rys_j_exact_n1_lagrange_basis(_mm256_load_pd(tf));
    _mm256_storeu_pd(
        roots,
        rys_j_exact_n1_lagrange_table_value(
            g_rys_tbl.roots_tbl[N].data(), base, basis));
    _mm256_storeu_pd(
        weights,
        rys_j_exact_n1_lagrange_table_value(
            g_rys_tbl.wts_tbl[N].data(), base, basis));

    if (mask != 0) {
        for (int lane = 0; lane < 4; ++lane) {
            if ((mask & (1 << lane)) != 0) {
                rys_roots_weights_fixed<N>(
                    targ[lane], roots + lane, weights + lane);
            }
        }
    }
}

// WHAT/WHY/PURPOSE: identical N=1 Lagrange policy for ss|ss, but stream only
// the weight table because those three J producers never consume the root.
__attribute__((noinline, hot)) __m256d
rys_j_exact_n1_weight4_lagrange(__m256d targ) noexcept {
    constexpr int N = 1;
    alignas(32) double targ4[4];
    alignas(32) double weights4[4];
    _mm256_store_pd(targ4, targ);
    const double tm = g_rys_tbl.tmax[N];
    int mask = 0;
    if (!(targ4[0] < tm && targ4[1] < tm &&
          targ4[2] < tm && targ4[3] < tm)) {
        mask = (targ4[0] >= tm ? 1 : 0) |
               (targ4[1] >= tm ? 2 : 0) |
               (targ4[2] >= tm ? 4 : 0) |
               (targ4[3] >= tm ? 8 : 0);
        if (mask == 15) {
            for (int lane = 0; lane < 4; ++lane) {
                double root;
                rys_roots_weights_fixed<N>(
                    targ4[lane], &root, weights4 + lane);
            }
            return _mm256_load_pd(weights4);
        }
    }

    int base[4];
    alignas(32) double tf[4];
    for (int lane = 0; lane < 4; ++lane) {
        const double safe_targ =
            targ4[lane] >= tm ? tm - 0.1 : targ4[lane];
        const double fi = safe_targ * RysRootWeightTable::INV_DX;
        const int ki = static_cast<int>(fi);
        tf[lane] = fi - ki;
        base[lane] = ki;
    }
    __m256d result = rys_j_exact_n1_lagrange_table_value(
        g_rys_tbl.wts_tbl[N].data(), base,
        rys_j_exact_n1_lagrange_basis(_mm256_load_pd(tf)));
    if (mask != 0) {
        _mm256_store_pd(weights4, result);
        for (int lane = 0; lane < 4; ++lane) {
            if ((mask & (1 << lane)) != 0) {
                double root;
                rys_roots_weights_fixed<N>(
                    targ4[lane], &root, weights4 + lane);
            }
        }
        result = _mm256_load_pd(weights4);
    }
    return result;
}
#endif

#if defined(TLNO_J_SPEED_PROBE_EXACT_N2_LAGRANGE) && \
    !defined(TLNO_J_SPEED_PROBE_EXACT_ROOTS)
#error "TLNO_J_SPEED_PROBE_EXACT_N2_LAGRANGE requires TLNO_J_SPEED_PROBE_EXACT_ROOTS"
#endif

// Evaluate the established N=2 Rys table with the direct degree-four Lagrange
// basis for grid nodes 0..4, consuming two sample rows at a time.  The generic
// Newton evaluator keeps R[10] and W[10] live together; this form instead keeps
// four output accumulators and five shared cardinal coefficients.  It changes
// floating-point association, but not the five samples, tmax, mixed-lane
// scalar repair, fixed<2> fallback, or SOA output order.  The packed pp|ps
// producer selects it only once the primitive workload amortizes the outlined
// evaluator.
__attribute__((noinline, hot)) bool
rys_n2_batch4_lagrange_table(const double* __restrict targ,
                             double* __restrict roots,
                             double* __restrict weights) noexcept {
    constexpr int N = 2;
    const double tm = g_rys_tbl.tmax[N];
    int mask = 0;
    if (!(targ[0] < tm && targ[1] < tm && targ[2] < tm && targ[3] < tm)) {
        mask = (targ[0] >= tm ? 1 : 0) | (targ[1] >= tm ? 2 : 0) |
               (targ[2] >= tm ? 4 : 0) | (targ[3] >= tm ? 8 : 0);
        if (mask == 15) return false;
    }

    const double* __restrict rtbl = g_rys_tbl.roots_tbl[N].data();
    const double* __restrict wtbl = g_rys_tbl.wts_tbl[N].data();
    int base[4];
    alignas(32) double tf[4];
    for (int lane = 0; lane < 4; ++lane) {
        const double safe_targ = (targ[lane] >= tm) ? (tm - 0.1) : targ[lane];
        const double fi = safe_targ * RysRootWeightTable::INV_DX;
        const int ki = static_cast<int>(fi);
        tf[lane] = fi - ki;
        base[lane] = ki * N;
    }

    const __m256d t = _mm256_load_pd(tf);
    __m256d root0 = _mm256_setzero_pd();
    __m256d root1 = _mm256_setzero_pd();
    __m256d weight0 = _mm256_setzero_pd();
    __m256d weight1 = _mm256_setzero_pd();

// Load two adjacent grid rows for all four target lanes, transpose them to
// SOA sample vectors, and immediately consume each vector.  No R[10]/W[10]
// scratch exists, and the macro expands to straight-line AVX2 (no hot call).
#define TLNO_RYS_N2_ACCUMULATE_TWO_ROWS(TABLE, OFFSET, L0, L1, OUT0, OUT1) \
    do {                                                                    \
        const __m256d lane0 = _mm256_loadu_pd((TABLE) + base[0] + (OFFSET)); \
        const __m256d lane1 = _mm256_loadu_pd((TABLE) + base[1] + (OFFSET)); \
        const __m256d lane2 = _mm256_loadu_pd((TABLE) + base[2] + (OFFSET)); \
        const __m256d lane3 = _mm256_loadu_pd((TABLE) + base[3] + (OFFSET)); \
        const __m256d even01 = _mm256_unpacklo_pd(lane0, lane1);             \
        const __m256d odd01 = _mm256_unpackhi_pd(lane0, lane1);              \
        const __m256d even23 = _mm256_unpacklo_pd(lane2, lane3);             \
        const __m256d odd23 = _mm256_unpackhi_pd(lane2, lane3);              \
        const __m256d row0_root0 =                                            \
            _mm256_permute2f128_pd(even01, even23, 0x20);                    \
        (OUT0) = _mm256_fmadd_pd((L0), row0_root0, (OUT0));                  \
        const __m256d row0_root1 =                                            \
            _mm256_permute2f128_pd(odd01, odd23, 0x20);                      \
        (OUT1) = _mm256_fmadd_pd((L0), row0_root1, (OUT1));                  \
        const __m256d row1_root0 =                                            \
            _mm256_permute2f128_pd(even01, even23, 0x31);                    \
        (OUT0) = _mm256_fmadd_pd((L1), row1_root0, (OUT0));                  \
        const __m256d row1_root1 =                                            \
            _mm256_permute2f128_pd(odd01, odd23, 0x31);                      \
        (OUT1) = _mm256_fmadd_pd((L1), row1_root1, (OUT1));                  \
    } while (false)

#define TLNO_RYS_N2_ACCUMULATE_LAST_ROW(TABLE, L4, OUT0, OUT1)                \
    do {                                                                      \
        const __m256d row4_root0 = _mm256_set_pd(                             \
            (TABLE)[base[3] + 8], (TABLE)[base[2] + 8],                       \
            (TABLE)[base[1] + 8], (TABLE)[base[0] + 8]);                      \
        const __m256d row4_root1 = _mm256_set_pd(                             \
            (TABLE)[base[3] + 9], (TABLE)[base[2] + 9],                       \
            (TABLE)[base[1] + 9], (TABLE)[base[0] + 9]);                      \
        (OUT0) = _mm256_fmadd_pd((L4), row4_root0, (OUT0));                  \
        (OUT1) = _mm256_fmadd_pd((L4), row4_root1, (OUT1));                  \
    } while (false)

    // Build the five cardinal coefficients from the falling-factorial basis.
    // This is algebraically the same degree-four interpolant as Newton; the
    // coefficient setup uses six multiplies and four FMAs, then each Li feeds
    // all four outputs.
    const __m256d b2 = _mm256_mul_pd(
        _mm256_mul_pd(t, _mm256_sub_pd(t, _mm256_set1_pd(1.0))),
        _mm256_set1_pd(0.5));
    const __m256d b3 = _mm256_mul_pd(
        _mm256_mul_pd(b2, _mm256_sub_pd(t, _mm256_set1_pd(2.0))),
        _mm256_set1_pd(1.0 / 3.0));
    const __m256d b4 = _mm256_mul_pd(
        _mm256_mul_pd(b3, _mm256_sub_pd(t, _mm256_set1_pd(3.0))),
        _mm256_set1_pd(0.25));
    const __m256d b4x2 = _mm256_add_pd(b4, b4);
    const __m256d b4x4 = _mm256_add_pd(b4x2, b4x2);
    const __m256d l4 = b4;
    const __m256d l3 = _mm256_sub_pd(b3, b4x4);
    const __m256d l2 = _mm256_add_pd(
        _mm256_add_pd(b4x4, b4x2),
        _mm256_fnmadd_pd(_mm256_set1_pd(3.0), b3, b2));
    const __m256d l1 = _mm256_fnmadd_pd(
        _mm256_set1_pd(4.0), b4,
        _mm256_fmadd_pd(
            _mm256_set1_pd(3.0), b3,
            _mm256_fnmadd_pd(_mm256_set1_pd(2.0), b2, t)));
    const __m256d l0 = _mm256_add_pd(
        _mm256_sub_pd(
            _mm256_add_pd(
                _mm256_sub_pd(_mm256_set1_pd(1.0), t), b2),
            b3),
        b4);

    TLNO_RYS_N2_ACCUMULATE_TWO_ROWS(rtbl, 0, l0, l1, root0, root1);
    TLNO_RYS_N2_ACCUMULATE_TWO_ROWS(wtbl, 0, l0, l1, weight0, weight1);
    TLNO_RYS_N2_ACCUMULATE_TWO_ROWS(rtbl, 4, l2, l3, root0, root1);
    TLNO_RYS_N2_ACCUMULATE_TWO_ROWS(wtbl, 4, l2, l3, weight0, weight1);
    TLNO_RYS_N2_ACCUMULATE_LAST_ROW(rtbl, l4, root0, root1);
    TLNO_RYS_N2_ACCUMULATE_LAST_ROW(wtbl, l4, weight0, weight1);

#undef TLNO_RYS_N2_ACCUMULATE_LAST_ROW
#undef TLNO_RYS_N2_ACCUMULATE_TWO_ROWS

    _mm256_storeu_pd(roots, root0);
    _mm256_storeu_pd(roots + 4, root1);
    _mm256_storeu_pd(weights, weight0);
    _mm256_storeu_pd(weights + 4, weight1);

    if (mask != 0) {
        for (int lane = 0; lane < 4; ++lane) {
            if (mask & (1 << lane)) {
                double lane_roots[N];
                double lane_weights[N];
                rys_roots_weights_fixed<N>(
                    targ[lane], lane_roots, lane_weights);
                for (int root = 0; root < N; ++root) {
                    roots[root * 4 + lane] = lane_roots[root];
                    weights[root * 4 + lane] = lane_weights[root];
                }
            }
        }
    }
    return true;
}

__attribute__((noinline, hot)) void
rys_ppps_n2_batch4_lagrange(
    const double* __restrict targ,
    double* __restrict roots,
    double* __restrict weights) noexcept {
    const bool used_table =
        rys_n2_batch4_lagrange_table(targ, roots, weights);
    if (used_table) return;
    for (int lane = 0; lane < 4; ++lane) {
        double lane_roots[2];
        double lane_weights[2];
        rys_roots_weights_fixed<2>(
            targ[lane], lane_roots, lane_weights);
        for (int root = 0; root < 2; ++root) {
            roots[4 * root + lane] = lane_roots[root];
            weights[4 * root + lane] = lane_weights[root];
        }
    }
}

// The (ss|ss) shellset kernel consumes only the one-root quadrature weight
// (Boys F0); computing and loading the root as well would double the lookup
// work.  This is the N=1 weight half of rys_roots_weights_batch4, specialized
// so four unrelated shellsets share one lane-parallel Newton stencil.
inline bool rys_weight1_batch4(const double* __restrict targ,
                               __m256d& weights) noexcept {
    const double tm = g_rys_tbl.tmax[1];
    int mask = 0;
    if (!(targ[0] < tm && targ[1] < tm && targ[2] < tm && targ[3] < tm)) {
        mask = (targ[0] >= tm ? 1 : 0) | (targ[1] >= tm ? 2 : 0) |
               (targ[2] >= tm ? 4 : 0) | (targ[3] >= tm ? 8 : 0);
        if (mask == 15) {
            // Weight half of rys_hermite_asymptotic_table<1>, one quartet per
            // lane. No root or table stencil is computed for this packet.
            const __m256d invT = _mm256_div_pd(
                _mm256_set1_pd(1.0), _mm256_loadu_pd(targ));
            weights = _mm256_mul_pd(
                _mm256_set1_pd(g_rys_hermite_asymp.weights[1][0]),
                _mm256_sqrt_pd(invT));
            return true;
        }
    }
    const double* __restrict table = g_rys_tbl.wts_tbl[1].data();
    int base[4];
    alignas(32) double tf[4];
    for (int lane = 0; lane < 4; ++lane) {
        const double safe_targ = (targ[lane] >= tm) ? (tm - 0.1) : targ[lane];
        const double fi = safe_targ * RysRootWeightTable::INV_DX;
        const int ki = static_cast<int>(fi);
        tf[lane] = fi - ki;
        base[lane] = ki;
    }
    const __m256d t = _mm256_load_pd(tf);
    const __m256d tm1 = _mm256_sub_pd(t, _mm256_set1_pd(1.0));
    const __m256d tm2 = _mm256_sub_pd(t, _mm256_set1_pd(2.0));
    const __m256d tm3 = _mm256_sub_pd(t, _mm256_set1_pd(3.0));
    __m256d f0, f1, f2, f3;
    rys_tp4(_mm256_loadu_pd(table + base[0]),
            _mm256_loadu_pd(table + base[1]),
            _mm256_loadu_pd(table + base[2]),
            _mm256_loadu_pd(table + base[3]),
            f0, f1, f2, f3);
    const __m256d f4 = _mm256_set_pd(table[base[3] + 4], table[base[2] + 4],
                                    table[base[1] + 4], table[base[0] + 4]);
    weights = rys_newton5v(f0, f1, f2, f3, f4, t, tm1, tm2, tm3);
    if (mask != 0) {
        alignas(32) double warr[4];
        _mm256_storeu_pd(warr, weights);
        for (int l = 0; l < 4; ++l) {
            if (mask & (1 << l)) {
                double dummy_rt;
                rys_roots_weights_fixed<1>(targ[l], &dummy_rt, &warr[l]);
            }
        }
        weights = _mm256_loadu_pd(warr);
    }
    return true;
}

// Degree-5/6 minimax-style rational F0 approximant used by the arbitrary-depth
// (ss|ss) 1x16 kernel below.  It is independently dense-grid audited on
// 0 <= T <= 5; tiles outside that domain retain the Rys table/root path.
constexpr double kRysSpBoysMediumMaxT = 5.0;
constexpr std::array<double, 6> kRysSsssMediumNumerator = {
    4.530897432932928792e-01,
    2.493218860524493985e-01,
    9.744677397300806343e-02,
    1.170817621951673534e-02,
    1.332818811711418720e-03,
   -1.055457740652338466e-04};
constexpr std::array<double, 7> kRysSsssMediumDenominator = {
    8.293911803473561495e-01,
    8.087735012999943685e-01,
    3.356041331722972032e-01,
    7.267983752225257166e-02,
    7.517711686613109089e-03,
    7.522259300840605789e-05,
   -3.964421115112796989e-05};

// A common-denominator Chebyshev rational for the three Boys moments needed by
// the low-s/p recurrences on 0 <= T <= 5.  The coefficients were fit to the
// analytic hypergeometric definitions, not to integral output.  A 2,000,001-
// point linear grid gave maximum absolute errors 2.22e-15, 2.89e-15, and
// 9.99e-16 for F0, F1, and F2.  On 30,001 independent 80-digit samples F0/F1
// stayed below 1.30e-15/2.78e-15; the common denominator remained at least
// 2.65e-1 away from zero.  The reproducible audit lives in
// experiments/sp_boys_moment_audit.  Outside this bounded interval the
// established Rys table/fallback remains authoritative.
constexpr std::array<double, 9> kRysPsSsBoys0Numerator = {
    4.96127718679698071e-01,
    2.87553186050661269e-01,
    6.24510885119250736e-02,
    6.02268420965473665e-03,
    5.52422861036275920e-04,
    1.87539396379352580e-05,
    9.06218866037224385e-07,
   -6.84314314282036777e-10,
    1.98219155384745173e-10};
constexpr std::array<double, 9> kRysPsSsBoys1Numerator = {
    7.99161284338462397e-02,
   -1.97690642300051705e-03,
    6.18488472112222170e-03,
   -3.64858872383147786e-04,
    6.53425183048812449e-05,
   -4.12521537755292034e-06,
    2.48844207744468439e-07,
   -9.11544087880926472e-09,
    1.74351125974485602e-10};
constexpr std::array<double, 9> kRysPsPsBoys2Numerator = {
    3.47617050493254018e-02,
   -1.42197930052216023e-02,
    3.58594252674166629e-03,
   -4.82661640824230925e-04,
    5.29859188457158445e-05,
   -4.16737046673423730e-06,
    2.37919536944138088e-07,
   -8.98550720808410919e-09,
    1.74351269316072588e-10};
constexpr std::array<double, 9> kRysPsSsBoysDenominator = {
    1.00000000000000000e+00,
    9.16015850384607200e-01,
    2.07675644643858293e-01,
    2.85245310625180382e-02,
    2.54565774903270370e-03,
    1.48720292676182090e-04,
    5.41869498444562479e-06,
    1.07173071935449471e-07,
    7.79102819919273042e-10};

// Global binary64 rational for F0/F1/F2 on 0 <= T <= 12.  Coefficients are
// ascending powers of x=T/12 and must be evaluated by Horner in x: converting
// this quotient back to Chebyshev form loses a decimal digit near T=0.  The
// common denominator has positive coefficients, hence no pole on the domain.
// Independent 2,000,001-point SciPy, 30,001-point 80-digit mpmath, and dense
// AVX2-FMA/quadmath audits bound max absolute error by 4.42e-15.  The <=5
// paths above remain cheaper; this global form replaces their root-table
// fallback for wider or mixed packets up to T=12.
constexpr double kRysSpBoysGlobalMaxT = 12.0;
constexpr std::array<double, 11> kRysSpBoysGlobal0Numerator = {
    2.41937677820681651e-02, 4.59134964995126515e-02,
    1.74595547877308410e-01, 1.57853277390500923e-01,
    2.39143275203376471e-01, 1.19579496552165182e-01,
    8.85913144671726521e-02, 2.14757762292305220e-02,
    6.75432863509305235e-03, 2.84178198290433012e-04,
    2.30837155773410297e-05};
constexpr std::array<double, 11> kRysSpBoysGlobal1Numerator = {
    8.06458926068938896e-03, -1.05021868010309744e-02,
    3.87174747029577979e-02, -3.04573853061233905e-02,
    4.21333619935586531e-02, -2.17778024591769905e-02,
    1.39189778672930160e-02, -4.58548492601399283e-03,
    1.27662632437713463e-03, -2.04521006909865843e-04,
    1.57856065954400959e-05};
constexpr std::array<double, 11> kRysSpBoysGlobal2Numerator = {
    4.83875355641373833e-03, -1.29373169580262112e-02,
    2.83330816479895435e-02, -3.51550181864331568e-02,
    3.44286576425524016e-02, -2.37292349300069830e-02,
    1.24509681114232541e-02, -4.70733682928108858e-03,
    1.23922031037814536e-03, -2.03509570534611328e-04,
    1.57855958392209315e-05};
constexpr std::array<double, 11> kRysSpBoysGlobalDenominator = {
    2.41937677820681651e-02, 1.42688567627795027e-01,
    3.96959562326089477e-01, 6.86376884473860471e-01,
    8.16446762194204312e-01, 6.96232653023352799e-01,
    4.27984669598052925e-01, 1.84090723310696314e-01,
    5.08748381894112353e-02, 7.35927099556330996e-03,
    3.32662320442847503e-04};

inline void rys_sp_boys012_global(
    __m256d targ,
    __m256d& boys0,
    __m256d& boys1,
    __m256d& boys2) noexcept {
    const __m256d x = _mm256_mul_pd(
        targ, _mm256_set1_pd(1.0 / kRysSpBoysGlobalMaxT));
    boys0 = _mm256_set1_pd(kRysSpBoysGlobal0Numerator.back());
    boys1 = _mm256_set1_pd(kRysSpBoysGlobal1Numerator.back());
    boys2 = _mm256_set1_pd(kRysSpBoysGlobal2Numerator.back());
    __m256d denominator =
        _mm256_set1_pd(kRysSpBoysGlobalDenominator.back());
#pragma GCC unroll 10
    for (int k = 9; k >= 0; --k) {
        const std::size_t index = static_cast<std::size_t>(k);
        boys0 = _mm256_fmadd_pd(
            boys0, x,
            _mm256_set1_pd(kRysSpBoysGlobal0Numerator[index]));
        boys1 = _mm256_fmadd_pd(
            boys1, x,
            _mm256_set1_pd(kRysSpBoysGlobal1Numerator[index]));
        boys2 = _mm256_fmadd_pd(
            boys2, x,
            _mm256_set1_pd(kRysSpBoysGlobal2Numerator[index]));
        denominator = _mm256_fmadd_pd(
            denominator, x,
            _mm256_set1_pd(kRysSpBoysGlobalDenominator[index]));
    }
    const __m256d inverse = _mm256_div_pd(
        _mm256_set1_pd(1.0), denominator);
    boys0 = _mm256_mul_pd(boys0, inverse);
    boys1 = _mm256_mul_pd(boys1, inverse);
    boys2 = _mm256_mul_pd(boys2, inverse);
}

inline void rys_psss_boys01_medium(
    __m256d targ, __m256d& boys0, __m256d& boys1) noexcept {
    const __m256d z = _mm256_fmadd_pd(
        targ, _mm256_set1_pd(2.0 / kRysSpBoysMediumMaxT),
        _mm256_set1_pd(-1.0));
    const __m256d twice_z = _mm256_add_pd(z, z);
    __m256d p0_next = _mm256_setzero_pd();
    __m256d p0_next2 = _mm256_setzero_pd();
    __m256d p1_next = _mm256_setzero_pd();
    __m256d p1_next2 = _mm256_setzero_pd();
    __m256d q_next = _mm256_setzero_pd();
    __m256d q_next2 = _mm256_setzero_pd();
#pragma GCC unroll 8
    for (int k = 8; k >= 1; --k) {
        const std::size_t index = static_cast<std::size_t>(k);
        const __m256d p0_current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, p0_next, p0_next2),
            _mm256_set1_pd(kRysPsSsBoys0Numerator[index]));
        const __m256d p1_current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, p1_next, p1_next2),
            _mm256_set1_pd(kRysPsSsBoys1Numerator[index]));
        const __m256d q_current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, q_next, q_next2),
            _mm256_set1_pd(kRysPsSsBoysDenominator[index]));
        p0_next2 = p0_next;
        p0_next = p0_current;
        p1_next2 = p1_next;
        p1_next = p1_current;
        q_next2 = q_next;
        q_next = q_current;
    }
    const __m256d p0 = _mm256_add_pd(
        _mm256_fmsub_pd(z, p0_next, p0_next2),
        _mm256_set1_pd(kRysPsSsBoys0Numerator[0]));
    const __m256d p1 = _mm256_add_pd(
        _mm256_fmsub_pd(z, p1_next, p1_next2),
        _mm256_set1_pd(kRysPsSsBoys1Numerator[0]));
    const __m256d denominator = _mm256_add_pd(
        _mm256_fmsub_pd(z, q_next, q_next2),
        _mm256_set1_pd(kRysPsSsBoysDenominator[0]));
    const __m256d inverse = _mm256_div_pd(
        _mm256_set1_pd(1.0), denominator);
    boys0 = _mm256_mul_pd(p0, inverse);
    boys1 = _mm256_mul_pd(p1, inverse);
}

inline void rys_sp_boys012_medium(
    __m256d targ,
    __m256d& boys0,
    __m256d& boys1,
    __m256d& boys2) noexcept {
    const __m256d z = _mm256_fmadd_pd(
        targ, _mm256_set1_pd(2.0 / kRysSpBoysMediumMaxT),
        _mm256_set1_pd(-1.0));
    const __m256d twice_z = _mm256_add_pd(z, z);
    __m256d p0_next = _mm256_setzero_pd();
    __m256d p0_next2 = _mm256_setzero_pd();
    __m256d p1_next = _mm256_setzero_pd();
    __m256d p1_next2 = _mm256_setzero_pd();
    __m256d p2_next = _mm256_setzero_pd();
    __m256d p2_next2 = _mm256_setzero_pd();
    __m256d q_next = _mm256_setzero_pd();
    __m256d q_next2 = _mm256_setzero_pd();
#pragma GCC unroll 8
    for (int k = 8; k >= 1; --k) {
        const std::size_t index = static_cast<std::size_t>(k);
        const __m256d p0_current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, p0_next, p0_next2),
            _mm256_set1_pd(kRysPsSsBoys0Numerator[index]));
        const __m256d p1_current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, p1_next, p1_next2),
            _mm256_set1_pd(kRysPsSsBoys1Numerator[index]));
        const __m256d p2_current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, p2_next, p2_next2),
            _mm256_set1_pd(kRysPsPsBoys2Numerator[index]));
        const __m256d q_current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, q_next, q_next2),
            _mm256_set1_pd(kRysPsSsBoysDenominator[index]));
        p0_next2 = p0_next;
        p0_next = p0_current;
        p1_next2 = p1_next;
        p1_next = p1_current;
        p2_next2 = p2_next;
        p2_next = p2_current;
        q_next2 = q_next;
        q_next = q_current;
    }
    const __m256d p0 = _mm256_add_pd(
        _mm256_fmsub_pd(z, p0_next, p0_next2),
        _mm256_set1_pd(kRysPsSsBoys0Numerator[0]));
    const __m256d p1 = _mm256_add_pd(
        _mm256_fmsub_pd(z, p1_next, p1_next2),
        _mm256_set1_pd(kRysPsSsBoys1Numerator[0]));
    const __m256d p2 = _mm256_add_pd(
        _mm256_fmsub_pd(z, p2_next, p2_next2),
        _mm256_set1_pd(kRysPsPsBoys2Numerator[0]));
    const __m256d denominator = _mm256_add_pd(
        _mm256_fmsub_pd(z, q_next, q_next2),
        _mm256_set1_pd(kRysPsSsBoysDenominator[0]));
    const __m256d inverse = _mm256_div_pd(
        _mm256_set1_pd(1.0), denominator);
    boys0 = _mm256_mul_pd(p0, inverse);
    boys1 = _mm256_mul_pd(p1, inverse);
    boys2 = _mm256_mul_pd(p2, inverse);
}

inline __m256d rys_ssss_weight1_exact(__m256d targ) noexcept {
    alignas(32) double targ4[4];
    alignas(32) double weight4[4];
    _mm256_store_pd(targ4, targ);
    __m256d weight;
    if (!rys_weight1_batch4(targ4, weight)) {
        double root;
        for (int lane = 0; lane < 4; ++lane) {
            rys_roots_weights_fixed<1>(
                targ4[lane], &root, &weight4[lane]);
        }
        weight = _mm256_load_pd(weight4);
    }
    return weight;
}

// Four independent Horner pairs expose sufficient ILP to hide most of the
// rational evaluator latency for a 1x16 output tile.
inline void rys_ssss_weight1_medium4(
    __m256d t0,
    __m256d t1,
    __m256d t2,
    __m256d t3,
    __m256d& result0,
    __m256d& result1,
    __m256d& result2,
    __m256d& result3) noexcept {
    const __m256d scale =
        _mm256_set1_pd(2.0 / kRysSpBoysMediumMaxT);
    const __m256d minus_one = _mm256_set1_pd(-1.0);
    const __m256d z0 = _mm256_fmadd_pd(t0, scale, minus_one);
    const __m256d z1 = _mm256_fmadd_pd(t1, scale, minus_one);
    const __m256d z2 = _mm256_fmadd_pd(t2, scale, minus_one);
    const __m256d z3 = _mm256_fmadd_pd(t3, scale, minus_one);

    result0 = _mm256_set1_pd(kRysSsssMediumNumerator.back());
    result1 = result0;
    result2 = result0;
    result3 = result0;
    __m256d denominator0 =
        _mm256_set1_pd(kRysSsssMediumDenominator.back());
    __m256d denominator1 = denominator0;
    __m256d denominator2 = denominator0;
    __m256d denominator3 = denominator0;
    const __m256d q5 =
        _mm256_set1_pd(kRysSsssMediumDenominator[5]);
    denominator0 = _mm256_fmadd_pd(denominator0, z0, q5);
    denominator1 = _mm256_fmadd_pd(denominator1, z1, q5);
    denominator2 = _mm256_fmadd_pd(denominator2, z2, q5);
    denominator3 = _mm256_fmadd_pd(denominator3, z3, q5);
#pragma GCC unroll 5
    for (int k = 4; k >= 0; --k) {
        const __m256d p = _mm256_set1_pd(
            kRysSsssMediumNumerator[static_cast<std::size_t>(k)]);
        const __m256d q = _mm256_set1_pd(
            kRysSsssMediumDenominator[static_cast<std::size_t>(k)]);
        result0 = _mm256_fmadd_pd(result0, z0, p);
        result1 = _mm256_fmadd_pd(result1, z1, p);
        result2 = _mm256_fmadd_pd(result2, z2, p);
        result3 = _mm256_fmadd_pd(result3, z3, p);
        denominator0 = _mm256_fmadd_pd(denominator0, z0, q);
        denominator1 = _mm256_fmadd_pd(denominator1, z1, q);
        denominator2 = _mm256_fmadd_pd(denominator2, z2, q);
        denominator3 = _mm256_fmadd_pd(denominator3, z3, q);
    }
    result0 = _mm256_div_pd(result0, denominator0);
    result1 = _mm256_div_pd(result1, denominator1);
    result2 = _mm256_div_pd(result2, denominator2);
    result3 = _mm256_div_pd(result3, denominator3);
}

// F0-only form of the global 0..12 rational.  Four independent Horner chains
// retain the ILP of the 1x16 ss|ss tile while avoiding unused F1/F2 work.
inline void rys_ssss_weight1_global4(
    __m256d t0,
    __m256d t1,
    __m256d t2,
    __m256d t3,
    __m256d& result0,
    __m256d& result1,
    __m256d& result2,
    __m256d& result3) noexcept {
    const __m256d inverse_domain =
        _mm256_set1_pd(1.0 / kRysSpBoysGlobalMaxT);
    const __m256d x0 = _mm256_mul_pd(t0, inverse_domain);
    const __m256d x1 = _mm256_mul_pd(t1, inverse_domain);
    const __m256d x2 = _mm256_mul_pd(t2, inverse_domain);
    const __m256d x3 = _mm256_mul_pd(t3, inverse_domain);
    result0 = _mm256_set1_pd(kRysSpBoysGlobal0Numerator.back());
    result1 = result0;
    result2 = result0;
    result3 = result0;
    __m256d denominator0 =
        _mm256_set1_pd(kRysSpBoysGlobalDenominator.back());
    __m256d denominator1 = denominator0;
    __m256d denominator2 = denominator0;
    __m256d denominator3 = denominator0;
#pragma GCC unroll 10
    for (int k = 9; k >= 0; --k) {
        const std::size_t index = static_cast<std::size_t>(k);
        const __m256d p = _mm256_set1_pd(
            kRysSpBoysGlobal0Numerator[index]);
        const __m256d q = _mm256_set1_pd(
            kRysSpBoysGlobalDenominator[index]);
        result0 = _mm256_fmadd_pd(result0, x0, p);
        result1 = _mm256_fmadd_pd(result1, x1, p);
        result2 = _mm256_fmadd_pd(result2, x2, p);
        result3 = _mm256_fmadd_pd(result3, x3, p);
        denominator0 = _mm256_fmadd_pd(denominator0, x0, q);
        denominator1 = _mm256_fmadd_pd(denominator1, x1, q);
        denominator2 = _mm256_fmadd_pd(denominator2, x2, q);
        denominator3 = _mm256_fmadd_pd(denominator3, x3, q);
    }
    result0 = _mm256_div_pd(result0, denominator0);
    result1 = _mm256_div_pd(result1, denominator1);
    result2 = _mm256_div_pd(result2, denominator2);
    result3 = _mm256_div_pd(result3, denominator3);
}

// One-node Rys is determined by two Boys moments: weight=F0(T) and
// root=F1(T)/F0(T).  These Chebyshev interpolants were generated from those
// analytic functions, not fitted to molecular output.  A 2,000,001-point
// linear/log grid over 0 <= T <= 32 gave maximum relative errors 1.3e-14 for
// F0 and 6.0e-14 for F1/F0.  At T=32 the leading Hermite limit below has
// relative errors 1.6e-15 and 8.2e-14 respectively, decreasing thereafter.
constexpr std::array<double, 19> kRysJRoot1LowWeight = {
    6.57186257657690565e-01, -2.66817291710163718e-01,
    6.11450766743830212e-02, -1.23329742967013094e-02,
    2.14537771790371343e-03, -3.24238500012859426e-04,
    4.30892255459405197e-05, -5.09514596306707115e-06,
    5.41772106106005832e-07, -5.22794261301397216e-08,
    4.61459165846855516e-09, -3.75137719961754618e-10,
    2.82539203099210918e-11, -1.98174078717584339e-12,
    1.30038721981010171e-13, -8.01503807304158942e-15,
    4.65695734205492855e-16, -2.55890068872223400e-17,
    1.33045935934109000e-18};
constexpr std::array<double, 19> kRysJRoot1LowRoot = {
    2.10167585921943095e-01, -1.05722700810158776e-01,
    1.65724477871015893e-02, -1.09064339583581950e-03,
    -1.64531110254449302e-04, 5.16675312564016006e-05,
    -4.53301623596724828e-06, -4.87174794875736790e-07,
    1.84385108088177468e-07, -1.77703252059755237e-08,
    -1.53227514537760939e-09, 6.60924950002947094e-10,
    -6.83504928423506929e-11, -4.76463485810597099e-12,
    2.36158165318070036e-12, -2.60797962842130257e-13,
    -1.44017604778747762e-14, 8.41659846375106734e-15,
    -1.01944900548932589e-15};
constexpr std::array<double, 21> kRysJRoot1MidWeight = {
    3.30062918373955905e-01, -8.86800033534240889e-02,
    1.75293753022885343e-02, -3.74776076515702080e-03,
    8.08282251474098500e-04, -1.70126330440873141e-04,
    3.42812509499873318e-05, -6.53949360369973774e-06,
    1.17386577518789958e-06, -1.97759902298114603e-07,
    3.12531984597394858e-08, -4.63708542678856792e-09,
    6.46958721191583030e-10, -8.50456531239809815e-11,
    1.05564333040373567e-11, -1.24005730315507398e-12,
    1.38161055592514317e-13, -1.46311629672554681e-14,
    1.47574224656360612e-15, -1.42036690242214087e-16,
    1.29738649276352039e-17};
constexpr std::array<double, 21> kRysJRoot1MidRoot = {
    7.11537226452196858e-02, -3.68951567554813425e-02,
    9.14208709302884344e-03, -2.10056054423035437e-03,
    4.31643139017169901e-04, -7.54045604085306281e-05,
    1.00505941316780668e-05, -6.10382903301413325e-07,
    -1.65732303933813199e-07, 7.62507302594023568e-08,
    -1.86777956669226086e-08, 3.27290535476152063e-09,
    -3.83948483883645240e-10, 7.15559282090678219e-12,
    1.11353413522059425e-11, -3.77398932716651221e-12,
    8.07336476586845996e-13, -1.24459180200275050e-13,
    1.13719316890544041e-14, 7.15671886262279712e-16,
    -5.90013195903678538e-16};
constexpr std::array<double, 21> kRysJRoot1HighWeight = {
    1.97230575237035965e-01, -4.77686364043661757e-02,
    8.63438263660545359e-03, -1.73194686582131443e-03,
    3.64581809754845911e-04, -7.89110048463089185e-05,
    1.73892285299449126e-05, -3.87930834633110079e-06,
    8.72755558762472875e-07, -1.97410029252897375e-07,
    4.47668050955943390e-08, -1.01483228068627912e-08,
    2.29266910743478395e-09, -5.14490389340096947e-10,
    1.14300660746253554e-10, -2.50572150182099063e-11,
    5.40379946454710415e-12, -1.14329626138682699e-12,
    2.36742239605959923e-13, -4.78310725473828379e-14,
    9.10993958982254411e-15};
constexpr std::array<double, 21> kRysJRoot1HighRoot = {
    2.55153927451760992e-02, -1.22680414743856700e-02,
    2.94919029144750933e-03, -7.08899344767851994e-04,
    1.70350810295424041e-04, -4.09081546622854560e-05,
    9.80923080843633207e-06, -2.34525013104700256e-06,
    5.57743412111833025e-07, -1.31464366558905925e-07,
    3.05584343604006018e-08, -6.95876052125396932e-09,
    1.53929472823488875e-09, -3.27098490832936065e-10,
    6.57377139851486144e-11, -1.21827895615502426e-11,
    1.97870205579571448e-12, -2.43083181770028851e-13,
    5.55179818591676872e-15, 9.65639992593502155e-15,
    -4.05868088185959340e-15};

template <const auto& Coefficients>
__attribute__((always_inline)) inline __m256d
rys_j_root1_chebyshev(__m256d z) noexcept {
    __m256d next = _mm256_setzero_pd();
    __m256d next2 = _mm256_setzero_pd();
    const __m256d twice_z = _mm256_add_pd(z, z);
#pragma GCC unroll 32
    for (int k = static_cast<int>(Coefficients.size()) - 1; k >= 1; --k) {
        const __m256d current = _mm256_add_pd(
            _mm256_fmsub_pd(twice_z, next, next2),
            _mm256_set1_pd(Coefficients[static_cast<std::size_t>(k)]));
        next2 = next;
        next = current;
    }
    return _mm256_add_pd(
        _mm256_fmsub_pd(z, next, next2),
        _mm256_set1_pd(Coefficients[0]));
}

template <const auto& WeightCoefficients, const auto& RootCoefficients>
__attribute__((always_inline)) inline void
rys_j_root1_interval(__m256d targ, double scale, double shift,
                     __m256d& root, __m256d& weight) noexcept {
    __m256d z = _mm256_fmadd_pd(
        targ, _mm256_set1_pd(scale), _mm256_set1_pd(shift));
    // Mixed-range packets evaluate each represented branch.  Clamping the
    // inactive lanes avoids overflow before blendv discards them.
    z = _mm256_max_pd(
        _mm256_set1_pd(-1.0),
        _mm256_min_pd(_mm256_set1_pd(1.0), z));
    weight = rys_j_root1_chebyshev<WeightCoefficients>(z);
    root = rys_j_root1_chebyshev<RootCoefficients>(z);
}

__attribute__((always_inline)) inline void
rys_j_root1_asymptotic(__m256d targ,
                       __m256d& root, __m256d& weight) noexcept {
    const __m256d safe_targ =
        _mm256_max_pd(targ, _mm256_set1_pd(32.0));
    root = _mm256_div_pd(
        _mm256_set1_pd(0.5), safe_targ);
    weight = _mm256_div_pd(
        _mm256_set1_pd(8.86226925452758014e-01),
        _mm256_sqrt_pd(safe_targ));
}


// WHAT: evaluate the high-precision one-root quadrature used by J entirely in
// four-lane AVX2 arithmetic; no root table, scalar lane extraction or call is
// required by the default path.
// WHY: the former two algebraic endpoint fits had percent-level interior
// error.  Exact one-node moments recover F0 and F1/F0 without restoring the
// multi-root table traffic whose performance ceiling this experiment probes.
// PURPOSE: numerical repair of N=1 is isolated from ActiveRoots and from the
// exact ERI, packed and combined-J/K instantiations.  The private LEGACY macro
// exists only to reproduce the old accuracy baseline; it changes no ABI.
__attribute__((always_inline)) inline void
rys_j_speed_probe_root_weight1(
    __m256d targ, __m256d& root, __m256d& weight) noexcept {
#if defined(TLNO_J_SPEED_PROBE_EXACT_ROOTS)
    alignas(32) double targ4[4];
    alignas(32) double roots4[4];
    alignas(32) double weights4[4];
    _mm256_store_pd(targ4, targ);
#ifdef TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE
    rys_j_exact_n1_batch4_lagrange(targ4, roots4, weights4);
#else
    if (!rys_roots_weights_batch4<1>(targ4, roots4, weights4)) {
        for (int lane = 0; lane < 4; ++lane) {
            rys_roots_weights_fixed<1>(
                targ4[lane], &roots4[lane], &weights4[lane]);
        }
    }
#endif
    root = _mm256_load_pd(roots4);
    weight = _mm256_load_pd(weights4);
#elif defined(TLNO_J_SPEED_PROBE_LEGACY_ROOT1)
    const __m256d one = _mm256_set1_pd(1.0);
    root = _mm256_div_pd(
        one,
        _mm256_fmadd_pd(_mm256_set1_pd(2.0), targ,
                        _mm256_set1_pd(3.0)));
    weight = _mm256_div_pd(
        one,
        _mm256_sqrt_pd(_mm256_fmadd_pd(
            _mm256_set1_pd(4.0 / kPi), targ, one)));
#else
    const __m256d low_mask = _mm256_cmp_pd(
        targ, _mm256_set1_pd(4.0), _CMP_LE_OQ);
    const int low_bits = _mm256_movemask_pd(low_mask);
    if (low_bits == 15) {
        rys_j_root1_interval<kRysJRoot1LowWeight, kRysJRoot1LowRoot>(
            targ, 0.5, -1.0, root, weight);
        return;
    }
    const __m256d through_mid_mask = _mm256_cmp_pd(
        targ, _mm256_set1_pd(12.0), _CMP_LE_OQ);
    const int through_mid_bits = _mm256_movemask_pd(through_mid_mask);
    if (low_bits == 0 && through_mid_bits == 15) {
        rys_j_root1_interval<kRysJRoot1MidWeight, kRysJRoot1MidRoot>(
            targ, 0.25, -2.0, root, weight);
        return;
    }
    const __m256d through_high_mask = _mm256_cmp_pd(
        targ, _mm256_set1_pd(32.0), _CMP_LE_OQ);
    const int through_high_bits = _mm256_movemask_pd(through_high_mask);
    if (through_mid_bits == 0 && through_high_bits == 15) {
        rys_j_root1_interval<kRysJRoot1HighWeight, kRysJRoot1HighRoot>(
            targ, 0.1, -2.2, root, weight);
        return;
    }
    if (through_high_bits == 0) {
        rys_j_root1_asymptotic(targ, root, weight);
        return;
    }

    // Rare packets that cross an interval boundary remain vector-only.  Each
    // represented polynomial is evaluated once and blended by lane.
    const __m256d mid_mask = _mm256_andnot_pd(low_mask, through_mid_mask);
    const __m256d high_mask =
        _mm256_andnot_pd(through_mid_mask, through_high_mask);
    const __m256d large_mask = _mm256_cmp_pd(
        targ, _mm256_set1_pd(32.0), _CMP_GT_OQ);
    root = _mm256_setzero_pd();
    weight = _mm256_setzero_pd();
    if (low_bits != 0) {
        __m256d branch_root, branch_weight;
        rys_j_root1_interval<kRysJRoot1LowWeight, kRysJRoot1LowRoot>(
            targ, 0.5, -1.0, branch_root, branch_weight);
        root = _mm256_blendv_pd(root, branch_root, low_mask);
        weight = _mm256_blendv_pd(weight, branch_weight, low_mask);
    }
    if (_mm256_movemask_pd(mid_mask) != 0) {
        __m256d branch_root, branch_weight;
        rys_j_root1_interval<kRysJRoot1MidWeight, kRysJRoot1MidRoot>(
            targ, 0.25, -2.0, branch_root, branch_weight);
        root = _mm256_blendv_pd(root, branch_root, mid_mask);
        weight = _mm256_blendv_pd(weight, branch_weight, mid_mask);
    }
    if (_mm256_movemask_pd(high_mask) != 0) {
        __m256d branch_root, branch_weight;
        rys_j_root1_interval<kRysJRoot1HighWeight, kRysJRoot1HighRoot>(
            targ, 0.1, -2.2, branch_root, branch_weight);
        root = _mm256_blendv_pd(root, branch_root, high_mask);
        weight = _mm256_blendv_pd(weight, branch_weight, high_mask);
    }
    if (_mm256_movemask_pd(large_mask) != 0) {
        __m256d branch_root, branch_weight;
        rys_j_root1_asymptotic(
            targ, branch_root, branch_weight);
        root = _mm256_blendv_pd(root, branch_root, large_mask);
        weight = _mm256_blendv_pd(weight, branch_weight, large_mask);
    }
#endif
}

// WHAT: select the quadrature order at compile time from the destination
// policy, then centralize the corresponding four-lane evaluator.
// WHY: the speed-only J experiment has explicitly deferred numerical accuracy;
// paying every exact Rys root in classes above ss|ps hides the producer-side
// speed ceiling it is meant to measure.  Packed ERI, reference-count, and
// combined-G policies must nevertheless retain their requested quadrature
// order.
// PURPOSE: every Emit-aware four-lane producer can size its root storage and
// root loop from one auditable rule.  Callers that do not take a bounded
// low-s/p moment path still use the established table evaluator and scalar
// fallback here.
#if defined(TLNO_J_SPEED_PROBE_EXACT_ROOTS) && \
    defined(TLNO_J_SPEED_PROBE_LEGACY_ROOT1)
#error "TLNO J root policies EXACT_ROOTS and LEGACY_ROOT1 are mutually exclusive"
#endif

#if defined(TLNO_J_SPEED_PROBE_EXACT_ROOTS)
inline constexpr bool kRysJSpeedProbeExactRoots = true;
#else
inline constexpr bool kRysJSpeedProbeExactRoots = false;
#endif

template <class Emit, int ExactN>
inline constexpr int ActiveRoots =
    std::is_same_v<Emit, RysDirectJSpeedProbeEmit> &&
            !kRysJSpeedProbeExactRoots
        ? 1
        : ExactN;

template <class Emit, int ExactN>
__attribute__((always_inline)) inline void
rys_roots_weights_batch4_for_emit(
    const double* __restrict targ,
    double* __restrict roots,
    double* __restrict weights) noexcept {
    static_assert(ExactN >= 1 && ExactN <= RysRootWeightTable::NMAX);
    if constexpr (std::is_same_v<Emit, RysDirectJSpeedProbeEmit> &&
                  !kRysJSpeedProbeExactRoots) {
        __m256d root;
        __m256d weight;
        rys_j_speed_probe_root_weight1(
            _mm256_loadu_pd(targ), root, weight);
        _mm256_storeu_pd(roots, root);
        _mm256_storeu_pd(weights, weight);
    } else {
        bool used_table = false;
#ifdef TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE
        if constexpr (std::is_same_v<Emit, RysDirectJSpeedProbeEmit> &&
                      ExactN == 1) {
            rys_j_exact_n1_batch4_lagrange(
                targ, roots, weights);
            used_table = true;
        } else
#endif
#ifdef TLNO_J_SPEED_PROBE_EXACT_N2_LAGRANGE
        if constexpr (std::is_same_v<Emit, RysDirectJSpeedProbeEmit> &&
                      ExactN == 2) {
            used_table = rys_n2_batch4_lagrange_table(
                targ, roots, weights);
        } else
#endif
        {
            used_table = rys_roots_weights_batch4<ExactN>(
                targ, roots, weights);
        }
        if (!used_table) {
            for (int lane = 0; lane < 4; ++lane) {
                double lane_roots[ExactN];
                double lane_weights[ExactN];
                rys_roots_weights_fixed<ExactN>(
                    targ[lane], lane_roots, lane_weights);
                for (int root = 0; root < ExactN; ++root) {
                    roots[4 * root + lane] = lane_roots[root];
                    weights[4 * root + lane] = lane_weights[root];
                }
            }
        }
    }
}

#include "integrals/two_electron/fock/detail/ssss_receivers.ipp"

template <bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_ssss_shell_bin_pair_fallback(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values) {
    alignas(32) double targ4[4];
    alignas(32) double weight4[4];
    alignas(32) double raw8[8];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    // WHAT: the J marker consumes one fixed ket pair at a time; exact ERI/G
    // policies retain the existing two-slot latency-hiding schedule.
    // WHY: two simultaneous contracted ss|ss YMMs exceeded the register budget
    // once final J metadata became live and GCC turned both into stack RMWs.
    // PURPOSE: the speed probe has one register-resident ERI accumulator and
    // still visits exactly the same ket positions and primitive-vector tests.
    constexpr int ket_step = std::is_same_v <
        Emit, RysDirectJSpeedProbeEmit > ? 1 : 2;
    const RysJkEmitContext* j_context = nullptr;
    if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
        j_context = rys_jk_context_slot();
    }

    for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
         bra_pos += 4) {
        const std::size_t valid_lanes =
            std::min<std::size_t>(4, bra_pair_count - bra_pos);
        const std::size_t ket_limit = same_bin
            ? bra_pos + valid_lanes
            : ket_pair_count;
        RysJSsssPairView moving[4] = {};
        double moving_delta[4] = {};
        std::uint64_t moving_terms[4] = {};
        if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
            for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                moving[lane] = rys_j_ssss_pair_view<Emit>(
                    *j_context, bd,
                    bra_bin.pair_group_ids[bra_pos + lane]);
            }
        }
        for (std::size_t ket_pos = 0; ket_pos < ket_limit;
             ket_pos += static_cast<std::size_t>(ket_step)) {
            unsigned shell_live[2] = {0u, 0u};
            unsigned shell_q_live[2] = {0u, 0u};
            if constexpr (Screened) {
                for (int slot = 0; slot < ket_step; ++slot) {
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    if (ket_lane < ket_limit) {
                        shell_live[slot] = rys_sp_shell_mask_bra4(
                            bra_bin, bra_pos, valid_lanes,
                            ket_bin, ket_lane, same_bin,
                            shell_q_live[slot]);
                    }
                }
                if ((shell_live[0] | shell_live[1]) == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        (shell_q_live[0] | shell_q_live[1]) == 0u) {
                        // WHAT/WHY: only Q==0 ends the sorted tail; a zero
                        // caused by density is not monotone.
                        break;
                    }
                    continue;
                }
            } else if constexpr (PrimitiveFast) {
                for (int slot = 0; slot < ket_step; ++slot) {
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    if (ket_lane < ket_limit) {
                        shell_live[slot] =
                            rys_sp_shell_logical_bra4(
                                bra_pos, valid_lanes,
                                ket_lane, same_bin);
                    }
                }
            }
            // Two independent ket shell pairs share each load of the four bra
            // primitives.  Besides halving those loads this gives the divider
            // and square-root units two independent accumulation chains.
            __m256d acc[2] = {_mm256_setzero_pd(), _mm256_setzero_pd()};
            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t off =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + off);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + off);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + off);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + off);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + off);
                __m256d vbra_fast;
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + off);
                }
                for (int slot = 0; slot < ket_step; ++slot) {
                    if constexpr (Screened) {
                        if (shell_live[slot] == 0u) continue;
                    }
                    const std::size_t ket_lane = ket_pos + static_cast<std::size_t>(slot);
                    for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                        const std::size_t koff =
                            static_cast<std::size_t>(ik) * ket_bin.stride + ket_lane;
                        if constexpr (PrimitiveFast) {
                            const __m256d vket_fast = _mm256_set1_pd(
                                ket_bin.primitive_fast_schwarz[koff]);
                            if (rys_primitive_fast_survival_mask(
                                    vbra_fast, vket_fast,
                                    shell_live[slot]) == 0u) {
                                continue;
                            }
                        }
                        const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                        const __m256d vpq = _mm256_add_pd(vp, vq);
                        const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                        const __m256d vdx = _mm256_sub_pd(
                            vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                        const __m256d vdy = _mm256_sub_pd(
                            vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                        const __m256d vdz = _mm256_sub_pd(
                            vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                        const __m256d vr2 = _mm256_fmadd_pd(
                            vdx, vdx,
                            _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                        const __m256d vtarg = _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)), vr2);
                        __m256d vweight;
                        if constexpr (std::is_same_v <
                                          Emit, RysDirectJSpeedProbeEmit > ) {
#ifdef TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE
                            vweight =
                                rys_j_exact_n1_weight4_lagrange(vtarg);
#else
                            __m256d unused_root;
                            rys_j_speed_probe_root_weight1(
                                vtarg, unused_root, vweight);
#endif
                        } else {
                            _mm256_store_pd(targ4, vtarg);
                            if (!rys_weight1_batch4(targ4, vweight)) {
                                double root;
                                for (int lane = 0; lane < 4; ++lane) {
                                    rys_roots_weights_fixed<1>(
                                        targ4[lane], &root, &weight4[lane]);
                                }
                                vweight = _mm256_load_pd(weight4);
                            }
                        }
                        // sqrt(1/(p+q)) replaces a second vector divide by
                        // sqrt(p+q), with equivalent full-double arithmetic.
                        const __m256d vpref = _mm256_mul_pd(
                            _mm256_mul_pd(
                                _mm256_mul_pd(vtwo_pi52, vbw),
                                _mm256_set1_pd(ket_bin.w[koff])),
                            _mm256_sqrt_pd(vinvpq));
                        acc[slot] = _mm256_fmadd_pd(vpref, vweight, acc[slot]);
                    }
                }
            }
            if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
                // WHAT: retain four moving J destinations across the complete
                // ket sweep and close each fixed ket destination immediately.
                // WHY: scalar emission performs two G RMWs per ss|ss ERI;
                // output stationarity reduces a 4x2 tile to at most six RMWs
                // and the contracted ERIs never enter `raw8`.
                // PURPOSE: J materializes only at its final M-by-M outputs,
                // while triangle, screening and primitive-vector semantics
                // remain identical to the producer above.
                for (int slot = 0; slot < ket_step; ++slot) {
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    if (ket_lane >= ket_limit) continue;
                    const unsigned active_bits = Screened
                        ? shell_live[slot]
                        : rys_sp_shell_logical_bra4(
                              bra_pos, valid_lanes, ket_lane, same_bin);
                    if (active_bits == 0u) continue;
                    const RysJSsssPairView fixed = rys_j_ssss_pair_view<Emit>(
                        *j_context, bd, ket_bin.pair_group_ids[ket_lane]);
                    double fixed_delta = 0.0;
                    std::uint64_t fixed_terms = 0u;
                    rys_j_ssss_accumulate_packet<Emit, false>(
                        *j_context, fixed, moving, active_bits, acc[slot],
                        moving_delta, moving_terms, fixed_delta, fixed_terms);
                    rys_j_ssss_commit<Emit>(
                        *j_context, fixed.matrix_index,
                        fixed_delta, fixed_terms);
                }
            } else {
                _mm256_store_pd(raw8, acc[0]);
                _mm256_store_pd(raw8 + 4, acc[1]);

                for (int slot = 0; slot < 2; ++slot) {
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    if (ket_lane >= ket_limit) continue;
                    const std::uint32_t ket_group =
                        ket_bin.pair_group_ids[ket_lane];
                    for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                        if constexpr (Screened) {
                            if ((shell_live[slot] & (1u << lane)) == 0u) {
                                continue;
                            }
                        } else if (same_bin &&
                                   ket_lane > bra_pos + lane) {
                            continue;
                        }
                        const std::uint32_t bra_group =
                            bra_bin.pair_group_ids[bra_pos + lane];
                        const auto& lhs_refs = low_batch_pair_components(bd)[
                            static_cast<std::size_t>(bra_group)];
                        const auto& rhs_refs = low_batch_pair_components(bd)[
                            static_cast<std::size_t>(ket_group)];
                        const bool same_pair_group = bra_group == ket_group;
                        const double raw_lane = raw8[4 * slot + lane];
                        rys_emit_block<Emit>(
                            eri_values, lhs_refs, rhs_refs, same_pair_group,
                            [raw_lane](const PackedPairComponentH& lhs_ref,
                                       const PackedPairComponentH& rhs_ref) noexcept {
                                return raw_lane * lhs_ref.norm * rhs_ref.norm;
                            });
                    }
                }
            }
        }
        if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
            for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                rys_j_ssss_commit<Emit>(
                    *j_context, moving[lane].matrix_index,
                    moving_delta[lane], moving_terms[lane]);
            }
        }
    }
}

// Emit policy threaded through (see the "Emission destination" block above).
// It travels with the value all the way down to rys_emit_pairpair(). There is
// deliberately no default: signature and call must change together.
template <class Emit>
inline void rys_emit_ssss_shell_pair_value(
    const BasisDataHGP& bd,
    std::uint32_t bra_group,
    std::uint32_t ket_group,
    double raw,
    std::vector<double>& eri_values) {
    const auto& lhs_refs =
        low_batch_pair_components(bd)[static_cast<std::size_t>(bra_group)];
    const auto& rhs_refs =
        low_batch_pair_components(bd)[static_cast<std::size_t>(ket_group)];
    const bool same_pair_group = bra_group == ket_group;
    if (lhs_refs.size() == 1 && rhs_refs.size() == 1) {
        const PackedPairComponentH& lhs_ref = lhs_refs.front();
        const PackedPairComponentH& rhs_ref = rhs_refs.front();
        if (!same_pair_group ||
            rhs_ref.pair_index <= lhs_ref.pair_index) {
            rys_emit_pairpair<Emit>(
                eri_values, lhs_ref.pair_index, rhs_ref.pair_index,
                raw * lhs_ref.norm * rhs_ref.norm);
        }
        return;
    }
    rys_emit_block<Emit>(
        eri_values, lhs_refs, rhs_refs, same_pair_group,
        [raw](const PackedPairComponentH& lhs_ref,
              const PackedPairComponentH& rhs_ref) noexcept {
            return raw * lhs_ref.norm * rhs_ref.norm;
        });
}

template <bool Screened>
inline bool rys_sp_bin_has_one_exponent_family(
    const RysLowPairBinH& bin) noexcept {
    const std::size_t pair_count =
        rys_sp_active_pair_count<Screened>(bin);
    if (pair_count <= 1u) return true;
    for (int ip = 0; ip < bin.primitive_count; ++ip) {
        const std::size_t base =
            static_cast<std::size_t>(ip) * bin.stride;
        const double reference = bin.s[base];
        for (std::size_t lane = 1; lane < pair_count; ++lane) {
            if (bin.s[base + lane] != reference) return false;
        }
    }
    return true;
}

struct RysSpExponentFamilyFactor {
    double rho;
    double invpq;
    double prefactor_scale;
};

// ss|ss consumes only rho and the square-root prefactor.  Keeping invpq in
// this table made the n=6 family occupy 31.1 KiB, effectively the full L1d,
// even though the kernel never loaded that field.  The compact form is 20.3
// KiB at the same 6^4 primitive interactions.
struct RysSsExponentFamilyFactor {
    double rho;
    double prefactor_scale;
};

inline void fill_rys_ss_exponent_family_factors(
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<RysSsExponentFamilyFactor>& factors) {
    const int bra_primitive_count = bra_bin.primitive_count;
    const int ket_primitive_count = ket_bin.primitive_count;
    const std::size_t factor_count =
        static_cast<std::size_t>(bra_primitive_count) *
        static_cast<std::size_t>(ket_primitive_count);
    factors.resize(factor_count);
    for (int ib = 0; ib < bra_primitive_count; ++ib) {
        const double p =
            bra_bin.s[static_cast<std::size_t>(ib) * bra_bin.stride];
        for (int ik = 0; ik < ket_primitive_count; ++ik) {
            const double q =
                ket_bin.s[static_cast<std::size_t>(ik) * ket_bin.stride];
            const double invpq = 1.0 / (p + q);
            const std::size_t uv =
                static_cast<std::size_t>(ib) *
                    static_cast<std::size_t>(ket_primitive_count) +
                static_cast<std::size_t>(ik);
            factors[uv] = {
                p * q * invpq,
                kTwoPi52 * std::sqrt(invpq)};
        }
    }
}

inline void fill_rys_sp_exponent_family_factors(
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<RysSpExponentFamilyFactor>& factors) {
    const int bra_primitive_count = bra_bin.primitive_count;
    const int ket_primitive_count = ket_bin.primitive_count;
    const std::size_t factor_count =
        static_cast<std::size_t>(bra_primitive_count) *
        static_cast<std::size_t>(ket_primitive_count);
    factors.resize(factor_count);
    for (int ib = 0; ib < bra_primitive_count; ++ib) {
        const double p =
            bra_bin.s[static_cast<std::size_t>(ib) * bra_bin.stride];
        for (int ik = 0; ik < ket_primitive_count; ++ik) {
            const double q =
                ket_bin.s[static_cast<std::size_t>(ik) * ket_bin.stride];
            const double invpq = 1.0 / (p + q);
            const std::size_t uv =
                static_cast<std::size_t>(ib) *
                    static_cast<std::size_t>(ket_primitive_count) +
                static_cast<std::size_t>(ik);
            factors[uv] = {
                p * q * invpq,
                invpq,
                kTwoPi52 * std::sqrt(invpq)};
        }
    }
}

// One scalar bra pair is contracted against sixteen ket pairs at a time.  This
// accepts arbitrary independent bra/ket primitive-pair counts.  Every P-Q
// distance and every primitive-pair interaction is still performed.  Packed
// ERI uses the bounded F0 rational on all-low-T 1x16 tiles and the established
// table/fallback elsewhere; only the speed-only J marker substitutes its
// documented register-only approximant and closes two Coulomb destinations
// directly.  Exponent-family invariants are shared only when the bin data
// proves that all pair lanes use that same family.
template < bool SharedExponentFamily, bool Screened, bool PrimitiveFast,
          class Emit >
inline void accumulate_rys_ssss_wide_shell_bin_pair_impl(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values) {
    const int bra_primitive_count = bra_bin.primitive_count;
    const int ket_primitive_count = ket_bin.primitive_count;
    // Reuse per-thread scratch across bin pairs.  Resizing may allocate during
    // plan warm-up, but steady-state calls at the same or smaller contraction
    // depth do not place an allocator in the hot path.
    thread_local std::vector<RysSsExponentFamilyFactor> family_factors;
    if constexpr (SharedExponentFamily) {
        fill_rys_ss_exponent_family_factors(
            bra_bin, ket_bin, family_factors);
    }

    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const __m256d vmedium_max =
        _mm256_set1_pd(kRysSpBoysMediumMaxT);
    const __m256d vglobal_max =
        _mm256_set1_pd(kRysSpBoysGlobalMaxT);
    alignas(32) double raw16[16];
    alignas(32) double raw4[4];
    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    const RysJkEmitContext* j_context = nullptr;
    if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
        j_context = rys_jk_context_slot();
    }

    for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
         ++bra_pos) {
        const std::size_t ket_limit =
            same_bin ? bra_pos + 1u : ket_pair_count;
        // Primitive FASTSCHWARZ uses the natural AVX2 width as its survival
        // unit.  Route that instantiation through the 1x4 loop below; the
        // unscreened/shell-only endpoint retains the tuned 1x16 kernel.
        // WHAT: keep the J speed probe on the one-accumulator 1x4 remainder
        // when exponents vary lane-by-lane or screening is disabled; exact
        // policies keep 1x16.
        // WHY: four simultaneous non-shared sums force GCC to implement every
        // FMA as a stack RMW, while the unscreened shared case spills one final
        // sum during output.  The screened shared specialization is the only
        // four-sum body that remains register-resident end to end.
        // PURPOSE: every J-owned ss|ss variant crosses the output seam without
        // ERI materialization; the wider proven schedule remains where it fits.
        constexpr bool j_narrow =
            std::is_same_v<Emit, RysDirectJSpeedProbeEmit> &&
            (!SharedExponentFamily || !Screened);
        const std::size_t full_limit = (PrimitiveFast || j_narrow)
            ? 0u
            : ket_limit & ~std::size_t{15u};
        const std::uint32_t bra_group =
            bra_bin.pair_group_ids[bra_pos];
        RysJSsssPairView fixed = {};
        double fixed_delta = 0.0;
        std::uint64_t fixed_terms = 0u;
        if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
            fixed = rys_j_ssss_pair_view<Emit>(
                *j_context, bd, bra_group);
        }

        for (std::size_t ket_pos = 0; ket_pos < full_limit;
             ket_pos += 16) {
            unsigned shell_live[4] = {0u, 0u, 0u, 0u};
            unsigned shell_q_live[4] = {0u, 0u, 0u, 0u};
            if constexpr (Screened) {
                unsigned any_shell_live = 0u;
                unsigned any_shell_q_live = 0u;
                for (std::size_t group = 0; group < 4; ++group) {
                    shell_live[group] = rys_sp_shell_mask_ket4(
                        bra_bin, bra_pos, ket_bin,
                        ket_pos + 4u * group, 4u,
                        shell_q_live[group]);
                    any_shell_live |= shell_live[group];
                    any_shell_q_live |= shell_q_live[group];
                }
                if (any_shell_live == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        any_shell_q_live == 0u) {
                        // WHAT/WHY: a density zero skips this block of 16;
                        // only a Q zero may end the following blocks.
                        break;
                    }
                    continue;
                }
            }

            __m256d sum0 = _mm256_setzero_pd();
            __m256d sum1 = _mm256_setzero_pd();
            __m256d sum2 = _mm256_setzero_pd();
            __m256d sum3 = _mm256_setzero_pd();

#pragma GCC unroll 1
            for (int ib = 0; ib < bra_primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride +
                    bra_pos;
                const __m256d vp = _mm256_set1_pd(bra_bin.s[boff]);
                const __m256d vPx = _mm256_set1_pd(bra_bin.Wx[boff]);
                const __m256d vPy = _mm256_set1_pd(bra_bin.Wy[boff]);
                const __m256d vPz = _mm256_set1_pd(bra_bin.Wz[boff]);
                const __m256d vbw = _mm256_set1_pd(bra_bin.w[boff]);
#pragma GCC unroll 1
                for (int ik = 0; ik < ket_primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride +
                        ket_pos;
                    const __m256d vq0 =
                        _mm256_loadu_pd(ket_bin.s.data() + koff);
                    const __m256d vq1 =
                        _mm256_loadu_pd(ket_bin.s.data() + koff + 4);
                    const __m256d vq2 =
                        _mm256_loadu_pd(ket_bin.s.data() + koff + 8);
                    const __m256d vq3 =
                        _mm256_loadu_pd(ket_bin.s.data() + koff + 12);
                    __m256d vrho0, vrho1, vrho2, vrho3;
                    __m256d vscale0, vscale1, vscale2, vscale3;
                    if constexpr (SharedExponentFamily) {
                        const std::size_t uv =
                            static_cast<std::size_t>(ib) *
                                static_cast<std::size_t>(
                                    ket_primitive_count) +
                            static_cast<std::size_t>(ik);
                        const RysSsExponentFamilyFactor& factor =
                            family_factors[uv];
                        vrho0 = _mm256_set1_pd(factor.rho);
                        vrho1 = vrho0;
                        vrho2 = vrho0;
                        vrho3 = vrho0;
                        vscale0 = _mm256_set1_pd(factor.prefactor_scale);
                        vscale1 = vscale0;
                        vscale2 = vscale0;
                        vscale3 = vscale0;
                    } else {
                        const __m256d vpq0 = _mm256_add_pd(vp, vq0);
                        const __m256d vpq1 = _mm256_add_pd(vp, vq1);
                        const __m256d vpq2 = _mm256_add_pd(vp, vq2);
                        const __m256d vpq3 = _mm256_add_pd(vp, vq3);
                        const __m256d vinv0 = _mm256_div_pd(vone, vpq0);
                        const __m256d vinv1 = _mm256_div_pd(vone, vpq1);
                        const __m256d vinv2 = _mm256_div_pd(vone, vpq2);
                        const __m256d vinv3 = _mm256_div_pd(vone, vpq3);
                        vrho0 =
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq0, vinv0));
                        vrho1 =
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq1, vinv1));
                        vrho2 =
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq2, vinv2));
                        vrho3 =
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq3, vinv3));
                        vscale0 = _mm256_mul_pd(
                            vtwo_pi52, _mm256_sqrt_pd(vinv0));
                        vscale1 = _mm256_mul_pd(
                            vtwo_pi52, _mm256_sqrt_pd(vinv1));
                        vscale2 = _mm256_mul_pd(
                            vtwo_pi52, _mm256_sqrt_pd(vinv2));
                        vscale3 = _mm256_mul_pd(
                            vtwo_pi52, _mm256_sqrt_pd(vinv3));
                    }

                    auto make_t =
                        [&](std::size_t lane_offset, __m256d vrho) {
                            const __m256d vdx = _mm256_sub_pd(
                                vPx, _mm256_loadu_pd(
                                    ket_bin.Wx.data() + koff + lane_offset));
                            const __m256d vdy = _mm256_sub_pd(
                                vPy, _mm256_loadu_pd(
                                    ket_bin.Wy.data() + koff + lane_offset));
                            const __m256d vdz = _mm256_sub_pd(
                                vPz, _mm256_loadu_pd(
                                    ket_bin.Wz.data() + koff + lane_offset));
                            const __m256d vr2 = _mm256_fmadd_pd(
                                vdx, vdx,
                                _mm256_fmadd_pd(
                                    vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                            return _mm256_mul_pd(vrho, vr2);
                        };
                    const __m256d vt0 = make_t(0, vrho0);
                    const __m256d vt1 = make_t(4, vrho1);
                    const __m256d vt2 = make_t(8, vrho2);
                    const __m256d vt3 = make_t(12, vrho3);
                    __m256d vw0, vw1, vw2, vw3;
                    if constexpr (std::is_same_v <
                                      Emit, RysDirectJSpeedProbeEmit > ) {
#ifdef TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE
                        vw0 = rys_j_exact_n1_weight4_lagrange(vt0);
                        vw1 = rys_j_exact_n1_weight4_lagrange(vt1);
                        vw2 = rys_j_exact_n1_weight4_lagrange(vt2);
                        vw3 = rys_j_exact_n1_weight4_lagrange(vt3);
#else
                        __m256d unused_root;
                        rys_j_speed_probe_root_weight1(
                            vt0, unused_root, vw0);
                        rys_j_speed_probe_root_weight1(
                            vt1, unused_root, vw1);
                        rys_j_speed_probe_root_weight1(
                            vt2, unused_root, vw2);
                        rys_j_speed_probe_root_weight1(
                            vt3, unused_root, vw3);
#endif
                    } else if constexpr (
                        rys_emit_uses_fast_sp_producer_v<Emit>) {
                        const __m256d vmax01 =
                            _mm256_max_pd(vt0, vt1);
                        const __m256d vmax23 =
                            _mm256_max_pd(vt2, vt3);
                        const __m256d vmax =
                            _mm256_max_pd(vmax01, vmax23);
                        if (_mm256_movemask_pd(_mm256_cmp_pd(
                                vmax, vmedium_max, _CMP_LE_OQ)) == 15) {
                            rys_ssss_weight1_medium4(
                                vt0, vt1, vt2, vt3,
                                vw0, vw1, vw2, vw3);
                        } else if (_mm256_movemask_pd(_mm256_cmp_pd(
                                       vmax, vglobal_max,
                                       _CMP_LE_OQ)) == 15) {
                            rys_ssss_weight1_global4(
                                vt0, vt1, vt2, vt3,
                                vw0, vw1, vw2, vw3);
                        } else {
                            vw0 = rys_ssss_weight1_exact(vt0);
                            vw1 = rys_ssss_weight1_exact(vt1);
                            vw2 = rys_ssss_weight1_exact(vt2);
                            vw3 = rys_ssss_weight1_exact(vt3);
                        }
                    } else {
                        // Preserve the Phase-A non-Packed policy byte for byte
                        // at the evaluator boundary: its established nprim=3
                        // host already used the medium-domain rational and the
                        // exact table/fallback outside it.  Only PackedTensor
                        // receives the new global [5,12] extension.
                        const __m256d vmax01 =
                            _mm256_max_pd(vt0, vt1);
                        const __m256d vmax23 =
                            _mm256_max_pd(vt2, vt3);
                        const __m256d vmax =
                            _mm256_max_pd(vmax01, vmax23);
                        if (_mm256_movemask_pd(_mm256_cmp_pd(
                                vmax, vmedium_max, _CMP_LE_OQ)) == 15) {
                            rys_ssss_weight1_medium4(
                                vt0, vt1, vt2, vt3,
                                vw0, vw1, vw2, vw3);
                        } else {
                            vw0 = rys_ssss_weight1_exact(vt0);
                            vw1 = rys_ssss_weight1_exact(vt1);
                            vw2 = rys_ssss_weight1_exact(vt2);
                            vw3 = rys_ssss_weight1_exact(vt3);
                        }
                    }
                    const __m256d vkw0 =
                        _mm256_loadu_pd(ket_bin.w.data() + koff);
                    const __m256d vkw1 =
                        _mm256_loadu_pd(ket_bin.w.data() + koff + 4);
                    const __m256d vkw2 =
                        _mm256_loadu_pd(ket_bin.w.data() + koff + 8);
                    const __m256d vkw3 =
                        _mm256_loadu_pd(ket_bin.w.data() + koff + 12);
                    sum0 = _mm256_fmadd_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vscale0, vbw), vkw0),
                        vw0, sum0);
                    sum1 = _mm256_fmadd_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vscale1, vbw), vkw1),
                        vw1, sum1);
                    sum2 = _mm256_fmadd_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vscale2, vbw), vkw2),
                        vw2, sum2);
                    sum3 = _mm256_fmadd_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vscale3, vbw), vkw3),
                        vw3, sum3);
                }
            }
            if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
                // WHAT: close sixteen moving ket destinations directly and
                // retain only one scalar delta for the fixed bra destination.
                // WHY: all four YMMs already hold final contracted ERIs;
                // storing raw16 would add a write/read round trip before the
                // two Coulomb terms and repeatedly RMW the fixed destination.
                // PURPOSE: an arbitrary-length ket sweep writes every moving
                // J pair once and the common pair once at the end of bra_pos.
                const unsigned active0 = Screened ? shell_live[0] : 0x0fu;
                const unsigned active1 = Screened ? shell_live[1] : 0x0fu;
                const unsigned active2 = Screened ? shell_live[2] : 0x0fu;
                const unsigned active3 = Screened ? shell_live[3] : 0x0fu;
                if (active0 != 0u)
                    rys_j_ssss_consume_moving_packet<Emit, true>(
                        *j_context, bd, ket_bin, ket_pos,
                        active0, sum0, fixed, fixed_delta, fixed_terms);
                if (active1 != 0u)
                    rys_j_ssss_consume_moving_packet<Emit, true>(
                        *j_context, bd, ket_bin, ket_pos + 4u,
                        active1, sum1, fixed, fixed_delta, fixed_terms);
                if (active2 != 0u)
                    rys_j_ssss_consume_moving_packet<Emit, true>(
                        *j_context, bd, ket_bin, ket_pos + 8u,
                        active2, sum2, fixed, fixed_delta, fixed_terms);
                if (active3 != 0u)
                    rys_j_ssss_consume_moving_packet<Emit, true>(
                        *j_context, bd, ket_bin, ket_pos + 12u,
                        active3, sum3, fixed, fixed_delta, fixed_terms);
            } else {
                _mm256_store_pd(raw16, sum0);
                _mm256_store_pd(raw16 + 4, sum1);
                _mm256_store_pd(raw16 + 8, sum2);
                _mm256_store_pd(raw16 + 12, sum3);
                for (std::size_t lane = 0; lane < 16; ++lane) {
                    if constexpr (Screened) {
                        if ((shell_live[lane / 4u] &
                             (1u << (lane % 4u))) == 0u) {
                            continue;
                        }
                    }
                    const std::size_t ket_lane = ket_pos + lane;
                    rys_emit_ssss_shell_pair_value<Emit>(
                        bd, bra_group, ket_bin.pair_group_ids[ket_lane],
                        raw16[lane], eri_values);
                }
            }
        }

        // A short remainder retains the existing table/root Boys evaluator for
        // every exact policy.  The speed-only J marker uses the same explicit
        // approximation as its full tile.  Padded AoSoA storage makes the final
        // four-lane load safe; only real lanes are consumed or emitted.
        for (std::size_t ket_pos = full_limit; ket_pos < ket_limit;
             ket_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, ket_limit - ket_pos);
            unsigned shell_live = 0u;
            unsigned shell_q_live = 0u;
            if constexpr (Screened) {
                shell_live = rys_sp_shell_mask_ket4(
                    bra_bin, bra_pos, ket_bin, ket_pos, valid_lanes,
                    shell_q_live);
                if (shell_live == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        shell_q_live == 0u) {
                        // WHAT/WHY: the tail is monotone only in Q.
                        break;
                    }
                    continue;
                }
            } else if constexpr (PrimitiveFast) {
                shell_live = rys_low_bits(valid_lanes);
            }

            __m256d sum = _mm256_setzero_pd();
            for (int ib = 0; ib < bra_primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride +
                    bra_pos;
                const __m256d vp = _mm256_set1_pd(bra_bin.s[boff]);
                const __m256d vPx = _mm256_set1_pd(bra_bin.Wx[boff]);
                const __m256d vPy = _mm256_set1_pd(bra_bin.Wy[boff]);
                const __m256d vPz = _mm256_set1_pd(bra_bin.Wz[boff]);
                const __m256d vbw = _mm256_set1_pd(bra_bin.w[boff]);
                __m256d vbra_fast;
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_set1_pd(
                        bra_bin.primitive_fast_schwarz[boff]);
                }
                for (int ik = 0; ik < ket_primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride +
                        ket_pos;
                    if constexpr (PrimitiveFast) {
                        const __m256d vket_fast = _mm256_loadu_pd(
                            ket_bin.primitive_fast_schwarz.data() +
                            koff);
                        if (rys_primitive_fast_survival_mask(
                                vbra_fast, vket_fast,
                                shell_live) == 0u) {
                            continue;
                        }
                    }
                    const __m256d vq =
                        _mm256_loadu_pd(ket_bin.s.data() + koff);
                    __m256d vrho;
                    __m256d vscale;
                    if constexpr (SharedExponentFamily) {
                        const std::size_t uv =
                            static_cast<std::size_t>(ib) *
                                static_cast<std::size_t>(
                                    ket_primitive_count) +
                            static_cast<std::size_t>(ik);
                        const RysSsExponentFamilyFactor& factor =
                            family_factors[uv];
                        vrho = _mm256_set1_pd(factor.rho);
                        vscale = _mm256_set1_pd(factor.prefactor_scale);
                    } else {
                        const __m256d vinvpq = _mm256_div_pd(
                            vone, _mm256_add_pd(vp, vq));
                        vrho = _mm256_mul_pd(
                            vp, _mm256_mul_pd(vq, vinvpq));
                        vscale = _mm256_mul_pd(
                            vtwo_pi52, _mm256_sqrt_pd(vinvpq));
                    }
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_loadu_pd(ket_bin.Wx.data() + koff));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_loadu_pd(ket_bin.Wy.data() + koff));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_loadu_pd(ket_bin.Wz.data() + koff));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(
                            vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    const __m256d vtarg = _mm256_mul_pd(vrho, vr2);
                    __m256d vweight;
                    if constexpr (std::is_same_v <
                                      Emit, RysDirectJSpeedProbeEmit > ) {
#ifdef TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE
                        vweight = rys_j_exact_n1_weight4_lagrange(vtarg);
#else
                        __m256d unused_root;
                        rys_j_speed_probe_root_weight1(
                            vtarg, unused_root, vweight);
#endif
                    } else {
                        vweight = rys_ssss_weight1_exact(vtarg);
                    }
                    const __m256d vkw =
                        _mm256_loadu_pd(ket_bin.w.data() + koff);
                    sum = _mm256_fmadd_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vscale, vbw), vkw),
                        vweight, sum);
                }
            }
            if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
                const unsigned active_bits = Screened
                    ? shell_live
                    : rys_low_bits(valid_lanes);
                if (active_bits != 0u) {
                    rys_j_ssss_consume_moving_packet<Emit, true>(
                        *j_context, bd, ket_bin, ket_pos,
                        active_bits, sum, fixed, fixed_delta, fixed_terms);
                }
            } else {
                _mm256_store_pd(raw4, sum);
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((shell_live & (1u << lane)) == 0u) continue;
                    }
                    const std::size_t ket_lane = ket_pos + lane;
                    rys_emit_ssss_shell_pair_value<Emit>(
                        bd, bra_group, ket_bin.pair_group_ids[ket_lane],
                        raw4[lane], eri_values);
                }
            }
        }
        if constexpr (rys_emit_uses_preemit_ssss_j_v<Emit>) {
            rys_j_ssss_commit<Emit>(
                *j_context, fixed.matrix_index, fixed_delta, fixed_terms);
        }
    }
}

template <bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_ssss_wide_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values) {
    const bool shared_exponent_family =
        rys_sp_bin_has_one_exponent_family<Screened>(bra_bin) &&
        rys_sp_bin_has_one_exponent_family<Screened>(ket_bin);
    if (shared_exponent_family) {
        accumulate_rys_ssss_wide_shell_bin_pair_impl <
            true, Screened, PrimitiveFast, Emit > (
            bd, bra_bin, ket_bin, same_bin, eri_values);
    } else {
        accumulate_rys_ssss_wide_shell_bin_pair_impl <
            false, Screened, PrimitiveFast, Emit > (
            bd, bra_bin, ket_bin, same_bin, eri_values);
    }
}

template <bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_ssss_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values) {
    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    // The production packed stream and the isolated fast-s/p J experiment use
    // the arbitrary-depth 1x16 host.  Other emitters retain the historical
    // nine-by-nine selector so validation/control code generation is unchanged.
    const bool wide_contraction_supported =
        rys_emit_uses_fast_sp_producer_v<Emit> ||
        (bra_bin.primitive_count == 9 && ket_bin.primitive_count == 9);
    if (wide_contraction_supported && same_bin && ket_pair_count >= 16) {
        accumulate_rys_ssss_wide_shell_bin_pair <
            Screened, PrimitiveFast, Emit > (
            bd, bra_bin, ket_bin, true, eri_values);
        return;
    }
    if (wide_contraction_supported && !same_bin &&
        std::max(bra_pair_count, ket_pair_count) >= 16) {
        // The 1x16 kernel wants the larger population in its ket/vector
        // dimension.  (ss|ss) pair exchange leaves the emitted canonical
        // pair-pair index and normalization unchanged.
        if (ket_pair_count >= bra_pair_count) {
            accumulate_rys_ssss_wide_shell_bin_pair <
                Screened, PrimitiveFast, Emit > (
                bd, bra_bin, ket_bin, false, eri_values);
        } else {
            accumulate_rys_ssss_wide_shell_bin_pair <
                Screened, PrimitiveFast, Emit > (
                bd, ket_bin, bra_bin, false, eri_values);
        }
        return;
    }
    accumulate_rys_ssss_shell_bin_pair_fallback <
        Screened, PrimitiveFast, Emit > (
        bd, bra_bin, ket_bin, same_bin, eri_values);
}

#include "integrals/two_electron/fock/detail/sp_row_receivers.ipp"

// Cross-shellset kernel for all (ss|ps) integrals. ERI pair symmetry lets us
// choose whichever pair population is larger as the four-lane bra dimension;
// BraP selects C00 (ps lanes) versus D00 (one scalar ps ket). The three p
// components stay in registers and are emitted only after full contraction.
template <bool BraP, bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_ssps_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values) {
    alignas(32) double targ4[4];
    alignas(32) double roots4[4];
    alignas(32) double weights4[4];
    constexpr int ket_tile = 2;
    alignas(32) double raw[ket_tile][3][4];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    const bool use_exponent_family =
        rys_emit_uses_fast_sp_producer_v<Emit> &&
        rys_sp_bin_has_one_exponent_family<Screened>(bra_bin) &&
        rys_sp_bin_has_one_exponent_family<Screened>(ket_bin);
    const int ket_primitive_count = ket_bin.primitive_count;
    thread_local std::vector<RysSpExponentFamilyFactor> family_factors;
    if (use_exponent_family) {
        fill_rys_sp_exponent_family_factors(
            bra_bin, ket_bin, family_factors);
    }

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_count;
         ket_pos += static_cast<std::size_t>(ket_tile)) {
        const int ket_slots = static_cast<int>(
            std::min<std::size_t>(ket_tile, ket_pair_count - ket_pos));
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_count - bra_pos);
            unsigned shell_live[ket_tile] = {};
            unsigned shell_q_live[ket_tile] = {};
            if constexpr (Screened) {
                unsigned live_union = 0u;
                unsigned q_live_union = 0u;
                for (int slot = 0; slot < ket_slots; ++slot) {
                    shell_live[slot] = rys_sp_shell_mask_bra4(
                        bra_bin, bra_pos, valid_lanes,
                        ket_bin,
                        ket_pos + static_cast<std::size_t>(slot),
                        false, shell_q_live[slot]);
                    live_union |= shell_live[slot];
                    q_live_union |= shell_q_live[slot];
                }
                if (live_union == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        q_live_union == 0u) {
                        // WHAT/WHY: do not turn a non-monotone P rejection
                        // into a break of the Q-descending bra lanes.
                        break;
                    }
                    continue;
                }
            } else if constexpr (PrimitiveFast) {
                const unsigned logical_bits =
                    rys_low_bits(valid_lanes);
                for (int slot = 0; slot < ket_slots; ++slot) {
                    shell_live[slot] = logical_bits;
                }
            }

            __m256d acc[ket_tile][3];
            for (auto& slot : acc) {
                for (auto& value : slot) value = _mm256_setzero_pd();
            }
            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                __m256d vbra_fast;
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                __m256d vPAx, vPAy, vPAz;
                if constexpr (BraP) {
                    vPAx = _mm256_loadu_pd(bra_bin.Lx.data() + boff);
                    vPAy = _mm256_loadu_pd(bra_bin.Ly.data() + boff);
                    vPAz = _mm256_loadu_pd(bra_bin.Lz.data() + boff);
                }
                for (int slot = 0; slot < ket_slots; ++slot) {
                    if constexpr (Screened) {
                        if (shell_live[slot] == 0u) continue;
                    }
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                        const std::size_t koff =
                            static_cast<std::size_t>(ik) * ket_bin.stride + ket_lane;
                        if constexpr (PrimitiveFast) {
                            const __m256d vket_fast = _mm256_set1_pd(
                                ket_bin.primitive_fast_schwarz[koff]);
                            if (rys_primitive_fast_survival_mask(
                                    vbra_fast, vket_fast,
                                    shell_live[slot]) == 0u) {
                                continue;
                            }
                        }
                        const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                        __m256d vinvpq;
                        __m256d vrho;
                        __m256d vprefactor_scale;
                        if (use_exponent_family) {
                            const std::size_t uv =
                                static_cast<std::size_t>(ib) *
                                    static_cast<std::size_t>(
                                        ket_primitive_count) +
                                static_cast<std::size_t>(ik);
                            const RysSpExponentFamilyFactor& factor =
                                family_factors[uv];
                            vinvpq = _mm256_set1_pd(factor.invpq);
                            vrho = _mm256_set1_pd(factor.rho);
                            vprefactor_scale =
                                _mm256_set1_pd(factor.prefactor_scale);
                        } else {
                            vinvpq = _mm256_div_pd(
                                vone, _mm256_add_pd(vp, vq));
                            vrho = _mm256_mul_pd(
                                vp, _mm256_mul_pd(vq, vinvpq));
                            vprefactor_scale = _mm256_sqrt_pd(vinvpq);
                        }
                        const __m256d vdx = _mm256_sub_pd(
                            vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                        const __m256d vdy = _mm256_sub_pd(
                            vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                        const __m256d vdz = _mm256_sub_pd(
                            vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                        const __m256d vr2 = _mm256_fmadd_pd(
                            vdx, vdx,
                            _mm256_fmadd_pd(
                                vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                        const __m256d vtarg = _mm256_mul_pd(vrho, vr2);
                        if constexpr (
                            rys_emit_uses_fast_sp_producer_v<Emit>) {
                            const int medium_bits = _mm256_movemask_pd(
                                _mm256_cmp_pd(
                                vtarg,
                                _mm256_set1_pd(kRysSpBoysMediumMaxT),
                                _CMP_LE_OQ));
                            const int global_bits = _mm256_movemask_pd(
                                _mm256_cmp_pd(
                                    vtarg,
                                    _mm256_set1_pd(kRysSpBoysGlobalMaxT),
                                    _CMP_LE_OQ));
                            __m256d boys0;
                            __m256d boys1;
                            __m256d unused_boys2;
                            if (medium_bits == 15) {
                                rys_psss_boys01_medium(
                                    vtarg, boys0, boys1);
                            } else if (global_bits == 15) {
                                rys_sp_boys012_global(
                                    vtarg, boys0, boys1, unused_boys2);
                            }
                            if (medium_bits == 15 || global_bits == 15) {
                                const __m256d vprefactor =
                                    use_exponent_family
                                        ? _mm256_mul_pd(
                                              _mm256_mul_pd(
                                                  vprefactor_scale, vbw),
                                              _mm256_set1_pd(
                                                  ket_bin.w[koff]))
                                        : _mm256_mul_pd(
                                              _mm256_mul_pd(
                                                  _mm256_mul_pd(
                                                      vtwo_pi52, vbw),
                                                  _mm256_set1_pd(
                                                      ket_bin.w[koff])),
                                              vprefactor_scale);
                                __m256d vx;
                                __m256d vy;
                                __m256d vz;
                                if constexpr (BraP) {
                                    const __m256d f1_scale = _mm256_mul_pd(
                                        _mm256_mul_pd(vq, vinvpq), boys1);
                                    vx = _mm256_fnmadd_pd(
                                        f1_scale, vdx,
                                        _mm256_mul_pd(boys0, vPAx));
                                    vy = _mm256_fnmadd_pd(
                                        f1_scale, vdy,
                                        _mm256_mul_pd(boys0, vPAy));
                                    vz = _mm256_fnmadd_pd(
                                        f1_scale, vdz,
                                        _mm256_mul_pd(boys0, vPAz));
                                } else {
                                    const __m256d f1_scale = _mm256_mul_pd(
                                        _mm256_mul_pd(vp, vinvpq), boys1);
                                    vx = _mm256_fmadd_pd(
                                        f1_scale, vdx,
                                        _mm256_mul_pd(
                                            boys0,
                                            _mm256_set1_pd(
                                                ket_bin.Lx[koff])));
                                    vy = _mm256_fmadd_pd(
                                        f1_scale, vdy,
                                        _mm256_mul_pd(
                                            boys0,
                                            _mm256_set1_pd(
                                                ket_bin.Ly[koff])));
                                    vz = _mm256_fmadd_pd(
                                        f1_scale, vdz,
                                        _mm256_mul_pd(
                                            boys0,
                                            _mm256_set1_pd(
                                                ket_bin.Lz[koff])));
                                }
                                acc[slot][0] = _mm256_fmadd_pd(
                                    vprefactor, vx, acc[slot][0]);
                                acc[slot][1] = _mm256_fmadd_pd(
                                    vprefactor, vy, acc[slot][1]);
                                acc[slot][2] = _mm256_fmadd_pd(
                                    vprefactor, vz, acc[slot][2]);
                                continue;
                            }
                        }
                        _mm256_store_pd(targ4, vtarg);
                        // The packed low-T packet above has already consumed
                        // F0/F1 directly.  Mixed or T>12 packets, plus every
                        // non-packed policy, retain the Emit-aware N=1 table
                        // and scalar fallback here.
                        rys_roots_weights_batch4_for_emit<Emit, 1>(
                            targ4, roots4, weights4);
                        const __m256d vroot = _mm256_load_pd(roots4);
                        const __m256d vprefactor =
                            use_exponent_family
                                ? _mm256_mul_pd(
                                      _mm256_mul_pd(
                                          vprefactor_scale, vbw),
                                      _mm256_set1_pd(ket_bin.w[koff]))
                                : _mm256_mul_pd(
                                      _mm256_mul_pd(
                                          _mm256_mul_pd(vtwo_pi52, vbw),
                                          _mm256_set1_pd(ket_bin.w[koff])),
                                      vprefactor_scale);
                        const __m256d vww = _mm256_mul_pd(
                            vprefactor, _mm256_load_pd(weights4));
                        __m256d vx, vy, vz;
                        if constexpr (BraP) {
                            const __m256d vscale = _mm256_mul_pd(
                                _mm256_mul_pd(vq, vinvpq), vroot);
                            vx = _mm256_fnmadd_pd(vscale, vdx, vPAx);
                            vy = _mm256_fnmadd_pd(vscale, vdy, vPAy);
                            vz = _mm256_fnmadd_pd(vscale, vdz, vPAz);
                        } else {
                            const __m256d vscale = _mm256_mul_pd(
                                _mm256_mul_pd(vp, vinvpq), vroot);
                            vx = _mm256_fmadd_pd(
                                vscale, vdx,
                                _mm256_set1_pd(ket_bin.Lx[koff]));
                            vy = _mm256_fmadd_pd(
                                vscale, vdy,
                                _mm256_set1_pd(ket_bin.Ly[koff]));
                            vz = _mm256_fmadd_pd(
                                vscale, vdz,
                                _mm256_set1_pd(ket_bin.Lz[koff]));
                        }
                        acc[slot][0] = _mm256_fmadd_pd(
                            vww, vx, acc[slot][0]);
                        acc[slot][1] = _mm256_fmadd_pd(
                            vww, vy, acc[slot][1]);
                        acc[slot][2] = _mm256_fmadd_pd(
                            vww, vz, acc[slot][2]);
                    }
                }
            }
            if constexpr (rys_emit_uses_preemit_ssps_positive_k_v<Emit>) {
#ifdef TLNO_RYS_DIRECT_EXCHANGE
                const RysJkEmitContext& context = *rys_jk_context_slot();
                for (int slot = 0; slot < ket_slots; ++slot) {
                    const unsigned active_bits = Screened
                        ? shell_live[slot]
                        : rys_low_bits(valid_lanes);
                    if (active_bits == 0u) continue;
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    const std::uint32_t ket_group =
                        ket_bin.pair_group_ids[ket_lane];
                    rys_k_ssps_preemit_packet<BraP>(
                        context, bd, bra_bin, bra_pos, ket_group,
                        active_bits, acc[slot][0], acc[slot][1], acc[slot][2]);
                }
#endif
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_ssps_j_v<Emit>) {
                const RysJkEmitContext& context = *rys_jk_context_slot();
                for (int slot = 0; slot < ket_slots; ++slot) {
                    const unsigned active_bits = Screened
                        ? shell_live[slot]
                        : rys_low_bits(valid_lanes);
                    if (active_bits == 0u) continue;
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    const std::uint32_t ket_group =
                        ket_bin.pair_group_ids[ket_lane];
                    const auto& fixed_refs = low_batch_pair_components(bd)[
                        static_cast<std::size_t>(ket_group)];
                    rys_j_ssps_common_preemit_packet<BraP>(
                        context, bd, bra_bin, bra_pos, fixed_refs,
                        active_bits, acc[slot][0], acc[slot][1], acc[slot][2]);
                }
            } else {
                for (int slot = 0; slot < ket_slots; ++slot) {
                    for (int component = 0; component < 3; ++component) {
                        _mm256_store_pd(
                            raw[slot][component], acc[slot][component]);
                    }
                }

                for (int slot = 0; slot < ket_slots; ++slot) {
                    const std::size_t ket_lane =
                        ket_pos + static_cast<std::size_t>(slot);
                    const std::uint32_t ket_group =
                        ket_bin.pair_group_ids[ket_lane];
                    const auto& rhs_refs = low_batch_pair_components(bd)[
                        static_cast<std::size_t>(ket_group)];
                    for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                        if constexpr (Screened) {
                            if ((shell_live[slot] & (1u << lane)) == 0u) {
                                continue;
                            }
                        }
                        const std::uint32_t bra_group =
                            bra_bin.pair_group_ids[bra_pos + lane];
                        const auto& lhs_refs = low_batch_pair_components(bd)[
                            static_cast<std::size_t>(bra_group)];
                        rys_emit_block<Emit>(
                            eri_values, lhs_refs, rhs_refs, false,
                            [&raw, slot, lane](
                                const PackedPairComponentH& lhs_ref,
                                const PackedPairComponentH& rhs_ref) noexcept {
                                const std::size_t component = BraP
                                    ? static_cast<std::size_t>(
                                          lhs_ref.component_index)
                                    : static_cast<std::size_t>(
                                          rhs_ref.component_index);
                                return raw[slot][component][lane] *
                                       lhs_ref.norm * rhs_ref.norm;
                            });
                    }
                }
            }
        }
    }
}

#include "integrals/two_electron/fock/detail/sp_packet_receivers.ipp"

// WHAT: consume a complete ss|ps packet directly from its three contracted
// YMM values, using one scalar ket shell pair (`ket_step=1`).
// WHY: the old kernel stored E[component][lane] into `raw` and then performed
// two J plus four K RMWs per ERI.  J has a better topology: one family is four
// distinct destinations, while the partner destination is shared by all four
// lanes and can be closed with one horizontal reduction.
// PURPOSE: no intermediate ERI block exists on this path and J materializes
// only at its final AO-matrix destinations.  Scheduler, masks and
// primitive-fast whole-vector semantics remain unchanged; v0 deliberately
// uses the documented approximate all-register one-root evaluator above.
template <bool BraP, bool Screened, bool PrimitiveFast>
__attribute__((noinline)) void accumulate_rys_ssps_j_speed_probe_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& /*eri_values*/) {
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const RysJkEmitContext& context = *rys_jk_context_slot();
    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_count; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        const auto& fixed_refs = low_batch_pair_components(bd)[
            static_cast<std::size_t>(ket_group)];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_count - bra_pos);
            unsigned shell_live = rys_low_bits(valid_lanes);
            unsigned shell_q_live = 0u;
            if constexpr (Screened) {
                shell_live = rys_j_sp_shell_mask_bra4(
                    bra_bin, bra_pos, valid_lanes,
                    ket_bin, ket_pos, shell_q_live);
                if (shell_live == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        shell_q_live == 0u) {
                        // WHAT/WHY/PURPOSE: terminate only a Q-monotone dead
                        // tail; density-only rejection remains a continue.
                        break;
                    }
                    continue;
                }
            }

            __m256d e0 = _mm256_setzero_pd();
            __m256d e1 = _mm256_setzero_pd();
            __m256d e2 = _mm256_setzero_pd();
            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                __m256d vbra_fast = _mm256_setzero_pd();
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                __m256d vPAx = _mm256_setzero_pd();
                __m256d vPAy = _mm256_setzero_pd();
                __m256d vPAz = _mm256_setzero_pd();
                if constexpr (BraP) {
                    vPAx = _mm256_loadu_pd(bra_bin.Lx.data() + boff);
                    vPAy = _mm256_loadu_pd(bra_bin.Ly.data() + boff);
                    vPAz = _mm256_loadu_pd(bra_bin.Lz.data() + boff);
                }
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if constexpr (PrimitiveFast) {
                        const __m256d vket_fast = _mm256_set1_pd(
                            ket_bin.primitive_fast_schwarz[koff]);
                        if (rys_primitive_fast_survival_mask(
                                vbra_fast, vket_fast, shell_live) == 0u) {
                            continue;
                        }
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(
                            vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    const __m256d vtarg = _mm256_mul_pd(
                        _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                        vr2);
                    __m256d vroot;
                    __m256d vweight;
                    rys_j_speed_probe_root_weight1(
                        vtarg, vroot, vweight);
                    const __m256d vww = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(
                                _mm256_mul_pd(vtwo_pi52, vbw),
                                _mm256_set1_pd(ket_bin.w[koff])),
                            _mm256_sqrt_pd(vinvpq)),
                        vweight);
                    __m256d vx, vy, vz;
                    if constexpr (BraP) {
                        const __m256d vscale = _mm256_mul_pd(
                            _mm256_mul_pd(vq, vinvpq), vroot);
                        vx = _mm256_fnmadd_pd(vscale, vdx, vPAx);
                        vy = _mm256_fnmadd_pd(vscale, vdy, vPAy);
                        vz = _mm256_fnmadd_pd(vscale, vdz, vPAz);
                    } else {
                        const __m256d vscale = _mm256_mul_pd(
                            _mm256_mul_pd(vp, vinvpq), vroot);
                        vx = _mm256_fmadd_pd(
                            vscale, vdx,
                            _mm256_set1_pd(ket_bin.Lx[koff]));
                        vy = _mm256_fmadd_pd(
                            vscale, vdy,
                            _mm256_set1_pd(ket_bin.Ly[koff]));
                        vz = _mm256_fmadd_pd(
                            vscale, vdz,
                            _mm256_set1_pd(ket_bin.Lz[koff]));
                    }
                    e0 = _mm256_fmadd_pd(vww, vx, e0);
                    e1 = _mm256_fmadd_pd(vww, vy, e1);
                    e2 = _mm256_fmadd_pd(vww, vz, e2);
                }
            }

            const __m256d active_mask = rys_j_active_lane_mask(shell_live);
            e0 = _mm256_and_pd(e0, active_mask);
            e1 = _mm256_and_pd(e1, active_mask);
            e2 = _mm256_and_pd(e2, active_mask);

            std::size_t moving_indices[3][4] = {};
            double moving_density[3][4] = {};
            double moving_scale[3][4] = {};
            for (std::size_t lane = 0; lane < 4; ++lane) {
                const std::uint32_t group =
                    bra_bin.pair_group_ids[bra_pos + lane];
                const auto& refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(group)];
                constexpr int moving_component_count = BraP ? 3 : 1;
                for (int component = 0;
                     component < moving_component_count; ++component) {
                    const RysJkPairInfo& info = context.pair_info[
                        refs[static_cast<std::size_t>(component)].pair_index];
                    const std::size_t index =
                        info.first_row + static_cast<std::size_t>(info.second);
                    moving_indices[component][lane] = index;
                    moving_density[component][lane] = context.p[index];
                    moving_scale[component][lane] =
                        static_cast<double>(info.scale);
                }
            }

            if constexpr (BraP) {
                const PackedPairComponentH& fixed = fixed_refs.front();
                const RysJkPairInfo& fi =
                    context.pair_info[fixed.pair_index];
                const std::size_t fixed_index =
                    fi.first_row + static_cast<std::size_t>(fi.second);
                const double common =
                    4.0 * static_cast<double>(fi.scale);
                __m256d fixed_sum = _mm256_setzero_pd();
                rys_j_ssps_commit_brap_component<0>(
                    context, shell_live, moving_indices, moving_density,
                    moving_scale, common, context.p[fixed_index], e0,
                    fixed_sum);
                rys_j_ssps_commit_brap_component<1>(
                    context, shell_live, moving_indices, moving_density,
                    moving_scale, common, context.p[fixed_index], e1,
                    fixed_sum);
                rys_j_ssps_commit_brap_component<2>(
                    context, shell_live, moving_indices, moving_density,
                    moving_scale, common, context.p[fixed_index], e2,
                    fixed_sum);
                context.g[fixed_index] += rys_j_horizontal_sum(fixed_sum);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
                // WHAT: one fixed-side write reduces three p components from
                // every live lane, while the moving-side writes above each
                // represent one logical term.
                // WHY: counting only physical commits would hide a missing
                // component behind the horizontal reduction.
                // PURPOSE: the ledger reaches exactly two J terms per
                // canonical ss|ps ERI without expanding the release path.
                g_rys_direct_j_speed_probe_audit.logical_j_terms_committed +=
                    3u * static_cast<std::uint64_t>(
                        __builtin_popcount(shell_live & 0x0fu));
#endif
                // For BraP the moving side owns three components and the
                // scalar fixed side repeats its sole component three times.
                if ((shell_live & 1u) != 0u)
                    rys_j_ssps_record_lane<true, 0>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
                if ((shell_live & 2u) != 0u)
                    rys_j_ssps_record_lane<true, 1>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
                if ((shell_live & 4u) != 0u)
                    rys_j_ssps_record_lane<true, 2>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
                if ((shell_live & 8u) != 0u)
                    rys_j_ssps_record_lane<true, 3>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
            } else {
                const __m256d lhs_scale = _mm256_set_pd(
                    moving_scale[0][3], moving_scale[0][2],
                    moving_scale[0][1], moving_scale[0][0]);
                const __m256d lhs_density = _mm256_set_pd(
                    moving_density[0][3], moving_density[0][2],
                    moving_density[0][1], moving_density[0][0]);
                __m256d lhs_sum = _mm256_setzero_pd();
                rys_j_ssps_commit_ketp_component<0>(
                    context, fixed_refs, shell_live, lhs_scale, lhs_density,
                    e0, lhs_sum);
                rys_j_ssps_commit_ketp_component<1>(
                    context, fixed_refs, shell_live, lhs_scale, lhs_density,
                    e1, lhs_sum);
                rys_j_ssps_commit_ketp_component<2>(
                    context, fixed_refs, shell_live, lhs_scale, lhs_density,
                    e2, lhs_sum);
                if ((shell_live & 1u) != 0u)
                    rys_j_ssps_commit_distinct<0, 3u>(
                        context, moving_indices[0], lhs_sum);
                if ((shell_live & 2u) != 0u)
                    rys_j_ssps_commit_distinct<1, 3u>(
                        context, moving_indices[0], lhs_sum);
                if ((shell_live & 4u) != 0u)
                    rys_j_ssps_commit_distinct<2, 3u>(
                        context, moving_indices[0], lhs_sum);
                if ((shell_live & 8u) != 0u)
                    rys_j_ssps_commit_distinct<3, 3u>(
                        context, moving_indices[0], lhs_sum);
                if ((shell_live & 1u) != 0u)
                    rys_j_ssps_record_lane<false, 0>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
                if ((shell_live & 2u) != 0u)
                    rys_j_ssps_record_lane<false, 1>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
                if ((shell_live & 4u) != 0u)
                    rys_j_ssps_record_lane<false, 2>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
                if ((shell_live & 8u) != 0u)
                    rys_j_ssps_record_lane<false, 3>(
                        context, bd, bra_bin, bra_pos,
                        fixed_refs, e0, e1, e2);
            }
        }
    }
}

// WHAT: retire the failed ps|ps producer-fused slice completely.
// WHY: AVX2 lanes enumerate shell quartets, so direct J/K still needs nine
// live E_ij vectors; assembly showed those vectors spill despite the root
// tape, replacing one materialization with another and missing the gate.
// PURPOSE: ps|ps follows the proven raw/scalar path until a producer layout
// gives several lanes one J/K destination without retaining E9.

// Cross-shellset (ps|ps) kernel. For one Rys root the Cartesian 3x3 block is
// C_i*D_j + delta_ij*B00. Keeping the nine contractions live across both roots
// avoids materializing a G table; independent shellsets again occupy AVX lanes.
template <bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_psps_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values) {
    alignas(32) double targ4[4];
    alignas(32) double roots8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double weights8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double raw[9][4];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);

    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    for (std::size_t ket_pos = 0; ket_pos < ket_pair_count; ++ket_pos) {
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_count - bra_pos);
            if (same_bin && ket_pos >= bra_pos + valid_lanes) continue;
            unsigned shell_live = 0u;
            unsigned shell_q_live = 0u;
            if constexpr (Screened) {
                shell_live = rys_sp_shell_mask_bra4(
                    bra_bin, bra_pos, valid_lanes,
                    ket_bin, ket_pos, same_bin, shell_q_live);
                if (shell_live == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        shell_q_live == 0u) {
                        // WHAT/WHY: break uses the Q part of the mask, never P.
                        break;
                    }
                    continue;
                }
            } else if constexpr (PrimitiveFast) {
                shell_live = rys_sp_shell_logical_bra4(
                    bra_pos, valid_lanes, ket_pos, same_bin);
            }
            RysJPspsRowSidecar psps_row_sidecar;
            if constexpr (
                rys_emit_uses_pairspace_preemit_psps_row_j_v<Emit>) {
                // Resolve every pair index, norm and density before E9 becomes
                // live.  No metadata call is allowed between the final
                // recurrence and the three row reductions below.
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_sp_shell_logical_bra4(
                          bra_pos, valid_lanes, ket_pos, same_bin);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                rys_j_psps_prepare_row_sidecar(
                    psps_row_sidecar, *rys_jk_context_slot(), bd,
                    bra_bin, bra_pos, ket_group, rhs_refs, active_bits);
            }
            // Keep the proven ps|ps recurrence and exact raw/scalar emitter.
            __m256d e00 = _mm256_setzero_pd();
            __m256d e01 = _mm256_setzero_pd();
            __m256d e02 = _mm256_setzero_pd();
            __m256d e10 = _mm256_setzero_pd();
            __m256d e11 = _mm256_setzero_pd();
            __m256d e12 = _mm256_setzero_pd();
            __m256d e20 = _mm256_setzero_pd();
            __m256d e21 = _mm256_setzero_pd();
            __m256d e22 = _mm256_setzero_pd();

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPAx = _mm256_loadu_pd(bra_bin.Lx.data() + boff);
                const __m256d vPAy = _mm256_loadu_pd(bra_bin.Ly.data() + boff);
                const __m256d vPAz = _mm256_loadu_pd(bra_bin.Lz.data() + boff);
                __m256d vbra_fast;
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if constexpr (PrimitiveFast) {
                        const __m256d vket_fast = _mm256_set1_pd(
                            ket_bin.primitive_fast_schwarz[koff]);
                        if (rys_primitive_fast_survival_mask(
                                vbra_fast, vket_fast,
                                shell_live) == 0u) {
                            continue;
                        }
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vinvpq = _mm256_div_pd(
                        vone, _mm256_add_pd(vp, vq));
                    const __m256d vrho = _mm256_mul_pd(
                        vp, _mm256_mul_pd(vq, vinvpq));
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    const __m256d vtarg = _mm256_mul_pd(vrho, vr2);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    if constexpr (
                        rys_emit_uses_fast_sp_producer_v<Emit>) {
                        const int medium_bits = _mm256_movemask_pd(
                            _mm256_cmp_pd(
                                vtarg,
                                _mm256_set1_pd(kRysSpBoysMediumMaxT),
                                _CMP_LE_OQ));
                        const int global_bits = _mm256_movemask_pd(
                            _mm256_cmp_pd(
                                vtarg,
                                _mm256_set1_pd(kRysSpBoysGlobalMaxT),
                                _CMP_LE_OQ));
                        __m256d boys0;
                        __m256d boys1;
                        __m256d boys2;
                        if (medium_bits == 15) {
                            rys_sp_boys012_medium(
                                vtarg, boys0, boys1, boys2);
                        } else if (global_bits == 15) {
                            rys_sp_boys012_global(
                                vtarg, boys0, boys1, boys2);
                        }
                        if (medium_bits == 15 || global_bits == 15) {
                            const __m256d va = _mm256_mul_pd(vq, vinvpq);
                            const __m256d vb = _mm256_mul_pd(vp, vinvpq);
                            const __m256d C1x = _mm256_mul_pd(
                                _mm256_sub_pd(_mm256_setzero_pd(), va), vdx);
                            const __m256d C1y = _mm256_mul_pd(
                                _mm256_sub_pd(_mm256_setzero_pd(), va), vdy);
                            const __m256d C1z = _mm256_mul_pd(
                                _mm256_sub_pd(_mm256_setzero_pd(), va), vdz);
                            const __m256d D0x =
                                _mm256_set1_pd(ket_bin.Lx[koff]);
                            const __m256d D0y =
                                _mm256_set1_pd(ket_bin.Ly[koff]);
                            const __m256d D0z =
                                _mm256_set1_pd(ket_bin.Lz[koff]);
                            const __m256d D1x = _mm256_mul_pd(vb, vdx);
                            const __m256d D1y = _mm256_mul_pd(vb, vdy);
                            const __m256d D1z = _mm256_mul_pd(vb, vdz);
                            const __m256d h = _mm256_mul_pd(vhalf, vinvpq);
                            auto moment_product = [&](__m256d C0,
                                                      __m256d C1,
                                                      __m256d D0,
                                                      __m256d D1,
                                                      bool diagonal) {
                                __m256d first = _mm256_fmadd_pd(
                                    C0, D1, _mm256_mul_pd(C1, D0));
                                if (diagonal) first = _mm256_add_pd(first, h);
                                return _mm256_fmadd_pd(
                                    boys2, _mm256_mul_pd(C1, D1),
                                    _mm256_fmadd_pd(
                                        boys1, first,
                                        _mm256_mul_pd(
                                            boys0,
                                            _mm256_mul_pd(C0, D0))));
                            };
                            e00 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAx, C1x, D0x, D1x, true),
                                e00);
                            e01 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAx, C1x, D0y, D1y, false),
                                e01);
                            e02 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAx, C1x, D0z, D1z, false),
                                e02);
                            e10 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAy, C1y, D0x, D1x, false),
                                e10);
                            e11 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAy, C1y, D0y, D1y, true),
                                e11);
                            e12 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAy, C1y, D0z, D1z, false),
                                e12);
                            e20 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAz, C1z, D0x, D1x, false),
                                e20);
                            e21 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAz, C1z, D0y, D1y, false),
                                e21);
                            e22 = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    vPAz, C1z, D0z, D1z, true),
                                e22);
                            continue;
                        }
                    }
                    _mm256_store_pd(targ4, vtarg);
                    rys_roots_weights_batch4_for_emit<Emit, 2>(
                        targ4, roots8, weights8);
                    for (int root = 0; root < ActiveRoots<Emit, 2>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots8 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights8 + 4 * root));
                        const __m256d vqscale =
                            _mm256_mul_pd(_mm256_mul_pd(vq, vinvpq), vrt);
                        const __m256d vpscale =
                            _mm256_mul_pd(_mm256_mul_pd(vp, vinvpq), vrt);
                        const __m256d Cx = _mm256_fnmadd_pd(vqscale, vdx, vPAx);
                        const __m256d Cy = _mm256_fnmadd_pd(vqscale, vdy, vPAy);
                        const __m256d Cz = _mm256_fnmadd_pd(vqscale, vdz, vPAz);
                        const __m256d Dx = _mm256_fmadd_pd(
                            vpscale, vdx, _mm256_set1_pd(ket_bin.Lx[koff]));
                        const __m256d Dy = _mm256_fmadd_pd(
                            vpscale, vdy, _mm256_set1_pd(ket_bin.Ly[koff]));
                        const __m256d Dz = _mm256_fmadd_pd(
                            vpscale, vdz, _mm256_set1_pd(ket_bin.Lz[koff]));
                        const __m256d B00 =
                            _mm256_mul_pd(_mm256_mul_pd(vhalf, vinvpq), vrt);
                        e00 = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(Cx, Dx, B00), e00);
                        e01 = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(Cx, Dy), e01);
                        e02 = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(Cx, Dz), e02);
                        e10 = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(Cy, Dx), e10);
                        e11 = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(Cy, Dy, B00), e11);
                        e12 = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(Cy, Dz), e12);
                        e20 = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(Cz, Dx), e20);
                        e21 = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(Cz, Dy), e21);
                        e22 = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(Cz, Dz, B00), e22);
                    }
                }
            }
            if constexpr (rys_emit_uses_preemit_psps_positive_k_v<Emit>) {
#ifdef TLNO_RYS_DIRECT_EXCHANGE
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_sp_shell_logical_bra4(
                          bra_pos, valid_lanes, ket_pos, same_bin);
                if (active_bits != 0u) {
                    const std::uint32_t ket_group =
                        ket_bin.pair_group_ids[ket_pos];
                    rys_k_psps_preemit_packet_v2(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, active_bits,
                        e00, e01, e02, e10, e11, e12, e20, e21, e22);
                }
#endif
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_psps_row_j_v<Emit>) {
                __m256d fixed_delta0;
                __m256d fixed_delta1;
                __m256d fixed_delta2;
                if (psps_row_sidecar.alias_lane < 0) {
                    // The overwhelmingly common branch contains no dynamic
                    // lane extraction or scalar scratch, so each E row can
                    // die immediately after forming its J destinations.
                    rys_j_psps_digest_row<true, 0>(
                        psps_row_sidecar, e00, e01, e02,
                        fixed_delta0, fixed_delta1, fixed_delta2);
                    rys_j_psps_digest_row<false, 1>(
                        psps_row_sidecar, e10, e11, e12,
                        fixed_delta0, fixed_delta1, fixed_delta2);
                    rys_j_psps_digest_row<false, 2>(
                        psps_row_sidecar, e20, e21, e22,
                        fixed_delta0, fixed_delta1, fixed_delta2);
                    const double no_alias_delta[3] = {};
                    rys_j_psps_finish_rows(
                        psps_row_sidecar,
                        fixed_delta0, fixed_delta1, fixed_delta2,
                        no_alias_delta);
                } else {
                    double alias_delta[3] = {};
                    rys_j_psps_digest_row<true, 0>(
                        psps_row_sidecar, e00, e01, e02,
                        fixed_delta0, fixed_delta1, fixed_delta2);
                    rys_j_psps_digest_alias_row<0>(
                        psps_row_sidecar, e00, e01, e02, alias_delta);
                    rys_j_psps_digest_row<false, 1>(
                        psps_row_sidecar, e10, e11, e12,
                        fixed_delta0, fixed_delta1, fixed_delta2);
                    rys_j_psps_digest_alias_row<1>(
                        psps_row_sidecar, e10, e11, e12, alias_delta);
                    rys_j_psps_digest_row<false, 2>(
                        psps_row_sidecar, e20, e21, e22,
                        fixed_delta0, fixed_delta1, fixed_delta2);
                    rys_j_psps_digest_alias_row<2>(
                        psps_row_sidecar, e20, e21, e22, alias_delta);
                    rys_j_psps_finish_rows(
                        psps_row_sidecar,
                        fixed_delta0, fixed_delta1, fixed_delta2,
                        alias_delta);
                }
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_sp_j_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_sp_shell_logical_bra4(
                          bra_pos, valid_lanes, ket_pos, same_bin);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                RysJSpPreemitPacket<3, 3> digest(
                    *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                    ket_group, rhs_refs, active_bits);
                digest.consume(0, 0, e00);
                digest.consume(0, 1, e01);
                digest.consume(0, 2, e02);
                digest.consume(1, 0, e10);
                digest.consume(1, 1, e11);
                digest.consume(1, 2, e12);
                digest.consume(2, 0, e20);
                digest.consume(2, 1, e21);
                digest.consume(2, 2, e22);
                digest.finish();
            } else {
                // WHAT: retain the established raw[component][lane] emission
                // for packed/checksum policies and the scalar J/K control.
                // WHY/PURPOSE: all living consumers see the exact established
                // values and canonical s8 aliases.
                _mm256_store_pd(raw[0], e00);
                _mm256_store_pd(raw[1], e01);
                _mm256_store_pd(raw[2], e02);
                _mm256_store_pd(raw[3], e10);
                _mm256_store_pd(raw[4], e11);
                _mm256_store_pd(raw[5], e12);
                _mm256_store_pd(raw[6], e20);
                _mm256_store_pd(raw[7], e21);
                _mm256_store_pd(raw[8], e22);

                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((shell_live & (1u << lane)) == 0u) continue;
                    } else if (same_bin && ket_pos > bra_pos + lane) {
                        continue;
                    }
                    const std::uint32_t bra_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& lhs_refs = low_batch_pair_components(bd)[
                        static_cast<std::size_t>(bra_group)];
                    const bool same_pair_group = bra_group == ket_group;
                    rys_emit_block<Emit>(
                        eri_values, lhs_refs, rhs_refs, same_pair_group,
                        [&raw, lane](
                            const PackedPairComponentH& lhs_ref,
                            const PackedPairComponentH& rhs_ref) noexcept {
                            const std::size_t component =
                                3u * static_cast<std::size_t>(
                                         lhs_ref.component_index) +
                                static_cast<std::size_t>(
                                    rhs_ref.component_index);
                            return raw[component][lane] *
                                   lhs_ref.norm * rhs_ref.norm;
                        });
                }
            }
        }
    }
}

// All symmetry-equivalent (ss|pp) shellsets. The pp horizontal transfer is
// folded into the primitive recurrence (direct Boys moments for bounded
// packed packets, roots otherwise):
//   (p_a p_b|ss) = [a+b|0] + (A-B)_b [a|0].
// This leaves nine register-resident contractions and no radial/HRR buffer.
template <bool BraPP, bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_sspp_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values) {
    alignas(32) double targ4[4];
    alignas(32) double roots8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double weights8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double raw[9][4];
    static constexpr int pp_unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);

    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    for (std::size_t ket_pos = 0; ket_pos < ket_pair_count; ++ket_pos) {
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_count - bra_pos);
            unsigned shell_live = 0u;
            unsigned shell_q_live = 0u;
            if constexpr (Screened) {
                shell_live = rys_sp_shell_mask_bra4(
                    bra_bin, bra_pos, valid_lanes,
                    ket_bin, ket_pos, false, shell_q_live);
                if (shell_live == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        shell_q_live == 0u) {
                        // WHAT/WHY: a dead density lane only continues.
                        break;
                    }
                    continue;
                }
            } else if constexpr (PrimitiveFast) {
                shell_live = rys_low_bits(valid_lanes);
            }
            RysJSsppRowSidecar<BraPP> sspp_row_sidecar;
            if constexpr (
                rys_emit_uses_pairspace_preemit_sspp_row_j_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_low_bits(valid_lanes);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                rys_j_sspp_prepare_row_sidecar<BraPP>(
                    sspp_row_sidecar, *rys_jk_context_slot(), bd,
                    bra_bin, bra_pos, ket_group, rhs_refs, active_bits);
            }
            __m256d acc_hi[6];
            __m256d acc_lo[3];
            for (auto& v : acc_hi) v = _mm256_setzero_pd();
            for (auto& v : acc_lo) v = _mm256_setzero_pd();
            const __m256d vRx = BraPP
                ? _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos)
                : _mm256_set1_pd(ket_bin.Rx[ket_pos]);
            const __m256d vRy = BraPP
                ? _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos)
                : _mm256_set1_pd(ket_bin.Ry[ket_pos]);
            const __m256d vRz = BraPP
                ? _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos)
                : _mm256_set1_pd(ket_bin.Rz[ket_pos]);
            const __m256d vR[3] = {vRx, vRy, vRz};

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                __m256d vbra_fast;
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if constexpr (PrimitiveFast) {
                        const __m256d vket_fast = _mm256_set1_pd(
                            ket_bin.primitive_fast_schwarz[koff]);
                        if (rys_primitive_fast_survival_mask(
                                vbra_fast, vket_fast,
                                shell_live) == 0u) {
                            continue;
                        }
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vinvpq = _mm256_div_pd(
                        vone, _mm256_add_pd(vp, vq));
                    const __m256d vrho = _mm256_mul_pd(
                        vp, _mm256_mul_pd(vq, vinvpq));
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    const __m256d vtarg = _mm256_mul_pd(vrho, vr2);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    if constexpr (
                        rys_emit_uses_fast_sp_producer_v<Emit>) {
                        const int medium_bits = _mm256_movemask_pd(
                            _mm256_cmp_pd(
                                vtarg,
                                _mm256_set1_pd(kRysSpBoysMediumMaxT),
                                _CMP_LE_OQ));
                        const int global_bits = _mm256_movemask_pd(
                            _mm256_cmp_pd(
                                vtarg,
                                _mm256_set1_pd(kRysSpBoysGlobalMaxT),
                                _CMP_LE_OQ));
                        __m256d boys0;
                        __m256d boys1;
                        __m256d boys2;
                        if (medium_bits == 15) {
                            rys_sp_boys012_medium(
                                vtarg, boys0, boys1, boys2);
                        } else if (global_bits == 15) {
                            rys_sp_boys012_global(
                                vtarg, boys0, boys1, boys2);
                        }
                        if (medium_bits == 15 || global_bits == 15) {
                            __m256d X0[3];
                            __m256d X1[3];
                            __m256d ratio;
                            __m256d hpair;
                            if constexpr (BraPP) {
                                ratio = _mm256_mul_pd(vq, vinvpq);
                                X0[0] = _mm256_loadu_pd(
                                    bra_bin.Lx.data() + boff);
                                X0[1] = _mm256_loadu_pd(
                                    bra_bin.Ly.data() + boff);
                                X0[2] = _mm256_loadu_pd(
                                    bra_bin.Lz.data() + boff);
                                X1[0] = _mm256_mul_pd(
                                    _mm256_sub_pd(
                                        _mm256_setzero_pd(), ratio),
                                    vdx);
                                X1[1] = _mm256_mul_pd(
                                    _mm256_sub_pd(
                                        _mm256_setzero_pd(), ratio),
                                    vdy);
                                X1[2] = _mm256_mul_pd(
                                    _mm256_sub_pd(
                                        _mm256_setzero_pd(), ratio),
                                    vdz);
                                hpair = _mm256_mul_pd(
                                    vhalf,
                                    _mm256_loadu_pd(
                                        bra_bin.invs.data() + boff));
                            } else {
                                ratio = _mm256_mul_pd(vp, vinvpq);
                                X0[0] =
                                    _mm256_set1_pd(ket_bin.Lx[koff]);
                                X0[1] =
                                    _mm256_set1_pd(ket_bin.Ly[koff]);
                                X0[2] =
                                    _mm256_set1_pd(ket_bin.Lz[koff]);
                                X1[0] = _mm256_mul_pd(ratio, vdx);
                                X1[1] = _mm256_mul_pd(ratio, vdy);
                                X1[2] = _mm256_mul_pd(ratio, vdz);
                                hpair = _mm256_mul_pd(
                                    vhalf,
                                    _mm256_set1_pd(ket_bin.invs[koff]));
                            }
                            const __m256d bpair1 = _mm256_sub_pd(
                                _mm256_setzero_pd(),
                                _mm256_mul_pd(hpair, ratio));
                            auto moment_product = [&](__m256d lhs0,
                                                      __m256d lhs1,
                                                      __m256d rhs0,
                                                      __m256d rhs1,
                                                      bool diagonal) {
                                __m256d zero = _mm256_mul_pd(lhs0, rhs0);
                                __m256d first = _mm256_fmadd_pd(
                                    lhs0, rhs1,
                                    _mm256_mul_pd(lhs1, rhs0));
                                if (diagonal) {
                                    zero = _mm256_add_pd(zero, hpair);
                                    first = _mm256_add_pd(first, bpair1);
                                }
                                return _mm256_fmadd_pd(
                                    boys2, _mm256_mul_pd(lhs1, rhs1),
                                    _mm256_fmadd_pd(
                                        boys1, first,
                                        _mm256_mul_pd(boys0, zero)));
                            };
                            acc_hi[0] = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    X0[0], X1[0], X0[0], X1[0], true),
                                acc_hi[0]);
                            acc_hi[1] = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    X0[0], X1[0], X0[1], X1[1], false),
                                acc_hi[1]);
                            acc_hi[2] = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    X0[0], X1[0], X0[2], X1[2], false),
                                acc_hi[2]);
                            acc_hi[3] = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    X0[1], X1[1], X0[1], X1[1], true),
                                acc_hi[3]);
                            acc_hi[4] = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    X0[1], X1[1], X0[2], X1[2], false),
                                acc_hi[4]);
                            acc_hi[5] = _mm256_fmadd_pd(
                                vpref,
                                moment_product(
                                    X0[2], X1[2], X0[2], X1[2], true),
                                acc_hi[5]);
                            for (int axis = 0; axis < 3; ++axis) {
                                const __m256d low = _mm256_fmadd_pd(
                                    boys1, X1[axis],
                                    _mm256_mul_pd(boys0, X0[axis]));
                                acc_lo[axis] = _mm256_fmadd_pd(
                                    vpref, low, acc_lo[axis]);
                            }
                            continue;
                        }
                    }
                    _mm256_store_pd(targ4, vtarg);
                    rys_roots_weights_batch4_for_emit<Emit, 2>(
                        targ4, roots8, weights8);
                    for (int root = 0; root < ActiveRoots<Emit, 2>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots8 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights8 + 4 * root));
                        __m256d X[3];
                        __m256d Bpair;
                        if constexpr (BraPP) {
                            const __m256d scale =
                                _mm256_mul_pd(_mm256_mul_pd(vq, vinvpq), vrt);
                            X[0] = _mm256_fnmadd_pd(
                                scale, vdx,
                                _mm256_loadu_pd(bra_bin.Lx.data() + boff));
                            X[1] = _mm256_fnmadd_pd(
                                scale, vdy,
                                _mm256_loadu_pd(bra_bin.Ly.data() + boff));
                            X[2] = _mm256_fnmadd_pd(
                                scale, vdz,
                                _mm256_loadu_pd(bra_bin.Lz.data() + boff));
                            Bpair = _mm256_mul_pd(
                                _mm256_mul_pd(vhalf,
                                    _mm256_loadu_pd(bra_bin.invs.data() + boff)),
                                _mm256_fnmadd_pd(
                                    _mm256_mul_pd(vq, vinvpq), vrt, vone));
                        } else {
                            const __m256d scale =
                                _mm256_mul_pd(_mm256_mul_pd(vp, vinvpq), vrt);
                            X[0] = _mm256_fmadd_pd(
                                scale, vdx, _mm256_set1_pd(ket_bin.Lx[koff]));
                            X[1] = _mm256_fmadd_pd(
                                scale, vdy, _mm256_set1_pd(ket_bin.Ly[koff]));
                            X[2] = _mm256_fmadd_pd(
                                scale, vdz, _mm256_set1_pd(ket_bin.Lz[koff]));
                            Bpair = _mm256_mul_pd(
                                _mm256_mul_pd(vhalf,
                                    _mm256_set1_pd(ket_bin.invs[koff])),
                                _mm256_fnmadd_pd(
                                    _mm256_mul_pd(vp, vinvpq), vrt, vone));
                        }
                        acc_hi[0] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(X[0], X[0], Bpair), acc_hi[0]);
                        acc_hi[1] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(X[0], X[1]), acc_hi[1]);
                        acc_hi[2] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(X[0], X[2]), acc_hi[2]);
                        acc_hi[3] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(X[1], X[1], Bpair), acc_hi[3]);
                        acc_hi[4] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(X[1], X[2]), acc_hi[4]);
                        acc_hi[5] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(X[2], X[2], Bpair), acc_hi[5]);
                        for (int a = 0; a < 3; ++a) {
                            acc_lo[a] = _mm256_fmadd_pd(vww, X[a], acc_lo[a]);
                        }
                    }
                }
            }
            if constexpr (rys_emit_uses_preemit_sspp_positive_k_v<Emit>) {
#ifdef TLNO_RYS_DIRECT_EXCHANGE
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_low_bits(valid_lanes);
                if (active_bits != 0u) {
                    const std::uint32_t ket_group =
                        ket_bin.pair_group_ids[ket_pos];
                    rys_k_sspp_preemit_packet(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, BraPP, active_bits, vR, acc_lo, acc_hi);
                }
#endif
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_sspp_row_j_v<Emit>) {
                if constexpr (BraPP) {
                    __m256d fixed_delta = _mm256_setzero_pd();
                    rys_j_sspp_digest_moving_pp_component<0>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<1>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<2>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<3>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<4>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<5>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<6>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<7>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_digest_moving_pp_component<8>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        fixed_delta);
                    rys_j_sspp_finish_moving_pp(
                        sspp_row_sidecar, fixed_delta);
                } else {
                    __m256d moving_delta = _mm256_setzero_pd();
                    rys_j_sspp_digest_fixed_pp_component<0>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<1>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<2>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<3>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<4>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<5>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<6>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<7>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_digest_fixed_pp_component<8>(
                        sspp_row_sidecar, vR, acc_lo, acc_hi,
                        moving_delta);
                    rys_j_sspp_finish_moving_ss(
                        sspp_row_sidecar, moving_delta);
                }
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_sp_j_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_low_bits(valid_lanes);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                if constexpr (BraPP) {
                    RysJSpPreemitPacket<9, 1> digest(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, rhs_refs, active_bits);
                    for (int component = 0; component < 9; ++component) {
                        const int first_axis = component / 3;
                        const int second_axis = component - 3 * first_axis;
                        digest.consume(
                            component, 0,
                            _mm256_fmadd_pd(
                                vR[second_axis], acc_lo[first_axis],
                                acc_hi[pp_unique[component]]));
                    }
                    digest.finish();
                } else {
                    RysJSpPreemitPacket<1, 9> digest(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, rhs_refs, active_bits);
                    for (int component = 0; component < 9; ++component) {
                        const int first_axis = component / 3;
                        const int second_axis = component - 3 * first_axis;
                        digest.consume(
                            0, component,
                            _mm256_fmadd_pd(
                                vR[second_axis], acc_lo[first_axis],
                                acc_hi[pp_unique[component]]));
                    }
                    digest.finish();
                }
            } else {
                for (int component = 0; component < 9; ++component) {
                    const int first_axis = component / 3;
                    const int second_axis = component - 3 * first_axis;
                    const __m256d value = _mm256_fmadd_pd(
                        vR[second_axis], acc_lo[first_axis],
                        acc_hi[pp_unique[component]]);
                    _mm256_store_pd(raw[component], value);
                }
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((shell_live & (1u << lane)) == 0u) continue;
                    }
                    const std::uint32_t bra_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& lhs_refs = low_batch_pair_components(bd)[
                        static_cast<std::size_t>(bra_group)];
                    rys_emit_block<Emit>(
                        eri_values, lhs_refs, rhs_refs, false,
                        [&raw, lane](const PackedPairComponentH& lhs_ref,
                                     const PackedPairComponentH& rhs_ref) noexcept {
                            const std::size_t component = BraPP
                                ? static_cast<std::size_t>(
                                      lhs_ref.component_index)
                                : static_cast<std::size_t>(
                                      rhs_ref.component_index);
                            return raw[component][lane] *
                                   lhs_ref.norm * rhs_ref.norm;
                        });
                }
            }
        }
    }
}

// Only six of the nine degree<=2 axis entries are needed by (ps|pp): the
// pre-HRR pp side carries degree 0..2 while the opposite p side carries only
// degree 0..1.  Keeping the compact orientation-specific table materially
// reduces the live vector set in the already register-heavy 27-output kernel.
template <bool BraPP>
inline void rys_axis_table_pspp_v(__m256d C, __m256d D,
                                  __m256d B00, __m256d Bpair,
                                  __m256d G[6]) noexcept {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    if constexpr (BraPP) {
        // Row-major [bra degree 0..2][ket degree 0..1].
        G[0] = one;
        G[1] = D;
        G[2] = C;
        G[3] = _mm256_fmadd_pd(C, D, B00);
        G[4] = _mm256_fmadd_pd(C, C, Bpair);
        G[5] = _mm256_fmadd_pd(
            D, G[4], _mm256_mul_pd(_mm256_mul_pd(two, B00), C));
    } else {
        // Row-major [bra degree 0..1][ket degree 0..2].
        G[0] = one;
        G[1] = D;
        G[2] = _mm256_fmadd_pd(D, D, Bpair);
        G[3] = C;
        G[4] = _mm256_fmadd_pd(C, D, B00);
        G[5] = _mm256_fmadd_pd(
            D, G[4],
            _mm256_fmadd_pd(B00, D, _mm256_mul_pd(Bpair, C)));
    }
}

template <bool BraPP>
inline __m256d rys_axis_product_pspp_v(const __m256d Gx[6],
                                       const __m256d Gy[6],
                                       const __m256d Gz[6],
                                       const int bra_degree[3],
                                       const int ket_degree[3]) noexcept {
    constexpr int stride = BraPP ? 2 : 3;
    return _mm256_mul_pd(
        _mm256_mul_pd(Gx[stride * bra_degree[0] + ket_degree[0]],
                      Gy[stride * bra_degree[1] + ket_degree[1]]),
        Gz[stride * bra_degree[2] + ket_degree[2]]);
}

// All symmetry-equivalent (ps|pp) shellsets. A compact degree<=2 per-axis VRR
// table supplies the 27 radial products, and the pp HRR is folded into each
// contraction before accumulation. This is the shellset analogue of the
// existing factorized right01 HRR, but shares roots and coefficient setup over
// four independent shell pairs.
template <bool BraPP, bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_pspp_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values) {
    static constexpr int p_degree[3][3] = {
        {1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    static constexpr int d_degree[6][3] = {
        {2, 0, 0}, {1, 1, 0}, {1, 0, 1},
        {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};
    static constexpr int pp_unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    alignas(32) double targ4[4];
    alignas(32) double roots8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double weights8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double raw[27][4];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);

    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    for (std::size_t ket_pos = 0; ket_pos < ket_pair_count; ++ket_pos) {
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_count - bra_pos);
            unsigned shell_live = 0u;
            unsigned shell_q_live = 0u;
            if constexpr (Screened) {
                shell_live = rys_sp_shell_mask_bra4(
                    bra_bin, bra_pos, valid_lanes,
                    ket_bin, ket_pos, false, shell_q_live);
                if (shell_live == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        shell_q_live == 0u) {
                        // WHAT/WHY: the P classification does not order the tail.
                        break;
                    }
                    continue;
                }
            } else if constexpr (PrimitiveFast) {
                shell_live = rys_low_bits(valid_lanes);
            }
            RysJPsppRowSidecar<BraPP> pspp_row_sidecar;
            if constexpr (
                rys_emit_uses_pairspace_preemit_pspp_row_j_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_low_bits(valid_lanes);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                rys_j_pspp_prepare_row_sidecar<BraPP>(
                    pspp_row_sidecar, *rys_jk_context_slot(), bd,
                    bra_bin, bra_pos, ket_group, rhs_refs, active_bits);
            }
            // Contract the six unique degree-two components and three lower
            // p components separately.  Ordered pp output is reconstructed by
            // HRR once below, instead of repeating both radial products and
            // the R*lower FMA for every primitive root.
            __m256d acc_hi[18];  // [opposite p][unique d]
            __m256d acc_lo[9];   // [opposite p][pp first p]
            for (auto& v : acc_hi) v = _mm256_setzero_pd();
            for (auto& v : acc_lo) v = _mm256_setzero_pd();
            const __m256d vRx = BraPP
                ? _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos)
                : _mm256_set1_pd(ket_bin.Rx[ket_pos]);
            const __m256d vRy = BraPP
                ? _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos)
                : _mm256_set1_pd(ket_bin.Ry[ket_pos]);
            const __m256d vRz = BraPP
                ? _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos)
                : _mm256_set1_pd(ket_bin.Rz[ket_pos]);
            const __m256d vR[3] = {vRx, vRy, vRz};

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPinv =
                    _mm256_loadu_pd(bra_bin.invs.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPAx = _mm256_loadu_pd(bra_bin.Lx.data() + boff);
                const __m256d vPAy = _mm256_loadu_pd(bra_bin.Ly.data() + boff);
                const __m256d vPAz = _mm256_loadu_pd(bra_bin.Lz.data() + boff);
                __m256d vbra_fast;
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if constexpr (PrimitiveFast) {
                        const __m256d vket_fast = _mm256_set1_pd(
                            ket_bin.primitive_fast_schwarz[koff]);
                        if (rys_primitive_fast_survival_mask(
                                vbra_fast, vket_fast,
                                shell_live) == 0u) {
                            continue;
                        }
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vqinv =
                        _mm256_set1_pd(ket_bin.invs[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)), vr2));
                    if constexpr (
                        rys_emit_uses_fast_sp_producer_v<Emit>) {
                        const std::size_t primitive_interactions =
                            static_cast<std::size_t>(
                                bra_bin.primitive_count) *
                            static_cast<std::size_t>(
                                ket_bin.primitive_count);
                        if (primitive_interactions >= 81u) {
                            rys_ppps_n2_batch4_lagrange(
                                targ4, roots8, weights8);
                        } else {
                            // Preserve the original inlined evaluator below
                            // the amortization boundary.  Outlining this cheap
                            // path regressed the one-primitive class even when
                            // the Lagrange table was never selected.
                            rys_roots_weights_batch4_for_emit<Emit, 2>(
                                targ4, roots8, weights8);
                        }
                    } else {
                        rys_roots_weights_batch4_for_emit<Emit, 2>(
                            targ4, roots8, weights8);
                    }
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    for (int root = 0; root < ActiveRoots<Emit, 2>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots8 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights8 + 4 * root));
                        const __m256d qop = _mm256_mul_pd(vq, vinvpq);
                        const __m256d pop = _mm256_mul_pd(vp, vinvpq);
                        const __m256d qt = _mm256_mul_pd(qop, vrt);
                        const __m256d pt = _mm256_mul_pd(pop, vrt);
                        const __m256d Cx = _mm256_fnmadd_pd(qt, vdx, vPAx);
                        const __m256d Cy = _mm256_fnmadd_pd(qt, vdy, vPAy);
                        const __m256d Cz = _mm256_fnmadd_pd(qt, vdz, vPAz);
                        const __m256d Dx = _mm256_fmadd_pd(
                            pt, vdx, _mm256_set1_pd(ket_bin.Lx[koff]));
                        const __m256d Dy = _mm256_fmadd_pd(
                            pt, vdy, _mm256_set1_pd(ket_bin.Ly[koff]));
                        const __m256d Dz = _mm256_fmadd_pd(
                            pt, vdz, _mm256_set1_pd(ket_bin.Lz[koff]));
                        const __m256d B00 =
                            _mm256_mul_pd(_mm256_mul_pd(vhalf, vinvpq), vrt);
                        const __m256d B10 = _mm256_mul_pd(
                            _mm256_mul_pd(vhalf, vPinv),
                            _mm256_fnmadd_pd(qop, vrt, vone));
                        const __m256d B01 = _mm256_mul_pd(
                            _mm256_mul_pd(vhalf, vqinv),
                            _mm256_fnmadd_pd(pop, vrt, vone));
                        __m256d Gx[6], Gy[6], Gz[6];
                        if constexpr (BraPP) {
                            rys_axis_table_pspp_v<true>(Cx, Dx, B00, B10, Gx);
                            rys_axis_table_pspp_v<true>(Cy, Dy, B00, B10, Gy);
                            rys_axis_table_pspp_v<true>(Cz, Dz, B00, B10, Gz);
                        } else {
                            rys_axis_table_pspp_v<false>(Cx, Dx, B00, B01, Gx);
                            rys_axis_table_pspp_v<false>(Cy, Dy, B00, B01, Gy);
                            rys_axis_table_pspp_v<false>(Cz, Dz, B00, B01, Gz);
                        }
                        for (int ps = 0; ps < 3; ++ps) {
                            for (int d = 0; d < 6; ++d) {
                                const __m256d full = BraPP
                                    ? rys_axis_product_pspp_v<true>(
                                          Gx, Gy, Gz, d_degree[d], p_degree[ps])
                                    : rys_axis_product_pspp_v<false>(
                                          Gx, Gy, Gz, p_degree[ps], d_degree[d]);
                                const int component = 6 * ps + d;
                                acc_hi[component] = _mm256_fmadd_pd(
                                    vww, full, acc_hi[component]);
                            }
                            for (int p = 0; p < 3; ++p) {
                                const __m256d lower = BraPP
                                    ? rys_axis_product_pspp_v<true>(
                                          Gx, Gy, Gz, p_degree[p], p_degree[ps])
                                    : rys_axis_product_pspp_v<false>(
                                          Gx, Gy, Gz, p_degree[ps], p_degree[p]);
                                const int component = 3 * ps + p;
                                acc_lo[component] = _mm256_fmadd_pd(
                                    vww, lower, acc_lo[component]);
                            }
                        }
                    }
                }
            }
            if constexpr (
                rys_emit_uses_preemit_pspp_positive_k_v<Emit>) {
#ifdef TLNO_RYS_DIRECT_EXCHANGE
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_low_bits(valid_lanes);
                if (active_bits != 0u) {
                    const std::uint32_t ket_group =
                        ket_bin.pair_group_ids[ket_pos];
                    rys_k_pspp_packet_v3<BraPP>(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, active_bits, vR, acc_lo, acc_hi);
                }
#endif
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_pspp_row_j_v<Emit>) {
                if constexpr (BraPP) {
                    __m256d fixed_delta0 = _mm256_setzero_pd();
                    __m256d fixed_delta1 = _mm256_setzero_pd();
                    __m256d fixed_delta2 = _mm256_setzero_pd();
#define TLNO_RYS_J_PSPP_MOVING_PP_ROW(PP) \
                    rys_j_pspp_digest_moving_pp_row<PP>( \
                        pspp_row_sidecar, vR, acc_lo, acc_hi, \
                        fixed_delta0, fixed_delta1, fixed_delta2)
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(0);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(1);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(2);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(3);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(4);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(5);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(6);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(7);
                    TLNO_RYS_J_PSPP_MOVING_PP_ROW(8);
#undef TLNO_RYS_J_PSPP_MOVING_PP_ROW
                    rys_j_pspp_finish_moving_pp(
                        pspp_row_sidecar,
                        fixed_delta0, fixed_delta1, fixed_delta2);
                } else {
                    __m256d moving_delta0 = _mm256_setzero_pd();
                    __m256d moving_delta1 = _mm256_setzero_pd();
                    __m256d moving_delta2 = _mm256_setzero_pd();
#define TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(PP) \
                    rys_j_pspp_digest_fixed_pp_column<PP>( \
                        pspp_row_sidecar, vR, acc_lo, acc_hi, \
                        moving_delta0, moving_delta1, moving_delta2)
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(0);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(1);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(2);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(3);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(4);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(5);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(6);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(7);
                    TLNO_RYS_J_PSPP_FIXED_PP_COLUMN(8);
#undef TLNO_RYS_J_PSPP_FIXED_PP_COLUMN
                    rys_j_pspp_finish_moving_ps(
                        pspp_row_sidecar,
                        moving_delta0, moving_delta1, moving_delta2);
                }
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_sp_j_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_low_bits(valid_lanes);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                if constexpr (BraPP) {
                    RysJSpPreemitPacket<9, 3> digest(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, rhs_refs, active_bits);
                    for (int pp = 0; pp < 9; ++pp) {
                        const int first_axis = pp / 3;
                        const int second_axis = pp - 3 * first_axis;
                        for (int ps = 0; ps < 3; ++ps) {
                            digest.consume(
                                pp, ps,
                                _mm256_fmadd_pd(
                                    vR[second_axis],
                                    acc_lo[3 * ps + first_axis],
                                    acc_hi[6 * ps + pp_unique[pp]]));
                        }
                    }
                    digest.finish();
                } else {
                    RysJSpPreemitPacket<3, 9> digest(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, rhs_refs, active_bits);
                    for (int pp = 0; pp < 9; ++pp) {
                        const int first_axis = pp / 3;
                        const int second_axis = pp - 3 * first_axis;
                        for (int ps = 0; ps < 3; ++ps) {
                            digest.consume(
                                ps, pp,
                                _mm256_fmadd_pd(
                                    vR[second_axis],
                                    acc_lo[3 * ps + first_axis],
                                    acc_hi[6 * ps + pp_unique[pp]]));
                        }
                    }
                    digest.finish();
                }
            } else {
                for (int pp = 0; pp < 9; ++pp) {
                    const int first_axis = pp / 3;
                    const int second_axis = pp - 3 * first_axis;
                    for (int ps = 0; ps < 3; ++ps) {
                        const __m256d value = _mm256_fmadd_pd(
                            vR[second_axis], acc_lo[3 * ps + first_axis],
                            acc_hi[6 * ps + pp_unique[pp]]);
                        const int component = BraPP
                            ? 3 * pp + ps : 9 * ps + pp;
                        _mm256_store_pd(raw[component], value);
                    }
                }
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((shell_live & (1u << lane)) == 0u) continue;
                    }
                    const std::uint32_t bra_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& lhs_refs = low_batch_pair_components(bd)[
                        static_cast<std::size_t>(bra_group)];
                    rys_emit_block<Emit>(
                        eri_values, lhs_refs, rhs_refs, false,
                        [&raw, lane](const PackedPairComponentH& lhs_ref,
                                     const PackedPairComponentH& rhs_ref) noexcept {
                            const std::size_t component = BraPP
                                ? 3u * static_cast<std::size_t>(
                                      lhs_ref.component_index) +
                                      static_cast<std::size_t>(
                                          rhs_ref.component_index)
                                : 9u * static_cast<std::size_t>(
                                      lhs_ref.component_index) +
                                      static_cast<std::size_t>(
                                          rhs_ref.component_index);
                            return raw[component][lane] *
                                   lhs_ref.norm * rhs_ref.norm;
                        });
                }
            }
        }
    }
}

inline void rys_axis_table_pppp_v(__m256d C, __m256d D,
                                  __m256d B00, __m256d B10, __m256d B01,
                                  __m256d G[3][3]) noexcept {
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    G[0][0] = one;
    G[0][1] = D;
    G[0][2] = _mm256_fmadd_pd(D, D, B01);
    G[1][0] = C;
    G[1][1] = _mm256_fmadd_pd(C, D, B00);
    G[1][2] = _mm256_fmadd_pd(
        D, G[1][1],
        _mm256_fmadd_pd(B00, D, _mm256_mul_pd(B01, C)));
    G[2][0] = _mm256_fmadd_pd(C, C, B10);
    G[2][1] = _mm256_fmadd_pd(
        D, G[2][0], _mm256_mul_pd(_mm256_mul_pd(two, B00), C));
    G[2][2] = _mm256_fmadd_pd(
        D, G[2][1],
        _mm256_fmadd_pd(_mm256_mul_pd(two, B00), G[1][1],
                        _mm256_mul_pd(B01, G[2][0])));
}

inline __m256d rys_axis_product_pppp_v(const __m256d Gx[3][3],
                                       const __m256d Gy[3][3],
                                       const __m256d Gz[3][3],
                                       const int bra_degree[3],
                                       const int ket_degree[3]) noexcept {
    return _mm256_mul_pd(
        _mm256_mul_pd(Gx[bra_degree[0]][ket_degree[0]],
                      Gy[bra_degree[1]][ket_degree[1]]),
        Gz[bra_degree[2]][ket_degree[2]]);
}

constexpr int kRysPpppRadialDegree[9][3] = {
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {2, 0, 0}, {1, 1, 0}, {1, 0, 1},
    {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};

// Four primitive interactions produce twelve three-root descriptors. Keeping
// the radial tables in a short L1-resident tape lets a nine-component row stay
// in the sixteen AVX2 registers while it consumes the whole block.  The old
// component-major loop otherwise performs a load/modify/store on each of 81
// accumulators for every root because all 81 cannot be register resident.
constexpr int kRysPpppDescriptorCapacity = 12;
struct alignas(32) RysPpppRootDescriptor {
    __m256d weight;
    __m256d axis[3][3][3];
};

__attribute__((noinline)) void rys_pppp_accumulate_descriptor_row(
    __m256d* acc_row,
    int bra,
    const RysPpppRootDescriptor* descriptors,
    int descriptor_count) noexcept {
    __m256d row[9];
#pragma GCC unroll 9
    for (int ket = 0; ket < 9; ++ket) row[ket] = acc_row[ket];
    const int bra_x = kRysPpppRadialDegree[bra][0];
    const int bra_y = kRysPpppRadialDegree[bra][1];
    const int bra_z = kRysPpppRadialDegree[bra][2];
    for (int descriptor = 0; descriptor < descriptor_count; ++descriptor) {
        const RysPpppRootDescriptor& radial = descriptors[descriptor];
        const __m256d wz0 = _mm256_mul_pd(
            radial.weight, radial.axis[2][bra_z][0]);
        const __m256d wz1 = _mm256_mul_pd(
            radial.weight, radial.axis[2][bra_z][1]);
        const __m256d wz2 = _mm256_mul_pd(
            radial.weight, radial.axis[2][bra_z][2]);
        __m256d xy = _mm256_mul_pd(
            radial.axis[0][bra_x][1], radial.axis[1][bra_y][0]);
        row[0] = _mm256_fmadd_pd(xy, wz0, row[0]);
        row[5] = _mm256_fmadd_pd(xy, wz1, row[5]);
        xy = _mm256_mul_pd(
            radial.axis[0][bra_x][0], radial.axis[1][bra_y][1]);
        row[1] = _mm256_fmadd_pd(xy, wz0, row[1]);
        row[7] = _mm256_fmadd_pd(xy, wz1, row[7]);
        xy = _mm256_mul_pd(
            radial.axis[0][bra_x][0], radial.axis[1][bra_y][0]);
        row[2] = _mm256_fmadd_pd(xy, wz1, row[2]);
        row[8] = _mm256_fmadd_pd(xy, wz2, row[8]);
        xy = _mm256_mul_pd(
            radial.axis[0][bra_x][2], radial.axis[1][bra_y][0]);
        row[3] = _mm256_fmadd_pd(xy, wz0, row[3]);
        xy = _mm256_mul_pd(
            radial.axis[0][bra_x][1], radial.axis[1][bra_y][1]);
        row[4] = _mm256_fmadd_pd(xy, wz0, row[4]);
        xy = _mm256_mul_pd(
            radial.axis[0][bra_x][0], radial.axis[1][bra_y][2]);
        row[6] = _mm256_fmadd_pd(xy, wz0, row[6]);
    }
#pragma GCC unroll 9
    for (int ket = 0; ket < 9; ++ket) acc_row[ket] = row[ket];
}

inline void rys_pppp_accumulate_descriptor_block(
    __m256d (&acc)[81],
    const RysPpppRootDescriptor* descriptors,
    int descriptor_count) noexcept {
    for (int bra = 0; bra < 9; ++bra) {
        rys_pppp_accumulate_descriptor_row(
            acc + 9 * bra, bra, descriptors, descriptor_count);
    }
}

// Cross-shellset (pp|pp).  The primitive contraction is performed in the
// compact pre-HRR basis {px,py,pz,dxx,dxy,dxz,dyy,dyz,dzz} on each side.
// Ordered 9x9 pp output is reconstructed once after contraction, avoiding the
// four-term two-sided HRR expansion at every root while sharing quadrature
// setup across four independent bra shell pairs.
template < bool DescriptorBlocked, bool Screened, bool PrimitiveFast,
          class Emit >
inline void accumulate_rys_pppp_shell_bin_pair_impl(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values) {
    static constexpr int pp_unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    alignas(32) double targ4[4];
    alignas(32) double roots12[4 * ActiveRoots<Emit, 3>];
    alignas(32) double weights12[4 * ActiveRoots<Emit, 3>];
    alignas(32) double raw[81][4];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);

    const std::size_t bra_pair_count =
        rys_sp_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_count =
        rys_sp_active_pair_count<Screened>(ket_bin);
    static_assert(
        !DescriptorBlocked ||
        (rys_emit_uses_fast_sp_producer_v<Emit> && !PrimitiveFast));
    for (std::size_t ket_pos = 0; ket_pos < ket_pair_count; ++ket_pos) {
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_count;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_count - bra_pos);
            if (same_bin && ket_pos >= bra_pos + valid_lanes) continue;
            unsigned shell_live = 0u;
            unsigned shell_q_live = 0u;
            if constexpr (Screened) {
                shell_live = rys_sp_shell_mask_bra4(
                    bra_bin, bra_pos, valid_lanes,
                    ket_bin, ket_pos, same_bin, shell_q_live);
                if (shell_live == 0u) {
                    const RysSpShellScreenContext* const screen =
                        g_rys_sp_shell_screen_context;
                    if (screen != nullptr &&
                        screen->bounds_sorted_descending &&
                        shell_q_live == 0u) {
                        // WHAT/WHY: triangular pppp also cuts only by Q.
                        break;
                    }
                    continue;
                }
            } else if constexpr (PrimitiveFast) {
                shell_live = rys_sp_shell_logical_bra4(
                    bra_pos, valid_lanes, ket_pos, same_bin);
            }

            RysJPpppRowSidecar pppp_row_sidecar;
            if constexpr (
                rys_emit_uses_pairspace_preemit_pppp_row_j_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_sp_shell_logical_bra4(
                          bra_pos, valid_lanes, ket_pos, same_bin);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                rys_j_pppp_prepare_row_sidecar(
                    pppp_row_sidecar, *rys_jk_context_slot(), bd,
                    bra_bin, bra_pos, ket_group, rhs_refs, active_bits);
            }
            __m256d acc[81];
            for (auto& v : acc) v = _mm256_setzero_pd();
            RysPpppRootDescriptor descriptors[kRysPpppDescriptorCapacity];
            int descriptor_count = 0;
            const __m256d vBraR[3] = {
                _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos)};
            const __m256d vKetR[3] = {
                _mm256_set1_pd(ket_bin.Rx[ket_pos]),
                _mm256_set1_pd(ket_bin.Ry[ket_pos]),
                _mm256_set1_pd(ket_bin.Rz[ket_pos])};

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPinv =
                    _mm256_loadu_pd(bra_bin.invs.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPAx = _mm256_loadu_pd(bra_bin.Lx.data() + boff);
                const __m256d vPAy = _mm256_loadu_pd(bra_bin.Ly.data() + boff);
                const __m256d vPAz = _mm256_loadu_pd(bra_bin.Lz.data() + boff);
                __m256d vbra_fast;
                if constexpr (PrimitiveFast) {
                    vbra_fast = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if constexpr (PrimitiveFast) {
                        const __m256d vket_fast = _mm256_set1_pd(
                            ket_bin.primitive_fast_schwarz[koff]);
                        if (rys_primitive_fast_survival_mask(
                                vbra_fast, vket_fast,
                                shell_live) == 0u) {
                            continue;
                        }
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vqinv = _mm256_set1_pd(ket_bin.invs[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)), vr2));
                    rys_roots_weights_batch4_for_emit<Emit, 3>(
                        targ4, roots12, weights12);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    for (int root = 0; root < ActiveRoots<Emit, 3>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots12 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights12 + 4 * root));
                        const __m256d qop = _mm256_mul_pd(vq, vinvpq);
                        const __m256d pop = _mm256_mul_pd(vp, vinvpq);
                        const __m256d qt = _mm256_mul_pd(qop, vrt);
                        const __m256d pt = _mm256_mul_pd(pop, vrt);
                        const __m256d Cx = _mm256_fnmadd_pd(qt, vdx, vPAx);
                        const __m256d Cy = _mm256_fnmadd_pd(qt, vdy, vPAy);
                        const __m256d Cz = _mm256_fnmadd_pd(qt, vdz, vPAz);
                        const __m256d Dx = _mm256_fmadd_pd(
                            pt, vdx, _mm256_set1_pd(ket_bin.Lx[koff]));
                        const __m256d Dy = _mm256_fmadd_pd(
                            pt, vdy, _mm256_set1_pd(ket_bin.Ly[koff]));
                        const __m256d Dz = _mm256_fmadd_pd(
                            pt, vdz, _mm256_set1_pd(ket_bin.Lz[koff]));
                        const __m256d B00 =
                            _mm256_mul_pd(_mm256_mul_pd(vhalf, vinvpq), vrt);
                        const __m256d B10 = _mm256_mul_pd(
                            _mm256_mul_pd(vhalf, vPinv),
                            _mm256_fnmadd_pd(qop, vrt, vone));
                        const __m256d B01 = _mm256_mul_pd(
                            _mm256_mul_pd(vhalf, vqinv),
                            _mm256_fnmadd_pd(pop, vrt, vone));
                        __m256d Gx[3][3], Gy[3][3], Gz[3][3];
                        rys_axis_table_pppp_v(Cx, Dx, B00, B10, B01, Gx);
                        rys_axis_table_pppp_v(Cy, Dy, B00, B10, B01, Gy);
                        rys_axis_table_pppp_v(Cz, Dz, B00, B10, B01, Gz);
                        if constexpr (DescriptorBlocked) {
                            RysPpppRootDescriptor& descriptor =
                                descriptors[descriptor_count++];
                            descriptor.weight = vww;
#pragma GCC unroll 3
                            for (int i = 0; i < 3; ++i) {
#pragma GCC unroll 3
                                for (int j = 0; j < 3; ++j) {
                                    descriptor.axis[0][i][j] = Gx[i][j];
                                    descriptor.axis[1][i][j] = Gy[i][j];
                                    descriptor.axis[2][i][j] = Gz[i][j];
                                }
                            }
                            if (descriptor_count ==
                                kRysPpppDescriptorCapacity) {
                                rys_pppp_accumulate_descriptor_block(
                                    acc, descriptors, descriptor_count);
                                descriptor_count = 0;
                            }
                        } else {
                            for (int bra = 0; bra < 9; ++bra) {
                                for (int ket = 0; ket < 9; ++ket) {
                                    const int component = 9 * bra + ket;
                                    const __m256d value =
                                        rys_axis_product_pppp_v(
                                            Gx, Gy, Gz,
                                            kRysPpppRadialDegree[bra],
                                            kRysPpppRadialDegree[ket]);
                                    acc[component] = _mm256_fmadd_pd(
                                        vww, value, acc[component]);
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (DescriptorBlocked) {
                if (descriptor_count != 0) {
                    rys_pppp_accumulate_descriptor_block(
                        acc, descriptors, descriptor_count);
                }
            }

            if constexpr (rys_emit_uses_preemit_pppp_positive_k_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_sp_shell_logical_bra4(bra_pos, valid_lanes, ket_pos, same_bin);
                if (active_bits) rys_k_pppp_preemit_packet(
                    *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                    ket_bin.pair_group_ids[ket_pos], active_bits, vBraR, vKetR, acc);
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_pppp_row_j_v<Emit>) {
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    !defined(TLNO_BUILD_RYS_VALIDATION) || \
    defined(TLNO_VALIDATE_RYS_SPHERICAL_PACKED)
                if constexpr (
                    rys_emit_uses_pairspace_preemit_pppp_tiled45_j_v<Emit>) {
                    rys_j_pppp_digest_tiled45(
                        pppp_row_sidecar, vBraR, vKetR, acc);
                } else {
                    rys_j_pppp_digest_rows(
                        pppp_row_sidecar, vBraR, vKetR, acc);
                }
#else
                rys_j_pppp_digest_rows(
                    pppp_row_sidecar, vBraR, vKetR, acc);
#endif
            } else if constexpr (
                rys_emit_uses_pairspace_preemit_sp_j_v<Emit>) {
                const unsigned active_bits = (Screened || PrimitiveFast)
                    ? shell_live
                    : rys_sp_shell_logical_bra4(
                          bra_pos, valid_lanes, ket_pos, same_bin);
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                RysJSpPreemitPacket<9, 9> digest(
                    *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                    ket_group, rhs_refs, active_bits);
                for (int ab = 0; ab < 9; ++ab) {
                    const int a = ab / 3;
                    const int b = ab - 3 * a;
                    const int bra_hi = 3 + pp_unique[ab];
                    for (int cd = 0; cd < 9; ++cd) {
                        const int c = cd / 3;
                        const int d = cd - 3 * c;
                        const int ket_hi = 3 + pp_unique[cd];
                        const __m256d hi = _mm256_fmadd_pd(
                            vBraR[b], acc[9 * a + ket_hi],
                            acc[9 * bra_hi + ket_hi]);
                        const __m256d lo = _mm256_fmadd_pd(
                            vBraR[b], acc[9 * a + c],
                            acc[9 * bra_hi + c]);
                        digest.consume(
                            ab, cd, _mm256_fmadd_pd(vKetR[d], lo, hi));
                    }
                }
                digest.finish();
            } else {
                for (int ab = 0; ab < 9; ++ab) {
                    const int a = ab / 3;
                    const int b = ab - 3 * a;
                    const int bra_hi = 3 + pp_unique[ab];
                    for (int cd = 0; cd < 9; ++cd) {
                        const int c = cd / 3;
                        const int d = cd - 3 * c;
                        const int ket_hi = 3 + pp_unique[cd];
                        const __m256d hi = _mm256_fmadd_pd(
                            vBraR[b], acc[9 * a + ket_hi],
                            acc[9 * bra_hi + ket_hi]);
                        const __m256d lo = _mm256_fmadd_pd(
                            vBraR[b], acc[9 * a + c],
                            acc[9 * bra_hi + c]);
                        const __m256d value =
                            _mm256_fmadd_pd(vKetR[d], lo, hi);
                        _mm256_store_pd(raw[9 * ab + cd], value);
                    }
                }
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& rhs_refs = low_batch_pair_components(bd)[
                    static_cast<std::size_t>(ket_group)];
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((shell_live & (1u << lane)) == 0u) continue;
                    } else if (same_bin && ket_pos > bra_pos + lane) {
                        continue;
                    }
                    const std::uint32_t bra_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& lhs_refs = low_batch_pair_components(bd)[
                        static_cast<std::size_t>(bra_group)];
                    const bool same_pair_group = bra_group == ket_group;
                    rys_emit_block<Emit>(
                        eri_values, lhs_refs, rhs_refs, same_pair_group,
                        [&raw, lane](const PackedPairComponentH& lhs_ref,
                                     const PackedPairComponentH& rhs_ref) noexcept {
                            const std::size_t component =
                                9u * static_cast<std::size_t>(
                                         lhs_ref.component_index) +
                                static_cast<std::size_t>(
                                    rhs_ref.component_index);
                            return raw[component][lane] *
                                   lhs_ref.norm * rhs_ref.norm;
                        });
                }
            }
        }
    }
}

template <bool Screened, bool PrimitiveFast, class Emit>
inline void accumulate_rys_pppp_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values) {
    if constexpr (rys_emit_uses_fast_sp_producer_v<Emit> &&
                  !PrimitiveFast) {
        const std::size_t primitive_interactions =
            static_cast<std::size_t>(bra_bin.primitive_count) *
            static_cast<std::size_t>(ket_bin.primitive_count);
        if (primitive_interactions >= 16u) {
            accumulate_rys_pppp_shell_bin_pair_impl <
                true, Screened, PrimitiveFast, Emit > (
                bd, bra_bin, ket_bin, same_bin, eri_values);
            return;
        }
    }
    accumulate_rys_pppp_shell_bin_pair_impl <
        false, Screened, PrimitiveFast, Emit > (
        bd, bra_bin, ket_bin, same_bin, eri_values);
}

template <class Emit>
inline void accumulate_rys_ssss_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& bins = bd.rys_low_pair_bins[0];
    for (std::size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].pair_count == 0) continue;
        for (std::size_t j = 1; j <= i; ++j) {
            if (bins[j].pair_count == 0) continue;
            if (i == j) {
                accumulate_rys_ssss_shell_bin_pair<false, false, Emit>(
                    bd, bins[i], bins[j], true, eri_values);
                continue;
            }
            // Put the larger shell-pair population in the SIMD dimension to
            // minimize partially occupied four-lane packs.
            const RysLowPairBinH* bra = &bins[i];
            const RysLowPairBinH* ket = &bins[j];
            if (ket->pair_count > bra->pair_count) std::swap(bra, ket);
            accumulate_rys_ssss_shell_bin_pair<false, false, Emit>(
                bd, *bra, *ket, false, eri_values);
        }
    }
}

template <class Emit>
inline void accumulate_rys_ssps_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& ss_bins = bd.rys_low_pair_bins[0];
    const auto& ps_bins = bd.rys_low_pair_bins[1];
    for (std::size_t si = 1; si < ss_bins.size(); ++si) {
        if (ss_bins[si].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < ps_bins.size(); ++pi) {
            if (ps_bins[pi].pair_count == 0) continue;
            if (ps_bins[pi].pair_count >= ss_bins[si].pair_count) {
                accumulate_rys_ssps_shell_bin_pair <
                    true, false, false, Emit > (
                    bd, ps_bins[pi], ss_bins[si], eri_values);
            } else {
                accumulate_rys_ssps_shell_bin_pair <
                    false, false, false, Emit > (
                    bd, ss_bins[si], ps_bins[pi], eri_values);
            }
        }
    }
}

template <class Emit>
inline void accumulate_rys_psps_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& bins = bd.rys_low_pair_bins[1];
    for (std::size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].pair_count == 0) continue;
        for (std::size_t j = 1; j <= i; ++j) {
            if (bins[j].pair_count == 0) continue;
            if (i == j) {
                accumulate_rys_psps_shell_bin_pair<false, false, Emit>(
                    bd, bins[i], bins[j], true, eri_values);
                continue;
            }
            const RysLowPairBinH* bra = &bins[i];
            const RysLowPairBinH* ket = &bins[j];
            if (ket->pair_count > bra->pair_count) std::swap(bra, ket);
            accumulate_rys_psps_shell_bin_pair<false, false, Emit>(
                bd, *bra, *ket, false, eri_values);
        }
    }
}

template <class Emit>
inline void accumulate_rys_sspp_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& ss_bins = bd.rys_low_pair_bins[0];
    const auto& pp_bins = bd.rys_low_pair_bins[2];
    for (std::size_t si = 1; si < ss_bins.size(); ++si) {
        if (ss_bins[si].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < pp_bins.size(); ++pi) {
            if (pp_bins[pi].pair_count == 0) continue;
            if (pp_bins[pi].pair_count >= ss_bins[si].pair_count) {
                accumulate_rys_sspp_shell_bin_pair<true, false, false, Emit>(
                    bd, pp_bins[pi], ss_bins[si], eri_values);
            } else {
                accumulate_rys_sspp_shell_bin_pair<false, false, false, Emit>(
                    bd, ss_bins[si], pp_bins[pi], eri_values);
            }
        }
    }
}

template <class Emit>
inline void accumulate_rys_pspp_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& ps_bins = bd.rys_low_pair_bins[1];
    const auto& pp_bins = bd.rys_low_pair_bins[2];
    for (std::size_t si = 1; si < ps_bins.size(); ++si) {
        if (ps_bins[si].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < pp_bins.size(); ++pi) {
            if (pp_bins[pi].pair_count == 0) continue;
            if (pp_bins[pi].pair_count >= ps_bins[si].pair_count) {
                accumulate_rys_pspp_shell_bin_pair<true, false, false, Emit>(
                    bd, pp_bins[pi], ps_bins[si], eri_values);
            } else {
                accumulate_rys_pspp_shell_bin_pair<false, false, false, Emit>(
                    bd, ps_bins[si], pp_bins[pi], eri_values);
            }
        }
    }
}

template <class Emit>
inline void accumulate_rys_pppp_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& bins = bd.rys_low_pair_bins[2];
    for (std::size_t i = 1; i < bins.size(); ++i) {
        if (bins[i].pair_count == 0) continue;
        for (std::size_t j = 1; j <= i; ++j) {
            if (bins[j].pair_count == 0) continue;
            if (i == j) {
                accumulate_rys_pppp_shell_bin_pair<false, false, Emit>(
                    bd, bins[i], bins[j], true, eri_values);
                continue;
            }
            const RysLowPairBinH* bra = &bins[i];
            const RysLowPairBinH* ket = &bins[j];
            if (ket->pair_count > bra->pair_count) std::swap(bra, ket);
            accumulate_rys_pppp_shell_bin_pair<false, false, Emit>(
                bd, *bra, *ket, false, eri_values);
        }
    }
}

enum class RysSpShellBatchKind : std::uint8_t {
    Ssss,
    SspsBraP,
    SspsKetP,
    Psps,
    SsppBraPP,
    SsppKetPP,
    PsppBraPP,
    PsppKetPP,
    Pppp
};

inline int rys_primitive_fast_class_index(
    RysSpShellBatchKind kind) noexcept {
    switch (kind) {
        case RysSpShellBatchKind::Ssss:
            return 0;
        case RysSpShellBatchKind::SspsBraP:
        case RysSpShellBatchKind::SspsKetP:
            return 1;
        case RysSpShellBatchKind::Psps:
            return 2;
        case RysSpShellBatchKind::SsppBraPP:
        case RysSpShellBatchKind::SsppKetPP:
            return 3;
        case RysSpShellBatchKind::PsppBraPP:
        case RysSpShellBatchKind::PsppKetPP:
            return 4;
        case RysSpShellBatchKind::Pppp:
            return 5;
    }
    return -1;
}

struct RysSpShellBatchTask {
    RysSpShellBatchKind kind = RysSpShellBatchKind::Ssss;
    const RysLowPairBinH* bra = nullptr;
    const RysLowPairBinH* ket = nullptr;
    bool same_bin = false;
    // WHAT: zero means that Q itself needs no lane mask; density_mask can still
    // select the masked kernel independently. WHY/PURPOSE: a task in which
    // only P is Partial must not repeat the Q comparison already proved.
    // Metric-specific caches guarantee that active Q values belong to one
    // family only.
    double shell_screen_threshold = 0.0;
    // WHAT: true only for a task whose Q-live quartets mix density-live and
    // density-dead lanes. WHY: WholeLive uses the original Q mask and
    // WholeDead never reaches the kernel; neither may pay local P accesses.
    bool density_mask = false;
    // Already squared once at the pure-spherical API boundary, matching
    // simint_compute_ostei's screen_tol2 contract.
    double primitive_fastschwarz_threshold_squared = 0.0;
    // Only Partial tasks override these logical ends.  Whole-bin and
    // primitive-only tasks retain the complete immutable bin.
    // Pair-group ids are already uint32_t; matching that width keeps the hot
    // task record compact. UINT32_MAX is the unambiguous whole-bin sentinel.
    std::uint32_t bra_pair_count_override =
        std::numeric_limits<std::uint32_t>::max();
    std::uint32_t ket_pair_count_override =
        std::numeric_limits<std::uint32_t>::max();
};

#ifdef TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST
class RysJVeloxTaskScope {
public:
    RysJVeloxTaskScope(bool enabled,
                       const BasisDataHGP& bd,
                       const RysSpShellBatchTask& task) noexcept
        : state_(enabled ? rys_j_velox_digest_slot() : nullptr) {
        if (state_ == nullptr) return;
        state_->begin();
        const auto register_bin = [&](
            const RysLowPairBinH* bin,
            std::uint32_t override_count) noexcept -> bool {
            const std::size_t count = override_count ==
                    std::numeric_limits<std::uint32_t>::max()
                ? bin->pair_count
                : static_cast<std::size_t>(override_count);
            const auto& components = low_batch_pair_components(bd);
            for (std::size_t i = 0; i < count; ++i) {
                const std::uint32_t group = bin->pair_group_ids[i];
                if (group >= components.size()) return false;
                for (const PackedPairComponentH& ref : components[group]) {
                    if (!state_->register_pair(ref.pair_index)) return false;
                }
            }
            return true;
        };
        const bool complete =
            register_bin(task.bra, task.bra_pair_count_override) &&
            register_bin(task.ket, task.ket_pair_count_override);
        if (complete) state_->activate();
    }

    ~RysJVeloxTaskScope() noexcept {
        if (state_ != nullptr) state_->finish();
    }
    RysJVeloxTaskScope(const RysJVeloxTaskScope&) = delete;
    RysJVeloxTaskScope& operator = (const RysJVeloxTaskScope&) = delete;

private:
    RysJVeloxTaskDigest* state_ = nullptr;
};
#endif

inline void append_rys_sp_shell_batch_tasks(
    const BasisDataHGP& bd,
    std::vector<RysSpShellBatchTask>& tasks) {
    const auto& ss_bins = bd.rys_low_pair_bins[0];
    const auto& ps_bins = bd.rys_low_pair_bins[1];
    const auto& pp_bins = bd.rys_low_pair_bins[2];

    for (std::size_t i = 1; i < ss_bins.size(); ++i) {
        if (ss_bins[i].pair_count == 0) continue;
        for (std::size_t j = 1; j <= i; ++j) {
            if (ss_bins[j].pair_count == 0) continue;
            const RysLowPairBinH* bra = &ss_bins[i];
            const RysLowPairBinH* ket = &ss_bins[j];
            if (i != j && ket->pair_count > bra->pair_count) {
                std::swap(bra, ket);
            }
            tasks.push_back(
                {RysSpShellBatchKind::Ssss, bra, ket, i == j});
        }
    }

    for (std::size_t si = 1; si < ss_bins.size(); ++si) {
        if (ss_bins[si].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < ps_bins.size(); ++pi) {
            if (ps_bins[pi].pair_count == 0) continue;
            if (ps_bins[pi].pair_count >= ss_bins[si].pair_count) {
                tasks.push_back(
                    {RysSpShellBatchKind::SspsBraP,
                     &ps_bins[pi], &ss_bins[si], false});
            } else {
                tasks.push_back(
                    {RysSpShellBatchKind::SspsKetP,
                     &ss_bins[si], &ps_bins[pi], false});
            }
        }
    }

    for (std::size_t i = 1; i < ps_bins.size(); ++i) {
        if (ps_bins[i].pair_count == 0) continue;
        for (std::size_t j = 1; j <= i; ++j) {
            if (ps_bins[j].pair_count == 0) continue;
            const RysLowPairBinH* bra = &ps_bins[i];
            const RysLowPairBinH* ket = &ps_bins[j];
            if (i != j && ket->pair_count > bra->pair_count) {
                std::swap(bra, ket);
            }
            tasks.push_back(
                {RysSpShellBatchKind::Psps, bra, ket, i == j});
        }
    }

    for (std::size_t si = 1; si < ss_bins.size(); ++si) {
        if (ss_bins[si].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < pp_bins.size(); ++pi) {
            if (pp_bins[pi].pair_count == 0) continue;
            if (pp_bins[pi].pair_count >= ss_bins[si].pair_count) {
                tasks.push_back(
                    {RysSpShellBatchKind::SsppBraPP,
                     &pp_bins[pi], &ss_bins[si], false});
            } else {
                tasks.push_back(
                    {RysSpShellBatchKind::SsppKetPP,
                     &ss_bins[si], &pp_bins[pi], false});
            }
        }
    }

    for (std::size_t si = 1; si < ps_bins.size(); ++si) {
        if (ps_bins[si].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < pp_bins.size(); ++pi) {
            if (pp_bins[pi].pair_count == 0) continue;
            if (pp_bins[pi].pair_count >= ps_bins[si].pair_count) {
                tasks.push_back(
                    {RysSpShellBatchKind::PsppBraPP,
                     &pp_bins[pi], &ps_bins[si], false});
            } else {
                tasks.push_back(
                    {RysSpShellBatchKind::PsppKetPP,
                     &ps_bins[si], &pp_bins[pi], false});
            }
        }
    }

    for (std::size_t i = 1; i < pp_bins.size(); ++i) {
        if (pp_bins[i].pair_count == 0) continue;
        for (std::size_t j = 1; j <= i; ++j) {
            if (pp_bins[j].pair_count == 0) continue;
            const RysLowPairBinH* bra = &pp_bins[i];
            const RysLowPairBinH* ket = &pp_bins[j];
            if (i != j && ket->pair_count > bra->pair_count) {
                std::swap(bra, ket);
            }
            tasks.push_back(
                {RysSpShellBatchKind::Pppp, bra, ket, i == j});
        }
    }
}

enum class RysSpBinScreenResult : std::uint8_t {
    Rejected,
    AcceptedWhole,
    Partial
};

enum class RysDensityTaskClass : std::uint8_t {
    WholeDead,
    WholeLive,
    Partial
};

struct RysSpDensityBinBounds {
    double rho_min = 0.0;
    double rho_max = 0.0;
    double exchange_upper_max = 0.0;
};

// WHAT: the single predicate for "a usable per-bin density summary exists".
// WHY: make_rys_sp_density_bin_bounds gives up if EITHER of the two pointers
// is missing and returns an EMPTY vector, while the caller decided whether to
// index it by looking only at shell_rho.  With two different predicates, the
// day an installer sets shell_rho but not pair_group_density_upper, the call
// site evaluates &density_bounds[i] on an empty vector: an out-of-range read,
// silent and allocator-dependent.  Today both pointers are installed together
// (the two assignment sites write them in the same block), so it is not
// reachable; the predicates are unified so that it cannot become reachable
// without anyone noticing.
inline bool rys_sp_density_bin_bounds_available(
    const RysBatchScreen* screen) noexcept {
    return screen != nullptr && screen->shell_rho != nullptr &&
           (screen->j_only ||
            screen->pair_group_density_upper != nullptr);
}

inline std::vector<RysSpDensityBinBounds>
make_rys_sp_density_bin_bounds(
    const std::vector<RysLowPairBinH>& bins,
    const std::vector<std::uint32_t>& live_counts) {
    const RysBatchScreen* const screen = g_rys_batch_screen;
    if (!rys_sp_density_bin_bounds_available(screen)) {
        // WHAT: reserve no summaries unless this is a J/K build with density.
        // WHY: Schwarz and FASTSCHWARZ alone must not acquire allocations
        // from an optimization that exists only for P.
        return {};
    }
    std::vector<RysSpDensityBinBounds> bounds(bins.size());

    // WHAT: summarize each live s/p bin once with the min/max of rho and the
    // per-pair-group exchange upper bound.
    // WHY: the scheduler reuses the summary for all bin pairs and avoids the
    // O(n_l*n_r) prepass that was duplicating the hot kernel's mask.
    for (std::size_t index = 0; index < bins.size(); ++index) {
        const RysLowPairBinH& bin = bins[index];
        const std::size_t count = live_counts.empty()
            ? bin.pair_count
            : std::min<std::size_t>(live_counts[index], bin.pair_count);
        if (count == 0u) continue;
        double rho_min = std::numeric_limits<double>::infinity();
        double rho_max = 0.0;
        double exchange_upper_max = 0.0;
        for (std::size_t lane = 0; lane < count; ++lane) {
            const std::uint32_t pair_group = bin.pair_group_ids[lane];
            const double rho = screen->shell_rho[pair_group];
            rho_min = std::min(rho_min, rho);
            rho_max = std::max(rho_max, rho);
            if (!screen->j_only) {
                exchange_upper_max = std::max(
                    exchange_upper_max,
                    screen->pair_group_density_upper[pair_group]);
            }
        }
        bounds[index] = {rho_min, rho_max, exchange_upper_max};
    }
    return bounds;
}

inline RysDensityTaskClass classify_rys_sp_density_task(
    const RysLowPairBinH& lhs,
    const RysLowPairBinH& rhs,
    const RysSpDensityBinBounds* lhs_density,
    const RysSpDensityBinBounds* rhs_density,
    RysSpBinScreenResult integral_class) noexcept {
    const RysBatchScreen* const screen = g_rys_batch_screen;
    if (screen == nullptr || screen->shell_rho == nullptr) {
        return RysDensityTaskClass::WholeLive;
    }
    if (lhs_density == nullptr || rhs_density == nullptr) {
        // WHAT/WHY: if the dynamic summary is missing, keep the work through
        // Partial; never turn incomplete metadata into an unsafe rejection.
        return RysDensityTaskClass::Partial;
    }

    // WHAT: classify with conservative constant-cost bounds.  WholeDead uses
    // upper bounds for Q, Coulomb and exchange; WholeLive uses as its floor
    // the exact minimum Q if the whole rectangle is live, or q_break if the
    // task is Q-partial, always over the quartets the scheduler can emit.
    // WHY: the previous exact classifier swept the whole rectangle for
    // WholeLive/WholeDead and swept Partial tasks again inside the mask; that
    // double work cancelled the roots saved in 6-31G and 6-31G(d,p).
    const double q_max = lhs.shell_screen_bound_max *
                         rhs.shell_screen_bound_max;
    const double coulomb_upper = 4.0 * std::max(
        lhs_density->rho_max, rhs_density->rho_max);
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    if (screen->k_only) {
        const double exchange_upper = std::min(
            lhs_density->exchange_upper_max,
            rhs_density->exchange_upper_max);
        if (q_max * exchange_upper < screen->density_break) {
            return RysDensityTaskClass::WholeDead;
        }
        return RysDensityTaskClass::Partial;
    }
#endif
    if (screen->j_only) {
        // WHAT: classify J-only tasks from their Coulomb envelope alone.
        // WHY: exchange_upper belongs to K and can only turn J-dead work into a
        // false Partial/WholeLive task; it can never make a J rejection safer.
        // PURPOSE: WholeDead and Partial decisions match the two terms that the
        // speed probe actually contracts, before any roots are evaluated.
        if (q_max * coulomb_upper < screen->density_break) {
            return RysDensityTaskClass::WholeDead;
        }
        const double q_lower =
            integral_class == RysSpBinScreenResult::AcceptedWhole
                ? lhs.shell_screen_bound_min * rhs.shell_screen_bound_min
                : screen->q_break;
        const double coulomb_lower = 4.0 * std::max(
            lhs_density->rho_min, rhs_density->rho_min);
        return q_lower * coulomb_lower >= screen->density_break
            ? RysDensityTaskClass::WholeLive
            : RysDensityTaskClass::Partial;
    }
    const double exchange_upper = std::min(
        lhs_density->exchange_upper_max,
        rhs_density->exchange_upper_max);
    if (q_max * std::max(coulomb_upper, exchange_upper) <
        screen->density_break) {
        return RysDensityTaskClass::WholeDead;
    }

    // WHAT: for a Q-partial task, use q_break as the lower bound of all its
    // Q-live quartets; AcceptedWhole can use the tighter exact minimum.
    // WHY: this turns off P loads when density cannot reject anything,
    // without turning off the Q mask that the partial task still needs.
    const double q_lower =
        integral_class == RysSpBinScreenResult::AcceptedWhole
            ? lhs.shell_screen_bound_min * rhs.shell_screen_bound_min
            : screen->q_break;
    const double coulomb_lower = 4.0 * std::max(
        lhs_density->rho_min, rhs_density->rho_min);
    if (q_lower * coulomb_lower >= screen->density_break) {
        return RysDensityTaskClass::WholeLive;
    }
    return RysDensityTaskClass::Partial;
}

inline RysSpBinScreenResult classify_rys_sp_bin_pair(
    const RysLowPairBinH& lhs,
    const RysLowPairBinH& rhs,
    bool triangular,
    double threshold) {
    RysSphericalScreeningDiagnostics* const diag =
        g_rys_spherical_screening_diagnostics;
    const std::uint64_t work =
        rys_spherical_bin_pair_work(lhs, rhs, triangular);
    if (diag != nullptr) {
        diag->scheduled_pair_group_quartets += work;
    }

    if (lhs.shell_screen_bound_max * rhs.shell_screen_bound_max <
        threshold) {
        if (diag != nullptr) {
            ++diag->whole_bin_pairs_rejected;
            diag->rejected_pair_group_quartets += work;
        }
        return RysSpBinScreenResult::Rejected;
    }

    if (lhs.shell_screen_bound_min * rhs.shell_screen_bound_min >=
        threshold) {
        if (diag != nullptr) {
            ++diag->whole_bin_pairs_accepted;
            diag->accepted_pair_group_quartets += work;
        }
        return RysSpBinScreenResult::AcceptedWhole;
    }

    if (diag != nullptr) {
        ++diag->partial_bin_pairs;
        std::uint64_t accepted = 0;
        for (std::size_t li = 0; li < lhs.pair_count; ++li) {
            const std::size_t rhs_end =
                triangular ? li + 1u : rhs.pair_count;
            const double lhs_bound = lhs.shell_screen_bound[li];
            for (std::size_t ri = 0; ri < rhs_end; ++ri) {
                if (lhs_bound * rhs.shell_screen_bound[ri] >=
                    threshold) {
                    ++accepted;
                }
            }
        }
        diag->accepted_pair_group_quartets += accepted;
        diag->rejected_pair_group_quartets += work - accepted;
    }
    return RysSpBinScreenResult::Partial;
}

// q_floor s/p: one global partner maximum turns each metric-sorted full bin
// into an immutable live prefix plus a suffix that cannot survive any s/p task.
inline std::uint32_t rys_sp_global_floor_live_pair_count(
    const RysLowPairBinH& bin,
    double sp_pair_group_metric_max,
    double threshold) {
    if (threshold <= 0.0) {
        return static_cast<std::uint32_t>(bin.pair_count);
    }
    if (bin.pair_count == 0 || !(sp_pair_group_metric_max > 0.0)) return 0;

    // Search only real lanes: [pair_count,stride) is finite SIMD padding, not
    // part of the logical prefix.  Testing the product directly preserves the
    // kernel's >= boundary and avoids a rounded threshold/Qmax division that
    // could discard a borderline survivor.
    const auto first = bin.shell_screen_bound.begin();
    const auto last = first +
        static_cast<std::ptrdiff_t>(bin.pair_count);
    const auto first_dead = std::partition_point(
        first, last,
        [sp_pair_group_metric_max, threshold](double metric) noexcept {
            return metric * sp_pair_group_metric_max >= threshold;
        });
    return static_cast<std::uint32_t>(first_dead - first);
}

// Shell-pair screening for the six logical s/p classes. It evaluates the
// active metric predicate exactly at pair-group lane granularity while
// preserving each original contraction-length bin: whole bin pairs use the
// unmasked kernel, rejected pairs create no task, and only partial pairs carry
// a positive threshold into the in-kernel four-lane mask.
inline void append_rys_sp_shell_batch_tasks_in_bin_screened(
    const BasisDataHGP& bd,
    const RysScreenedSpFullBins& sorted_full_bins,
    double threshold,
    std::vector<RysSpShellBatchTask>& tasks) {
    const auto& ss_bins = bd.rys_low_pair_bins[0];
    const auto& ps_bins = bd.rys_low_pair_bins[1];
    const auto& pp_bins = bd.rys_low_pair_bins[2];
    const auto& sorted_ss_bins = sorted_full_bins.bins[0];
    const auto& sorted_ps_bins = sorted_full_bins.bins[1];
    const auto& sorted_pp_bins = sorted_full_bins.bins[2];

    // Threshold-dependent state stays local to this build.  The cached AoSoA
    // bins remain immutable, while each bin pays one logarithmic prefix search
    // regardless of how many Partial tasks reuse it. Primitive-only screening
    // has threshold==0 and deliberately allocates/traverses none of this state.
    const auto make_live_pair_counts =
        [&](const std::vector<RysLowPairBinH>& bins) {
            std::vector<std::uint32_t> counts(bins.size(), 0u);
            for (std::size_t index = 0; index < bins.size(); ++index) {
                counts[index] = rys_sp_global_floor_live_pair_count(
                    bins[index], sorted_full_bins.sp_pair_group_metric_max,
                    threshold);
            }
            return counts;
        };
    std::vector<std::uint32_t> live_ss_counts;
    std::vector<std::uint32_t> live_ps_counts;
    std::vector<std::uint32_t> live_pp_counts;
    if (threshold > 0.0) {
        live_ss_counts = make_live_pair_counts(sorted_ss_bins);
        live_ps_counts = make_live_pair_counts(sorted_ps_bins);
        live_pp_counts = make_live_pair_counts(sorted_pp_bins);
    }
    std::vector<RysSpDensityBinBounds> density_ss_bounds;
    std::vector<RysSpDensityBinBounds> density_ps_bounds;
    std::vector<RysSpDensityBinBounds> density_pp_bounds;
    const RysBatchScreen* const batch_screen = g_rys_batch_screen;
    // The same predicate make_rys_sp_density_bin_bounds uses to decide whether
    // it returns tables: if the two diverge, this `density_active` indexes an
    // empty vector a few lines below.
    const bool density_active =
        rys_sp_density_bin_bounds_available(batch_screen);
    if (density_active) {
        // WHAT: build three O(number of shell pairs) tables, only when density
        // screening is on. WHY: Schwarz/FASTSCHWARZ without P keeps its
        // previous allocations and scheduler.
        density_ss_bounds = make_rys_sp_density_bin_bounds(
            sorted_ss_bins, live_ss_counts);
        density_ps_bounds = make_rys_sp_density_bin_bounds(
            sorted_ps_bins, live_ps_counts);
        density_pp_bounds = make_rys_sp_density_bin_bounds(
            sorted_pp_bins, live_pp_counts);
    }
    // Whole-bin classification and its accepted/rejected quartet diagnostics
    // still cover the original bins.  Floor-pruned suffixes never enter a SIMD
    // mask, so the sp_masked_* counters intentionally describe only the live
    // prefix visited by Partial tasks.

    const auto append_same =
        [&](const std::vector<RysLowPairBinH>& original_bins,
            const std::vector<RysLowPairBinH>& sorted_bins,
            const std::vector<std::uint32_t>& live_counts,
            const std::vector<RysSpDensityBinBounds>& density_bounds,
            RysSpShellBatchKind kind) {
            for (std::size_t i = 1; i < sorted_bins.size(); ++i) {
                if (sorted_bins[i].pair_count == 0) continue;
                for (std::size_t j = 1; j <= i; ++j) {
                    if (sorted_bins[j].pair_count == 0) continue;
                    const bool same_bin = i == j;
                    const RysSpBinScreenResult result =
                        classify_rys_sp_bin_pair(
                            sorted_bins[i], sorted_bins[j],
                            same_bin, threshold);
                    if (result == RysSpBinScreenResult::Rejected) {
                        continue;
                    }
                    const RysDensityTaskClass density_class =
                        classify_rys_sp_density_task(
                            sorted_bins[i], sorted_bins[j],
                            density_active ? &density_bounds[i] : nullptr,
                            density_active ? &density_bounds[j] : nullptr,
                            result);
                    if (density_class == RysDensityTaskClass::WholeDead) {
                        // WHAT: a task entirely dead by density is not created.
                        // WHY: this rejects it before Boys/roots.
                        continue;
                    }
                    const bool density_mask =
                        density_class == RysDensityTaskClass::Partial;
                    const bool q_mask =
                        result == RysSpBinScreenResult::Partial;
#ifdef TLNO_JK_FORCE_COMBINED_Q_DENSITY_MASK
                    // WHAT: the A/B control recompiles the previous behavior.
                    // WHY/PURPOSE: it measures the value of separating Q/P
                    // with two fresh builds of the same live source.
                    const bool scheduled_q_mask = q_mask || density_mask;
#else
                    const bool scheduled_q_mask = q_mask;
#endif
                    // WHAT: route uniform tasks before dispatch: the original
                    // for Q+density WholeLive, none for WholeDead and a local
                    // mask only for a mixture. WHY: forcing all density to
                    // partial paid the overhead even when P rejected nothing.
                    const bool partial =
                        result == RysSpBinScreenResult::Partial || density_mask;
                    const auto& selected =
                        partial ? sorted_bins : original_bins;
                    const RysLowPairBinH* bra = &selected[i];
                    const RysLowPairBinH* ket = &selected[j];
                    std::size_t bra_pair_count =
                        partial ? live_counts[i] : bra->pair_count;
                    std::size_t ket_pair_count =
                        partial ? live_counts[j] : ket->pair_count;
                    if (!same_bin &&
                        ket_pair_count > bra_pair_count) {
                        std::swap(bra, ket);
                        std::swap(bra_pair_count, ket_pair_count);
                    }
                    RysSpShellBatchTask task{
                        kind, bra, ket, same_bin};
                    task.density_mask = density_mask;
                    if (partial) {
                        task.shell_screen_threshold =
                            scheduled_q_mask ? threshold : 0.0;
                        task.bra_pair_count_override =
                            static_cast<std::uint32_t>(bra_pair_count);
                        task.ket_pair_count_override =
                            static_cast<std::uint32_t>(ket_pair_count);
                    }
                    tasks.push_back(task);
                }
            }
        };

    const auto append_cross =
        [&](const std::vector<RysLowPairBinH>& original_low_bins,
            const std::vector<RysLowPairBinH>& original_high_bins,
            const std::vector<RysLowPairBinH>& sorted_low_bins,
            const std::vector<RysLowPairBinH>& sorted_high_bins,
            const std::vector<std::uint32_t>& live_low_counts,
            const std::vector<std::uint32_t>& live_high_counts,
            const std::vector<RysSpDensityBinBounds>& density_low_bounds,
            const std::vector<RysSpDensityBinBounds>& density_high_bounds,
            RysSpShellBatchKind high_is_bra,
            RysSpShellBatchKind high_is_ket) {
            for (std::size_t li = 1;
                 li < sorted_low_bins.size(); ++li) {
                if (sorted_low_bins[li].pair_count == 0) continue;
                for (std::size_t hi = 1;
                     hi < sorted_high_bins.size(); ++hi) {
                    if (sorted_high_bins[hi].pair_count == 0) continue;
                    const RysSpBinScreenResult result =
                        classify_rys_sp_bin_pair(
                            sorted_low_bins[li],
                            sorted_high_bins[hi],
                            false, threshold);
                    if (result == RysSpBinScreenResult::Rejected) {
                        continue;
                    }
                    const RysDensityTaskClass density_class =
                        classify_rys_sp_density_task(
                            sorted_low_bins[li], sorted_high_bins[hi],
                            density_active ? &density_low_bounds[li] : nullptr,
                            density_active ? &density_high_bounds[hi] : nullptr,
                            result);
                    if (density_class == RysDensityTaskClass::WholeDead) {
                        // WHAT/WHY: a cross task dead by P is removed in the
                        // scheduler so that it never reaches a roots kernel.
                        continue;
                    }
                    const bool density_mask =
                        density_class == RysDensityTaskClass::Partial;
                    const bool q_mask =
                        result == RysSpBinScreenResult::Partial;
#ifdef TLNO_JK_FORCE_COMBINED_Q_DENSITY_MASK
                    // WHAT/WHY/PURPOSE: the same exact control for cross-class
                    // tasks; A keeps Q active when only P is mixed.
                    const bool scheduled_q_mask = q_mask || density_mask;
#else
                    const bool scheduled_q_mask = q_mask;
#endif
                    // WHAT/WHY: as for same-class, only a real density mixture
                    // needs local P loads in the mask.
                    const bool partial =
                        result == RysSpBinScreenResult::Partial || density_mask;
                    const auto& selected_low =
                        partial ? sorted_low_bins
                                : original_low_bins;
                    const auto& selected_high =
                        partial ? sorted_high_bins
                                : original_high_bins;
                    const double task_threshold =
                        scheduled_q_mask ? threshold : 0.0;
                    const std::size_t low_pair_count = partial
                        ? live_low_counts[li]
                        : selected_low[li].pair_count;
                    const std::size_t high_pair_count = partial
                        ? live_high_counts[hi]
                        : selected_high[hi].pair_count;
                    if (high_pair_count >= low_pair_count) {
                        RysSpShellBatchTask task{
                            high_is_bra, &selected_high[hi],
                            &selected_low[li], false};
                        task.shell_screen_threshold = task_threshold;
                        task.density_mask = density_mask;
                        if (partial) {
                            task.bra_pair_count_override =
                                static_cast<std::uint32_t>(high_pair_count);
                            task.ket_pair_count_override =
                                static_cast<std::uint32_t>(low_pair_count);
                        }
                        tasks.push_back(task);
                    } else {
                        RysSpShellBatchTask task{
                            high_is_ket, &selected_low[li],
                            &selected_high[hi], false};
                        task.shell_screen_threshold = task_threshold;
                        task.density_mask = density_mask;
                        if (partial) {
                            task.bra_pair_count_override =
                                static_cast<std::uint32_t>(low_pair_count);
                            task.ket_pair_count_override =
                                static_cast<std::uint32_t>(high_pair_count);
                        }
                        tasks.push_back(task);
                    }
                }
            }
        };

    append_same(
        ss_bins, sorted_ss_bins, live_ss_counts, density_ss_bounds,
        RysSpShellBatchKind::Ssss);
    append_cross(
        ss_bins, ps_bins, sorted_ss_bins, sorted_ps_bins,
        live_ss_counts, live_ps_counts,
        density_ss_bounds, density_ps_bounds,
        RysSpShellBatchKind::SspsBraP,
        RysSpShellBatchKind::SspsKetP);
    append_same(
        ps_bins, sorted_ps_bins, live_ps_counts, density_ps_bounds,
        RysSpShellBatchKind::Psps);
    append_cross(
        ss_bins, pp_bins, sorted_ss_bins, sorted_pp_bins,
        live_ss_counts, live_pp_counts,
        density_ss_bounds, density_pp_bounds,
        RysSpShellBatchKind::SsppBraPP,
        RysSpShellBatchKind::SsppKetPP);
    append_cross(
        ps_bins, pp_bins, sorted_ps_bins, sorted_pp_bins,
        live_ps_counts, live_pp_counts,
        density_ps_bounds, density_pp_bounds,
        RysSpShellBatchKind::PsppBraPP,
        RysSpShellBatchKind::PsppKetPP);
    append_same(
        pp_bins, sorted_pp_bins, live_pp_counts, density_pp_bounds,
        RysSpShellBatchKind::Pppp);
}

template <bool Screened, bool PrimitiveFast, class Emit>
inline void run_rys_sp_shell_batch_task_impl(
    const BasisDataHGP& bd,
    const RysSpShellBatchTask& task,
    std::vector<double>& eri_values) {
    switch (task.kind) {
        case RysSpShellBatchKind::Ssss:
            accumulate_rys_ssss_shell_bin_pair <
                Screened, PrimitiveFast, Emit > (
                bd, *task.bra, *task.ket, task.same_bin, eri_values);
            return;
        case RysSpShellBatchKind::SspsBraP:
            if constexpr (std::is_same_v <
                              Emit, RysDirectJSpeedProbeEmit > ) {
                // WHAT: route every ps-lane | ss-scalar packet to the J-only
                // producer, including masked and tail instantiations.
                // WHY/PURPOSE: a per-packet fallback would silently mix the
                // old raw receiver into the measured candidate.
                accumulate_rys_ssps_j_speed_probe_shell_bin_pair <
                    true, Screened, PrimitiveFast > (
                    bd, *task.bra, *task.ket, eri_values);
            } else {
                accumulate_rys_ssps_shell_bin_pair <
                    true, Screened, PrimitiveFast, Emit > (
                    bd, *task.bra, *task.ket, eri_values);
            }
            return;
        case RysSpShellBatchKind::SspsKetP:
            if constexpr (std::is_same_v <
                              Emit, RysDirectJSpeedProbeEmit > ) {
                // WHAT/WHY/PURPOSE: symmetric orientation of the same
                // complete ss|ps ownership rule; no center-distinct gate is
                // needed because J has no cross-index K aliases.
                accumulate_rys_ssps_j_speed_probe_shell_bin_pair <
                    false, Screened, PrimitiveFast > (
                    bd, *task.bra, *task.ket, eri_values);
            } else {
                accumulate_rys_ssps_shell_bin_pair <
                    false, Screened, PrimitiveFast, Emit > (
                    bd, *task.bra, *task.ket, eri_values);
            }
            return;
        case RysSpShellBatchKind::Psps:
            accumulate_rys_psps_shell_bin_pair <
                Screened, PrimitiveFast, Emit > (
                bd, *task.bra, *task.ket, task.same_bin, eri_values);
            return;
        case RysSpShellBatchKind::SsppBraPP:
            accumulate_rys_sspp_shell_bin_pair <
                true, Screened, PrimitiveFast, Emit > (
                bd, *task.bra, *task.ket, eri_values);
            return;
        case RysSpShellBatchKind::SsppKetPP:
            accumulate_rys_sspp_shell_bin_pair <
                false, Screened, PrimitiveFast, Emit > (
                bd, *task.bra, *task.ket, eri_values);
            return;
        case RysSpShellBatchKind::PsppBraPP:
            accumulate_rys_pspp_shell_bin_pair <
                true, Screened, PrimitiveFast, Emit > (
                bd, *task.bra, *task.ket, eri_values);
            return;
        case RysSpShellBatchKind::PsppKetPP:
            accumulate_rys_pspp_shell_bin_pair <
                false, Screened, PrimitiveFast, Emit > (
                bd, *task.bra, *task.ket, eri_values);
            return;
        case RysSpShellBatchKind::Pppp:
            accumulate_rys_pppp_shell_bin_pair <
                Screened, PrimitiveFast, Emit > (
                bd, *task.bra, *task.ket, task.same_bin, eri_values);
            return;
    }
}

template <class Emit>
inline void run_rys_sp_shell_batch_task(
    const BasisDataHGP& bd,
    const RysSpShellBatchTask& task,
    std::vector<double>& eri_values) {
#ifdef TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST
    RysJVeloxTaskScope velox_task_scope(
        std::is_same_v<Emit, RysDirectJCommonProducerEmit>, bd, task);
#endif
    const bool q_screened =
        task.shell_screen_threshold > 0.0;
    // WHAT: density-Partial installs the masked route even if Q is WholeLive.
    // WHY/PURPOSE: separating both states keeps the local rejection by P
    // while allowing the already-proved Schwarz SIMD mask to be skipped; the
    // density test keeps its own q because it is part of the q*cap bound.
    const bool shell_masked = q_screened || task.density_mask;
    const bool primitive_fast =
        task.primitive_fastschwarz_threshold_squared > 0.0;
    if (shell_masked || primitive_fast) {
        const auto resolve_pair_count = [](
            const RysLowPairBinH* bin,
            std::uint32_t override_count) noexcept -> std::size_t {
            return override_count ==
                    std::numeric_limits<std::uint32_t>::max()
                ? bin->pair_count
                : static_cast<std::size_t>(override_count);
        };
        const RysSpShellScreenContext screen{
            task.shell_screen_threshold, q_screened,
            task.density_mask,
            task.primitive_fastschwarz_threshold_squared,
            rys_primitive_fast_class_index(task.kind),
            task.bra,
            resolve_pair_count(task.bra, task.bra_pair_count_override),
            task.ket,
            resolve_pair_count(task.ket, task.ket_pair_count_override)};
        RysSpShellScreenContextGuard screen_guard(&screen);
        if (shell_masked &&
            g_rys_spherical_screening_diagnostics != nullptr) {
            ++g_rys_spherical_screening_diagnostics->sp_masked_tasks;
        }
        if (shell_masked) {
            if (primitive_fast) {
                run_rys_sp_shell_batch_task_impl<true, true, Emit>(
                    bd, task, eri_values);
            } else {
                run_rys_sp_shell_batch_task_impl<true, false, Emit>(
                    bd, task, eri_values);
            }
        } else {
            run_rys_sp_shell_batch_task_impl<false, true, Emit>(
                bd, task, eri_values);
        }
        return;
    }
    run_rys_sp_shell_batch_task_impl<false, false, Emit>(
        bd, task, eri_values);
}


// Bound ordering depends only on geometry and basis, so the chunk set is built
// once and cached with the plan regardless of which threshold is active.

// Shared execution for a prepared task list: dynamic scheduling with per-task
// stream accumulators reduced in stable task order.
template <class Emit>
inline void run_rys_sp_shell_batch_task_list(
    const BasisDataHGP& bd,
    const std::vector<RysSpShellBatchTask>& tasks,
    std::vector<double>& eri_values) {
#ifdef _OPENMP
    const bool use_parallel_task_accumulators =
        omp_get_max_threads() > 1 && tasks.size() >= 2u;
#else
    constexpr bool use_parallel_task_accumulators = false;
#endif
    if (!use_parallel_task_accumulators) {
        // The pure-spherical endpoint is currently serial.  Running directly
        // into its parent stream accumulator avoids allocating and reducing
        // one accumulator per task (the chunked route created thousands).
        for (const RysSpShellBatchTask& task : tasks) {
            run_rys_sp_shell_batch_task<Emit>(
                bd, task, eri_values);
        }
        return;
    }

    const bool streaming = eri_streaming();
    EriStreamAccumulator* const parent_accumulator =
        streaming ? &eri_stream_accumulator() : nullptr;
    const auto* const pair_components = g_low_batch_pair_components;
    std::vector<EriStreamAccumulator> task_accumulators(
        streaming ? tasks.size() : 0u);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (tasks.size() >= 2)
#endif
    for (int task_index = 0;
         task_index < static_cast<int>(tasks.size());
         ++task_index) {
        EriStreamModeGuard task_stream_mode(
            streaming, pair_components,
            streaming
                ? &task_accumulators[static_cast<std::size_t>(task_index)]
                : nullptr);
        run_rys_sp_shell_batch_task<Emit>(
            bd, tasks[static_cast<std::size_t>(task_index)], eri_values);
    }

    if (streaming) {
        for (const EriStreamAccumulator& accumulator : task_accumulators) {
            merge_eri_stream_accumulator(*parent_accumulator, accumulator);
        }
    }
}


// Keep the serial low-L producer behind an explicit code-generation boundary.
// When the Cartesian oracle shared this TU, its second caller made GCC split
// the OpenMP dispatch automatically.  Moving that oracle to its own object
// removed the incidental partial-inlining decision and folded the parallel
// setup into the one-thread hot function.  This boundary expresses the real
// runtime split without coupling production back to validation.
__attribute__((noinline)) void accumulate_rys_sp_shell_batches_serial_packed(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    accumulate_rys_ssss_shell_batches<RysPackedTensorEmit>(bd, eri_values);
    accumulate_rys_ssps_shell_batches<RysPackedTensorEmit>(bd, eri_values);
    accumulate_rys_psps_shell_batches<RysPackedTensorEmit>(bd, eri_values);
    accumulate_rys_sspp_shell_batches<RysPackedTensorEmit>(bd, eri_values);
    accumulate_rys_pspp_shell_batches<RysPackedTensorEmit>(bd, eri_values);
    accumulate_rys_pppp_shell_batches<RysPackedTensorEmit>(bd, eri_values);
}

template <class Emit>
inline void accumulate_rys_sp_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
#ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    if (max_threads > 1) {
        std::vector<RysSpShellBatchTask> tasks;
        append_rys_sp_shell_batch_tasks(bd, tasks);
        const bool streaming = eri_streaming();
        EriStreamAccumulator* const parent_accumulator =
            streaming ? &eri_stream_accumulator() : nullptr;
        const auto* const pair_components = g_low_batch_pair_components;
        std::vector<EriStreamAccumulator> task_accumulators(
            streaming ? tasks.size() : 0u);

#pragma omp parallel for schedule(dynamic, 1) if (tasks.size() >= 2)
        for (int task_index = 0;
             task_index < static_cast<int>(tasks.size());
             ++task_index) {
            EriStreamModeGuard task_stream_mode(
                streaming, pair_components,
                streaming
                    ? &task_accumulators[static_cast<std::size_t>(task_index)]
                    : nullptr);
            run_rys_sp_shell_batch_task<Emit>(
                bd, tasks[static_cast<std::size_t>(task_index)], eri_values);
        }

        if (streaming) {
            for (const EriStreamAccumulator& accumulator : task_accumulators) {
                merge_eri_stream_accumulator(
                    *parent_accumulator, accumulator);
            }
        }
        return;
    }
#endif
    if constexpr (std::is_same_v<Emit, RysPackedTensorEmit>) {
        accumulate_rys_sp_shell_batches_serial_packed(bd, eri_values);
    } else {
        // Non-packed consumers retain the original inline instantiation. In
        // particular, changing this boundary perturbs the bitwise J payload
        // identity even when its aggregate moments still agree.
        accumulate_rys_ssss_shell_batches<Emit>(bd, eri_values);
        accumulate_rys_ssps_shell_batches<Emit>(bd, eri_values);
        accumulate_rys_psps_shell_batches<Emit>(bd, eri_values);
        accumulate_rys_sspp_shell_batches<Emit>(bd, eri_values);
        accumulate_rys_pspp_shell_batches<Emit>(bd, eri_values);
        accumulate_rys_pppp_shell_batches<Emit>(bd, eri_values);
    }
}

// Keep one binary64 normalization for every direct-pure d evaluator and its
// derivative tables.  In particular, using the neighbouring rounding produced
// by `1.0 / std::sqrt(3.0)` in only some kernels would make the tabulated
// Hessian differ by one ULP from the polynomial it differentiates.  Sharing
// this literal makes the polynomial, gradients, and Hessian use the same
// binary64 coefficient.  Algebraically H_z2 = diag(-a,-a,2a), so its trace
// vanishes by construction; this is why a ket d has no B01 same-side trace
// term.  This states coefficient consistency, not bitwise equivalence between
// differently associated FMA expressions.
inline constexpr double kRysDSolidInvSqrt3 =
    0.57735026918962576451;

#include "integrals/two_electron/fock/detail/ssds_receivers.ipp"

// Pure-spherical (ss|ds) cross-shellset kernel.  Four unrelated shellsets
// occupy the AVX2 lanes, two Rys roots are evaluated lane-parallel, and the six
// Cartesian d contractions are transformed to five real-spherical components
// once after primitive contraction.
template <bool BraDS, bool Screened, class Emit>
inline void accumulate_rys_ssds_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    alignas(32) double targ4[4];
    alignas(32) double roots8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double weights8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double raw[5][4];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const __m256d vinv_sqrt3 = _mm256_set1_pd(kRysDSolidInvSqrt3);
    const std::size_t bra_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(ket_bin);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_end; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_end;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_end - bra_pos);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask);
                if (active_mask == 0u) {
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : rys_low_bits(valid_lanes);
            __m256d acc[6];  // xx, xy, xz, yy, yz, zz
            for (auto& value : acc) value = _mm256_setzero_pd();

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)), vr2));
                    rys_roots_weights_batch4_for_emit<Emit, 2>(
                        targ4, roots8, weights8);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    for (int root = 0; root < ActiveRoots<Emit, 2>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots8 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights8 + 4 * root));
                        __m256d X[3];
                        __m256d Bpair;
                        if constexpr (BraDS) {
                            const __m256d ratio = _mm256_mul_pd(vq, vinvpq);
                            const __m256d scale = _mm256_mul_pd(ratio, vrt);
                            X[0] = _mm256_fnmadd_pd(
                                scale, vdx,
                                _mm256_loadu_pd(bra_bin.Lx.data() + boff));
                            X[1] = _mm256_fnmadd_pd(
                                scale, vdy,
                                _mm256_loadu_pd(bra_bin.Ly.data() + boff));
                            X[2] = _mm256_fnmadd_pd(
                                scale, vdz,
                                _mm256_loadu_pd(bra_bin.Lz.data() + boff));
                            Bpair = _mm256_mul_pd(
                                _mm256_mul_pd(
                                    vhalf,
                                    _mm256_loadu_pd(bra_bin.invs.data() + boff)),
                                _mm256_fnmadd_pd(ratio, vrt, vone));
                        } else {
                            const __m256d ratio = _mm256_mul_pd(vp, vinvpq);
                            const __m256d scale = _mm256_mul_pd(ratio, vrt);
                            X[0] = _mm256_fmadd_pd(
                                scale, vdx, _mm256_set1_pd(ket_bin.Lx[koff]));
                            X[1] = _mm256_fmadd_pd(
                                scale, vdy, _mm256_set1_pd(ket_bin.Ly[koff]));
                            X[2] = _mm256_fmadd_pd(
                                scale, vdz, _mm256_set1_pd(ket_bin.Lz[koff]));
                            Bpair = _mm256_mul_pd(
                                _mm256_mul_pd(
                                    vhalf, _mm256_set1_pd(ket_bin.invs[koff])),
                                _mm256_fnmadd_pd(ratio, vrt, vone));
                        }
                        acc[0] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(X[0], X[0], Bpair), acc[0]);
                        acc[1] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(X[0], X[1]), acc[1]);
                        acc[2] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(X[0], X[2]), acc[2]);
                        acc[3] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(X[1], X[1], Bpair), acc[3]);
                        acc[4] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(X[1], X[2]), acc[4]);
                        acc[5] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(X[2], X[2], Bpair), acc[5]);
                    }
                }
            }

            const __m256d pure_xy = acc[1];
            const __m256d pure_yz = acc[4];
            const __m256d pure_z2 = _mm256_mul_pd(
                vinv_sqrt3,
                _mm256_sub_pd(
                    acc[5],
                    _mm256_mul_pd(vhalf, _mm256_add_pd(acc[0], acc[3]))));
            const __m256d pure_xz = acc[2];
            const __m256d pure_x2_y2 =
                _mm256_mul_pd(vhalf, _mm256_sub_pd(acc[0], acc[3]));
            const auto& rhs_refs =
                bd.packed_spherical_pair_components[static_cast<std::size_t>(ket_group)];
            if constexpr (
                rys_emit_uses_pairspace_preemit_ssds_j_v<Emit>) {
                const unsigned live_bits = Screened
                    ? active_mask
                    : rys_low_bits(valid_lanes);
                if (live_bits != 0u) {
                    rys_j_ssds_common_preemit_packet<BraDS>(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        rhs_refs, live_bits, pure_xy, pure_yz, pure_z2,
                        pure_xz, pure_x2_y2);
                }
            } else {
                _mm256_store_pd(raw[0], pure_xy);
                _mm256_store_pd(raw[1], pure_yz);
                _mm256_store_pd(raw[2], pure_z2);
                _mm256_store_pd(raw[3], pure_xz);
                _mm256_store_pd(raw[4], pure_x2_y2);
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((active_mask &
                             (1u << static_cast<unsigned>(lane))) == 0u) continue;
                    }
                    const std::uint32_t bra_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& lhs_refs =
                        bd.packed_spherical_pair_components[
                            static_cast<std::size_t>(bra_group)];
                    rys_emit_spherical_lane_block<Emit>(
                        eri_values, lhs_refs, rhs_refs, false,
                        [&raw, lane](const PackedPairComponentH& lhs_ref,
                                     const PackedPairComponentH& rhs_ref) noexcept {
                            const std::size_t component = BraDS
                                ? static_cast<std::size_t>(
                                      lhs_ref.component_index)
                                : static_cast<std::size_t>(
                                      rhs_ref.component_index);
                            return raw[component][lane];
                        });
                }
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
inline void accumulate_rys_ssds_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& ss_bins = bd.rys_low_pair_bins[0];
    const auto& ds_bins = bd.rys_ds_pair_bins;
    for (std::size_t si = 1; si < ss_bins.size(); ++si) {
        if (ss_bins[si].pair_count == 0) continue;
        for (std::size_t di = 1; di < ds_bins.size(); ++di) {
            if (ds_bins[di].pair_count == 0) continue;
            if (ds_bins[di].pair_count >= ss_bins[si].pair_count) {
                accumulate_rys_ssds_spherical_shell_bin_pair<true, false, Emit>(
                    bd, ds_bins[di], ss_bins[si], eri_values);
            } else {
                accumulate_rys_ssds_spherical_shell_bin_pair<false, false, Emit>(
                    bd, ss_bins[si], ds_bins[di], eri_values);
            }
        }
    }
}

// Pure-spherical (ds|ps) cross-shellset kernel.  The five real d harmonics are
// traceless, so the B10/B01 same-side recurrence term cancels in the 6 -> 5
// transform.  This permits a fixed 15-output microkernel:
//
//   H_d(Xd) * Xp[k] + B00 * d(H_d)/d(Xd[k]),
//
// evaluated for four unrelated shellsets in AVX2 lanes.  Primitive pairs and
// both Rys roots are fully contracted before any result is emitted.
template <bool BraDS, bool Screened, class Emit>
inline void accumulate_rys_dsps_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    alignas(32) double targ4[4];
    alignas(32) double roots8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double weights8[4 * ActiveRoots<Emit, 2>];
    alignas(32) double raw[15][4];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo = _mm256_set1_pd(2.0);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const __m256d vinv_sqrt3 = _mm256_set1_pd(kRysDSolidInvSqrt3);
    const std::size_t bra_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(ket_bin);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_end; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_end;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_end - bra_pos);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask);
                if (active_mask == 0u) {
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : rys_low_bits(valid_lanes);
            __m256d acc[15];
            for (auto& value : acc) value = _mm256_setzero_pd();

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPAx =
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff);
                const __m256d vPAy =
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff);
                const __m256d vPAz =
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff);
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                    const __m256d vdx = _mm256_sub_pd(
                        vPx, _mm256_set1_pd(ket_bin.Wx[koff]));
                    const __m256d vdy = _mm256_sub_pd(
                        vPy, _mm256_set1_pd(ket_bin.Wy[koff]));
                    const __m256d vdz = _mm256_sub_pd(
                        vPz, _mm256_set1_pd(ket_bin.Wz[koff]));
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vdx, vdx,
                        _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdz, vdz)));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                            vr2));
                    rys_roots_weights_batch4_for_emit<Emit, 2>(
                        targ4, roots8, weights8);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    for (int root = 0; root < ActiveRoots<Emit, 2>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots8 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights8 + 4 * root));
                        const __m256d vqscale =
                            _mm256_mul_pd(_mm256_mul_pd(vq, vinvpq), vrt);
                        const __m256d vpscale =
                            _mm256_mul_pd(_mm256_mul_pd(vp, vinvpq), vrt);
                        const __m256d C[3] = {
                            _mm256_fnmadd_pd(vqscale, vdx, vPAx),
                            _mm256_fnmadd_pd(vqscale, vdy, vPAy),
                            _mm256_fnmadd_pd(vqscale, vdz, vPAz),
                        };
                        const __m256d D[3] = {
                            _mm256_fmadd_pd(
                                vpscale, vdx,
                                _mm256_set1_pd(ket_bin.Lx[koff])),
                            _mm256_fmadd_pd(
                                vpscale, vdy,
                                _mm256_set1_pd(ket_bin.Ly[koff])),
                            _mm256_fmadd_pd(
                                vpscale, vdz,
                                _mm256_set1_pd(ket_bin.Lz[koff])),
                        };
                        const __m256d* const Xd = BraDS ? C : D;
                        const __m256d* const Xp = BraDS ? D : C;
                        const __m256d B00 = _mm256_mul_pd(
                            _mm256_mul_pd(vhalf, vinvpq), vrt);

                        const __m256d hxy = _mm256_mul_pd(Xd[0], Xd[1]);
                        const __m256d hyz = _mm256_mul_pd(Xd[1], Xd[2]);
                        const __m256d hz2 = _mm256_mul_pd(
                            vinv_sqrt3,
                            _mm256_sub_pd(
                                _mm256_mul_pd(Xd[2], Xd[2]),
                                _mm256_mul_pd(
                                    vhalf,
                                    _mm256_fmadd_pd(
                                        Xd[0], Xd[0],
                                        _mm256_mul_pd(Xd[1], Xd[1])))));
                        const __m256d hxz = _mm256_mul_pd(Xd[0], Xd[2]);
                        const __m256d hx2y2 = _mm256_mul_pd(
                            vhalf,
                            _mm256_sub_pd(
                                _mm256_mul_pd(Xd[0], Xd[0]),
                                _mm256_mul_pd(Xd[1], Xd[1])));
                        const __m256d Bx = _mm256_mul_pd(B00, Xd[0]);
                        const __m256d By = _mm256_mul_pd(B00, Xd[1]);
                        const __m256d Bz = _mm256_mul_pd(B00, Xd[2]);

                        // d=xy, p=x/y/z
                        acc[0] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(B00, Xd[1],
                                                _mm256_mul_pd(hxy, Xp[0])),
                            acc[0]);
                        acc[1] = _mm256_fmadd_pd(
                            vww, _mm256_fmadd_pd(B00, Xd[0],
                                                _mm256_mul_pd(hxy, Xp[1])),
                            acc[1]);
                        acc[2] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(hxy, Xp[2]), acc[2]);
                        // d=yz, p=x/y/z
                        acc[3] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(hyz, Xp[0]), acc[3]);
                        acc[4] = _mm256_fmadd_pd(
                            vww, _mm256_add_pd(_mm256_mul_pd(hyz, Xp[1]), Bz),
                            acc[4]);
                        acc[5] = _mm256_fmadd_pd(
                            vww, _mm256_add_pd(_mm256_mul_pd(hyz, Xp[2]), By),
                            acc[5]);
                        // d=(zz-(xx+yy)/2)/sqrt(3), p=x/y/z
                        acc[6] = _mm256_fmadd_pd(
                            vww,
                            _mm256_fnmadd_pd(
                                vinv_sqrt3, Bx, _mm256_mul_pd(hz2, Xp[0])),
                            acc[6]);
                        acc[7] = _mm256_fmadd_pd(
                            vww,
                            _mm256_fnmadd_pd(
                                vinv_sqrt3, By, _mm256_mul_pd(hz2, Xp[1])),
                            acc[7]);
                        acc[8] = _mm256_fmadd_pd(
                            vww,
                            _mm256_fmadd_pd(
                                _mm256_mul_pd(vtwo, vinv_sqrt3), Bz,
                                _mm256_mul_pd(hz2, Xp[2])),
                            acc[8]);
                        // d=xz, p=x/y/z
                        acc[9] = _mm256_fmadd_pd(
                            vww, _mm256_add_pd(_mm256_mul_pd(hxz, Xp[0]), Bz),
                            acc[9]);
                        acc[10] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(hxz, Xp[1]), acc[10]);
                        acc[11] = _mm256_fmadd_pd(
                            vww, _mm256_add_pd(_mm256_mul_pd(hxz, Xp[2]), Bx),
                            acc[11]);
                        // d=(xx-yy)/2, p=x/y/z
                        acc[12] = _mm256_fmadd_pd(
                            vww, _mm256_add_pd(_mm256_mul_pd(hx2y2, Xp[0]), Bx),
                            acc[12]);
                        acc[13] = _mm256_fmadd_pd(
                            vww, _mm256_sub_pd(_mm256_mul_pd(hx2y2, Xp[1]), By),
                            acc[13]);
                        acc[14] = _mm256_fmadd_pd(
                            vww, _mm256_mul_pd(hx2y2, Xp[2]), acc[14]);
                    }
                }
            }

            for (int component = 0; component < 15; ++component) {
                _mm256_store_pd(raw[component], acc[component]);
            }
            const auto& rhs_refs =
                bd.packed_spherical_pair_components[static_cast<std::size_t>(ket_group)];
            for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                if constexpr (Screened) {
                    if ((active_mask &
                         (1u << static_cast<unsigned>(lane))) == 0u) continue;
                }
                const std::uint32_t bra_group =
                    bra_bin.pair_group_ids[bra_pos + lane];
                const auto& lhs_refs =
                    bd.packed_spherical_pair_components[static_cast<std::size_t>(bra_group)];
                rys_emit_spherical_lane_block<Emit>(
                    eri_values, lhs_refs, rhs_refs, false,
                    [&raw, lane](const PackedPairComponentH& lhs_ref,
                                 const PackedPairComponentH& rhs_ref) noexcept {
                        const std::size_t d_component = BraDS
                            ? static_cast<std::size_t>(lhs_ref.component_index)
                            : static_cast<std::size_t>(rhs_ref.component_index);
                        const std::size_t p_component = BraDS
                            ? static_cast<std::size_t>(rhs_ref.component_index)
                            : static_cast<std::size_t>(lhs_ref.component_index);
                        return raw[3u * d_component + p_component][lane];
                    });
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
inline void accumulate_rys_dsps_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& ds_bins = bd.rys_ds_pair_bins;
    const auto& ps_bins = bd.rys_low_pair_bins[1];
    for (std::size_t di = 1; di < ds_bins.size(); ++di) {
        if (ds_bins[di].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < ps_bins.size(); ++pi) {
            if (ps_bins[pi].pair_count == 0) continue;
            if (ds_bins[di].pair_count >= ps_bins[pi].pair_count) {
                accumulate_rys_dsps_spherical_shell_bin_pair<true, false, Emit>(
                    bd, ds_bins[di], ps_bins[pi], eri_values);
            } else {
                accumulate_rys_dsps_spherical_shell_bin_pair<false, false, Emit>(
                    bd, ps_bins[pi], ds_bins[di], eri_values);
            }
        }
    }
}

// Unit-normalized real d solid harmonics and first derivatives in the same
// ordering used by the spherical pair-component tables.
inline void rys_d_solid_harmonics_v(
    const __m256d X[3], __m256d H[5], __m256d dH[5][3]) noexcept {
    const __m256d zero = _mm256_setzero_pd();
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d invsqrt3 = _mm256_set1_pd(kRysDSolidInvSqrt3);
    const __m256d x = X[0], y = X[1], z = X[2];
    const __m256d x2 = _mm256_mul_pd(x, x);
    const __m256d y2 = _mm256_mul_pd(y, y);
    const __m256d z2 = _mm256_mul_pd(z, z);

    H[0] = _mm256_mul_pd(x, y);
    H[1] = _mm256_mul_pd(y, z);
    H[2] = _mm256_mul_pd(
        invsqrt3,
        _mm256_sub_pd(z2, _mm256_mul_pd(half, _mm256_add_pd(x2, y2))));
    H[3] = _mm256_mul_pd(x, z);
    H[4] = _mm256_mul_pd(half, _mm256_sub_pd(x2, y2));

    dH[0][0] = y;    dH[0][1] = x;    dH[0][2] = zero;
    dH[1][0] = zero; dH[1][1] = z;    dH[1][2] = y;
    dH[2][0] = _mm256_sub_pd(zero, _mm256_mul_pd(invsqrt3, x));
    dH[2][1] = _mm256_sub_pd(zero, _mm256_mul_pd(invsqrt3, y));
    dH[2][2] = _mm256_mul_pd(_mm256_mul_pd(two, invsqrt3), z);
    dH[3][0] = z;    dH[3][1] = zero; dH[3][2] = x;
    dH[4][0] = x;    dH[4][1] = _mm256_sub_pd(zero, y);
    dH[4][2] = zero;
}

// Constant Hessians of the five normalized real d harmonics above.  Keeping
// these as scalar compile-time data avoids materializing a 45-YMM constant
// array in every high-L kernel.
inline constexpr double kRysDSolidHessian[5][3][3] = {
    {{0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}},
    {{-kRysDSolidInvSqrt3, 0.0, 0.0},
     {0.0, -kRysDSolidInvSqrt3, 0.0},
     {0.0, 0.0, 2.0 * kRysDSolidInvSqrt3}},
    {{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
    {{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}},
};

// JIT-style component specialization: a dp|{ds,pp} fragment needs one d
// harmonic at a time, so do not build the other four values or derivatives.
template <int Component>
inline void rys_d_solid_harmonic_component_v(
    const __m256d X[3], __m256d& H, __m256d dH[3]) noexcept {
    static_assert(Component >= 0 && Component < 5, "invalid d component");
    const __m256d zero = _mm256_setzero_pd();
    const __m256d x = X[0], y = X[1], z = X[2];
    if constexpr (Component == 0) {
        H = _mm256_mul_pd(x, y);
        dH[0] = y; dH[1] = x; dH[2] = zero;
    } else if constexpr (Component == 1) {
        H = _mm256_mul_pd(y, z);
        dH[0] = zero; dH[1] = z; dH[2] = y;
    } else if constexpr (Component == 2) {
        const __m256d half = _mm256_set1_pd(0.5);
        const __m256d invsqrt3 =
            _mm256_set1_pd(kRysDSolidInvSqrt3);
        const __m256d x2 = _mm256_mul_pd(x, x);
        const __m256d y2 = _mm256_mul_pd(y, y);
        const __m256d z2 = _mm256_mul_pd(z, z);
        H = _mm256_mul_pd(
            invsqrt3,
            _mm256_sub_pd(z2, _mm256_mul_pd(half, _mm256_add_pd(x2, y2))));
        dH[0] = _mm256_sub_pd(zero, _mm256_mul_pd(invsqrt3, x));
        dH[1] = _mm256_sub_pd(zero, _mm256_mul_pd(invsqrt3, y));
        dH[2] = _mm256_mul_pd(
            _mm256_set1_pd(2.0 * kRysDSolidInvSqrt3), z);
    } else if constexpr (Component == 3) {
        H = _mm256_mul_pd(x, z);
        dH[0] = z; dH[1] = zero; dH[2] = x;
    } else {
        const __m256d half = _mm256_set1_pd(0.5);
        H = _mm256_mul_pd(
            half,
            _mm256_sub_pd(_mm256_mul_pd(x, x), _mm256_mul_pd(y, y)));
        dH[0] = x; dH[1] = _mm256_sub_pd(zero, y); dH[2] = zero;
    }
}

// Dynamic root data reused by the five compile-time first-d fragments of
// dd|ss and dd|ps.  The d-d horizontal transfer is evaluated directly in the
// pure basis; no 36-column Cartesian dd temporary is formed.
struct alignas(32) RysDDLowRootRecordV {
    __m256d weight;
    __m256d xdd[3];
    __m256d xlow[3];
    __m256d b00;
    __m256d bside;
    __m256d bsidesq;
};

// OtherL selects the class of the partner pair: 0 = ss, 1 = ps, 2 = pp.
// `rlow` is the partner's internal displacement (A-B), analogous to `rdd` for
// the dd pair.  Only the pp case uses it, where the two p shells live on
// different centers; for ss and ps the partner has a single angular shell and
// the parameter is never read.
// B coefficient of the side OPPOSITE the dd pair: B01 when the dd is in the
// bra, B10 when it is in the ket (`bside` already holds the dd side's one).
//
// WHY it goes in a SEPARATE record and not in the common one: only the pp
// partner needs it.  ss and ps have degree <= 1, and the Gaussian smoothing of
// a degree-1 polynomial produces no term; the ds ket is a solid harmonic with
// zero Laplacian and does not use it either.  Putting it in the common record
// raised EVERY record from 320 to 352 bytes and penalized the cache traversal
// of dd|ss and dd|ps, although neither of them ever reads it.
struct alignas(32) RysDDLowRootRecordV4 : RysDDLowRootRecordV {
    __m256d bother;
};

// The class fixes the record type: only OtherL == 2 (a pp partner) pays the
// extra 32 bytes.
template <int OtherL>
using RysDDLowRecordFor = std::conditional_t <
    OtherL == 2, RysDDLowRootRecordV4, RysDDLowRootRecordV > ;

template <int DComponent, int OtherL>
inline void contract_rys_dd_low_fragment(
    const std::vector<RysDDLowRecordFor<OtherL>>& records,
    std::size_t active_record_count,
    const __m256d rdd[3],
    const __m256d rlow[3] [[maybe_unused]],
    double* __restrict raw) noexcept {
    static_assert(OtherL >= 0 && OtherL <= 3,
                  "dd kernel: ss/ps/pp mas el tag ds (OtherL == 3)");
    constexpr int NOther =
        OtherL == 0 ? 1 : (OtherL == 1 ? 3 : (OtherL == 2 ? 9 : 5));
    constexpr int NFragment = 5 * NOther;
    __m256d acc[NFragment];
    for (auto& value : acc) value = _mm256_setzero_pd();

    for (std::size_t record_pos = 0;
         record_pos < active_record_count; ++record_pos) {
        const RysDDLowRecordFor<OtherL>& record = records[record_pos];
        __m256d ha, dha[3];
        rys_d_solid_harmonic_component_v<DComponent>(record.xdd, ha, dha);
        const __m256d shifted[3] = {
            _mm256_add_pd(record.xdd[0], rdd[0]),
            _mm256_add_pd(record.xdd[1], rdd[1]),
            _mm256_add_pd(record.xdd[2], rdd[2]),
        };
        __m256d hb[5], dhb[5][3];
        rys_d_solid_harmonics_v(shifted, hb, dhb);
        // ds ket: the s does not shift the center, so the ket's d harmonic is
        // evaluated at xlow without adding rlow (unlike the pp ket).
        [[maybe_unused]] __m256d hlow[5], dhlow[5][3];
        if constexpr (OtherL == 3) {
            rys_d_solid_harmonics_v(record.xlow, hlow, dhlow);
        }

        for (int b = 0; b < 5; ++b) {
            __m256d dot = _mm256_mul_pd(dha[0], dhb[b][0]);
            dot = _mm256_fmadd_pd(dha[1], dhb[b][1], dot);
            dot = _mm256_fmadd_pd(dha[2], dhb[b][2], dot);
            __m256d pair_value = _mm256_fmadd_pd(
                record.bside, dot, _mm256_mul_pd(ha, hb[b]));
            if (DComponent == b) {
                pair_value = _mm256_add_pd(pair_value, record.bsidesq);
            }

            if constexpr (OtherL == 0) {
                acc[b] = _mm256_fmadd_pd(
                    record.weight, pair_value, acc[b]);
            } else if constexpr (OtherL == 2) {
                // pp partner:  V_ij = P*Q + B00*(dP.dQ) + B00^2*(ddP:ddQ)/2
                // with Q(y) = y_i (y_j + S_j), whose smoothing in B01 adds
                // B01*delta_ij.  The contraction of second derivatives is
                // symmetrized and cancels the 1/2, so the term becomes
                // B00^2 * d_i d_j P.  Structure checked against exact
                // Gaussian moments and modeled on contract_rys_dp_matrix_fragment,
                // which already solves a pp ket with this same layout.
                __m256d dP[3];
                for (int p = 0; p < 3; ++p) {
                    __m256d derivative = _mm256_fmadd_pd(
                        dha[p], hb[b], _mm256_mul_pd(ha, dhb[b][p]));
                    __m256d contraction = _mm256_setzero_pd();
                    for (int axis = 0; axis < 3; ++axis) {
                        const double ca =
                            kRysDSolidHessian[DComponent][p][axis];
                        if (ca != 0.0) {
                            contraction = _mm256_fmadd_pd(
                                _mm256_set1_pd(ca), dhb[b][axis], contraction);
                        }
                        const double cb = kRysDSolidHessian[b][p][axis];
                        if (cb != 0.0) {
                            contraction = _mm256_fmadd_pd(
                                _mm256_set1_pd(cb), dha[axis], contraction);
                        }
                    }
                    dP[p] = _mm256_fmadd_pd(
                        record.bside, contraction, derivative);
                }
                const __m256d b00sq =
                    _mm256_mul_pd(record.b00, record.b00);
                for (int pi = 0; pi < 3; ++pi) {
                    for (int pj = 0; pj < 3; ++pj) {
                        const __m256d shifted_low =
                            _mm256_add_pd(record.xlow[pj], rlow[pj]);
                        __m256d ket = _mm256_mul_pd(
                            record.xlow[pi], shifted_low);
                        if (pi == pj) {
                            ket = _mm256_add_pd(ket, record.bother);
                        }
                        __m256d first = _mm256_mul_pd(dP[pi], shifted_low);
                        first = _mm256_fmadd_pd(
                            dP[pj], record.xlow[pi], first);
                        // d_i d_j P: third derivatives of degree-2 harmonics
                        // vanish, so only Hessians take part.
                        __m256d second = _mm256_mul_pd(
                            _mm256_set1_pd(
                                kRysDSolidHessian[DComponent][pi][pj]),
                            hb[b]);
                        second = _mm256_fmadd_pd(
                            ha,
                            _mm256_set1_pd(kRysDSolidHessian[b][pi][pj]),
                            second);
                        second = _mm256_fmadd_pd(
                            dha[pi], dhb[b][pj], second);
                        second = _mm256_fmadd_pd(
                            dha[pj], dhb[b][pi], second);
                        __m256d cross = _mm256_setzero_pd();
                        for (int s = 0; s < 3; ++s) {
                            const double c1 =
                                kRysDSolidHessian[DComponent][pi][s] *
                                kRysDSolidHessian[b][pj][s];
                            const double c2 =
                                kRysDSolidHessian[DComponent][pj][s] *
                                kRysDSolidHessian[b][pi][s];
                            const double c = c1 + c2;
                            if (c != 0.0) {
                                cross = _mm256_add_pd(
                                    cross, _mm256_set1_pd(c));
                            }
                        }
                        second = _mm256_fmadd_pd(record.bside, cross, second);
                        __m256d value = _mm256_fmadd_pd(
                            record.b00, first, _mm256_mul_pd(pair_value, ket));
                        value = _mm256_fmadd_pd(b00sq, second, value);
                        const int local = 9 * b + 3 * pi + pj;
                        acc[local] = _mm256_fmadd_pd(
                            record.weight, value, acc[local]);
                    }
                }
            } else if constexpr (OtherL == 3) {
                // ds partner:  Q(y) = h_c(y), the ket's d solid harmonic.  The
                // Laplacian of a d harmonic is zero, so the smoothing in B01
                // adds no term and Q is evaluated directly.
                // The Gaussian series is truncated at k = 2 because the third
                // derivative of a degree-2 polynomial vanishes:
                //   V_c = P*Q + B00*(dP . dQ) + (B00^2/2)*sum_ij ddP_ij*ddQ_ij
                // The 1/2 factor DOES remain here, unlike in the pp ket:
                // there the symmetrization of the two indices cancels it, while
                // here all nine (i,j) run against a constant Hessian that is
                // already symmetric.
                const __m256d b00sq_half = _mm256_mul_pd(
                    _mm256_set1_pd(0.5),
                    _mm256_mul_pd(record.b00, record.b00));
                __m256d dP[3];
                for (int p = 0; p < 3; ++p) {
                    __m256d derivative = _mm256_fmadd_pd(
                        dha[p], hb[b], _mm256_mul_pd(ha, dhb[b][p]));
                    __m256d contraction = _mm256_setzero_pd();
                    for (int axis = 0; axis < 3; ++axis) {
                        const double ca =
                            kRysDSolidHessian[DComponent][p][axis];
                        if (ca != 0.0) {
                            contraction = _mm256_fmadd_pd(
                                _mm256_set1_pd(ca), dhb[b][axis], contraction);
                        }
                        const double cb = kRysDSolidHessian[b][p][axis];
                        if (cb != 0.0) {
                            contraction = _mm256_fmadd_pd(
                                _mm256_set1_pd(cb), dha[axis], contraction);
                        }
                    }
                    dP[p] = _mm256_fmadd_pd(
                        record.bside, contraction, derivative);
                }
                // ddP_ij does not depend on the ket component c: it is computed
                // once and reused for all five components.
                __m256d ddP[3][3];
                for (int pi = 0; pi < 3; ++pi) {
                    for (int pj = 0; pj < 3; ++pj) {
                        __m256d second = _mm256_mul_pd(
                            _mm256_set1_pd(
                                kRysDSolidHessian[DComponent][pi][pj]),
                            hb[b]);
                        second = _mm256_fmadd_pd(
                            ha,
                            _mm256_set1_pd(kRysDSolidHessian[b][pi][pj]),
                            second);
                        second = _mm256_fmadd_pd(
                            dha[pi], dhb[b][pj], second);
                        second = _mm256_fmadd_pd(
                            dha[pj], dhb[b][pi], second);
                        __m256d cross = _mm256_setzero_pd();
                        for (int t = 0; t < 3; ++t) {
                            const double c1 =
                                kRysDSolidHessian[DComponent][pi][t] *
                                kRysDSolidHessian[b][pj][t];
                            const double c2 =
                                kRysDSolidHessian[DComponent][pj][t] *
                                kRysDSolidHessian[b][pi][t];
                            const double c = c1 + c2;
                            if (c != 0.0) {
                                cross = _mm256_add_pd(
                                    cross, _mm256_set1_pd(c));
                            }
                        }
                        ddP[pi][pj] = _mm256_fmadd_pd(
                            record.bside, cross, second);
                    }
                }
                for (int c = 0; c < 5; ++c) {
                    __m256d value = _mm256_mul_pd(pair_value, hlow[c]);
                    __m256d dot2 = _mm256_mul_pd(dP[0], dhlow[c][0]);
                    dot2 = _mm256_fmadd_pd(dP[1], dhlow[c][1], dot2);
                    dot2 = _mm256_fmadd_pd(dP[2], dhlow[c][2], dot2);
                    value = _mm256_fmadd_pd(record.b00, dot2, value);
                    __m256d hess = _mm256_setzero_pd();
                    for (int pi = 0; pi < 3; ++pi) {
                        for (int pj = 0; pj < 3; ++pj) {
                            const double q = kRysDSolidHessian[c][pi][pj];
                            if (q != 0.0) {
                                hess = _mm256_fmadd_pd(
                                    _mm256_set1_pd(q), ddP[pi][pj], hess);
                            }
                        }
                    }
                    value = _mm256_fmadd_pd(b00sq_half, hess, value);
                    const int local = 5 * b + c;
                    acc[local] = _mm256_fmadd_pd(
                        record.weight, value, acc[local]);
                }
            } else {
                for (int p = 0; p < 3; ++p) {
                    __m256d derivative = _mm256_fmadd_pd(
                        dha[p], hb[b], _mm256_mul_pd(ha, dhb[b][p]));
                    __m256d contraction = _mm256_setzero_pd();
                    for (int axis = 0; axis < 3; ++axis) {
                        const double ca =
                            kRysDSolidHessian[DComponent][p][axis];
                        if (ca != 0.0) {
                            contraction = _mm256_fmadd_pd(
                                _mm256_set1_pd(ca), dhb[b][axis],
                                contraction);
                        }
                        const double cb = kRysDSolidHessian[b][p][axis];
                        if (cb != 0.0) {
                            contraction = _mm256_fmadd_pd(
                                _mm256_set1_pd(cb), dha[axis], contraction);
                        }
                    }
                    derivative = _mm256_fmadd_pd(
                        record.bside, contraction, derivative);
                    const __m256d value = _mm256_fmadd_pd(
                        record.b00, derivative,
                        _mm256_mul_pd(pair_value, record.xlow[p]));
                    const int local = 3 * b + p;
                    acc[local] = _mm256_fmadd_pd(
                        record.weight, value, acc[local]);
                }
            }
        }
    }

    for (int b = 0; b < 5; ++b) {
        for (int other = 0; other < NOther; ++other) {
            const int local = NOther * b + other;
            const int component =
                ((5 * DComponent + b) * NOther + other) * 4;
            _mm256_store_pd(raw + component, acc[local]);
        }
    }
}

template <int OtherL, bool BraDD, bool Screened, class Emit>
inline void accumulate_rys_dd_low_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    static_assert(OtherL >= 0 && OtherL <= 3,
                  "dd kernel: ss/ps/pp mas el tag ds (OtherL == 3)");
    // Roots: (La+Lb+Lc+Ld)/2 + 1 with the dd bra fixed at 4.  ss and ps give
    // 3; pp gives 4, because the partner contributes two units of angular
    // momentum.  OtherL 2 (pp) and 3 (the ds tag) both add L = 2 to the ket.
    constexpr int OtherAngular = OtherL >= 2 ? 2 : OtherL;
    constexpr int NRoots = (4 + OtherAngular) / 2 + 1;
    constexpr int NOther =
        OtherL == 0 ? 1 : (OtherL == 1 ? 3 : (OtherL == 2 ? 9 : 5));
    constexpr int NOut = 25 * NOther;
    alignas(32) double targ4[4];
    alignas(32) double roots[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double weights[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double raw[NOut][4];
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;
    const std::size_t record_count =
        static_cast<std::size_t>(bra_bin.primitive_count) *
        static_cast<std::size_t>(ket_bin.primitive_count) *
        ActiveRoots<Emit, NRoots>;
    std::vector<RysDDLowRecordFor<OtherL>> records(record_count);
    const std::size_t bra_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(ket_bin);

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_end; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_end;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_end - bra_pos);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask);
                if (active_mask == 0u) {
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : rys_low_bits(valid_lanes);
            const __m256d bra_r[3] = {
                _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos),
            };
            const __m256d ket_r[3] = {
                _mm256_set1_pd(ket_bin.Rx[ket_pos]),
                _mm256_set1_pd(ket_bin.Ry[ket_pos]),
                _mm256_set1_pd(ket_bin.Rz[ket_pos]),
            };
            const __m256d* const rdd = BraDD ? bra_r : ket_r;
            // Partner's internal displacement: the side opposite the dd pair.
            const __m256d* const rlow = BraDD ? ket_r : bra_r;
            std::size_t record_index = 0;

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vinvp =
                    _mm256_loadu_pd(bra_bin.invs.data() + boff);
                const __m256d vpx =
                    _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vpy =
                    _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vpz =
                    _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vpa[3] = {
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff),
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff),
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff),
                };
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vinvq = _mm256_set1_pd(ket_bin.invs[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(one, vpq);
                    const __m256d delta[3] = {
                        _mm256_sub_pd(vpx, _mm256_set1_pd(ket_bin.Wx[koff])),
                        _mm256_sub_pd(vpy, _mm256_set1_pd(ket_bin.Wy[koff])),
                        _mm256_sub_pd(vpz, _mm256_set1_pd(ket_bin.Wz[koff])),
                    };
                    const __m256d vr2 = _mm256_fmadd_pd(
                        delta[0], delta[0],
                        _mm256_fmadd_pd(
                            delta[1], delta[1],
                            _mm256_mul_pd(delta[2], delta[2])));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                            vr2));
                    rys_roots_weights_batch4_for_emit<Emit, NRoots>(
                        targ4, roots, weights);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    const __m256d qop = _mm256_mul_pd(vq, vinvpq);
                    const __m256d pop = _mm256_mul_pd(vp, vinvpq);
                    for (int root = 0;
                         root < ActiveRoots<Emit, NRoots>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots + 4 * root);
                        RysDDLowRecordFor<OtherL>& record =
                            records[record_index++];
                        record.weight = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights + 4 * root));
                        const __m256d qscale = _mm256_mul_pd(qop, vrt);
                        const __m256d pscale = _mm256_mul_pd(pop, vrt);
                        const __m256d c[3] = {
                            _mm256_fnmadd_pd(qscale, delta[0], vpa[0]),
                            _mm256_fnmadd_pd(qscale, delta[1], vpa[1]),
                            _mm256_fnmadd_pd(qscale, delta[2], vpa[2]),
                        };
                        const __m256d d[3] = {
                            _mm256_fmadd_pd(
                                pscale, delta[0],
                                _mm256_set1_pd(ket_bin.Lx[koff])),
                            _mm256_fmadd_pd(
                                pscale, delta[1],
                                _mm256_set1_pd(ket_bin.Ly[koff])),
                            _mm256_fmadd_pd(
                                pscale, delta[2],
                                _mm256_set1_pd(ket_bin.Lz[koff])),
                        };
                        for (int axis = 0; axis < 3; ++axis) {
                            record.xdd[axis] = BraDD ? c[axis] : d[axis];
                            record.xlow[axis] = BraDD ? d[axis] : c[axis];
                        }
                        record.b00 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvpq), vrt);
                        const __m256d b10 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvp),
                            _mm256_fnmadd_pd(qop, vrt, one));
                        const __m256d b01 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvq),
                            _mm256_fnmadd_pd(pop, vrt, one));
                        record.bside = BraDD ? b10 : b01;
                        record.bsidesq =
                            _mm256_mul_pd(record.bside, record.bside);
                        // Opposite side: it exists only in the four-root
                        // record of the pp partner.  See RysDDLowRootRecordV4.
                        if constexpr (OtherL == 2) {
                            record.bother = BraDD ? b01 : b10;
                        }
                    }
                }
            }

            contract_rys_dd_low_fragment<0, OtherL>(
                records, record_index, rdd, rlow, &raw[0][0]);
            contract_rys_dd_low_fragment<1, OtherL>(
                records, record_index, rdd, rlow, &raw[0][0]);
            contract_rys_dd_low_fragment<2, OtherL>(
                records, record_index, rdd, rlow, &raw[0][0]);
            contract_rys_dd_low_fragment<3, OtherL>(
                records, record_index, rdd, rlow, &raw[0][0]);
            contract_rys_dd_low_fragment<4, OtherL>(
                records, record_index, rdd, rlow, &raw[0][0]);

            const auto& rhs_refs = bd.packed_spherical_pair_components[
                static_cast<std::size_t>(ket_group)];
            for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                if constexpr (Screened) {
                    if ((active_mask &
                         (1u << static_cast<unsigned>(lane))) == 0u) continue;
                }
                const std::uint32_t bra_group =
                    bra_bin.pair_group_ids[bra_pos + lane];
                const auto& lhs_refs = bd.packed_spherical_pair_components[
                    static_cast<std::size_t>(bra_group)];
                rys_emit_spherical_lane_block<Emit>(
                    eri_values, lhs_refs, rhs_refs, false,
                    [&raw, lane](const PackedPairComponentH& lhs_ref,
                                 const PackedPairComponentH& rhs_ref) noexcept {
                        const int dd_component = BraDD
                            ? static_cast<int>(lhs_ref.component_index)
                            : static_cast<int>(rhs_ref.component_index);
                        int component = dd_component;
                        if constexpr (OtherL >= 1) {
                            const int low_component = BraDD
                                ? static_cast<int>(rhs_ref.component_index)
                                : static_cast<int>(lhs_ref.component_index);
                            // The stride is NOther, not a fixed 3.  The fragment
                            // stores at ((5*DComponent + b)*NOther + other), so
                            // the emitter must use the SAME NOther.  With the
                            // hard-wired 3, dd|pp (NOther == 9) fell onto the ss
                            // path and read only 25 of the 225 raw entries: it
                            // emitted too little and into the wrong slot, with
                            // no compilation error at all.
                            component = NOther * dd_component + low_component;
                        }
                        return raw[component][lane];
                    });
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <int OtherL, class Emit>
inline void accumulate_rys_dd_low_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& dd_bins = bd.rys_dd_pair_bins;
    // OtherL == 3 is the explicit tag of the ds partner.  Its bins do NOT live
    // in rys_low_pair_bins, which indexes only ss/ps/pp and has size 3: indexing
    // it with 3 would be out of range.  The if constexpr picks the right source
    // at compile time and avoids that access entirely.
    const auto& low_bins = [&]() -> const std::vector<RysLowPairBinH>& {
        if constexpr (OtherL == 3) {
            return bd.rys_ds_pair_bins;
        } else {
            return bd.rys_low_pair_bins[static_cast<std::size_t>(OtherL)];
        }
    }();
    for (std::size_t di = 1; di < dd_bins.size(); ++di) {
        if (dd_bins[di].pair_count == 0) continue;
        for (std::size_t li = 1; li < low_bins.size(); ++li) {
            if (low_bins[li].pair_count == 0) continue;
            if (dd_bins[di].pair_count >= low_bins[li].pair_count) {
                accumulate_rys_dd_low_spherical_shell_bin_pair<OtherL, true, false, Emit>(
                    bd, dd_bins[di], low_bins[li], eri_values);
            } else {
                accumulate_rys_dd_low_spherical_shell_bin_pair<OtherL, false, false, Emit>(
                    bd, low_bins[li], dd_bins[di], eri_values);
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
// WHAT: enables the dd|pp specialization (OtherL == 2), currently OFF.
//
// WHY off: the branch is new and its only real gate is comparing checksums
// against the generic route, which serves dd|pp while this stays false.  A
// convention error in the output index or a lost factor in the B00^2 term
// does NOT break compilation: it produces subtly wrong integrals.  While
// the knob is off the engine behaves exactly as before and the branch is
// verifiable dead code.
//
// HOW to validate it: build one binary and run the nine cases with
// TLNO_RYS_DD_PP_BATCH=1 and without it.  If the branch is correct, the
// differences must be reassociation (~1e-13 relative), because it computes
// the same integrals in another order.  Any larger discrepancy means that
// the branch is wrong and must stay off.
// Same contract as the dd|pp knob: off by default until the checksum gate
// against the generic route backs it.  It is registered in the coverage
// under this same condition so that the class is not computed twice.
// GATE RUN 2026-08-04 (scripts/gate_dd_families.py).  The contract required
// by the comment above is met, and the verdict is twofold:
//
//   1. VALUE: correct.  Six molecules with d shells (631gd, 631gdp, def2svp,
//      def2tzvp), each family alone and both together: the relative drift of
//      the production checksum is at most 1.8e-14, 2700x below the
//      tolerance with which the nine-case sweep compares engines.  That
//      drift is the reassociation of a sum of ~1e8 terms under -ffast-math
//      when a different kernel produces the class, not a different value.
//      (Comparing by exact equality here is a mistake: with -ffast-math the
//      last bits move by themselves.)
//
//   2. PERFORMANCE: none.  Molecular on/off ratios fall in 0.93-1.10 with a
//      geomean of ~1.00.  The most favorable case, benzene/def2tzvp, gives
//      5.114 s against 5.031 s in the median but 4.473 s against 4.487 s in
//      the minimum -- they cross, so it is noise.
//
// Conclusion: these two classes are NOT a hidden slow generic route.  The
// generic route costs them the same as the specialized kernel, so the
// families stay off by default, now by measurement and not for lack of
// validation.  They are not deleted because the kernel is written and
// validated; if the generic route ever degrades for dd|pp or dd|ds, the
// replacement already exists, and so does the gate to authorize it again.
inline bool rys_dd_ds_shell_batch_enabled() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("TLNO_RYS_DD_DS_BATCH");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

inline bool rys_dd_pp_shell_batch_enabled() noexcept {
    static const bool enabled = []() {
        const char* value = std::getenv("TLNO_RYS_DD_PP_BATCH");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
    }();
    return enabled;
}

template <class Emit>
inline void accumulate_rys_dd_low_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    accumulate_rys_dd_low_spherical_shell_batches<0, Emit>(bd, eri_values);
    accumulate_rys_dd_low_spherical_shell_batches<1, Emit>(bd, eri_values);
    if (rys_dd_pp_shell_batch_enabled()) {
        accumulate_rys_dd_low_spherical_shell_batches<2, Emit>(bd, eri_values);
    }
    if (rys_dd_ds_shell_batch_enabled()) {
        accumulate_rys_dd_low_spherical_shell_batches<3, Emit>(bd, eri_values);
    }
}

// Root-record fragmentation has a fixed setup cost, just like dp|{ds,pp}.
// Keep tiny molecules on the compact compiled-quartet path while enabling the
// cross-shellset kernel once there is enough dd x {ss,ps} work to amortize it.
// MEASURED 2026-08-02 and CLOSED.  Switches were instrumented to send this
// whole family to the generic route and measure specialized vs generic on
// exactly the same integrals.  Result: the family gains only
// 1.00x-1.06x (c9_grid/def2svp 1.02x, isoleucine/631gdp 1.04x,
// benzene/def2tzvp 1.02x); the new dd|pp and dd|ds classes measured 0.99x-1.03x.
// The cause is that the "generic" route is ALREADY specialized at compile
// time for each (La,Lb,Lc,Ld), so there is no slow fallback to rescue
// classes from.  The switches were removed: they cost a branch in the
// production dispatch and added nothing after the decision.  Do NOT
// reintroduce them expecting gains from specializing more classes: none exist.

inline bool rys_dd_low_shell_batch_profitable(
    const BasisDataHGP& bd) noexcept {
    std::uint64_t dd_count = 0;
    std::uint64_t low_count = 0;
    for (std::size_t i = 1; i < bd.rys_dd_pair_bins.size(); ++i) {
        dd_count += bd.rys_dd_pair_bins[i].pair_count;
    }
    for (int other_l = 0; other_l <= 1; ++other_l) {
        const auto& bins =
            bd.rys_low_pair_bins[static_cast<std::size_t>(other_l)];
        for (std::size_t i = 1; i < bins.size(); ++i) {
            low_count += bins[i].pair_count;
        }
    }
    constexpr std::uint64_t kMinimumShellPairProducts = 256;
    return dd_count != 0 &&
           low_count >=
               (kMinimumShellPairProducts + dd_count - 1) / dd_count;
}

// Cross-shellset pure (ds|ds).  Harmonicity removes B10/B01 traces.  The
// normalized d Hessians are mutually orthogonal with squared norm two, so the
// second cross-contraction reduces to delta(m,n)*B00^2.
template <bool Screened, class Emit>
inline void accumulate_rys_dsds_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    alignas(32) double targ4[4];
    alignas(32) double roots12[4 * ActiveRoots<Emit, 3>];
    alignas(32) double weights12[4 * ActiveRoots<Emit, 3>];
    alignas(32) double raw[25][4];
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    for (std::size_t ket_pos = 0; ket_pos < ket_bin.pair_count; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_bin.pair_count;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_bin.pair_count - bra_pos);
            if (same_bin && ket_pos >= bra_pos + valid_lanes) continue;
            const unsigned logical_mask = rys_sp_shell_logical_bra4(
                bra_pos, valid_lanes, ket_pos, same_bin);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask) & logical_mask;
                monotone_mask &= logical_mask;
                if (active_mask == 0u) {
                    // Only partial chunks instantiate Screened=true; they are
                    // metric-descending, so an empty monotone mask proves the
                    // rest of this bra tail cannot pass for the fixed ket.
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : logical_mask;
            __m256d acc[25];
            for (auto& value : acc) value = _mm256_setzero_pd();

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPA[3] = {
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff),
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff),
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff),
                };
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(one, vpq);
                    const __m256d vd[3] = {
                        _mm256_sub_pd(vPx, _mm256_set1_pd(ket_bin.Wx[koff])),
                        _mm256_sub_pd(vPy, _mm256_set1_pd(ket_bin.Wy[koff])),
                        _mm256_sub_pd(vPz, _mm256_set1_pd(ket_bin.Wz[koff])),
                    };
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vd[0], vd[0],
                        _mm256_fmadd_pd(vd[1], vd[1],
                                        _mm256_mul_pd(vd[2], vd[2])));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                            vr2));
                    rys_roots_weights_batch4_for_emit<Emit, 3>(
                        targ4, roots12, weights12);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    const __m256d qop = _mm256_mul_pd(vq, vinvpq);
                    const __m256d pop = _mm256_mul_pd(vp, vinvpq);
                    for (int root = 0; root < ActiveRoots<Emit, 3>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots12 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights12 + 4 * root));
                        const __m256d qscale = _mm256_mul_pd(qop, vrt);
                        const __m256d pscale = _mm256_mul_pd(pop, vrt);
                        const __m256d C[3] = {
                            _mm256_fnmadd_pd(qscale, vd[0], vPA[0]),
                            _mm256_fnmadd_pd(qscale, vd[1], vPA[1]),
                            _mm256_fnmadd_pd(qscale, vd[2], vPA[2]),
                        };
                        const __m256d D[3] = {
                            _mm256_fmadd_pd(
                                pscale, vd[0],
                                _mm256_set1_pd(ket_bin.Lx[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[1],
                                _mm256_set1_pd(ket_bin.Ly[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[2],
                                _mm256_set1_pd(ket_bin.Lz[koff])),
                        };
                        const __m256d B00 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvpq), vrt);
                        const __m256d B00sq = _mm256_mul_pd(B00, B00);
                        __m256d HC[5], HD[5], dHC[5][3], dHD[5][3];
                        rys_d_solid_harmonics_v(C, HC, dHC);
                        rys_d_solid_harmonics_v(D, HD, dHD);
                        for (int a = 0; a < 5; ++a) {
                            for (int b = 0; b < 5; ++b) {
                                __m256d value = _mm256_mul_pd(HC[a], HD[b]);
                                __m256d dot = _mm256_mul_pd(dHC[a][0], dHD[b][0]);
                                dot = _mm256_fmadd_pd(
                                    dHC[a][1], dHD[b][1], dot);
                                dot = _mm256_fmadd_pd(
                                    dHC[a][2], dHD[b][2], dot);
                                value = _mm256_fmadd_pd(B00, dot, value);
                                if (a == b) value = _mm256_add_pd(value, B00sq);
                                const int component = 5 * a + b;
                                acc[component] = _mm256_fmadd_pd(
                                    vww, value, acc[component]);
                            }
                        }
                    }
                }
            }

            for (int component = 0; component < 25; ++component) {
                _mm256_store_pd(raw[component], acc[component]);
            }
            const auto& rhs_refs =
                bd.packed_spherical_pair_components[static_cast<std::size_t>(ket_group)];
            for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                if (same_bin && ket_pos > bra_pos + lane) continue;
                if constexpr (Screened) {
                    if ((active_mask &
                         (1u << static_cast<unsigned>(lane))) == 0u) continue;
                }
                const std::uint32_t bra_group =
                    bra_bin.pair_group_ids[bra_pos + lane];
                const auto& lhs_refs =
                    bd.packed_spherical_pair_components[static_cast<std::size_t>(bra_group)];
                const bool same_pair_group = bra_group == ket_group;
                rys_emit_spherical_lane_block<Emit>(
                    eri_values, lhs_refs, rhs_refs, same_pair_group,
                    [&raw, lane](const PackedPairComponentH& lhs_ref,
                                 const PackedPairComponentH& rhs_ref) noexcept {
                        const int component =
                            5 * static_cast<int>(lhs_ref.component_index) +
                            static_cast<int>(rhs_ref.component_index);
                        return raw[component][lane];
                    });
            }
        }
    }
}

// Cross-shellset pure (ds|pp).  The 5x9 final block is evaluated as a fixed
// small matrix product.  Harmonic traces again cancel, leaving only first and
// second cross derivatives in addition to the pp horizontal-transfer term.
template <bool BraDS, bool Screened, class Emit>
inline void accumulate_rys_dspp_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    alignas(32) double targ4[4];
    alignas(32) double roots12[4 * ActiveRoots<Emit, 3>];
    alignas(32) double weights12[4 * ActiveRoots<Emit, 3>];
    alignas(32) double raw[45][4];
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d invsqrt3 = _mm256_set1_pd(kRysDSolidInvSqrt3);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const __m256d ddH[5][3][3] = {
        {{zero, one, zero}, {one, zero, zero}, {zero, zero, zero}},
        {{zero, zero, zero}, {zero, zero, one}, {zero, one, zero}},
        {{_mm256_sub_pd(zero, invsqrt3), zero, zero},
         {zero, _mm256_sub_pd(zero, invsqrt3), zero},
         {zero, zero, _mm256_mul_pd(two, invsqrt3)}},
        {{zero, zero, one}, {zero, zero, zero}, {one, zero, zero}},
        {{one, zero, zero}, {zero, _mm256_sub_pd(zero, one), zero},
         {zero, zero, zero}},
    };
    const std::size_t bra_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(ket_bin);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_end; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_end;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_end - bra_pos);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask);
                if (active_mask == 0u) {
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : rys_low_bits(valid_lanes);
            __m256d acc[45];
            for (auto& value : acc) value = zero;
            const __m256d bra_R[3] = {
                _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos),
            };
            const __m256d ket_R[3] = {
                _mm256_set1_pd(ket_bin.Rx[ket_pos]),
                _mm256_set1_pd(ket_bin.Ry[ket_pos]),
                _mm256_set1_pd(ket_bin.Rz[ket_pos]),
            };

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vinvp =
                    _mm256_loadu_pd(bra_bin.invs.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPA[3] = {
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff),
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff),
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff),
                };
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vinvq = _mm256_set1_pd(ket_bin.invs[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(one, vpq);
                    const __m256d vd[3] = {
                        _mm256_sub_pd(vPx, _mm256_set1_pd(ket_bin.Wx[koff])),
                        _mm256_sub_pd(vPy, _mm256_set1_pd(ket_bin.Wy[koff])),
                        _mm256_sub_pd(vPz, _mm256_set1_pd(ket_bin.Wz[koff])),
                    };
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vd[0], vd[0],
                        _mm256_fmadd_pd(vd[1], vd[1],
                                        _mm256_mul_pd(vd[2], vd[2])));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                            vr2));
                    rys_roots_weights_batch4_for_emit<Emit, 3>(
                        targ4, roots12, weights12);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    const __m256d qop = _mm256_mul_pd(vq, vinvpq);
                    const __m256d pop = _mm256_mul_pd(vp, vinvpq);
                    for (int root = 0; root < ActiveRoots<Emit, 3>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots12 + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights12 + 4 * root));
                        const __m256d qscale = _mm256_mul_pd(qop, vrt);
                        const __m256d pscale = _mm256_mul_pd(pop, vrt);
                        const __m256d C[3] = {
                            _mm256_fnmadd_pd(qscale, vd[0], vPA[0]),
                            _mm256_fnmadd_pd(qscale, vd[1], vPA[1]),
                            _mm256_fnmadd_pd(qscale, vd[2], vPA[2]),
                        };
                        const __m256d D[3] = {
                            _mm256_fmadd_pd(
                                pscale, vd[0],
                                _mm256_set1_pd(ket_bin.Lx[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[1],
                                _mm256_set1_pd(ket_bin.Ly[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[2],
                                _mm256_set1_pd(ket_bin.Lz[koff])),
                        };
                        const __m256d B00 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvpq), vrt);
                        const __m256d B00sq = _mm256_mul_pd(B00, B00);
                        const __m256d B10 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvp),
                            _mm256_fnmadd_pd(qop, vrt, one));
                        const __m256d B01 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvq),
                            _mm256_fnmadd_pd(pop, vrt, one));
                        const __m256d* const Xd = BraDS ? C : D;
                        const __m256d* const Xpp = BraDS ? D : C;
                        const __m256d* const Rpp = BraDS ? ket_R : bra_R;
                        const __m256d Bside = BraDS ? B01 : B10;
                        __m256d H[5], dH[5][3];
                        rys_d_solid_harmonics_v(Xd, H, dH);
                        for (int dcomp = 0; dcomp < 5; ++dcomp) {
                            for (int pi = 0; pi < 3; ++pi) {
                                for (int pj = 0; pj < 3; ++pj) {
                                    const __m256d shifted =
                                        _mm256_add_pd(Xpp[pj], Rpp[pj]);
                                    __m256d pp =
                                        _mm256_mul_pd(Xpp[pi], shifted);
                                    if (pi == pj) {
                                        pp = _mm256_add_pd(pp, Bside);
                                    }
                                    __m256d first = _mm256_mul_pd(
                                        dH[dcomp][pi], shifted);
                                    first = _mm256_fmadd_pd(
                                        dH[dcomp][pj], Xpp[pi], first);
                                    __m256d value = _mm256_fmadd_pd(
                                        B00, first,
                                        _mm256_mul_pd(H[dcomp], pp));
                                    value = _mm256_fmadd_pd(
                                        B00sq, ddH[dcomp][pi][pj], value);
                                    const int component =
                                        9 * dcomp + 3 * pi + pj;
                                    acc[component] = _mm256_fmadd_pd(
                                        vww, value, acc[component]);
                                }
                            }
                        }
                    }
                }
            }

            for (int component = 0; component < 45; ++component) {
                _mm256_store_pd(raw[component], acc[component]);
            }
            const auto& rhs_refs =
                bd.packed_spherical_pair_components[static_cast<std::size_t>(ket_group)];
            for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                if constexpr (Screened) {
                    if ((active_mask &
                         (1u << static_cast<unsigned>(lane))) == 0u) continue;
                }
                const std::uint32_t bra_group =
                    bra_bin.pair_group_ids[bra_pos + lane];
                const auto& lhs_refs =
                    bd.packed_spherical_pair_components[static_cast<std::size_t>(bra_group)];
                rys_emit_spherical_lane_block<Emit>(
                    eri_values, lhs_refs, rhs_refs, false,
                    [&raw, lane](const PackedPairComponentH& lhs_ref,
                                 const PackedPairComponentH& rhs_ref) noexcept {
                        const int d_component = BraDS
                            ? static_cast<int>(lhs_ref.component_index)
                            : static_cast<int>(rhs_ref.component_index);
                        const int pp_component = BraDS
                            ? static_cast<int>(rhs_ref.component_index)
                            : static_cast<int>(lhs_ref.component_index);
                        const int component = 9 * d_component + pp_component;
                        return raw[component][lane];
                    });
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
inline void accumulate_rys_ds_matrix_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& ds_bins = bd.rys_ds_pair_bins;
    for (std::size_t i = 1; i < ds_bins.size(); ++i) {
        if (ds_bins[i].pair_count == 0) continue;
        for (std::size_t j = 1; j <= i; ++j) {
            if (ds_bins[j].pair_count == 0) continue;
            if (i == j) {
                accumulate_rys_dsds_spherical_shell_bin_pair<false, Emit>(
                    bd, ds_bins[i], ds_bins[j], true, eri_values);
            } else {
                const RysLowPairBinH* bra = &ds_bins[i];
                const RysLowPairBinH* ket = &ds_bins[j];
                if (ket->pair_count > bra->pair_count) std::swap(bra, ket);
                accumulate_rys_dsds_spherical_shell_bin_pair<false, Emit>(
                    bd, *bra, *ket, false, eri_values);
            }
        }
    }

    const auto& pp_bins = bd.rys_low_pair_bins[2];
    for (std::size_t di = 1; di < ds_bins.size(); ++di) {
        if (ds_bins[di].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < pp_bins.size(); ++pi) {
            if (pp_bins[pi].pair_count == 0) continue;
            if (ds_bins[di].pair_count >= pp_bins[pi].pair_count) {
                accumulate_rys_dspp_spherical_shell_bin_pair<true, false, Emit>(
                    bd, ds_bins[di], pp_bins[pi], eri_values);
            } else {
                accumulate_rys_dspp_spherical_shell_bin_pair<false, false, Emit>(
                    bd, pp_bins[pi], ds_bins[di], eri_values);
            }
        }
    }
}

#ifdef TLNO_RYS_DIRECT_EXCHANGE
// Close dp|ps with two output-stationary contractions.  Re-reading the 45
// lane values keeps only four carries live while reducing 180 scalar K RMWs
// to 32 commits.  Both shell pairs are necessarily AO-off-diagonal, hence the
// exact positive-K factor is 2 (their RysJkPairInfo scales are both one).
template <bool BraDP>
__attribute__((noinline, noclone)) void rys_k_dpps_preemit_packet_v1(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    unsigned active_bits,
    const __m256d (&acc)[45]) noexcept {
    const std::size_t M = context.M;
    const double* const p = context.p;
    double* const k = context.g;
    for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t moving_group = moving_bin.pair_group_ids[
            moving_pos + lane];
        const std::uint32_t dp_group = BraDP ? moving_group : fixed_group;
        const std::uint32_t ps_group = BraDP ? fixed_group : moving_group;
        const auto [d_shell, dp_p_shell] =
            bd.pair_groups[static_cast<std::size_t>(dp_group)];
        const auto [ps_p_shell, s_shell] =
            bd.pair_groups[static_cast<std::size_t>(ps_group)];
#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
        assert(bd.groups[static_cast<std::size_t>(d_shell)].L == 2);
        assert(bd.groups[static_cast<std::size_t>(dp_p_shell)].L == 1);
        assert(bd.groups[static_cast<std::size_t>(ps_p_shell)].L == 1);
        assert(bd.groups[static_cast<std::size_t>(s_shell)].L == 0);
#endif
        const int d0 = bd.spherical_group_bases[
            static_cast<std::size_t>(d_shell)];
        const int p0 = bd.spherical_group_bases[
            static_cast<std::size_t>(dp_p_shell)];
        const int c0 = bd.spherical_group_bases[
            static_cast<std::size_t>(ps_p_shell)];
        const int z = bd.spherical_group_bases[
            static_cast<std::size_t>(s_shell)];

#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
        if constexpr (!BraDP) {
            // The generic reversed emitter traverses lhs ps components before
            // rhs dp components.  Reproduce its pk -> di -> pj ledger order in
            // diagnostics only; this entire prepass is absent from release.
            for (int pk = 0; pk < 3; ++pk) {
                const std::uint32_t ps_pair =
                    EriPackedGAccessor::pair_index(c0 + pk, z);
                for (int di = 0; di < 5; ++di) {
                    for (int pj = 0; pj < 3; ++pj) {
                        const int component = 9 * di + 3 * pj + pk;
                        const std::uint32_t dp_pair =
                            EriPackedGAccessor::pair_index(
                                d0 + di, p0 + pj);
                        rys_jk_record_one(
                            context, ps_pair, dp_pair,
                            acc[component][lane]);
                    }
                }
            }
        }
#endif

        // First contraction pass: retain only the three D-C destinations and
        // D-Z for one d row.  BraDP records here in its exact producer order
        // di -> pj -> pk; reversed diagnostics used the prepass above.
        for (int di = 0; di < 5; ++di) {
            double dc[3] = {};
            double dz = 0.0;
            for (int pj = 0; pj < 3; ++pj) {
                for (int pk = 0; pk < 3; ++pk) {
                    const int component = 9 * di + 3 * pj + pk;
                    const double value = acc[component][lane];
                    const std::uint32_t dp_pair =
                        EriPackedGAccessor::pair_index(d0 + di, p0 + pj);
                    const std::uint32_t ps_pair =
                        EriPackedGAccessor::pair_index(c0 + pk, z);
                    if constexpr (BraDP) {
                        rys_jk_record_one(
                            context, dp_pair, ps_pair, value);
                    }
                    const double q = 2.0 * value;
                    if constexpr (BraDP) {
                        const std::size_t pr =
                            static_cast<std::size_t>(p0 + pj) * M;
                        dc[pk] += q * p[pr + static_cast<std::size_t>(z)];
                        dz += q * p[
                            pr + static_cast<std::size_t>(c0 + pk)];
                    } else {
                        const std::size_t zr =
                            static_cast<std::size_t>(z) * M;
                        const std::size_t cr =
                            static_cast<std::size_t>(c0 + pk) * M;
                        dc[pk] += q * p[
                            zr + static_cast<std::size_t>(p0 + pj)];
                        dz += q * p[
                            cr + static_cast<std::size_t>(p0 + pj)];
                    }
                }
            }
            if constexpr (BraDP) {
                double* const drow =
                    k + static_cast<std::size_t>(d0 + di) * M;
                for (int pk = 0; pk < 3; ++pk)
                    drow[c0 + pk] += dc[pk];
                drow[z] += dz;
            } else {
                for (int pk = 0; pk < 3; ++pk)
                    k[static_cast<std::size_t>(c0 + pk) * M + d0 + di] +=
                        dc[pk];
                k[static_cast<std::size_t>(z) * M + d0 + di] += dz;
            }
        }

        // Second pass: reread acc and close one P-C row plus P-Z at a time.
        // If both p shells coincide, all nine directed p-p destinations remain
        // distinct here; do not triangularize or apply diagonal half scaling.
        for (int pj = 0; pj < 3; ++pj) {
            double pc[3] = {};
            double pz = 0.0;
            for (int di = 0; di < 5; ++di) {
                for (int pk = 0; pk < 3; ++pk) {
                    const int component = 9 * di + 3 * pj + pk;
                    const double q = 2.0 * acc[component][lane];
                    if constexpr (BraDP) {
                        const std::size_t dr =
                            static_cast<std::size_t>(d0 + di) * M;
                        pc[pk] += q * p[dr + static_cast<std::size_t>(z)];
                        pz += q * p[
                            dr + static_cast<std::size_t>(c0 + pk)];
                    } else {
                        const std::size_t zr =
                            static_cast<std::size_t>(z) * M;
                        const std::size_t cr =
                            static_cast<std::size_t>(c0 + pk) * M;
                        pc[pk] += q * p[
                            zr + static_cast<std::size_t>(d0 + di)];
                        pz += q * p[
                            cr + static_cast<std::size_t>(d0 + di)];
                    }
                }
            }
            if constexpr (BraDP) {
                double* const prow =
                    k + static_cast<std::size_t>(p0 + pj) * M;
                for (int pk = 0; pk < 3; ++pk)
                    prow[c0 + pk] += pc[pk];
                prow[z] += pz;
            } else {
                for (int pk = 0; pk < 3; ++pk)
                    k[static_cast<std::size_t>(c0 + pk) * M + p0 + pj] +=
                        pc[pk];
                k[static_cast<std::size_t>(z) * M + p0 + pj] += pz;
            }
        }
    }
}
#endif

// Cross-shellset (dp|ss)/(dp|ps), including reversed pair orientations.
// The final pure (d,p) shell-pair polynomial is formed directly after each
// root and contracted into 15/45 outputs; no Cartesian 18-column temporary or
// runtime sparse transform is needed.
template <int OtherL, bool BraDP, bool Screened, class Emit>
inline void accumulate_rys_dp_low_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    static_assert(OtherL == 0 || OtherL == 1, "dp shell batch supports ss/ps");
    constexpr int NRoots = (3 + OtherL) / 2 + 1;
    constexpr int NOut = 15 * (OtherL == 0 ? 1 : 3);
    alignas(32) double targ4[4];
    alignas(32) double roots[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double weights[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double raw[NOut][4];
    const __m256d zero = _mm256_setzero_pd();
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d invsqrt3 = _mm256_set1_pd(kRysDSolidInvSqrt3);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const __m256d ddH[5][3][3] = {
        {{zero, one, zero}, {one, zero, zero}, {zero, zero, zero}},
        {{zero, zero, zero}, {zero, zero, one}, {zero, one, zero}},
        {{_mm256_sub_pd(zero, invsqrt3), zero, zero},
         {zero, _mm256_sub_pd(zero, invsqrt3), zero},
         {zero, zero, _mm256_mul_pd(two, invsqrt3)}},
        {{zero, zero, one}, {zero, zero, zero}, {one, zero, zero}},
        {{one, zero, zero}, {zero, _mm256_sub_pd(zero, one), zero},
         {zero, zero, zero}},
    };
    const std::size_t bra_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(ket_bin);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_end; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_end;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_end - bra_pos);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask);
                if (active_mask == 0u) {
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : rys_low_bits(valid_lanes);
            __m256d acc[NOut];
            for (auto& value : acc) value = zero;
            const __m256d bra_R[3] = {
                _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos),
            };
            const __m256d ket_R[3] = {
                _mm256_set1_pd(ket_bin.Rx[ket_pos]),
                _mm256_set1_pd(ket_bin.Ry[ket_pos]),
                _mm256_set1_pd(ket_bin.Rz[ket_pos]),
            };

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vinvp =
                    _mm256_loadu_pd(bra_bin.invs.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPA[3] = {
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff),
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff),
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff),
                };
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vinvq = _mm256_set1_pd(ket_bin.invs[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(one, vpq);
                    const __m256d vd[3] = {
                        _mm256_sub_pd(vPx, _mm256_set1_pd(ket_bin.Wx[koff])),
                        _mm256_sub_pd(vPy, _mm256_set1_pd(ket_bin.Wy[koff])),
                        _mm256_sub_pd(vPz, _mm256_set1_pd(ket_bin.Wz[koff])),
                    };
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vd[0], vd[0],
                        _mm256_fmadd_pd(vd[1], vd[1],
                                        _mm256_mul_pd(vd[2], vd[2])));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                            vr2));
                    rys_roots_weights_batch4_for_emit<Emit, NRoots>(
                        targ4, roots, weights);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    const __m256d qop = _mm256_mul_pd(vq, vinvpq);
                    const __m256d pop = _mm256_mul_pd(vp, vinvpq);
                    for (int root = 0;
                         root < ActiveRoots<Emit, NRoots>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights + 4 * root));
                        const __m256d qscale = _mm256_mul_pd(qop, vrt);
                        const __m256d pscale = _mm256_mul_pd(pop, vrt);
                        const __m256d C[3] = {
                            _mm256_fnmadd_pd(qscale, vd[0], vPA[0]),
                            _mm256_fnmadd_pd(qscale, vd[1], vPA[1]),
                            _mm256_fnmadd_pd(qscale, vd[2], vPA[2]),
                        };
                        const __m256d D[3] = {
                            _mm256_fmadd_pd(
                                pscale, vd[0],
                                _mm256_set1_pd(ket_bin.Lx[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[1],
                                _mm256_set1_pd(ket_bin.Ly[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[2],
                                _mm256_set1_pd(ket_bin.Lz[koff])),
                        };
                        const __m256d B00 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvpq), vrt);
                        const __m256d B10 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvp),
                            _mm256_fnmadd_pd(qop, vrt, one));
                        const __m256d B01 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvq),
                            _mm256_fnmadd_pd(pop, vrt, one));
                        const __m256d* const Xdp = BraDP ? C : D;
                        const __m256d* const Xlow = BraDP ? D : C;
                        const __m256d* const Rdp = BraDP ? bra_R : ket_R;
                        const __m256d Bside = BraDP ? B10 : B01;
                        __m256d H[5], dH[5][3];
                        rys_d_solid_harmonics_v(Xdp, H, dH);

                        for (int dcomp = 0; dcomp < 5; ++dcomp) {
                            for (int pair_p = 0; pair_p < 3; ++pair_p) {
                                const __m256d shifted =
                                    _mm256_add_pd(Xdp[pair_p], Rdp[pair_p]);
                                const __m256d F = _mm256_fmadd_pd(
                                    Bside, dH[dcomp][pair_p],
                                    _mm256_mul_pd(H[dcomp], shifted));
                                const int dp_component = 3 * dcomp + pair_p;
                                if constexpr (OtherL == 0) {
                                    acc[dp_component] = _mm256_fmadd_pd(
                                        vww, F, acc[dp_component]);
                                } else {
                                    for (int low_p = 0; low_p < 3; ++low_p) {
                                        __m256d dF = _mm256_fmadd_pd(
                                            Bside, ddH[dcomp][pair_p][low_p],
                                            _mm256_mul_pd(
                                                dH[dcomp][low_p], shifted));
                                        if (pair_p == low_p) {
                                            dF = _mm256_add_pd(dF, H[dcomp]);
                                        }
                                        const __m256d value = _mm256_fmadd_pd(
                                            B00, dF,
                                            _mm256_mul_pd(F, Xlow[low_p]));
                                        const int component =
                                            3 * dp_component + low_p;
                                        acc[component] = _mm256_fmadd_pd(
                                            vww, value, acc[component]);
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if constexpr (
                OtherL == 1 && rys_emit_uses_preemit_dpps_positive_k_v<Emit>) {
#ifdef TLNO_RYS_DIRECT_EXCHANGE
                const unsigned active_bits = Screened
                    ? active_mask : rys_low_bits(valid_lanes);
                if (active_bits != 0u) {
                    rys_k_dpps_preemit_packet_v1<BraDP>(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, active_bits, acc);
                }
#endif
            } else {
                for (int component = 0; component < NOut; ++component)
                    _mm256_store_pd(raw[component], acc[component]);
                const auto& rhs_refs =
                    bd.packed_spherical_pair_components[static_cast<std::size_t>(ket_group)];
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((active_mask &
                             (1u << static_cast<unsigned>(lane))) == 0u) continue;
                    }
                    const std::uint32_t bra_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& lhs_refs =
                        bd.packed_spherical_pair_components[static_cast<std::size_t>(bra_group)];
                    rys_emit_spherical_lane_block<Emit>(
                        eri_values, lhs_refs, rhs_refs, false,
                        [&raw, lane](const PackedPairComponentH& lhs_ref,
                                     const PackedPairComponentH& rhs_ref) noexcept {
                            const int dp_component = BraDP
                                ? static_cast<int>(lhs_ref.component_index)
                                : static_cast<int>(rhs_ref.component_index);
                            int component = dp_component;
                            if constexpr (OtherL == 1) {
                                const int low_component = BraDP
                                    ? static_cast<int>(rhs_ref.component_index)
                                    : static_cast<int>(lhs_ref.component_index);
                                component = 3 * dp_component + low_component;
                            }
                            return raw[component][lane];
                        });
                }
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <int OtherL, class Emit>
inline void accumulate_rys_dp_low_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& dp_bins = bd.rys_dp_pair_bins;
    const auto& low_bins = bd.rys_low_pair_bins[static_cast<std::size_t>(OtherL)];
    for (std::size_t di = 1; di < dp_bins.size(); ++di) {
        if (dp_bins[di].pair_count == 0) continue;
        for (std::size_t li = 1; li < low_bins.size(); ++li) {
            if (low_bins[li].pair_count == 0) continue;
            if (dp_bins[di].pair_count >= low_bins[li].pair_count) {
                accumulate_rys_dp_low_spherical_shell_bin_pair<OtherL, true, false, Emit>(
                    bd, dp_bins[di], low_bins[li], eri_values);
            } else {
                accumulate_rys_dp_low_spherical_shell_bin_pair<OtherL, false, false, Emit>(
                    bd, low_bins[li], dp_bins[di], eri_values);
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
inline void accumulate_rys_dp_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    accumulate_rys_dp_low_spherical_shell_batches<0, Emit>(bd, eri_values);
    accumulate_rys_dp_low_spherical_shell_batches<1, Emit>(bd, eri_values);
}

// Root-level data shared by the five output fragments of dp|ds and dp|pp.
// Geometry and contractions remain dynamic, while the angular pattern, root
// count, fragment shape, and all derivative structure are compile-time fixed.
struct alignas(32) RysDPMatrixRootRecordV {
    __m256d weight;
    __m256d xdp[3];
    __m256d xother[3];
    __m256d b00;
    __m256d b00sq;
    __m256d bdp;
    __m256d bother;
};

template <int DComponent, bool OtherPP>
inline void contract_rys_dp_matrix_fragment(
    const std::vector<RysDPMatrixRootRecordV>& records,
    std::size_t active_record_count,
    const __m256d rdp[3],
    const __m256d rother[3],
    double* __restrict raw) noexcept {
    constexpr int NOther = OtherPP ? 9 : 5;
    constexpr int NFragment = 3 * NOther;
    __m256d acc[NFragment];
    for (auto& value : acc) value = _mm256_setzero_pd();

    for (std::size_t record_pos = 0;
         record_pos < active_record_count; ++record_pos) {
        const RysDPMatrixRootRecordV& record = records[record_pos];
        __m256d hd, dhd[3];
        rys_d_solid_harmonic_component_v<DComponent>(record.xdp, hd, dhd);
        __m256d hother[5], dhother[5][3];
        if constexpr (!OtherPP) {
            rys_d_solid_harmonics_v(record.xother, hother, dhother);
        }

        for (int pair_p = 0; pair_p < 3; ++pair_p) {
            const __m256d shifted_dp =
                _mm256_add_pd(record.xdp[pair_p], rdp[pair_p]);
            const __m256d f = _mm256_fmadd_pd(
                record.bdp, dhd[pair_p], _mm256_mul_pd(hd, shifted_dp));
            __m256d df[3];
            for (int axis = 0; axis < 3; ++axis) {
                df[axis] = _mm256_fmadd_pd(
                    record.bdp,
                    _mm256_set1_pd(
                        kRysDSolidHessian[DComponent][pair_p][axis]),
                    _mm256_mul_pd(dhd[axis], shifted_dp));
                if (axis == pair_p) df[axis] = _mm256_add_pd(df[axis], hd);
            }

            if constexpr (!OtherPP) {
                for (int other_d = 0; other_d < 5; ++other_d) {
                    __m256d dot = _mm256_mul_pd(df[0], dhother[other_d][0]);
                    dot = _mm256_fmadd_pd(df[1], dhother[other_d][1], dot);
                    dot = _mm256_fmadd_pd(df[2], dhother[other_d][2], dot);
                    __m256d second = DComponent == other_d
                        ? shifted_dp : _mm256_setzero_pd();
                    for (int axis = 0; axis < 3; ++axis) {
                        const double coefficient =
                            kRysDSolidHessian[other_d][axis][pair_p];
                        if (coefficient != 0.0) {
                            second = _mm256_fmadd_pd(
                                _mm256_set1_pd(coefficient), dhd[axis], second);
                        }
                    }
                    __m256d value = _mm256_fmadd_pd(
                        record.b00, dot, _mm256_mul_pd(f, hother[other_d]));
                    value = _mm256_fmadd_pd(record.b00sq, second, value);
                    const int local = pair_p * NOther + other_d;
                    acc[local] = _mm256_fmadd_pd(
                        record.weight, value, acc[local]);
                }
            } else {
                for (int pi = 0; pi < 3; ++pi) {
                    for (int pj = 0; pj < 3; ++pj) {
                        const __m256d shifted_other =
                            _mm256_add_pd(record.xother[pj], rother[pj]);
                        __m256d pp = _mm256_mul_pd(
                            record.xother[pi], shifted_other);
                        if (pi == pj) pp = _mm256_add_pd(pp, record.bother);
                        __m256d first = _mm256_mul_pd(df[pi], shifted_other);
                        first = _mm256_fmadd_pd(
                            df[pj], record.xother[pi], first);
                        __m256d ddf = _mm256_mul_pd(
                            _mm256_set1_pd(
                                kRysDSolidHessian[DComponent][pi][pj]),
                            shifted_dp);
                        if (pj == pair_p) ddf = _mm256_add_pd(ddf, dhd[pi]);
                        if (pi == pair_p) ddf = _mm256_add_pd(ddf, dhd[pj]);
                        __m256d value = _mm256_fmadd_pd(
                            record.b00, first, _mm256_mul_pd(f, pp));
                        value = _mm256_fmadd_pd(record.b00sq, ddf, value);
                        const int local = pair_p * NOther + 3 * pi + pj;
                        acc[local] = _mm256_fmadd_pd(
                            record.weight, value, acc[local]);
                    }
                }
            }
        }
    }

    for (int pair_p = 0; pair_p < 3; ++pair_p) {
        for (int other = 0; other < NOther; ++other) {
            const int local = pair_p * NOther + other;
            const int component =
                ((3 * DComponent + pair_p) * NOther + other) * 4;
            _mm256_store_pd(raw + component, acc[local]);
        }
    }
}

#ifdef TLNO_RYS_DIRECT_EXCHANGE
// Digest the already-materialized pure dp|pp block with two small
// output-stationary contractions. The fragmented producer still owns raw135.
template <bool BraDP>
__attribute__((noinline, noclone)) void rys_k_dppp_preemit_packet_v1(
    const RysJkEmitContext& context, const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin, std::size_t moving_pos,
    std::uint32_t fixed_group, unsigned active_bits,
    const double (&raw)[135][4]) noexcept {
    const std::size_t M = context.M;
    const double* const p = context.p;
    double* const k = context.g;
    for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t moving_group =
            moving_bin.pair_group_ids[moving_pos + lane];
        const std::uint32_t dp_group = BraDP ? moving_group : fixed_group;
        const std::uint32_t pp_group = BraDP ? fixed_group : moving_group;
        const auto [d_shell, dp_p_shell] =
            bd.pair_groups[static_cast<std::size_t>(dp_group)];
        const auto [a_shell, b_shell] =
            bd.pair_groups[static_cast<std::size_t>(pp_group)];
        const bool same_pp = a_shell == b_shell;
#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
        assert(bd.groups[static_cast<std::size_t>(d_shell)].L == 2);
        assert(bd.groups[static_cast<std::size_t>(dp_p_shell)].L == 1);
        assert(bd.groups[static_cast<std::size_t>(a_shell)].L == 1);
        assert(bd.groups[static_cast<std::size_t>(b_shell)].L == 1);
#endif
        const int d0 = bd.spherical_group_bases[
            static_cast<std::size_t>(d_shell)];
        const int p0 = bd.spherical_group_bases[
            static_cast<std::size_t>(dp_p_shell)];
        const int a0 = bd.spherical_group_bases[
            static_cast<std::size_t>(a_shell)];
        const int b0 = bd.spherical_group_bases[
            static_cast<std::size_t>(b_shell)];

#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
        if constexpr (!BraDP) {
            // Reversed fallback traverses lhs pp before rhs dp. This ledger
            // prepass is absent from release.
            for (int pa = 0; pa < 3; ++pa) {
                const int pb_limit = same_pp ? pa + 1 : 3;
                for (int pb = 0; pb < pb_limit; ++pb) {
                    const std::uint32_t pp_pair =
                        EriPackedGAccessor::pair_index(a0 + pa, b0 + pb);
                    for (int di = 0; di < 5; ++di) {
                        for (int pj = 0; pj < 3; ++pj) {
                            const int component =
                                (3 * di + pj) * 9 + 3 * pa + pb;
                            const std::uint32_t dp_pair =
                                EriPackedGAccessor::pair_index(
                                    d0 + di, p0 + pj);
                            rys_jk_record_one(
                                context, pp_pair, dp_pair,
                                raw[component][lane]);
                        }
                    }
                }
            }
        }
#endif

        // D-side pass: six carries for rectangular pp; paired carries combine
        // into three commits when both p shells are identical.
        for (int di = 0; di < 5; ++di) {
            double d_a[3] = {}, d_b[3] = {};
            for (int pj = 0; pj < 3; ++pj) {
                for (int pa = 0; pa < 3; ++pa) {
                    const int pb_limit = same_pp ? pa + 1 : 3;
                    for (int pb = 0; pb < pb_limit; ++pb) {
                        const int component =
                            (3 * di + pj) * 9 + 3 * pa + pb;
                        const double value = raw[component][lane];
                        if constexpr (BraDP) {
                            const std::uint32_t dp_pair =
                                EriPackedGAccessor::pair_index(
                                    d0 + di, p0 + pj);
                            const std::uint32_t pp_pair =
                                EriPackedGAccessor::pair_index(
                                    a0 + pa, b0 + pb);
                            rys_jk_record_one(
                                context, dp_pair, pp_pair, value);
                        }
                        const double q =
                            same_pp && pa == pb ? value : 2.0 * value;
                        if constexpr (BraDP) {
                            const std::size_t pr =
                                static_cast<std::size_t>(p0 + pj) * M;
                            d_a[pa] += q * p[
                                pr + static_cast<std::size_t>(b0 + pb)];
                            d_b[pb] += q * p[
                                pr + static_cast<std::size_t>(a0 + pa)];
                        } else {
                            const std::size_t ar =
                                static_cast<std::size_t>(a0 + pa) * M;
                            const std::size_t br =
                                static_cast<std::size_t>(b0 + pb) * M;
                            d_a[pa] += q * p[
                                br + static_cast<std::size_t>(p0 + pj)];
                            d_b[pb] += q * p[
                                ar + static_cast<std::size_t>(p0 + pj)];
                        }
                    }
                }
            }
            if (same_pp) {
                for (int pa = 0; pa < 3; ++pa) {
                    const double delta = d_a[pa] + d_b[pa];
                    if constexpr (BraDP)
                        k[static_cast<std::size_t>(d0 + di) * M + a0 + pa] +=
                            delta;
                    else
                        k[static_cast<std::size_t>(a0 + pa) * M + d0 + di] +=
                            delta;
                }
            } else {
                for (int pa = 0; pa < 3; ++pa) {
                    if constexpr (BraDP)
                        k[static_cast<std::size_t>(d0 + di) * M + a0 + pa] +=
                            d_a[pa];
                    else
                        k[static_cast<std::size_t>(a0 + pa) * M + d0 + di] +=
                            d_a[pa];
                }
                for (int pb = 0; pb < 3; ++pb) {
                    if constexpr (BraDP)
                        k[static_cast<std::size_t>(d0 + di) * M + b0 + pb] +=
                            d_b[pb];
                    else
                        k[static_cast<std::size_t>(b0 + pb) * M + d0 + di] +=
                            d_b[pb];
                }
            }
        }

        // P-side pass rereads raw so only six carries remain live. Directed
        // p-p destinations are retained if dp P aliases either pp shell.
        for (int pj = 0; pj < 3; ++pj) {
            double p_a[3] = {}, p_b[3] = {};
            for (int di = 0; di < 5; ++di) {
                for (int pa = 0; pa < 3; ++pa) {
                    const int pb_limit = same_pp ? pa + 1 : 3;
                    for (int pb = 0; pb < pb_limit; ++pb) {
                        const int component =
                            (3 * di + pj) * 9 + 3 * pa + pb;
                        const double value = raw[component][lane];
                        const double q =
                            same_pp && pa == pb ? value : 2.0 * value;
                        if constexpr (BraDP) {
                            const std::size_t dr =
                                static_cast<std::size_t>(d0 + di) * M;
                            p_a[pa] += q * p[
                                dr + static_cast<std::size_t>(b0 + pb)];
                            p_b[pb] += q * p[
                                dr + static_cast<std::size_t>(a0 + pa)];
                        } else {
                            const std::size_t ar =
                                static_cast<std::size_t>(a0 + pa) * M;
                            const std::size_t br =
                                static_cast<std::size_t>(b0 + pb) * M;
                            p_a[pa] += q * p[
                                br + static_cast<std::size_t>(d0 + di)];
                            p_b[pb] += q * p[
                                ar + static_cast<std::size_t>(d0 + di)];
                        }
                    }
                }
            }
            if (same_pp) {
                for (int pa = 0; pa < 3; ++pa) {
                    const double delta = p_a[pa] + p_b[pa];
                    if constexpr (BraDP)
                        k[static_cast<std::size_t>(p0 + pj) * M + a0 + pa] +=
                            delta;
                    else
                        k[static_cast<std::size_t>(a0 + pa) * M + p0 + pj] +=
                            delta;
                }
            } else {
                for (int pa = 0; pa < 3; ++pa) {
                    if constexpr (BraDP)
                        k[static_cast<std::size_t>(p0 + pj) * M + a0 + pa] +=
                            p_a[pa];
                    else
                        k[static_cast<std::size_t>(a0 + pa) * M + p0 + pj] +=
                            p_a[pa];
                }
                for (int pb = 0; pb < 3; ++pb) {
                    if constexpr (BraDP)
                        k[static_cast<std::size_t>(p0 + pj) * M + b0 + pb] +=
                            p_b[pb];
                    else
                        k[static_cast<std::size_t>(b0 + pb) * M + p0 + pj] +=
                            p_b[pb];
                }
            }
        }
    }
}

#endif

// Fragmented cross-shellset dp|ds or dp|pp.  Rys roots and geometric
// intermediates are generated once into a compact L1/L2 record stream.  Five
// compile-time d-component fragments then consume that stream with bounded
// live state, mirroring the paper's high-L fragmentation on a CPU SIMD core.
template <bool OtherPP, bool BraDP, bool Screened, class Emit>
inline void accumulate_rys_dp_matrix_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    constexpr int NRoots = 3;
    constexpr int NOther = OtherPP ? 9 : 5;
    constexpr int NOut = 15 * NOther;
    alignas(32) double targ4[4];
    alignas(32) double roots[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double weights[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double raw[NOut][4];
    const __m256d one = _mm256_set1_pd(1.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const std::size_t record_count =
        static_cast<std::size_t>(bra_bin.primitive_count) *
        static_cast<std::size_t>(ket_bin.primitive_count) *
        ActiveRoots<Emit, NRoots>;
    std::vector<RysDPMatrixRootRecordV> records(record_count);
    const std::size_t bra_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(ket_bin);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_end; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_end;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_end - bra_pos);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask);
                if (active_mask == 0u) {
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : rys_low_bits(valid_lanes);
            const __m256d bra_r[3] = {
                _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos),
                _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos),
            };
            const __m256d ket_r[3] = {
                _mm256_set1_pd(ket_bin.Rx[ket_pos]),
                _mm256_set1_pd(ket_bin.Ry[ket_pos]),
                _mm256_set1_pd(ket_bin.Rz[ket_pos]),
            };
            const __m256d* const rdp = BraDP ? bra_r : ket_r;
            const __m256d* const rother = BraDP ? ket_r : bra_r;
            std::size_t record_index = 0;

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vinvp =
                    _mm256_loadu_pd(bra_bin.invs.data() + boff);
                const __m256d vpx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vpy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vpz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vpa[3] = {
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff),
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff),
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff),
                };
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vinvq = _mm256_set1_pd(ket_bin.invs[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(one, vpq);
                    const __m256d delta[3] = {
                        _mm256_sub_pd(vpx, _mm256_set1_pd(ket_bin.Wx[koff])),
                        _mm256_sub_pd(vpy, _mm256_set1_pd(ket_bin.Wy[koff])),
                        _mm256_sub_pd(vpz, _mm256_set1_pd(ket_bin.Wz[koff])),
                    };
                    const __m256d vr2 = _mm256_fmadd_pd(
                        delta[0], delta[0],
                        _mm256_fmadd_pd(
                            delta[1], delta[1],
                            _mm256_mul_pd(delta[2], delta[2])));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                            vr2));
                    rys_roots_weights_batch4_for_emit<Emit, NRoots>(
                        targ4, roots, weights);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    const __m256d qop = _mm256_mul_pd(vq, vinvpq);
                    const __m256d pop = _mm256_mul_pd(vp, vinvpq);
                    for (int root = 0;
                         root < ActiveRoots<Emit, NRoots>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots + 4 * root);
                        RysDPMatrixRootRecordV& record =
                            records[record_index++];
                        record.weight = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights + 4 * root));
                        const __m256d qscale = _mm256_mul_pd(qop, vrt);
                        const __m256d pscale = _mm256_mul_pd(pop, vrt);
                        const __m256d c[3] = {
                            _mm256_fnmadd_pd(qscale, delta[0], vpa[0]),
                            _mm256_fnmadd_pd(qscale, delta[1], vpa[1]),
                            _mm256_fnmadd_pd(qscale, delta[2], vpa[2]),
                        };
                        const __m256d d[3] = {
                            _mm256_fmadd_pd(
                                pscale, delta[0],
                                _mm256_set1_pd(ket_bin.Lx[koff])),
                            _mm256_fmadd_pd(
                                pscale, delta[1],
                                _mm256_set1_pd(ket_bin.Ly[koff])),
                            _mm256_fmadd_pd(
                                pscale, delta[2],
                                _mm256_set1_pd(ket_bin.Lz[koff])),
                        };
                        for (int axis = 0; axis < 3; ++axis) {
                            record.xdp[axis] = BraDP ? c[axis] : d[axis];
                            record.xother[axis] = BraDP ? d[axis] : c[axis];
                        }
                        record.b00 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvpq), vrt);
                        record.b00sq =
                            _mm256_mul_pd(record.b00, record.b00);
                        const __m256d b10 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvp),
                            _mm256_fnmadd_pd(qop, vrt, one));
                        const __m256d b01 = _mm256_mul_pd(
                            _mm256_mul_pd(half, vinvq),
                            _mm256_fnmadd_pd(pop, vrt, one));
                        record.bdp = BraDP ? b10 : b01;
                        record.bother = BraDP ? b01 : b10;
                    }
                }
            }

            contract_rys_dp_matrix_fragment<0, OtherPP>(
                records, record_index, rdp, rother, &raw[0][0]);
            contract_rys_dp_matrix_fragment<1, OtherPP>(
                records, record_index, rdp, rother, &raw[0][0]);
            contract_rys_dp_matrix_fragment<2, OtherPP>(
                records, record_index, rdp, rother, &raw[0][0]);
            contract_rys_dp_matrix_fragment<3, OtherPP>(
                records, record_index, rdp, rother, &raw[0][0]);
            contract_rys_dp_matrix_fragment<4, OtherPP>(
                records, record_index, rdp, rother, &raw[0][0]);

            if constexpr (
                OtherPP && rys_emit_uses_preemit_dppp_positive_k_v<Emit>) {
#ifdef TLNO_RYS_DIRECT_EXCHANGE
                const unsigned active_bits = Screened
                    ? active_mask : rys_low_bits(valid_lanes);
                if (active_bits != 0u) {
                    rys_k_dppp_preemit_packet_v1<BraDP>(
                        *rys_jk_context_slot(), bd, bra_bin, bra_pos,
                        ket_group, active_bits, raw);
                }
#endif
            } else {
                const auto& rhs_refs = bd.packed_spherical_pair_components[
                    static_cast<std::size_t>(ket_group)];
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    if constexpr (Screened) {
                        if ((active_mask &
                             (1u << static_cast<unsigned>(lane))) == 0u) continue;
                    }
                    const std::uint32_t bra_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& lhs_refs = bd.packed_spherical_pair_components[
                        static_cast<std::size_t>(bra_group)];
                    rys_emit_spherical_lane_block<Emit>(
                        eri_values, lhs_refs, rhs_refs, false,
                        [&raw, lane](const PackedPairComponentH& lhs_ref,
                                     const PackedPairComponentH& rhs_ref) noexcept {
                            const int dp_component = BraDP
                                ? static_cast<int>(lhs_ref.component_index)
                                : static_cast<int>(rhs_ref.component_index);
                            const int other_component = BraDP
                                ? static_cast<int>(rhs_ref.component_index)
                                : static_cast<int>(lhs_ref.component_index);
                            const int component =
                                dp_component * NOther + other_component;
                            return raw[component][lane];
                        });
                }
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
inline void accumulate_rys_dp_matrix_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& dp_bins = bd.rys_dp_pair_bins;
    const auto& ds_bins = bd.rys_ds_pair_bins;
    for (std::size_t di = 1; di < dp_bins.size(); ++di) {
        if (dp_bins[di].pair_count == 0) continue;
        for (std::size_t si = 1; si < ds_bins.size(); ++si) {
            if (ds_bins[si].pair_count == 0) continue;
            if (dp_bins[di].pair_count >= ds_bins[si].pair_count) {
                accumulate_rys_dp_matrix_spherical_shell_bin_pair<false, true, false, Emit>(
                    bd, dp_bins[di], ds_bins[si], eri_values);
            } else {
                accumulate_rys_dp_matrix_spherical_shell_bin_pair<false, false, false, Emit>(
                    bd, ds_bins[si], dp_bins[di], eri_values);
            }
        }
    }

    const auto& pp_bins = bd.rys_low_pair_bins[2];
    for (std::size_t di = 1; di < dp_bins.size(); ++di) {
        if (dp_bins[di].pair_count == 0) continue;
        for (std::size_t pi = 1; pi < pp_bins.size(); ++pi) {
            if (pp_bins[pi].pair_count == 0) continue;
            if (dp_bins[di].pair_count >= pp_bins[pi].pair_count) {
                accumulate_rys_dp_matrix_spherical_shell_bin_pair<true, true, false, Emit>(
                    bd, dp_bins[di], pp_bins[pi], eri_values);
            } else {
                accumulate_rys_dp_matrix_spherical_shell_bin_pair<true, false, false, Emit>(
                    bd, pp_bins[pi], dp_bins[di], eri_values);
            }
        }
    }
}

// The fragmented kernel pays a fixed root-record allocation/traversal cost.
// That cost is well amortized on the protein-sized shell sets for which it was
// designed, but not on tiny systems (for example H2O/6-31G(d)).  Count the
// shell-pair products handled by the specialization and leave small workloads
// on the lower-overhead compiled quartet path.
inline bool rys_dp_matrix_shell_batch_profitable(
    const BasisDataHGP& bd) noexcept {
    std::uint64_t dp_count = 0;
    std::uint64_t other_count = 0;
    for (std::size_t i = 1; i < bd.rys_dp_pair_bins.size(); ++i) {
        dp_count += bd.rys_dp_pair_bins[i].pair_count;
    }
    for (std::size_t i = 1; i < bd.rys_ds_pair_bins.size(); ++i) {
        other_count += bd.rys_ds_pair_bins[i].pair_count;
    }
    const auto& pp_bins = bd.rys_low_pair_bins[2];
    for (std::size_t i = 1; i < pp_bins.size(); ++i) {
        other_count += pp_bins[i].pair_count;
    }
    constexpr std::uint64_t kMinimumShellPairProducts = 256;
    return dp_count != 0 &&
           other_count >=
               (kMinimumShellPairProducts + dp_count - 1) / dp_count;
}

// Unit-normalized real f solid harmonics in the same PySCF ordering as
// kSphC2S_L3.  Evaluating the seven harmonic polynomials directly is the
// inline small-matrix equivalent of the sparse 7x10 Cartesian transform.
// The same-side Rys recurrence traces cancel because every H_l is harmonic.
template <bool WithDerivative>
inline void rys_f_solid_harmonics_v(
    const __m256d X[3], __m256d H[7], __m256d dH[7][3]) noexcept {
    const __m256d zero = _mm256_setzero_pd();
    const __m256d two = _mm256_set1_pd(2.0);
    const __m256d three = _mm256_set1_pd(3.0);
    const __m256d four = _mm256_set1_pd(4.0);
    const __m256d six = _mm256_set1_pd(6.0);
    const __m256d eight = _mm256_set1_pd(8.0);
    const __m256d half = _mm256_set1_pd(0.5);
    const __m256d a = _mm256_set1_pd(1.0 / std::sqrt(24.0));
    const __m256d b = _mm256_set1_pd(1.0 / std::sqrt(40.0));
    const __m256d c = _mm256_set1_pd(1.0 / std::sqrt(60.0));
    const __m256d x = X[0], y = X[1], z = X[2];
    const __m256d x2 = _mm256_mul_pd(x, x);
    const __m256d y2 = _mm256_mul_pd(y, y);
    const __m256d z2 = _mm256_mul_pd(z, z);
    const __m256d xy = _mm256_mul_pd(x, y);
    const __m256d xz = _mm256_mul_pd(x, z);
    const __m256d yz = _mm256_mul_pd(y, z);

    H[0] = _mm256_mul_pd(a, _mm256_mul_pd(
        y, _mm256_sub_pd(_mm256_mul_pd(three, x2), y2)));
    H[1] = _mm256_mul_pd(xy, z);
    H[2] = _mm256_mul_pd(b, _mm256_mul_pd(
        y, _mm256_sub_pd(_mm256_mul_pd(four, z2),
                         _mm256_add_pd(x2, y2))));
    H[3] = _mm256_mul_pd(c, _mm256_mul_pd(
        z, _mm256_sub_pd(_mm256_mul_pd(two, z2),
                         _mm256_mul_pd(three, _mm256_add_pd(x2, y2)))));
    H[4] = _mm256_mul_pd(b, _mm256_mul_pd(
        x, _mm256_sub_pd(_mm256_mul_pd(four, z2),
                         _mm256_add_pd(x2, y2))));
    H[5] = _mm256_mul_pd(half, _mm256_mul_pd(z, _mm256_sub_pd(x2, y2)));
    H[6] = _mm256_mul_pd(a, _mm256_mul_pd(
        x, _mm256_sub_pd(x2, _mm256_mul_pd(three, y2))));

    if constexpr (WithDerivative) {
        dH[0][0] = _mm256_mul_pd(_mm256_mul_pd(six, a), xy);
        dH[0][1] = _mm256_mul_pd(
            _mm256_mul_pd(three, a), _mm256_sub_pd(x2, y2));
        dH[0][2] = zero;

        dH[1][0] = yz;
        dH[1][1] = xz;
        dH[1][2] = xy;

        dH[2][0] = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_set1_pd(-2.0), b), xy);
        dH[2][1] = _mm256_mul_pd(
            b, _mm256_sub_pd(
                   _mm256_sub_pd(_mm256_mul_pd(four, z2), x2),
                   _mm256_mul_pd(three, y2)));
        dH[2][2] = _mm256_mul_pd(_mm256_mul_pd(eight, b), yz);

        dH[3][0] = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_set1_pd(-6.0), c), xz);
        dH[3][1] = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_set1_pd(-6.0), c), yz);
        dH[3][2] = _mm256_mul_pd(
            _mm256_mul_pd(three, c),
            _mm256_sub_pd(_mm256_mul_pd(two, z2),
                          _mm256_add_pd(x2, y2)));

        dH[4][0] = _mm256_mul_pd(
            b, _mm256_sub_pd(
                   _mm256_sub_pd(_mm256_mul_pd(four, z2),
                                 _mm256_mul_pd(three, x2)),
                   y2));
        dH[4][1] = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_set1_pd(-2.0), b), xy);
        dH[4][2] = _mm256_mul_pd(_mm256_mul_pd(eight, b), xz);

        dH[5][0] = xz;
        dH[5][1] = _mm256_sub_pd(zero, yz);
        dH[5][2] = _mm256_mul_pd(half, _mm256_sub_pd(x2, y2));

        dH[6][0] = _mm256_mul_pd(
            _mm256_mul_pd(three, a), _mm256_sub_pd(x2, y2));
        dH[6][1] = _mm256_mul_pd(
            _mm256_mul_pd(_mm256_set1_pd(-6.0), a), xy);
        dH[6][2] = zero;
    }
}

// Cross-shellset (fs|ss)/(fs|ps), including reversed pair orientations.
// Four independent shellsets occupy AVX2 lanes.  Primitive quartets are fully
// contracted into 7 or 21 pure outputs before the inline transform/scatter.
template <int OtherL, bool BraFS, bool Screened, class Emit>
inline void accumulate_rys_fs_low_spherical_shell_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    std::vector<double>& eri_values [[maybe_unused]]) {
    static_assert(OtherL == 0 || OtherL == 1, "fs shell batch supports ss/ps");
    constexpr int NRoots = (3 + OtherL) / 2 + 1;
    constexpr int NOut = 7 * (OtherL == 0 ? 1 : 3);
    alignas(32) double targ4[4];
    alignas(32) double roots[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double weights[4 * ActiveRoots<Emit, NRoots>];
    alignas(32) double raw[NOut][4];
    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const std::size_t bra_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(bra_bin);
    const std::size_t ket_pair_end =
        rys_df_q_floor_active_pair_count<Screened>(ket_bin);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    for (std::size_t ket_pos = 0; ket_pos < ket_pair_end; ++ket_pos) {
        const std::uint32_t ket_group = ket_bin.pair_group_ids[ket_pos];
        for (std::size_t bra_pos = 0; bra_pos < bra_pair_end;
             bra_pos += 4) {
            const std::size_t valid_lanes =
                std::min<std::size_t>(4, bra_pair_end - bra_pos);
            unsigned active_mask = 0u;
            if constexpr (Screened) {
                unsigned monotone_mask = 0u;
                active_mask = rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes, ket_group,
                    true, monotone_mask);
                if (active_mask == 0u) {
                    if (monotone_mask == 0u) break;
                    continue;
                }
            }
            const unsigned primitive_active_bits = Screened
                ? active_mask
                : rys_low_bits(valid_lanes);
            __m256d acc[NOut];
            for (auto& value : acc) value = _mm256_setzero_pd();

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vPx = _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy = _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz = _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vbw = _mm256_loadu_pd(bra_bin.w.data() + boff);
                const __m256d vPA[3] = {
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff),
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff),
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff),
                };
                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            primitive_active_bits,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const __m256d vq = _mm256_set1_pd(ket_bin.s[koff]);
                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                    const __m256d vd[3] = {
                        _mm256_sub_pd(vPx, _mm256_set1_pd(ket_bin.Wx[koff])),
                        _mm256_sub_pd(vPy, _mm256_set1_pd(ket_bin.Wy[koff])),
                        _mm256_sub_pd(vPz, _mm256_set1_pd(ket_bin.Wz[koff])),
                    };
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vd[0], vd[0],
                        _mm256_fmadd_pd(vd[1], vd[1],
                                        _mm256_mul_pd(vd[2], vd[2])));
                    _mm256_store_pd(
                        targ4,
                        _mm256_mul_pd(
                            _mm256_mul_pd(vp, _mm256_mul_pd(vq, vinvpq)),
                            vr2));
                    rys_roots_weights_batch4_for_emit<Emit, NRoots>(
                        targ4, roots, weights);
                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    const __m256d vqop = _mm256_mul_pd(vq, vinvpq);
                    const __m256d vpop = _mm256_mul_pd(vp, vinvpq);
                    for (int root = 0;
                         root < ActiveRoots<Emit, NRoots>; ++root) {
                        const __m256d vrt = _mm256_load_pd(roots + 4 * root);
                        const __m256d vww = _mm256_mul_pd(
                            vpref, _mm256_load_pd(weights + 4 * root));
                        const __m256d qscale = _mm256_mul_pd(vqop, vrt);
                        const __m256d pscale = _mm256_mul_pd(vpop, vrt);
                        const __m256d C[3] = {
                            _mm256_fnmadd_pd(qscale, vd[0], vPA[0]),
                            _mm256_fnmadd_pd(qscale, vd[1], vPA[1]),
                            _mm256_fnmadd_pd(qscale, vd[2], vPA[2]),
                        };
                        const __m256d D[3] = {
                            _mm256_fmadd_pd(
                                pscale, vd[0],
                                _mm256_set1_pd(ket_bin.Lx[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[1],
                                _mm256_set1_pd(ket_bin.Ly[koff])),
                            _mm256_fmadd_pd(
                                pscale, vd[2],
                                _mm256_set1_pd(ket_bin.Lz[koff])),
                        };
                        const __m256d* const Xf = BraFS ? C : D;
                        __m256d H[7], dH[7][3];
                        rys_f_solid_harmonics_v<(OtherL == 1)>(Xf, H, dH);
                        if constexpr (OtherL == 0) {
                            for (int f = 0; f < 7; ++f) {
                                acc[f] = _mm256_fmadd_pd(vww, H[f], acc[f]);
                            }
                        } else {
                            const __m256d* const Xp = BraFS ? D : C;
                            const __m256d B00 = _mm256_mul_pd(
                                _mm256_mul_pd(vhalf, vinvpq), vrt);
                            for (int f = 0; f < 7; ++f) {
                                for (int pcomp = 0; pcomp < 3; ++pcomp) {
                                    const __m256d value = _mm256_fmadd_pd(
                                        B00, dH[f][pcomp],
                                        _mm256_mul_pd(H[f], Xp[pcomp]));
                                    const int component = 3 * f + pcomp;
                                    acc[component] = _mm256_fmadd_pd(
                                        vww, value, acc[component]);
                                }
                            }
                        }
                    }
                }
            }

            for (int component = 0; component < NOut; ++component) {
                _mm256_store_pd(raw[component], acc[component]);
            }
            const auto& rhs_refs =
                bd.packed_spherical_pair_components[static_cast<std::size_t>(ket_group)];
            for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                if constexpr (Screened) {
                    if ((active_mask &
                         (1u << static_cast<unsigned>(lane))) == 0u) continue;
                }
                const std::uint32_t bra_group =
                    bra_bin.pair_group_ids[bra_pos + lane];
                const auto& lhs_refs =
                    bd.packed_spherical_pair_components[static_cast<std::size_t>(bra_group)];
                rys_emit_spherical_lane_block<Emit>(
                    eri_values, lhs_refs, rhs_refs, false,
                    [&raw, lane](const PackedPairComponentH& lhs_ref,
                                 const PackedPairComponentH& rhs_ref) noexcept {
                        const int f_component = BraFS
                            ? static_cast<int>(lhs_ref.component_index)
                            : static_cast<int>(rhs_ref.component_index);
                        int component = f_component;
                        if constexpr (OtherL == 1) {
                            const int p_component = BraFS
                                ? static_cast<int>(rhs_ref.component_index)
                                : static_cast<int>(lhs_ref.component_index);
                            component = 3 * f_component + p_component;
                        }
                        return raw[component][lane];
                    });
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <int OtherL, class Emit>
inline void accumulate_rys_fs_low_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    const auto& fs_bins = bd.rys_fs_pair_bins;
    const auto& low_bins = bd.rys_low_pair_bins[static_cast<std::size_t>(OtherL)];
    for (std::size_t fi = 1; fi < fs_bins.size(); ++fi) {
        if (fs_bins[fi].pair_count == 0) continue;
        for (std::size_t li = 1; li < low_bins.size(); ++li) {
            if (low_bins[li].pair_count == 0) continue;
            if (fs_bins[fi].pair_count >= low_bins[li].pair_count) {
                accumulate_rys_fs_low_spherical_shell_bin_pair<OtherL, true, false, Emit>(
                    bd, fs_bins[fi], low_bins[li], eri_values);
            } else {
                accumulate_rys_fs_low_spherical_shell_bin_pair<OtherL, false, false, Emit>(
                    bd, low_bins[li], fs_bins[fi], eri_values);
            }
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
inline void accumulate_rys_fs_spherical_shell_batches(
    const BasisDataHGP& bd,
    std::vector<double>& eri_values) {
    accumulate_rys_fs_low_spherical_shell_batches<0, Emit>(bd, eri_values);
    accumulate_rys_fs_low_spherical_shell_batches<1, Emit>(bd, eri_values);
}

#endif  // __AVX2__ && __FMA__

template <int ISum, int KSum, int GStride>
inline void build_rys_2d_axis_soa_fixed(double* __restrict G,
                                        const double* __restrict C00,
                                        const double* __restrict D00,
                                        const double* __restrict B00,
                                        const double* __restrict B10,
                                        const double* __restrict B01) noexcept {
    constexpr int B = kRysSoaBatch;
#if defined(__AVX2__) && defined(__FMA__)
    if constexpr (B == 4 && ISum <= 2 && KSum <= 2) {
        // Fully unrolled build for the pure s/p shapes (pp|ps, ps|pp, pp|pp).
        // The fused path measured unrolled axis builds 2.7-3.6x faster than
        // the loopy recurrence; libcint's g0_2d4d build is unrolled codegen.
        // One __m256d per (i,k) entry, straight-line VRR:
        //   G[i+1,0] = C00*G[i,0] + i*B10*G[i-1,0]
        //   G[i,k+1] = D00*G[i,k] + i*B00*G[i-1,k] + k*B01*G[i,k-1]
        const __m256d vc = _mm256_loadu_pd(C00);
        const __m256d vd = _mm256_loadu_pd(D00);
        const __m256d one = _mm256_set1_pd(1.0);
        __m256d g10 = one, g20 = one, g11 = one, g21 = one;
        _mm256_storeu_pd(G, one);
        if constexpr (ISum >= 1) {
            g10 = vc;
            _mm256_storeu_pd(G + GStride * B, g10);
        }
        if constexpr (ISum >= 2) {
            g20 = _mm256_fmadd_pd(vc, g10, _mm256_loadu_pd(B10));
            _mm256_storeu_pd(G + 2 * GStride * B, g20);
        }
        if constexpr (KSum >= 1) {
            const __m256d vb00 = _mm256_loadu_pd(B00);
            const __m256d g01 = vd;
            _mm256_storeu_pd(G + B, g01);
            if constexpr (ISum >= 1) {
                g11 = _mm256_fmadd_pd(vd, g10, vb00);
                _mm256_storeu_pd(G + (GStride + 1) * B, g11);
            }
            if constexpr (ISum >= 2) {
                const __m256d b2 = _mm256_add_pd(vb00, vb00);
                g21 = _mm256_fmadd_pd(vd, g20, _mm256_mul_pd(b2, g10));
                _mm256_storeu_pd(G + (2 * GStride + 1) * B, g21);
            }
            if constexpr (KSum >= 2) {
                const __m256d vb01 = _mm256_loadu_pd(B01);
                const __m256d g02 = _mm256_fmadd_pd(vd, g01, vb01);
                _mm256_storeu_pd(G + 2 * B, g02);
                if constexpr (ISum >= 1) {
                    const __m256d g12 = _mm256_fmadd_pd(vd, g11,
                        _mm256_fmadd_pd(vb00, g01, _mm256_mul_pd(vb01, g10)));
                    _mm256_storeu_pd(G + (GStride + 2) * B, g12);
                }
                if constexpr (ISum >= 2) {
                    const __m256d b2 = _mm256_add_pd(vb00, vb00);
                    const __m256d g22 = _mm256_fmadd_pd(vd, g21,
                        _mm256_fmadd_pd(b2, g11, _mm256_mul_pd(vb01, g20)));
                    _mm256_storeu_pd(G + (2 * GStride + 2) * B, g22);
                }
            }
        }
        return;
    }
#endif
#pragma omp simd
    for (int lane = 0; lane < B; ++lane) {
        G[lane] = 1.0;
    }

    if constexpr (ISum >= 1) {
        constexpr int dst0 = GStride * B;
#pragma omp simd
        for (int lane = 0; lane < B; ++lane) {
            G[dst0 + lane] = C00[lane];
        }
        for (int i = 1; i < ISum; ++i) {
            const int dst = (i + 1) * GStride * B;
            const int cur = i * GStride * B;
            const int prv = (i - 1) * GStride * B;
            const double ii = static_cast<double>(i);
#pragma omp simd
            for (int lane = 0; lane < B; ++lane) {
                G[dst + lane] = C00[lane] * G[cur + lane]
                                + ii * B10[lane] * G[prv + lane];
            }
        }
    }

    for (int k = 0; k < KSum; ++k) {
        const double kk = static_cast<double>(k);
        for (int i = 0; i <= ISum; ++i) {
            const int dst = (i * GStride + (k + 1)) * B;
            const int cur = (i * GStride + k) * B;
            const int im1 = ((i - 1) * GStride + k) * B;
            const int km1 = (i * GStride + (k - 1)) * B;
            const double ii = static_cast<double>(i);
#pragma omp simd
            for (int lane = 0; lane < B; ++lane) {
                double v = D00[lane] * G[cur + lane];
                if (i >= 1) {
                    v += ii * B00[lane] * G[im1 + lane];
                }
                if (k >= 1) {
                    v += kk * B01[lane] * G[km1 + lane];
                }
                G[dst + lane] = v;
            }
        }
    }
}

inline void build_rys_2d_axis_soa_runtime(int isum,
                                          int ksum,
                                          int gstride,
                                          double* __restrict G,
                                          const double* __restrict C00,
                                          const double* __restrict D00,
                                          const double* __restrict B00,
                                          const double* __restrict B10,
                                          const double* __restrict B01) noexcept {
    constexpr int B = kRysSoaBatch;
#pragma omp simd
    for (int lane = 0; lane < B; ++lane) {
        G[lane] = 1.0;
    }

    if (isum >= 1) {
        const int dst0 = gstride * B;
#pragma omp simd
        for (int lane = 0; lane < B; ++lane) {
            G[dst0 + lane] = C00[lane];
        }
        for (int i = 1; i < isum; ++i) {
            const int dst = (i + 1) * gstride * B;
            const int cur = i * gstride * B;
            const int prv = (i - 1) * gstride * B;
            const double ii = static_cast<double>(i);
#pragma omp simd
            for (int lane = 0; lane < B; ++lane) {
                G[dst + lane] = C00[lane] * G[cur + lane]
                                + ii * B10[lane] * G[prv + lane];
            }
        }
    }

    for (int k = 0; k < ksum; ++k) {
        const double kk = static_cast<double>(k);
        for (int i = 0; i <= isum; ++i) {
            const int dst = (i * gstride + (k + 1)) * B;
            const int cur = (i * gstride + k) * B;
            const int im1 = ((i - 1) * gstride + k) * B;
            const int km1 = (i * gstride + (k - 1)) * B;
            const double ii = static_cast<double>(i);
#pragma omp simd
            for (int lane = 0; lane < B; ++lane) {
                double v = D00[lane] * G[cur + lane];
                if (i >= 1) {
                    v += ii * B00[lane] * G[im1 + lane];
                }
                if (k >= 1) {
                    v += kk * B01[lane] * G[km1 + lane];
                }
                G[dst + lane] = v;
            }
        }
    }
}

template < int ISum, int KSum, int NRoots, int GStride, int GSize,
          int NaRows, int NcRows, class Emit >
inline void accumulate_rys_quartets_soa_fixed(
    const RysPrimitivePairSlab& bras,
    const RysPrimitivePairSlab& kets,
    const std::array<double, 3>& A,
    const std::array<double, 3>& C,
    const int* __restrict axs,
    const int* __restrict ays,
    const int* __restrict azs,
    const int* __restrict cxs,
    const int* __restrict cys,
    const int* __restrict czs,
    double* __restrict radial) {
    constexpr int B = kRysSoaBatch;
    const int n_bras = bras.size();
    const int n_kets = kets.size();
    if (n_bras == 0 || n_kets == 0) return;
    const int n_primitive_quartets = n_bras * n_kets;

    double Gx[GSize * B], Gy[GSize * B], Gz[GSize * B];
    double roots[NRoots * B], weights[NRoots * B];

    double Px[B], Py[B], Pz[B], Qx[B], Qy[B], Qz[B];
    double PQx[B], PQy[B], PQz[B], prefactor[B];
    double q_over_pq[B], p_over_pq[B], inv2p[B], inv2q[B], inv2pq[B];

    double ww[B], B00[B], B10[B], B01[B];
    double C00x[B], C00y[B], C00z[B], D00x[B], D00y[B], D00z[B];

    // Full batches only: the padded-lane scheme replicated the last valid prim
    // into idle lanes at prefactor=0, wasting whole vector ops (~18% of the
    // work at ~10 prims/block, the pp-class regime). The 1..B-1 remainder is
    // handled by the zero-waste scalar tail below.
    const int n_full_batches = (n_primitive_quartets / B) * B;
    for (int batch_start = 0; batch_start < n_full_batches; batch_start += B) {
        const int valid = std::min(B, n_primitive_quartets - batch_start);
        bool header_done = false;
#if defined(__AVX2__) && defined(__FMA__)
        // Vectorized batch header: when the 4 lanes share one bra primitive and
        // read 4 contiguous ket primitives (the common case for contracted
        // low-L blocks), every geometric quantity is one AVX op and the Rys
        // table lookup runs lane-parallel. Falls through to the scalar per-lane
        // header at bra-row boundaries or when a lane needs the asymptotic
        // branch of the root table.
        if constexpr (B == 4 && NRoots <= 3) {
            const int vib = batch_start / n_kets;
            const int vik = batch_start - vib * n_kets;
            if (vik + B <= n_kets) {
                const std::size_t ibs = static_cast<std::size_t>(vib);
                const std::size_t iks = static_cast<std::size_t>(vik);
                const double p = bras.s[ibs];
                const __m256d vp = _mm256_set1_pd(p);
                const __m256d vq = _mm256_loadu_pd(kets.s.data() + iks);
                const __m256d vpq = _mm256_add_pd(vp, vq);
                const __m256d vone = _mm256_set1_pd(1.0);
                const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                const __m256d vPx = _mm256_set1_pd(bras.Wx[ibs]);
                const __m256d vPy = _mm256_set1_pd(bras.Wy[ibs]);
                const __m256d vPz = _mm256_set1_pd(bras.Wz[ibs]);
                const __m256d vQx = _mm256_loadu_pd(kets.Wx.data() + iks);
                const __m256d vQy = _mm256_loadu_pd(kets.Wy.data() + iks);
                const __m256d vQz = _mm256_loadu_pd(kets.Wz.data() + iks);
                const __m256d vPQx = _mm256_sub_pd(vPx, vQx);
                const __m256d vPQy = _mm256_sub_pd(vPy, vQy);
                const __m256d vPQz = _mm256_sub_pd(vPz, vQz);
                _mm256_storeu_pd(Px, vPx); _mm256_storeu_pd(Py, vPy);
                _mm256_storeu_pd(Pz, vPz);
                _mm256_storeu_pd(Qx, vQx); _mm256_storeu_pd(Qy, vQy);
                _mm256_storeu_pd(Qz, vQz);
                _mm256_storeu_pd(PQx, vPQx); _mm256_storeu_pd(PQy, vPQy);
                _mm256_storeu_pd(PQz, vPQz);
                _mm256_storeu_pd(q_over_pq, _mm256_mul_pd(vq, vinvpq));
                _mm256_storeu_pd(p_over_pq, _mm256_mul_pd(vp, vinvpq));
                const __m256d vhalf = _mm256_set1_pd(0.5);
                _mm256_storeu_pd(inv2p, _mm256_set1_pd(0.5 * bras.invs[ibs]));
                _mm256_storeu_pd(inv2q, _mm256_mul_pd(vhalf, _mm256_loadu_pd(kets.invs.data() + iks)));
                _mm256_storeu_pd(inv2pq, _mm256_mul_pd(vhalf, vinvpq));
                _mm256_storeu_pd(prefactor, _mm256_div_pd(
                    _mm256_mul_pd(_mm256_set1_pd(kTwoPi52 * bras.w[ibs]),
                                  _mm256_loadu_pd(kets.w.data() + iks)),
                    _mm256_sqrt_pd(vpq)));
                const __m256d vrpq2 = _mm256_fmadd_pd(vPQx, vPQx,
                    _mm256_fmadd_pd(vPQy, vPQy, _mm256_mul_pd(vPQz, vPQz)));
                alignas(32) double targ4[4];
                _mm256_store_pd(targ4, _mm256_mul_pd(
                    _mm256_mul_pd(_mm256_mul_pd(vp, vq), vinvpq), vrpq2));
#ifdef TLNO_J_SPEED_PROBE_EXACT_N2_LAGRANGE
                if constexpr (std::is_same_v<Emit, RysDirectJSpeedProbeEmit> &&
                              NRoots == 2) {
                    rys_roots_weights_batch4_for_emit<Emit, NRoots>(
                        targ4, roots, weights);
                } else
#endif
                {
                    if (!rys_roots_weights_batch4<NRoots>(
                            targ4, roots, weights)) {
                        double rtmp[NRoots], wtmp[NRoots];
                        for (int lane = 0; lane < B; ++lane) {
                            rys_roots_weights_fixed<NRoots>(
                                targ4[lane], rtmp, wtmp);
                            for (int n = 0; n < NRoots; ++n) {
                                roots[n * B + lane] = rtmp[n];
                                weights[n * B + lane] = wtmp[n];
                            }
                        }
                    }
                }
                header_done = true;
            }
        }
#endif  // __AVX2__ && __FMA__
        if (!header_done)
        for (int lane = 0; lane < B; ++lane) {
            const int flat = batch_start + ((lane < valid) ? lane : valid - 1);
            const int ib = flat / n_kets;
            const int ik = flat - ib * n_kets;
            const std::size_t ibs = static_cast<std::size_t>(ib);
            const std::size_t iks = static_cast<std::size_t>(ik);

            const double p = bras.s[ibs];
            const double q = kets.s[iks];
            const double pq = p + q;
            const double inv_pq = 1.0 / pq;
            Px[lane] = bras.Wx[ibs];
            Py[lane] = bras.Wy[ibs];
            Pz[lane] = bras.Wz[ibs];
            Qx[lane] = kets.Wx[iks];
            Qy[lane] = kets.Wy[iks];
            Qz[lane] = kets.Wz[iks];
            PQx[lane] = Px[lane] - Qx[lane];
            PQy[lane] = Py[lane] - Qy[lane];
            PQz[lane] = Pz[lane] - Qz[lane];
            q_over_pq[lane] = q * inv_pq;
            p_over_pq[lane] = p * inv_pq;
            inv2p[lane] = 0.5 * bras.invs[ibs];
            inv2q[lane] = 0.5 * kets.invs[iks];
            inv2pq[lane] = 0.5 * inv_pq;
            prefactor[lane] = (lane < valid)
                                  ? kTwoPi52 * bras.w[ibs] * kets.w[iks] / std::sqrt(pq)
                                  : 0.0;

            const double rpq2 = PQx[lane] * PQx[lane]
                                + PQy[lane] * PQy[lane]
                                + PQz[lane] * PQz[lane];
            const double Targ = (p * q * inv_pq) * rpq2;
            double rtmp[NRoots], wtmp[NRoots];
            rys_roots_weights_fixed<NRoots>(Targ, rtmp, wtmp);
            for (int n = 0; n < NRoots; ++n) {
                roots[n * B + lane] = rtmp[n];
                weights[n * B + lane] = wtmp[n];
            }
        }

        for (int n = 0; n < NRoots; ++n) {
#if defined(__AVX2__) && defined(__FMA__)
            if constexpr (B == 4) {
                // Explicit AVX2 coefficient setup: one vector op per array
                // (t2/weights are lane-contiguous at n*B).
                const __m256d t2v = _mm256_loadu_pd(roots + n * B);
                const __m256d onev = _mm256_set1_pd(1.0);
                const __m256d qop = _mm256_loadu_pd(q_over_pq);
                const __m256d pop = _mm256_loadu_pd(p_over_pq);
                _mm256_storeu_pd(ww, _mm256_mul_pd(_mm256_loadu_pd(prefactor),
                                                   _mm256_loadu_pd(weights + n * B)));
                _mm256_storeu_pd(B00, _mm256_mul_pd(_mm256_loadu_pd(inv2pq), t2v));
                _mm256_storeu_pd(B10, _mm256_mul_pd(_mm256_loadu_pd(inv2p),
                                     _mm256_fnmadd_pd(qop, t2v, onev)));
                _mm256_storeu_pd(B01, _mm256_mul_pd(_mm256_loadu_pd(inv2q),
                                     _mm256_fnmadd_pd(pop, t2v, onev)));
                const __m256d qt = _mm256_mul_pd(qop, t2v);
                const __m256d pt = _mm256_mul_pd(pop, t2v);
                _mm256_storeu_pd(C00x, _mm256_fnmadd_pd(qt, _mm256_loadu_pd(PQx),
                    _mm256_sub_pd(_mm256_loadu_pd(Px), _mm256_set1_pd(A[0]))));
                _mm256_storeu_pd(C00y, _mm256_fnmadd_pd(qt, _mm256_loadu_pd(PQy),
                    _mm256_sub_pd(_mm256_loadu_pd(Py), _mm256_set1_pd(A[1]))));
                _mm256_storeu_pd(C00z, _mm256_fnmadd_pd(qt, _mm256_loadu_pd(PQz),
                    _mm256_sub_pd(_mm256_loadu_pd(Pz), _mm256_set1_pd(A[2]))));
                _mm256_storeu_pd(D00x, _mm256_fmadd_pd(pt, _mm256_loadu_pd(PQx),
                    _mm256_sub_pd(_mm256_loadu_pd(Qx), _mm256_set1_pd(C[0]))));
                _mm256_storeu_pd(D00y, _mm256_fmadd_pd(pt, _mm256_loadu_pd(PQy),
                    _mm256_sub_pd(_mm256_loadu_pd(Qy), _mm256_set1_pd(C[1]))));
                _mm256_storeu_pd(D00z, _mm256_fmadd_pd(pt, _mm256_loadu_pd(PQz),
                    _mm256_sub_pd(_mm256_loadu_pd(Qz), _mm256_set1_pd(C[2]))));
            } else
#endif
            {
#pragma omp simd
            for (int lane = 0; lane < B; ++lane) {
                const double t2 = roots[n * B + lane];
                ww[lane] = prefactor[lane] * weights[n * B + lane];
                B00[lane] = inv2pq[lane] * t2;
                B10[lane] = inv2p[lane] * (1.0 - q_over_pq[lane] * t2);
                B01[lane] = inv2q[lane] * (1.0 - p_over_pq[lane] * t2);
                C00x[lane] = (Px[lane] - A[0]) - q_over_pq[lane] * PQx[lane] * t2;
                C00y[lane] = (Py[lane] - A[1]) - q_over_pq[lane] * PQy[lane] * t2;
                C00z[lane] = (Pz[lane] - A[2]) - q_over_pq[lane] * PQz[lane] * t2;
                D00x[lane] = (Qx[lane] - C[0]) + p_over_pq[lane] * PQx[lane] * t2;
                D00y[lane] = (Qy[lane] - C[1]) + p_over_pq[lane] * PQy[lane] * t2;
                D00z[lane] = (Qz[lane] - C[2]) + p_over_pq[lane] * PQz[lane] * t2;
            }
            }

            build_rys_2d_axis_soa_fixed<ISum, KSum, GStride>(
                Gx, C00x, D00x, B00, B10, B01);
            build_rys_2d_axis_soa_fixed<ISum, KSum, GStride>(
                Gy, C00y, D00y, B00, B10, B01);
            build_rys_2d_axis_soa_fixed<ISum, KSum, GStride>(
                Gz, C00z, D00z, B00, B10, B01);

#if defined(__AVX2__) && defined(__FMA__)
            if constexpr (B == 4) {
                // Radial contract, explicit AVX2. Two proven fused-path lessons
                // applied here: (1) fold ww into Gz once per root (Gzw) so the
                // inner product reads 3 streams instead of 4 (WFOLD pattern;
                // same product, FP reassociation only); (2) process TWO output
                // entries per iteration with the hadd-pair reduction instead of
                // one omp-simd horizontal reduction per entry (the pragma
                // reduction was measured 2.2-2.7 cyc/lane-fma on pp classes).
                alignas(32) double Gzw[GSize * B];
                {
                    const __m256d w = _mm256_loadu_pd(ww);
                    for (int t = 0; t < GSize; ++t)
                        _mm256_storeu_pd(Gzw + t * B,
                            _mm256_mul_pd(_mm256_loadu_pd(Gz + t * B), w));
                }
                for (int al = 0; al < NaRows; ++al) {
                    double* __restrict row = radial + static_cast<std::size_t>(al) * NcRows;
                    const int gx_a = axs[al] * GStride;
                    const int gy_a = ays[al] * GStride;
                    const int gz_a = azs[al] * GStride;
                    int cf = 0;
                    for (; cf + 2 <= NcRows; cf += 2) {
                        const __m256d p0 = _mm256_mul_pd(
                            _mm256_mul_pd(_mm256_loadu_pd(Gx + (gx_a + cxs[cf]) * B),
                                          _mm256_loadu_pd(Gy + (gy_a + cys[cf]) * B)),
                            _mm256_loadu_pd(Gzw + (gz_a + czs[cf]) * B));
                        const __m256d p1 = _mm256_mul_pd(
                            _mm256_mul_pd(_mm256_loadu_pd(Gx + (gx_a + cxs[cf + 1]) * B),
                                          _mm256_loadu_pd(Gy + (gy_a + cys[cf + 1]) * B)),
                            _mm256_loadu_pd(Gzw + (gz_a + czs[cf + 1]) * B));
                        const __m256d h = _mm256_hadd_pd(p0, p1);
                        const __m128d s = _mm_add_pd(_mm256_castpd256_pd128(h),
                                                     _mm256_extractf128_pd(h, 1));
                        _mm_storeu_pd(row + cf, _mm_add_pd(_mm_loadu_pd(row + cf), s));
                    }
                    for (; cf < NcRows; ++cf) {
                        const __m256d p = _mm256_mul_pd(
                            _mm256_mul_pd(_mm256_loadu_pd(Gx + (gx_a + cxs[cf]) * B),
                                          _mm256_loadu_pd(Gy + (gy_a + cys[cf]) * B)),
                            _mm256_loadu_pd(Gzw + (gz_a + czs[cf]) * B));
                        const __m128d s2 = _mm_add_pd(_mm256_castpd256_pd128(p),
                                                      _mm256_extractf128_pd(p, 1));
                        row[cf] += _mm_cvtsd_f64(_mm_hadd_pd(s2, s2));
                    }
                }
            } else
#endif
            for (int al = 0; al < NaRows; ++al) {
                double* __restrict row = radial + static_cast<std::size_t>(al) * NcRows;
                const int gx_a = axs[al] * GStride;
                const int gy_a = ays[al] * GStride;
                const int gz_a = azs[al] * GStride;
                for (int cf = 0; cf < NcRows; ++cf) {
                    const double* __restrict gx = Gx + (gx_a + cxs[cf]) * B;
                    const double* __restrict gy = Gy + (gy_a + cys[cf]) * B;
                    const double* __restrict gz = Gz + (gz_a + czs[cf]) * B;
                    double sum = 0.0;
#pragma omp simd reduction(+:sum)
                    for (int lane = 0; lane < B; ++lane) {
                        sum += ww[lane] * gx[lane] * gy[lane] * gz[lane];
                    }
                    row[cf] += sum;
                }
            }
        }
    }
    // Zero-waste scalar tail for the 1..B-1 remaining primitive quartets.
    if (n_full_batches < n_primitive_quartets) {
        double Gsx[GSize], Gsy[GSize], Gsz[GSize];
        double* Gs3[3] = {Gsx, Gsy, Gsz};
        double rts[NRoots], wts[NRoots];
        for (int flat = n_full_batches; flat < n_primitive_quartets; ++flat) {
            const int ib = flat / n_kets;
            const int ik = flat - ib * n_kets;
            const std::size_t ibs = static_cast<std::size_t>(ib);
            const std::size_t iks = static_cast<std::size_t>(ik);
            const double p = bras.s[ibs], q = kets.s[iks];
            const double pq = p + q, inv_pq = 1.0 / pq;
            const double P[3] = {bras.Wx[ibs], bras.Wy[ibs], bras.Wz[ibs]};
            const double Q[3] = {kets.Wx[iks], kets.Wy[iks], kets.Wz[iks]};
            const double PQ[3] = {P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]};
            const double rpq2 = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];
            const double Targ = p * q * inv_pq * rpq2;
            const double prefac = kTwoPi52 * bras.w[ibs] * kets.w[iks] / std::sqrt(pq);
            rys_roots_weights_fixed<NRoots>(Targ, rts, wts);
            const double q_over = q * inv_pq, p_over = p * inv_pq;
            const double i2p = 0.5 * bras.invs[ibs];
            const double i2q = 0.5 * kets.invs[iks];
            const double i2pq = 0.5 * inv_pq;
            for (int n = 0; n < NRoots; ++n) {
                const double t2 = rts[n];
                const double wwn = prefac * wts[n];
                const double B00s = i2pq * t2;
                const double B10s = i2p * (1.0 - q_over * t2);
                const double B01s = i2q * (1.0 - p_over * t2);
                for (int ax = 0; ax < 3; ++ax) {
                    double* G = Gs3[ax];
                    const double C00 = (P[ax] - A[ax]) - q_over * PQ[ax] * t2;
                    const double D00 = (Q[ax] - C[ax]) + p_over * PQ[ax] * t2;
                    G[0] = 1.0;
                    if constexpr (ISum >= 1) {
                        G[GStride] = C00;
                        for (int i = 1; i < ISum; ++i) {
                            G[(i + 1) * GStride] =
                                C00 * G[i * GStride]
                                + static_cast<double>(i) * B10s * G[(i - 1) * GStride];
                        }
                    }
                    for (int k = 0; k < KSum; ++k) {
                        for (int i = 0; i <= ISum; ++i) {
                            double v = D00 * G[i * GStride + k];
                            if (i >= 1) {
                                v += static_cast<double>(i) * B00s * G[(i - 1) * GStride + k];
                            }
                            if (k >= 1) {
                                v += static_cast<double>(k) * B01s * G[i * GStride + (k - 1)];
                            }
                            G[i * GStride + (k + 1)] = v;
                        }
                    }
                }
                for (int al = 0; al < NaRows; ++al) {
                    const double* __restrict Gxr = Gsx + axs[al] * GStride;
                    const double* __restrict Gyr = Gsy + ays[al] * GStride;
                    const double* __restrict Gzr = Gsz + azs[al] * GStride;
                    double* __restrict row = radial + static_cast<std::size_t>(al) * NcRows;
#pragma omp simd
                    for (int cf = 0; cf < NcRows; ++cf) {
                        row[cf] += wwn * Gxr[cxs[cf]] * Gyr[cys[cf]] * Gzr[czs[cf]];
                    }
                }
            }
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
// Lean register-resident batch kernel for the Ltot<=1 classes (ss|ss, ss|ps,
// ps|ss). These shapes need no G table at all: the single-root quartet value
// is ww (ss|ss), ww*C00[axis] (ps|ss), or ww*D00[axis] (ss|ps). Everything
// lives in ymm registers with persistent vector accumulators reduced once at
// the end, so the L1 footprint is near zero -- this both beats the scalar loop
// outright and stops the cache-eviction interference the full batch kernel
// inflicts on scalar neighbours.
template < int ISum, int KSum, int NRoots, int NaRows, int NcRows,
          class Emit >
inline void accumulate_rys_quartets_lean_fixed(
    const RysPrimitivePairSlab& bras,
    const RysPrimitivePairSlab& kets,
    const std::array<double, 3>& A,
    const std::array<double, 3>& C,
    const int* __restrict axs,
    const int* __restrict ays,
    const int* __restrict azs,
    const int* __restrict cxs,
    const int* __restrict cys,
    const int* __restrict czs,
    double* __restrict radial) {
    static_assert(ISum + KSum <= 1 && NRoots == 1, "lean kernel covers Ltot<=1");
    // Radial rows are CUMULATIVE monomials (HRR needs the lower rows): the
    // L-carrying side has 4 rows (constant + x,y,z) unless AOffset strips the
    // constant (La>=1 bra), giving 3. Row value = ww for the constant row,
    // ww*C00/D00[axis] for the one-hot rows.
    static_assert(NaRows * NcRows <= 4, "unexpected component count");
    const int n_bras = static_cast<int>(bras.s.size());
    const int n_kets = static_cast<int>(kets.s.size());
    if (n_bras == 0 || n_kets == 0) return;

    __m256d acc[NaRows * NcRows];
    for (int r = 0; r < NaRows * NcRows; ++r) acc[r] = _mm256_setzero_pd();
    alignas(32) double targ4[4], rts4[4], wts4[4];

    for (int ib = 0; ib < n_bras; ++ib) {
        const std::size_t ibs = static_cast<std::size_t>(ib);
        const double p = bras.s[ibs];
        const __m256d vp = _mm256_set1_pd(p);
        const __m256d vPx = _mm256_set1_pd(bras.Wx[ibs]);
        const __m256d vPy = _mm256_set1_pd(bras.Wy[ibs]);
        const __m256d vPz = _mm256_set1_pd(bras.Wz[ibs]);
        const double wb_pref = kTwoPi52 * bras.w[ibs];
        // (P - A) per axis, needed only for the ps|ss C00.
        const __m256d vPAx = _mm256_set1_pd(bras.Wx[ibs] - A[0]);
        const __m256d vPAy = _mm256_set1_pd(bras.Wy[ibs] - A[1]);
        const __m256d vPAz = _mm256_set1_pd(bras.Wz[ibs] - A[2]);
        int ik = 0;
        for (; ik + 4 <= n_kets; ik += 4) {
            const std::size_t iks = static_cast<std::size_t>(ik);
            const __m256d vq = _mm256_loadu_pd(kets.s.data() + iks);
            const __m256d vpq = _mm256_add_pd(vp, vq);
            const __m256d vinvpq = _mm256_div_pd(_mm256_set1_pd(1.0), vpq);
            const __m256d vQx = _mm256_loadu_pd(kets.Wx.data() + iks);
            const __m256d vQy = _mm256_loadu_pd(kets.Wy.data() + iks);
            const __m256d vQz = _mm256_loadu_pd(kets.Wz.data() + iks);
            const __m256d vPQx = _mm256_sub_pd(vPx, vQx);
            const __m256d vPQy = _mm256_sub_pd(vPy, vQy);
            const __m256d vPQz = _mm256_sub_pd(vPz, vQz);
            const __m256d vrpq2 = _mm256_fmadd_pd(vPQx, vPQx,
                _mm256_fmadd_pd(vPQy, vPQy, _mm256_mul_pd(vPQz, vPQz)));
            const __m256d vqop = _mm256_mul_pd(vq, vinvpq);
            _mm256_store_pd(targ4,
                _mm256_mul_pd(_mm256_mul_pd(vp, vqop), vrpq2));
#ifdef TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE
            if constexpr (std::is_same_v <
                              Emit, RysDirectJSpeedProbeEmit > ) {
                rys_roots_weights_batch4_for_emit<Emit, 1>(
                    targ4, rts4, wts4);
            } else
#endif
            {
                if (!rys_roots_weights_batch4<1>(
                        targ4, rts4, wts4)) {
                    double rt, wt;
                    for (int l = 0; l < 4; ++l) {
                        rys_roots_weights_fixed<1>(
                            targ4[l], &rt, &wt);
                        rts4[l] = rt;
                        wts4[l] = wt;
                    }
                }
            }
            const __m256d vpref = _mm256_div_pd(
                _mm256_mul_pd(_mm256_set1_pd(wb_pref),
                              _mm256_loadu_pd(kets.w.data() + iks)),
                _mm256_sqrt_pd(vpq));
            const __m256d vww = _mm256_mul_pd(vpref, _mm256_load_pd(wts4));
            if constexpr (ISum == 0 && KSum == 0) {
                acc[0] = _mm256_add_pd(acc[0], vww);
            } else {
                const __m256d vt2 = _mm256_load_pd(rts4);
                if constexpr (ISum == 1) {
                    // C00[ax] = (P-A)[ax] - q_over_pq * t2 * PQ[ax]
                    const __m256d vqt = _mm256_mul_pd(vqop, vt2);
                    const __m256d c0 = _mm256_fnmadd_pd(vqt, vPQx, vPAx);
                    const __m256d c1 = _mm256_fnmadd_pd(vqt, vPQy, vPAy);
                    const __m256d c2 = _mm256_fnmadd_pd(vqt, vPQz, vPAz);
                    for (int al = 0; al < NaRows; ++al) {
                        if (axs[al] + ays[al] + azs[al] == 0) {
                            acc[al] = _mm256_add_pd(acc[al], vww);  // constant row
                        } else {
                            const __m256d sel = axs[al] ? c0 : (ays[al] ? c1 : c2);
                            acc[al] = _mm256_fmadd_pd(vww, sel, acc[al]);
                        }
                    }
                } else {
                    // D00[ax] = (Q-C)[ax] + p_over_pq * t2 * PQ[ax]
                    const __m256d vpop = _mm256_mul_pd(vp, vinvpq);
                    const __m256d vpt = _mm256_mul_pd(vpop, vt2);
                    const __m256d d0 = _mm256_fmadd_pd(vpt, vPQx,
                        _mm256_sub_pd(vQx, _mm256_set1_pd(C[0])));
                    const __m256d d1 = _mm256_fmadd_pd(vpt, vPQy,
                        _mm256_sub_pd(vQy, _mm256_set1_pd(C[1])));
                    const __m256d d2 = _mm256_fmadd_pd(vpt, vPQz,
                        _mm256_sub_pd(vQz, _mm256_set1_pd(C[2])));
                    for (int cf = 0; cf < NcRows; ++cf) {
                        if (cxs[cf] + cys[cf] + czs[cf] == 0) {
                            acc[cf] = _mm256_add_pd(acc[cf], vww);  // constant row
                        } else {
                            const __m256d sel = cxs[cf] ? d0 : (cys[cf] ? d1 : d2);
                            acc[cf] = _mm256_fmadd_pd(vww, sel, acc[cf]);
                        }
                    }
                }
            }
        }
        // Scalar ket tail (n_kets % 4), accumulated straight into radial.
        for (; ik < n_kets; ++ik) {
            const std::size_t iks = static_cast<std::size_t>(ik);
            const double q = kets.s[iks];
            const double pq = p + q;
            const double inv_pq = 1.0 / pq;
            const double PQ[3] = {bras.Wx[ibs] - kets.Wx[iks],
                                  bras.Wy[ibs] - kets.Wy[iks],
                                  bras.Wz[ibs] - kets.Wz[iks]};
            const double rpq2 = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];
            const double q_over = q * inv_pq;
            const double Targ = p * q_over * rpq2;
            const double pref = wb_pref * kets.w[iks] / std::sqrt(pq);
            double rt, wt;
            rys_roots_weights_fixed<1>(Targ, &rt, &wt);
            const double ww = pref * wt;
            if constexpr (ISum == 0 && KSum == 0) {
                radial[0] += ww;
            } else if constexpr (ISum == 1) {
                const double qt = q_over * rt;
                const double C00[3] = {
                    (bras.Wx[ibs] - A[0]) - qt * PQ[0],
                    (bras.Wy[ibs] - A[1]) - qt * PQ[1],
                    (bras.Wz[ibs] - A[2]) - qt * PQ[2]};
                for (int al = 0; al < NaRows; ++al) {
                    radial[al] += (axs[al] + ays[al] + azs[al] == 0)
                        ? ww : ww * C00[axs[al] ? 0 : (ays[al] ? 1 : 2)];
                }
            } else {
                const double pt = (p * inv_pq) * rt;
                const double D00[3] = {
                    (kets.Wx[iks] - C[0]) + pt * PQ[0],
                    (kets.Wy[iks] - C[1]) + pt * PQ[1],
                    (kets.Wz[iks] - C[2]) + pt * PQ[2]};
                for (int cf = 0; cf < NcRows; ++cf) {
                    radial[cf] += (cxs[cf] + cys[cf] + czs[cf] == 0)
                        ? ww : ww * D00[cxs[cf] ? 0 : (cys[cf] ? 1 : 2)];
                }
            }
        }
    }
    // One horizontal reduction per output component.
    for (int r = 0; r < NaRows * NcRows; ++r) {
        const __m128d lo = _mm256_castpd256_pd128(acc[r]);
        const __m128d hi = _mm256_extractf128_pd(acc[r], 1);
        const __m128d s = _mm_add_pd(lo, hi);
        radial[r] += _mm_cvtsd_f64(_mm_hadd_pd(s, s));
    }
}

// Monomial descriptor for the lean2 kernel: total degree and the (at most two)
// axes carrying it. deg 0 -> {}, deg 1 -> {a0}, deg 2 -> {a0 <= a1}.
struct Lean2Row {
    int deg;
    int a0;
    int a1;
};

inline Lean2Row lean2_classify(int mx, int my, int mz) noexcept {
    Lean2Row row{mx + my + mz, 0, 0};
    if (row.deg == 1) {
        row.a0 = mx ? 0 : (my ? 1 : 2);
    } else if (row.deg == 2) {
        if (mx == 2) { row.a0 = 0; row.a1 = 0; }
        else if (my == 2) { row.a0 = 1; row.a1 = 1; }
        else if (mz == 2) { row.a0 = 2; row.a1 = 2; }
        else if (mx && my) { row.a0 = 0; row.a1 = 1; }
        else if (mx && mz) { row.a0 = 0; row.a1 = 2; }
        else { row.a0 = 1; row.a1 = 2; }
    }
    return row;
}

// Register-resident batch kernel for the Ltot==2 classes (ps|ps, ss|pp,
// pp|ss and their b/d-carrying variants). Every radial cell is a closed-form
// product of per-axis factors: C00, D00, G11 = C00*D00 + B00,
// G20 = C00^2 + B10, G02 = D00^2 + B01 -- so no G table is materialized and
// the per-block L1 footprint is a fixed few hundred bytes independent of the
// molecule size (the SOA batch kernel's array round-trips are what erode the
// large-N ratio, which is the regime that matters for protein-scale targets).
template < int ISum, int KSum, int NRoots, int NaRows, int NcRows,
          class Emit >
inline void accumulate_rys_quartets_lean2_fixed(
    const RysPrimitivePairSlab& bras,
    const RysPrimitivePairSlab& kets,
    const std::array<double, 3>& A,
    const std::array<double, 3>& C,
    const int* __restrict axs,
    const int* __restrict ays,
    const int* __restrict azs,
    const int* __restrict cxs,
    const int* __restrict cys,
    const int* __restrict czs,
    double* __restrict radial) {
    static_assert(ISum + KSum == 2 && NRoots == 2, "lean2 kernel covers Ltot==2");
    // Largest shapes: (0,1|x,y) with cumulative rows both sides -> 4*4 = 16.
    static_assert(NaRows * NcRows <= 16, "unexpected component count");
    const int n_bras = static_cast<int>(bras.s.size());
    const int n_kets = static_cast<int>(kets.s.size());
    if (n_bras == 0 || n_kets == 0) return;

    Lean2Row arow[NaRows], crow[NcRows];
    for (int al = 0; al < NaRows; ++al) arow[al] = lean2_classify(axs[al], ays[al], azs[al]);
    for (int cf = 0; cf < NcRows; ++cf) crow[cf] = lean2_classify(cxs[cf], cys[cf], czs[cf]);

    __m256d acc[NaRows * NcRows];
    for (int r = 0; r < NaRows * NcRows; ++r) acc[r] = _mm256_setzero_pd();
    alignas(32) double targ4[4], rts4[8], wts4[8];
    // Per-axis factor families for the current root (L1-resident scratch).
    alignas(32) double fC[3][4], fD[3][4], fG11[3][4], fG20[3][4], fG02[3][4];

    for (int ib = 0; ib < n_bras; ++ib) {
        const std::size_t ibs = static_cast<std::size_t>(ib);
        const double p = bras.s[ibs];
        const __m256d vp = _mm256_set1_pd(p);
        const __m256d vinv2p = _mm256_set1_pd(0.5 * bras.invs[ibs]);
        const __m256d vPx = _mm256_set1_pd(bras.Wx[ibs]);
        const __m256d vPy = _mm256_set1_pd(bras.Wy[ibs]);
        const __m256d vPz = _mm256_set1_pd(bras.Wz[ibs]);
        const double wb_pref = kTwoPi52 * bras.w[ibs];
        const __m256d vPA[3] = {
            _mm256_set1_pd(bras.Wx[ibs] - A[0]),
            _mm256_set1_pd(bras.Wy[ibs] - A[1]),
            _mm256_set1_pd(bras.Wz[ibs] - A[2])};
        int ik = 0;
        for (; ik + 4 <= n_kets; ik += 4) {
            const std::size_t iks = static_cast<std::size_t>(ik);
            const __m256d vq = _mm256_loadu_pd(kets.s.data() + iks);
            const __m256d vpq = _mm256_add_pd(vp, vq);
            const __m256d vinvpq = _mm256_div_pd(_mm256_set1_pd(1.0), vpq);
            const __m256d vQ[3] = {
                _mm256_loadu_pd(kets.Wx.data() + iks),
                _mm256_loadu_pd(kets.Wy.data() + iks),
                _mm256_loadu_pd(kets.Wz.data() + iks)};
            const __m256d vPQ[3] = {
                _mm256_sub_pd(vPx, vQ[0]),
                _mm256_sub_pd(vPy, vQ[1]),
                _mm256_sub_pd(vPz, vQ[2])};
            const __m256d vrpq2 = _mm256_fmadd_pd(vPQ[0], vPQ[0],
                _mm256_fmadd_pd(vPQ[1], vPQ[1], _mm256_mul_pd(vPQ[2], vPQ[2])));
            const __m256d vqop = _mm256_mul_pd(vq, vinvpq);
            const __m256d vpop = _mm256_mul_pd(vp, vinvpq);
            _mm256_store_pd(targ4, _mm256_mul_pd(_mm256_mul_pd(vp, vqop), vrpq2));
#ifdef TLNO_J_SPEED_PROBE_EXACT_N2_LAGRANGE
            if constexpr (std::is_same_v<Emit, RysDirectJSpeedProbeEmit>) {
                rys_roots_weights_batch4_for_emit<Emit, NRoots>(
                    targ4, rts4, wts4);
            } else
#endif
            {
                if (!rys_roots_weights_batch4<2>(targ4, rts4, wts4)) {
                    double rt[2], wt[2];
                    for (int l = 0; l < 4; ++l) {
                        rys_roots_weights_fixed<2>(targ4[l], rt, wt);
                        rts4[l] = rt[0]; rts4[4 + l] = rt[1];
                        wts4[l] = wt[0]; wts4[4 + l] = wt[1];
                    }
                }
            }
            const __m256d vpref = _mm256_div_pd(
                _mm256_mul_pd(_mm256_set1_pd(wb_pref),
                              _mm256_loadu_pd(kets.w.data() + iks)),
                _mm256_sqrt_pd(vpq));
            const __m256d vinv2q = _mm256_mul_pd(_mm256_set1_pd(0.5),
                _mm256_loadu_pd(kets.invs.data() + iks));
            const __m256d vinv2pq = _mm256_mul_pd(_mm256_set1_pd(0.5), vinvpq);
            const __m256d vQC[3] = {
                _mm256_sub_pd(vQ[0], _mm256_set1_pd(C[0])),
                _mm256_sub_pd(vQ[1], _mm256_set1_pd(C[1])),
                _mm256_sub_pd(vQ[2], _mm256_set1_pd(C[2]))};

            for (int n = 0; n < 2; ++n) {
                const __m256d vt2 = _mm256_load_pd(rts4 + n * 4);
                const __m256d vww = _mm256_mul_pd(vpref, _mm256_load_pd(wts4 + n * 4));
                const __m256d vB00 = _mm256_mul_pd(vinv2pq, vt2);
                const __m256d vB10 = _mm256_mul_pd(vinv2p,
                    _mm256_fnmadd_pd(vqop, vt2, _mm256_set1_pd(1.0)));
                const __m256d vB01 = _mm256_mul_pd(vinv2q,
                    _mm256_fnmadd_pd(vpop, vt2, _mm256_set1_pd(1.0)));
                const __m256d vqt = _mm256_mul_pd(vqop, vt2);
                const __m256d vpt = _mm256_mul_pd(vpop, vt2);
                for (int ax = 0; ax < 3; ++ax) {
                    const __m256d c00 = _mm256_fnmadd_pd(vqt, vPQ[ax], vPA[ax]);
                    const __m256d d00 = _mm256_fmadd_pd(vpt, vPQ[ax], vQC[ax]);
                    _mm256_store_pd(fC[ax], c00);
                    _mm256_store_pd(fD[ax], d00);
                    if constexpr (ISum >= 1 && KSum >= 1) {
                        _mm256_store_pd(fG11[ax], _mm256_fmadd_pd(c00, d00, vB00));
                    }
                    if constexpr (ISum == 2) {
                        _mm256_store_pd(fG20[ax], _mm256_fmadd_pd(c00, c00, vB10));
                    }
                    if constexpr (KSum == 2) {
                        _mm256_store_pd(fG02[ax], _mm256_fmadd_pd(d00, d00, vB01));
                    }
                }
                for (int al = 0; al < NaRows; ++al) {
                    const Lean2Row& ar = arow[al];
                    for (int cf = 0; cf < NcRows; ++cf) {
                        const Lean2Row& cr = crow[cf];
                        __m256d value;
                        if constexpr (ISum == 2) {
                            if (ar.deg == 0) {
                                value = _mm256_set1_pd(1.0);
                            } else if (ar.deg == 1) {
                                value = _mm256_load_pd(fC[ar.a0]);
                            } else {
                                value = (ar.a0 == ar.a1)
                                    ? _mm256_load_pd(fG20[ar.a0])
                                    : _mm256_mul_pd(_mm256_load_pd(fC[ar.a0]),
                                                    _mm256_load_pd(fC[ar.a1]));
                            }
                        } else if constexpr (KSum == 2) {
                            if (cr.deg == 0) {
                                value = _mm256_set1_pd(1.0);
                            } else if (cr.deg == 1) {
                                value = _mm256_load_pd(fD[cr.a0]);
                            } else {
                                value = (cr.a0 == cr.a1)
                                    ? _mm256_load_pd(fG02[cr.a0])
                                    : _mm256_mul_pd(_mm256_load_pd(fD[cr.a0]),
                                                    _mm256_load_pd(fD[cr.a1]));
                            }
                        } else {
                            static_assert(
                                ISum == 1 && KSum == 1,
                                "lean2 kernel supports only (2,0), (1,1), and (0,2)");
                            if (ar.deg == 0 && cr.deg == 0) {
                                value = _mm256_set1_pd(1.0);
                            } else if (ar.deg == 1 && cr.deg == 0) {
                                value = _mm256_load_pd(fC[ar.a0]);
                            } else if (ar.deg == 0 && cr.deg == 1) {
                                value = _mm256_load_pd(fD[cr.a0]);
                            } else {
                                value = (ar.a0 == cr.a0)
                                    ? _mm256_load_pd(fG11[ar.a0])
                                    : _mm256_mul_pd(_mm256_load_pd(fC[ar.a0]),
                                                    _mm256_load_pd(fD[cr.a0]));
                            }
                        }
                        acc[al * NcRows + cf] =
                            _mm256_fmadd_pd(vww, value, acc[al * NcRows + cf]);
                    }
                }
            }
        }
        // Scalar ket tail.
        for (; ik < n_kets; ++ik) {
            const std::size_t iks = static_cast<std::size_t>(ik);
            const double q = kets.s[iks];
            const double pq = p + q;
            const double inv_pq = 1.0 / pq;
            const double PQ[3] = {bras.Wx[ibs] - kets.Wx[iks],
                                  bras.Wy[ibs] - kets.Wy[iks],
                                  bras.Wz[ibs] - kets.Wz[iks]};
            const double rpq2 = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];
            const double q_over = q * inv_pq;
            const double p_over = p * inv_pq;
            const double Targ = p * q_over * rpq2;
            double rt[2], wt[2];
            rys_roots_weights_fixed<2>(Targ, rt, wt);
            const double pref = wb_pref * kets.w[iks] / std::sqrt(pq);
            const double i2p = 0.5 * bras.invs[ibs];
            const double i2q = 0.5 * kets.invs[iks];
            const double i2pq = 0.5 * inv_pq;
            for (int n = 0; n < 2; ++n) {
                const double t2 = rt[n];
                const double ww = pref * wt[n];
                const double B00s = i2pq * t2;
                const double B10s = i2p * (1.0 - q_over * t2);
                const double B01s = i2q * (1.0 - p_over * t2);
                double sC[3], sD[3];
                for (int ax = 0; ax < 3; ++ax) {
                    const double base_a = (ax == 0 ? bras.Wx[ibs] - A[0]
                                          : ax == 1 ? bras.Wy[ibs] - A[1]
                                                    : bras.Wz[ibs] - A[2]);
                    const double base_c = (ax == 0 ? kets.Wx[iks] - C[0]
                                          : ax == 1 ? kets.Wy[iks] - C[1]
                                                    : kets.Wz[iks] - C[2]);
                    sC[ax] = base_a - q_over * t2 * PQ[ax];
                    sD[ax] = base_c + p_over * t2 * PQ[ax];
                }
                for (int al = 0; al < NaRows; ++al) {
                    const Lean2Row& ar = arow[al];
                    for (int cf = 0; cf < NcRows; ++cf) {
                        const Lean2Row& cr = crow[cf];
                        double value;
                        if (ar.deg == 0 && cr.deg == 0) value = 1.0;
                        else if (ar.deg == 1 && cr.deg == 0) value = sC[ar.a0];
                        else if (ar.deg == 0 && cr.deg == 1) value = sD[cr.a0];
                        else if (ar.deg == 1 && cr.deg == 1)
                            value = (ar.a0 == cr.a0) ? sC[ar.a0] * sD[cr.a0] + B00s
                                                     : sC[ar.a0] * sD[cr.a0];
                        else if (ar.deg == 2)
                            value = (ar.a0 == ar.a1) ? sC[ar.a0] * sC[ar.a0] + B10s
                                                     : sC[ar.a0] * sC[ar.a1];
                        else
                            value = (cr.a0 == cr.a1) ? sD[cr.a0] * sD[cr.a0] + B01s
                                                     : sD[cr.a0] * sD[cr.a1];
                        radial[al * NcRows + cf] += ww * value;
                    }
                }
            }
        }
    }
    for (int r = 0; r < NaRows * NcRows; ++r) {
        const __m128d lo = _mm256_castpd256_pd128(acc[r]);
        const __m128d hi = _mm256_extractf128_pd(acc[r], 1);
        const __m128d s = _mm_add_pd(lo, hi);
        radial[r] += _mm_cvtsd_f64(_mm_hadd_pd(s, s));
    }
}
#endif  // __AVX2__ && __FMA__

template < int ISum, int KSum, int NRoots, int GStride, int GSize,
          int NaRows, int NcRows >
inline void accumulate_rys_quartets_scalar_fixed(
    const std::vector<RysPrimitivePair>& bras,
    const std::vector<RysPrimitivePair>& kets,
    const std::array<double, 3>& A,
    const std::array<double, 3>& C,
    const int* __restrict axs,
    const int* __restrict ays,
    const int* __restrict azs,
    const int* __restrict cxs,
    const int* __restrict cys,
    const int* __restrict czs,
    double* __restrict radial) {
    double Gx[GSize], Gy[GSize], Gz[GSize];
    double* Gax[3] = {Gx, Gy, Gz};
    double roots[NRoots], weights[NRoots];

    for (const RysPrimitivePair& bp : bras) {
        const double p = bp.s;
        const double invp = bp.invs;
        const double P[3] = {bp.Wx, bp.Wy, bp.Wz};
        for (const RysPrimitivePair& kp : kets) {
            const double q = kp.s;
            const double invq = kp.invs;
            const double Q[3] = {kp.Wx, kp.Wy, kp.Wz};
            const double pq = p + q;
            const double inv_pq = 1.0 / pq;
            const double PQ[3] = {P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]};
            const double rpq2 = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];
            const double rho = p * q * inv_pq;
            const double Targ = rho * rpq2;
            const double prefactor = kTwoPi52 * bp.w * kp.w / std::sqrt(pq);

            rys_roots_weights_fixed<NRoots>(Targ, roots, weights);

            const double q_over_pq = q * inv_pq;
            const double p_over_pq = p * inv_pq;
            const double inv2p = 0.5 * invp;
            const double inv2q = 0.5 * invq;
            const double inv2pq = 0.5 * inv_pq;

            for (int n = 0; n < NRoots; ++n) {
                const double t2 = roots[n];
                const double ww = prefactor * weights[n];
                const double B00 = inv2pq * t2;
                const double B10 = inv2p * (1.0 - q_over_pq * t2);
                const double B01 = inv2q * (1.0 - p_over_pq * t2);

                for (int ax = 0; ax < 3; ++ax) {
                    double* G = Gax[ax];
                    const double C00 = (P[ax] - A[ax]) - q_over_pq * PQ[ax] * t2;
                    const double D00 = (Q[ax] - C[ax]) + p_over_pq * PQ[ax] * t2;
                    G[0] = 1.0;
                    if constexpr (ISum >= 1) {
                        G[GStride] = C00;
                        for (int i = 1; i < ISum; ++i) {
                            G[(i + 1) * GStride] =
                                C00 * G[i * GStride]
                                + static_cast<double>(i) * B10 * G[(i - 1) * GStride];
                        }
                    }
                    for (int k = 0; k < KSum; ++k) {
                        for (int i = 0; i <= ISum; ++i) {
                            double v = D00 * G[i * GStride + k];
                            if (i >= 1) {
                                v += static_cast<double>(i) * B00 * G[(i - 1) * GStride + k];
                            }
                            if (k >= 1) {
                                v += static_cast<double>(k) * B01 * G[i * GStride + (k - 1)];
                            }
                            G[i * GStride + (k + 1)] = v;
                        }
                    }
                }
                for (int al = 0; al < NaRows; ++al) {
                    const double* __restrict Gxr = Gx + axs[al] * GStride;
                    const double* __restrict Gyr = Gy + ays[al] * GStride;
                    const double* __restrict Gzr = Gz + azs[al] * GStride;
                    double* __restrict row = radial + static_cast<std::size_t>(al) * NcRows;
#pragma omp simd
                    for (int cf = 0; cf < NcRows; ++cf) {
                        row[cf] += ww * Gxr[cxs[cf]] * Gyr[cys[cf]] * Gzr[czs[cf]];
                    }
                }
            }
        }
    }
}

inline void accumulate_rys_quartets_soa_runtime(
    const RysPrimitivePairSlab& bras,
    const RysPrimitivePairSlab& kets,
    const std::array<double, 3>& A,
    const std::array<double, 3>& C,
    int isum,
    int ksum,
    int nroots,
    int gstride,
    int na_rows,
    int nc_rows,
    const int* __restrict axs,
    const int* __restrict ays,
    const int* __restrict azs,
    const int* __restrict cxs,
    const int* __restrict cys,
    const int* __restrict czs,
    double* __restrict radial) {
    constexpr int B = kRysSoaBatch;
    const int n_bras = bras.size();
    const int n_kets = kets.size();
    if (n_bras == 0 || n_kets == 0) return;
    const int n_primitive_quartets = n_bras * n_kets;

    constexpr int kG = kMaxLSum + 1;
    double Gx[kG * kG * B], Gy[kG * kG * B], Gz[kG * kG * B];
    double roots[kRysMaxRoots * B], weights[kRysMaxRoots * B];

    double Px[B], Py[B], Pz[B], Qx[B], Qy[B], Qz[B];
    double PQx[B], PQy[B], PQz[B], prefactor[B];
    double q_over_pq[B], p_over_pq[B], inv2p[B], inv2q[B], inv2pq[B];

    double ww[B], B00[B], B10[B], B01[B];
    double C00x[B], C00y[B], C00z[B], D00x[B], D00y[B], D00z[B];

    for (int batch_start = 0; batch_start < n_primitive_quartets; batch_start += B) {
        const int valid = std::min(B, n_primitive_quartets - batch_start);
        for (int lane = 0; lane < B; ++lane) {
            const int flat = batch_start + ((lane < valid) ? lane : valid - 1);
            const int ib = flat / n_kets;
            const int ik = flat - ib * n_kets;
            const std::size_t ibs = static_cast<std::size_t>(ib);
            const std::size_t iks = static_cast<std::size_t>(ik);

            const double p = bras.s[ibs];
            const double q = kets.s[iks];
            const double pq = p + q;
            const double inv_pq = 1.0 / pq;
            Px[lane] = bras.Wx[ibs];
            Py[lane] = bras.Wy[ibs];
            Pz[lane] = bras.Wz[ibs];
            Qx[lane] = kets.Wx[iks];
            Qy[lane] = kets.Wy[iks];
            Qz[lane] = kets.Wz[iks];
            PQx[lane] = Px[lane] - Qx[lane];
            PQy[lane] = Py[lane] - Qy[lane];
            PQz[lane] = Pz[lane] - Qz[lane];
            q_over_pq[lane] = q * inv_pq;
            p_over_pq[lane] = p * inv_pq;
            inv2p[lane] = 0.5 * bras.invs[ibs];
            inv2q[lane] = 0.5 * kets.invs[iks];
            inv2pq[lane] = 0.5 * inv_pq;
            prefactor[lane] = (lane < valid)
                                  ? kTwoPi52 * bras.w[ibs] * kets.w[iks] / std::sqrt(pq)
                                  : 0.0;

            const double rpq2 = PQx[lane] * PQx[lane]
                                + PQy[lane] * PQy[lane]
                                + PQz[lane] * PQz[lane];
            const double Targ = (p * q * inv_pq) * rpq2;
            double rtmp[kRysMaxRoots], wtmp[kRysMaxRoots];
            rys_roots_weights(nroots, Targ, rtmp, wtmp);
            for (int n = 0; n < nroots; ++n) {
                roots[n * B + lane] = rtmp[n];
                weights[n * B + lane] = wtmp[n];
            }
        }

        for (int n = 0; n < nroots; ++n) {
#pragma omp simd
            for (int lane = 0; lane < B; ++lane) {
                const double t2 = roots[n * B + lane];
                ww[lane] = prefactor[lane] * weights[n * B + lane];
                B00[lane] = inv2pq[lane] * t2;
                B10[lane] = inv2p[lane] * (1.0 - q_over_pq[lane] * t2);
                B01[lane] = inv2q[lane] * (1.0 - p_over_pq[lane] * t2);
                C00x[lane] = (Px[lane] - A[0]) - q_over_pq[lane] * PQx[lane] * t2;
                C00y[lane] = (Py[lane] - A[1]) - q_over_pq[lane] * PQy[lane] * t2;
                C00z[lane] = (Pz[lane] - A[2]) - q_over_pq[lane] * PQz[lane] * t2;
                D00x[lane] = (Qx[lane] - C[0]) + p_over_pq[lane] * PQx[lane] * t2;
                D00y[lane] = (Qy[lane] - C[1]) + p_over_pq[lane] * PQy[lane] * t2;
                D00z[lane] = (Qz[lane] - C[2]) + p_over_pq[lane] * PQz[lane] * t2;
            }

            build_rys_2d_axis_soa_runtime(isum, ksum, gstride, Gx, C00x, D00x, B00, B10, B01);
            build_rys_2d_axis_soa_runtime(isum, ksum, gstride, Gy, C00y, D00y, B00, B10, B01);
            build_rys_2d_axis_soa_runtime(isum, ksum, gstride, Gz, C00z, D00z, B00, B10, B01);

            for (int al = 0; al < na_rows; ++al) {
                double* __restrict row = radial + static_cast<std::size_t>(al) * nc_rows;
                const int gx_a = axs[al] * gstride;
                const int gy_a = ays[al] * gstride;
                const int gz_a = azs[al] * gstride;
                for (int cf = 0; cf < nc_rows; ++cf) {
                    const double* __restrict gx = Gx + (gx_a + cxs[cf]) * B;
                    const double* __restrict gy = Gy + (gy_a + cys[cf]) * B;
                    const double* __restrict gz = Gz + (gz_a + czs[cf]) * B;
                    double sum = 0.0;
#pragma omp simd reduction(+:sum)
                    for (int lane = 0; lane < B; ++lane) {
                        sum += ww[lane] * gx[lane] * gy[lane] * gz[lane];
                    }
                    row[cf] += sum;
                }
            }
        }
    }
}

// ── Per-quartet driver (Rys quadrature) ---------------------------------------
// Same output contract as compute_quartet_block: produces the raw-Cartesian
// [Na_block × Nb_block × Nc_block × Nd_block] block (before N_A scaling).
// Builds per-primitive (a0|c0) via Rys 2D integrals, accumulates into the same
// T_radial layout as the HGP path, then reuses hrr_ket / hrr_bra unchanged.
void compute_quartet_block_rys_impl(const ContractedShellGroupH& gA,
                                    const ContractedShellGroupH& gB,
                                    const ContractedShellGroupH& gC,
                                    const ContractedShellGroupH& gD,
                                    const std::vector<RysPrimitivePair>* cached_bras,
                                    const std::vector<RysPrimitivePair>* cached_kets,
                                    const RysPrimitivePairSlab* cached_bra_slab,
                                    const RysPrimitivePairSlab* cached_ket_slab,
                                    std::vector<double>& block_radial) {
    const int La = gA.L, Lb = gB.L, Lc = gC.L, Ld = gD.L;
    const int La_sum = La + Lb;
    const int Lc_sum = Lc + Ld;
    const int Ltot   = La_sum + Lc_sum;
    const int nroots = Ltot / 2 + 1;

    const int Na_block = n_cart(La);
    const int Nb_block = n_cart(Lb);
    const int Nc_block = n_cart(Lc);
    const int Nd_block = n_cart(Ld);
    block_radial.assign(static_cast<std::size_t>(Na_block) * Nb_block
                            * Nc_block * Nd_block, 0.0);

    const int a_lo_offset   = offset_for_L(La);
    const int Na_full       = n_cart_cum(La_sum);
    const int Na_block_slice = Na_full - a_lo_offset;
    const int Nc_full       = n_cart_cum(Lc_sum);

    thread_local std::vector<double> T_radial;
    T_radial.assign(static_cast<std::size_t>(Na_block_slice) * Nc_full, 0.0);

    // Flat Cartesian-exponent index arrays for the contraction hot loop, so it
    // does not reload the g_cart {lx,ly,lz} structs on every primitive/root.
    constexpr int kCMax = CartTriplesTable::kMaxStored;  // 165
    int axs[kCMax], ays[kCMax], azs[kCMax];
    for (int al = 0; al < Na_block_slice; ++al) {
        const auto& at = g_cart.at(a_lo_offset + al);
        axs[al] = at[0]; ays[al] = at[1]; azs[al] = at[2];
    }
    int cxs[kCMax], cys[kCMax], czs[kCMax];
    for (int cf = 0; cf < Nc_full; ++cf) {
        const auto& ct = g_cart.at(cf);
        cxs[cf] = ct[0]; cys[cf] = ct[1]; czs[cf] = ct[2];
    }

    const auto& A = gA.center;
    const auto& B = gB.center;
    const auto& C = gC.center;
    const auto& D = gD.center;
    const std::array<double, 3> AB = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
    const std::array<double, 3> CD = {C[0] - D[0], C[1] - D[1], C[2] - D[2]};

    const int gstride = Lc_sum + 1;

    thread_local std::vector<RysPrimitivePair> bras_scratch, kets_scratch;
    thread_local RysPrimitivePairSlab bra_slab_scratch, ket_slab_scratch;
    const std::vector<RysPrimitivePair>* bras_ptr = cached_bras;
    const std::vector<RysPrimitivePair>* kets_ptr = cached_kets;
    const RysPrimitivePairSlab* bra_slab_ptr = cached_bra_slab;
    const RysPrimitivePairSlab* ket_slab_ptr = cached_ket_slab;
    if (bras_ptr == nullptr) {
        bras_scratch = build_rys_primitive_pairs_h(gA, gB);
        bras_ptr = &bras_scratch;
    }
    if (kets_ptr == nullptr) {
        kets_scratch = build_rys_primitive_pairs_h(gC, gD);
        kets_ptr = &kets_scratch;
    }
    const auto& bras = *bras_ptr;
    const auto& kets = *kets_ptr;

    // The runtime SoA kernel consumes pretransposed primitive-pair slabs.
    if (bra_slab_ptr == nullptr) {
        bra_slab_scratch = build_rys_primitive_pair_slab_h(bras);
        bra_slab_ptr = &bra_slab_scratch;
    }
    if (ket_slab_ptr == nullptr) {
        ket_slab_scratch = build_rys_primitive_pair_slab_h(kets);
        ket_slab_ptr = &ket_slab_scratch;
    }
    accumulate_rys_quartets_soa_runtime(
        *bra_slab_ptr, *ket_slab_ptr, A, C, La_sum, Lc_sum, nroots, gstride,
        Na_block_slice, Nc_full, axs, ays, azs, cxs, cys, czs, T_radial.data());

    thread_local std::vector<double> K_out;
    thread_local std::vector<double> B_out;
    hrr_ket(Lc, Ld, CD, T_radial.data(), Nc_full, Na_block_slice, K_out);
    hrr_bra(La, Lb, AB, K_out, Nc_block, Nd_block, B_out);
    block_radial.swap(B_out);
}

template <int La, int Lb, int Lc, int Ld>
inline double eval_rys_lowd_hrr_raw(const double* __restrict T_radial,
                                    int NcFull,
                                    const double* __restrict AB,
                                    const double* __restrict CD,
                                    int a_local,
                                    int b_local,
                                    int c_local,
                                    int d_local) noexcept {
    static_assert(Lb >= 0 && Lb <= 2, "low-d bra HRR supports Lb <= 2");
    static_assert(Ld >= 0 && Ld <= 2, "low-d ket HRR supports Ld <= 2");
    constexpr int AOffset = (La == 0) ? 0 : La * (La + 1) * (La + 2) / 6;
    constexpr int COffset = (Lc == 0) ? 0 : Lc * (Lc + 1) * (Lc + 2) / 6;
    const auto& a = g_cart.at(AOffset + a_local);
    const auto& b = g_cart.at(offset_for_L(Lb) + b_local);
    const auto& c = g_cart.at(COffset + c_local);
    const auto& d = g_cart.at(offset_for_L(Ld) + d_local);

    double raw = 0.0;
    for (int ux = 0; ux <= b[0]; ++ux) {
        const int bx_coef = rys_small_binom(b[0], ux);
        const double abx = rys_small_pow(AB[0], b[0] - ux);
        for (int uy = 0; uy <= b[1]; ++uy) {
            const int bxy_coef = bx_coef * rys_small_binom(b[1], uy);
            const double abxy = abx * rys_small_pow(AB[1], b[1] - uy);
            for (int uz = 0; uz <= b[2]; ++uz) {
                const int b_coef = bxy_coef * rys_small_binom(b[2], uz);
                const double ab_fac = abxy * rys_small_pow(AB[2], b[2] - uz);
                const int arow = cart_flat_idx(a[0] + ux, a[1] + uy, a[2] + uz) - AOffset;
                const double* __restrict t_row =
                    T_radial + static_cast<std::size_t>(arow) * static_cast<std::size_t>(NcFull);
                for (int vx = 0; vx <= d[0]; ++vx) {
                    const int dx_coef = rys_small_binom(d[0], vx);
                    const double cdx = rys_small_pow(CD[0], d[0] - vx);
                    for (int vy = 0; vy <= d[1]; ++vy) {
                        const int dxy_coef = dx_coef * rys_small_binom(d[1], vy);
                        const double cdxy = cdx * rys_small_pow(CD[1], d[1] - vy);
                        for (int vz = 0; vz <= d[2]; ++vz) {
                            const int d_coef = dxy_coef * rys_small_binom(d[2], vz);
                            const double cd_fac = cdxy * rys_small_pow(CD[2], d[2] - vz);
                            const int cflat = cart_flat_idx(c[0] + vx, c[1] + vy, c[2] + vz);
                            raw += static_cast<double>(b_coef * d_coef) * ab_fac * cd_fac * t_row[cflat];
                        }
                    }
                }
            }
        }
    }
    return raw;
}

// ============================================================================
// FUSED per-class d/f kernel (libcint _g0_2d4d style): the Horizontal
// Recurrence (HRR) is FOLDED into the per-AXIS 2D build as 1D transfers, so
// there is no separate 3D radial intermediate and no per-component HRR pass.
// Scalar-per-quartet (SIMD lives over the Rys roots / the contract), which
// avoids the cross-quartet deinterleave + L1 bloat that sank the lane=quartet
// attempt.  Produces the final (ab|cd) Cartesian block DIRECTLY in the engine's
// scatter/GeneratedMap layout
//     out[(a*Nb+b)*(Nc*Nd) + (c*Nd+d)]  ==  out[lhs_component*(Nc*Nd)+rhs_component]
// using the engine's EXACT prefactor + 2D recurrence (so it matches the rys
// path bit-for-bit to rounding); the scatter then just reads raw = out[idx].
// ----------------------------------------------------------------------------
// One axis, one root: VRR 2D -> bra-HRR (i->j) -> ket-HRR (k->l) into I.
// I layout: ((i*(Lb+1)+j)*GK + k)*(Ld+1) + l   with GK = Lc+Ld+1 (k full range).
template <int La, int Lb, int Lc, int Ld>
static inline void fused_axis_build(double C00, double D00, double B00,
                                    double B10, double B01, double AB, double CD,
                                    double* __restrict I, int r, double seed = 1.0) {
    constexpr int LAB = La + Lb, LCD = Lc + Ld, GK = LCD + 1;
    constexpr int NR = (La + Lb + Lc + Ld) / 2 + 1;            // root-innermost stride
    double g[(LAB + 1) * GK];                                   // VRR 2D scratch
    g[0] = seed;
    for (int i = 0; i < LAB; ++i)                               // raise bra i (0..LAB)
        g[(i + 1) * GK] = C00 * g[i * GK] + (i ? i * B10 * g[(i - 1) * GK] : 0.0);
    for (int kk = 0; kk < LCD; ++kk)                           // raise ket k (0..LCD)
        for (int i = 0; i <= LAB; ++i) {
            double v = D00 * g[i * GK + kk];                    //   k+1 from k
            if (i)  v += i  * B00 * g[(i - 1) * GK + kk];       //   + i * B00 * (i-1,k)
            if (kk) v += kk * B01 * g[i * GK + kk - 1];         //   + k * B01 * (i,k-1)
            g[i * GK + kk + 1] = v;
        }
    double Bb[(LAB + 1) * (Lb + 1) * GK];                       // bra-HRR: i full (0..LAB)
    for (int i = 0; i <= LAB; ++i)                              // j=0 plane = VRR copy
        for (int kk = 0; kk <= LCD; ++kk)
            Bb[(i * (Lb + 1) + 0) * GK + kk] = g[i * GK + kk];
    for (int j = 1; j <= Lb; ++j)                              // (a,b)=(a+1,b-1)+AB*(a,b-1)
        for (int i = 0; i <= LAB - j; ++i)                     //   i-range shrinks with j
            for (int kk = 0; kk <= LCD; ++kk)
                Bb[(i * (Lb + 1) + j) * GK + kk] =
                    Bb[((i + 1) * (Lb + 1) + (j - 1)) * GK + kk] +
                    AB * Bb[(i * (Lb + 1) + (j - 1)) * GK + kk];
    for (int i = 0; i <= La; ++i)                              // ket-HRR l=0 plane = Bb
        for (int j = 0; j <= Lb; ++j)
            for (int kk = 0; kk <= LCD; ++kk)
                I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + 0) * NR + r] =
                    Bb[(i * (Lb + 1) + j) * GK + kk];
    for (int l = 1; l <= Ld; ++l)                             // (c,d)=(c+1,d-1)+CD*(c,d-1)
        for (int i = 0; i <= La; ++i)                          //   k-range shrinks with l
            for (int j = 0; j <= Lb; ++j)
                for (int kk = 0; kk <= LCD - l; ++kk)
                    I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + l) * NR + r] =
                        I[(((i * (Lb + 1) + j) * GK + (kk + 1)) * (Ld + 1) + (l - 1)) * NR + r] +
                        CD * I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + (l - 1)) * NR + r];
}

// ── AVX2 root-SIMD 2D build ──────────────────────────────────────────────────
// Carry the Rys roots as SIMD lanes through the VRR+HRR recurrence with EXPLICIT
// intrinsics (NOT #pragma omp simd, which lost).  4-wide AVX2 for full chunks of
// 4 roots + a scalar remainder; tight stride NRoots (no cache-hostile padding).
// Writes G root-innermost (G[iidx*NRoots + r]).  Microbench: ~2-3.8x over scalar.
// The selected implementation is the compact loopy AVX recurrence. The
// generated scalar/AVX variants were larger and neutral-to-slower.
#if defined(__AVX2__) && defined(__FMA__)
template <int W> struct RysVec;
template <>
struct RysVec<1> {
    using T = double;
    static T ld(const double* p) { return *p; }
    static void st(double* p, T v) { *p = v; }
    static T set1(double x) { return x; }
    static T mul(T a, T b) { return a * b; }
    static T fma(T a, T b, T c) { return a * b + c; }
};

template <>
struct RysVec<4> {
    using T = __m256d;
    static T ld(const double* p) { return _mm256_loadu_pd(p); }
    static void st(double* p, T v) { _mm256_storeu_pd(p, v); }
    static T set1(double x) { return _mm256_set1_pd(x); }
    static T mul(T a, T b) { return _mm256_mul_pd(a, b); }
    static T fma(T a, T b, T c) { return _mm256_fmadd_pd(a, b, c); }
};

// Build W roots (lanes base..base+W-1) of one axis into G[iidx*STRIDE + base + lane].
// Wt (optional): if non-null, seed g[0]=Wt[base..] so the whole axis is scaled by
// the per-root weight for free (used to fold the contract weight into Gz).
template <int La, int Lb, int Lc, int Ld, int W, int STRIDE>
static inline void rys_build_chunk(const double* C00, const double* D00, const double* B00,
        const double* B10, const double* B01, double AB, double CD, double* __restrict I, int base,
        const double* Wt = nullptr) {
    using VV = RysVec<W>; using T = typename VV::T;
    constexpr int LAB = La + Lb, LCD = Lc + Ld, GK = LCD + 1;
    T g[(LAB + 1) * GK], Bb[(LAB + 1) * (Lb + 1) * GK];
    const T c00 = VV::ld(C00 + base), d00 = VV::ld(D00 + base), b00 = VV::ld(B00 + base),
            b10 = VV::ld(B10 + base), b01 = VV::ld(B01 + base);
    g[0] = Wt ? VV::ld(Wt + base) : VV::set1(1.0);
    for (int i = 0; i < LAB; ++i)
        g[(i + 1) * GK] = i ? VV::fma(VV::set1((double)i), VV::mul(b10, g[(i - 1) * GK]), VV::mul(c00, g[i * GK]))
                            : VV::mul(c00, g[i * GK]);
    for (int kk = 0; kk < LCD; ++kk) for (int i = 0; i <= LAB; ++i) {
        T v = VV::mul(d00, g[i * GK + kk]);
        if (i)  v = VV::fma(VV::set1((double)i),  VV::mul(b00, g[(i - 1) * GK + kk]), v);
        if (kk) v = VV::fma(VV::set1((double)kk), VV::mul(b01, g[i * GK + kk - 1]),  v);
        g[i * GK + kk + 1] = v;
    }
    for (int i = 0; i <= LAB; ++i) for (int kk = 0; kk <= LCD; ++kk) Bb[(i * (Lb + 1)) * GK + kk] = g[i * GK + kk];
    const T ab = VV::set1(AB), cd = VV::set1(CD);
    for (int j = 1; j <= Lb; ++j) for (int i = 0; i <= LAB - j; ++i) for (int kk = 0; kk <= LCD; ++kk)
        Bb[(i * (Lb + 1) + j) * GK + kk] = VV::fma(ab, Bb[(i * (Lb + 1) + (j - 1)) * GK + kk],
                                                       Bb[((i + 1) * (Lb + 1) + (j - 1)) * GK + kk]);
    for (int i = 0; i <= La; ++i) for (int j = 0; j <= Lb; ++j) for (int kk = 0; kk <= LCD; ++kk)
        VV::st(&I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1)) * STRIDE + base], Bb[(i * (Lb + 1) + j) * GK + kk]);
    for (int l = 1; l <= Ld; ++l) for (int i = 0; i <= La; ++i) for (int j = 0; j <= Lb; ++j) for (int kk = 0; kk <= LCD - l; ++kk) {
        const T a = VV::ld(&I[(((i * (Lb + 1) + j) * GK + (kk + 1)) * (Ld + 1) + (l - 1)) * STRIDE + base]);
        const T b = VV::ld(&I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + (l - 1)) * STRIDE + base]);
        VV::st(&I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + l) * STRIDE + base], VV::fma(cd, b, a));
    }
}
// One axis, all NRoots: AVX 4-chunks + scalar remainder (stride = NRoots, tight).
template <int La, int Lb, int Lc, int Ld, int NRoots>
static inline void rys_build_axis_avx(const double* C00, const double* D00, const double* B00,
        const double* B10, const double* B01, double AB, double CD, double* __restrict G,
        const double* Wt = nullptr) {
    int b = 0;
    for (; b + 4 <= NRoots; b += 4)   // AVX chunk (loopy)
        rys_build_chunk<La, Lb, Lc, Ld, 4, NRoots>(C00, D00, B00, B10, B01, AB, CD, G, b, Wt);
    for (; b < NRoots; ++b)           // remainder root (loopy)
        rys_build_chunk<La, Lb, Lc, Ld, 1, NRoots>(C00, D00, B00, B10, B01, AB, CD, G, b, Wt);
}
#endif

// ── SIMD gout (the contract) ────────────────────────────────────────────────
// libcint-style contract: the per-axis 2D values are stored ROOT-INNERMOST
// (Gx[component*NR + r], roots contiguous).  Sum over roots is done 2-wide (SSE),
// processing TWO output components per iteration and combining their partial sums
// with one hadd.  This beats a 4-wide-over-roots loop because the typical
// nroots (3..7) leaves awkward scalar/mask tails for 4-wide (measured: 4-wide is
// up to 27% slower on ff|ff); 2-wide + 2-components/iter has tiny tails and low
// register pressure.  ~2.4x faster than the old gather-over-o contract.
// Accumulates into out over primitive pairs.  Falls back to scalar without SSE.
#if defined(__AVX2__) && defined(__FMA__)
static inline double hsum128(__m128d v) noexcept {
    return _mm_cvtsd_f64(_mm_add_sd(v, _mm_unpackhi_pd(v, v)));
}
// One output component's root-sum, 2-wide.
static inline __m128d gout_pair_sum(const double* __restrict ax, const double* __restrict ay,
                                    const double* __restrict az, const double* __restrict wt,
                                    int NR) noexcept {
    __m128d a = _mm_setzero_pd();
    int r = 0;
    for (; r + 2 <= NR; r += 2) {
        const __m128d prod = _mm_mul_pd(_mm_mul_pd(_mm_loadu_pd(ax + r), _mm_loadu_pd(ay + r)),
                                        _mm_loadu_pd(az + r));
        a = _mm_fmadd_pd(prod, _mm_loadu_pd(wt + r), a);
    }
    return a;  // caller adds the odd-NR remainder
}
// Weight-folded twin: azw already carries the per-root weight, so the inner
// product is gx*gy (1 mul) then fma(prod, azw, acc) (1 fma) over 3 streams.
static inline __m128d gout_pair_sum_wfold(const double* __restrict ax, const double* __restrict ay,
                                          const double* __restrict azw, int NR) noexcept {
    __m128d a = _mm_setzero_pd();
    int r = 0;
    for (; r + 2 <= NR; r += 2) {
        const __m128d prod = _mm_mul_pd(_mm_loadu_pd(ax + r), _mm_loadu_pd(ay + r));
        a = _mm_fmadd_pd(prod, _mm_loadu_pd(azw + r), a);
    }
    return a;  // caller adds the odd-NR remainder
}
#endif

// Gzw carries the per-root weight (seeded into
// the Z 2D-build base), so no wt stream is loaded and the inner product is
// gx*gy*(wt*gz) = 1 mul + 1 fma over 3 streams.
template <int NR>
static inline void simd_gout_acc_wfold(const double* __restrict Gx,
                                       const double* __restrict Gy,
                                       const double* __restrict Gzw,
                                       const int* __restrict ix, const int* __restrict iy,
                                       const int* __restrict iz, int nout,
                                       double* __restrict out) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    int o = 0;
    for (; o + 2 <= nout; o += 2) {
        const double* ax0 = Gx + std::size_t(ix[o]) * NR;
        const double* ay0 = Gy + std::size_t(iy[o]) * NR;
        const double* az0 = Gzw + std::size_t(iz[o]) * NR;
        const double* ax1 = Gx + std::size_t(ix[o + 1]) * NR;
        const double* ay1 = Gy + std::size_t(iy[o + 1]) * NR;
        const double* az1 = Gzw + std::size_t(iz[o + 1]) * NR;
        __m128d s = _mm_hadd_pd(gout_pair_sum_wfold(ax0, ay0, az0, NR),
                                gout_pair_sum_wfold(ax1, ay1, az1, NR));
        if (NR & 1) {                                          // odd-root remainder
            const int r = NR - 1;
            s = _mm_add_pd(s, _mm_set_pd(az1[r] * ax1[r] * ay1[r],
                                         az0[r] * ax0[r] * ay0[r]));
        }
        _mm_storeu_pd(out + o, _mm_add_pd(_mm_loadu_pd(out + o), s));
    }
    if (o < nout) {                                            // tail component
        const double* ax = Gx + std::size_t(ix[o]) * NR;
        const double* ay = Gy + std::size_t(iy[o]) * NR;
        const double* az = Gzw + std::size_t(iz[o]) * NR;
        double sd = hsum128(gout_pair_sum_wfold(ax, ay, az, NR));
        if (NR & 1) { const int r = NR - 1; sd += az[r] * ax[r] * ay[r]; }
        out[o] += sd;
    }
#else
    for (int o = 0; o < nout; ++o) {
        const double* ax = Gx + std::size_t(ix[o]) * NR;
        const double* ay = Gy + std::size_t(iy[o]) * NR;
        const double* az = Gzw + std::size_t(iz[o]) * NR;
        double s = 0.0;
        for (int r = 0; r < NR; ++r) s += ax[r] * ay[r] * az[r];
        out[o] += s;
    }
#endif
}

// Runtime-NR twin for the high-L fallback (ff|ff etc.).
static inline void simd_gout_acc_rt(const double* __restrict Gx,
                                    const double* __restrict Gy,
                                    const double* __restrict Gz,
                                    const double* __restrict wt, int NR,
                                    const int* __restrict ix, const int* __restrict iy,
                                    const int* __restrict iz, int nout,
                                    double* __restrict out) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    int o = 0;
    for (; o + 2 <= nout; o += 2) {
        const double* ax0 = Gx + std::size_t(ix[o]) * NR, *ay0 = Gy + std::size_t(iy[o]) * NR, *az0 = Gz + std::size_t(iz[o]) * NR;
        const double* ax1 = Gx + std::size_t(ix[o + 1]) * NR, *ay1 = Gy + std::size_t(iy[o + 1]) * NR, *az1 = Gz + std::size_t(iz[o + 1]) * NR;
        __m128d s = _mm_hadd_pd(gout_pair_sum(ax0, ay0, az0, wt, NR),
                                gout_pair_sum(ax1, ay1, az1, wt, NR));
        if (NR & 1) {
            const int r = NR - 1;
            s = _mm_add_pd(s, _mm_set_pd(wt[r] * ax1[r] * ay1[r] * az1[r],
                                         wt[r] * ax0[r] * ay0[r] * az0[r]));
        }
        _mm_storeu_pd(out + o, _mm_add_pd(_mm_loadu_pd(out + o), s));
    }
    if (o < nout) {
        const double* ax = Gx + std::size_t(ix[o]) * NR, *ay = Gy + std::size_t(iy[o]) * NR, *az = Gz + std::size_t(iz[o]) * NR;
        double sd = hsum128(gout_pair_sum(ax, ay, az, wt, NR));
        if (NR & 1) { const int r = NR - 1; sd += wt[r] * ax[r] * ay[r] * az[r]; }
        out[o] += sd;
    }
#else
    for (int o = 0; o < nout; ++o) {
        const double* ax = Gx + std::size_t(ix[o]) * NR, *ay = Gy + std::size_t(iy[o]) * NR, *az = Gz + std::size_t(iz[o]) * NR;
        double s = 0.0;
        for (int r = 0; r < NR; ++r) s += wt[r] * ax[r] * ay[r] * az[r];
        out[o] += s;
    }
#endif
}

// Full fused (ab|cd) block summed over every primitive pair.  out is sized
// Na*Nb*Nc*Nd and indexed out[(a*Nb+b)*(Nc*Nd)+(c*Nd+d)] (scatter layout).
template <int La, int Lb, int Lc, int Ld, int NRoots>
static void fused_rys_block(const std::array<double, 3>& A,
                            const std::array<double, 3>& B,
                            const std::array<double, 3>& C,
                            const std::array<double, 3>& D,
                            const std::vector<RysPrimitivePair>& bras,
                            const std::vector<RysPrimitivePair>& kets,
                            double* __restrict out) {
    constexpr int Na = (La + 1) * (La + 2) / 2, Nb = (Lb + 1) * (Lb + 2) / 2;
    constexpr int Nc = (Lc + 1) * (Lc + 2) / 2, Nd = (Ld + 1) * (Ld + 2) / 2;
    constexpr int GK = Lc + Ld + 1;
    constexpr int ISZ = (La + 1) * (Lb + 1) * GK * (Ld + 1);   // per-axis I size
    constexpr int nout = Na * Nb * Nc * Nd;
    const double AB[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
    const double CD[3] = {C[0] - D[0], C[1] - D[1], C[2] - D[2]};
    // Per-output-component offsets into the per-axis I (g_cart Cartesian order,
    // identical to the scatter's a_local/b_local/c_local/d_local enumeration).
    // Class-only data: built ONCE per instantiated class (magic static), not per
    // block -- the per-block rebuild dominated shallow-contraction d-classes
    // (dd|dd is 1 primitive quartet per block; 3*nout bounds-checked index
    // computations per call was several us of pure overhead).
    struct FusedIdxMaps {
        int ix[nout], iy[nout], iz[nout];
        FusedIdxMaps() {
            constexpr int aoff = La * (La + 1) * (La + 2) / 6;
            constexpr int boff = Lb * (Lb + 1) * (Lb + 2) / 6;
            constexpr int coff = Lc * (Lc + 1) * (Lc + 2) / 6;
            constexpr int doff = Ld * (Ld + 1) * (Ld + 2) / 6;
            int o = 0;
            for (int al = 0; al < Na; ++al) {
                const auto& at = g_cart.at(aoff + al);
                for (int bl = 0; bl < Nb; ++bl) {
                    const auto& bt = g_cart.at(boff + bl);
                    for (int cl = 0; cl < Nc; ++cl) {
                        const auto& ct = g_cart.at(coff + cl);
                        for (int dl = 0; dl < Nd; ++dl, ++o) {
                            const auto& dt = g_cart.at(doff + dl);
                            ix[o] = ((at[0] * (Lb + 1) + bt[0]) * GK + ct[0]) * (Ld + 1) + dt[0];
                            iy[o] = ((at[1] * (Lb + 1) + bt[1]) * GK + ct[1]) * (Ld + 1) + dt[1];
                            iz[o] = ((at[2] * (Lb + 1) + bt[2]) * GK + ct[2]) * (Ld + 1) + dt[2];
                        }
                    }
                }
            }
        }
    };
    static const FusedIdxMaps idx_maps;   // C++11 magic static: thread-safe once-init
    const int* __restrict ix = idx_maps.ix;
    const int* __restrict iy = idx_maps.iy;
    const int* __restrict iz = idx_maps.iz;
    for (int o = 0; o < nout; ++o) out[o] = 0.0;
    double roots[NRoots], weights[NRoots], wbuf[NRoots];
    double Gx[ISZ * NRoots], Gy[ISZ * NRoots], Gz[ISZ * NRoots];  // root-innermost g
    for (const RysPrimitivePair& bp : bras) {
        const double p = bp.s;
        const double P[3] = {bp.Wx, bp.Wy, bp.Wz};
        for (const RysPrimitivePair& kp : kets) {
            const double q = kp.s;
            const double Q[3] = {kp.Wx, kp.Wy, kp.Wz};
            const double pq = p + q;
            const double inv_pq = 1.0 / pq;
            const double PQ[3] = {P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]};
            const double rpq2 = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];
            const double Targ = p * q * inv_pq * rpq2;          // T = rho * |PQ|^2
            const double prefactor = kTwoPi52 * bp.w * kp.w / std::sqrt(pq);
            rys_roots_weights_fixed<NRoots>(Targ, roots, weights);
            const double q_over_pq = q * inv_pq, p_over_pq = p * inv_pq;
            const double inv2p = 0.5 / p, inv2q = 0.5 / q, inv2pq = 0.5 * inv_pq;
#if defined(__AVX2__) && defined(__FMA__)
            double C00xa[NRoots], D00xa[NRoots], C00ya[NRoots], D00ya[NRoots], C00za[NRoots], D00za[NRoots];
            double B00a[NRoots], B10a[NRoots], B01a[NRoots];
            for (int n = 0; n < NRoots; ++n) {
                const double t2 = roots[n];
                wbuf[n] = prefactor * weights[n];
                B00a[n] = inv2pq * t2;
                B10a[n] = inv2p * (1.0 - q_over_pq * t2);
                B01a[n] = inv2q * (1.0 - p_over_pq * t2);
                C00xa[n] = (P[0] - A[0]) - q_over_pq * PQ[0] * t2; D00xa[n] = (Q[0] - C[0]) + p_over_pq * PQ[0] * t2;
                C00ya[n] = (P[1] - A[1]) - q_over_pq * PQ[1] * t2; D00ya[n] = (Q[1] - C[1]) + p_over_pq * PQ[1] * t2;
                C00za[n] = (P[2] - A[2]) - q_over_pq * PQ[2] * t2; D00za[n] = (Q[2] - C[2]) + p_over_pq * PQ[2] * t2;
            }
            rys_build_axis_avx<La, Lb, Lc, Ld, NRoots>(C00xa, D00xa, B00a, B10a, B01a, AB[0], CD[0], Gx);
            rys_build_axis_avx<La, Lb, Lc, Ld, NRoots>(C00ya, D00ya, B00a, B10a, B01a, AB[1], CD[1], Gy);
            rys_build_axis_avx<La, Lb, Lc, Ld, NRoots>(
                C00za, D00za, B00a, B10a, B01a, AB[2], CD[2], Gz, wbuf);
#else
            for (int n = 0; n < NRoots; ++n) {
                const double t2 = roots[n];
                wbuf[n] = prefactor * weights[n];
                const double B00 = inv2pq * t2;
                const double B10 = inv2p * (1.0 - q_over_pq * t2);
                const double B01 = inv2q * (1.0 - p_over_pq * t2);
                const double C00x = (P[0] - A[0]) - q_over_pq * PQ[0] * t2;
                const double D00x = (Q[0] - C[0]) + p_over_pq * PQ[0] * t2;
                const double C00y = (P[1] - A[1]) - q_over_pq * PQ[1] * t2;
                const double D00y = (Q[1] - C[1]) + p_over_pq * PQ[1] * t2;
                const double C00z = (P[2] - A[2]) - q_over_pq * PQ[2] * t2;
                const double D00z = (Q[2] - C[2]) + p_over_pq * PQ[2] * t2;
                fused_axis_build<La, Lb, Lc, Ld>(
                    C00x, D00x, B00, B10, B01, AB[0], CD[0], Gx, n);
                fused_axis_build<La, Lb, Lc, Ld>(
                    C00y, D00y, B00, B10, B01, AB[1], CD[1], Gy, n);
                fused_axis_build<La, Lb, Lc, Ld>(
                    C00z, D00z, B00, B10, B01, AB[2], CD[2], Gz, n, wbuf[n]);
            }
#endif
            simd_gout_acc_wfold<NRoots>(Gx, Gy, Gz, ix, iy, iz, nout, out);
        }
    }
}

#if defined(__AVX2__) && defined(__FMA__)
// Four-shell counterpart of fused_axis_build.  Every YMM lane is an
// independent contracted shell pair; the Rys root is handled by the enclosing
// loop.  This is the SIMD dimension used by LibintX/Simint and, unlike the
// root-SIMD scalar-quartet fallback above, remains four lanes wide for every
// angular class.
template <int La, int Lb, int Lc, int Ld>
static inline void fused_axis_build_shell4(
    __m256d C00, __m256d D00, __m256d B00, __m256d B10, __m256d B01,
    __m256d AB, __m256d CD, __m256d seed,
    __m256d* __restrict I) noexcept {
    constexpr int LAB = La + Lb;
    constexpr int LCD = Lc + Ld;
    constexpr int GK = LCD + 1;
    __m256d g[(LAB + 1) * GK];
    __m256d Bb[(LAB + 1) * (Lb + 1) * GK];

    g[0] = seed;
    for (int i = 0; i < LAB; ++i) {
        __m256d value = _mm256_mul_pd(C00, g[i * GK]);
        if (i != 0) {
            value = _mm256_fmadd_pd(
                _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(i)), B10),
                g[(i - 1) * GK], value);
        }
        g[(i + 1) * GK] = value;
    }
    for (int kk = 0; kk < LCD; ++kk) {
        for (int i = 0; i <= LAB; ++i) {
            __m256d value = _mm256_mul_pd(D00, g[i * GK + kk]);
            if (i != 0) {
                value = _mm256_fmadd_pd(
                    _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(i)), B00),
                    g[(i - 1) * GK + kk], value);
            }
            if (kk != 0) {
                value = _mm256_fmadd_pd(
                    _mm256_mul_pd(_mm256_set1_pd(static_cast<double>(kk)), B01),
                    g[i * GK + kk - 1], value);
            }
            g[i * GK + kk + 1] = value;
        }
    }

    for (int i = 0; i <= LAB; ++i) {
        for (int kk = 0; kk <= LCD; ++kk) {
            Bb[(i * (Lb + 1)) * GK + kk] = g[i * GK + kk];
        }
    }
    for (int j = 1; j <= Lb; ++j) {
        for (int i = 0; i <= LAB - j; ++i) {
            for (int kk = 0; kk <= LCD; ++kk) {
                Bb[(i * (Lb + 1) + j) * GK + kk] =
                    _mm256_fmadd_pd(
                        AB,
                        Bb[(i * (Lb + 1) + (j - 1)) * GK + kk],
                        Bb[((i + 1) * (Lb + 1) + (j - 1)) * GK + kk]);
            }
        }
    }
    for (int i = 0; i <= La; ++i) {
        for (int j = 0; j <= Lb; ++j) {
            for (int kk = 0; kk <= LCD; ++kk) {
                I[((i * (Lb + 1) + j) * GK + kk) * (Ld + 1)] =
                    Bb[(i * (Lb + 1) + j) * GK + kk];
            }
        }
    }
    for (int l = 1; l <= Ld; ++l) {
        for (int i = 0; i <= La; ++i) {
            for (int j = 0; j <= Lb; ++j) {
                for (int kk = 0; kk <= LCD - l; ++kk) {
                    const std::size_t dst =
                        static_cast<std::size_t>(
                            ((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + l);
                    const std::size_t up =
                        static_cast<std::size_t>(
                            ((i * (Lb + 1) + j) * GK + kk + 1) * (Ld + 1) +
                            l - 1);
                    const std::size_t prev = dst - 1;
                    I[dst] = _mm256_fmadd_pd(CD, I[prev], I[up]);
                }
            }
        }
    }
}

// Radial-only counterpart used by the direct spherical projector.  It stops
// before either horizontal transfer, retaining the compact
// [a+0 | c+0] table.  The pair projectors fold both HRRs and C2S into the
// subsequent contraction, so no four-centre Cartesian tensor is formed.
template <int LAB, int LCD>
static inline void fused_axis_build_radial_shell4(
    __m256d C00, __m256d D00, __m256d B00, __m256d B10, __m256d B01,
    __m256d seed, __m256d* __restrict g) noexcept {
    constexpr int GK = LCD + 1;
    g[0] = seed;
    for (int i = 0; i < LAB; ++i) {
        __m256d value = _mm256_mul_pd(C00, g[i * GK]);
        if (i != 0) {
            value = _mm256_fmadd_pd(
                _mm256_mul_pd(
                    _mm256_set1_pd(static_cast<double>(i)), B10),
                g[(i - 1) * GK], value);
        }
        g[(i + 1) * GK] = value;
    }
    for (int k = 0; k < LCD; ++k) {
        for (int i = 0; i <= LAB; ++i) {
            __m256d value = _mm256_mul_pd(D00, g[i * GK + k]);
            if (i != 0) {
                value = _mm256_fmadd_pd(
                    _mm256_mul_pd(
                        _mm256_set1_pd(static_cast<double>(i)), B00),
                    g[(i - 1) * GK + k], value);
            }
            if (k != 0) {
                value = _mm256_fmadd_pd(
                    _mm256_mul_pd(
                        _mm256_set1_pd(static_cast<double>(k)), B01),
                    g[i * GK + k - 1], value);
            }
            g[i * GK + k + 1] = value;
        }
    }
}

struct RysGenericShell4Scratch {
    // Component-major AoSoA: four independent shell quartets are contiguous.
    std::vector<double> cart4;
    std::vector<double> radial4;
    std::vector<double> radial_lhs_projector4;
    std::vector<int> radial_lhs_active_offsets;
    std::vector<int> radial_lhs_active_columns;
    std::vector<double> spherical4;
};
thread_local RysGenericShell4Scratch g_rys_generic_shell4_scratch;

#ifndef TLNO_AVX2_COMPACT_KET_DP
// Retained with mixed-bra radial dispatch (optimization study, 2026-09-12).
// Preserve the one-axis VRR dependencies; only unused three-axis ket columns
// and their final projection intermediates are removed.
// Native W8 keeps its independently measured control for any W4 fallback.
#if defined(__AVX512F__)
#define TLNO_AVX2_COMPACT_KET_DP 0
#else
#define TLNO_AVX2_COMPACT_KET_DP 1
#endif
#endif
#ifndef TLNO_AVX2_COMPACT_KET_FS
#if defined(__AVX512F__)
#define TLNO_AVX2_COMPACT_KET_FS 0
#else
#define TLNO_AVX2_COMPACT_KET_FS 1
#endif
#endif
#if (TLNO_AVX2_COMPACT_KET_DP != 0 && TLNO_AVX2_COMPACT_KET_DP != 1) || \
    (TLNO_AVX2_COMPACT_KET_FS != 0 && TLNO_AVX2_COMPACT_KET_FS != 1)
#error "TLNO_AVX2_COMPACT_KET_DP/FS must be 0 or 1"
#endif

template <int La, int Lb, int Lc, int Ld, bool RadialKernel, class Emit>
// Generic q_floor stays inside the exact screened mask instead of changing the
// shell4 ABI or duplicating its portfolio.  ket is already traversed from high
// to low Q, so the existing break removes its dead suffix.  When ket_pos==0
// also rejects a whole bra packet, the current and every later bra packet are
// below the same floor and the complete remaining rectangle can return.  U
// never enters this exact-mask block; its pair_count loops remain verbatim.
static void accumulate_rys_generic_spherical_shell4_bin_pair(
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values,
    bool exact_shell_mask) {
    constexpr int Na = (La + 1) * (La + 2) / 2;
    constexpr int Nb = (Lb + 1) * (Lb + 2) / 2;
    constexpr int Nc = (Lc + 1) * (Lc + 2) / 2;
    constexpr int Nd = (Ld + 1) * (Ld + 2) / 2;
    constexpr int LhsCart = Na * Nb;
    constexpr int RhsCart = Nc * Nd;
    constexpr int NOut = LhsCart * RhsCart;
    constexpr int NRoots = (La + Lb + Lc + Ld) / 2 + 1;
    constexpr int LAB = La + Lb;
    constexpr int LCD = Lc + Ld;
    constexpr int GK = Lc + Ld + 1;
    constexpr int ISize = (La + 1) * (Lb + 1) * GK * (Ld + 1);
    constexpr int PairClassKey =
        10 * (La * (La + 1) / 2 + Lb) +
        (Lc * (Lc + 1) / 2 + Ld);
    constexpr int BraRadialOffset = offset_for_L(La);
    constexpr int BraRadial =
        n_cart_cum(LAB) - BraRadialOffset;
    // Class-specific experiment, independently controlled from native W8.
    // HRR produces ket degrees Lc..Lc+Ld, so lower three-axis products never
    // reach the sparse projector. Keep lower one-axis VRR dependencies and
    // the existing full-ket default: historical compact variants did not
    // establish a general speedup. Every radial map/projector access below
    // already accounts for KetRadialOffset; the Cartesian route is unchanged.
    constexpr bool CompactKet = RadialKernel &&
        ((TLNO_AVX2_COMPACT_KET_DP && (PairClassKey == 44 || PairClassKey == 54)) ||
         (TLNO_AVX2_COMPACT_KET_FS && (PairClassKey == 86 || PairClassKey == 96)));
    constexpr int KetRadialOffset = CompactKet ? offset_for_L(Lc) : 0;
    constexpr int KetRadial =
        n_cart_cum(LCD) - KetRadialOffset;
    constexpr int RadialOut = BraRadial * KetRadial;
    constexpr int RadialAxisSize = (LAB + 1) * (LCD + 1);
    // Measured one-thread portfolio: these five shallow-HRR classes retain the
    // original full-Cartesian shell4 body. Other generic s-through-f classes
    // can use the root-fused radial projector when the dispatcher finds a
    // homogeneous four-pair bra pack.
    constexpr bool UseRadialProjector =
        RadialKernel &&
        (PairClassKey != 53 &&  // dd|ds
         PairClassKey != 63 &&  // fs|ds
         PairClassKey != 64 &&  // fs|dp
         PairClassKey != 66 &&  // fs|fs
         PairClassKey != 73);   // fp|ds

    struct IndexMaps {
        int ix[UseRadialProjector ? 1 : NOut];
        int iy[UseRadialProjector ? 1 : NOut];
        int iz[UseRadialProjector ? 1 : NOut];
        int rix[UseRadialProjector ? RadialOut : 1];
        int riy[UseRadialProjector ? RadialOut : 1];
        int riz[UseRadialProjector ? RadialOut : 1];
        IndexMaps() {
            if constexpr (!UseRadialProjector) {
                constexpr int aoff = La * (La + 1) * (La + 2) / 6;
                constexpr int boff = Lb * (Lb + 1) * (Lb + 2) / 6;
                constexpr int coff = Lc * (Lc + 1) * (Lc + 2) / 6;
                constexpr int doff = Ld * (Ld + 1) * (Ld + 2) / 6;
                int out = 0;
                for (int a = 0; a < Na; ++a) {
                    const auto& av = g_cart.at(aoff + a);
                    for (int b = 0; b < Nb; ++b) {
                        const auto& bv = g_cart.at(boff + b);
                        for (int c = 0; c < Nc; ++c) {
                            const auto& cv = g_cart.at(coff + c);
                            for (int d = 0; d < Nd; ++d, ++out) {
                                const auto& dv = g_cart.at(doff + d);
                                ix[out] =
                                    ((av[0] * (Lb + 1) + bv[0]) * GK +
                                     cv[0]) *
                                        (Ld + 1) +
                                    dv[0];
                                iy[out] =
                                    ((av[1] * (Lb + 1) + bv[1]) * GK +
                                     cv[1]) *
                                        (Ld + 1) +
                                    dv[1];
                                iz[out] =
                                    ((av[2] * (Lb + 1) + bv[2]) * GK +
                                     cv[2]) *
                                        (Ld + 1) +
                                    dv[2];
                            }
                        }
                    }
                }
            } else {
                int radial_out = 0;
                for (int a = 0; a < BraRadial; ++a) {
                    const auto& av = g_cart.at(BraRadialOffset + a);
                    for (int c = 0; c < KetRadial; ++c, ++radial_out) {
                        const auto& cv = g_cart.at(KetRadialOffset + c);
                        rix[radial_out] = av[0] * GK + cv[0];
                        riy[radial_out] = av[1] * GK + cv[1];
                        riz[radial_out] = av[2] * GK + cv[2];
                    }
                }
            }
        }
    };
    static const IndexMaps maps;

    const FinalSphPairProjector& lhs_projector =
        get_final_sph_pair_projector(La, Lb);
    const FinalSphPairProjector& rhs_projector =
        get_final_sph_pair_projector(Lc, Ld);
    const int lhs_sph = lhs_projector.sph_rows;
    const int rhs_sph = rhs_projector.sph_rows;
    const auto& pair_components = low_batch_pair_components(bd);
    auto& scratch = g_rys_generic_shell4_scratch;
    if constexpr (UseRadialProjector) {
        scratch.radial4.resize(static_cast<std::size_t>(RadialOut) * 4u);
        scratch.radial_lhs_projector4.resize(
            static_cast<std::size_t>(lhs_sph) *
            static_cast<std::size_t>(BraRadial) * 4u);
        scratch.radial_lhs_active_offsets.resize(
            static_cast<std::size_t>(lhs_sph) + 1u);
        scratch.radial_lhs_active_columns.resize(
            static_cast<std::size_t>(lhs_sph) *
            static_cast<std::size_t>(BraRadial));
    } else {
        scratch.cart4.resize(static_cast<std::size_t>(NOut) * 4u);
    }
    scratch.spherical4.resize(
        static_cast<std::size_t>(lhs_sph) *
        static_cast<std::size_t>(rhs_sph) * 4u);

    const __m256d vone = _mm256_set1_pd(1.0);
    const __m256d vhalf = _mm256_set1_pd(0.5);
    const __m256d vtwo_pi52 = _mm256_set1_pd(kTwoPi52);
    const double primitive_fast_threshold_squared =
        rys_df_primitive_fast_threshold_squared();
    const bool primitive_fast = primitive_fast_threshold_squared > 0.0;

    alignas(32) double targ4[4];
    alignas(32) double roots4[ActiveRoots<Emit, NRoots> * 4];
    alignas(32) double weights4[ActiveRoots<Emit, NRoots> * 4];
    __m256d Gx[ISize], Gy[ISize], Gzw[ISize];
    __m256d RGx[ActiveRoots<Emit, NRoots>][RadialAxisSize];
    __m256d RGy[ActiveRoots<Emit, NRoots>][RadialAxisSize];
    __m256d RGzw[ActiveRoots<Emit, NRoots>][RadialAxisSize];
    __m256d mid_row[(RhsCart > KetRadial) ? RhsCart : KetRadial];

    for (std::size_t bra_pos = 0; bra_pos < bra_bin.pair_count;
         bra_pos += 4) {
        const std::size_t valid_lanes =
            std::min<std::size_t>(4, bra_bin.pair_count - bra_pos);
        std::uint32_t first_bra_group = 0;
        bool uniform_bra_projector = true;
        if constexpr (UseRadialProjector) {
            first_bra_group = bra_bin.pair_group_ids[bra_pos];
            if constexpr (Lb != 0) {
                const double first_rx = bra_bin.Rx[bra_pos];
                const double first_ry = bra_bin.Ry[bra_pos];
                const double first_rz = bra_bin.Rz[bra_pos];
                for (std::size_t lane = 1; lane < valid_lanes; ++lane) {
                    const std::size_t pos = bra_pos + lane;
                    uniform_bra_projector =
                        uniform_bra_projector &&
                        bra_bin.Rx[pos] == first_rx &&
                        bra_bin.Ry[pos] == first_ry &&
                        bra_bin.Rz[pos] == first_rz;
                }
            }
            if (!uniform_bra_projector) {
                std::fill(
                    scratch.radial_lhs_projector4.begin(),
                    scratch.radial_lhs_projector4.end(), 0.0);
                for (std::size_t lane = 0; lane < valid_lanes; ++lane) {
                    const std::uint32_t pair_group =
                        bra_bin.pair_group_ids[bra_pos + lane];
                    const auto& projector =
                        bd.rys_radial_pair_projectors[
                            static_cast<std::size_t>(pair_group)];
                    for (int row = 0; row < lhs_sph; ++row) {
                        for (const SphProjectorTerm& term :
                             projector.bra_rows[
                                 static_cast<std::size_t>(row)]) {
                            const std::size_t dst =
                                (static_cast<std::size_t>(row) *
                                     static_cast<std::size_t>(BraRadial) +
                                 static_cast<std::size_t>(term.index)) *
                                    4u +
                                lane;
                            scratch.radial_lhs_projector4[dst] =
                                term.coeff;
                        }
                    }
                }
                int active_column_count = 0;
                scratch.radial_lhs_active_offsets[0] = 0;
                for (int lhs_row = 0; lhs_row < lhs_sph; ++lhs_row) {
                    for (int a = 0; a < BraRadial; ++a) {
                        const double* const coeff =
                            scratch.radial_lhs_projector4.data() +
                            (static_cast<std::size_t>(lhs_row) *
                                 static_cast<std::size_t>(BraRadial) +
                             static_cast<std::size_t>(a)) *
                                4u;
                        if (coeff[0] != 0.0 || coeff[1] != 0.0 ||
                            coeff[2] != 0.0 || coeff[3] != 0.0) {
                            scratch.radial_lhs_active_columns[
                                static_cast<std::size_t>(
                                    active_column_count++)] = a;
                        }
                    }
                    scratch.radial_lhs_active_offsets[
                        static_cast<std::size_t>(lhs_row) + 1u] =
                        active_column_count;
                }
            }
        }
        const std::size_t ket_limit =
            same_bin ? bra_pos + valid_lanes : ket_bin.pair_count;
        for (std::size_t ket_pos = 0; ket_pos < ket_limit; ++ket_pos) {
            const unsigned logical_mask = rys_sp_shell_logical_bra4(
                bra_pos, valid_lanes, ket_pos, same_bin);
            unsigned active_mask = logical_mask;
            if (exact_shell_mask) {
                unsigned monotone_mask = 0u;
                active_mask &= rys_batch_screen_lane_mask(
                    bra_bin, bra_pos, valid_lanes,
                    ket_bin.pair_group_ids[ket_pos],
                    true, monotone_mask);
                monotone_mask &= logical_mask;
                if (active_mask == 0u) {
                    // Partial chunks are Q-descending on both sides.  Here
                    // ket is the advancing dimension; once every logical bra
                    // lane fails, every later ket in this chunk also fails.
                    if (monotone_mask == 0u) {
                        if (ket_pos == 0u) {
                            // ket[0] is the largest opposing Q.  If it cannot
                            // rescue any lane, later bra packets (also lower Q)
                            // cannot pass against any ket.  This is the generic
                            // q_floor on the second axis.  It is safe for a
                            // diagonal task too: ket_pos==0 has no triangular
                            // lanes removed from logical_mask.
                            RysSphericalScreeningDiagnostics* const diag =
                                g_rys_spherical_screening_diagnostics;
                            if (diag != nullptr) {
                                ++diag->generic_q_floor_tail_stops;
                                diag->generic_q_floor_pruned_bra_pairs +=
                                    bra_bin.pair_count - bra_pos;
                            }
                            return;
                        }
                        break;
                    }
                    continue;
                }
            }
            bool active[4];
            for (int lane = 0; lane < 4; ++lane) {
                active[lane] =
                    (active_mask &
                     (1u << static_cast<unsigned>(lane))) != 0u;
            }
            const __m256d lane_mask = _mm256_castsi256_pd(_mm256_set_epi64x(
                active[3] ? -1ll : 0ll, active[2] ? -1ll : 0ll,
                active[1] ? -1ll : 0ll, active[0] ? -1ll : 0ll));

            if constexpr (UseRadialProjector) {
                std::fill(
                    scratch.radial4.begin(), scratch.radial4.end(), 0.0);
            } else {
                std::fill(
                    scratch.cart4.begin(), scratch.cart4.end(), 0.0);
            }

            for (int ib = 0; ib < bra_bin.primitive_count; ++ib) {
                const std::size_t boff =
                    static_cast<std::size_t>(ib) * bra_bin.stride + bra_pos;
                __m256d primitive_bra_screen;
                if (primitive_fast) {
                    primitive_bra_screen = _mm256_loadu_pd(
                        bra_bin.primitive_fast_schwarz.data() + boff);
                }
                const __m256d vp = _mm256_loadu_pd(bra_bin.s.data() + boff);
                const __m256d vinvp =
                    _mm256_loadu_pd(bra_bin.invs.data() + boff);
                const __m256d vPx =
                    _mm256_loadu_pd(bra_bin.Wx.data() + boff);
                const __m256d vPy =
                    _mm256_loadu_pd(bra_bin.Wy.data() + boff);
                const __m256d vPz =
                    _mm256_loadu_pd(bra_bin.Wz.data() + boff);
                const __m256d vPAx =
                    _mm256_loadu_pd(bra_bin.Lx.data() + boff);
                const __m256d vPAy =
                    _mm256_loadu_pd(bra_bin.Ly.data() + boff);
                const __m256d vPAz =
                    _mm256_loadu_pd(bra_bin.Lz.data() + boff);
                const __m256d vbw = _mm256_and_pd(
                    _mm256_loadu_pd(bra_bin.w.data() + boff), lane_mask);
                const __m256d vABx =
                    _mm256_loadu_pd(bra_bin.Rx.data() + bra_pos);
                const __m256d vABy =
                    _mm256_loadu_pd(bra_bin.Ry.data() + bra_pos);
                const __m256d vABz =
                    _mm256_loadu_pd(bra_bin.Rz.data() + bra_pos);

                for (int ik = 0; ik < ket_bin.primitive_count; ++ik) {
                    const std::size_t koff =
                        static_cast<std::size_t>(ik) * ket_bin.stride + ket_pos;
                    if (primitive_fast &&
                        rys_df_primitive_fast_packet_dead(
                            primitive_bra_screen, ket_bin, koff,
                            active_mask,
                            primitive_fast_threshold_squared)) {
                        continue;
                    }
                    const double q = ket_bin.s[koff];
                    const __m256d vq = _mm256_set1_pd(q);
                    const __m256d vinvq =
                        _mm256_set1_pd(ket_bin.invs[koff]);
                    const __m256d vQx =
                        _mm256_set1_pd(ket_bin.Wx[koff]);
                    const __m256d vQy =
                        _mm256_set1_pd(ket_bin.Wy[koff]);
                    const __m256d vQz =
                        _mm256_set1_pd(ket_bin.Wz[koff]);
                    const __m256d vQCx =
                        _mm256_set1_pd(ket_bin.Lx[koff]);
                    const __m256d vQCy =
                        _mm256_set1_pd(ket_bin.Ly[koff]);
                    const __m256d vQCz =
                        _mm256_set1_pd(ket_bin.Lz[koff]);
                    const __m256d vCDx =
                        _mm256_set1_pd(ket_bin.Rx[ket_pos]);
                    const __m256d vCDy =
                        _mm256_set1_pd(ket_bin.Ry[ket_pos]);
                    const __m256d vCDz =
                        _mm256_set1_pd(ket_bin.Rz[ket_pos]);

                    const __m256d vpq = _mm256_add_pd(vp, vq);
                    const __m256d vinvpq = _mm256_div_pd(vone, vpq);
                    const __m256d vq_over_pq =
                        _mm256_mul_pd(vq, vinvpq);
                    const __m256d vp_over_pq =
                        _mm256_mul_pd(vp, vinvpq);
                    const __m256d vPQx = _mm256_sub_pd(vPx, vQx);
                    const __m256d vPQy = _mm256_sub_pd(vPy, vQy);
                    const __m256d vPQz = _mm256_sub_pd(vPz, vQz);
                    const __m256d vr2 = _mm256_fmadd_pd(
                        vPQx, vPQx,
                        _mm256_fmadd_pd(
                            vPQy, vPQy, _mm256_mul_pd(vPQz, vPQz)));
                    const __m256d vtarg = _mm256_mul_pd(
                        _mm256_mul_pd(vp, vq_over_pq), vr2);
                    _mm256_store_pd(targ4, vtarg);
                    rys_roots_weights_batch4_for_emit<Emit, NRoots>(
                        targ4, roots4, weights4);

                    const __m256d vpref = _mm256_mul_pd(
                        _mm256_mul_pd(
                            _mm256_mul_pd(vtwo_pi52, vbw),
                            _mm256_set1_pd(ket_bin.w[koff])),
                        _mm256_sqrt_pd(vinvpq));
                    const __m256d vinv2p =
                        _mm256_mul_pd(vhalf, vinvp);
                    const __m256d vinv2q =
                        _mm256_mul_pd(vhalf, vinvq);
                    const __m256d vinv2pq =
                        _mm256_mul_pd(vhalf, vinvpq);

                    for (int root = 0;
                         root < ActiveRoots<Emit, NRoots>; ++root) {
                        const __m256d vt2 =
                            _mm256_load_pd(roots4 + root * 4);
                        const __m256d vweight =
                            _mm256_mul_pd(
                                vpref,
                                _mm256_load_pd(weights4 + root * 4));
                        const __m256d vB00 =
                            _mm256_mul_pd(vinv2pq, vt2);
                        const __m256d vB10 = _mm256_mul_pd(
                            vinv2p,
                            _mm256_fnmadd_pd(vq_over_pq, vt2, vone));
                        const __m256d vB01 = _mm256_mul_pd(
                            vinv2q,
                            _mm256_fnmadd_pd(vp_over_pq, vt2, vone));
                        const __m256d vC00x = _mm256_fnmadd_pd(
                            _mm256_mul_pd(vq_over_pq, vPQx), vt2, vPAx);
                        const __m256d vC00y = _mm256_fnmadd_pd(
                            _mm256_mul_pd(vq_over_pq, vPQy), vt2, vPAy);
                        const __m256d vC00z = _mm256_fnmadd_pd(
                            _mm256_mul_pd(vq_over_pq, vPQz), vt2, vPAz);
                        const __m256d vD00x = _mm256_fmadd_pd(
                            _mm256_mul_pd(vp_over_pq, vPQx), vt2, vQCx);
                        const __m256d vD00y = _mm256_fmadd_pd(
                            _mm256_mul_pd(vp_over_pq, vPQy), vt2, vQCy);
                        const __m256d vD00z = _mm256_fmadd_pd(
                            _mm256_mul_pd(vp_over_pq, vPQz), vt2, vQCz);

                        if constexpr (UseRadialProjector) {
                            fused_axis_build_radial_shell4<LAB, LCD>(
                                vC00x, vD00x, vB00, vB10, vB01,
                                vone, RGx[root]);
                            fused_axis_build_radial_shell4<LAB, LCD>(
                                vC00y, vD00y, vB00, vB10, vB01,
                                vone, RGy[root]);
                            fused_axis_build_radial_shell4<LAB, LCD>(
                                vC00z, vD00z, vB00, vB10, vB01,
                                vweight, RGzw[root]);
                        } else {
                            fused_axis_build_shell4<La, Lb, Lc, Ld>(
                                vC00x, vD00x, vB00, vB10, vB01,
                                vABx, vCDx, vone, Gx);
                            fused_axis_build_shell4<La, Lb, Lc, Ld>(
                                vC00y, vD00y, vB00, vB10, vB01,
                                vABy, vCDy, vone, Gy);
                            fused_axis_build_shell4<La, Lb, Lc, Ld>(
                                vC00z, vD00z, vB00, vB10, vB01,
                                vABz, vCDz, vweight, Gzw);

                            for (int out = 0; out < NOut; ++out) {
                                double* const dst =
                                    scratch.cart4.data() +
                                    static_cast<std::size_t>(out) * 4u;
                                const __m256d xy = _mm256_mul_pd(
                                    Gx[maps.ix[out]], Gy[maps.iy[out]]);
                                const __m256d old =
                                    _mm256_loadu_pd(dst);
                                _mm256_storeu_pd(
                                    dst,
                                    _mm256_fmadd_pd(
                                        xy, Gzw[maps.iz[out]], old));
                            }
                        }
                    }
                    if constexpr (UseRadialProjector) {
                        for (int out = 0; out < RadialOut; ++out) {
                            double* const dst =
                                scratch.radial4.data() +
                                static_cast<std::size_t>(out) * 4u;
                            __m256d value = _mm256_loadu_pd(dst);
                            for (int root = 0;
                                 root < ActiveRoots<Emit, NRoots>; ++root) {
                                const __m256d xy = _mm256_mul_pd(
                                    RGx[root][maps.rix[out]],
                                    RGy[root][maps.riy[out]]);
                                value = _mm256_fmadd_pd(
                                    xy, RGzw[root][maps.riz[out]], value);
                            }
                            _mm256_storeu_pd(dst, value);
                        }
                    }
                }
            }

            if constexpr (UseRadialProjector) {
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& ket_projector =
                    bd.rys_radial_pair_projectors[
                        static_cast<std::size_t>(ket_group)];
                const auto transform_rhs = [&](int lhs_row) {
                    for (int rhs_row = 0;
                         rhs_row < rhs_sph; ++rhs_row) {
                        __m256d sum = _mm256_setzero_pd();
                        for (const SphProjectorTerm& term :
                             ket_projector.ket_rows[
                                 static_cast<std::size_t>(rhs_row)]) {
                            const int ket_index =
                                term.index - KetRadialOffset;
                            sum = _mm256_fmadd_pd(
                                _mm256_set1_pd(term.coeff),
                                mid_row[ket_index], sum);
                        }
                        double* const dst =
                            scratch.spherical4.data() +
                            (static_cast<std::size_t>(lhs_row) *
                                 static_cast<std::size_t>(rhs_sph) +
                             static_cast<std::size_t>(rhs_row)) *
                                4u;
                        _mm256_storeu_pd(dst, sum);
                    }
                };

                if (uniform_bra_projector) {
                    const auto& bra_projector =
                        bd.rys_radial_pair_projectors[
                            static_cast<std::size_t>(first_bra_group)];
                    for (int lhs_row = 0;
                         lhs_row < lhs_sph; ++lhs_row) {
                        const auto& terms =
                            bra_projector.bra_rows[
                                static_cast<std::size_t>(lhs_row)];
                        for (int j = 0; j < KetRadial; ++j) {
                            __m256d sum = _mm256_setzero_pd();
                            for (const SphProjectorTerm& term : terms) {
                                const double* const src =
                                    scratch.radial4.data() +
                                    (static_cast<std::size_t>(term.index) *
                                         static_cast<std::size_t>(KetRadial) +
                                     static_cast<std::size_t>(j)) *
                                        4u;
                                sum = _mm256_fmadd_pd(
                                    _mm256_set1_pd(term.coeff),
                                    _mm256_loadu_pd(src), sum);
                            }
                            mid_row[j] = sum;
                        }
                        transform_rhs(lhs_row);
                    }
                } else {
                    for (int lhs_row = 0;
                         lhs_row < lhs_sph; ++lhs_row) {
                        const int active_begin =
                            scratch.radial_lhs_active_offsets[
                                static_cast<std::size_t>(lhs_row)];
                        const int active_end =
                            scratch.radial_lhs_active_offsets[
                                static_cast<std::size_t>(lhs_row) + 1u];
                        for (int j = 0; j < KetRadial; ++j) {
                            __m256d sum = _mm256_setzero_pd();
                            for (int k = active_begin;
                                 k < active_end; ++k) {
                                const int a =
                                    scratch.radial_lhs_active_columns[
                                        static_cast<std::size_t>(k)];
                                const double* const coeff =
                                    scratch.radial_lhs_projector4.data() +
                                    (static_cast<std::size_t>(lhs_row) *
                                         static_cast<std::size_t>(
                                             BraRadial) +
                                     static_cast<std::size_t>(a)) *
                                        4u;
                                const double* const src =
                                    scratch.radial4.data() +
                                    (static_cast<std::size_t>(a) *
                                         static_cast<std::size_t>(
                                             KetRadial) +
                                     static_cast<std::size_t>(j)) *
                                        4u;
                                sum = _mm256_fmadd_pd(
                                    _mm256_loadu_pd(coeff),
                                    _mm256_loadu_pd(src), sum);
                            }
                            mid_row[j] = sum;
                        }
                        transform_rhs(lhs_row);
                    }
                }
            } else for (int lhs_row = 0; lhs_row < lhs_sph; ++lhs_row) {
                for (int j = 0; j < RhsCart; ++j) {
                    __m256d sum = _mm256_setzero_pd();
                    for (const SphProjectorTerm& term :
                         lhs_projector.rows[
                             static_cast<std::size_t>(lhs_row)]) {
                        const double* const src =
                            scratch.cart4.data() +
                            (static_cast<std::size_t>(term.index) *
                                 static_cast<std::size_t>(RhsCart) +
                             static_cast<std::size_t>(j)) *
                                4u;
                        sum = _mm256_fmadd_pd(
                            _mm256_set1_pd(term.coeff),
                            _mm256_loadu_pd(src), sum);
                    }
                    mid_row[j] = sum;
                }
                for (int rhs_row = 0; rhs_row < rhs_sph; ++rhs_row) {
                    __m256d sum = _mm256_setzero_pd();
                    for (const SphProjectorTerm& term :
                         rhs_projector.rows[
                             static_cast<std::size_t>(rhs_row)]) {
                        sum = _mm256_fmadd_pd(
                            _mm256_set1_pd(term.coeff),
                            mid_row[term.index], sum);
                    }
                    double* const dst =
                        scratch.spherical4.data() +
                        (static_cast<std::size_t>(lhs_row) *
                             static_cast<std::size_t>(rhs_sph) +
                         static_cast<std::size_t>(rhs_row)) *
                            4u;
                    _mm256_storeu_pd(dst, sum);
                }
            }

#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
            constexpr bool emit_values_individually = true;
#else
            constexpr bool emit_values_individually =
                !rys_emit_uses_packed_checksum_fold_v<Emit>;
#endif
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
            // Materialization has no stream accumulator: emit every canonical
            // value through the audited packed writer instead of the fold.
            if (emit_values_individually || !eri_streaming()) {
#else
            if constexpr (emit_values_individually) {
#endif
            for (int lane = 0; lane < 4; ++lane) {
                if (!active[lane]) continue;
                const std::uint32_t bra_group =
                    bra_bin.pair_group_ids[
                        bra_pos + static_cast<std::size_t>(lane)];
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& lhs_refs =
                    pair_components[static_cast<std::size_t>(bra_group)];
                const auto& rhs_refs =
                    pair_components[static_cast<std::size_t>(ket_group)];
                const bool same_pair_group = bra_group == ket_group;
                // Block emission: the J/K context is resolved once here
                // instead of once per integral (see rys_emit_block).  The
                // packed policy generates the same loop as before.
                const double* const sph4 = scratch.spherical4.data();
                const std::size_t lane_off = static_cast<std::size_t>(lane);
                const std::size_t rhs_n = static_cast<std::size_t>(rhs_sph);
                rys_emit_block<Emit>(
                    eri_values, lhs_refs, rhs_refs, same_pair_group,
                    [sph4, lane_off, rhs_n](
                        const PackedPairComponentH& l,
                        const PackedPairComponentH& r) noexcept -> double {
                        return sph4[(static_cast<std::size_t>(
                                         l.component_index) * rhs_n +
                                     static_cast<std::size_t>(
                                         r.component_index)) * 4u + lane_off];
                    });
            }
            } else {
            double checksum[4] = {0.0, 0.0, 0.0, 0.0};
            for (int lane = 0; lane < 4; ++lane) {
                if (!active[lane]) continue;
                const std::uint32_t bra_group =
                    bra_bin.pair_group_ids[
                        bra_pos + static_cast<std::size_t>(lane)];
                const std::uint32_t ket_group =
                    ket_bin.pair_group_ids[ket_pos];
                const auto& lhs_refs =
                    pair_components[static_cast<std::size_t>(bra_group)];
                const auto& rhs_refs =
                    pair_components[static_cast<std::size_t>(ket_group)];
                const bool same_pair_group = bra_group == ket_group;
                double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0, sum3 = 0.0;
                std::size_t ordinal = 0;
                for (const PackedPairComponentH& lhs_ref : lhs_refs) {
                    for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                        if (same_pair_group &&
                            rhs_ref.pair_index > lhs_ref.pair_index) {
                            continue;
                        }
                        const std::size_t component =
                            (static_cast<std::size_t>(
                                 lhs_ref.component_index) *
                                 static_cast<std::size_t>(rhs_sph) +
                             static_cast<std::size_t>(
                                 rhs_ref.component_index)) *
                                4u +
                            static_cast<std::size_t>(lane);
                        const double value =
                            scratch.spherical4[component];
                        switch (ordinal++ & 3u) {
                            case 0: sum0 += value; break;
                            case 1: sum1 += value; break;
                            case 2: sum2 += value; break;
                            default: sum3 += value; break;
                        }
                    }
                }
                checksum[lane] = (sum0 + sum1) + (sum2 + sum3);
            }
            eri_stream_accumulator().checksum +=
                (checksum[0] + checksum[1]) +
                (checksum[2] + checksum[3]);
            (void)eri_values;
            }
        }
    }
}

constexpr bool rys_generic_shell4_radial_class(
    int bra_code, int ket_code) noexcept {
    if (bra_code < 0 || bra_code > 9 ||
        ket_code < 0 || ket_code > bra_code) {
        return false;
    }
    const int pair_class_key = 10 * bra_code + ket_code;
    switch (pair_class_key) {
        case 53:  // dd|ds
        case 63:  // fs|ds
        case 66:  // fs|fs
        case 73:  // fp|ds
            return false;
        case 64:  // fs|dp
            return false;
        default:
            return true;
    }
}

inline bool rys_generic_shell4_uniform_bra_bin(
    const RysLowPairBinH& bra_bin) noexcept {
    if (bra_bin.pair_count == 0) return false;
    const std::uint32_t first_group = bra_bin.pair_group_ids.front();
    for (std::size_t pos = 1; pos < bra_bin.pair_count; ++pos) {
        if (bra_bin.pair_group_ids[pos] != first_group) return false;
    }
    return true;
}

// Retained W4 compact/radial bundle: clean nine-molecule GM 1.035740x,
// GCC13/v3, four paired blocks; the 2026-09-12 optimization study recorded
// the dispersion and a separate full-value gate. The gain is for the bundle,
// not this predicate alone. Radial kernels already have lane-dependent bra
// projectors; profitable mixed bins need no shared pair_group_id.
#ifndef TLNO_AVX2_MIXED_RADIAL_DISPATCH
#if defined(__AVX512F__)
#define TLNO_AVX2_MIXED_RADIAL_DISPATCH 0
#else
#define TLNO_AVX2_MIXED_RADIAL_DISPATCH 1
#endif
#endif
#ifndef TLNO_AVX2_MIXED_RADIAL_DPDP_MIN_PRODUCTS
#define TLNO_AVX2_MIXED_RADIAL_DPDP_MIN_PRODUCTS 2
#endif
#ifndef TLNO_AVX2_MIXED_RADIAL_DDDP_MIN_PRODUCTS
#define TLNO_AVX2_MIXED_RADIAL_DDDP_MIN_PRODUCTS 2
#endif
#ifndef TLNO_AVX2_MIXED_RADIAL_FDFS_MIN_PRODUCTS
#define TLNO_AVX2_MIXED_RADIAL_FDFS_MIN_PRODUCTS 2
#endif
#ifndef TLNO_AVX2_MIXED_RADIAL_FFFS_MIN_PRODUCTS
#define TLNO_AVX2_MIXED_RADIAL_FFFS_MIN_PRODUCTS 3
#endif
#if TLNO_AVX2_MIXED_RADIAL_DISPATCH != 0 && TLNO_AVX2_MIXED_RADIAL_DISPATCH != 1
#error "TLNO_AVX2_MIXED_RADIAL_DISPATCH must be 0 or 1"
#endif
#if TLNO_AVX2_MIXED_RADIAL_DPDP_MIN_PRODUCTS < 0 || \
    TLNO_AVX2_MIXED_RADIAL_DDDP_MIN_PRODUCTS < 0 || \
    TLNO_AVX2_MIXED_RADIAL_FDFS_MIN_PRODUCTS < 0 || \
    TLNO_AVX2_MIXED_RADIAL_FFFS_MIN_PRODUCTS < 0
#error "Mixed radial minimum products must be nonnegative; zero disables a class"
#endif

#if TLNO_AVX2_MIXED_RADIAL_DISPATCH
template <int PairClassKey>
inline bool rys_generic_shell4_mixed_radial_selected(
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool exact_shell_mask) noexcept {
    constexpr bool compact_ready =
        ((PairClassKey == 44 || PairClassKey == 54) && TLNO_AVX2_COMPACT_KET_DP) ||
        ((PairClassKey == 86 || PairClassKey == 96) && TLNO_AVX2_COMPACT_KET_FS);
    constexpr std::uint64_t minimum_products =
        PairClassKey == 44 ? TLNO_AVX2_MIXED_RADIAL_DPDP_MIN_PRODUCTS :
        PairClassKey == 54 ? TLNO_AVX2_MIXED_RADIAL_DDDP_MIN_PRODUCTS :
        PairClassKey == 86 ? TLNO_AVX2_MIXED_RADIAL_FDFS_MIN_PRODUCTS :
        PairClassKey == 96 ? TLNO_AVX2_MIXED_RADIAL_FFFS_MIN_PRODUCTS : 0;
    if constexpr (!compact_ready || minimum_products == 0) {
        return false;
    } else {
        // Widen both operands before multiplication: primitive_count is int.
        const std::uint64_t primitive_products =
            static_cast<std::uint64_t>(bra_bin.primitive_count) *
            static_cast<std::uint64_t>(ket_bin.primitive_count);
        // Fully accepted screened chunks can have exact_shell_mask=false.
        // Exclude their enclosing batch-screen context too. A density-only
        // context whose effective Schwarz is zero may still reach route U;
        // this predicate does not exclude that separate TLS context.
        return primitive_products >= minimum_products &&
               !exact_shell_mask && g_rys_batch_screen == nullptr;
    }
}
#define TLNO_RYS_GENERIC_USE_RADIAL(BC, KC)                                  \
    (rys_generic_shell4_mixed_radial_selected<10 * (BC) + (KC)>(              \
         bra_bin, ket_bin, exact_shell_mask) ||                              \
     rys_generic_shell4_uniform_bra_bin(bra_bin))
#else
// Disabled expansion is precisely the existing uniform-bin predicate.
#define TLNO_RYS_GENERIC_USE_RADIAL(BC, KC)                                  \
    rys_generic_shell4_uniform_bra_bin(bra_bin)
#endif

template <class Emit>
inline bool dispatch_rys_generic_spherical_shell4_bin_pair(
    int bra_code, int ket_code,
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    const RysLowPairBinH& ket_bin,
    bool same_bin,
    std::vector<double>& eri_values,
    bool exact_shell_mask = false) {
#define TLNO_RYS_SHELL4_CASE(BC, KC, LA, LB, LC, LD)                         \
    case 10 * (BC) + (KC): {                                                  \
        constexpr bool radial_class =                                         \
            rys_generic_shell4_radial_class((BC), (KC));                       \
        if constexpr (radial_class) {                                          \
            const bool uniform_bra_geometry =                                  \
                TLNO_RYS_GENERIC_USE_RADIAL(BC, KC);                    \
            if (uniform_bra_geometry) {                                         \
                accumulate_rys_generic_spherical_shell4_bin_pair<              \
                    LA, LB, LC, LD, true, Emit>(                                \
                    bd, bra_bin, ket_bin, same_bin, eri_values,                 \
                    exact_shell_mask);                                          \
            } else {                                                           \
                accumulate_rys_generic_spherical_shell4_bin_pair<              \
                    LA, LB, LC, LD, false, Emit>(                               \
                    bd, bra_bin, ket_bin, same_bin, eri_values,                 \
                    exact_shell_mask);                                          \
            }                                                                  \
        } else {                                                               \
            accumulate_rys_generic_spherical_shell4_bin_pair<                  \
                LA, LB, LC, LD, false, Emit>(                                  \
                bd, bra_bin, ket_bin, same_bin, eri_values,                    \
                exact_shell_mask);                                              \
        }                                                                      \
        return true;                                                           \
    }
    switch (10 * bra_code + ket_code) {
        TLNO_RYS_SHELL4_CASE(0, 0, 0, 0, 0, 0);
        TLNO_RYS_SHELL4_CASE(1, 0, 1, 0, 0, 0);
        TLNO_RYS_SHELL4_CASE(1, 1, 1, 0, 1, 0);
        TLNO_RYS_SHELL4_CASE(2, 0, 1, 1, 0, 0);
        TLNO_RYS_SHELL4_CASE(2, 1, 1, 1, 1, 0);
        TLNO_RYS_SHELL4_CASE(2, 2, 1, 1, 1, 1);
        TLNO_RYS_SHELL4_CASE(3, 0, 2, 0, 0, 0);
        TLNO_RYS_SHELL4_CASE(3, 1, 2, 0, 1, 0);
        TLNO_RYS_SHELL4_CASE(3, 2, 2, 0, 1, 1);
        TLNO_RYS_SHELL4_CASE(3, 3, 2, 0, 2, 0);
        TLNO_RYS_SHELL4_CASE(4, 0, 2, 1, 0, 0);
        TLNO_RYS_SHELL4_CASE(4, 1, 2, 1, 1, 0);
        TLNO_RYS_SHELL4_CASE(4, 2, 2, 1, 1, 1);
        TLNO_RYS_SHELL4_CASE(4, 3, 2, 1, 2, 0);
        TLNO_RYS_SHELL4_CASE(4, 4, 2, 1, 2, 1);
        TLNO_RYS_SHELL4_CASE(5, 0, 2, 2, 0, 0);
        TLNO_RYS_SHELL4_CASE(5, 1, 2, 2, 1, 0);
        TLNO_RYS_SHELL4_CASE(5, 2, 2, 2, 1, 1);
        TLNO_RYS_SHELL4_CASE(5, 3, 2, 2, 2, 0);
        TLNO_RYS_SHELL4_CASE(5, 4, 2, 2, 2, 1);
        TLNO_RYS_SHELL4_CASE(5, 5, 2, 2, 2, 2);
        TLNO_RYS_SHELL4_CASE(6, 0, 3, 0, 0, 0);
        TLNO_RYS_SHELL4_CASE(6, 1, 3, 0, 1, 0);
        TLNO_RYS_SHELL4_CASE(6, 2, 3, 0, 1, 1);
        TLNO_RYS_SHELL4_CASE(6, 3, 3, 0, 2, 0);
        TLNO_RYS_SHELL4_CASE(6, 4, 3, 0, 2, 1);
        TLNO_RYS_SHELL4_CASE(6, 5, 3, 0, 2, 2);
        TLNO_RYS_SHELL4_CASE(6, 6, 3, 0, 3, 0);
        TLNO_RYS_SHELL4_CASE(7, 0, 3, 1, 0, 0);
        TLNO_RYS_SHELL4_CASE(7, 1, 3, 1, 1, 0);
        TLNO_RYS_SHELL4_CASE(7, 2, 3, 1, 1, 1);
        TLNO_RYS_SHELL4_CASE(7, 3, 3, 1, 2, 0);
        TLNO_RYS_SHELL4_CASE(7, 4, 3, 1, 2, 1);
        TLNO_RYS_SHELL4_CASE(7, 5, 3, 1, 2, 2);
        TLNO_RYS_SHELL4_CASE(7, 6, 3, 1, 3, 0);
        TLNO_RYS_SHELL4_CASE(7, 7, 3, 1, 3, 1);
        TLNO_RYS_SHELL4_CASE(8, 0, 3, 2, 0, 0);
        TLNO_RYS_SHELL4_CASE(8, 1, 3, 2, 1, 0);
        TLNO_RYS_SHELL4_CASE(8, 2, 3, 2, 1, 1);
        TLNO_RYS_SHELL4_CASE(8, 3, 3, 2, 2, 0);
        TLNO_RYS_SHELL4_CASE(8, 4, 3, 2, 2, 1);
        TLNO_RYS_SHELL4_CASE(8, 5, 3, 2, 2, 2);
        TLNO_RYS_SHELL4_CASE(8, 6, 3, 2, 3, 0);
        TLNO_RYS_SHELL4_CASE(8, 7, 3, 2, 3, 1);
        TLNO_RYS_SHELL4_CASE(8, 8, 3, 2, 3, 2);
        TLNO_RYS_SHELL4_CASE(9, 0, 3, 3, 0, 0);
        TLNO_RYS_SHELL4_CASE(9, 1, 3, 3, 1, 0);
        TLNO_RYS_SHELL4_CASE(9, 2, 3, 3, 1, 1);
        TLNO_RYS_SHELL4_CASE(9, 3, 3, 3, 2, 0);
        TLNO_RYS_SHELL4_CASE(9, 4, 3, 3, 2, 1);
        TLNO_RYS_SHELL4_CASE(9, 5, 3, 3, 2, 2);
        TLNO_RYS_SHELL4_CASE(9, 6, 3, 3, 3, 0);
        TLNO_RYS_SHELL4_CASE(9, 7, 3, 3, 3, 1);
        TLNO_RYS_SHELL4_CASE(9, 8, 3, 3, 3, 2);
        TLNO_RYS_SHELL4_CASE(9, 9, 3, 3, 3, 3);
        default: break;
    }
#undef TLNO_RYS_SHELL4_CASE
    return false;
}
#undef TLNO_RYS_GENERIC_USE_RADIAL

inline bool rys_spherical_class_covered_by_specialized_batch(
    int bra_code, int ket_code,
    bool use_dp_matrix_shell_batch,
    bool use_dd_low_shell_batch) noexcept {
    const int key = 10 * bra_code + ket_code;
    switch (key) {
        case 0:   // ss|ss
        case 10:  // ps|ss
        case 11:  // ps|ps
        case 20:  // pp|ss
        case 21:  // pp|ps
        case 22:  // pp|pp
        case 30:  // ds|ss
        case 31:  // ds|ps
        case 32:  // ds|pp
        case 33:  // ds|ds
        case 40:  // dp|ss
        case 41:  // dp|ps
        case 60:  // fs|ss
        case 61:  // fs|ps
            return true;
        case 42:  // dp|pp
        case 43:  // dp|ds
            return use_dp_matrix_shell_batch;
        case 50:  // dd|ss
        case 51:  // dd|ps
            return use_dd_low_shell_batch;
        case 53:  // dd|ds
            return use_dd_low_shell_batch && rys_dd_ds_shell_batch_enabled();
        case 52:  // dd|pp
            // ESSENTIAL next to the knob: if the specialization runs but the
            // class is not declared covered, the generic route emits the SAME
            // quartets and all of dd|pp is counted twice.  The two conditions
            // must always move together.
            return use_dd_low_shell_batch && rys_dd_pp_shell_batch_enabled();
        default:
            return false;
    }
}

template <class Emit>
void accumulate_rys_generic_spherical_shell4_batches(
    const BasisDataHGP& bd,
    bool use_dp_matrix_shell_batch,
    bool use_dd_low_shell_batch,
    std::vector<double>& eri_values) {
    for (int bra_code = 0; bra_code < 10; ++bra_code) {
        const auto& bra_bins =
            bd.rys_all_pair_bins[static_cast<std::size_t>(bra_code)];
        if (bra_bins.empty()) continue;
        for (int ket_code = 0; ket_code <= bra_code; ++ket_code) {
            if (rys_spherical_class_covered_by_specialized_batch(
                    bra_code, ket_code,
                    use_dp_matrix_shell_batch,
                    use_dd_low_shell_batch)) {
                continue;
            }
            const auto& ket_bins =
                bd.rys_all_pair_bins[static_cast<std::size_t>(ket_code)];
            if (ket_bins.empty()) continue;

            if (bra_code != ket_code) {
                for (std::size_t bi = 1; bi < bra_bins.size(); ++bi) {
                    if (bra_bins[bi].pair_count == 0) continue;
                    for (std::size_t ki = 1; ki < ket_bins.size(); ++ki) {
                        if (ket_bins[ki].pair_count == 0) continue;
                        (void)dispatch_rys_generic_spherical_shell4_bin_pair<Emit>(
                            bra_code, ket_code, bd,
                            bra_bins[bi], ket_bins[ki], false, eri_values);
                    }
                }
                continue;
            }

            for (std::size_t bi = 1; bi < bra_bins.size(); ++bi) {
                if (bra_bins[bi].pair_count == 0) continue;
                for (std::size_t ki = 1; ki <= bi; ++ki) {
                    if (ki >= ket_bins.size() ||
                        ket_bins[ki].pair_count == 0) {
                        continue;
                    }
                    (void)dispatch_rys_generic_spherical_shell4_bin_pair<Emit>(
                        bra_code, ket_code, bd,
                        bra_bins[bi], ket_bins[ki], bi == ki, eri_values);
                }
            }
        }
    }
}
#endif

template <int La, int Lb, int Lc, int Ld>
inline void fused_rys_block_compiled_to_vector(
    const std::array<double, 3>& A,
    const std::array<double, 3>& B,
    const std::array<double, 3>& C,
    const std::array<double, 3>& D,
    const std::vector<RysPrimitivePair>& bras,
    const std::vector<RysPrimitivePair>& kets,
    std::vector<double>& out) {
    constexpr int NRoots = (La + Lb + Lc + Ld) / 2 + 1;
    constexpr int NOut =
        ((La + 1) * (La + 2) / 2) *
        ((Lb + 1) * (Lb + 2) / 2) *
        ((Lc + 1) * (Lc + 2) / 2) *
        ((Ld + 1) * (Ld + 2) / 2);
    out.resize(static_cast<std::size_t>(NOut));
    fused_rys_block<La, Lb, Lc, Ld, NRoots>(
        A, B, C, D, bras, kets, out.data());
}

// Dispatch only classes that remain after the accepted s/p and ds shellset
// kernels.  This is the AOT equivalent of the paper's angular-pattern JIT:
// one branch per class, then a fully fixed kernel with dynamic molecule data.
inline bool try_fused_rys_block_compiled_df_runtime(
    int La, int Lb, int Lc, int Ld,
    const std::array<double, 3>& A,
    const std::array<double, 3>& B,
    const std::array<double, 3>& C,
    const std::array<double, 3>& D,
    const std::vector<RysPrimitivePair>& bras,
    const std::vector<RysPrimitivePair>& kets,
    std::vector<double>& out) {
#define TLNO_RYS_D_CLASS_KEY(LA, LB, LC, LD) \
    (((((LA) * 5) + (LB)) * 5 + (LC)) * 5 + (LD))
#define TLNO_RYS_D_COMPILED_CASE(LA, LB, LC, LD)                            \
    case TLNO_RYS_D_CLASS_KEY(LA, LB, LC, LD):                              \
        fused_rys_block_compiled_to_vector<LA, LB, LC, LD>(                 \
            A, B, C, D, bras, kets, out);                                   \
        return true
    const int key = TLNO_RYS_D_CLASS_KEY(La, Lb, Lc, Ld);
    switch (key) {
        TLNO_RYS_D_COMPILED_CASE(0, 0, 2, 1);  // ss|dp
        TLNO_RYS_D_COMPILED_CASE(1, 0, 2, 1);  // ps|dp
        TLNO_RYS_D_COMPILED_CASE(2, 0, 2, 0);  // ds|ds
        TLNO_RYS_D_COMPILED_CASE(2, 0, 1, 1);  // ds|pp
        TLNO_RYS_D_COMPILED_CASE(2, 1, 1, 0);  // dp|ps
        TLNO_RYS_D_COMPILED_CASE(2, 0, 2, 1);  // ds|dp
        TLNO_RYS_D_COMPILED_CASE(0, 0, 2, 2);  // ss|dd
        TLNO_RYS_D_COMPILED_CASE(2, 1, 0, 0);  // dp|ss
        TLNO_RYS_D_COMPILED_CASE(2, 1, 1, 1);  // dp|pp
        TLNO_RYS_D_COMPILED_CASE(1, 0, 2, 2);  // ps|dd
        TLNO_RYS_D_COMPILED_CASE(2, 1, 2, 1);  // dp|dp
        TLNO_RYS_D_COMPILED_CASE(1, 1, 2, 0);  // pp|ds
        TLNO_RYS_D_COMPILED_CASE(1, 1, 2, 1);  // pp|dp
        TLNO_RYS_D_COMPILED_CASE(2, 1, 2, 0);  // dp|ds
        TLNO_RYS_D_COMPILED_CASE(2, 0, 2, 2);  // ds|dd
        TLNO_RYS_D_COMPILED_CASE(2, 2, 1, 0);  // dd|ps
        TLNO_RYS_D_COMPILED_CASE(2, 2, 1, 1);  // dd|pp
        TLNO_RYS_D_COMPILED_CASE(2, 2, 0, 0);  // dd|ss
        TLNO_RYS_D_COMPILED_CASE(2, 2, 2, 1);  // dd|dp
        TLNO_RYS_D_COMPILED_CASE(2, 1, 2, 2);  // dp|dd
        TLNO_RYS_D_COMPILED_CASE(1, 1, 2, 2);  // pp|dd
        TLNO_RYS_D_COMPILED_CASE(2, 2, 2, 0);  // dd|ds
        TLNO_RYS_D_COMPILED_CASE(2, 2, 2, 2);  // dd|dd

        // Dominant f-containing classes from the def2-TZVP profile.  Deep
        // ff combinations below the measured cutoff stay on the compact
        // runtime fallback to avoid unnecessary instruction-cache growth.
        TLNO_RYS_D_COMPILED_CASE(1, 0, 3, 0);  // ps|fs
        TLNO_RYS_D_COMPILED_CASE(1, 0, 3, 1);  // ps|fp
        TLNO_RYS_D_COMPILED_CASE(3, 0, 1, 0);  // fs|ps
        TLNO_RYS_D_COMPILED_CASE(0, 0, 3, 0);  // ss|fs
        TLNO_RYS_D_COMPILED_CASE(1, 0, 3, 2);  // ps|fd
        TLNO_RYS_D_COMPILED_CASE(3, 1, 1, 0);  // fp|ps
        TLNO_RYS_D_COMPILED_CASE(0, 0, 3, 1);  // ss|fp
        TLNO_RYS_D_COMPILED_CASE(3, 1, 2, 1);  // fp|dp
        TLNO_RYS_D_COMPILED_CASE(3, 0, 2, 1);  // fs|dp
        TLNO_RYS_D_COMPILED_CASE(1, 1, 3, 1);  // pp|fp
        TLNO_RYS_D_COMPILED_CASE(3, 0, 0, 0);  // fs|ss
        TLNO_RYS_D_COMPILED_CASE(3, 1, 1, 1);  // fp|pp
        TLNO_RYS_D_COMPILED_CASE(2, 0, 3, 1);  // ds|fp
        TLNO_RYS_D_COMPILED_CASE(3, 1, 3, 1);  // fp|fp
        TLNO_RYS_D_COMPILED_CASE(2, 1, 3, 1);  // dp|fp
        TLNO_RYS_D_COMPILED_CASE(3, 0, 1, 1);  // fs|pp
        TLNO_RYS_D_COMPILED_CASE(3, 0, 2, 0);  // fs|ds
        TLNO_RYS_D_COMPILED_CASE(1, 1, 3, 0);  // pp|fs
        TLNO_RYS_D_COMPILED_CASE(3, 1, 2, 0);  // fp|ds
        TLNO_RYS_D_COMPILED_CASE(3, 0, 3, 1);  // fs|fp
        TLNO_RYS_D_COMPILED_CASE(3, 1, 0, 0);  // fp|ss
        TLNO_RYS_D_COMPILED_CASE(0, 0, 3, 2);  // ss|fd
        TLNO_RYS_D_COMPILED_CASE(2, 1, 3, 0);  // dp|fs
        TLNO_RYS_D_COMPILED_CASE(1, 1, 3, 2);  // pp|fd
        TLNO_RYS_D_COMPILED_CASE(2, 0, 3, 2);  // ds|fd
        TLNO_RYS_D_COMPILED_CASE(1, 0, 3, 3);  // ps|ff
        TLNO_RYS_D_COMPILED_CASE(3, 0, 3, 2);  // fs|fd
        TLNO_RYS_D_COMPILED_CASE(3, 1, 3, 2);  // fp|fd
        TLNO_RYS_D_COMPILED_CASE(3, 0, 3, 0);  // fs|fs

        // Reversed orientations that remain material under canonical pair
        // ordering on larger molecules.
        TLNO_RYS_D_COMPILED_CASE(3, 2, 1, 0);  // fd|ps
        TLNO_RYS_D_COMPILED_CASE(3, 2, 0, 0);  // fd|ss
        TLNO_RYS_D_COMPILED_CASE(3, 2, 1, 1);  // fd|pp
        TLNO_RYS_D_COMPILED_CASE(3, 2, 2, 0);  // fd|ds
        TLNO_RYS_D_COMPILED_CASE(3, 2, 3, 0);  // fd|fs
        TLNO_RYS_D_COMPILED_CASE(3, 2, 3, 1);  // fd|fp
        TLNO_RYS_D_COMPILED_CASE(3, 3, 1, 0);  // ff|ps

        // Complete the L<=3 angular-class set.  This block is deliberately
        // kept separate so its aggregate code-size/performance tradeoff can
        // be benchmarked and reverted independently.
        TLNO_RYS_D_COMPILED_CASE(3, 0, 2, 2);  // fs|dd
        TLNO_RYS_D_COMPILED_CASE(3, 0, 3, 3);  // fs|ff
        TLNO_RYS_D_COMPILED_CASE(3, 1, 2, 2);  // fp|dd
        TLNO_RYS_D_COMPILED_CASE(3, 1, 3, 0);  // fp|fs
        TLNO_RYS_D_COMPILED_CASE(3, 1, 3, 3);  // fp|ff
        TLNO_RYS_D_COMPILED_CASE(3, 2, 2, 1);  // fd|dp
        TLNO_RYS_D_COMPILED_CASE(3, 2, 2, 2);  // fd|dd
        TLNO_RYS_D_COMPILED_CASE(3, 2, 3, 2);  // fd|fd
        TLNO_RYS_D_COMPILED_CASE(3, 2, 3, 3);  // fd|ff
        TLNO_RYS_D_COMPILED_CASE(3, 3, 0, 0);  // ff|ss
        TLNO_RYS_D_COMPILED_CASE(3, 3, 1, 1);  // ff|pp
        TLNO_RYS_D_COMPILED_CASE(3, 3, 2, 0);  // ff|ds
        TLNO_RYS_D_COMPILED_CASE(3, 3, 2, 1);  // ff|dp
        TLNO_RYS_D_COMPILED_CASE(3, 3, 2, 2);  // ff|dd
        TLNO_RYS_D_COMPILED_CASE(3, 3, 3, 0);  // ff|fs
        TLNO_RYS_D_COMPILED_CASE(3, 3, 3, 1);  // ff|fp
        TLNO_RYS_D_COMPILED_CASE(3, 3, 3, 2);  // ff|fd
        TLNO_RYS_D_COMPILED_CASE(3, 3, 3, 3);  // ff|ff
        TLNO_RYS_D_COMPILED_CASE(0, 0, 3, 3);  // ss|ff
        TLNO_RYS_D_COMPILED_CASE(1, 1, 3, 3);  // pp|ff
        TLNO_RYS_D_COMPILED_CASE(2, 0, 3, 0);  // ds|fs
        TLNO_RYS_D_COMPILED_CASE(2, 0, 3, 3);  // ds|ff
        TLNO_RYS_D_COMPILED_CASE(2, 1, 3, 2);  // dp|fd
        TLNO_RYS_D_COMPILED_CASE(2, 1, 3, 3);  // dp|ff
        TLNO_RYS_D_COMPILED_CASE(2, 2, 3, 0);  // dd|fs
        TLNO_RYS_D_COMPILED_CASE(2, 2, 3, 1);  // dd|fp
        TLNO_RYS_D_COMPILED_CASE(2, 2, 3, 2);  // dd|fd
        TLNO_RYS_D_COMPILED_CASE(2, 2, 3, 3);  // dd|ff
        default: break;
    }
#undef TLNO_RYS_D_COMPILED_CASE
#undef TLNO_RYS_D_CLASS_KEY
    return false;
}

// Runtime (non-templated) twin of fused_axis_build/fused_rys_block, used by the
// HIGH-L fallback (ff|ff and any class whose smaller pair-member L>2, which the
// templated right01 dispatch cannot reach and which would explode into hundreds
// of template instantiations).  Same per-axis HRR-folded math, but runtime dims
// + thread_local scratch.  g sized (LAB+1)*GK, Bb sized (LAB+1)*(Lb+1)*GK.
static inline void fused_axis_build_runtime(int La, int Lb, int Lc, int Ld,
        double C00, double D00, double B00, double B10, double B01,
        double AB, double CD, double* __restrict I, int r, int NR,
        double* __restrict g, double* __restrict Bb) {
    const int LAB = La + Lb, LCD = Lc + Ld, GK = LCD + 1;
    g[0] = 1.0;
    for (int i = 0; i < LAB; ++i)                               // raise bra i (0..LAB)
        g[(i + 1) * GK] = C00 * g[i * GK] + (i ? i * B10 * g[(i - 1) * GK] : 0.0);
    for (int kk = 0; kk < LCD; ++kk)                           // raise ket k (0..LCD)
        for (int i = 0; i <= LAB; ++i) {
            double v = D00 * g[i * GK + kk];
            if (i)  v += i  * B00 * g[(i - 1) * GK + kk];
            if (kk) v += kk * B01 * g[i * GK + kk - 1];
            g[i * GK + kk + 1] = v;
        }
    for (int i = 0; i <= LAB; ++i)                              // bra-HRR j=0 plane = VRR
        for (int kk = 0; kk <= LCD; ++kk)
            Bb[(i * (Lb + 1)) * GK + kk] = g[i * GK + kk];
    for (int j = 1; j <= Lb; ++j)                              // bra-HRR raise j (i shrinks)
        for (int i = 0; i <= LAB - j; ++i)
            for (int kk = 0; kk <= LCD; ++kk)
                Bb[(i * (Lb + 1) + j) * GK + kk] =
                    Bb[((i + 1) * (Lb + 1) + (j - 1)) * GK + kk] +
                    AB * Bb[(i * (Lb + 1) + (j - 1)) * GK + kk];
    for (int i = 0; i <= La; ++i)                              // ket-HRR l=0 plane = Bb
        for (int j = 0; j <= Lb; ++j)
            for (int kk = 0; kk <= LCD; ++kk)
                I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1)) * NR + r] =
                    Bb[(i * (Lb + 1) + j) * GK + kk];
    for (int l = 1; l <= Ld; ++l)                             // ket-HRR raise l (k shrinks)
        for (int i = 0; i <= La; ++i)
            for (int j = 0; j <= Lb; ++j)
                for (int kk = 0; kk <= LCD - l; ++kk)
                    I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + l) * NR + r] =
                        I[(((i * (Lb + 1) + j) * GK + (kk + 1)) * (Ld + 1) + (l - 1)) * NR + r] +
                        CD * I[(((i * (Lb + 1) + j) * GK + kk) * (Ld + 1) + (l - 1)) * NR + r];
}

// Per-class (La,Lb,Lc,Ld) Cartesian->per-axis-I offset maps.  These depend only
// on the class, not the quartet, so cache them thread-locally and build each
// class once (the maps are large for ff|ff: nout=10000, so rebuilding them per
// quartet was a real cost in the runtime fallback).
inline void get_fused_offset_maps(int La, int Lb, int Lc, int Ld,
                                  const int * & ix, const int * & iy, const int * & iz) {
    struct MapEntry { std::vector<int> ix, iy, iz; bool built = false; };
    thread_local std::vector<MapEntry> cache(625);            // 5^4, L<=4 each
    const int key = ((La * 5 + Lb) * 5 + Lc) * 5 + Ld;
    MapEntry& e = cache[static_cast<std::size_t>(key)];
    if (!e.built) {
        const int Na = n_cart(La), Nb = n_cart(Lb), Nc = n_cart(Lc), Nd = n_cart(Ld);
        const int GK = Lc + Ld + 1, nout = Na * Nb * Nc * Nd;
        e.ix.resize(nout); e.iy.resize(nout); e.iz.resize(nout);
        const int aoff = offset_for_L(La), boff = offset_for_L(Lb);
        const int coff = offset_for_L(Lc), doff = offset_for_L(Ld);
        int o = 0;
        for (int al = 0; al < Na; ++al) { const auto& at = g_cart.at(aoff + al);
          for (int bl = 0; bl < Nb; ++bl) { const auto& bt = g_cart.at(boff + bl);
            for (int cl = 0; cl < Nc; ++cl) { const auto& ct = g_cart.at(coff + cl);
              for (int dl = 0; dl < Nd; ++dl, ++o) { const auto& dt = g_cart.at(doff + dl);
                e.ix[o] = ((at[0] * (Lb + 1) + bt[0]) * GK + ct[0]) * (Ld + 1) + dt[0];
                e.iy[o] = ((at[1] * (Lb + 1) + bt[1]) * GK + ct[1]) * (Ld + 1) + dt[1];
                e.iz[o] = ((at[2] * (Lb + 1) + bt[2]) * GK + ct[2]) * (Ld + 1) + dt[2];
              } } } }
        e.built = true;
    }
    ix = e.ix.data(); iy = e.iy.data(); iz = e.iz.data();
}

// Writes out[la*(Nb*Nc*Nd) + lb*(Nc*Nd) + lc*Nd + ld]  (== block_radial layout).
static void fused_rys_block_runtime(int La, int Lb, int Lc, int Ld, int NRoots,
        const std::array<double, 3>& A, const std::array<double, 3>& B,
        const std::array<double, 3>& C, const std::array<double, 3>& D,
        const std::vector<RysPrimitivePair>& bras,
        const std::vector<RysPrimitivePair>& kets,
        std::vector<double>& out) {
    const int Na = n_cart(La), Nb = n_cart(Lb), Nc = n_cart(Lc), Nd = n_cart(Ld);
    const int LAB = La + Lb, LCD = Lc + Ld, GK = LCD + 1;
    const int ISZ = (La + 1) * (Lb + 1) * GK * (Ld + 1);
    const int nout = Na * Nb * Nc * Nd;
    const double AB[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
    const double CD[3] = {C[0] - D[0], C[1] - D[1], C[2] - D[2]};
    thread_local std::vector<double> sg, sBb, sroots, sweights, sGx, sGy, sGz, swbuf;
    sg.resize((LAB + 1) * GK); sBb.resize((LAB + 1) * (Lb + 1) * GK);
    sroots.resize(NRoots); sweights.resize(NRoots);
    sGx.resize(static_cast<std::size_t>(ISZ) * NRoots);       // root-innermost G
    sGy.resize(static_cast<std::size_t>(ISZ) * NRoots);
    sGz.resize(static_cast<std::size_t>(ISZ) * NRoots);
    swbuf.resize(NRoots);
    const int * ix, *iy, *iz;
    get_fused_offset_maps(La, Lb, Lc, Ld, ix, iy, iz);        // cached per class
    out.assign(static_cast<std::size_t>(nout), 0.0);
    double* gg = sg.data(); double* Bb = sBb.data();
    double* roots = sroots.data(); double* weights = sweights.data();
    double* Gx = sGx.data(); double* Gy = sGy.data(); double* Gz = sGz.data();
    double* wbuf = swbuf.data();
    for (const RysPrimitivePair& bp : bras) {
        const double p = bp.s;
        const double P[3] = {bp.Wx, bp.Wy, bp.Wz};
        for (const RysPrimitivePair& kp : kets) {
            const double q = kp.s;
            const double Q[3] = {kp.Wx, kp.Wy, kp.Wz};
            const double pq = p + q, inv_pq = 1.0 / pq;
            const double PQ[3] = {P[0] - Q[0], P[1] - Q[1], P[2] - Q[2]};
            const double rpq2 = PQ[0] * PQ[0] + PQ[1] * PQ[1] + PQ[2] * PQ[2];
            const double Targ = p * q * inv_pq * rpq2;
            const double prefactor = kTwoPi52 * bp.w * kp.w / std::sqrt(pq);
            rys_roots_weights(NRoots, Targ, roots, weights);
            const double q_over_pq = q * inv_pq, p_over_pq = p * inv_pq;
            const double inv2p = 0.5 / p, inv2q = 0.5 / q, inv2pq = 0.5 * inv_pq;
            for (int n = 0; n < NRoots; ++n) {
                const double t2 = roots[n];
                wbuf[n] = prefactor * weights[n];
                const double B00 = inv2pq * t2;
                const double B10 = inv2p * (1.0 - q_over_pq * t2);
                const double B01 = inv2q * (1.0 - p_over_pq * t2);
                const double C00x = (P[0] - A[0]) - q_over_pq * PQ[0] * t2;
                const double D00x = (Q[0] - C[0]) + p_over_pq * PQ[0] * t2;
                const double C00y = (P[1] - A[1]) - q_over_pq * PQ[1] * t2;
                const double D00y = (Q[1] - C[1]) + p_over_pq * PQ[1] * t2;
                const double C00z = (P[2] - A[2]) - q_over_pq * PQ[2] * t2;
                const double D00z = (Q[2] - C[2]) + p_over_pq * PQ[2] * t2;
                fused_axis_build_runtime(
                    La, Lb, Lc, Ld, C00x, D00x, B00, B10, B01,
                    AB[0], CD[0], Gx, n, NRoots, gg, Bb);
                fused_axis_build_runtime(
                    La, Lb, Lc, Ld, C00y, D00y, B00, B10, B01,
                    AB[1], CD[1], Gy, n, NRoots, gg, Bb);
                fused_axis_build_runtime(
                    La, Lb, Lc, Ld, C00z, D00z, B00, B10, B01,
                    AB[2], CD[2], Gz, n, NRoots, gg, Bb);
            }
            simd_gout_acc_rt(Gx, Gy, Gz, wbuf, NRoots, ix, iy, iz, nout, out.data());
        }
    }
}

// Lean scatter for the FUSED path: the kernel already produced the dense
// (ab|cd) block, so the emit loop only reads blk[]. The generic shared loop
// below re-loads ref structs and does div/mod per element and (in stream mode)
// serializes on the global checksum accumulator; measured at ~650 cyc/block
// average (~16% of whole kernel, ~4us of a dd|dd block). Here: block-constant
// ref data is precomputed once into small SoA arrays, the schwarz/stream
// branches are hoisted out of the loop, and the stream reduction uses 4
// independent lanes (checksum order changes -> last-ulp drift, same standard
// as WFOLD/static-maps).
template <int Na, int Nb, int Nc, int Nd, class Emit>
static inline void scatter_fused_block_lean(
    const double* __restrict blk,
    const std::vector<PackedPairComponentH>& lhs_refs,
    const std::vector<PackedPairComponentH>& rhs_refs,
    bool same_pair_group,
    const std::vector<double>* pair_schwarz_bounds,
    double schwarz_threshold,
    std::vector<double>& eri_values) {
    constexpr int NR = Nc * Nd;
    const int nl = static_cast<int>(lhs_refs.size());
    const int nr = static_cast<int>(rhs_refs.size());
    // NOTE: shell GROUPS can hold several same-L shells (def2tzvp C has 2 d
    // shells), so refs can exceed Na*Nb -- sized from refs, thread_local so the
    // allocation amortizes to zero across blocks.
    thread_local std::vector<std::uint32_t> lpid_v, rpid_v;
    thread_local std::vector<int> lrow_v, rcol_v;
    thread_local std::vector<double> lnorm_v, rnorm_v;
    lpid_v.resize(static_cast<std::size_t>(nl)); lrow_v.resize(static_cast<std::size_t>(nl));
    lnorm_v.resize(static_cast<std::size_t>(nl));
    rpid_v.resize(static_cast<std::size_t>(nr)); rcol_v.resize(static_cast<std::size_t>(nr));
    rnorm_v.resize(static_cast<std::size_t>(nr));
    std::uint32_t* __restrict lpid = lpid_v.data(); std::uint32_t* __restrict rpid = rpid_v.data();
    int* __restrict lrow = lrow_v.data(); int* __restrict rcol = rcol_v.data();
    double* __restrict lnorm = lnorm_v.data(); double* __restrict rnorm = rnorm_v.data();
    for (int i = 0; i < nl; ++i) {
        lpid[i] = lhs_refs[i].pair_index;
        lrow[i] = static_cast<int>(lhs_refs[i].component_index) * NR;
        lnorm[i] = lhs_refs[i].norm;
    }
    for (int j = 0; j < nr; ++j) {
        rpid[j] = rhs_refs[j].pair_index;
        rcol[j] = static_cast<int>(rhs_refs[j].component_index);
        rnorm[j] = rhs_refs[j].norm;
    }
    const bool do_schwarz = (pair_schwarz_bounds != nullptr) && (schwarz_threshold > 0.0);
    const bool stream = eri_streaming();
#ifndef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    double acc0 = 0.0, acc1 = 0.0, acc2 = 0.0, acc3 = 0.0;
#endif
    for (int i = 0; i < nl; ++i) {
        const std::uint32_t ab = lpid[i];
        const double ln = lnorm[i];
        const double* __restrict row = blk + lrow[i];
        const std::size_t abbase = static_cast<std::size_t>(ab) * (ab + 1) / 2;
        const double lb = do_schwarz
            ? (*pair_schwarz_bounds)[static_cast<std::size_t>(ab)] : 0.0;
        for (int j = 0; j < nr; ++j) {
            const std::uint32_t cd = rpid[j];
            if (same_pair_group && cd > ab) continue;
            if (do_schwarz &&
                lb * (*pair_schwarz_bounds)[static_cast<std::size_t>(cd)] < schwarz_threshold)
                continue;
            const double val = row[rcol[j]] * ln * rnorm[j];
            if constexpr (rys_emit_uses_packed_checksum_fold_v<Emit>) {
                const std::size_t idx = (ab >= cd)
                    ? abbase + cd
                    : static_cast<std::size_t>(cd) * (cd + 1) / 2 + ab;
                if (stream) {
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
                    eri_emit(eri_values, idx, val);
#else
                    switch (j & 3) {       // 4 lanes: break the serial += chain
                        case 0: acc0 += val; break;
                        case 1: acc1 += val; break;
                        case 2: acc2 += val; break;
                        default: acc3 += val; break;
                    }
#endif
                } else {
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
                    if (g_eri_materialized_spherical_audit != nullptr) {
                        eri_emit(eri_values, idx, val);
                    } else {
                        eri_values[idx] = val;
                    }
#else
                    eri_values[idx] = val;
#endif
                }
            } else {
                rys_emit_pairpair<Emit>(eri_values, ab, cd, val);
            }
        }
    }
#ifndef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    if constexpr (rys_emit_uses_packed_checksum_fold_v<Emit>) {
        if (stream) {
            eri_stream_accumulator().checksum +=
                (acc0 + acc1) + (acc2 + acc3);
        }
    }
#endif
}

// Block HRR for the right01 (non-fused) s/p path.  The per-element scatter
// expands each output as binom(b)*binom(d) T_radial terms with a SERIAL
// raw += chain: at pp|pp that is 81 elements x 4 dependent FMAs = the whole
// per-block fixed cost (measured 2854 cyc/blk scatter vs 4952 kernel on
// benzene/def2svp -- the overhead that loses every shallow-contraction basis).
// Factorized two-stage HRR instead: ket stage K[row][cd] = T[row][c+1d] +
// CD_d*T[row][c] (all rows), then bra stage out[ab][cd] = K[a+1b][cd] +
// AB_b*K[a][cd] over the contiguous cd dim -- 171 independent FMAs for pp|pp.
// The dense block then goes through scatter_fused_block_lean like the fused
// path.  Index tables come from the same g_cart/cart_flat_idx primitives as
// eval_rys_lowd_hrr_raw, built once per class (magic static).
template <int La, int Lb, int Lc, int Ld>
struct Right01HrrIdx {
    static constexpr int Na = (La + 1) * (La + 2) / 2;
    static constexpr int Nb = (Lb + 1) * (Lb + 2) / 2;
    static constexpr int Nc = (Lc + 1) * (Lc + 2) / 2;
    static constexpr int Nd = (Ld + 1) * (Ld + 2) / 2;
    static constexpr int AOffset = (La == 0) ? 0 : La * (La + 1) * (La + 2) / 6;
    static constexpr int COffset = (Lc == 0) ? 0 : Lc * (Lc + 1) * (Lc + 2) / 6;
    int a_up[Na * Nb];   // slice row of (a+1_b)          (Lb==1)
    int a_lo[Na * Nb];   // slice row of a
    int b_ax[Na * Nb];   // axis of b                     (Lb==1)
    int c_up[Nc * Nd];   // full cart column of (c+1_d)   (Ld==1)
    int c_lo[Nc * Nd];   // full cart column of c
    int d_ax[Nc * Nd];   // axis of d                     (Ld==1)
    Right01HrrIdx() {
        for (int a = 0; a < Na; ++a)
            for (int b = 0; b < Nb; ++b) {
                const int e = a * Nb + b;
                const auto& at = g_cart.at(AOffset + a);
                a_lo[e] = a;
                if constexpr (Lb == 1) {
                    const auto& bt = g_cart.at(offset_for_L(Lb) + b);
                    const int ax = bt[0] ? 0 : (bt[1] ? 1 : 2);
                    b_ax[e] = ax;
                    a_up[e] = cart_flat_idx(at[0] + bt[0], at[1] + bt[1],
                                            at[2] + bt[2]) - AOffset;
                } else {
                    b_ax[e] = 0; a_up[e] = a;
                }
            }
        for (int c = 0; c < Nc; ++c)
            for (int d = 0; d < Nd; ++d) {
                const int q = c * Nd + d;
                const auto& ct = g_cart.at(COffset + c);
                c_lo[q] = COffset + c;
                if constexpr (Ld == 1) {
                    const auto& dt = g_cart.at(offset_for_L(Ld) + d);
                    const int ax = dt[0] ? 0 : (dt[1] ? 1 : 2);
                    d_ax[q] = ax;
                    c_up[q] = cart_flat_idx(ct[0] + dt[0], ct[1] + dt[1],
                                            ct[2] + dt[2]);
                } else {
                    d_ax[q] = 0; c_up[q] = c_lo[q];
                }
            }
    }
};

template <int La, int Lb, int Lc, int Ld, int NaSlice, int NcFull>
static inline void hrr_right01_block(const double* __restrict T_radial,
                                     const double* __restrict AB,
                                     const double* __restrict CD,
                                     double* __restrict out) {
    static_assert(Lb <= 1 && Ld <= 1, "block HRR is for p-right classes");
    using Idx = Right01HrrIdx<La, Lb, Lc, Ld>;
    constexpr int Na = Idx::Na, Nb = Idx::Nb, Nc = Idx::Nc, Nd = Idx::Nd;
    constexpr int NKet = Nc * Nd;
    static const Idx M;
    double K[NaSlice * NKet];
    if constexpr (Ld == 1) {
        for (int r = 0; r < NaSlice; ++r) {
            const double* __restrict t = T_radial + r * NcFull;
            double* __restrict k = K + r * NKet;
            for (int q = 0; q < NKet; ++q)
                k[q] = t[M.c_up[q]] + CD[M.d_ax[q]] * t[M.c_lo[q]];
        }
    } else {
        for (int r = 0; r < NaSlice; ++r) {
            const double* __restrict t = T_radial + r * NcFull;
            double* __restrict k = K + r * NKet;
            for (int q = 0; q < NKet; ++q) k[q] = t[M.c_lo[q]];
        }
    }
    if constexpr (Lb == 1) {
        for (int e = 0; e < Na * Nb; ++e) {
            const double abf = AB[M.b_ax[e]];
            const double* __restrict up = K + M.a_up[e] * NKet;
            const double* __restrict lo = K + M.a_lo[e] * NKet;
            double* __restrict o = out + e * NKet;
            for (int q = 0; q < NKet; ++q) o[q] = up[q] + abf * lo[q];
        }
    } else {
        for (int e = 0; e < Na; ++e) {
            const double* __restrict k = K + e * NKet;
            double* __restrict o = out + e * NKet;
            for (int q = 0; q < NKet; ++q) o[q] = k[q];
        }
    }
}

template <int La, int Lb, int Lc, int Ld, class Emit>
void scatter_pairpair_rys_right0123_direct(
    const ContractedShellGroupH& gA,
    const ContractedShellGroupH& gB,
    const ContractedShellGroupH& gC,
    const ContractedShellGroupH& gD,
    const std::vector<RysPrimitivePair>& bras,
    const std::vector<RysPrimitivePair>& kets,
    const RysPrimitivePairSlab& bra_slab,
    const RysPrimitivePairSlab& ket_slab,
    const std::vector<PackedPairComponentH>& lhs_refs,
    const std::vector<PackedPairComponentH>& rhs_refs,
    bool same_pair_group,
    const std::vector<double>* pair_schwarz_bounds,
    double schwarz_threshold,
    std::vector<double>& eri_values) {
    static_assert(La >= 0 && La <= kMaxLSingle, "La out of range");
    static_assert(Lb >= 0 && Lb <= 3, "Lb out of direct range");
    static_assert(Lc >= 0 && Lc <= kMaxLSingle, "Lc out of range");
    static_assert(Ld >= 0 && Ld <= 3, "Ld out of direct range");
    using GeneratedMap = RysRight01GeneratedMap<La, Lb, Lc, Ld>;
    constexpr int LaSum = La + Lb;
    constexpr int LcSum = Lc + Ld;
    constexpr int Ltot = LaSum + LcSum;
    constexpr int NRoots = Ltot / 2 + 1;
    constexpr int Na = (La + 1) * (La + 2) / 2;
    constexpr int Nb = (Lb + 1) * (Lb + 2) / 2;
    constexpr int Nc = (Lc + 1) * (Lc + 2) / 2;
    constexpr int Nd = (Ld + 1) * (Ld + 2) / 2;
    constexpr int AOffset = (La == 0) ? 0 : La * (La + 1) * (La + 2) / 6;
    constexpr int NaFull = (LaSum + 1) * (LaSum + 2) * (LaSum + 3) / 6;
    constexpr int NcFull = (LcSum + 1) * (LcSum + 2) * (LcSum + 3) / 6;
    constexpr int NaSlice = NaFull - AOffset;
    constexpr int GStride = LcSum + 1;
    constexpr int GSize = (LaSum + 1) * (LcSum + 1);

    // The promoted fused kernel handles every class containing d or higher.
    // Return before allocating the radial/map scratch used only by s/p.
    if constexpr (La >= 2 || Lb >= 2 || Lc >= 2 || Ld >= 2) {
        double fused_out[Na * Nb * Nc * Nd];
        fused_rys_block<La, Lb, Lc, Ld, NRoots>(gA.center, gB.center, gC.center,
                                                gD.center, bras, kets, fused_out);
        scatter_fused_block_lean<Na, Nb, Nc, Nd, Emit>(
            fused_out, lhs_refs, rhs_refs, same_pair_group,
            pair_schwarz_bounds, schwarz_threshold, eri_values);
        return;
    }
    double T_radial[NaSlice * NcFull]{};
    int axs[NaSlice], ays[NaSlice], azs[NaSlice];
    int cxs[NcFull], cys[NcFull], czs[NcFull];
    for (int al = 0; al < NaSlice; ++al) {
        const auto& at = g_cart.at(AOffset + al);
        axs[al] = at[0]; ays[al] = at[1]; azs[al] = at[2];
    }
    for (int cf = 0; cf < NcFull; ++cf) {
        const auto& ct = g_cart.at(cf);
        cxs[cf] = ct[0]; cys[cf] = ct[1]; czs[cf] = ct[2];
    }

    const auto& A = gA.center;
    const auto& B = gB.center;
    const auto& C = gC.center;
    const auto& D = gD.center;
    const double AB[3] = {A[0] - B[0], A[1] - B[1], A[2] - B[2]};
    const double CD[3] = {C[0] - D[0], C[1] - D[1], C[2] - D[2]};
    double AB_monomials[64]{};
    double CD_monomials[64]{};
    if constexpr (GeneratedMap::available) {
        rys_generated_fill_sp_monomials(AB, AB_monomials);
        rys_generated_fill_sp_monomials(CD, CD_monomials);
    }

    constexpr bool lowd_hrr_supported = (Lb <= 2 && Ld <= 2);
    if constexpr (GeneratedMap::available || lowd_hrr_supported) {
        if constexpr (Ltot <= 2) {
            // Low-total-L classes use closed-form register-resident kernels
            // instead of materializing the general G table.
#if defined(__AVX2__) && defined(__FMA__)
            constexpr bool lean_available = (Ltot <= 1);
#else
            constexpr bool lean_available = false;
#endif
            if constexpr (lean_available) {
#if defined(__AVX2__) && defined(__FMA__)
                accumulate_rys_quartets_lean_fixed <
                    LaSum, LcSum, NRoots, NaSlice, NcFull, Emit > (
                    bra_slab, ket_slab, A, C, axs, ays, azs,
                    cxs, cys, czs, T_radial);
#endif
            } else if constexpr (Ltot == 2) {
#if defined(__AVX2__) && defined(__FMA__)
                accumulate_rys_quartets_lean2_fixed <
                    LaSum, LcSum, NRoots, NaSlice, NcFull, Emit > (
                    bra_slab, ket_slab, A, C, axs, ays, azs,
                    cxs, cys, czs, T_radial);
#else
                accumulate_rys_quartets_soa_fixed < LaSum, LcSum, NRoots, GStride, GSize,
                                                  NaSlice, NcFull, Emit > (
                    bra_slab, ket_slab, A, C, axs, ays, azs,
                    cxs, cys, czs, T_radial);
#endif
            } else {
                accumulate_rys_quartets_scalar_fixed < LaSum, LcSum, NRoots, GStride, GSize,
                                                     NaSlice, NcFull > (
                    bras, kets, A, C, axs, ays, azs, cxs, cys, czs, T_radial);
            }
        } else {
            accumulate_rys_quartets_soa_fixed < LaSum, LcSum, NRoots, GStride, GSize,
                                              NaSlice, NcFull, Emit > (
                bra_slab, ket_slab, A, C, axs, ays, azs, cxs, cys, czs, T_radial);
        }
    } else {
        // Classes with angular momentum above the compact direct maps are
        // handled by the fused early return above.
        return;
    }

    // Block-HRR + lean scatter for the all-p classes with a p on the right
    // (pp|pp, pp|ps, ps|pp): replaces the per-element serial-FMA expansion
    // below (the dominant per-block fixed cost at shallow contraction).
    constexpr bool use_block_hrr =
        (La == 1 && Lc == 1 && Lb <= 1 && Ld <= 1 && (Lb + Ld) >= 1);
    if constexpr (use_block_hrr) {
        double blk[Na * Nb * Nc * Nd];
        hrr_right01_block<La, Lb, Lc, Ld, NaSlice, NcFull>(T_radial, AB, CD, blk);
        scatter_fused_block_lean<Na, Nb, Nc, Nd, Emit>(
            blk, lhs_refs, rhs_refs, same_pair_group,
            pair_schwarz_bounds, schwarz_threshold, eri_values);
        return;
    }
    for (const PackedPairComponentH& lhs_ref : lhs_refs) {
        const int lhs_component = static_cast<int>(lhs_ref.component_index);
        const int a_local = lhs_component / Nb;
        const int b_local = lhs_component - a_local * Nb;
        const double lhs_bound = (pair_schwarz_bounds == nullptr)
                                     ? 0.0
                                     : (*pair_schwarz_bounds)[static_cast<std::size_t>(lhs_ref.pair_index)];
        for (const PackedPairComponentH& rhs_ref : rhs_refs) {
            if (same_pair_group && rhs_ref.pair_index > lhs_ref.pair_index) continue;
            if (pair_schwarz_bounds != nullptr &&
                lhs_bound * (*pair_schwarz_bounds)[static_cast<std::size_t>(rhs_ref.pair_index)] <
                    schwarz_threshold) {
                continue;
            }
            const int rhs_component = static_cast<int>(rhs_ref.component_index);
            const int c_local = rhs_component / Nd;
            const int d_local = rhs_component - c_local * Nd;
            double raw = 0.0;
            if constexpr (GeneratedMap::available) {
                const int entry_index = lhs_component * (Nc * Nd) + rhs_component;
                const RysGeneratedScatterEntry& entry = GeneratedMap::entries[entry_index];
                for (int ti = 0; ti < static_cast<int>(entry.term_count); ++ti) {
                    const RysGeneratedScatterTerm& term = entry.terms[ti];
                    raw += rys_generated_coeff_value(term, AB_monomials, CD_monomials) *
                           T_radial[term.t_index];
                }
            } else if constexpr (lowd_hrr_supported) {
                raw = eval_rys_lowd_hrr_raw<La, Lb, Lc, Ld>(
                    T_radial, NcFull, AB, CD, a_local, b_local, c_local, d_local);
            }
            const double val = raw * lhs_ref.norm * rhs_ref.norm;
            rys_emit_pairpair<Emit>(eri_values, lhs_ref.pair_index, rhs_ref.pair_index, val);
        }
    }
}

// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
bool try_scatter_pairpair_rys_direct(
    const ContractedShellGroupH& gA,
    const ContractedShellGroupH& gB,
    const ContractedShellGroupH& gC,
    const ContractedShellGroupH& gD,
    const std::vector<RysPrimitivePair>& bras,
    const std::vector<RysPrimitivePair>& kets,
    const RysPrimitivePairSlab& bra_slab,
    const RysPrimitivePairSlab& ket_slab,
    const std::vector<PackedPairComponentH>& lhs_refs,
    const std::vector<PackedPairComponentH>& rhs_refs,
    bool same_pair_group,
    const std::vector<double>* pair_schwarz_bounds,
    double schwarz_threshold,
    std::vector<double>& eri_values) {
#define TLNO_TRY_RYS_EXACT_F(LA_VALUE, LB_VALUE, LC_VALUE, LD_VALUE)                      \
    do {                                                                                   \
        if (gA.L == (LA_VALUE) && gB.L == (LB_VALUE) &&                                    \
            gC.L == (LC_VALUE) && gD.L == (LD_VALUE)) {                                    \
            scatter_pairpair_rys_right0123_direct<LA_VALUE, LB_VALUE, LC_VALUE, LD_VALUE, Emit>( \
                gA, gB, gC, gD, bras, kets, bra_slab, ket_slab, lhs_refs, rhs_refs,         \
                same_pair_group, pair_schwarz_bounds, schwarz_threshold, eri_values);       \
            return true;                                                                   \
        }                                                                                  \
    } while (0)

    TLNO_TRY_RYS_EXACT_F(0, 0, 3, 3); // ss|ff
    TLNO_TRY_RYS_EXACT_F(1, 0, 3, 3); // ps|ff
    TLNO_TRY_RYS_EXACT_F(1, 1, 3, 3); // pp|ff
    TLNO_TRY_RYS_EXACT_F(2, 0, 3, 3); // ds|ff
    TLNO_TRY_RYS_EXACT_F(2, 1, 3, 3); // dp|ff
    TLNO_TRY_RYS_EXACT_F(3, 0, 3, 3); // fs|ff
    TLNO_TRY_RYS_EXACT_F(3, 1, 3, 3); // fp|ff
    TLNO_TRY_RYS_EXACT_F(3, 3, 0, 0); // ff|ss
    TLNO_TRY_RYS_EXACT_F(3, 3, 1, 0); // ff|ps
    TLNO_TRY_RYS_EXACT_F(3, 3, 1, 1); // ff|pp
    TLNO_TRY_RYS_EXACT_F(3, 3, 2, 0); // ff|ds
    TLNO_TRY_RYS_EXACT_F(3, 3, 2, 1); // ff|dp

#undef TLNO_TRY_RYS_EXACT_F

    if (gB.L > 2 || gD.L > 2) return false;

#define TLNO_DISPATCH_RYS_RIGHT01_LD(LA_VALUE, LB_VALUE, LC_VALUE)                         \
    do {                                                                                   \
        switch (gD.L) {                                                                    \
            case 0: scatter_pairpair_rys_right0123_direct<LA_VALUE, LB_VALUE, LC_VALUE, 0, Emit>( \
                        gA, gB, gC, gD, bras, kets, bra_slab, ket_slab, lhs_refs, rhs_refs, same_pair_group, \
                        pair_schwarz_bounds, schwarz_threshold, eri_values); return true;  \
            case 1: scatter_pairpair_rys_right0123_direct<LA_VALUE, LB_VALUE, LC_VALUE, 1, Emit>( \
                        gA, gB, gC, gD, bras, kets, bra_slab, ket_slab, lhs_refs, rhs_refs, same_pair_group, \
                        pair_schwarz_bounds, schwarz_threshold, eri_values); return true;  \
            case 2: scatter_pairpair_rys_right0123_direct<LA_VALUE, LB_VALUE, LC_VALUE, 2, Emit>( \
                        gA, gB, gC, gD, bras, kets, bra_slab, ket_slab, lhs_refs, rhs_refs, same_pair_group, \
                        pair_schwarz_bounds, schwarz_threshold, eri_values); return true;  \
            default: break;                                                               \
        }                                                                                  \
    } while (0)

#define TLNO_DISPATCH_RYS_RIGHT01_LC(LA_VALUE, LB_VALUE)                                  \
    do {                                                                                   \
        switch (gC.L) {                                                                    \
            case 0: TLNO_DISPATCH_RYS_RIGHT01_LD(LA_VALUE, LB_VALUE, 0); break;            \
            case 1: TLNO_DISPATCH_RYS_RIGHT01_LD(LA_VALUE, LB_VALUE, 1); break;            \
            case 2: TLNO_DISPATCH_RYS_RIGHT01_LD(LA_VALUE, LB_VALUE, 2); break;            \
            case 3: TLNO_DISPATCH_RYS_RIGHT01_LD(LA_VALUE, LB_VALUE, 3); break;            \
            case 4: TLNO_DISPATCH_RYS_RIGHT01_LD(LA_VALUE, LB_VALUE, 4); break;            \
            default: break;                                                               \
        }                                                                                  \
    } while (0)

#define TLNO_DISPATCH_RYS_RIGHT01_LB(LA_VALUE)                                            \
    do {                                                                                   \
        switch (gB.L) {                                                                    \
            case 0: TLNO_DISPATCH_RYS_RIGHT01_LC(LA_VALUE, 0); break;                      \
            case 1: TLNO_DISPATCH_RYS_RIGHT01_LC(LA_VALUE, 1); break;                      \
            case 2: TLNO_DISPATCH_RYS_RIGHT01_LC(LA_VALUE, 2); break;                      \
            default: break;                                                               \
        }                                                                                  \
    } while (0)

    switch (gA.L) {
        case 0: TLNO_DISPATCH_RYS_RIGHT01_LB(0); break;
        case 1: TLNO_DISPATCH_RYS_RIGHT01_LB(1); break;
        case 2: TLNO_DISPATCH_RYS_RIGHT01_LB(2); break;
        case 3: TLNO_DISPATCH_RYS_RIGHT01_LB(3); break;
        case 4: TLNO_DISPATCH_RYS_RIGHT01_LB(4); break;
        default: break;
    }

#undef TLNO_DISPATCH_RYS_RIGHT01_LB
#undef TLNO_DISPATCH_RYS_RIGHT01_LC
#undef TLNO_DISPATCH_RYS_RIGHT01_LD

    return false;
}


const RysSphericalSchwarzBounds&
get_rys_spherical_pair_schwarz_bounds(const BasisDataHGP& bd) {
    {
        const std::lock_guard<std::mutex> lock(
            bd.rys_spherical_schwarz_bounds_mutex);
        if (bd.rys_spherical_schwarz_bounds != nullptr) {
            return *bd.rys_spherical_schwarz_bounds;
        }
    }

    auto built = std::make_shared<RysSphericalSchwarzBounds>();
    const std::size_t n =
        static_cast<std::size_t>(bd.spherical_function_count);
    built->pair_bounds.assign(n * (n + 1u) / 2u, 0.0);
    built->pair_group_max.assign(bd.pair_groups.size(), 0.0);
    const int pair_group_count =
        static_cast<int>(bd.pair_groups.size());

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) if (pair_group_count >= 16)
#endif
    for (int pi = 0; pi < pair_group_count; ++pi) {
        const std::size_t pair_group = static_cast<std::size_t>(pi);
        const auto [lg, rg] = bd.pair_groups[pair_group];
        const auto& refs =
            bd.packed_spherical_pair_components[pair_group];
        if (refs.empty()) continue;

        const auto& gA = bd.groups[static_cast<std::size_t>(lg)];
        const auto& gB = bd.groups[static_cast<std::size_t>(rg)];
        thread_local std::vector<double> cart_self;
        compute_quartet_block_rys_impl(
            gA, gB, gA, gB,
            &bd.rys_pair_primitives[pair_group],
            &bd.rys_pair_primitives[pair_group],
            bd.rys_pair_slabs.empty() ? nullptr : &bd.rys_pair_slabs[pair_group],
            bd.rys_pair_slabs.empty() ? nullptr : &bd.rys_pair_slabs[pair_group],
            cart_self);

        const FinalSphPairProjector& projector =
            get_final_sph_pair_projector(gA.L, gB.L);
        const std::size_t cart_cols =
            static_cast<std::size_t>(projector.cart_cols);
        double group_max = 0.0;
        for (const PackedPairComponentH& ref : refs) {
            const auto& row = projector.rows[
                static_cast<std::size_t>(ref.component_index)];
            long double self = 0.0L;
            long double absolute_sum = 0.0L;
            for (const SphProjectorTerm& lhs : row) {
                const std::size_t lhs_offset =
                    static_cast<std::size_t>(lhs.index) * cart_cols;
                for (const SphProjectorTerm& rhs : row) {
                    const long double term =
                        static_cast<long double>(lhs.coeff) *
                        static_cast<long double>(rhs.coeff) *
                        static_cast<long double>(
                            cart_self[lhs_offset +
                                      static_cast<std::size_t>(rhs.index)]);
                    self += term;
                    absolute_sum += std::fabs(term);
                }
            }
            // The projected Coulomb self norm is non-negative analytically.
            // Accumulating in long double and padding by a roundoff guard makes
            // the published double a conservative numerical bound.
            const long double roundoff_guard =
                64.0L *
                static_cast<long double>(
                    std::numeric_limits<double>::epsilon()) *
                absolute_sum;
            const long double safe_self =
                self >= 0.0L
                    ? self + roundoff_guard
                    : absolute_sum + roundoff_guard;
            double bound =
                std::sqrt(static_cast<double>(safe_self));
            if (bound > 0.0) {
                bound = std::nextafter(
                    bound, std::numeric_limits<double>::infinity());
            }
            built->pair_bounds[
                static_cast<std::size_t>(ref.pair_index)] = bound;
            group_max = std::max(group_max, bound);
        }
        built->pair_group_max[pair_group] = group_max;
    }

    const std::lock_guard<std::mutex> lock(
        bd.rys_spherical_schwarz_bounds_mutex);
    if (bd.rys_spherical_schwarz_bounds == nullptr) {
        bd.rys_spherical_schwarz_bounds = built;
    }
    return *bd.rys_spherical_schwarz_bounds;
}

inline std::shared_ptr<const RysScreenedSpFullBins>
build_rys_spherical_sp_full_bins(
    const BasisDataHGP& bd,
    const std::vector<double>& pair_group_metric) {
    auto built = std::make_shared<RysScreenedSpFullBins>();
    for (int type = 0; type <= 2; ++type) {
        const auto& source =
            bd.rys_low_pair_bins[static_cast<std::size_t>(type)];
        auto& destination =
            built->bins[static_cast<std::size_t>(type)];
        destination.resize(source.size());
        for (std::size_t nprim = 1; nprim < source.size(); ++nprim) {
            const RysLowPairBinH& bin = source[nprim];
            if (bin.pair_count == 0) continue;

            std::vector<std::uint32_t> ids(
                bin.pair_group_ids.begin(),
                bin.pair_group_ids.begin() +
                    static_cast<std::ptrdiff_t>(bin.pair_count));
            std::stable_sort(
                ids.begin(), ids.end(),
                [&pair_group_metric](
                    std::uint32_t lhs, std::uint32_t rhs) {
                    const double lhs_bound =
                        pair_group_metric[
                            static_cast<std::size_t>(lhs)];
                    const double rhs_bound =
                        pair_group_metric[
                            static_cast<std::size_t>(rhs)];
                    if (lhs_bound != rhs_bound) {
                        return lhs_bound > rhs_bound;
                    }
                    return lhs < rhs;
                });

            RysLowPairBinH& sorted = destination[nprim];
            fill_rys_low_chunk(
                bd, ids.data(), ids.size(),
                static_cast<int>(nprim), sorted);
            sorted.shell_screen_bound.resize(sorted.stride);
            for (std::size_t lane = 0; lane < sorted.stride; ++lane) {
                sorted.shell_screen_bound[lane] =
                    pair_group_metric[
                        static_cast<std::size_t>(
                            sorted.pair_group_ids[lane])];
            }
            sorted.shell_screen_bound_max =
                sorted.shell_screen_bound.front();
            sorted.shell_screen_bound_min =
                sorted.shell_screen_bound[sorted.pair_count - 1u];
            // These sorted full bins feed only the six s/p x s/p task families.
            // Their own maximum is therefore the tight exact partner maximum;
            // including d/f would retain extra dead s/p lanes for no benefit.
            built->sp_pair_group_metric_max = std::max(
                built->sp_pair_group_metric_max,
                sorted.shell_screen_bound_max);
        }
    }
    return built;
}

inline std::shared_ptr<const RysScreenedSpFullBins>
get_rys_spherical_sp_full_bins(
    const BasisDataHGP& bd,
    const RysSphericalSchwarzBounds& schwarz) {
    {
        const std::lock_guard<std::mutex> lock(
            bd.rys_spherical_sp_full_bins_mutex);
        if (bd.rys_spherical_sp_full_bins != nullptr) {
            return bd.rys_spherical_sp_full_bins;
        }
    }
    const std::shared_ptr<const RysScreenedSpFullBins> built =
        build_rys_spherical_sp_full_bins(
            bd, schwarz.pair_group_max);

    const std::lock_guard<std::mutex> lock(
        bd.rys_spherical_sp_full_bins_mutex);
    if (bd.rys_spherical_sp_full_bins == nullptr) {
        bd.rys_spherical_sp_full_bins = built;
    }
    return bd.rys_spherical_sp_full_bins;
}

inline std::shared_ptr<const RysScreenedSpFullBins>
get_rys_spherical_decay_sp_full_bins(const BasisDataHGP& bd) {
    {
        const std::lock_guard<std::mutex> lock(
            bd.rys_spherical_decay_sp_full_bins_mutex);
        if (bd.rys_spherical_decay_sp_full_bins != nullptr) {
            return bd.rys_spherical_decay_sp_full_bins;
        }
    }
    const std::shared_ptr<const RysScreenedSpFullBins> built =
        build_rys_spherical_sp_full_bins(
            bd, bd.pair_decay_bounds);

    const std::lock_guard<std::mutex> lock(
        bd.rys_spherical_decay_sp_full_bins_mutex);
    if (bd.rys_spherical_decay_sp_full_bins == nullptr) {
        bd.rys_spherical_decay_sp_full_bins = built;
    }
    return bd.rys_spherical_decay_sp_full_bins;
}

inline std::shared_ptr<const RysScreenedSphericalChunks>
build_rys_screened_spherical_chunks(
    const BasisDataHGP& bd,
    const std::vector<double>& pair_group_metric) {
    auto built = std::make_shared<RysScreenedSphericalChunks>();
    for (int code = 0; code < 10; ++code) {
        const auto& bins =
            bd.rys_all_pair_bins[static_cast<std::size_t>(code)];
        auto& class_chunks =
            built->chunks[static_cast<std::size_t>(code)];
        auto& class_chunk_bounds =
            built->chunk_max_bound[static_cast<std::size_t>(code)];
        auto& class_min =
            built->bin_min_bound[static_cast<std::size_t>(code)];
        auto& class_max =
            built->bin_max_bound[static_cast<std::size_t>(code)];
        class_chunks.resize(bins.size());
        class_chunk_bounds.resize(bins.size());
        class_min.assign(bins.size(), 0.0);
        class_max.assign(bins.size(), 0.0);

        const std::size_t chunk_width =
            code <= 2 ? rys_screened_spherical_low_l_chunk_pairs()
                      : rys_screened_spherical_high_l_chunk_pairs();
        for (std::size_t nprim = 1; nprim < bins.size(); ++nprim) {
            const auto& bin = bins[nprim];
            if (bin.pair_count == 0) continue;
            std::vector<std::uint32_t> ids(
                bin.pair_group_ids.begin(),
                bin.pair_group_ids.begin() +
                    static_cast<std::ptrdiff_t>(bin.pair_count));
            std::stable_sort(
                ids.begin(), ids.end(),
                [&pair_group_metric](
                    std::uint32_t a, std::uint32_t b) {
                    const double lhs =
                        pair_group_metric[
                            static_cast<std::size_t>(a)];
                    const double rhs =
                        pair_group_metric[
                            static_cast<std::size_t>(b)];
                    if (lhs != rhs) return lhs > rhs;
                    return a < b;
                });
            class_max[nprim] =
                pair_group_metric[
                    static_cast<std::size_t>(ids.front())];
            class_min[nprim] =
                pair_group_metric[
                    static_cast<std::size_t>(ids.back())];

            const std::size_t count =
                (ids.size() + chunk_width - 1u) / chunk_width;
            class_chunks[nprim].resize(count);
            class_chunk_bounds[nprim].resize(count);
            for (std::size_t chunk = 0; chunk < count; ++chunk) {
                const std::size_t lo = chunk * chunk_width;
                const std::size_t hi =
                    std::min(ids.size(), lo + chunk_width);
                RysLowPairBinH& built_chunk =
                    class_chunks[nprim][chunk];
                fill_rys_low_chunk(
                    bd, ids.data() + lo, hi - lo,
                    static_cast<int>(nprim),
                    built_chunk);

                // q_floor d/f consumes the same metric-specific contiguous
                // bounds as the s/p prefix.  The real lanes are descending;
                // SIMD padding repeats the last real lane and is never searched.
                built_chunk.shell_screen_bound.resize(built_chunk.stride);
                for (std::size_t lane = 0;
                     lane < built_chunk.stride; ++lane) {
                    const double metric = pair_group_metric[
                        static_cast<std::size_t>(
                            built_chunk.pair_group_ids[lane])];
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
                    assert(tlno::core::math::is_finite_binary64(metric));
                    assert(metric >= 0.0);
#endif
                    built_chunk.shell_screen_bound[lane] = metric;
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
                    if (lane != 0 && lane < built_chunk.pair_count) {
                        assert(
                            built_chunk.shell_screen_bound[lane - 1u] >=
                            metric);
                    }
#endif
                }
                built_chunk.shell_screen_bound_max =
                    built_chunk.shell_screen_bound.front();
                built_chunk.shell_screen_bound_min =
                    built_chunk.shell_screen_bound[
                        built_chunk.pair_count - 1u];
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
                if (chunk != 0u) {
                    assert(
                        class_chunks[nprim][chunk - 1u]
                                .shell_screen_bound_min >=
                            built_chunk.shell_screen_bound_max);
                }
#endif
                class_chunk_bounds[nprim][chunk] =
                    built_chunk.shell_screen_bound_max;
            }
        }
    }
    return built;
}

inline std::shared_ptr<const RysScreenedSphericalChunks>
get_rys_screened_spherical_chunks(
    const BasisDataHGP& bd,
    const RysSphericalSchwarzBounds& schwarz) {
    {
        const std::lock_guard<std::mutex> lock(
            bd.rys_spherical_screened_chunks_mutex);
        if (bd.rys_spherical_screened_chunks != nullptr) {
            return bd.rys_spherical_screened_chunks;
        }
    }
    const std::shared_ptr<const RysScreenedSphericalChunks> built =
        build_rys_screened_spherical_chunks(
            bd, schwarz.pair_group_max);

    const std::lock_guard<std::mutex> lock(
        bd.rys_spherical_screened_chunks_mutex);
    if (bd.rys_spherical_screened_chunks == nullptr) {
        bd.rys_spherical_screened_chunks = built;
    }
    return bd.rys_spherical_screened_chunks;
}

inline std::shared_ptr<const RysScreenedSphericalChunks>
get_rys_decay_screened_spherical_chunks(const BasisDataHGP& bd) {
    {
        const std::lock_guard<std::mutex> lock(
            bd.rys_spherical_decay_screened_chunks_mutex);
        if (bd.rys_spherical_decay_screened_chunks != nullptr) {
            return bd.rys_spherical_decay_screened_chunks;
        }
    }
    const std::shared_ptr<const RysScreenedSphericalChunks> built =
        build_rys_screened_spherical_chunks(
            bd, bd.pair_decay_bounds);

    const std::lock_guard<std::mutex> lock(
        bd.rys_spherical_decay_screened_chunks_mutex);
    if (bd.rys_spherical_decay_screened_chunks == nullptr) {
        bd.rys_spherical_decay_screened_chunks = built;
    }
    return bd.rys_spherical_decay_screened_chunks;
}

// q_floor d/f for one metric-sorted chunk.  partner_qmax is the largest real
// metric in the opposing chunk, so every lane in the returned prefix may still
// have a partner while every lane in the suffix provably fails the existing
// Qi*Qj >= threshold predicate.  Direct multiplication preserves its equality
// boundary and avoids a rounded threshold/Qmax division.
inline std::uint32_t rys_df_q_floor_live_pair_count(
    const RysLowPairBinH& bin,
    double partner_qmax,
    double threshold) {
    if (threshold <= 0.0) {
        return static_cast<std::uint32_t>(bin.pair_count);
    }
    if (bin.pair_count == 0) return 0u;
    if (!tlno::core::math::is_finite_binary64(partner_qmax)) {
        // Invalid or infinite external metrics must over-retain, never prune.
        return static_cast<std::uint32_t>(bin.pair_count);
    }
    if (partner_qmax < 0.0) {
        // A negative screening metric is invalid: fail open instead of
        // turning corrupted metadata into a false all-dead prefix.
        return static_cast<std::uint32_t>(bin.pair_count);
    }
    if (partner_qmax == 0.0) return 0u;

    if (bin.shell_screen_bound.size() < bin.pair_count) {
        return static_cast<std::uint32_t>(bin.pair_count);
    }
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    assert(bin.shell_screen_bound.size() >= bin.stride);
#endif
    if (!tlno::core::math::is_finite_binary64(
            bin.shell_screen_bound_max) ||
        !tlno::core::math::is_finite_binary64(
            bin.shell_screen_bound_min) ||
        bin.shell_screen_bound_max < 0.0 ||
        bin.shell_screen_bound_min < 0.0) {
        return static_cast<std::uint32_t>(bin.pair_count);
    }
    if (bin.shell_screen_bound_max * partner_qmax < threshold) return 0u;
    if (bin.shell_screen_bound_min * partner_qmax >= threshold) {
        return static_cast<std::uint32_t>(bin.pair_count);
    }
    const auto first = bin.shell_screen_bound.begin();
    const auto last = first +
        static_cast<std::ptrdiff_t>(bin.pair_count);
    const auto first_dead = std::partition_point(
        first, last,
        [partner_qmax, threshold](double metric) noexcept {
            return metric * partner_qmax >= threshold;
        });
    const std::uint32_t live = static_cast<std::uint32_t>(
        first_dead - first);

#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    for (auto it = first; it != first_dead; ++it) {
        assert(*it * partner_qmax >= threshold);
    }
    for (auto it = first_dead; it != last; ++it) {
        assert(*it * partner_qmax < threshold);
    }
#endif
    return live;
}

template <class Emit, typename Callback>
inline void for_each_spherical_screened_bin_pair(
    const RysLowPairBinH& lhs_bin,
    const std::vector<RysLowPairBinH>& lhs_chunks,
    const std::vector<double>& lhs_chunk_max,
    double lhs_min,
    double lhs_max,
    const RysLowPairBinH& rhs_bin,
    const std::vector<RysLowPairBinH>& rhs_chunks,
    const std::vector<double>& rhs_chunk_max,
    double rhs_min,
    double rhs_max,
    bool triangular,
    bool use_prefix_q_floor,
    double threshold,
    Callback&& callback) {
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    assert(!use_prefix_q_floor || !triangular);
#endif
    RysSphericalScreeningDiagnostics* const diag =
        g_rys_spherical_screening_diagnostics;
    if (diag != nullptr) {
        diag->scheduled_pair_group_quartets +=
            rys_spherical_bin_pair_work(
                lhs_bin, rhs_bin, triangular);
    }
    if (lhs_max * rhs_max < threshold) {
        if (diag != nullptr) {
            ++diag->whole_bin_pairs_rejected;
            diag->rejected_pair_group_quartets +=
                rys_spherical_bin_pair_work(
                    lhs_bin, rhs_bin, triangular);
        }
        return;
    }
    if (lhs_min * rhs_min >= threshold &&
        (g_rys_density_screen == nullptr || threshold <= 0.0)) {
        if (diag != nullptr) {
            ++diag->whole_bin_pairs_accepted;
            diag->accepted_pair_group_quartets +=
                rys_spherical_bin_pair_work(
                    lhs_bin, rhs_bin, triangular);
        }
        // With no density screen, the full bins need no in-kernel mask: every
        // quartet passes. A density-screened J/K call deliberately continues
        // into the sorted chunks below even when Schwarz accepts this whole
        // bin, because the local CVHF density cap can still reject quartets.
        // Raw ERI calls retain this zero-overhead unmasked path.
        callback(
            lhs_bin, lhs_bin.pair_count,
            rhs_bin, rhs_bin.pair_count,
            triangular, false);
        return;
    }

    if (diag != nullptr) {
        ++diag->partial_bin_pairs;
    }
    if (lhs_chunks.empty() || rhs_chunks.empty()) {
        return;
    }
    const auto record_rejected_chunk =
        [diag, triangular](const RysLowPairBinH& lhs_chunk,
                           const RysLowPairBinH& rhs_chunk,
                           bool diagonal) noexcept {
            if (diag == nullptr) return;
            ++diag->chunk_pairs_rejected;
            diag->rejected_pair_group_quartets +=
                rys_spherical_bin_pair_work(
                    lhs_chunk, rhs_chunk,
                    triangular && diagonal);
        };
    for (std::size_t lhs = 0; lhs < lhs_chunks.size(); ++lhs) {
        const std::size_t rhs_end = std::min(
            triangular ? lhs + 1u : rhs_chunks.size(),
            rhs_chunks.size());

        // Both chunk sequences are Q-descending. If this lhs maximum cannot
        // pass even against the first (largest) rhs maximum, neither this row
        // nor any remaining lhs row can pass. Diagnostics account for the
        // skipped triangular tail only when explicitly requested.
        if (lhs_chunk_max[lhs] * rhs_chunk_max.front() < threshold) {
            if (diag != nullptr) {
                for (std::size_t lhs_tail = lhs;
                     lhs_tail < lhs_chunks.size(); ++lhs_tail) {
                    const std::size_t rhs_tail_end = std::min(
                        triangular ? lhs_tail + 1u : rhs_chunks.size(),
                        rhs_chunks.size());
                    for (std::size_t rhs_tail = 0;
                         rhs_tail < rhs_tail_end; ++rhs_tail) {
                        record_rejected_chunk(
                            lhs_chunks[lhs_tail], rhs_chunks[rhs_tail],
                            lhs_tail == rhs_tail);
                    }
                }
            }
            break;
        }
        for (std::size_t rhs = 0; rhs < rhs_end; ++rhs) {
            if (lhs_chunk_max[lhs] * rhs_chunk_max[rhs] <
                threshold) {
                // The rhs maxima are descending, so the rest of this row is
                // an exact rejected tail rather than a sequence of continues.
                if (diag != nullptr) {
                    for (std::size_t rhs_tail = rhs;
                         rhs_tail < rhs_end; ++rhs_tail) {
                        record_rejected_chunk(
                            lhs_chunks[lhs], rhs_chunks[rhs_tail],
                            lhs == rhs_tail);
                    }
                }
                break;
            }
            if (diag != nullptr) {
                ++diag->chunk_pairs_accepted;
                diag->accepted_pair_group_quartets +=
                    rys_spherical_bin_pair_work(
                        lhs_chunks[lhs], rhs_chunks[rhs],
                        triangular && lhs == rhs);
            }
            std::size_t lhs_live_pair_count =
                lhs_chunks[lhs].pair_count;
            std::size_t rhs_live_pair_count =
                rhs_chunks[rhs].pair_count;
            if (use_prefix_q_floor) {
                lhs_live_pair_count = rys_df_q_floor_live_pair_count(
                    lhs_chunks[lhs], rhs_chunk_max[rhs], threshold);
                rhs_live_pair_count = rys_df_q_floor_live_pair_count(
                    rhs_chunks[rhs], lhs_chunk_max[lhs], threshold);
                // The chunk-max predicate above guarantees that the leading
                // lane of each chunk survives against the other.  Within each
                // sorted chunk there is at most one q_floor frontier: a live
                // prefix followed by a dead suffix.
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
                assert(lhs_live_pair_count != 0u);
                assert(rhs_live_pair_count != 0u);
#endif
                if (diag != nullptr) {
                    const std::uint64_t candidate =
                        rys_spherical_bin_pair_work(
                            lhs_chunks[lhs], rhs_chunks[rhs], false);
                    const std::uint64_t visited =
                        static_cast<std::uint64_t>(
                            lhs_live_pair_count) *
                        static_cast<std::uint64_t>(
                            rhs_live_pair_count);
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
                    assert(visited <= candidate);
#endif
                    ++diag->df_q_floor_tasks;
                    diag->df_q_floor_candidate_pair_group_quartets +=
                        candidate;
                    diag->df_q_floor_visited_pair_group_quartets +=
                        visited;
                    diag->df_q_floor_pruned_pair_group_quartets +=
                        candidate - visited;
                }
            }
            if (use_prefix_q_floor) {
#if defined(__AVX2__) && defined(__FMA__)
                const RysDfQFloorPairWindow window{
                    &lhs_chunks[lhs], lhs_live_pair_count,
                    &rhs_chunks[rhs], rhs_live_pair_count};
                const RysDfQFloorPairWindowGuard window_guard(
                    g_rys_batch_screen, &window);
#endif
                callback(
                    lhs_chunks[lhs], lhs_live_pair_count,
                    rhs_chunks[rhs], rhs_live_pair_count,
                    triangular && lhs == rhs, true);
            } else {
                callback(
                    lhs_chunks[lhs], lhs_live_pair_count,
                    rhs_chunks[rhs], rhs_live_pair_count,
                    triangular && lhs == rhs, true);
            }
        }
    }
}

template <class Emit, typename Callback>
inline void for_each_spherical_screened_cross_class(
    const BasisDataHGP& bd,
    const RysScreenedSphericalChunks& chunks,
    int lhs_code,
    int rhs_code,
    bool use_prefix_q_floor,
    double threshold,
    Callback&& callback) {
    const auto& lhs_bins =
        bd.rys_all_pair_bins[static_cast<std::size_t>(lhs_code)];
    const auto& rhs_bins =
        bd.rys_all_pair_bins[static_cast<std::size_t>(rhs_code)];
    const auto& lhs_chunks =
        chunks.chunks[static_cast<std::size_t>(lhs_code)];
    const auto& rhs_chunks =
        chunks.chunks[static_cast<std::size_t>(rhs_code)];
    const auto& lhs_chunk_max =
        chunks.chunk_max_bound[static_cast<std::size_t>(lhs_code)];
    const auto& rhs_chunk_max =
        chunks.chunk_max_bound[static_cast<std::size_t>(rhs_code)];
    const auto& lhs_min =
        chunks.bin_min_bound[static_cast<std::size_t>(lhs_code)];
    const auto& rhs_min =
        chunks.bin_min_bound[static_cast<std::size_t>(rhs_code)];
    const auto& lhs_max =
        chunks.bin_max_bound[static_cast<std::size_t>(lhs_code)];
    const auto& rhs_max =
        chunks.bin_max_bound[static_cast<std::size_t>(rhs_code)];

    for (std::size_t li = 1; li < lhs_bins.size(); ++li) {
        if (lhs_bins[li].pair_count == 0) continue;
        for (std::size_t ri = 1; ri < rhs_bins.size(); ++ri) {
            if (rhs_bins[ri].pair_count == 0) continue;
            for_each_spherical_screened_bin_pair<Emit>(
                lhs_bins[li], lhs_chunks[li], lhs_chunk_max[li],
                lhs_min[li], lhs_max[li],
                rhs_bins[ri], rhs_chunks[ri], rhs_chunk_max[ri],
                rhs_min[ri], rhs_max[ri],
                false, use_prefix_q_floor, threshold, callback);
        }
    }
}

// Emit is explicit through both traversal helpers; omission is a compile error.
//
// MIND THE ORDER: here Emit comes FIRST, unlike in the rest of the engine.
// Callback is deduced from the lambda this function receives, and explicit
// template arguments are filled from the left: with Emit last, writing
// <Emit> in the call would put it in Callback's slot and break the
// deduction.  With Emit first, <Emit> takes its own slot and Callback is
// still deduced from the argument.  The general rule: Emit goes last when
// all the preceding parameters are explicit, and first when any of them is
// deduced.
template <class Emit, typename Callback>
inline void for_each_spherical_screened_same_class(
    const BasisDataHGP& bd,
    const RysScreenedSphericalChunks& chunks,
    int code,
    double threshold,
    Callback&& callback) {
    const auto& bins =
        bd.rys_all_pair_bins[static_cast<std::size_t>(code)];
    const auto& class_chunks =
        chunks.chunks[static_cast<std::size_t>(code)];
    const auto& chunk_max =
        chunks.chunk_max_bound[static_cast<std::size_t>(code)];
    const auto& bin_min =
        chunks.bin_min_bound[static_cast<std::size_t>(code)];
    const auto& bin_max =
        chunks.bin_max_bound[static_cast<std::size_t>(code)];
    for (std::size_t li = 1; li < bins.size(); ++li) {
        if (bins[li].pair_count == 0) continue;
        for (std::size_t ri = 1; ri <= li; ++ri) {
            if (ri >= bins.size() || bins[ri].pair_count == 0) continue;
            for_each_spherical_screened_bin_pair<Emit>(
                bins[li], class_chunks[li], chunk_max[li],
                bin_min[li], bin_max[li],
                bins[ri], class_chunks[ri], chunk_max[ri],
                bin_min[ri], bin_max[ri],
                li == ri, false, threshold, callback);
        }
    }
}

template <class Emit>
EriPackedGData
build_eri_packed_for_G_rys_spherical_stream_unscreened_impl(
    const BasisDataHGP& bd,
    bool materialize_validation) {
#ifdef _OPENMP
    if (omp_get_max_threads() != 1) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: pure-spherical checksum stream is not parallelized yet; "
            "set OMP_NUM_THREADS=1.");
    }
#endif
    EriPackedGData eri;
    EriStreamAccumulator stream_accumulator;

#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    EriMaterializedSphericalAudit materialized_audit;
    if (materialize_validation) {
        const std::size_t expected = spherical_packed_value_count(bd);
        eri.values.assign(expected, 0.0);
        materialized_audit.reset(expected);
    }
    EriMaterializedSphericalAuditGuard materialized_audit_guard(
        materialize_validation ? &materialized_audit : nullptr);
    LowBatchPairComponentsGuard materialized_pair_components(
        materialize_validation, &bd.packed_spherical_pair_components);
#else
    if (materialize_validation) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: materialized pure-spherical validation "
            "requires TLNO_VALIDATE_RYS_SPHERICAL_PACKED.");
    }
#endif

    // Pure-spherical canonical indices: point the shared low-L batch kernels at
    // the spherical pair table so their emitted values land on the same
    // canonical index the projected d/generic path uses.
    EriStreamModeGuard stream_mode(
        !materialize_validation, &bd.packed_spherical_pair_components,
        &stream_accumulator);

    const auto& groups = bd.groups;
    const auto& pair_groups = bd.pair_groups;
#if defined(__AVX2__) && defined(__FMA__)
    bool has_low_pair = false;
    bool has_fs_pair = false;
    bool has_dp_pair = false;
    bool has_ds_pair = false;
    bool has_pp_pair = false;
    bool has_dd_pair = false;
    bool all_shell4_pair_classes = true;
    for (std::size_t pg = 0; pg < pair_groups.size(); ++pg) {
        const auto [left_group, right_group] = pair_groups[pg];
        const auto& gL = groups[static_cast<std::size_t>(left_group)];
        const auto& gR = groups[static_cast<std::size_t>(right_group)];
        const bool ss = gL.L == 0 && gR.L == 0;
        const bool ps = gL.L == 1 && gR.L == 0;
        has_low_pair = has_low_pair || ss || ps;
        has_fs_pair = has_fs_pair || (gL.L == 3 && gR.L == 0);
        has_dp_pair = has_dp_pair || (gL.L == 2 && gR.L == 1);
        has_ds_pair = has_ds_pair || (gL.L == 2 && gR.L == 0);
        has_pp_pair = has_pp_pair || (gL.L == 1 && gR.L == 1);
        has_dd_pair = has_dd_pair || (gL.L == 2 && gR.L == 2);
        all_shell4_pair_classes =
            all_shell4_pair_classes &&
            rys_spherical_pair_class_code(gL.L, gR.L) >= 0;
    }

    accumulate_rys_sp_shell_batches<Emit>(bd, eri.values);
    accumulate_rys_ssds_spherical_shell_batches<Emit>(bd, eri.values);
    accumulate_rys_dsps_spherical_shell_batches<Emit>(bd, eri.values);
    const bool use_fs_shell_batch = has_fs_pair && has_low_pair;
    if (use_fs_shell_batch) {
        accumulate_rys_fs_spherical_shell_batches<Emit>(bd, eri.values);
    }
    const bool use_dp_shell_batch = has_dp_pair && has_low_pair;
    if (use_dp_shell_batch) {
        accumulate_rys_dp_spherical_shell_batches<Emit>(bd, eri.values);
    }
    const bool use_dp_matrix_shell_batch =
        has_dp_pair && (has_ds_pair || has_pp_pair) &&
        rys_dp_matrix_shell_batch_profitable(bd);
    if (use_dp_matrix_shell_batch) {
        accumulate_rys_dp_matrix_spherical_shell_batches<Emit>(bd, eri.values);
    }
    const bool use_ds_matrix_shell_batch =
        has_ds_pair && (has_ds_pair || has_pp_pair);
    if (use_ds_matrix_shell_batch) {
        accumulate_rys_ds_matrix_spherical_shell_batches<Emit>(bd, eri.values);
    }
    const bool use_dd_low_shell_batch =
        has_dd_pair && has_low_pair &&
        rys_dd_low_shell_batch_profitable(bd);
    if (use_dd_low_shell_batch) {
        accumulate_rys_dd_low_spherical_shell_batches<Emit>(bd, eri.values);
    }
    accumulate_rys_generic_spherical_shell4_batches<Emit>(
        bd, use_dp_matrix_shell_batch, use_dd_low_shell_batch, eri.values);
#endif

#if defined(__AVX2__) && defined(__FMA__)
    const bool need_high_l_quartet_fallback = !all_shell4_pair_classes;
#else
    const bool need_high_l_quartet_fallback = true;
#endif
    if (need_high_l_quartet_fallback) {
      for (std::size_t ab = 0; ab < pair_groups.size(); ++ab) {
       for (std::size_t cd = 0; cd <= ab; ++cd) {
        const auto [lg, rg] = pair_groups[ab];
        const auto [lg2, rg2] = pair_groups[cd];
        const auto& gA = groups[static_cast<std::size_t>(lg)];
        const auto& gB = groups[static_cast<std::size_t>(rg)];
        const auto& gC = groups[static_cast<std::size_t>(lg2)];
        const auto& gD = groups[static_cast<std::size_t>(rg2)];
#if defined(__AVX2__) && defined(__FMA__)
        if (rys_spherical_pair_class_code(gA.L, gB.L) >= 0 &&
            rys_spherical_pair_class_code(gC.L, gD.L) >= 0) {
            // Every s-through-f class not consumed by a specialized kernel
            // above was already emitted by the universal class-bin scheduler.
            continue;
        }
#endif
        const auto& lhs_refs = bd.packed_spherical_pair_components[ab];
        const auto& rhs_refs = bd.packed_spherical_pair_components[cd];
        if (lhs_refs.empty() || rhs_refs.empty()) continue;
        const auto& bras = bd.rys_pair_primitives[ab];
        const auto& kets = bd.rys_pair_primitives[cd];
        const auto& bra_slab = bd.rys_pair_slabs[ab];
        const auto& ket_slab = bd.rys_pair_slabs[cd];
        const FinalSphPairProjector& lhs_projector =
            get_final_sph_pair_projector(gA.L, gB.L);

        // s and p have identical Cartesian and pure components.  Preserve the
        // tuned direct Rys path exactly for these blocks.
        if (gA.L <= 1 && gB.L <= 1 && gC.L <= 1 && gD.L <= 1 &&
            try_scatter_pairpair_rys_direct<Emit>(
                gA, gB, gC, gD,
                bras, kets, bra_slab, ket_slab,
                lhs_refs, rhs_refs, ab == cd, nullptr, 0.0, eri.values)) {
            continue;
        }

        thread_local std::vector<double> cart_block;
        const bool used_compiled_block =
            try_fused_rys_block_compiled_df_runtime(
                gA.L, gB.L, gC.L, gD.L,
                gA.center, gB.center, gC.center, gD.center,
                bras, kets, cart_block);
        if (!used_compiled_block) {
            // WHAT: carry the J-only active-root policy through the scalar
            // high-L escape, whose runtime API no longer has an Emit type.
            // WHY: otherwise an unsupported-above-f quartet would silently
            // restore the exact root count after every AVX shell4 producer had
            // selected one.  Exact/reference/packed/G keep the old expression.
            // PURPOSE: root-count coverage remains complete even at this
            // currently unreachable edge of the living s-through-f bases.
            const int nroots = std::is_same_v <
                    Emit, RysDirectJSpeedProbeEmit > &&
                    !kRysJSpeedProbeExactRoots
                ? 1
                : (gA.L + gB.L + gC.L + gD.L) / 2 + 1;
            fused_rys_block_runtime(
                gA.L, gB.L, gC.L, gD.L, nroots,
                gA.center, gB.center, gC.center, gD.center,
                bras, kets, cart_block);
        }

        const FinalSphPairProjector& rhs_projector =
            get_final_sph_pair_projector(gC.L, gD.L);
        const int rhs_cart = rhs_projector.cart_cols;
        thread_local std::vector<double> mid;
        mid.assign(static_cast<std::size_t>(lhs_projector.sph_rows) *
                       static_cast<std::size_t>(rhs_cart),
                   0.0);
        for (int lhs_row = 0; lhs_row < lhs_projector.sph_rows; ++lhs_row) {
            double* const dst =
                mid.data() + static_cast<std::size_t>(lhs_row) *
                                 static_cast<std::size_t>(rhs_cart);
            for (const SphProjectorTerm& term :
                 lhs_projector.rows[static_cast<std::size_t>(lhs_row)]) {
                const double* const src =
                    cart_block.data() + static_cast<std::size_t>(term.index) *
                                            static_cast<std::size_t>(rhs_cart);
                for (int j = 0; j < rhs_cart; ++j) {
                    dst[j] += term.coeff * src[j];
                }
            }
        }

        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            const double* const lhs_row =
                mid.data() + static_cast<std::size_t>(lhs_ref.component_index) *
                                 static_cast<std::size_t>(rhs_cart);
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (ab == cd && rhs_ref.pair_index > lhs_ref.pair_index) continue;
                double value = 0.0;
                for (const SphProjectorTerm& term :
                     rhs_projector.rows[static_cast<std::size_t>(rhs_ref.component_index)]) {
                    value += lhs_row[term.index] * term.coeff;
                }
                rys_emit_pairpair<Emit>(
                    eri.values, lhs_ref.pair_index, rhs_ref.pair_index, value);
            }
        }
       }
      }
    }

#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    // WHY the policy guard: the fingerprint is fed by eri_emit, which only the
    // packed policy goes through.  Policies that consume the values themselves
    // (direct J/K) leave the counter at zero by construction, not because they
    // lost integrals, and without this guard the check would reject all of
    // them.  Completeness is still required where it means something.
    if (!materialize_validation &&
        rys_emit_uses_packed_checksum_fold_v<Emit>) {
        const std::size_t expected = spherical_packed_value_count(bd);
        if (stream_accumulator.fingerprint.canonical_count != expected) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: incomplete stream fingerprint: got " +
                std::to_string(stream_accumulator.fingerprint.canonical_count) +
                ", expected " + std::to_string(expected));
        }
    }
    if (!materialize_validation) {
        const EriStreamFP& fp = stream_accumulator.fingerprint;
        if (fp.canonical_count) std::fprintf(stderr,
            "STREAMFP checksum=%.17Lg absolute_sum=%.17Lg square_sum=%.17Lg m1=%.17Lg m2=%.17Lg m3=%.17Lg max_abs=%.17Lg count=%llu\n",
            fp.checksum, fp.absolute_sum, fp.square_sum,
            fp.indexed_moment_1, fp.indexed_moment_2, fp.indexed_moment_3,
            fp.max_abs, fp.canonical_count);
    }
#endif
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    if (materialize_validation) {
        materialized_audit.validate(true);
        std::fprintf(
            stderr,
            "MATERIALIZED_SPHERICAL_AUDIT expected=%zu written=%zu "
            "implicit_zeros=0 require_complete=1\n",
            materialized_audit.expected_count,
            materialized_audit.unique_writes);
        return eri;
    }
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    // The moments are exposed next to the checksum ONLY in fingerprint builds:
    // they are the reference against which the J/K policy is compared.  In
    // production the vector keeps one element and the contract is unchanged.
    eri.values.assign(
        {stream_accumulator.checksum,
         static_cast<double>(stream_accumulator.fingerprint.indexed_moment_1),
         static_cast<double>(stream_accumulator.fingerprint.indexed_moment_2),
         static_cast<double>(stream_accumulator.fingerprint.indexed_moment_3),
         static_cast<double>(stream_accumulator.fingerprint.canonical_count)});
#else
    eri.values.assign(1, stream_accumulator.checksum);
#endif
    return eri;
}

#if defined(__AVX2__) && defined(__FMA__)
// Emit is explicit at this dispatcher boundary; omission is a compile error.
template <class Emit>
inline void accumulate_rys_spherical_batches_screened(
    const BasisDataHGP& bd,
    const RysScreenedSpFullBins& sorted_sp_full_bins,
    const RysScreenedSphericalChunks& chunks,
    const std::vector<double>& pair_group_metric,
    double threshold,
    double primitive_fastschwarz_threshold_squared,
    std::vector<double>& eri_values,
    bool& all_shell4_pair_classes) {
    // The batch screen is assembled here from whatever the caller installed: if
    // there is no density context, the fields stay null and the lane test
    // takes the usual branch.
    const RysDensityScreenContext* const density_context =
        g_rys_density_screen;
    RysBatchScreen batch_screen;
    batch_screen.q_break = threshold;
    batch_screen.primitive_fastschwarz_threshold_squared =
        primitive_fastschwarz_threshold_squared;
    batch_screen.pair_group_metric = pair_group_metric.data();
    batch_screen.j_only = rys_emit_uses_j_density_screen_v<Emit>;
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    batch_screen.k_only = rys_emit_is_k_only_v<Emit>;
#endif
    if (density_context != nullptr) {
        batch_screen.shell_rho = density_context->shell_rho;
        // WHAT/WHY: share the factorizable O(N_pair_groups) bound with the s/p
        // scheduler to prove tasks dead without a quartet prepass.
        batch_screen.pair_group_density_upper =
            density_context->pair_group_density_upper;
        batch_screen.group_pair_density_max =
            density_context->group_pair_density_max;
        batch_screen.density_ng = density_context->ng;
        batch_screen.pair_group_left = density_context->pair_group_left;
        batch_screen.pair_group_right = density_context->pair_group_right;
        batch_screen.density_break = density_context->density_break;
        batch_screen.j_only = density_context->j_only;
#ifdef TLNO_RYS_DIRECT_EXCHANGE
        batch_screen.k_only = density_context->k_only;
#endif
    }
    const RysBatchScreenGuard batch_screen_guard(&batch_screen);
    bool has_low_pair = false;
    bool has_fs_pair = false;
    bool has_dp_pair = false;
    bool has_ds_pair = false;
    bool has_pp_pair = false;
    bool has_dd_pair = false;
    all_shell4_pair_classes = true;
    for (const auto& pair_group : bd.pair_groups) {
        const auto& gL =
            bd.groups[static_cast<std::size_t>(pair_group.first)];
        const auto& gR =
            bd.groups[static_cast<std::size_t>(pair_group.second)];
        const bool ss = gL.L == 0 && gR.L == 0;
        const bool ps = gL.L == 1 && gR.L == 0;
        has_low_pair = has_low_pair || ss || ps;
        has_fs_pair = has_fs_pair || (gL.L == 3 && gR.L == 0);
        has_dp_pair = has_dp_pair || (gL.L == 2 && gR.L == 1);
        has_ds_pair = has_ds_pair || (gL.L == 2 && gR.L == 0);
        has_pp_pair = has_pp_pair || (gL.L == 1 && gR.L == 1);
        has_dd_pair = has_dd_pair || (gL.L == 2 && gR.L == 2);
        all_shell4_pair_classes =
            all_shell4_pair_classes &&
            rys_spherical_pair_class_code(gL.L, gR.L) >= 0;
    }

    // The s/p classes retain their original contraction-length bins.  Every
    // partial task now applies the exact shell-pair predicate in four-lane
    // packets before expensive work: the tuned d/f batches and the residual
    // shell4 dispatcher share the same metric context and monotone-tail rule.
    std::vector<RysSpShellBatchTask> sp_tasks;
    append_rys_sp_shell_batch_tasks_in_bin_screened(
        bd, sorted_sp_full_bins, threshold, sp_tasks);
    if (primitive_fastschwarz_threshold_squared > 0.0) {
        for (RysSpShellBatchTask& task : sp_tasks) {
            task.primitive_fastschwarz_threshold_squared =
                primitive_fastschwarz_threshold_squared;
        }
    }
    run_rys_sp_shell_batch_task_list<Emit>(bd, sp_tasks, eri_values);

    for_each_spherical_screened_cross_class<Emit>(
        bd, chunks, 3, 0, true, threshold,
        [&](const RysLowPairBinH& ds,
            std::size_t ds_pair_count,
            const RysLowPairBinH& ss,
            std::size_t ss_pair_count,
            bool,
            bool exact_mask) {
            if (ds_pair_count >= ss_pair_count) {
                if (exact_mask) {
                    accumulate_rys_ssds_spherical_shell_bin_pair<true, true, Emit>(
                        bd, ds, ss, eri_values);
                } else {
                    accumulate_rys_ssds_spherical_shell_bin_pair<true, false, Emit>(
                        bd, ds, ss, eri_values);
                }
            } else {
                if (exact_mask) {
                    accumulate_rys_ssds_spherical_shell_bin_pair<false, true, Emit>(
                        bd, ss, ds, eri_values);
                } else {
                    accumulate_rys_ssds_spherical_shell_bin_pair<false, false, Emit>(
                        bd, ss, ds, eri_values);
                }
            }
        });
    for_each_spherical_screened_cross_class<Emit>(
        bd, chunks, 3, 1, true, threshold,
        [&](const RysLowPairBinH& ds,
            std::size_t ds_pair_count,
            const RysLowPairBinH& ps,
            std::size_t ps_pair_count,
            bool,
            bool exact_mask) {
            if (ds_pair_count >= ps_pair_count) {
                if (exact_mask) {
                    accumulate_rys_dsps_spherical_shell_bin_pair<true, true, Emit>(
                        bd, ds, ps, eri_values);
                } else {
                    accumulate_rys_dsps_spherical_shell_bin_pair<true, false, Emit>(
                        bd, ds, ps, eri_values);
                }
            } else {
                if (exact_mask) {
                    accumulate_rys_dsps_spherical_shell_bin_pair<false, true, Emit>(
                        bd, ps, ds, eri_values);
                } else {
                    accumulate_rys_dsps_spherical_shell_bin_pair<false, false, Emit>(
                        bd, ps, ds, eri_values);
                }
            }
        });

    const bool use_fs_shell_batch = has_fs_pair && has_low_pair;
    if (use_fs_shell_batch) {
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 6, 0, true, threshold,
            [&](const RysLowPairBinH& fs,
                std::size_t fs_pair_count,
                const RysLowPairBinH& ss,
                std::size_t ss_pair_count,
                bool,
                bool exact_mask) {
                if (fs_pair_count >= ss_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            0, true, true, Emit > (
                                bd, fs, ss, eri_values);
                    } else {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            0, true, false, Emit > (
                                bd, fs, ss, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            0, false, true, Emit > (
                                bd, ss, fs, eri_values);
                    } else {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            0, false, false, Emit > (
                                bd, ss, fs, eri_values);
                    }
                }
            });
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 6, 1, true, threshold,
            [&](const RysLowPairBinH& fs,
                std::size_t fs_pair_count,
                const RysLowPairBinH& ps,
                std::size_t ps_pair_count,
                bool,
                bool exact_mask) {
                if (fs_pair_count >= ps_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            1, true, true, Emit > (
                                bd, fs, ps, eri_values);
                    } else {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            1, true, false, Emit > (
                                bd, fs, ps, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            1, false, true, Emit > (
                                bd, ps, fs, eri_values);
                    } else {
                        accumulate_rys_fs_low_spherical_shell_bin_pair <
                            1, false, false, Emit > (
                                bd, ps, fs, eri_values);
                    }
                }
            });
    }

    const bool use_dp_shell_batch = has_dp_pair && has_low_pair;
    if (use_dp_shell_batch) {
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 4, 0, true, threshold,
            [&](const RysLowPairBinH& dp,
                std::size_t dp_pair_count,
                const RysLowPairBinH& ss,
                std::size_t ss_pair_count,
                bool,
                bool exact_mask) {
                if (dp_pair_count >= ss_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            0, true, true, Emit > (
                                bd, dp, ss, eri_values);
                    } else {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            0, true, false, Emit > (
                                bd, dp, ss, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            0, false, true, Emit > (
                                bd, ss, dp, eri_values);
                    } else {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            0, false, false, Emit > (
                                bd, ss, dp, eri_values);
                    }
                }
            });
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 4, 1, true, threshold,
            [&](const RysLowPairBinH& dp,
                std::size_t dp_pair_count,
                const RysLowPairBinH& ps,
                std::size_t ps_pair_count,
                bool,
                bool exact_mask) {
                if (dp_pair_count >= ps_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            1, true, true, Emit > (
                                bd, dp, ps, eri_values);
                    } else {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            1, true, false, Emit > (
                                bd, dp, ps, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            1, false, true, Emit > (
                                bd, ps, dp, eri_values);
                    } else {
                        accumulate_rys_dp_low_spherical_shell_bin_pair <
                            1, false, false, Emit > (
                                bd, ps, dp, eri_values);
                    }
                }
            });
    }

    const bool use_dp_matrix_shell_batch =
        has_dp_pair && (has_ds_pair || has_pp_pair) &&
        rys_dp_matrix_shell_batch_profitable(bd);
    if (use_dp_matrix_shell_batch) {
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 4, 3, true, threshold,
            [&](const RysLowPairBinH& dp,
                std::size_t dp_pair_count,
                const RysLowPairBinH& ds,
                std::size_t ds_pair_count,
                bool,
                bool exact_mask) {
                if (dp_pair_count >= ds_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            false, true, true, Emit > (
                                bd, dp, ds, eri_values);
                    } else {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            false, true, false, Emit > (
                                bd, dp, ds, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            false, false, true, Emit > (
                                bd, ds, dp, eri_values);
                    } else {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            false, false, false, Emit > (
                                bd, ds, dp, eri_values);
                    }
                }
            });
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 4, 2, true, threshold,
            [&](const RysLowPairBinH& dp,
                std::size_t dp_pair_count,
                const RysLowPairBinH& pp,
                std::size_t pp_pair_count,
                bool,
                bool exact_mask) {
                if (dp_pair_count >= pp_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            true, true, true, Emit > (
                                bd, dp, pp, eri_values);
                    } else {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            true, true, false, Emit > (
                                bd, dp, pp, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            true, false, true, Emit > (
                                bd, pp, dp, eri_values);
                    } else {
                        accumulate_rys_dp_matrix_spherical_shell_bin_pair <
                            true, false, false, Emit > (
                                bd, pp, dp, eri_values);
                    }
                }
            });
    }

    const bool use_ds_matrix_shell_batch =
        has_ds_pair && (has_ds_pair || has_pp_pair);
    if (use_ds_matrix_shell_batch) {
        for_each_spherical_screened_same_class<Emit>(
            bd, chunks, 3, threshold,
            [&](const RysLowPairBinH& lhs,
                std::size_t,
                const RysLowPairBinH& rhs,
                std::size_t,
                bool same_bin,
                bool exact_mask) {
                const RysLowPairBinH* bra = &lhs;
                const RysLowPairBinH* ket = &rhs;
                if (!same_bin && ket->pair_count > bra->pair_count) {
                    std::swap(bra, ket);
                }
                if (exact_mask) {
                    accumulate_rys_dsds_spherical_shell_bin_pair<true, Emit>(
                        bd, *bra, *ket, same_bin, eri_values);
                } else {
                    accumulate_rys_dsds_spherical_shell_bin_pair<false, Emit>(
                        bd, *bra, *ket, same_bin, eri_values);
                }
            });
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 3, 2, true, threshold,
            [&](const RysLowPairBinH& ds,
                std::size_t ds_pair_count,
                const RysLowPairBinH& pp,
                std::size_t pp_pair_count,
                bool,
                bool exact_mask) {
                if (ds_pair_count >= pp_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_dspp_spherical_shell_bin_pair<true, true, Emit>(
                            bd, ds, pp, eri_values);
                    } else {
                        accumulate_rys_dspp_spherical_shell_bin_pair<true, false, Emit>(
                            bd, ds, pp, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_dspp_spherical_shell_bin_pair<false, true, Emit>(
                            bd, pp, ds, eri_values);
                    } else {
                        accumulate_rys_dspp_spherical_shell_bin_pair<false, false, Emit>(
                            bd, pp, ds, eri_values);
                    }
                }
            });
    }

    const bool use_dd_low_shell_batch =
        has_dd_pair && has_low_pair &&
        rys_dd_low_shell_batch_profitable(bd);
    if (use_dd_low_shell_batch) {
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 5, 0, true, threshold,
            [&](const RysLowPairBinH& dd,
                std::size_t dd_pair_count,
                const RysLowPairBinH& ss,
                std::size_t ss_pair_count,
                bool,
                bool exact_mask) {
                if (dd_pair_count >= ss_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            0, true, true, Emit > (
                                bd, dd, ss, eri_values);
                    } else {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            0, true, false, Emit > (
                                bd, dd, ss, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            0, false, true, Emit > (
                                bd, ss, dd, eri_values);
                    } else {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            0, false, false, Emit > (
                                bd, ss, dd, eri_values);
                    }
                }
            });
        for_each_spherical_screened_cross_class<Emit>(
            bd, chunks, 5, 1, true, threshold,
            [&](const RysLowPairBinH& dd,
                std::size_t dd_pair_count,
                const RysLowPairBinH& ps,
                std::size_t ps_pair_count,
                bool,
                bool exact_mask) {
                if (dd_pair_count >= ps_pair_count) {
                    if (exact_mask) {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            1, true, true, Emit > (
                                bd, dd, ps, eri_values);
                    } else {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            1, true, false, Emit > (
                                bd, dd, ps, eri_values);
                    }
                } else {
                    if (exact_mask) {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            1, false, true, Emit > (
                                bd, ps, dd, eri_values);
                    } else {
                        accumulate_rys_dd_low_spherical_shell_bin_pair <
                            1, false, false, Emit > (
                                bd, ps, dd, eri_values);
                    }
                }
            });
        // Four-root classes.  These blocks MUST exist together with the
        // knobs: rys_spherical_class_covered_by_specialized_batch declares 52
        // and 53 covered as soon as they are switched on, so the generic route
        // stops emitting them.  Without a screened dispatch here, those two
        // classes are LOST entirely as soon as screening is active -- a hole
        // that does not show up when measuring only with zero thresholds.
        if (rys_dd_pp_shell_batch_enabled()) {
            for_each_spherical_screened_cross_class<Emit>(
                bd, chunks, 5, 2, true, threshold,
                [&](const RysLowPairBinH& dd,
                    std::size_t dd_pair_count,
                    const RysLowPairBinH& pp,
                    std::size_t pp_pair_count,
                    bool,
                    bool exact_mask) {
                    if (dd_pair_count >= pp_pair_count) {
                        if (exact_mask) {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                2, true, true, Emit > (
                                    bd, dd, pp, eri_values);
                        } else {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                2, true, false, Emit > (
                                    bd, dd, pp, eri_values);
                        }
                    } else {
                        if (exact_mask) {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                2, false, true, Emit > (
                                    bd, pp, dd, eri_values);
                        } else {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                2, false, false, Emit > (
                                    bd, pp, dd, eri_values);
                        }
                    }
                });
        }
        if (rys_dd_ds_shell_batch_enabled()) {
            for_each_spherical_screened_cross_class<Emit>(
                bd, chunks, 5, 3, true, threshold,
                [&](const RysLowPairBinH& dd,
                    std::size_t dd_pair_count,
                    const RysLowPairBinH& ds,
                    std::size_t ds_pair_count,
                    bool,
                    bool exact_mask) {
                    if (dd_pair_count >= ds_pair_count) {
                        if (exact_mask) {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                3, true, true, Emit > (
                                    bd, dd, ds, eri_values);
                        } else {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                3, true, false, Emit > (
                                    bd, dd, ds, eri_values);
                        }
                    } else {
                        if (exact_mask) {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                3, false, true, Emit > (
                                    bd, ds, dd, eri_values);
                        } else {
                            accumulate_rys_dd_low_spherical_shell_bin_pair <
                                3, false, false, Emit > (
                                    bd, ds, dd, eri_values);
                        }
                    }
                });
        }
    }

    // Every remaining s-through-f class uses the existing universal shell4
    // dispatcher.  The ownership table is shared with the unscreened route;
    // avoid hard-coding a class count here because two profitability switches
    // move dp|{ds,pp} and dd|{ss,ps} between specialized and generic kernels.
    for (int bra_code = 0; bra_code < 10; ++bra_code) {
        for (int ket_code = 0; ket_code <= bra_code; ++ket_code) {
            if (rys_spherical_class_covered_by_specialized_batch(
                    bra_code, ket_code,
                    use_dp_matrix_shell_batch,
                    use_dd_low_shell_batch)) {
                continue;
            }
            const auto dispatch =
                [&](const RysLowPairBinH& bra,
                    std::size_t,
                    const RysLowPairBinH& ket,
                    std::size_t,
                    bool same_bin,
                    bool exact_mask) {
                    if (g_rys_spherical_screening_diagnostics != nullptr) {
                        ++g_rys_spherical_screening_diagnostics
                              ->generic_accepted_tasks;
                        g_rys_spherical_screening_diagnostics
                            ->generic_accepted_pair_group_quartets +=
                            rys_spherical_bin_pair_work(
                                bra, ket, same_bin);
                    }
                    // exact_mask activates both the per-lane predicate and the
                    // generic two-axis monotone q_floor.  It does not alter the
                    // bin or carry logical ends through this ABI.  Never invert
                    // bra/ket here: their class codes fix La/Lb/Lc/Ld in
                    // shell4.
                    (void)dispatch_rys_generic_spherical_shell4_bin_pair<Emit>(
                        bra_code, ket_code, bd,
                        bra, ket, same_bin, eri_values, exact_mask);
                };
            if (bra_code == ket_code) {
                for_each_spherical_screened_same_class<Emit>(
                    bd, chunks, bra_code, threshold, dispatch);
            } else {
                // Do not install the specialized prefix window here.  Generic
                // shell4 gets the same q_floor effect adaptively: its exact
                // ket-tail break plus the ket[0] bra-tail return prune both
                // sorted axes without another TLS lookup, ABI argument or
                // machine-body instantiation.
                for_each_spherical_screened_cross_class<Emit>(
                    bd, chunks, bra_code, ket_code,
                    false,  // prefix window off; adaptive q_floor is in shell4
                    threshold, dispatch);
            }
        }
    }
}
#endif

template <class Emit>
EriPackedGData
build_eri_packed_for_G_rys_spherical_stream_screened_impl(
    const BasisDataHGP& bd,
    double schwarz_threshold,
    double decay_threshold,
    double primitive_fastschwarz_threshold_squared,
    bool materialize_validation) {
#ifdef _OPENMP
    if (omp_get_max_threads() != 1) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: pure-spherical checksum stream is not parallelized yet; "
            "set OMP_NUM_THREADS=1.");
    }
#endif
    const bool decay_screened = decay_threshold > 0.0;
    const double shell_screen_threshold =
        decay_screened ? decay_threshold : schwarz_threshold;
    const RysSphericalSchwarzBounds* schwarz = nullptr;
    if (!decay_screened) {
        schwarz = &get_rys_spherical_pair_schwarz_bounds(bd);
    }
    const std::vector<double>& pair_group_metric =
        decay_screened ? bd.pair_decay_bounds
                       : schwarz->pair_group_max;
#if defined(__AVX2__) && defined(__FMA__)
    const std::shared_ptr<const RysScreenedSpFullBins>
        sorted_sp_full_bins =
            decay_screened
                ? get_rys_spherical_decay_sp_full_bins(bd)
                : get_rys_spherical_sp_full_bins(bd, *schwarz);
#endif
    const std::shared_ptr<const RysScreenedSphericalChunks> chunks =
        decay_screened
            ? get_rys_decay_screened_spherical_chunks(bd)
            : get_rys_screened_spherical_chunks(bd, *schwarz);
    RysSphericalScreeningDiagnostics diagnostics;
    const bool collect_diagnostics =
        rys_spherical_screening_diagnostics_requested();
    RysSphericalScreeningDiagnosticsGuard diagnostics_guard(
        collect_diagnostics ? &diagnostics : nullptr);

    EriPackedGData eri;
    EriStreamAccumulator stream_accumulator;
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    EriMaterializedSphericalAudit materialized_audit;
    const bool validate_decay_omissions =
        materialize_validation && decay_screened &&
        primitive_fastschwarz_threshold_squared == 0.0;
    if (materialize_validation) {
        const std::size_t expected = spherical_packed_value_count(bd);
        eri.values.assign(expected, 0.0);
        materialized_audit.reset(expected);
        if (validate_decay_omissions) {
            for (std::size_t ab = 0;
                 ab < bd.pair_groups.size(); ++ab) {
                for (std::size_t cd = 0; cd <= ab; ++cd) {
                    if (bd.pair_decay_bounds[ab] *
                            bd.pair_decay_bounds[cd] >=
                        decay_threshold) {
                        continue;
                    }
                    const auto& lhs_refs =
                        bd.packed_spherical_pair_components[ab];
                    const auto& rhs_refs =
                        bd.packed_spherical_pair_components[cd];
                    for (const PackedPairComponentH& lhs_ref :
                         lhs_refs) {
                        for (const PackedPairComponentH& rhs_ref :
                             rhs_refs) {
                            if (ab == cd &&
                                rhs_ref.pair_index >
                                    lhs_ref.pair_index) {
                                continue;
                            }
                            materialized_audit.allow_missing(
                                EriPackedGAccessor::quartet_index(
                                    lhs_ref.pair_index,
                                    rhs_ref.pair_index));
                        }
                    }
                }
            }
        }
    }
    EriMaterializedSphericalAuditGuard materialized_audit_guard(
        materialize_validation ? &materialized_audit : nullptr);
    LowBatchPairComponentsGuard materialized_pair_components(
        materialize_validation, &bd.packed_spherical_pair_components);
#else
    if (materialize_validation) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: materialized pure-spherical validation "
            "requires TLNO_VALIDATE_RYS_SPHERICAL_PACKED.");
    }
#endif
    EriStreamModeGuard stream_mode(
        !materialize_validation, &bd.packed_spherical_pair_components,
        &stream_accumulator);
    const auto& groups = bd.groups;
    const auto& pair_groups = bd.pair_groups;

#if defined(__AVX2__) && defined(__FMA__)
    bool all_shell4_pair_classes = true;
    accumulate_rys_spherical_batches_screened<Emit>(
        bd, *sorted_sp_full_bins, *chunks, pair_group_metric,
        shell_screen_threshold,
        primitive_fastschwarz_threshold_squared,
        eri.values,
        all_shell4_pair_classes);
    const bool need_high_l_quartet_fallback =
        !all_shell4_pair_classes;
#else
    constexpr bool need_high_l_quartet_fallback = true;
#endif

    if (need_high_l_quartet_fallback) {
      for (std::size_t ab = 0; ab < pair_groups.size(); ++ab) {
       for (std::size_t cd = 0; cd <= ab; ++cd) {
        const auto [lg, rg] = pair_groups[ab];
        const auto [lg2, rg2] = pair_groups[cd];
        const auto& gA = groups[static_cast<std::size_t>(lg)];
        const auto& gB = groups[static_cast<std::size_t>(rg)];
        const auto& gC = groups[static_cast<std::size_t>(lg2)];
        const auto& gD = groups[static_cast<std::size_t>(rg2)];
#if defined(__AVX2__) && defined(__FMA__)
        if (rys_spherical_pair_class_code(gA.L, gB.L) >= 0 &&
            rys_spherical_pair_class_code(gC.L, gD.L) >= 0) {
            continue;
        }
#endif
        if (pair_group_metric[ab] *
                pair_group_metric[cd] <
            shell_screen_threshold) {
            continue;
        }
        const auto& lhs_refs =
            bd.packed_spherical_pair_components[ab];
        const auto& rhs_refs =
            bd.packed_spherical_pair_components[cd];
        if (lhs_refs.empty() || rhs_refs.empty()) continue;
        const auto& bras = bd.rys_pair_primitives[ab];
        const auto& kets = bd.rys_pair_primitives[cd];
        const auto& bra_slab = bd.rys_pair_slabs[ab];
        const auto& ket_slab = bd.rys_pair_slabs[cd];
        const FinalSphPairProjector& lhs_projector =
            get_final_sph_pair_projector(gA.L, gB.L);

        if (gA.L <= 1 && gB.L <= 1 &&
            gC.L <= 1 && gD.L <= 1 &&
            try_scatter_pairpair_rys_direct<Emit>(
                gA, gB, gC, gD,
                bras, kets, bra_slab, ket_slab,
                lhs_refs, rhs_refs, ab == cd,
                schwarz != nullptr ? &schwarz->pair_bounds : nullptr,
                schwarz != nullptr ? schwarz_threshold : 0.0,
                eri.values)) {
            continue;
        }

        thread_local std::vector<double> cart_block;
        const bool used_compiled_block =
            try_fused_rys_block_compiled_df_runtime(
                gA.L, gB.L, gC.L, gD.L,
                gA.center, gB.center, gC.center, gD.center,
                bras, kets, cart_block);
        if (!used_compiled_block) {
            // WHAT/WHY/PURPOSE: screened twin of the unscreened high-L
            // escape above; only J changes its root count, while every exact
            // policy retains the original runtime order.
            const int nroots = std::is_same_v <
                    Emit, RysDirectJSpeedProbeEmit > &&
                    !kRysJSpeedProbeExactRoots
                ? 1
                : (gA.L + gB.L + gC.L + gD.L) / 2 + 1;
            fused_rys_block_runtime(
                gA.L, gB.L, gC.L, gD.L, nroots,
                gA.center, gB.center, gC.center, gD.center,
                bras, kets, cart_block);
        }

        const FinalSphPairProjector& rhs_projector =
            get_final_sph_pair_projector(gC.L, gD.L);
        const int rhs_cart = rhs_projector.cart_cols;
        thread_local std::vector<double> mid;
        mid.assign(
            static_cast<std::size_t>(lhs_projector.sph_rows) *
                static_cast<std::size_t>(rhs_cart),
            0.0);
        for (int lhs_row_index = 0;
             lhs_row_index < lhs_projector.sph_rows;
             ++lhs_row_index) {
            double* const dst =
                mid.data() +
                static_cast<std::size_t>(lhs_row_index) *
                    static_cast<std::size_t>(rhs_cart);
            for (const SphProjectorTerm& term :
                 lhs_projector.rows[
                     static_cast<std::size_t>(lhs_row_index)]) {
                const double* const src =
                    cart_block.data() +
                    static_cast<std::size_t>(term.index) *
                        static_cast<std::size_t>(rhs_cart);
                for (int j = 0; j < rhs_cart; ++j) {
                    dst[j] += term.coeff * src[j];
                }
            }
        }

        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            const double lhs_bound =
                schwarz != nullptr
                    ? schwarz->pair_bounds[
                          static_cast<std::size_t>(
                              lhs_ref.pair_index)]
                    : 0.0;
            const double* const lhs_row =
                mid.data() +
                static_cast<std::size_t>(lhs_ref.component_index) *
                    static_cast<std::size_t>(rhs_cart);
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (ab == cd &&
                    rhs_ref.pair_index > lhs_ref.pair_index) {
                    continue;
                }
                if (schwarz != nullptr &&
                    lhs_bound *
                            schwarz->pair_bounds[
                                static_cast<std::size_t>(
                                    rhs_ref.pair_index)] <
                        schwarz_threshold) {
                    continue;
                }
                double value = 0.0;
                for (const SphProjectorTerm& term :
                     rhs_projector.rows[
                         static_cast<std::size_t>(
                             rhs_ref.component_index)]) {
                    value += lhs_row[term.index] * term.coeff;
                }
                rys_emit_pairpair<Emit>(
                    eri.values, lhs_ref.pair_index,
                    rhs_ref.pair_index, value);
            }
        }
       }
      }
    }

    if (collect_diagnostics) {
        const char* const metric_name =
            decay_screened ? "decay" : "schwarz";
        const char* const metric_semantics =
            decay_screened
                ? "heuristic_gaussian_shell_pair_indicator"
                : "contracted_ao_pair_cauchy_schwarz";
        std::uint64_t exact_total = 0;
        std::uint64_t exact_rejected = 0;
        std::uint64_t primitive_quartets_total = 0;
        std::uint64_t primitive_quartets_fast_eligible = 0;
        std::uint64_t primitive_quartets_sp_eligible = 0;
        for (std::size_t ab = 0; ab < pair_groups.size(); ++ab) {
            const auto [lg, rg] = pair_groups[ab];
            const auto& gA = groups[static_cast<std::size_t>(lg)];
            const auto& gB = groups[static_cast<std::size_t>(rg)];
            const bool ab_sp =
                rys_sp_pair_type(gA.L, gB.L) >= 0;
            const std::uint64_t ab_primitives =
                static_cast<std::uint64_t>(
                    bd.rys_pair_primitives[ab].size());
            if (rys_spherical_pair_class_code(gA.L, gB.L) < 0) {
                for (std::size_t cd = 0; cd <= ab; ++cd) {
                    primitive_quartets_total +=
                        ab_primitives *
                        static_cast<std::uint64_t>(
                            bd.rys_pair_primitives[cd].size());
                }
                continue;
            }
            for (std::size_t cd = 0; cd <= ab; ++cd) {
                const auto [lg2, rg2] = pair_groups[cd];
                const auto& gC =
                    groups[static_cast<std::size_t>(lg2)];
                const auto& gD =
                    groups[static_cast<std::size_t>(rg2)];
                const std::uint64_t primitive_work =
                    ab_primitives *
                    static_cast<std::uint64_t>(
                        bd.rys_pair_primitives[cd].size());
                primitive_quartets_total += primitive_work;
                if (rys_spherical_pair_class_code(gC.L, gD.L) < 0) {
                    continue;
                }
                primitive_quartets_fast_eligible += primitive_work;
                if (ab_sp &&
                    rys_sp_pair_type(gC.L, gD.L) >= 0) {
                    primitive_quartets_sp_eligible += primitive_work;
                }
                ++exact_total;
                if (pair_group_metric[ab] *
                        pair_group_metric[cd] <
                    shell_screen_threshold) {
                    ++exact_rejected;
                }
            }
        }
        const std::uint64_t leakage =
            exact_rejected > diagnostics.rejected_pair_group_quartets
                ? exact_rejected -
                      diagnostics.rejected_pair_group_quartets
                : 0;
        const double exact_rejected_fraction =
            exact_total == 0
                ? 0.0
                : static_cast<double>(exact_rejected) /
                      static_cast<double>(exact_total);
        const double scheduled_rejected_fraction =
            diagnostics.scheduled_pair_group_quartets == 0
                ? 0.0
                : static_cast<double>(
                      diagnostics.rejected_pair_group_quartets) /
                      static_cast<double>(
                          diagnostics.scheduled_pair_group_quartets);
        const double leakage_fraction =
            exact_rejected == 0
                ? 0.0
                : static_cast<double>(leakage) /
                      static_cast<double>(exact_rejected);
        const std::uint64_t primitive_quartets_df_eligible =
            primitive_quartets_fast_eligible -
            primitive_quartets_sp_eligible;
        const std::uint64_t primitive_quartets_outside_fast =
            primitive_quartets_total -
            primitive_quartets_fast_eligible;
        const double primitive_quartets_sp_fraction =
            primitive_quartets_total == 0
                ? 0.0
                : static_cast<double>(
                      primitive_quartets_sp_eligible) /
                      static_cast<double>(
                          primitive_quartets_total);
        std::fprintf(
            stderr,
            "SPHSCREEN_DIAG metric=%s metric_semantics=%s "
            "metric_scope=s_through_f_shell4 "
            "low_chunk=%zu high_chunk=%zu exact_total=%llu "
            "exact_rejected=%llu exact_rejected_frac=%.9f "
            "scheduled_total=%llu scheduled_accepted=%llu "
            "scheduled_rejected=%llu scheduled_rejected_frac=%.9f "
            "rejected_leakage=%llu rejected_leakage_frac=%.9f "
            "whole_accept=%llu whole_reject=%llu partial=%llu "
            "chunk_accept=%llu chunk_reject=%llu "
            "generic_accept_tasks=%llu generic_accept_work=%llu "
            "generic_q_floor_stops=%llu "
            "generic_q_floor_pruned_bra_pairs=%llu "
            "df_q_floor_tasks=%llu df_q_floor_candidate=%llu "
            "df_q_floor_visited=%llu df_q_floor_pruned=%llu "
            "sp_masked_tasks=%llu sp_masked_tiles=%llu "
            "sp_masked_tiles_rejected=%llu "
            "sp_masked_lanes=%llu sp_masked_lanes_accepted=%llu "
            "sp_masked_lanes_rejected=%llu "
            "primitive_fast_vectors_tested=%llu "
            "primitive_fast_vectors_rejected=%llu "
            "primitive_fast_lanes_tested=%llu "
            "primitive_fast_lanes_rejected_simd=%llu "
            "primitive_fast_lanes_rejected_strict=%llu\n",
            metric_name,
            metric_semantics,
            rys_screened_spherical_low_l_chunk_pairs(),
            rys_screened_spherical_high_l_chunk_pairs(),
            static_cast<unsigned long long>(exact_total),
            static_cast<unsigned long long>(exact_rejected),
            exact_rejected_fraction,
            static_cast<unsigned long long>(
                diagnostics.scheduled_pair_group_quartets),
            static_cast<unsigned long long>(
                diagnostics.accepted_pair_group_quartets),
            static_cast<unsigned long long>(
                diagnostics.rejected_pair_group_quartets),
            scheduled_rejected_fraction,
            static_cast<unsigned long long>(leakage),
            leakage_fraction,
            static_cast<unsigned long long>(
                diagnostics.whole_bin_pairs_accepted),
            static_cast<unsigned long long>(
                diagnostics.whole_bin_pairs_rejected),
            static_cast<unsigned long long>(
                diagnostics.partial_bin_pairs),
            static_cast<unsigned long long>(
                diagnostics.chunk_pairs_accepted),
            static_cast<unsigned long long>(
                diagnostics.chunk_pairs_rejected),
            static_cast<unsigned long long>(
                diagnostics.generic_accepted_tasks),
            static_cast<unsigned long long>(
                diagnostics.generic_accepted_pair_group_quartets),
            static_cast<unsigned long long>(
                diagnostics.generic_q_floor_tail_stops),
            static_cast<unsigned long long>(
                diagnostics.generic_q_floor_pruned_bra_pairs),
            static_cast<unsigned long long>(
                diagnostics.df_q_floor_tasks),
            static_cast<unsigned long long>(
                diagnostics.df_q_floor_candidate_pair_group_quartets),
            static_cast<unsigned long long>(
                diagnostics.df_q_floor_visited_pair_group_quartets),
            static_cast<unsigned long long>(
                diagnostics.df_q_floor_pruned_pair_group_quartets),
            static_cast<unsigned long long>(
                diagnostics.sp_masked_tasks),
            static_cast<unsigned long long>(
                diagnostics.sp_masked_tiles),
            static_cast<unsigned long long>(
                diagnostics.sp_masked_tiles_rejected),
            static_cast<unsigned long long>(
                diagnostics.sp_masked_shell_lanes),
            static_cast<unsigned long long>(
                diagnostics.sp_masked_shell_lanes_accepted),
            static_cast<unsigned long long>(
                diagnostics.sp_masked_shell_lanes_rejected),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_vectors_tested),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_vectors_rejected),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_lanes_tested),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_lanes_rejected_simd),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_lanes_rejected_strict));
        static constexpr const char* kPrimitiveClassNames[6] = {
            "ssss", "ssps", "psps", "sspp", "pspp", "pppp"};
        for (std::size_t primitive_class = 0;
             primitive_class < 6; ++primitive_class) {
            std::fprintf(
                stderr,
                "SPHFAST_CLASS class=%s vectors_tested=%llu "
                "vectors_rejected=%llu lanes_tested=%llu "
                "lanes_rejected_simd=%llu "
                "lanes_rejected_strict=%llu\n",
                kPrimitiveClassNames[primitive_class],
                static_cast<unsigned long long>(
                    diagnostics.primitive_fast_class_vectors_tested[
                        primitive_class]),
                static_cast<unsigned long long>(
                    diagnostics.primitive_fast_class_vectors_rejected[
                        primitive_class]),
                static_cast<unsigned long long>(
                    diagnostics.primitive_fast_class_lanes_tested[
                        primitive_class]),
                static_cast<unsigned long long>(
                    diagnostics.primitive_fast_class_lanes_rejected_simd[
                        primitive_class]),
                static_cast<unsigned long long>(
                diagnostics.primitive_fast_class_lanes_rejected_strict[
                        primitive_class]));
        }
        std::fprintf(
            stderr,
            "SPHFAST_DF scope=df_specialized_and_generic_shell4 "
            "vectors_tested=%llu vectors_rejected=%llu "
            "lanes_tested=%llu lanes_rejected_simd=%llu "
            "lanes_rejected_strict=%llu\n",
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_df_vectors_tested),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_df_vectors_rejected),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_df_lanes_tested),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_df_lanes_rejected_simd),
            static_cast<unsigned long long>(
                diagnostics.primitive_fast_df_lanes_rejected_strict));
        std::fprintf(
            stderr,
            "SPHFAST_SCOPE scope=s_through_f_shell4_pair_classes "
            "primitive_quartets_total=%llu "
            "primitive_quartets_fast_eligible=%llu "
            "primitive_quartets_sp_eligible=%llu "
            "primitive_quartets_df_eligible=%llu "
            "primitive_quartets_outside_fast=%llu "
            "primitive_quartets_sp_fraction=%.9f\n",
            static_cast<unsigned long long>(
                primitive_quartets_total),
            static_cast<unsigned long long>(
                primitive_quartets_fast_eligible),
            static_cast<unsigned long long>(
                primitive_quartets_sp_eligible),
            static_cast<unsigned long long>(
                primitive_quartets_df_eligible),
            static_cast<unsigned long long>(
                primitive_quartets_outside_fast),
            primitive_quartets_sp_fraction);
    }

#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    if (!materialize_validation) {
        const EriStreamFP& fp = stream_accumulator.fingerprint;
        if (fp.canonical_count) std::fprintf(stderr,
            "STREAMFP checksum=%.17Lg absolute_sum=%.17Lg square_sum=%.17Lg m1=%.17Lg m2=%.17Lg m3=%.17Lg max_abs=%.17Lg count=%llu\n",
            fp.checksum, fp.absolute_sum, fp.square_sum,
            fp.indexed_moment_1, fp.indexed_moment_2,
            fp.indexed_moment_3, fp.max_abs,
            fp.canonical_count);
    }
#endif
#ifdef TLNO_VALIDATE_RYS_SPHERICAL_PACKED
    if (materialize_validation) {
        materialized_audit.validate(
            false, validate_decay_omissions);
        std::fprintf(
            stderr,
            "MATERIALIZED_SPHERICAL_AUDIT expected=%zu written=%zu "
            "implicit_zeros=%zu require_complete=0\n",
            materialized_audit.expected_count,
            materialized_audit.unique_writes,
            materialized_audit.expected_count -
                materialized_audit.unique_writes);
        if (decay_screened) {
            std::fprintf(
                stderr,
                "MATERIALIZED_DECAY_AUDIT "
                "semantics=heuristic_nonrigorous "
                "omission_predicate_checked=%d\n",
                validate_decay_omissions ? 1 : 0);
        }
        return eri;
    }
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    // The moments are exposed next to the checksum ONLY in fingerprint builds:
    // they are the reference against which the J/K policy is compared.  In
    // production the vector keeps one element and the contract is unchanged.
    eri.values.assign(
        {stream_accumulator.checksum,
         static_cast<double>(stream_accumulator.fingerprint.indexed_moment_1),
         static_cast<double>(stream_accumulator.fingerprint.indexed_moment_2),
         static_cast<double>(stream_accumulator.fingerprint.indexed_moment_3),
         static_cast<double>(stream_accumulator.fingerprint.canonical_count)});
#else
    eri.values.assign(1, stream_accumulator.checksum);
#endif
    return eri;
}

#include "native_quartets.ipp"

template <class Emit>
EriPackedGData
build_eri_packed_for_G_rys_spherical_stream_impl(
    const BasisDataHGP& bd,
    const EriPackedBuildOptions& options,
    bool materialize_validation = false) {
    if (!tlno::core::math::is_finite_binary64(options.schwarz_threshold) ||
        options.schwarz_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: schwarz_threshold must be finite and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(options.decay_threshold) ||
        options.decay_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: decay_threshold must be finite and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(
            options.primitive_fastschwarz_threshold) ||
        options.primitive_fastschwarz_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: primitive_fastschwarz_threshold must be "
            "finite and >= 0.");
    }
    // Native W8/SSE2 evaluate Schwarz and decay independently in every packet.
    // The legacy specialized W4 scheduler still supports one shell metric.
    if (!portable_kernel && options.schwarz_threshold != 0.0 &&
        options.decay_threshold != 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: pure-spherical Schwarz+decay screening is "
            "not part of the independent decay checkpoint; select exactly "
            "one shell-level metric.");
    }
    if constexpr (portable_kernel) {
        return build_native_quartet_stream<Emit>(bd, options, materialize_validation);
    } else {
    if (options.schwarz_threshold == 0.0 &&
        options.decay_threshold == 0.0 &&
        options.primitive_fastschwarz_threshold == 0.0) {
        // Preserve the established endpoint as a distinct, untouched branch:
        // no bounds, sorting, chunks, or screened dispatch are constructed.
        return build_eri_packed_for_G_rys_spherical_stream_unscreened_impl<Emit>(
            bd, materialize_validation);
    }
    const double primitive_fastschwarz_threshold_squared =
        options.primitive_fastschwarz_threshold *
        options.primitive_fastschwarz_threshold;
    return build_eri_packed_for_G_rys_spherical_stream_screened_impl<Emit>(
        bd, options.schwarz_threshold, options.decay_threshold,
        primitive_fastschwarz_threshold_squared,
        materialize_validation);
    }
}

#ifdef TLNO_BUILD_RYS_VALIDATION
#include "../../../../validation/rys_validation_internal.inc"
#endif

template <class Emit>
EriPackedGData
build_eri_packed_for_G_rys_impl(
    const std::shared_ptr<const BasisDataHGP>& bd_owner,
    const EriPackedBuildOptions& options) {
    if (!bd_owner) {
        throw std::runtime_error("eri_ao_tensor_hgp: null packed ERI plan.");
    }
    // The public engine has one production route. Keeping this decision out of
    // the options prevents callers from falling back to Cartesian
    // materialization or rebuilding work that belongs in the cached plan.
    return build_eri_packed_for_G_rys_spherical_stream_impl<Emit>(
        *bd_owner, options);
}

void compress_sph_projector_terms(const std::vector<double>& dense,
                                  int n_rows,
                                  int n_cols,
                                  std::vector<std::vector<SphProjectorTerm>>& terms) {
    terms.assign(static_cast<std::size_t>(n_rows), {});
    for (int row = 0; row < n_rows; ++row) {
        auto& out = terms[static_cast<std::size_t>(row)];
        for (int col = 0; col < n_cols; ++col) {
            const double v = dense[static_cast<std::size_t>(row) *
                                       static_cast<std::size_t>(n_cols) +
                                   static_cast<std::size_t>(col)];
            if (v != 0.0) out.push_back({col, v});
        }
    }
}


void build_pair_sph_projector_terms_runtime(
    const ContractedShellGroupH& gL,
    const ContractedShellGroupH& gR,
    const std::array<double, 3>& LR,
    bool bra_side,
    std::vector<std::vector<SphProjectorTerm>>& terms) {
    const int LL = gL.L;
    const int LRang = gR.L;
    const int Lsum = LL + LRang;
    const int offset = bra_side ? offset_for_L(LL) : 0;
    const int ncols = bra_side ? (n_cart_cum(Lsum) - offset) : n_cart_cum(Lsum);
    const int Nsl = n_sph(LL);
    const int Nsr = n_sph(LRang);
    const int left_offset = offset_for_L(LL);
    const int right_offset = offset_for_L(LRang);
    const SphC2SBlock& c2sL = kSphC2SBlocks[LL];
    const SphC2SBlock& c2sR = kSphC2SBlocks[LRang];

    std::vector<double> dense(static_cast<std::size_t>(Nsl) *
                                  static_cast<std::size_t>(Nsr) *
                                  static_cast<std::size_t>(ncols),
                              0.0);

    for (int il = 0; il < c2sL.count; ++il) {
        const SphC2STriple& el = c2sL.data[il];
        const auto& a = g_cart.at(left_offset + el.cart);
        const double cl = el.coeff * gL.n_ang[static_cast<std::size_t>(el.cart)];
        for (int ir = 0; ir < c2sR.count; ++ir) {
            const SphC2STriple& er = c2sR.data[ir];
            const auto& b = g_cart.at(right_offset + er.cart);
            const int pair = el.sph * Nsr + er.sph;
            const double cab = cl * er.coeff * gR.n_ang[static_cast<std::size_t>(er.cart)];

            for (int ux = 0; ux <= b[0]; ++ux) {
                const int cx = rys_small_binom_any(b[0], ux);
                const double px = rys_small_pow_any(LR[0], b[0] - ux);
                for (int uy = 0; uy <= b[1]; ++uy) {
                    const int cxy = cx * rys_small_binom_any(b[1], uy);
                    const double pxy = px * rys_small_pow_any(LR[1], b[1] - uy);
                    for (int uz = 0; uz <= b[2]; ++uz) {
                        const int cxyz = cxy * rys_small_binom_any(b[2], uz);
                        const double pxyz = pxy * rys_small_pow_any(LR[2], b[2] - uz);
                        int row = cart_flat_idx(a[0] + ux, a[1] + uy, a[2] + uz);
                        if (bra_side) row -= offset;
                        dense[static_cast<std::size_t>(pair) *
                                  static_cast<std::size_t>(ncols) +
                              static_cast<std::size_t>(row)] +=
                            cab * static_cast<double>(cxyz) * pxyz;
                    }
                }
            }
        }
    }

    compress_sph_projector_terms(dense, Nsl * Nsr, ncols, terms);
}

#include "integrals/two_electron/fock/detail/k_pspp_receiver.ipp"

}  // anonymous namespace

struct EriPackedPlanBackend {
    explicit EriPackedPlanBackend(
        std::shared_ptr<const BasisDataHGP> basis_data_in) noexcept
        : basis_data(std::move(basis_data_in)) {}

    std::shared_ptr<const BasisDataHGP> basis_data;
};

struct EriPackedPlanBackendAccess {
    static const std::shared_ptr<const EriPackedPlanBackend>& get(
        const EriPackedPlan& plan) noexcept {
        return plan.backend_;
    }

    static void set(
        EriPackedPlan& plan,
        std::shared_ptr<const EriPackedPlanBackend> backend) noexcept {
        plan.backend_ = std::move(backend);
    }
};

namespace {

std::shared_ptr<const BasisDataHGP> packed_plan_basis_data(
    const EriPackedPlan& plan) noexcept {
    const auto& backend = EriPackedPlanBackendAccess::get(plan);
    return backend ? backend->basis_data : nullptr;
}

}  // anonymous namespace

// ── Public API ----------------------------------------------------------------

EriPackedPlan build_eri_packed_plan_for_G_rys(const std::vector<tlno::core::basis::Shell>& basis,
                                             const EriPackedBuildOptions& options) {
    if (!tlno::core::math::is_finite_binary64(options.schwarz_threshold) ||
        options.schwarz_threshold < 0.0) {
        throw std::runtime_error("eri_ao_tensor_hgp: schwarz_threshold must be finite and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(options.decay_threshold) ||
        options.decay_threshold < 0.0) {
        throw std::runtime_error("eri_ao_tensor_hgp: decay_threshold must be finite and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(
            options.primitive_fastschwarz_threshold) ||
        options.primitive_fastschwarz_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: primitive_fastschwarz_threshold must be "
            "finite and >= 0.");
    }
    auto bd = get_basis_data_h_shared(basis);
    if constexpr (portable_kernel) {
        (void)get_native_quartet_schedule(*bd);
        if (options.schwarz_threshold > 0 || options.density_threshold > 0)
            (void)get_rys_spherical_pair_schwarz_bounds(*bd);
    }
    EriPackedPlan plan;
    plan.profile = build_packed_plan_profile_h(*bd);
    EriPackedPlanBackendAccess::set(
        plan, std::make_shared<const EriPackedPlanBackend>(std::move(bd)));
    return plan;
}

EriPackedGData build_eri_packed_for_G_rys(const EriPackedPlan& plan,
                                          const EriPackedBuildOptions& options) {
    auto bd = packed_plan_basis_data(plan);
    return build_eri_packed_for_G_rys_impl<RysPackedTensorEmit>(bd, options);
}

#include "integrals/two_electron/fock/detail/direct_build.ipp"

#ifdef TLNO_BUILD_RYS_VALIDATION
#include "../../../../validation/rys_validation_endpoints.inc"
#endif

}  // namespace tlno::integrals::twoe
