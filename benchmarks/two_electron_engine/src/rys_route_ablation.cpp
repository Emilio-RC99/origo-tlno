// Validation/benchmark-only causal surfaces for the retained Rys routes.
//
// This translation unit deliberately includes the authoritative implementation
// so it can select internal route twins without adding switches to the public
// ABI or to the production dispatcher.  It is built only by
// scripts/build_route_ablation_bench.sh into a disposable/result directory.
// Nothing in this file is linked into a production application.

#define TLNO_BUILD_RYS_AUDIT 1
#define TLNO_RYS_ABLATION_ALL55 1

#include "integrals/two_electron/rys/rys_engine.ipp"
#include "core/math/binary64.hpp"
#include "core/shell/shell_contract.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tlno::integrals::twoe::route_ablation {
namespace {

#if !defined(__AVX2__) || !defined(__FMA__)
#error "rys_route_ablation requires AVX2 and FMA"
#endif

enum class Route {
    Production,
    GenericAuto,
    GenericEarly,
    GenericCart,
};

inline constexpr int kNativeLanes = tlno::core::simd::lanes;

struct Mode {
    Route route = Route::Production;
    int packed_quartets = 0;
    int scalar_repeats = 0;
    bool roots_batch4 = false;
    bool roots_scalar4 = false;
    std::string label;
};

std::vector<std::array<int, 3>> cartesian_components(int L) {
    std::vector<std::array<int, 3>> out;
    for (int nx = L; nx >= 0; --nx) {
        for (int ny = L - nx; ny >= 0; --ny) {
            out.push_back({nx, ny, L - nx - ny});
        }
    }
    return out;
}

std::array<int, 4> parse_class(const std::string& text) {
    std::string cleaned;
    cleaned.reserve(text.size() * 2u);
    for (const char c : text) {
        if (c == 's' || c == 'S') cleaned += "0 ";
        else if (c == 'p' || c == 'P') cleaned += "1 ";
        else if (c == 'd' || c == 'D') cleaned += "2 ";
        else if (c == 'f' || c == 'F') cleaned += "3 ";
        else if (c >= '0' && c <= '3') {
            cleaned.push_back(c);
            cleaned.push_back(' ');
        } else if (c == ',' || c == '|' || c == ':' || c == '/' ||
                   c == ' ' || c == '\t') {
            cleaned.push_back(' ');
        } else {
            throw std::runtime_error("invalid character in --class");
        }
    }
    std::istringstream input(cleaned);
    std::array<int, 4> angular{};
    for (int i = 0; i < 4; ++i) {
        if (!(input >> angular[static_cast<std::size_t>(i)])) {
            throw std::runtime_error(
                "--class must contain four s-through-f angular momenta");
        }
    }
    int extra = 0;
    if (input >> extra) {
        throw std::runtime_error("--class contains more than four momenta");
    }
    return angular;
}

char angular_symbol(int L) {
    static constexpr char symbols[] = {'s', 'p', 'd', 'f'};
    return (L >= 0 && L <= 3) ? symbols[L] : '?';
}

std::string class_label(const std::array<int, 4>& angular) {
    std::string out;
    out.push_back(angular_symbol(angular[0]));
    out.push_back(angular_symbol(angular[1]));
    out.push_back('|');
    out.push_back(angular_symbol(angular[2]));
    out.push_back(angular_symbol(angular[3]));
    return out;
}

int pair_code(int left, int right) {
    const int high = std::max(left, right);
    const int low = std::min(left, right);
    return high * (high + 1) / 2 + low;
}

std::vector<tlno::core::basis::Shell> make_basis(
    const std::array<int, 4>& angular,
    double exponent,
    int nprim,
    double scale) {
    const std::array<std::array<double, 3>, 4> centers = {{
        {{0.00, 0.00, 0.00}},
        {{1.37 * scale, 0.19 * scale, 0.11 * scale}},
        {{0.43 * scale, 1.21 * scale, 0.29 * scale}},
        {{0.23 * scale, 0.31 * scale, 1.51 * scale}},
    }};
    std::vector<tlno::core::basis::Shell> basis;
    for (int group = 0; group < 4; ++group) {
        const int L = angular[static_cast<std::size_t>(group)];
        for (const auto& component : cartesian_components(L)) {
            tlno::core::basis::RawShell raw;
            raw.label = std::string(1, angular_symbol(L));
            raw.ang = component;
            for (int primitive = nprim; primitive >= 1; --primitive) {
                raw.exponents.push_back(
                    exponent * static_cast<double>(primitive));
                raw.coefficients.push_back(1.0);
            }
            basis.push_back(tlno::core::shell::make_shell(
                raw, centers[static_cast<std::size_t>(group)]));
        }
    }
    return basis;
}

std::size_t find_pair_group(
    const BasisDataHGP& basis, int first, int second) {
    for (std::size_t index = 0; index < basis.pair_groups.size(); ++index) {
        const auto pair = basis.pair_groups[index];
        if ((pair.first == first && pair.second == second) ||
            (pair.first == second && pair.second == first)) {
            return index;
        }
    }
    throw std::runtime_error("requested shell pair was not found");
}

RysLowPairBinH replicate_pair_bin(
    const BasisDataHGP& basis,
    std::size_t pair_group,
    int count) {
    if (count <= 0) throw std::runtime_error("pair-bin count must be positive");
    std::vector<std::uint32_t> ids(
        static_cast<std::size_t>(count),
        static_cast<std::uint32_t>(pair_group));
    RysLowPairBinH result;
    fill_rys_low_chunk(
        basis, ids.data(), ids.size(),
        static_cast<int>(basis.rys_pair_slabs.at(pair_group).size()), result);
    return result;
}

bool production_specialized(
    int bra_code, int ket_code, std::uint64_t pair_products) {
    const bool use_dp_matrix = pair_products >= 256u;
    const bool use_dd_low = pair_products >= 256u;
    return rys_spherical_class_covered_by_specialized_batch(
        bra_code, ket_code, use_dp_matrix, use_dd_low);
}

template <class Emit>
void dispatch_specialized(
    int key,
    const BasisDataHGP& basis,
    const RysLowPairBinH& bra,
    const RysLowPairBinH& ket,
    std::vector<double>& scratch) {
    switch (key) {
        case 0:
            accumulate_rys_ssss_shell_bin_pair<false, false, Emit>(
                basis, bra, ket, false, scratch); return;
        case 10:
            accumulate_rys_ssps_shell_bin_pair<true, false, false, Emit>(
                basis, bra, ket, scratch); return;
        case 11:
            accumulate_rys_psps_shell_bin_pair<false, false, Emit>(
                basis, bra, ket, false, scratch); return;
        case 20:
            accumulate_rys_sspp_shell_bin_pair<true, false, false, Emit>(
                basis, bra, ket, scratch); return;
        case 21:
            accumulate_rys_pspp_shell_bin_pair<true, false, false, Emit>(
                basis, bra, ket, scratch); return;
        case 22:
            accumulate_rys_pppp_shell_bin_pair<false, false, Emit>(
                basis, bra, ket, false, scratch); return;
        case 30:
            accumulate_rys_ssds_spherical_shell_bin_pair<true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 31:
            accumulate_rys_dsps_spherical_shell_bin_pair<true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 32:
            accumulate_rys_dspp_spherical_shell_bin_pair<true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 33:
            accumulate_rys_dsds_spherical_shell_bin_pair<false, Emit>(
                basis, bra, ket, false, scratch); return;
        case 40:
            accumulate_rys_dp_low_spherical_shell_bin_pair<0, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 41:
            accumulate_rys_dp_low_spherical_shell_bin_pair<1, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 42:
            accumulate_rys_dp_matrix_spherical_shell_bin_pair<true, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 43:
            accumulate_rys_dp_matrix_spherical_shell_bin_pair<false, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 50:
            accumulate_rys_dd_low_spherical_shell_bin_pair<0, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 51:
            accumulate_rys_dd_low_spherical_shell_bin_pair<1, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 52:
            accumulate_rys_dd_low_spherical_shell_bin_pair<2, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 53:
            accumulate_rys_dd_low_spherical_shell_bin_pair<3, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 60:
            accumulate_rys_fs_low_spherical_shell_bin_pair<0, true, false, Emit>(
                basis, bra, ket, scratch); return;
        case 61:
            accumulate_rys_fs_low_spherical_shell_bin_pair<1, true, false, Emit>(
                basis, bra, ket, scratch); return;
        default:
            throw std::runtime_error(
                "benchmark selected a nonexistent specialized route");
    }
}

// Force either twin of the same shell4 body after the benchmark has checked
// the production dispatch predicate.  This keeps the class switch and timed
// scheduler symmetric: GenericEarly calls RadialKernel=true directly, while
// GenericCart calls false.  The five classes for which the radial body does
// not exist are rejected at compile time and use the auto/cart null control in
// the driver.  The changed intermediate/projector placement is an algorithmic
// route toggle, not one removed instruction.
template <bool RadialKernel, class Emit>
bool dispatch_generic_forced(
    int bra_code,
    int ket_code,
    const BasisDataHGP& basis,
    const RysLowPairBinH& bra,
    const RysLowPairBinH& ket,
    std::vector<double>& scratch) {
#define TLNO_ABLATION_FORCED_CASE(BC, KC, LA, LB, LC, LD)                    \
    case 10 * (BC) + (KC): {                                                 \
        constexpr bool radial_class =                                        \
            rys_generic_shell4_radial_class((BC), (KC));                     \
        if constexpr (RadialKernel && !radial_class) {                        \
            return false;                                                    \
        } else {                                                             \
            accumulate_rys_generic_spherical_shell4_bin_pair<                \
                LA, LB, LC, LD, RadialKernel, Emit>(                          \
                basis, bra, ket, false, scratch, false);                     \
            return true;                                                     \
        }                                                                    \
    }
    switch (10 * bra_code + ket_code) {
        TLNO_ABLATION_FORCED_CASE(0, 0, 0, 0, 0, 0);
        TLNO_ABLATION_FORCED_CASE(1, 0, 1, 0, 0, 0);
        TLNO_ABLATION_FORCED_CASE(1, 1, 1, 0, 1, 0);
        TLNO_ABLATION_FORCED_CASE(2, 0, 1, 1, 0, 0);
        TLNO_ABLATION_FORCED_CASE(2, 1, 1, 1, 1, 0);
        TLNO_ABLATION_FORCED_CASE(2, 2, 1, 1, 1, 1);
        TLNO_ABLATION_FORCED_CASE(3, 0, 2, 0, 0, 0);
        TLNO_ABLATION_FORCED_CASE(3, 1, 2, 0, 1, 0);
        TLNO_ABLATION_FORCED_CASE(3, 2, 2, 0, 1, 1);
        TLNO_ABLATION_FORCED_CASE(3, 3, 2, 0, 2, 0);
        TLNO_ABLATION_FORCED_CASE(4, 0, 2, 1, 0, 0);
        TLNO_ABLATION_FORCED_CASE(4, 1, 2, 1, 1, 0);
        TLNO_ABLATION_FORCED_CASE(4, 2, 2, 1, 1, 1);
        TLNO_ABLATION_FORCED_CASE(4, 3, 2, 1, 2, 0);
        TLNO_ABLATION_FORCED_CASE(4, 4, 2, 1, 2, 1);
        TLNO_ABLATION_FORCED_CASE(5, 0, 2, 2, 0, 0);
        TLNO_ABLATION_FORCED_CASE(5, 1, 2, 2, 1, 0);
        TLNO_ABLATION_FORCED_CASE(5, 2, 2, 2, 1, 1);
        TLNO_ABLATION_FORCED_CASE(5, 3, 2, 2, 2, 0);
        TLNO_ABLATION_FORCED_CASE(5, 4, 2, 2, 2, 1);
        TLNO_ABLATION_FORCED_CASE(5, 5, 2, 2, 2, 2);
        TLNO_ABLATION_FORCED_CASE(6, 0, 3, 0, 0, 0);
        TLNO_ABLATION_FORCED_CASE(6, 1, 3, 0, 1, 0);
        TLNO_ABLATION_FORCED_CASE(6, 2, 3, 0, 1, 1);
        TLNO_ABLATION_FORCED_CASE(6, 3, 3, 0, 2, 0);
        TLNO_ABLATION_FORCED_CASE(6, 4, 3, 0, 2, 1);
        TLNO_ABLATION_FORCED_CASE(6, 5, 3, 0, 2, 2);
        TLNO_ABLATION_FORCED_CASE(6, 6, 3, 0, 3, 0);
        TLNO_ABLATION_FORCED_CASE(7, 0, 3, 1, 0, 0);
        TLNO_ABLATION_FORCED_CASE(7, 1, 3, 1, 1, 0);
        TLNO_ABLATION_FORCED_CASE(7, 2, 3, 1, 1, 1);
        TLNO_ABLATION_FORCED_CASE(7, 3, 3, 1, 2, 0);
        TLNO_ABLATION_FORCED_CASE(7, 4, 3, 1, 2, 1);
        TLNO_ABLATION_FORCED_CASE(7, 5, 3, 1, 2, 2);
        TLNO_ABLATION_FORCED_CASE(7, 6, 3, 1, 3, 0);
        TLNO_ABLATION_FORCED_CASE(7, 7, 3, 1, 3, 1);
        TLNO_ABLATION_FORCED_CASE(8, 0, 3, 2, 0, 0);
        TLNO_ABLATION_FORCED_CASE(8, 1, 3, 2, 1, 0);
        TLNO_ABLATION_FORCED_CASE(8, 2, 3, 2, 1, 1);
        TLNO_ABLATION_FORCED_CASE(8, 3, 3, 2, 2, 0);
        TLNO_ABLATION_FORCED_CASE(8, 4, 3, 2, 2, 1);
        TLNO_ABLATION_FORCED_CASE(8, 5, 3, 2, 2, 2);
        TLNO_ABLATION_FORCED_CASE(8, 6, 3, 2, 3, 0);
        TLNO_ABLATION_FORCED_CASE(8, 7, 3, 2, 3, 1);
        TLNO_ABLATION_FORCED_CASE(8, 8, 3, 2, 3, 2);
        TLNO_ABLATION_FORCED_CASE(9, 0, 3, 3, 0, 0);
        TLNO_ABLATION_FORCED_CASE(9, 1, 3, 3, 1, 0);
        TLNO_ABLATION_FORCED_CASE(9, 2, 3, 3, 1, 1);
        TLNO_ABLATION_FORCED_CASE(9, 3, 3, 3, 2, 0);
        TLNO_ABLATION_FORCED_CASE(9, 4, 3, 3, 2, 1);
        TLNO_ABLATION_FORCED_CASE(9, 5, 3, 3, 2, 2);
        TLNO_ABLATION_FORCED_CASE(9, 6, 3, 3, 3, 0);
        TLNO_ABLATION_FORCED_CASE(9, 7, 3, 3, 3, 1);
        TLNO_ABLATION_FORCED_CASE(9, 8, 3, 3, 3, 2);
        TLNO_ABLATION_FORCED_CASE(9, 9, 3, 3, 3, 3);
        default: break;
    }
#undef TLNO_ABLATION_FORCED_CASE
    return false;
}

struct Fingerprint {
    std::uint64_t count = 0;
    long double checksum = 0.0L;
    long double absolute_sum = 0.0L;
    long double square_sum = 0.0L;
    long double indexed_1 = 0.0L;
    long double indexed_2 = 0.0L;
    long double indexed_3 = 0.0L;
    long double max_abs = 0.0L;
};

std::uint64_t mix64(std::uint64_t value) noexcept {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

long double index_weight(std::uint64_t index, std::uint64_t seed) noexcept {
    const std::uint64_t mixed = mix64(index ^ seed);
    const long long centered =
        static_cast<long long>((mixed >> 40) & 0xffffffull) - 0x800000ll;
    return static_cast<long double>(centered) / 8388608.0L;
}

thread_local Fingerprint* g_fingerprint = nullptr;

struct FingerprintEmit {
    static inline void emit(
        std::vector<double>&,
        std::uint32_t lhs_pair,
        std::uint32_t rhs_pair,
        double value) noexcept {
        Fingerprint& fp = *g_fingerprint;
        const std::uint64_t index = static_cast<std::uint64_t>(
            EriPackedGAccessor::quartet_index(lhs_pair, rhs_pair));
        const long double x = static_cast<long double>(value);
        const long double ax = x < 0.0L ? -x : x;
        ++fp.count;
        fp.checksum += x;
        fp.absolute_sum += ax;
        fp.square_sum += x * x;
        fp.indexed_1 += x * index_weight(index, 0x243f6a8885a308d3ull);
        fp.indexed_2 += x * index_weight(index, 0x13198a2e03707344ull);
        fp.indexed_3 += x * index_weight(index, 0xa4093822299f31d0ull);
        fp.max_abs = std::max(fp.max_abs, ax);
    }
};

class FingerprintGuard {
public:
    explicit FingerprintGuard(Fingerprint& fingerprint)
        : previous_(g_fingerprint) {
        g_fingerprint = &fingerprint;
    }
    ~FingerprintGuard() { g_fingerprint = previous_; }
    FingerprintGuard(const FingerprintGuard&) = delete;
    FingerprintGuard& operator=(const FingerprintGuard&) = delete;
private:
    Fingerprint* previous_;
};

struct RouteWorkload {
    std::shared_ptr<const BasisDataHGP> basis_owner;
    RysLowPairBinH bra;
    RysLowPairBinH ket;
    int bra_code = 0;
    int ket_code = 0;
    int key = 0;
    int route_calls = 1;
    std::uint64_t quartets = 0;
    std::uint64_t spherical_per_quartet = 0;
    std::uint64_t primitive_quartets_per_quartet = 0;
    bool specialized = false;
    bool radial_eligible = false;
    bool generic_bra_uniform = false;
    bool generic_radial_dispatch_predicate = false;
    bool generic_radial_selected = false;
    std::vector<double> scratch;
};

RouteWorkload make_workload(
    const std::vector<tlno::core::basis::Shell>& shells,
    const std::array<int, 4>& angular,
    const Mode& mode,
    int batch) {
    RouteWorkload work;
    work.basis_owner = get_basis_data_h_shared(shells);
    const BasisDataHGP& basis = *work.basis_owner;
    if (basis.groups.size() != angular.size()) {
        throw std::runtime_error(
            "synthetic basis did not form exactly four contracted shells");
    }
    for (std::size_t group = 0; group < angular.size(); ++group) {
        if (basis.groups[group].L != angular[group]) {
            throw std::runtime_error(
                "synthetic contracted-shell order differs from --class");
        }
    }
    const std::size_t bra_group = find_pair_group(basis, 0, 1);
    const std::size_t ket_group = find_pair_group(basis, 2, 3);
    if (bra_group == ket_group) {
        throw std::runtime_error(
            "route ablation requires distinct bra and ket shell pairs");
    }
    int bra_count = batch;
    int ket_count = batch;
    if (mode.packed_quartets > 0) {
        bra_count = mode.packed_quartets;
        ket_count = 1;
    } else if (mode.scalar_repeats > 0) {
        bra_count = 1;
        ket_count = 1;
        work.route_calls = mode.scalar_repeats;
    }
    work.bra = replicate_pair_bin(basis, bra_group, bra_count);
    work.ket = replicate_pair_bin(basis, ket_group, ket_count);
    work.bra_code = pair_code(angular[0], angular[1]);
    work.ket_code = pair_code(angular[2], angular[3]);
    if (work.bra_code < work.ket_code) {
        throw std::runtime_error("class is not in canonical pair order");
    }
    work.key = 10 * work.bra_code + work.ket_code;
    const std::uint64_t pair_products =
        static_cast<std::uint64_t>(bra_count) *
        static_cast<std::uint64_t>(ket_count);
    work.specialized =
        mode.route == Route::Production &&
        production_specialized(
            work.bra_code, work.ket_code, pair_products);
    work.radial_eligible =
        rys_generic_shell4_radial_class(work.bra_code, work.ket_code);
    // The generic dispatch predicate is eligible class AND homogeneous bra
    // bin.  Re-evaluate it with the same core helper, then also record whether
    // this requested mode actually enters that generic radial instantiation.
    // This avoids inferring route selection from class eligibility alone.
    work.generic_bra_uniform =
        rys_generic_shell4_uniform_bra_bin(work.bra);
    work.generic_radial_dispatch_predicate =
        work.radial_eligible && work.generic_bra_uniform;
    work.generic_radial_selected =
        !work.specialized && mode.route != Route::GenericCart &&
        work.generic_radial_dispatch_predicate;
    if (mode.route == Route::GenericEarly &&
        !work.generic_radial_selected) {
        throw std::runtime_error(
            "generic-early did not select a homogeneous radial shell4 bin");
    }
    work.quartets = pair_products *
                     static_cast<std::uint64_t>(work.route_calls);
    work.spherical_per_quartet = 1u;
    for (const int L : angular) {
        work.spherical_per_quartet *=
            static_cast<std::uint64_t>(2 * L + 1);
    }
    work.primitive_quartets_per_quartet =
        static_cast<std::uint64_t>(basis.rys_pair_slabs[bra_group].size()) *
        static_cast<std::uint64_t>(basis.rys_pair_slabs[ket_group].size());
    return work;
}

template <class Emit>
void execute_route(Route route, RouteWorkload& work) {
    const BasisDataHGP& basis = *work.basis_owner;
    for (int call = 0; call < work.route_calls; ++call) {
        if (route == Route::Production && work.specialized) {
            dispatch_specialized<Emit>(
                work.key, basis, work.bra, work.ket, work.scratch);
        } else if (route == Route::GenericEarly) {
            if (!dispatch_generic_forced<true, Emit>(
                    work.bra_code, work.ket_code, basis,
                    work.bra, work.ket, work.scratch)) {
                throw std::runtime_error("forced radial dispatch failed");
            }
        } else if (route == Route::GenericCart) {
            if (!dispatch_generic_forced<false, Emit>(
                    work.bra_code, work.ket_code, basis,
                    work.bra, work.ket, work.scratch)) {
                throw std::runtime_error("generic Cartesian dispatch failed");
            }
        } else {
            if (!dispatch_rys_generic_spherical_shell4_bin_pair<Emit>(
                    work.bra_code, work.ket_code, basis,
                    work.bra, work.ket, false, work.scratch)) {
                throw std::runtime_error("generic shell4 dispatch failed");
            }
        }
    }
}

Fingerprint fingerprint_route(Route route, RouteWorkload& work) {
    Fingerprint result;
    FingerprintGuard fingerprint_guard(result);
    const LowBatchPairComponentsGuard pair_guard(
        true, &work.basis_owner->packed_spherical_pair_components);
    execute_route<FingerprintEmit>(route, work);
    return result;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    if (values.empty()) return 0.0;
    const std::size_t middle = values.size() / 2u;
    if ((values.size() & 1u) != 0u) return values[middle];
    return 0.5 * (values[middle - 1u] + values[middle]);
}

struct RouteTiming {
    double best_seconds = 0.0;
    double median_seconds = 0.0;
    double checksum_per_quartet = 0.0;
};

RouteTiming time_route(
    Route route,
    RouteWorkload& work,
    int iterations,
    int repeats,
    int warmup) {
    EriStreamAccumulator accumulator;
    EriStreamModeGuard stream_guard(
        true, &work.basis_owner->packed_spherical_pair_components,
        &accumulator);
    const auto one_set = [&]() {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            reset_eri_stream_accumulator(accumulator);
            execute_route<RysPackedTensorEmit>(route, work);
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - begin).count();
    };
    for (int repeat = 0; repeat < warmup; ++repeat) (void)one_set();
    std::vector<double> seconds;
    seconds.reserve(static_cast<std::size_t>(repeats));
    for (int repeat = 0; repeat < repeats; ++repeat) {
        seconds.push_back(one_set());
    }
    RouteTiming result;
    result.best_seconds = *std::min_element(seconds.begin(), seconds.end());
    result.median_seconds = median(seconds);
    result.checksum_per_quartet =
        accumulator.checksum / static_cast<double>(work.quartets);
    return result;
}

Mode parse_mode(const std::string& text) {
    if (text == "production")
        return {Route::Production, 0, 0, false, false, text};
    if (text == "generic-auto")
        return {Route::GenericAuto, 0, 0, false, false, text};
    if (text == "generic-early")
        return {Route::GenericEarly, 0, 0, false, false, text};
    if (text == "generic-cart")
        return {Route::GenericCart, 0, 0, false, false, text};
    if (text == "production-pack-native")
        return {Route::Production, kNativeLanes, 0, false, false, text};
    if (text == "production-pack1xnative")
        return {Route::Production, 0, kNativeLanes, false, false, text};
    if (text == "generic-pack-native")
        return {Route::GenericAuto, kNativeLanes, 0, false, false, text};
    if (text == "generic-pack1xnative")
        return {Route::GenericAuto, 0, kNativeLanes, false, false, text};
    // Legacy fixed-width controls are retained for reproducibility of older
    // campaign logs. New AVX2/AVX-512 evidence must use the native-width modes.
    if (text == "production-pack4")
        return {Route::Production, 4, 0, false, false, text};
    if (text == "production-pack1x4")
        return {Route::Production, 0, 4, false, false, text};
    if (text == "generic-pack4")
        return {Route::GenericAuto, 4, 0, false, false, text};
    if (text == "generic-pack1x4")
        return {Route::GenericAuto, 0, 4, false, false, text};
    if (text == "roots-batch4")
        return {Route::Production, 0, 0, true, false, text};
    if (text == "roots-scalar4")
        return {Route::Production, 0, 0, false, true, text};
    throw std::runtime_error("unknown --mode: " + text);
}

template <int N>
__attribute__((noinline)) bool roots_batch_call(
    const double* arguments, double* roots, double* weights) noexcept {
    return rys_roots_weights_batch4<N>(arguments, roots, weights);
}

template <int N>
__attribute__((noinline)) bool roots_scalar_call(
    const double* arguments, double* roots, double* weights) noexcept {
    for (int lane = 0; lane < 4; ++lane) {
        double lane_roots[N];
        double lane_weights[N];
        rys_roots_weights_fixed<N>(
            arguments[lane], lane_roots, lane_weights);
        for (int root = 0; root < N; ++root) {
            roots[4 * root + lane] = lane_roots[root];
            weights[4 * root + lane] = lane_weights[root];
        }
    }
    return true;
}

bool roots_call(
    int nroots,
    bool batch,
    const double* arguments,
    double* roots,
    double* weights) {
#define TLNO_ABLATION_ROOT_CASE(N)                                            \
    case N: return batch                                                      \
        ? roots_batch_call<N>(arguments, roots, weights)                      \
        : roots_scalar_call<N>(arguments, roots, weights)
    switch (nroots) {
        TLNO_ABLATION_ROOT_CASE(1);
        TLNO_ABLATION_ROOT_CASE(2);
        TLNO_ABLATION_ROOT_CASE(3);
        TLNO_ABLATION_ROOT_CASE(4);
        TLNO_ABLATION_ROOT_CASE(5);
        TLNO_ABLATION_ROOT_CASE(6);
        TLNO_ABLATION_ROOT_CASE(7);
        default: throw std::runtime_error("--nroots must be in 1..7");
    }
#undef TLNO_ABLATION_ROOT_CASE
}

inline void compiler_escape(const double* roots, const double* weights) {
#if defined(__GNUC__) || defined(__clang__)
    asm volatile("" : : "g"(roots), "g"(weights) : "memory");
#else
    (void)roots;
    (void)weights;
#endif
}

int run_roots(
    const Mode& mode,
    int nroots,
    int iterations,
    int repeats,
    int warmup) {
    if (nroots < 1 || nroots > 7) {
        throw std::runtime_error("--nroots must be in 1..7");
    }
    const bool batch = mode.roots_batch4;
    const double table_max = g_rys_tbl.tmax[nroots];
    const double span = std::max(0.001, std::min(12.0, table_max * 0.80));
    std::array<std::array<double, 4>, 256> argument_bank{};
    for (std::size_t row = 0; row < argument_bank.size(); ++row) {
        const double jitter =
            (static_cast<double>(row) + 0.5) /
            static_cast<double>(argument_bank.size()) * 0.01;
        argument_bank[row] = {
            span * (0.07 + jitter), span * (0.23 + jitter),
            span * (0.47 + jitter), span * (0.79 + jitter)};
    }

    alignas(32) double roots[4 * 7]{};
    alignas(32) double weights[4 * 7]{};
    const auto one_set = [&]() {
        const auto begin = std::chrono::steady_clock::now();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            const auto& arguments = argument_bank[
                static_cast<std::size_t>(iteration) & 255u];
            if (!roots_call(
                    nroots, batch, arguments.data(), roots, weights)) {
                throw std::runtime_error(
                    "batch4 root lookup unexpectedly requested fallback");
            }
            compiler_escape(roots, weights);
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - begin).count();
    };
    for (int repeat = 0; repeat < warmup; ++repeat) (void)one_set();
    std::vector<double> seconds;
    for (int repeat = 0; repeat < repeats; ++repeat) {
        seconds.push_back(one_set());
    }

    const std::array<double, 4> validation_arguments = {
        span * 0.11, span * 0.31, span * 0.53, span * 0.73};
    alignas(32) double reference_roots[4 * 7]{};
    alignas(32) double reference_weights[4 * 7]{};
    // The explicit switch is needed because the reference order is dynamic.
    if (!roots_call(nroots, false, validation_arguments.data(),
                    reference_roots, reference_weights) ||
        !roots_call(nroots, batch, validation_arguments.data(),
                    roots, weights)) {
        throw std::runtime_error("root validation dispatch failed");
    }
    double max_abs = 0.0;
    double max_rel = 0.0;
    double checksum = 0.0;
    for (int index = 0; index < 4 * nroots; ++index) {
        for (const auto& pair : {
                 std::pair<double, double>{roots[index], reference_roots[index]},
                 std::pair<double, double>{weights[index], reference_weights[index]}}) {
            const double difference = std::abs(pair.first - pair.second);
            const double magnitude =
                std::max(std::abs(pair.first), std::abs(pair.second));
            max_abs = std::max(max_abs, difference);
            max_rel = std::max(
                max_rel, difference / std::max(magnitude, 1.0e-300));
        }
        checksum += roots[index] + weights[index];
    }
    const bool gate = max_abs <= 2.0e-14 + 2.0e-12 *
        std::max(1.0, std::abs(checksum));
    const double best = *std::min_element(seconds.begin(), seconds.end());
    const double med = median(seconds);
    const double calls = static_cast<double>(iterations);

    std::cout << std::setprecision(15);
    std::cout << "engine=tlno_rys_quartet_batched_root_lookup_control\n";
    std::cout << "mode=" << mode.label << "\n";
    std::cout << "nroots=" << nroots << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "repeats=" << repeats << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "quartets_per_call=4\n";
    std::cout << "simd_lane_semantics="
              << (batch ? "four_independent_quartet_arguments"
                        : "four_scalar_quartet_calls") << "\n";
    std::cout << "root_axis_simd_retained_elsewhere=1\n";
    std::cout << "root_axis_simd_active_in_measured_production_s_f_route=0\n";
    std::cout << "root_axis_simd_measured=0\n";
    std::cout << "measurement_scope=quadrature_lookup_only\n";
    std::cout << "best_s=" << best << "\n";
    std::cout << "median_s=" << med << "\n";
    std::cout << "ns_per_quartet_best=" << best * 1.0e9 / (4.0 * calls)
              << "\n";
    std::cout << "ns_per_quartet_median=" << med * 1.0e9 / (4.0 * calls)
              << "\n";
    std::cout << "checksum=" << checksum << "\n";
    std::cout << "root_validation_max_abs=" << max_abs << "\n";
    std::cout << "root_validation_max_rel=" << max_rel << "\n";
    std::cout << "root_validation_gate=" << (gate ? "pass" : "fail")
              << "\n";
    return gate ? 0 : 1;
}

void print_usage(const char* executable) {
    std::cerr
        << "Usage: " << executable
        << " --mode MODE [--class ff|ff] [--batch N] [--nprim N]"
        << " [--iterations N] [--repeats N] [--warmup N]"
        << " [--nroots 1..7]\n"
        << "Modes: production, generic-auto, generic-early, generic-cart,\n"
        << "       production-pack-native, production-pack1xnative,\n"
        << "       generic-pack-native, generic-pack1xnative,\n"
        << "       production-pack4, production-pack1x4, generic-pack4,\n"
        << "       generic-pack1x4, roots-batch4, roots-scalar4\n";
}

}  // namespace

int parse_integer(const char* text) {
    std::size_t consumed = 0;
    const int value = std::stoi(text, &consumed);
    if (text[consumed] != '\0') {
        throw std::runtime_error("Invalid integer argument.");
    }
    return value;
}

double parse_number(const char* text) {
    std::size_t consumed = 0;
    const double value = std::stod(text, &consumed);
    if (text[consumed] != '\0') {
        throw std::runtime_error("Invalid numeric argument.");
    }
    return value;
}

int run(int argc, char** argv) {
    std::array<int, 4> angular = {3, 3, 3, 3};
    std::string mode_text;
    int batch = 4;
    int nprim = 3;
    int iterations = 100;
    int repeats = 5;
    int warmup = 1;
    int nroots = 1;
    double exponent = 1.0;
    double scale = 1.0;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto require_value = [&]() -> const char* {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value after " + argument);
            }
            return argv[++i];
        };
        if (argument == "--mode") mode_text = require_value();
        else if (argument == "--class" || argument == "-c")
            angular = parse_class(require_value());
        else if (argument == "--batch") batch = parse_integer(require_value());
        else if (argument == "--nprim") nprim = parse_integer(require_value());
        else if (argument == "--iterations")
            iterations = parse_integer(require_value());
        else if (argument == "--repeats") repeats = parse_integer(require_value());
        else if (argument == "--warmup") warmup = parse_integer(require_value());
        else if (argument == "--nroots") nroots = parse_integer(require_value());
        else if (argument == "--exp") exponent = parse_number(require_value());
        else if (argument == "--scale") scale = parse_number(require_value());
        else if (argument == "--help" || argument == "-h") {
            print_usage(argv[0]);
            return 0;
        } else {
            throw std::runtime_error("unknown argument: " + argument);
        }
    }
    if (mode_text.empty()) throw std::runtime_error("--mode is required");
    if (batch <= 0 || nprim <= 0 || iterations <= 0 || repeats <= 0 ||
        warmup < 0) {
        throw std::runtime_error(
            "batch/nprim/iterations/repeats must be positive; warmup >= 0");
    }
    if (!tlno::core::math::is_finite_binary64(exponent) ||
        !tlno::core::math::is_finite_binary64(scale) ||
        exponent <= 0.0 || scale <= 0.0) {
        throw std::runtime_error("exp and scale must be finite and positive");
    }
    const Mode mode = parse_mode(mode_text);
    if (mode.roots_batch4 || mode.roots_scalar4) {
        return run_roots(mode, nroots, iterations, repeats, warmup);
    }

    const auto shells = make_basis(angular, exponent, nprim, scale);
    RouteWorkload work = make_workload(shells, angular, mode, batch);
    const Fingerprint fingerprint = fingerprint_route(mode.route, work);
    const std::uint64_t expected_count =
        work.quartets * work.spherical_per_quartet;
    const bool count_gate = fingerprint.count == expected_count;
    const RouteTiming timing = time_route(
        mode.route, work, iterations, repeats, warmup);
    const long double normalized_fp_checksum =
        fingerprint.checksum / static_cast<long double>(work.quartets);
    const double self_difference = std::abs(
        timing.checksum_per_quartet -
        static_cast<double>(normalized_fp_checksum));
    const double self_scale = std::max({
        1.0, std::abs(timing.checksum_per_quartet),
        std::abs(static_cast<double>(normalized_fp_checksum))});
    const bool checksum_gate =
        self_difference <= 2.0e-12 + 5.0e-10 * self_scale;
    const double calls = static_cast<double>(iterations);
    const double spherical = calls * static_cast<double>(
        work.quartets * work.spherical_per_quartet);
    const double quartets = calls * static_cast<double>(work.quartets);
    const bool uses_generic_radial_early =
        work.generic_radial_selected;
    const char* const selected_route = work.specialized
        ? "production_specialized"
        : uses_generic_radial_early
            ? "generic_radial_shell4"
            : "generic_cart_shell4";

    std::cout << std::setprecision(15);
    std::cout << "engine=tlno_rys_route_ablation\n";
    std::cout << "mode=" << mode.label << "\n";
    std::cout << "class=" << class_label(angular) << "\n";
    std::cout << "la=" << angular[0] << "\n";
    std::cout << "lb=" << angular[1] << "\n";
    std::cout << "lc=" << angular[2] << "\n";
    std::cout << "ld=" << angular[3] << "\n";
    std::cout << "nroots="
              << (angular[0] + angular[1] + angular[2] + angular[3]) / 2 + 1
              << "\n";
    std::cout << "nprim=" << nprim << "\n";
    std::cout << "batch_argument=" << batch << "\n";
    std::cout << "bra_pairs=" << work.bra.pair_count << "\n";
    std::cout << "ket_pairs=" << work.ket.pair_count << "\n";
    std::cout << "route_calls_per_iteration=" << work.route_calls << "\n";
    std::cout << "quartets_per_call=" << work.quartets << "\n";
    std::cout << "spherical_per_quartet=" << work.spherical_per_quartet
              << "\n";
    std::cout << "primitive_quartets_per_quartet="
              << work.primitive_quartets_per_quartet << "\n";
    std::cout << "iterations=" << iterations << "\n";
    std::cout << "repeats=" << repeats << "\n";
    std::cout << "warmup=" << warmup << "\n";
    std::cout << "used_specialized=" << (work.specialized ? 1 : 0) << "\n";
    std::cout << "selected_route=" << selected_route << "\n";
    std::cout << "radial_eligible=" << (work.radial_eligible ? 1 : 0) << "\n";
    std::cout << "generic_bra_uniform="
              << (work.generic_bra_uniform ? 1 : 0) << "\n";
    std::cout << "generic_radial_dispatch_predicate="
              << (work.generic_radial_dispatch_predicate ? 1 : 0) << "\n";
    std::cout << "generic_radial_selected="
              << (work.generic_radial_selected ? 1 : 0) << "\n";
    std::cout << "generic_radial_early_spherical_selected="
              << (uses_generic_radial_early ? 1 : 0) << "\n";
    std::cout << "timed_emit=RysPackedTensorEmit\n";
    std::cout << "fingerprint_emit=FingerprintEmit\n";
    std::cout << "native_simd_lanes=" << kNativeLanes << "\n";
    std::cout << "simd_lane_semantics=" << kNativeLanes
              << "_independent_shell_quartets\n";
    std::cout << "root_axis_simd_retained_elsewhere=1\n";
    std::cout << "root_axis_simd_active_in_measured_production_s_f_route=0\n";
    std::cout << "root_axis_simd_measured=0\n";
    std::cout << "best_s=" << timing.best_seconds << "\n";
    std::cout << "median_s=" << timing.median_seconds << "\n";
    std::cout << "ns_per_quartet_best="
              << timing.best_seconds * 1.0e9 / quartets << "\n";
    std::cout << "ns_per_quartet_median="
              << timing.median_seconds * 1.0e9 / quartets << "\n";
    std::cout << "ns_per_spherical_integral_best="
              << timing.best_seconds * 1.0e9 / spherical << "\n";
    std::cout << "ns_per_spherical_integral_median="
              << timing.median_seconds * 1.0e9 / spherical << "\n";
    std::cout << "checksum_per_quartet="
              << timing.checksum_per_quartet << "\n";
    std::cout << "fp_count=" << fingerprint.count << "\n";
    std::cout << "fp_count_expected=" << expected_count << "\n";
    std::cout << "fp_checksum_per_quartet="
              << normalized_fp_checksum << "\n";
    std::cout << "fp_abs_sum_per_quartet="
              << fingerprint.absolute_sum /
                     static_cast<long double>(work.quartets) << "\n";
    std::cout << "fp_square_sum_per_quartet="
              << fingerprint.square_sum /
                     static_cast<long double>(work.quartets) << "\n";
    std::cout << "fp_indexed_1_per_quartet="
              << fingerprint.indexed_1 /
                     static_cast<long double>(work.quartets) << "\n";
    std::cout << "fp_indexed_2_per_quartet="
              << fingerprint.indexed_2 /
                     static_cast<long double>(work.quartets) << "\n";
    std::cout << "fp_indexed_3_per_quartet="
              << fingerprint.indexed_3 /
                     static_cast<long double>(work.quartets) << "\n";
    std::cout << "fp_max_abs=" << fingerprint.max_abs << "\n";
    std::cout << "count_gate=" << (count_gate ? "pass" : "fail") << "\n";
    std::cout << "timed_vs_fp_checksum_abs=" << self_difference << "\n";
    std::cout << "timed_vs_fp_checksum_gate="
              << (checksum_gate ? "pass" : "fail") << "\n";
    return count_gate && checksum_gate ? 0 : 1;
}

}  // namespace tlno::integrals::twoe::route_ablation

int main(int argc, char** argv) {
    try {
        return tlno::integrals::twoe::route_ablation::run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "rys_route_ablation_bench: " << error.what() << "\n";
        return 2;
    }
}
