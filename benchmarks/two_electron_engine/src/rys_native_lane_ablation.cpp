// Validation/benchmark-only native-width quartet-lane causal surface.
//
// This TU directly hosts the experimental avx2_avx512 Rys implementation.
// It exercises the same NativePacket<W> producer used by the portable W4/W8
// route, with either W independent quartets live in one packet or one live
// quartet per invocation repeated W times.  Production ABI and dispatch are
// unchanged.
#define TLNO_BUILD_RYS_AUDIT 1
#include "integrals/two_electron/rys/rys_engine.ipp"

#include "core/basis/types.hpp"
#include "core/shell/shell_contract.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <utility>

namespace tlno::integrals::twoe::native_lane_ablation {
namespace {

constexpr int W = tlno::core::simd::lanes;

static_assert(W == 4 || W == 8, "Native lane ablation requires AVX2 W4 or AVX-512 W8");

std::vector<std::array<int, 3>> cartesian_components(int L) {
    std::vector<std::array<int, 3>> out;
    for (int nx = L; nx >= 0; --nx) {
        for (int ny = L - nx; ny >= 0; --ny) {
            out.push_back({nx, ny, L - nx - ny});
        }
    }
    return out;
}

char angular_symbol(int L) {
    static constexpr char symbols[] = {'s', 'p', 'd', 'f'};
    return (L >= 0 && L <= 3) ? symbols[L] : '?';
}

std::array<int, 4> parse_class(const std::string& text) {
    std::string cleaned;
    for (const char c : text) {
        if (c == 's' || c == 'S') cleaned += "0 ";
        else if (c == 'p' || c == 'P') cleaned += "1 ";
        else if (c == 'd' || c == 'D') cleaned += "2 ";
        else if (c == 'f' || c == 'F') cleaned += "3 ";
        else if (c >= '0' && c <= '3') {
            cleaned.push_back(c);
            cleaned.push_back(' ');
        } else if (c == '|' || c == ',' || c == ':' || c == '/' ||
                   c == ' ' || c == '\t') {
            cleaned.push_back(' ');
        } else {
            throw std::runtime_error("invalid --class character");
        }
    }
    std::istringstream input(cleaned);
    std::array<int, 4> L{};
    for (int i = 0; i < 4; ++i) {
        if (!(input >> L[static_cast<std::size_t>(i)]))
            throw std::runtime_error("--class must contain four s--f momenta");
    }
    int extra = 0;
    if (input >> extra) throw std::runtime_error("too many --class momenta");
    return L;
}

std::string class_label(const std::array<int, 4>& L) {
    std::string out;
    out.push_back(angular_symbol(L[0]));
    out.push_back(angular_symbol(L[1]));
    out.push_back('|');
    out.push_back(angular_symbol(L[2]));
    out.push_back(angular_symbol(L[3]));
    return out;
}

std::vector<tlno::core::basis::Shell> make_basis(
    const std::array<int, 4>& angular, int nprim) {
    const std::array<std::array<double, 3>, 4> centers = {{
        {{0.00, 0.00, 0.00}},
        {{1.37, 0.19, 0.11}},
        {{0.43, 1.21, 0.29}},
        {{0.23, 0.31, 1.51}},
    }};
    std::vector<tlno::core::basis::Shell> basis;
    for (int group = 0; group < 4; ++group) {
        const int L = angular[static_cast<std::size_t>(group)];
        for (const auto& component : cartesian_components(L)) {
            tlno::core::basis::RawShell raw;
            raw.label = std::string(1, angular_symbol(L));
            raw.ang = component;
            for (int primitive = nprim; primitive >= 1; --primitive) {
                raw.exponents.push_back(static_cast<double>(primitive));
                raw.coefficients.push_back(1.0);
            }
            basis.push_back(
                tlno::core::shell::make_shell(raw, centers[static_cast<std::size_t>(group)]));
        }
    }
    return basis;
}

std::size_t find_pair_group(const BasisDataHGP& bd, int first, int second) {
    for (std::size_t i = 0; i < bd.pair_groups.size(); ++i) {
        const auto [a, b] = bd.pair_groups[i];
        if ((a == first && b == second) || (a == second && b == first))
            return i;
    }
    throw std::runtime_error("synthetic pair group not found");
}

NativePairBin make_native_bin(
    const BasisDataHGP& bd, std::size_t pair_group, int copies) {
    if (copies <= 0) throw std::runtime_error("copies must be positive");
    const auto [a, b] = bd.pair_groups.at(pair_group);
    NativePairBin bin;
    bin.la = bd.groups.at(a).L;
    bin.lb = bd.groups.at(b).L;
    bin.primitives = bd.rys_pair_primitives.at(pair_group).size();
    bin.ids.assign(
        static_cast<std::size_t>(copies),
        static_cast<std::uint32_t>(pair_group));
    bin.pack(bd);
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
    // W8 production caches this immutable projector metadata.  Construct the
    // same cache for the synthetic duplicate-ID packet instead of disabling it.
    bin.radial_projector_cache =
        std::make_shared<const NativeRadialPairProjectorCache<W>>(
            bd, bin.ids, bin.la, bin.lb);
#endif
    return bin;
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

thread_local Fingerprint* g_fp = nullptr;

struct FingerprintEmit {
    static inline void emit(
        std::vector<double>&,
        std::uint32_t lhs_pair,
        std::uint32_t rhs_pair,
        double value) noexcept {
        Fingerprint& fp = *g_fp;
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
    explicit FingerprintGuard(Fingerprint& fp) : previous_(g_fp) { g_fp = &fp; }
    ~FingerprintGuard() { g_fp = previous_; }
private:
    Fingerprint* previous_;
};

struct Work {
    std::shared_ptr<const BasisDataHGP> bd;
    NativePairBin lhs;
    NativePairBin rhs;
    NativeQuartetShape shape;
    int calls = 1;
    std::uint64_t quartets = 0;
    std::uint64_t spherical_per_quartet = 0;
    std::uint64_t primitive_quartets_per_quartet = 0;

    Work(
        std::shared_ptr<const BasisDataHGP> owner,
        NativePairBin left,
        NativePairBin right,
        NativeQuartetShape quartet_shape,
        int route_calls)
        : bd(std::move(owner)),
          lhs(std::move(left)),
          rhs(std::move(right)),
          shape(std::move(quartet_shape)),
          calls(route_calls) {}
};

Work make_work(
    const std::array<int, 4>& angular,
    int nprim,
    bool full_packet) {
    const auto shells = make_basis(angular, nprim);
    EriPackedBuildOptions options;
    options.schwarz_threshold = 0.0;
    options.primitive_fastschwarz_threshold = 0.0;
    options.decay_threshold = 0.0;
    const EriPackedPlan plan = build_eri_packed_plan_for_G_rys(shells, options);
    const auto bd = packed_plan_basis_data(plan);
    if (!bd) throw std::runtime_error("null synthetic basis data");

    const std::size_t lhs_id = find_pair_group(*bd, 0, 1);
    const std::size_t rhs_id = find_pair_group(*bd, 2, 3);
    const int lhs_copies = full_packet ? W : 1;
    NativePairBin lhs = make_native_bin(*bd, lhs_id, lhs_copies);
    NativePairBin rhs = make_native_bin(*bd, rhs_id, 1);
    Work work(
        bd,
        std::move(lhs),
        std::move(rhs),
        NativeQuartetShape(angular[0], angular[1], angular[2], angular[3]),
        full_packet ? 1 : W);
    work.quartets =
        static_cast<std::uint64_t>(lhs_copies) *
        static_cast<std::uint64_t>(work.calls);
    work.spherical_per_quartet =
        static_cast<std::uint64_t>((2 * angular[0] + 1) *
                                   (2 * angular[1] + 1) *
                                   (2 * angular[2] + 1) *
                                   (2 * angular[3] + 1));
    work.primitive_quartets_per_quartet =
        static_cast<std::uint64_t>(
            bd->rys_pair_primitives.at(lhs_id).size()) *
        static_cast<std::uint64_t>(
            bd->rys_pair_primitives.at(rhs_id).size());
    return work;
}

template <class Emit>
bool execute_once(Work& work, std::vector<double>& values) {
    bool specialized = false;
    for (int call = 0; call < work.calls; ++call) {
#if TLNO_NATIVE_SPECIALIZED
        if (dispatch_native_specialized<W, Emit>(
                *work.bd,
                work.lhs,
                work.rhs,
                work.shape,
                false,
                nullptr,
                0.0,
                0.0,
                values)) {
            specialized = true;
            continue;
        }
#endif
        native_quartet_bin_pair<W, Emit>(
            *work.bd,
            work.lhs,
            work.rhs,
            work.shape,
            false,
            nullptr,
            0.0,
            0.0,
            values);
    }
    return specialized;
}

Fingerprint fingerprint(Work& work, bool& specialized) {
    Fingerprint fp;
    FingerprintGuard guard(fp);
    std::vector<double> values;
    specialized = execute_once<FingerprintEmit>(work, values);
    return fp;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t n = values.size();
    if (n == 0) return 0.0;
    return (n & 1u) ? values[n / 2]
                    : 0.5 * (values[n / 2 - 1] + values[n / 2]);
}

struct Timing {
    double best = 0.0;
    double median = 0.0;
    double checksum_per_quartet = 0.0;
    bool specialized = false;
};

Timing time_work(Work& work, int iterations, int repeats, int warmup) {
    EriStreamAccumulator accumulator;
    EriStreamModeGuard stream_guard(
        true, &work.bd->packed_spherical_pair_components, &accumulator);
    std::vector<double> values;
    bool specialized = false;
    const auto one_set = [&]() {
        const auto begin = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            reset_eri_stream_accumulator(accumulator);
            specialized = execute_once<RysPackedTensorEmit>(work, values);
        }
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(end - begin).count();
    };
    for (int i = 0; i < warmup; ++i) (void)one_set();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) samples.push_back(one_set());
    return {
        *std::min_element(samples.begin(), samples.end()),
        median(samples),
        accumulator.checksum / static_cast<double>(work.quartets),
        specialized,
    };
}

void print_usage(const char* program) {
    std::cerr
        << "Usage: " << program
        << " --class ff|ff --mode full|one-repeated"
        << " [--nprim 3] [--iterations N] [--repeats 7] [--warmup 2]\n";
}

}  // namespace
}  // namespace tlno::twoe::native_lane_ablation

int main(int argc, char** argv) {
    using namespace tlno::integrals::twoe::native_lane_ablation;
    try {
        std::array<int, 4> angular = {3, 3, 3, 3};
        std::string mode;
        int nprim = 3;
        int iterations = 1000;
        int repeats = 7;
        int warmup = 2;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            auto require_value = [&]() -> const char* {
                if (++i >= argc) throw std::runtime_error("missing value for " + arg);
                return argv[i];
            };
            if (arg == "--class" || arg == "-c")
                angular = parse_class(require_value());
            else if (arg == "--mode")
                mode = require_value();
            else if (arg == "--nprim")
                nprim = std::stoi(require_value());
            else if (arg == "--iterations")
                iterations = std::stoi(require_value());
            else if (arg == "--repeats")
                repeats = std::stoi(require_value());
            else if (arg == "--warmup")
                warmup = std::stoi(require_value());
            else if (arg == "--help" || arg == "-h") {
                print_usage(argv[0]);
                return 0;
            } else {
                throw std::runtime_error("unknown argument: " + arg);
            }
        }
        if (mode != "full" && mode != "one-repeated")
            throw std::runtime_error("--mode must be full or one-repeated");
        if (nprim <= 0 || iterations <= 0 || repeats <= 0 || warmup < 0)
            throw std::runtime_error("invalid campaign size");

        const bool full = mode == "full";
        Work work = make_work(angular, nprim, full);
        bool fp_specialized = false;
        const Fingerprint fp = fingerprint(work, fp_specialized);
        const Timing timing =
            time_work(work, iterations, repeats, warmup);
        if (fp_specialized != timing.specialized)
            throw std::runtime_error("route selection changed between gates and timing");

        const std::uint64_t expected_count =
            work.quartets * work.spherical_per_quartet;
        const bool count_gate = fp.count == expected_count;
        const long double fp_checksum_per_quartet =
            fp.checksum / static_cast<long double>(work.quartets);
        const double difference = std::abs(
            timing.checksum_per_quartet -
            static_cast<double>(fp_checksum_per_quartet));
        const double scale = std::max({
            1.0,
            std::abs(timing.checksum_per_quartet),
            std::abs(static_cast<double>(fp_checksum_per_quartet)),
        });
        const bool timed_fp_gate =
            difference <= 2.0e-12 + 5.0e-10 * scale;
        const double total_quartets =
            static_cast<double>(iterations) *
            static_cast<double>(work.quartets);
        const double total_spherical =
            total_quartets * static_cast<double>(work.spherical_per_quartet);

        std::cout << std::setprecision(17);
        std::cout << "engine=tlno_native_lane_ablation\n";
        std::cout << "mode=" << mode << "\n";
        std::cout << "class=" << class_label(angular) << "\n";
        std::cout << "native_simd_lanes=" << W << "\n";
        std::cout << "lane_semantics=independent_shell_quartets\n";
        std::cout << "live_lanes_per_invocation=" << (full ? W : 1) << "\n";
        std::cout << "route_calls_per_iteration=" << work.calls << "\n";
        std::cout << "quartets_per_iteration=" << work.quartets << "\n";
        std::cout << "spherical_per_quartet=" << work.spherical_per_quartet << "\n";
        std::cout << "primitive_quartets_per_quartet="
                  << work.primitive_quartets_per_quartet << "\n";
        std::cout << "used_native_specialized="
                  << (timing.specialized ? 1 : 0) << "\n";
        std::cout << "nprim=" << nprim << "\n";
        std::cout << "iterations=" << iterations << "\n";
        std::cout << "repeats=" << repeats << "\n";
        std::cout << "warmup=" << warmup << "\n";
        std::cout << "best_s=" << timing.best << "\n";
        std::cout << "median_s=" << timing.median << "\n";
        std::cout << "ns_per_quartet_best="
                  << timing.best * 1.0e9 / total_quartets << "\n";
        std::cout << "ns_per_quartet_median="
                  << timing.median * 1.0e9 / total_quartets << "\n";
        std::cout << "ns_per_spherical_integral_median="
                  << timing.median * 1.0e9 / total_spherical << "\n";
        std::cout << "checksum_per_quartet="
                  << timing.checksum_per_quartet << "\n";
        std::cout << "fp_count=" << fp.count << "\n";
        std::cout << "fp_count_expected=" << expected_count << "\n";
        std::cout << "fp_checksum_per_quartet="
                  << fp_checksum_per_quartet << "\n";
        std::cout << "fp_abs_sum_per_quartet="
                  << fp.absolute_sum / static_cast<long double>(work.quartets)
                  << "\n";
        std::cout << "fp_square_sum_per_quartet="
                  << fp.square_sum / static_cast<long double>(work.quartets)
                  << "\n";
        std::cout << "fp_indexed_1_per_quartet="
                  << fp.indexed_1 / static_cast<long double>(work.quartets)
                  << "\n";
        std::cout << "fp_indexed_2_per_quartet="
                  << fp.indexed_2 / static_cast<long double>(work.quartets)
                  << "\n";
        std::cout << "fp_indexed_3_per_quartet="
                  << fp.indexed_3 / static_cast<long double>(work.quartets)
                  << "\n";
        std::cout << "fp_max_abs=" << fp.max_abs << "\n";
        std::cout << "count_gate=" << (count_gate ? "pass" : "fail") << "\n";
        std::cout << "timed_vs_fp_checksum_gate="
                  << (timed_fp_gate ? "pass" : "fail") << "\n";
        return count_gate && timed_fp_gate ? 0 : 3;
    } catch (const std::exception& ex) {
        std::cerr << "rys_native_lane_ablation_bench: " << ex.what() << "\n";
        return 1;
    }
}
