// Real-molecule one-electron kernel profile, binned by shell-pair class.
//
// The profile deliberately bypasses plan construction, dense-matrix clearing,
// scatter, and screening.  It measures the same prebuilt ShellPairPrim objects
// used by the cached production evaluator and reports one complete traversal of
// every unordered group pair in each angular-momentum class.  S+T is timed as
// the fused TLNO kernel; V is timed separately.
#include "benchmarks/common/molecule_io.hpp"
#include "integrals/one_electron/onee_integrals.hpp"
#include "integrals/one_electron/onee_matrices.hpp"
#include "core/basis/basis_general.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using tlno::integrals::onee::OneElectronPlan;

volatile double g_profile_sink = 0.0;

struct ClassWork {
    int l_lo = 0;
    int l_hi = 0;
    std::vector<int> pair_indices;
    std::size_t primitive_pairs = 0;
    std::size_t cartesian_pairs = 0;
    std::size_t max_block_size = 0;
};

double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    const std::size_t mid = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid), values.end());
    const double high = values[mid];
    if (values.size() % 2 != 0) return high;
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(mid - 1), values.end());
    return 0.5 * (values[mid - 1] + high);
}

template <class Body>
double time_traversal(Body&& body, int repeats, int warmups, double target_s) {
    const auto c0 = Clock::now();
    body();
    const double calibration =
        std::max(1.0e-9, std::chrono::duration<double>(Clock::now() - c0).count());
    const int iterations = std::clamp(
        static_cast<int>(std::ceil(target_s / calibration)), 1, 100000);

    for (int w = 0; w < warmups; ++w) {
        for (int it = 0; it < iterations; ++it) body();
    }
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const auto t0 = Clock::now();
        for (int it = 0; it < iterations; ++it) body();
        const auto t1 = Clock::now();
        samples.push_back(
            std::chrono::duration<double>(t1 - t0).count() /
            static_cast<double>(iterations));
    }
    return median(std::move(samples));
}

const char* angular_name(int l) {
    constexpr const char* names[] = {"s", "p", "d", "f", "g", "h"};
    return (l >= 0 && l < static_cast<int>(std::size(names))) ? names[l] : "L";
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string inp_path;
        std::string basis_name = "631gdp";
        int repeats = 7;
        int warmups = 2;
        double target_ms = 5.0;

        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if (arg == "--inp" && i + 1 < argc) {
                inp_path = argv[++i];
            } else if (arg == "--basis" && i + 1 < argc) {
                basis_name = argv[++i];
            } else if (arg == "--repeats" && i + 1 < argc) {
                repeats = std::max(1, std::stoi(argv[++i]));
            } else if (arg == "--warmups" && i + 1 < argc) {
                warmups = std::max(0, std::stoi(argv[++i]));
            } else if (arg == "--target-ms" && i + 1 < argc) {
                target_ms = std::max(0.01, std::stod(argv[++i]));
            } else if (arg == "--help" || arg == "-h") {
                std::cout
                    << "Usage: onee_class_profile --inp molecule.inp --basis NAME "
                    << "[--repeats N] [--warmups N] [--target-ms MS]\n";
                return 0;
            } else {
                throw std::runtime_error("unknown or incomplete argument: " + arg);
            }
        }
        if (inp_path.empty()) throw std::runtime_error("--inp is required");

        const auto mol = tlno::benchmarks::onee::read_inp_file(inp_path);
        tlno::core::molecule::MoleculeSpec spec;
        spec.symbols = mol.symbols;
        spec.coords = mol.coords_bohr;
        spec.charge = mol.charge;
        const auto basis_kind = tlno::core::basis::parse_basis_kind(basis_name);
        const auto build = tlno::core::basis::build_basis_general(spec, basis_kind); 

        // The class story is a raw-kernel story: retain every group and
        // primitive pair, matching the Libcint class driver with EXPCUTOFF=1e9.
        tlno::integrals::onee::set_onee_screening_enabled(false);
        const OneElectronPlan plan = tlno::integrals::onee::build_one_electron_plan(build);

        int max_l = 0;
        for (const auto& group : plan.groups) max_l = std::max(max_l, group.L);
        std::vector<ClassWork> classes;
        for (int hi = 0; hi <= max_l; ++hi) {
            for (int lo = 0; lo <= hi; ++lo) classes.push_back({lo, hi, {}, 0, 0, 0});
        }
        const auto class_index = [](int lo, int hi) {
            return hi * (hi + 1) / 2 + lo;
        };

        for (int pair_index = 0; 
             pair_index < static_cast<int>(plan.group_pairs.size()); ++pair_index) {
            const auto& pair = plan.group_pairs[static_cast<std::size_t>(pair_index)];
            const auto& ga = plan.groups[static_cast<std::size_t>(pair.ga)];
            const auto& gb = plan.groups[static_cast<std::size_t>(pair.gb)];
            const auto& spp = plan.pair_prims[static_cast<std::size_t>(pair_index)];
            const int lo = std::min(ga.L, gb.L);
            const int hi = std::max(ga.L, gb.L);
            ClassWork& work = classes[static_cast<std::size_t>(class_index(lo, hi))];
            work.pair_indices.push_back(pair_index);
            work.primitive_pairs +=
                static_cast<std::size_t>(spp.n_prim_a) *
                static_cast<std::size_t>(spp.n_prim_b);
            const std::size_t block_size =
                ga.ang.size() * gb.ang.size();
            work.cartesian_pairs += block_size;
            work.max_block_size = std::max(work.max_block_size, block_size);
        }

        std::cout
            << "engine,molecule,basis,ao_count,class,l_lo,l_hi,blocks,"
               "primitive_pairs,cartesian_pairs,st_median_s,v_median_s,"
               "st_ns_per_block,v_ns_per_block,st_ns_per_primitive_pair,"
               "v_ns_per_primitive_pair,st_ns_per_cartesian_pair,"
               "v_ns_per_cartesian_pair\n";
        std::cout << std::setprecision(17);

        const std::string molecule_name = [&] {
            const std::size_t slash = inp_path.find_last_of("/\\");
            std::string name = inp_path.substr(slash == std::string::npos ? 0 : slash + 1);
            const std::size_t dot = name.rfind('.');
            if (dot != std::string::npos) name.resize(dot);
            return name;
        }();

        for (const ClassWork& work : classes) {
            if (work.pair_indices.empty()) continue;
            std::vector<double> overlap(work.max_block_size, 0.0);
            std::vector<double> kinetic(work.max_block_size, 0.0);
            std::vector<double> nuclear(work.max_block_size, 0.0);

            const auto run_st = [&] {
                for (const int pair_index : work.pair_indices) {
                    const auto& pair = plan.group_pairs[static_cast<std::size_t>(pair_index)];
                    const auto& ga = plan.groups[static_cast<std::size_t>(pair.ga)];
                    const auto& gb = plan.groups[static_cast<std::size_t>(pair.gb)];
                    tlno::integrals::onee::contracted_shell_block_overlap_kinetic(
                        ga, gb, plan.pair_prims[static_cast<std::size_t>(pair_index)],
                        overlap.data(), kinetic.data(), 0.0);
                }
                g_profile_sink = g_profile_sink + overlap[0] + kinetic[0];
            };
            const auto run_v = [&] {
                for (const int pair_index : work.pair_indices) {
                    const auto& pair = plan.group_pairs[static_cast<std::size_t>(pair_index)];
                    const auto& ga = plan.groups[static_cast<std::size_t>(pair.ga)];
                    const auto& gb = plan.groups[static_cast<std::size_t>(pair.gb)];
                    tlno::integrals::onee::contracted_shell_block_nuclear(
                        ga, gb, plan.pair_prims[static_cast<std::size_t>(pair_index)],
                        nuclear.data(), build.nuclear_charges, build.nuclear_centers);
                }
                g_profile_sink = g_profile_sink + nuclear[0];
            };

            const double target_s = target_ms * 1.0e-3;
            const double st_s = time_traversal(run_st, repeats, warmups, target_s);
            const double v_s = time_traversal(run_v, repeats, warmups, target_s);
            const double blocks = static_cast<double>(work.pair_indices.size());
            const double primitives = static_cast<double>(work.primitive_pairs);
            const double cartesians = static_cast<double>(work.cartesian_pairs);

            std::cout
                << "tlno," << molecule_name << ',' << basis_name << ','
                << build.basis.size() << ','
                << angular_name(work.l_lo) << '|' << angular_name(work.l_hi) << ','
                << work.l_lo << ',' << work.l_hi << ','
                << work.pair_indices.size() << ',' << work.primitive_pairs << ','
                << work.cartesian_pairs << ',' << st_s << ',' << v_s << ','
                << st_s * 1.0e9 / blocks << ',' << v_s * 1.0e9 / blocks << ','
                << st_s * 1.0e9 / primitives << ',' << v_s * 1.0e9 / primitives << ','
                << st_s * 1.0e9 / cartesians << ',' << v_s * 1.0e9 / cartesians
                << '\n';
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
