// Exercise the actual production plan, scatter and OpenMP ownership.
#include "benchmarks/common/molecule_io.hpp"
#include "core/shell/cart2sph.hpp"
#include "integrals/one_electron/onee_matrices.hpp"
#include "core/basis/basis_general.hpp"
#include "core/math/binary64.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <tuple>
#include <utility>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace tlno::integrals::onee;
using tlno::core::basis::DenseMatrix;
namespace shell = tlno::core::shell;
using tlno::core::math::is_finite_binary64;

static auto pair_keys(const OneElectronPlan& plan) {
    std::vector<std::tuple<int,int,bool,bool>> keys;
    for (const auto& pair:plan.group_pairs)
        keys.emplace_back(pair.ga, pair.gb, pair.do_st, pair.do_v);
    std::sort(keys.begin(), keys.end());
    if (std::adjacent_find(keys.begin(), keys.end()) != keys.end())
        throw std::runtime_error("Duplicate group pair.");
    return keys;
}

static double compare(const OneElectronMatrices& actual, const OneElectronMatrices& reference) {
    double worst = 0;
    const DenseMatrix* a[] = {&actual.overlap, &actual.kinetic, &actual.nuclear};
    const DenseMatrix* b[] = {&reference.overlap, &reference.kinetic, &reference.nuclear};
    for (int op = 0; op < 3; ++op) {
        if (a[op]->size != b[op]->size) throw std::runtime_error("Matrix dimension");
        for (std::size_t i = 0; i < a[op]->values.size(); ++i) {
            const double got = a[op]->values[i], ref=b[op]->values[i];
            worst = std::max(worst, std::abs(got - ref));
            if (!is_finite_binary64(got) || std::abs(got-ref) > 2e-10 + 2e-11*std::abs(ref))
                throw std::runtime_error("Matrix numerical mismatch.");
        }
        for (int i = 0; i < a[op]->size; ++i) for (int j = 0; j < i; ++j)
            if ((*a[op])(i, j) != (*a[op])(j, i)) throw std::runtime_error("Symmetry.");
    }
    return worst;
}

static OneElectronMatrices transform_cart(const OneElectronMatrices& cart, const OneElectronPlan& plan) {
    std::vector<std::vector<std::pair<int,double>>> columns(plan.n_sph);
    for (std::size_t g = 0; g < plan.groups.size(); ++g) {
        const auto& grp = plan.groups[g];
        const int nc = shell::cart_count(grp.L), ns = shell::sph_count(grp.L);
        for (std::size_t chunk = 0; chunk < plan.sph_base[g].size(); ++chunk) {
            const int base = plan.sph_base[g][chunk];
            for (int c = 0; c < nc; ++c) {
                std::vector<double> row(nc, 0.0), out(ns, 0.0), scratch(ns, 0.0);
                row[c] = 1.0;
                shell::cart2sph_block(0, grp.L, row.data(), nc,out.data(), scratch.data());
                const int global = grp.basis_indices[chunk * nc + c];
                for (int sph = 0; sph<ns; ++sph)
                    if (out[sph] != 0.0) columns[base + sph].emplace_back(global, out[sph]);
            }
        }
    }
    OneElectronMatrices transformed(plan.n_sph);
    const DenseMatrix* from[] = {&cart.overlap, &cart.kinetic, &cart.nuclear};
    DenseMatrix* to[] = {&transformed.overlap, &transformed.kinetic, &transformed.nuclear};
    for (int op = 0; op < 3; ++op)
        for (int i = 0; i < plan.n_sph; ++i)
            for (int j = i; j < plan.n_sph; ++j) {
                double v = 0.0;
                for (const auto& a:columns[i]) for (const auto& b:columns[j])
                    v += a.second*(*from[op])(a.first, b.first) * b.second;
                (*to[op])(i,j) = v; (*to[op])(j,i) = v;
            }
    return transformed;
}

static void poison(OneElectronMatrices& actual) {
    std::fill(actual.overlap.values.begin(), actual.overlap.values.end(), 123.0);
    std::fill(actual.kinetic.values.begin(), actual.kinetic.values.end(), -234.0);
    std::fill(actual.nuclear.values.begin(), actual.nuclear.values.end(), 345.0);
}

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::runtime_error("Usage: plan_contract molecule.inp basis.");
        const auto spec = tlno::benchmarks::onee::molecule_spec(
            tlno::benchmarks::onee::read_inp_file(argv[1]));
        auto build = tlno::core::basis::build_basis_general(spec, tlno::core::basis::parse_basis_kind(argv[2]));
        double worst = 0;
#ifdef _OPENMP
        omp_set_dynamic(0); omp_set_num_threads(1);
#endif
        OneElectronMatrices unscreened_cart, unscreened_sph;
        for (bool screened : {false, true}) {
#ifdef _OPENMP
            omp_set_num_threads(1);
#endif
            set_onee_screening_enabled(screened);
            auto serial = build_one_electron_plan(build, true);
            auto prepared = build_one_electron_spherical_plan(build, true);
            auto moved = std::move(prepared);
            auto full = build_one_electron_spherical_plan(build, false);
            OneElectronMatrices reference(serial.n_sph), actual(moved.n_sph), single_use(full.n_sph);
            evaluate_one_electron_matrices_sph_into(build, serial, reference);
            auto cart_reference = evaluate_one_electron_matrices(build, serial);
            auto cart_actual = cart_reference;
            worst = std::max(worst, compare(transform_cart(cart_reference, serial), reference));
            worst = std::max(worst, compare(build_one_electron_matrices(build), cart_reference));
            worst = std::max(worst, compare(build_one_electron_spherical_matrices(build), reference));
            if (!screened) { unscreened_cart = cart_reference; unscreened_sph = reference; }
            else {
                worst = std::max(worst, compare(cart_reference, unscreened_cart));
                worst = std::max(worst, compare(reference, unscreened_sph));
            }
            std::cout << "case=" << argv[1] << " basis=" << argv[2]
                      << " screened=" << screened << " cart=" << cart_reference.overlap.size
                      << " sph=" << serial.n_sph << " groups=" << serial.groups.size()
                      << " pairs=" << serial.group_pairs.size()
                      << " cache_bytes=" << serial.nuclear_hermite_cache_bytes << '\n';
            // The same immutable plan must support independent thread counts.
            for (int threads : {1, 2, 3}) {
#ifdef _OPENMP
                omp_set_num_threads(threads);
#endif
                auto parallel = build_one_electron_spherical_plan(build, false);
                if (pair_keys(parallel) != pair_keys(serial)) 
                    throw std::runtime_error("Parallel screening changed pair set.");
                evaluate_one_electron_matrices_sph_into(build, parallel, single_use);
                worst = std::max(worst, compare(single_use, reference));
                evaluate_one_electron_matrices_sph_into(build, moved, actual);
                evaluate_one_electron_matrices_sph_into(build, full, single_use);
                worst = std::max(worst, compare(actual, reference));
                worst = std::max(worst, compare(single_use, reference));
                evaluate_one_electron_matrices_into(build, parallel, cart_actual);
                worst = std::max(worst, compare(cart_actual, cart_reference));
                poison(cart_actual);
                evaluate_one_electron_matrices_into(build, moved, cart_actual);
                worst = std::max(worst, compare(cart_actual, cart_reference));
                evaluate_one_electron_matrices_into(build, full, cart_actual);
                worst = std::max(worst, compare(cart_actual, cart_reference));
                // A reused output must overwrite the old matrices completely.
                std::fill(actual.overlap.values.begin(), actual.overlap.values.end(), 123.0);
                std::fill(actual.kinetic.values.begin(), actual.kinetic.values.end(), -234.0);
                std::fill(actual.nuclear.values.begin(), actual.nuclear.values.end(), 345.0);
                evaluate_one_electron_matrices_sph_into(build, moved, actual);
                worst = std::max(worst, compare(actual, reference));
            }
            // Reusable plans cache basis geometry, but never nuclear integrals.
            auto changed = build;
            for (std::size_t i = 0; i < changed.nuclear_centers.size(); ++i) {
                changed.nuclear_centers[i][1] += .20;
                if (i % 2) changed.nuclear_charges[i] *= -1;
            }
            evaluate_one_electron_matrices_sph_into(changed, serial, reference);
            evaluate_one_electron_matrices_sph_into(changed, moved, actual);
            worst = std::max(worst, compare(actual, reference));
            const auto changed_cart = evaluate_one_electron_matrices(changed, serial);
            evaluate_one_electron_matrices_into(changed, moved, cart_actual);
            worst = std::max(worst, compare(cart_actual, changed_cart));
            worst = std::max(worst, compare(transform_cart(changed_cart, serial), reference));
        }
        std::cout << "PASS: full/cached, screening on/off, "
                     "plans built/evaluated at 1/2/3 threads, moved plan, repeated outputs, changed nuclear potential, Cartesian/spherical consistency, screening error; max error=" << worst << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n'; return 1;
    }
}
