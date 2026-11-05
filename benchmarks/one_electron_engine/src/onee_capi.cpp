// C ABI wrapper around the TLNO one-electron engine.
//
// Purpose: let the Python benchmark harness time TLNO through the *same*
// ctypes/perf_counter path it uses for libcint, with output matrices
// pre-allocated outside the timed loop for both engines.  This removes the
// "two different harnesses" confound (the C++ binary vs Python timing) so the
// libcint-vs-TLNO ratio reflects integral work, not interpreter overhead.
//
// A handle owns the basis build result, the (reused) plan, and a (reused)
// output matrix set.  The timed entry points (tlno_onee_eval_full /
// tlno_onee_eval_cached) write into the handle's pre-allocated matrices and
// return nothing to copy; tlno_onee_copy() exposes the result for validation
// outside the timed region.

#include "benchmarks/common/molecule_io.hpp"
#include "core/basis/basis_general.hpp"
#include "integrals/one_electron/onee_matrices.hpp"

#include <cmath>
#include <cstring>
#include <memory>
#include <string>

namespace {

struct TlnoOneeHandle {
    tlno::core::basis::BasisBuildResult br;
    tlno::integrals::onee::OneElectronPlan plan;
    tlno::integrals::onee::OneElectronMatrices out;       // Cartesian S/T/V
    tlno::integrals::onee::OneElectronMatrices out_sph;   // spherical S/T/V
    bool has_plan = false;
};

} // namespace

extern "C" {

// Build the molecule + basis once (mirrors PySCF's gto.M, done outside timing).
// Returns an opaque handle, or nullptr on failure.
void* tlno_onee_create_from_inp(const char* inp_path, const char* basis_name) {
    try {
        if (!inp_path || !basis_name) return nullptr;
        auto handle = std::make_unique<TlnoOneeHandle>();
        const tlno::benchmarks::onee::MoleculeInput mol = 
            tlno::benchmarks::onee::read_inp_file(inp_path);
        tlno::core::molecule::MoleculeSpec spec;
        spec.symbols = mol.symbols;
        spec.coords = mol.coords_bohr;
        spec.charge = mol.charge;
        const tlno::core::basis::BasisKind bk =
            tlno::core::basis::parse_basis_kind(basis_name);
        handle->br = tlno::core::basis::build_basis_general(spec, bk);
        handle->out = tlno::integrals::onee::OneElectronMatrices(
            static_cast<int>(handle->br.basis.size()));
        return handle.release();
    } catch (...) {
        return nullptr;
    }
}

int tlno_onee_nbf(void* handle) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    return h ? static_cast<int>(h->br.basis.size()) : -1;
}

// Spherical AO count. The layout is basis-only (screening-independent), so this
// builds the plan once on demand if needed and can be queried before any eval.
int tlno_onee_nbf_sph(void* handle) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return -1;
    if (!h->has_plan) {
        h->plan = tlno::integrals::onee::build_one_electron_spherical_plan(h->br);
        h->has_plan = true;
    }
    return h->plan.n_sph;
}

// Global screening toggle (1 = production screening on, 0 = fully off).
void tlno_onee_set_screening(int enabled) {
    tlno::integrals::onee::set_onee_screening_enabled(enabled != 0);
}

int tlno_onee_screening_enabled() {
    return tlno::integrals::onee::onee_screening_enabled() ? 1 : 0;
}

// Runtime group-pair Gaussian-decay threshold override. Call after
// tlno_onee_set_screening(1), which resets it to the compiled default.
void tlno_onee_set_group_pair_decay_eps(double eps) {
    tlno::integrals::onee::set_onee_group_pair_decay_eps(eps);
}

double tlno_onee_group_pair_decay_eps() {
    return tlno::integrals::onee::onee_group_pair_decay_eps();
}

// Runtime nuclear primitive-pair threshold override for isolated sweep studies.
// Unlike the group-pair threshold, this takes effect immediately during eval.
void tlno_onee_set_nuclear_screen_eps(double eps) {
    tlno::integrals::onee::set_onee_nuclear_screen_eps(eps);
}

double tlno_onee_nuclear_screen_eps() {
    return tlno::integrals::onee::get_onee_nuclear_screen_eps();
}

// Build the plan once for repeated integral evaluations (the cached scope).
void tlno_onee_prepare_plan(void* handle) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return;
    h->plan = tlno::integrals::onee::build_one_electron_plan(h->br,true);
    h->has_plan = true;
}

// Timed entry, "full" scope: rebuild the plan (groups + SoA pair scalars +
// screening) and evaluate.  Mirrors libcint full scope (make_cintopt + driver).
void tlno_onee_eval_full(void* handle) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return;
    h->plan = tlno::integrals::onee::build_one_electron_plan(h->br, false);
    h->has_plan = true;
    tlno::integrals::onee::evaluate_one_electron_matrices_into(h->br, h->plan, h->out);
}

// Timed entry, "cached" scope: evaluate with the pre-built plan.  Mirrors
// libcint cached scope (reused CINTOpt).
void tlno_onee_eval_cached(void* handle) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h || !h->has_plan) return;
    tlno::integrals::onee::evaluate_one_electron_matrices_into(h->br, h->plan, h->out);
}

// Spherical counterparts transform contracted Cartesian blocks before scatter.
// Both integral evaluation and projection/scatter belong to the timed call.
void tlno_onee_eval_full_sph(void* handle) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return;
    h->plan = tlno::integrals::onee::build_one_electron_spherical_plan(h->br, false);
    h->has_plan = true;
    tlno::integrals::onee::evaluate_one_electron_matrices_sph_into(h->br, h->plan, h->out_sph);
}

void tlno_onee_eval_cached_sph(void* handle) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h || !h->has_plan) return;
    // Cartesian and spherical evaluation share the same immutable pair plan.
    tlno::integrals::onee::evaluate_one_electron_matrices_sph_into(h->br, h->plan, h->out_sph);
}

// Copy the current S/T/V (row-major, n*n each) into caller buffers.  For
// validation only; call outside the timed region.
void tlno_onee_copy(void* handle, double* S, double* T, double* V) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return;
    const std::size_t n2 = h->out.overlap.values.size();
    if (S) std::memcpy(S, h->out.overlap.values.data(), n2 * sizeof(double));
    if (T) std::memcpy(T, h->out.kinetic.values.data(), n2 * sizeof(double));
    if (V) std::memcpy(V, h->out.nuclear.values.data(), n2 * sizeof(double));
}

// Copy the current spherical S/T/V (row-major, n_sph*n_sph each).  Validation
// only; requires a prior eval_*_sph call.  Call outside the timed region.
void tlno_onee_copy_sph(void* handle, double* S, double* T, double* V) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return;
    const std::size_t n2 = h->out_sph.overlap.values.size();
    if (S) std::memcpy(S, h->out_sph.overlap.values.data(), n2 * sizeof(double));
    if (T) std::memcpy(T, h->out_sph.kinetic.values.data(), n2 * sizeof(double));
    if (V) std::memcpy(V, h->out_sph.nuclear.values.data(), n2 * sizeof(double));
}

// Shell-group-pair work accounting from the current plan: number of groups,
// total possible pairs (G*(G+1)/2), and pairs actually kept after screening.
// kept == total_possible when screening is disabled.  Requires a prior
// eval/prepare call to have populated the plan.
void tlno_onee_pair_stats(void* handle, int* n_groups, int* total_possible,
                          int* kept) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return;
    const int g = static_cast<int>(h->plan.groups.size());
    if (n_groups) *n_groups = g;
    if (total_possible) *total_possible = g * (g + 1) / 2;
    if (kept) *kept = static_cast<int>(h->plan.group_pairs.size());
}

// Primitive-pair accounting for the nuclear kernel at the active threshold.
// Counts only group pairs whose plan enables V.
void tlno_onee_nuclear_pair_stats(void* handle, long long* total, 
                                  long long* kept) {
    auto* h = static_cast<TlnoOneeHandle*>(handle);
    if (!h) return;
    double z_sum = 0.0;
    for (int z : h->br.nuclear_charges) {
        z_sum += std::abs(static_cast<double>(z));
    }
    const double eps = tlno::integrals::onee::get_onee_nuclear_screen_eps();
    const double threshold = z_sum > 0.0 ? eps / z_sum : 0.0;
    long long total_count = 0;
    long long kept_count = 0;
    for (std::size_t i = 0; i < h->plan.group_pairs.size(); ++i) {
        if (!h->plan.group_pairs[i].do_v) continue;
        const auto& spp = h->plan.pair_prims[i];
        const long long count =
            static_cast<long long>(spp.n_prim_a) * spp.n_prim_b;
        total_count += count;
        for (long long k = 0; k < count; ++k) {
            if (spp.screen[static_cast<std::size_t>(k)] >= threshold) {
                ++kept_count;
            }
        }
    }
    if (total) *total = total_count;
    if (kept) *kept = kept_count;
}

// Empty call to measure the Python+ctypes per-call overhead floor.
void tlno_onee_noop(void* handle) {
    (void)handle;
}

void tlno_onee_destroy(void* handle) {
    delete static_cast<TlnoOneeHandle*>(handle);
}

} // extern "C"
