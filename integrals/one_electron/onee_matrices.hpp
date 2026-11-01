#ifndef TLNO_INTEGRALS_ONE_ELECTRON_ONEE_MATRICES_HPP
#define TLNO_INTEGRALS_ONE_ELECTRON_ONEE_MATRICES_HPP

#include "onee_integrals.hpp"

namespace tlno::integrals::onee {

// One-electron output bundle (S, T, V). 1e-specific, so it lives with the one-electron engine.
namespace basis = tlno::core::basis;

struct OneElectronMatrices {
	basis::DenseMatrix overlap;
	basis::DenseMatrix kinetic;
	basis::DenseMatrix nuclear;

	explicit OneElectronMatrices(int n = 0) : overlap(n), kinetic(n), nuclear(n) {}
};

struct OneElectronGroupPair {
	int ga = 0;
	int gb = 0;
	bool do_st = true;
	bool do_v = true;
};

struct OneElectronPlan {
	std::vector<ShellGroup> groups;
	std::vector<OneElectronGroupPair> group_pairs;
	// Precomputed SoA primitive-pair scalars, parallel to group_pairs.
	std::vector<ShellPairPrim> pair_prims;
	// Bounded extra storage for reusable nuclear Hermite projectors.
	std::size_t nuclear_hermite_cache_bytes = 0;
	// Spherical-harmonic AO layout (built once, used by the spherical path). Each group bundles one 
    // or more shells of momentum L (chunks of cart_count(L) components); sph_base[g][chunk] 
    // is the spherical AO base of that shell, assigned in Cartesian-AO order so the spherical 
    // matrix matches the conventional (PySCF) AO ordering. n_sph is the total spherical AO count.
	int n_sph = 0;
	std::vector<std::vector<int>> sph_base;
};

OneElectronPlan build_one_electron_plan(const basis::BasisBuildResult& build);
// Set false for a plan used only once: constructing the Hermite cache pays
// off on reuse, but adds unnecessary allocation to a single evaluation.
OneElectronPlan build_one_electron_plan(
	const basis::BasisBuildResult& build, bool cache_nuclear_hermite);
// The same pair plan serves Cartesian and spherical output; spherical layout
// is prepared once and each contracted block is transformed before scatter.
OneElectronPlan build_one_electron_spherical_plan(
	const basis::BasisBuildResult& build, bool cache_nuclear_hermite = true);
OneElectronMatrices build_one_electron_spherical_matrices(const basis::BasisBuildResult& build);
OneElectronMatrices evaluate_one_electron_matrices(
	const basis::BasisBuildResult& build,
	const OneElectronPlan& plan);
// Evaluate into a caller-owned matrix set (resized/zeroed as needed) so tight
// benchmark / SCF-reuse loops avoid reallocating the output every call.
void evaluate_one_electron_matrices_into(
	const basis::BasisBuildResult& build,
	const OneElectronPlan& plan,
	OneElectronMatrices& mats);
// Spherical output: transform each contracted Cartesian shell block before
// scatter. Identity s/p classes keep their specialized kernels. Screened-away
// group pairs perform neither integral nor transformation work.
void evaluate_one_electron_matrices_sph_into(
	const basis::BasisBuildResult& build,
	const OneElectronPlan& plan,
	OneElectronMatrices& mats);
OneElectronMatrices build_one_electron_matrices(const basis::BasisBuildResult& build);

// Toggle all one-electron screening (group-pair Gaussian decay + nuclear
// primitive estimate) on or off. Benchmarks use this to separate raw kernel
// speed from screening gains.
void set_onee_screening_enabled(bool enabled);
bool onee_screening_enabled();

// Override the group-pair Gaussian-decay threshold at runtime. Note that
// set_onee_screening_enabled(true) resets it to the compiled default, so call
// this after enabling. Takes effect at the next plan build.
void set_onee_group_pair_decay_eps(double eps);
double onee_group_pair_decay_eps();

} // namespace tlno::integrals::onee

#endif
