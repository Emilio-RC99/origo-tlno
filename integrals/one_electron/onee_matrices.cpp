#include "onee_matrices.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "onee_integrals.hpp"
#include "../../core/math/general_math.hpp"
#include "../../core/shell/cart2sph.hpp"

#ifndef TLNO_ONEE_HERMITE_CACHE_BYTES
#define TLNO_ONEE_HERMITE_CACHE_BYTES (16 * 1024 * 1024)
#endif
static_assert(TLNO_ONEE_HERMITE_CACHE_BYTES >= 0, "negative Hermite cache budget");

namespace tlno::integrals::onee {

namespace basis = tlno::core::basis;
namespace shell = tlno::core::shell;

namespace {

constexpr double kPrimitiveMinAbsOverlapST = 0.0;
constexpr double kGroupPairMinAbsOverlapST = 0.0;
constexpr double kGroupPairDecayEps = 1.0e-15;

struct OneEScreeningConfig {
    double primitive_min_abs_overlap_st = kPrimitiveMinAbsOverlapST;
    double group_pair_min_abs_overlap_st = kGroupPairMinAbsOverlapST;
    double group_pair_decay_eps = kGroupPairDecayEps;
};

OneEScreeningConfig& mutable_onee_screening_config() {
    static OneEScreeningConfig config;
    return config;
}

const OneEScreeningConfig& get_onee_screening_config() {
    return mutable_onee_screening_config();
}

std::vector<ShellGroup> build_shell_groups(const basis::BasisBuildResult& build) {
    std::vector<ShellGroup> groups;
    const auto& basis = build.basis;
    for (int idx = 0; idx < static_cast<int>(basis.size()); ++idx) {
        const basis::Shell& sh = basis[static_cast<std::size_t>(idx)];
        const int L = sh.ang[0] + sh.ang[1] + sh.ang[2];
        bool found = false;
        for (auto& grp : groups) {
            if (grp.L != L) continue;
            if (grp.center != sh.center) continue;
            if (grp.exponents.size() != sh.primitives.size()) continue;
            bool exp_ok = true;
            for (int k = 0; k < static_cast<int>(sh.primitives.size()); ++k) {
                if (grp.exponents[static_cast<std::size_t>(k)] != sh.primitives[static_cast<std::size_t>(k)].exponent) {
                    exp_ok = false;
                    break;
                }
            }
            if (!exp_ok) continue;
            grp.basis_indices.push_back(idx);
            grp.ang.push_back(sh.ang);
            std::vector<double> ccs;
            for (const auto& pr : sh.primitives) {
                ccs.push_back(pr.coefficient);
            }
            grp.coeffs.push_back(std::move(ccs));
            found = true;
            break;
        }
        if (!found) {
            ShellGroup grp;
            grp.center = sh.center;
            grp.L = L;
            grp.basis_indices.push_back(idx);
            grp.ang.push_back(sh.ang);
            for (const auto& pr : sh.primitives) {
                grp.exponents.push_back(pr.exponent);
            }
            std::vector<double> ccs;
            for (const auto& pr : sh.primitives) {
                ccs.push_back(pr.coefficient);
            }
            grp.coeffs.push_back(std::move(ccs));
            groups.push_back(std::move(grp));
        }
    }
    return groups;
}

std::vector<OneElectronGroupPair> build_group_pairs(const std::vector<ShellGroup>& groups,
    const OneEScreeningConfig& screening) {
    const int n_groups = static_cast<int>(groups.size());

    struct GroupDecayScale {
        double alpha_min = std::numeric_limits<double>::max();
        double K = 0.0;
    };

    std::vector<GroupDecayScale> decay_scales(static_cast<std::size_t>(n_groups));
    if (screening.group_pair_decay_eps > 0.0) {
        for (int g = 0; g < n_groups; ++g) {
            const auto& exps = groups[static_cast<std::size_t>(g)].exponents;
            const auto& c0 = groups[static_cast<std::size_t>(g)].coeffs[0];
            for (std::size_t i = 0; i < exps.size(); ++i) {
                decay_scales[static_cast<std::size_t>(g)].alpha_min =
                    std::min(decay_scales[static_cast<std::size_t>(g)].alpha_min, exps[i]);
                decay_scales[static_cast<std::size_t>(g)].K +=
                    std::abs(c0[i]) * std::pow(basis::kPi / exps[i], 0.75);
            }
        }
    }

    const std::size_t total_pairs =
        static_cast<std::size_t>(n_groups) * static_cast<std::size_t>(n_groups + 1) / 2;
    const bool need_pair_bound =
        screening.group_pair_decay_eps > 0.0 ||
        screening.group_pair_min_abs_overlap_st > 0.0;

    // Each pair's screening decision is independent, so evaluate the bounds in
    // parallel into a flat flags table (k = ga(ga+1)/2 + gb), then compact
    // serially so the kept-pair order stays identical to the serial loop.
    constexpr std::uint8_t kKeep = 1, kDoSt = 2, kDoV = 4;
    std::vector<std::uint8_t> flags(total_pairs, 0);

#pragma omp parallel for schedule(dynamic, 8) if (n_groups >= 32)
    for (int ga = 0; ga < n_groups; ++ga) {
        const std::size_t row0 =
            static_cast<std::size_t>(ga) * static_cast<std::size_t>(ga + 1) / 2;
        for (int gb = 0; gb <= ga; ++gb) {
            double AB2 = 0.0;
            if (need_pair_bound) {
                AB2 = shell::dist2(groups[static_cast<std::size_t>(ga)].center,
                            groups[static_cast<std::size_t>(gb)].center);
            }

            bool keep = true;
            bool do_st = true;
            double bound = 0.0;
            if (screening.group_pair_min_abs_overlap_st > 0.0 ||
                screening.group_pair_decay_eps > 0.0) {
                if (screening.group_pair_decay_eps > 0.0) {
                    const GroupDecayScale& na =
                        decay_scales[static_cast<std::size_t>(ga)];
                    const GroupDecayScale& nb =
                        decay_scales[static_cast<std::size_t>(gb)];
                    const double mu_min = na.alpha_min * nb.alpha_min / (na.alpha_min + nb.alpha_min);
                    const double decay_estimate =
                        na.K * nb.K * std::exp(-mu_min * AB2);
                    if (decay_estimate < screening.group_pair_decay_eps) {
                        keep = false;
                    }
                }

                if (keep && need_pair_bound) {
                    const auto& ga_exp = groups[static_cast<std::size_t>(ga)].exponents;
                    const auto& gb_exp = groups[static_cast<std::size_t>(gb)].exponents;
                    const auto& ga_c0 = groups[static_cast<std::size_t>(ga)].coeffs[0];
                    const auto& gb_c0 = groups[static_cast<std::size_t>(gb)].coeffs[0];
                    for (int ai = 0; ai < static_cast<int>(ga_exp.size()); ++ai) {
                        for (int bi = 0; bi < static_cast<int>(gb_exp.size()); ++bi) {
                            const double p = ga_exp[static_cast<std::size_t>(ai)] + gb_exp[static_cast<std::size_t>(bi)];
                            const double chi = ga_exp[static_cast<std::size_t>(ai)] * gb_exp[static_cast<std::size_t>(bi)] / p;
                            bound += std::abs(ga_c0[static_cast<std::size_t>(ai)] * gb_c0[static_cast<std::size_t>(bi)]) *
                                     std::pow(basis::kPi / p, 1.5) * std::exp(-chi * AB2);
                        }
                    }
                }
                if (keep) {
                    do_st = bound >= screening.group_pair_min_abs_overlap_st;
                }
            }

            if (keep) {
                const bool do_v =
                    screening.group_pair_decay_eps <= 0.0 ||
                    bound >= screening.group_pair_decay_eps;
                flags[row0 + static_cast<std::size_t>(gb)] =
                    static_cast<std::uint8_t>(kKeep | (do_st ? kDoSt : 0) | (do_v ? kDoV : 0));
            }
        }
    }

    std::vector<OneElectronGroupPair> gp_list;
    gp_list.reserve(total_pairs);
    for (int ga = 0; ga < n_groups; ++ga) {
        const std::size_t row0 =
            static_cast<std::size_t>(ga) * static_cast<std::size_t>(ga + 1) / 2;
        for (int gb = 0; gb <= ga; ++gb) {
            const std::uint8_t f = flags[row0 + static_cast<std::size_t>(gb)];
            if (!(f & kKeep)) continue;
            OneElectronGroupPair gp;
            gp.ga = ga;
            gp.gb = gb;
            // The one-electron recurrences transfer/extend momentum on the
            // right (B) shell.  Orient mixed-L pairs with the higher-L group
            // on the left so the right-hand table dimension is the smaller
            // one.  Screening above remains on the original unordered pair;
            // equal-L pairs keep their existing deterministic order.
            if (groups[static_cast<std::size_t>(gp.gb)].L >
                groups[static_cast<std::size_t>(gp.ga)].L) {
                std::swap(gp.ga, gp.gb);
            }
            gp.do_st = (f & kDoSt) != 0;
            gp.do_v = (f & kDoV) != 0;
            gp_list.push_back(gp);
        }
    }

    return gp_list;
}

inline void store_upper(basis::DenseMatrix& matrix, int row, int col, double value) {
    if (row <= col) {
        matrix(row, col) = value;
    } else {
        matrix(col, row) = value;
    }
}

inline void eval_group_pair_sph(
    const basis::BasisBuildResult& build,
    const OneElectronPlan& plan,
    const OneEScreeningConfig& screening,
    int gp_idx,
    std::vector<double>& ov_blk,
    std::vector<double>& ki_blk,
    std::vector<double>& nuc_blk,
    OneElectronMatrices& mats) {

    const OneElectronGroupPair& gp = plan.group_pairs[static_cast<std::size_t>(gp_idx)];
    const ShellGroup& ga_grp = plan.groups[static_cast<std::size_t>(gp.ga)];
    const ShellGroup& gb_grp = plan.groups[static_cast<std::size_t>(gp.gb)];
    const ShellPairPrim& spp = plan.pair_prims[static_cast<std::size_t>(gp_idx)];
    const int n_a = static_cast<int>(ga_grp.ang.size());
    const int n_b = static_cast<int>(gb_grp.ang.size());
    const std::size_t blk_size = static_cast<std::size_t>(n_a * n_b);

    if (ov_blk.size() < blk_size) {
        ov_blk.resize(blk_size);
        ki_blk.resize(blk_size);
        nuc_blk.resize(blk_size);
    }
    if (gp.do_st) {
        std::fill(ov_blk.begin(), ov_blk.begin() + static_cast<std::ptrdiff_t>(blk_size), 0.0);
        std::fill(ki_blk.begin(), ki_blk.begin() + static_cast<std::ptrdiff_t>(blk_size), 0.0);
        contracted_shell_block_overlap_kinetic(
            ga_grp, gb_grp, spp, ov_blk.data(), ki_blk.data(),
            screening.primitive_min_abs_overlap_st);
    }
    if (gp.do_v) {
        std::fill(nuc_blk.begin(), nuc_blk.begin() + static_cast<std::ptrdiff_t>(blk_size), 0.0);
        contracted_shell_block_nuclear(
            ga_grp, gb_grp, spp, nuc_blk.data(),
            build.nuclear_charges, build.nuclear_centers);
    }

    const int La = ga_grp.L, Lb = gb_grp.L;
    const int nca = shell::cart_count(La), ncb = shell::cart_count(Lb);
    const int nsa = shell::sph_count(La), nsb = shell::sph_count(Lb);
    const int chunks_a = n_a / nca, chunks_b = n_b / ncb;
    const std::vector<int>& sb_a = plan.sph_base[static_cast<std::size_t>(gp.ga)];
    const std::vector<int>& sb_b = plan.sph_base[static_cast<std::size_t>(gp.gb)];
    double scratch[shell::cart_count(shell::kCart2SphMaxL) * shell::sph_count(shell::kCart2SphMaxL)];
    double sblk[shell::sph_count(shell::kCart2SphMaxL) * shell::sph_count(shell::kCart2SphMaxL)];
    // s/p blocks (L<2) are identity under cart->sph, and they dominate the block
    // count; scatter them straight through (as the Cartesian path does) instead
    // of paying the two-pass transform machinery.
    const bool identity = (La < 2 && Lb < 2);

    auto transform_store = [&](const std::vector<double>& blk, basis::DenseMatrix& out) {
        for (int ja = 0; ja < chunks_a; ++ja) {
            const int ca0 = ja * nca;
            const int base_a = sb_a[static_cast<std::size_t>(ja)];
            for (int jb = 0; jb < chunks_b; ++jb) {
                const int cb0 = jb * ncb;
                const int base_b = sb_b[static_cast<std::size_t>(jb)];
                if (identity) {
                    for (int i = 0; i < nca; ++i) {
                        const double* row = blk.data() + (ca0 + i) * n_b + cb0;
                        for (int j = 0; j < ncb; ++j) {
                            store_upper(out, base_a + i, base_b + j, row[j]);
                        }
                    }
                    continue;
                }
                shell::cart2sph_block(La, Lb, blk.data() + ca0 * n_b + cb0, n_b, sblk, scratch);
                for (int sa = 0; sa < nsa; ++sa) {
                    for (int sb = 0; sb < nsb; ++sb) {
                        store_upper(out, base_a + sa, base_b + sb, sblk[sa * nsb + sb]);
                    }
                }
            }
        }
    };

    if (gp.do_st) {
        transform_store(ov_blk, mats.overlap);
        transform_store(ki_blk, mats.kinetic);
    }
    if (gp.do_v) {
        transform_store(nuc_blk, mats.nuclear);
    }
}

template <bool DoST, bool DoV>
inline void scatter_group_pair_cart(const ShellGroup& ga_grp,
    const ShellGroup& gb_grp, bool diagonal_group,
    int n_a, int n_b,
    const std::vector<double>& ov_blk,
    const std::vector<double>& ki_blk,
    const std::vector<double>& nuc_blk,
    OneElectronMatrices& mats) {
    const std::size_t stride = static_cast<std::size_t>(mats.overlap.size);
    double* const ov = mats.overlap.values.data();
    double* const ki = mats.kinetic.values.data();
    double* const nu = mats.nuclear.values.data();

    const auto scatter_entry = [&](int ca, int cb) {
        const int ia = ga_grp.basis_indices[static_cast<std::size_t>(ca)];
        const int ib = gb_grp.basis_indices[static_cast<std::size_t>(cb)];
        const std::size_t ab =
            static_cast<std::size_t>(ia) * stride + static_cast<std::size_t>(ib);
        const std::size_t ba =
            static_cast<std::size_t>(ib) * stride + static_cast<std::size_t>(ia);
        const std::size_t src = static_cast<std::size_t>(ca * n_b + cb);
        if constexpr (DoST) {
            const double s = ov_blk[src];
            const double t = ki_blk[src];
            ov[ab] = s;
            ov[ba] = s;
            ki[ab] = t;
            ki[ba] = t;
        }
        if constexpr (DoV) {
            const double v = nuc_blk[src];
            nu[ab] = v;
            nu[ba] = v;
        }
    };

    const auto scatter_diagonal = [&](int c) {
        const int i = ga_grp.basis_indices[static_cast<std::size_t>(c)];
        const std::size_t out =
            static_cast<std::size_t>(i) * stride + static_cast<std::size_t>(i);
        const std::size_t src = static_cast<std::size_t>(c * n_b + c);
        if constexpr (DoST) {
            ov[out] = ov_blk[src];
            ki[out] = ki_blk[src];
        }
        if constexpr (DoV) {
            nu[out] = nuc_blk[src];
        }
    };

    if (!diagonal_group) {
        // Each unordered off-diagonal group pair occurs once in the plan, so
        // this thread exclusively owns both symmetric output rectangles.
        for (int ca = 0; ca < n_a; ++ca) {
            for (int cb = 0; cb < n_b; ++cb) {
                scatter_entry(ca, cb);
            }
        }
        return;
    }

    // A diagonal group block contains both orders of every AO pair.  The old
    // row-major store_upper loop left the ca > cb value in the upper triangle;
    // use that same half here, but write each symmetric destination only once.
    for (int ca = 0; ca < n_a; ++ca) {
        for (int cb = 0; cb < ca; ++cb) {
            scatter_entry(ca, cb);
        }
        scatter_diagonal(ca);
    }
}

inline void scatter_group_pair_cart(
    const OneElectronGroupPair& gp,
    const ShellGroup& ga_grp,
    const ShellGroup& gb_grp,
    int n_a,
    int n_b,
    const std::vector<double>& ov_blk,
    const std::vector<double>& ki_blk,
    const std::vector<double>& nuc_blk,
    OneElectronMatrices& mats) {
    const bool diagonal_group = gp.ga == gp.gb;
    if (gp.do_st) {
        if (gp.do_v) {
            scatter_group_pair_cart<true, true>(
                ga_grp, gb_grp, diagonal_group, n_a, n_b,
                ov_blk, ki_blk, nuc_blk, mats);
        } else {
            scatter_group_pair_cart<true, false>(
                ga_grp, gb_grp, diagonal_group, n_a, n_b,
                ov_blk, ki_blk, nuc_blk, mats);
        }
    } else if (gp.do_v) {
        scatter_group_pair_cart<false, true>(
            ga_grp, gb_grp, diagonal_group, n_a, n_b,
            ov_blk, ki_blk, nuc_blk, mats);
    }
}

} // namespace

OneElectronPlan build_one_electron_plan(const basis::BasisBuildResult& build) {
    return build_one_electron_plan(build, true);
}

OneElectronPlan build_one_electron_plan(
    const basis::BasisBuildResult& build, bool cache_nuclear_hermite) {
    OneElectronPlan plan;
    plan.groups = build_shell_groups(build);
    plan.group_pairs = build_group_pairs(plan.groups, get_onee_screening_config());
    // Precompute reusable SoA primitive-pair scalars so evaluation avoids
    // repeated transcendental operations and divisions.  Pairs are independent,
    // so build them in parallel (indexed writes keep the order deterministic).
    const int n_pairs = static_cast<int>(plan.group_pairs.size());
    plan.pair_prims.resize(plan.group_pairs.size());
    // As in the retained two-electron projector cache, preparation belongs to
    // the reusable plan. A deterministic global cap and 8 KiB per-pair cap
    // bound the extra memory independently of the number of OpenMP workers.
    constexpr std::size_t cache_budget = TLNO_ONEE_HERMITE_CACHE_BYTES;
    std::vector<unsigned char> cache_pair;
    if (cache_nuclear_hermite && cache_budget > 0) {
        cache_pair.assign(plan.group_pairs.size(), 0);
        for (std::size_t i=0; i<plan.group_pairs.size(); ++i) {
            const auto& gp = plan.group_pairs[i];
            const auto& a = plan.groups[gp.ga];
            const auto& b = plan.groups[gp.gb];
            if (!gp.do_v || a.L+b.L == 0) continue;
            const std::size_t bytes = 3*sizeof(double)*a.exponents.size()*b.exponents.size()*
                static_cast<std::size_t>(a.L+1)*(b.L+1)*(a.L+b.L+1);
            if (bytes <= 8192 && bytes <= cache_budget-plan.nuclear_hermite_cache_bytes) {
                cache_pair[i] = 1;
                plan.nuclear_hermite_cache_bytes += bytes;
            }
        }
    }
#pragma omp parallel for schedule(dynamic, 16) if (n_pairs >= 64)
    for (int i = 0; i < n_pairs; ++i) {
        const OneElectronGroupPair& gp = plan.group_pairs[static_cast<std::size_t>(i)];
        plan.pair_prims[static_cast<std::size_t>(i)] = build_shell_pair_prim(
            plan.groups[static_cast<std::size_t>(gp.ga)],
            plan.groups[static_cast<std::size_t>(gp.gb)]);
        if (!cache_pair.empty() && cache_pair[static_cast<std::size_t>(i)]) {
            cache_shell_pair_nucleus_hermite(plan.groups[gp.ga], plan.groups[gp.gb],
                plan.pair_prims[static_cast<std::size_t>(i)]);
        }
    }

    // Spherical AO layout. A group bundles one or more shells of the same L
    // (chunks of cart_count(L) Cartesian components, in canonical order). Each
    // shell maps to sph_count(L) spherical AOs; assign their bases in ascending
    // Cartesian-AO order (the first component's global index) so the spherical
    // matrix reproduces the conventional (PySCF) AO ordering.
    struct ShellRef { int cart_start; int group; int chunk; int L; };
    std::vector<ShellRef> shells;
    plan.sph_base.resize(plan.groups.size());
    for (int g = 0; g < static_cast<int>(plan.groups.size()); ++g) {
        const ShellGroup& grp = plan.groups[static_cast<std::size_t>(g)];
        const int nc = shell::cart_count(grp.L);
        const int n_chunks = static_cast<int>(grp.ang.size()) / nc;
        plan.sph_base[static_cast<std::size_t>(g)].assign(static_cast<std::size_t>(n_chunks), 0);
        for (int j = 0; j < n_chunks; ++j) {
            shells.push_back({grp.basis_indices[static_cast<std::size_t>(j * nc)], g, j, grp.L});
        }
    }
    std::sort(shells.begin(), shells.end(),
              [](const ShellRef& a, const ShellRef& b) { return a.cart_start < b.cart_start; });
    int sph_off = 0;
    for (const ShellRef& sh : shells) {
        plan.sph_base[static_cast<std::size_t>(sh.group)][static_cast<std::size_t>(sh.chunk)] = sph_off;
        sph_off += shell::sph_count(sh.L);
    }
    plan.n_sph = sph_off;
    return plan;
}

void evaluate_one_electron_matrices_into(
    const basis::BasisBuildResult& build,
    const OneElectronPlan& plan,
    OneElectronMatrices& mats) {
    const int n_bf = static_cast<int>(build.basis.size());
    if (mats.overlap.size != n_bf) {
        mats = OneElectronMatrices(n_bf);
    } else {
        // Zero all three matrices with one chunked parallel pass; the serial
        // fills were a fixed O(N^2) cost that did not shrink with threads.
        double* const ov = mats.overlap.values.data();
        double* const ki = mats.kinetic.values.data();
        double* const nu = mats.nuclear.values.data();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (n_bf >= 384)
#endif
        for (int row = 0; row < n_bf; ++row) {
            const std::size_t off = static_cast<std::size_t>(row) * static_cast<std::size_t>(n_bf);
            const std::size_t len = static_cast<std::size_t>(n_bf);
            std::fill(ov + off, ov + off + len, 0.0);
            std::fill(ki + off, ki + off + len, 0.0);
            std::fill(nu + off, nu + off + len, 0.0);
        }
    }
    const auto& screening = get_onee_screening_config();

#pragma omp parallel if (static_cast<int>(plan.group_pairs.size()) >= 16)
    {
        thread_local std::vector<double> ov_blk, ki_blk, nuc_blk;

#pragma omp for schedule(dynamic, 4)
        for (int gp_idx = 0; gp_idx < static_cast<int>(plan.group_pairs.size()); ++gp_idx) {
            const OneElectronGroupPair& gp = plan.group_pairs[static_cast<std::size_t>(gp_idx)];
            const ShellGroup& ga_grp = plan.groups[static_cast<std::size_t>(gp.ga)];
            const ShellGroup& gb_grp = plan.groups[static_cast<std::size_t>(gp.gb)];
            const ShellPairPrim& spp = plan.pair_prims[static_cast<std::size_t>(gp_idx)];
            const int n_a = static_cast<int>(ga_grp.ang.size());
            const int n_b = static_cast<int>(gb_grp.ang.size());
            const std::size_t blk_size = static_cast<std::size_t>(n_a * n_b);

            if (ov_blk.size() < blk_size) {
                ov_blk.resize(blk_size);
                ki_blk.resize(blk_size);
                nuc_blk.resize(blk_size);
            }

            if (gp.do_st) {
                std::fill(ov_blk.begin(), ov_blk.begin() + static_cast<std::ptrdiff_t>(blk_size), 0.0);
                std::fill(ki_blk.begin(), ki_blk.begin() + static_cast<std::ptrdiff_t>(blk_size), 0.0);
                contracted_shell_block_overlap_kinetic(
                    ga_grp, gb_grp, spp, ov_blk.data(), ki_blk.data(),
                    screening.primitive_min_abs_overlap_st);
            }
            if (gp.do_v) {
                std::fill(nuc_blk.begin(), nuc_blk.begin() + static_cast<std::ptrdiff_t>(blk_size), 0.0);
                contracted_shell_block_nuclear(
                    ga_grp, gb_grp, spp, nuc_blk.data(),
                    build.nuclear_charges, build.nuclear_centers);
            }

            scatter_group_pair_cart(
                gp, ga_grp, gb_grp, n_a, n_b,
                ov_blk, ki_blk, nuc_blk, mats);
        }
    }
}

void evaluate_one_electron_matrices_sph_into(
    const basis::BasisBuildResult& build,
    const OneElectronPlan& plan,
    OneElectronMatrices& mats) {
    const int n_sph = plan.n_sph;
    const int n_pairs = static_cast<int>(plan.group_pairs.size());
    if (mats.overlap.size != n_sph) mats = OneElectronMatrices(n_sph);
    double* const ov = mats.overlap.values.data();
    double* const ki = mats.kinetic.values.data();
    double* const nu = mats.nuclear.values.data();
    const auto& screening = get_onee_screening_config();
    const auto clear_row = [&](int row) {
        const std::size_t off=static_cast<std::size_t>(row)*n_sph;
        std::fill_n(ov+off,n_sph,0.0);
        std::fill_n(ki+off,n_sph,0.0);
        std::fill_n(nu+off,n_sph,0.0);
    };
    const auto mirror_row = [&](double* values,int row) {
        const std::size_t off=static_cast<std::size_t>(row)*n_sph;
        for (int col=0;col<row;++col) {
            const std::size_t from=static_cast<std::size_t>(col)*n_sph+row;
            values[off+col]=values[from];
        }
    };
#ifdef _OPENMP
    if (n_pairs>=16 && omp_get_max_threads()>1) {
        // One team owns initialization, pair evaluation and all three mirrors.
        // Barriers separate zeroing from scatter and scatter from mirroring.
#pragma omp parallel
        {
            thread_local std::vector<double> ov_blk,ki_blk,nuc_blk;
#pragma omp for schedule(static)
            for (int row=0;row<n_sph;++row) clear_row(row);
#pragma omp for schedule(dynamic,4)
            for (int gp_idx=0;gp_idx<n_pairs;++gp_idx)
                eval_group_pair_sph(build,plan,screening,gp_idx,ov_blk,ki_blk,nuc_blk,mats);
            // Complete each operator while its strided reads remain in cache.
#pragma omp for schedule(static,32)
            for (int row=0;row<n_sph;++row) mirror_row(ov,row);
#pragma omp for schedule(static,32)
            for (int row=0;row<n_sph;++row) mirror_row(ki,row);
#pragma omp for schedule(static,32) nowait
            for (int row=0;row<n_sph;++row) mirror_row(nu,row);
        }
        return;
    }
#endif
    thread_local std::vector<double> ov_blk,ki_blk,nuc_blk;
    for (int row=0;row<n_sph;++row) clear_row(row);
    for (int gp_idx=0;gp_idx<n_pairs;++gp_idx)
        eval_group_pair_sph(build,plan,screening,gp_idx,ov_blk,ki_blk,nuc_blk,mats);
    for (double* values:{ov,ki,nu})
        for (int row=0;row<n_sph;++row) mirror_row(values,row);
}

OneElectronMatrices evaluate_one_electron_matrices(
    const basis::BasisBuildResult& build,
    const OneElectronPlan& plan) {
    OneElectronMatrices mats(static_cast<int>(build.basis.size()));
    evaluate_one_electron_matrices_into(build, plan, mats);
    return mats;
}

OneElectronPlan build_one_electron_spherical_plan(
    const basis::BasisBuildResult& build,bool cache_nuclear_hermite) {
    return build_one_electron_plan(build,cache_nuclear_hermite);
}

OneElectronMatrices build_one_electron_spherical_matrices(const basis::BasisBuildResult& build) {
    const auto plan=build_one_electron_spherical_plan(build,false);
    OneElectronMatrices mats(plan.n_sph);
    evaluate_one_electron_matrices_sph_into(build,plan,mats);
    return mats;
}

OneElectronMatrices build_one_electron_matrices(const basis::BasisBuildResult& build) {
    const OneElectronPlan plan = build_one_electron_plan(build, false);
    return evaluate_one_electron_matrices(build, plan);
}

void set_onee_screening_enabled(bool enabled) {
    OneEScreeningConfig& cfg = mutable_onee_screening_config();
    cfg.group_pair_decay_eps = enabled ? kGroupPairDecayEps : 0.0;
    cfg.group_pair_min_abs_overlap_st = enabled ? kGroupPairMinAbsOverlapST : 0.0;
    cfg.primitive_min_abs_overlap_st = enabled ? kPrimitiveMinAbsOverlapST : 0.0;
    set_onee_nuclear_screen_eps(enabled ? kNuclearScreenEps : 0.0);
}

bool onee_screening_enabled() {
    return mutable_onee_screening_config().group_pair_decay_eps > 0.0;
}

void set_onee_group_pair_decay_eps(double eps) {
    mutable_onee_screening_config().group_pair_decay_eps = eps;
}

double onee_group_pair_decay_eps() {
    return mutable_onee_screening_config().group_pair_decay_eps;
}

} //namespace tlno::integrals::onee
