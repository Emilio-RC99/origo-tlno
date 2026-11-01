#include "onee_integrals.hpp"

#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <vector>

#include "../../core/math/general_math.hpp"
#include "../../core/simd/packet.hpp"

// Nuclear-attraction SIMD lane count (doubles per vector register). Overridable
// at compile time, e.g. -DTLNO_NUCLEAR_LANES = 1 for the lane ablation study.
#ifndef TLNO_NUCLEAR_LANES
#define TLNO_NUCLEAR_LANES tlno::core::simd::lanes
#endif

namespace tlno::integrals::onee {

namespace math = tlno::core::math;

namespace {

#if defined(__AVX512F__) && defined(__FMA__)
template<int W> struct NuclearPacketWidth { static constexpr int value = W; };
#endif

// Writes the VRR/HRR shape table directly into `out`, which must hold
// (la_max + lb_max + 1) * (lb_max + 1) doubles: the recursion needs bra rows
// up to la_max + lb_max as workspace, and every cell the caller reads
// (ia <= la_max + lb_max - ib) is written by the recursion, so no zero-fill
// or copy-out pass is needed.
void build_shape_table_1d(int la_max, int lb_max, double PA, double AB, double inv2p, 
double* out) {
    const int Ltot_max = la_max + lb_max;
    const int ncol = lb_max + 1;
    out[0] = 1.0;
    if (Ltot_max >= 1) out[static_cast<std::size_t>(ncol)] = PA;
    for (int n = 1; n < Ltot_max; ++n) {
        out[static_cast<std::size_t>(n + 1) * static_cast<std::size_t>(ncol)] = 
            PA * out[static_cast<std::size_t>(n) * static_cast<std::size_t>(ncol)] + 
            (static_cast<std::size_t>(n) * inv2p) * out[static_cast<std::size_t>(n - 1) * static_cast<std::size_t>(ncol)];
    }
    for (int ib = 0; ib < lb_max; ++ib) {
        const int imax = Ltot_max - ib - 1;
        for (int ia = 0; ia <= imax; ++ia) {
            out[static_cast<std::size_t>(ia) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(ib + 1)] = 
                out[static_cast<std::size_t>(ia + 1) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(ib)] + 
                AB * out[static_cast<std::size_t>(ia) * static_cast<std::size_t>(ncol) + static_cast<std::size_t>(ib)];
        }
    }
}

void build_full_hermite_table_1d(int la_max, int lb_max, double qx, 
                                  double alpha, double beta, double* out) {
    const double p     = alpha + beta;
    const double chi   = alpha * beta / p;
    const double PA    = - beta * qx / p;
    const double PB    = alpha * qx / p;
    const double inv2p = 0.5 / p;
    const int ltot     = la_max + lb_max;
    const int nt       = ltot + 1;
    const int col      = lb_max + 1;

    thread_local std::vector<double> W;
    const std::size_t wsz = static_cast<std::size_t>(ltot + 1) * col * nt;
    if (W.size() < wsz) W.assign(wsz, 0.0);
    else std::fill(W.begin(), W.begin() + static_cast<std::ptrdiff_t>(wsz), 0.0); 

    const auto widx = [&](int ia, int ib, int t) -> std::size_t {
        return static_cast<std::size_t>(ia * col + ib) * nt + t;
    };

    W[widx(0, 0, 0)] = std::exp(- chi * qx * qx);

    for (int ia = 0; ia < ltot; ++ia) {
        for (int t = 0; t <= ia + 1; ++t) {
            double v = PA * W[widx(ia, 0, t)];
            if (t > 0) v += inv2p * W[widx(ia, 0, t - 1)];
            if (t < ia) v += (t + 1.0) * W[widx(ia, 0, t + 1)];
            W[widx(ia + 1, 0, t)] = v;
        }
    }

    for (int ib = 0; ib < lb_max; ++ib) {
        for (int ia = 0; ia <= ltot; ++ia) {
            const int t_max = std::min(ia + ib + 1, ltot);
            for (int t = 0; t <= t_max; ++t) {
                double v = PB * W[widx(ia, ib, t)];
                if (t > 0) v += inv2p * W[widx(ia, ib, t - 1)];
                if (t < ia + ib) v += (t + 1.0) * W[widx(ia, ib, t + 1)];
                W[widx(ia, ib + 1, t)] = v;
            }
        }
    }

    for (int ia = 0; ia <= la_max; ++ia) {
        for (int ib = 0; ib <= lb_max; ++ib) {
            for (int t = 0; t <= ltot; ++t) {
                out[static_cast<std::size_t>(ia * col + ib) * nt + t] = W[widx(ia, ib, t)];
            }
        }
    }
}

// Seeded variant of build_full_hermite_table_1d: takes the precomputed Gaussian
// seed e0 = exp(-chi*qx^2) and per-pair scalars so repeated evaluation performs
// no exp() and no division. All other math is identical.
void build_full_hermite_table_1d_seed(int la_max, int lb_max, double qx, 
                                      double alpha, double beta, double p, 
                                      double inv2p, double e0_seed, double* out) {
    const double inv_p = inv2p + inv2p;      // = 1/p, avoids a division
    const double PA    = - beta * qx * inv_p;
    const double PB   =  alpha * qx * inv_p;
    const int ltot    = la_max + lb_max;
    const int nt      = ltot + 1;
    const int col     = lb_max + 1;

    thread_local std::vector<double> W;
    const std::size_t wsz = static_cast<std::size_t>(ltot + 1) * col * nt;
    if (W.size() < wsz) W.assign(wsz, 0.0);
    else std::fill(W.begin(), W.begin() + static_cast<std::ptrdiff_t>(wsz), 0.0);

    const auto widx = [&](int ia, int ib, int t) -> std::size_t {
        return static_cast<std::size_t>(ia * col + ib) * nt + t;
    };

    W[widx(0, 0, 0)] = e0_seed;

    for (int ia = 0; ia < ltot; ++ia) {
        for (int t = 0; t <= ia + 1; ++t) {
            double v = PA * W[widx(ia, 0, t)];
            if (t > 0) v += inv2p * W[widx(ia, 0, t - 1)];
            if (t < ia) v += (t + 1.0) * W[widx(ia, 0, t + 1)];
            W[widx(ia + 1, 0, t)] = v;
        }
    }

    for (int ib = 0; ib < lb_max; ++ib) {
        for (int ia = 0; ia <= ltot; ++ia) {
            const int t_max = std::min(ia + ib + 1, ltot);
            for (int t = 0; t <= t_max; ++t) {
                double v = PB * W[widx(ia, ib, t)];
                if (t > 0) v += inv2p * W[widx(ia, ib, t - 1)];
                if (t < ia + ib) v += (t + 1.0) * W[widx(ia, ib, t + 1)];
                W[widx(ia, ib + 1, t)] = v;
            } 
        }
    }

    for (int ia = 0; ia <= la_max; ++ia) {
        for (int ib = 0; ib <= lb_max; ++ib) {
            for (int t = 0; t <= ltot; ++t) {
                out[static_cast<std::size_t>(ia * col + ib) * nt + t] = W[widx(ia, ib, t)];
            }
        }
    }
}

} // namespace

// Precompute all primitive-pair scalars for one group pair (SoA layout).
// Index: k = ia * n_prim_b + ib.
ShellPairPrim build_shell_pair_prim(const ShellGroup& group_a, const ShellGroup& group_b) {
    ShellPairPrim spp;
    const int n_prim_a = static_cast<int>(group_a.exponents.size());
    const int n_prim_b = static_cast<int>(group_b.exponents.size());
    spp.n_prim_a = n_prim_a;
    spp.n_prim_b = n_prim_b;

    const double Ax = group_a.center[0], Ay = group_a.center[1], Az = group_a.center[2];
    const double Bx = group_b.center[0], By = group_b.center[1], Bz = group_b.center[2];
    spp.ABx = Ax - Bx;
    spp.ABy = Ay - By;
    spp.ABz = Az - Bz;
    spp.AB2 = spp.ABx * spp.ABx + spp.ABy * spp.ABy + spp.ABz * spp.ABz;
    const double AB2 = spp.AB2;

    const std::size_t npp = 
        static_cast<std::size_t>(n_prim_a) * static_cast<std::size_t>(n_prim_b);
    // One arena allocation for all 18 SoA sections, each padded to 8 doubles.
    const std::size_t sec = (npp + 7) & ~static_cast<std::size_t>(7);
    spp.storage.assign(18 * sec, 0.0);
    double* base = spp.storage.data();
    spp.p             = base;               spp.inv2p = base + sec;
    spp.chi           = base + 2 * sec;     spp.sss   = base + 3 * sec;
    spp.expmChiAB2    = base + 4 * sec;
    spp.alpha         = base + 5 * sec;     spp.beta  = base + 6 * sec; 
    spp.PAx           = base + 7 * sec;     spp.PAy   = base + 8 * sec;
    spp.PAz           = base + 9 * sec;
    spp.Px            = base + 10 * sec;    spp.Py    = base + 11 * sec;
    spp.Pz            = base + 12 * sec;
    spp.e0x           = base + 13 * sec;    spp.e0y   = base + 14 * sec;
    spp.e0z           = base + 15 * sec;
    spp.neg2pi_over_p = base + 16 * sec;
    spp.screen        = base + 17 * sec;

    // Max |contraction coeff| over angular components per primitive (for the
    // nuclear screening bound).  thread_local: this runs inside the parallel
    // plan build and small per-call vectors were a malloc hot spot.
    thread_local std::vector<double> maxca_t1, maxcb_t1;
    maxca_t1.assign(static_cast<std::size_t>(n_prim_a), 0.0);
    maxcb_t1.assign(static_cast<std::size_t>(n_prim_b), 0.0);
    std::vector<double>& maxca = maxca_t1;
    std::vector<double>& maxcb = maxcb_t1;
    for (const auto& crow : group_a.coeffs) 
        for (int ia = 0; ia < n_prim_a; ++ia) 
            maxca[static_cast<std::size_t>(ia)] = 
                std::max(maxca[static_cast<std::size_t>(ia)], std::abs(crow[static_cast<std::size_t>(ia)]));
    for (const auto& crow : group_b.coeffs) 
        for (int ib = 0; ib < n_prim_b; ++ib) 
            maxcb[static_cast<std::size_t>(ib)] = 
                std::max(maxcb[static_cast<std::size_t>(ib)], std::abs(crow[static_cast<std::size_t>(ib)]));

    for (int ia = 0; ia < n_prim_a; ++ia) {
        const double alpha = group_a.exponents[static_cast<std::size_t>(ia)];
        for (int ib = 0; ib < n_prim_b; ++ib) {
            const double beta = group_b.exponents[static_cast<std::size_t>(ib)];
            const std::size_t k = 
                static_cast<std::size_t>(ia) * static_cast<std::size_t>(n_prim_b) +
                static_cast<std::size_t>(ib);

            const double p = alpha + beta;
            const double inv_p = 1.0 / p;
            const double chi = alpha * beta * inv_p; 
            const double emc = std::exp(- chi * AB2);
            const double pi_over_p = tlno::core::basis::kPi * inv_p;

            spp.p[k]          =  p;
            spp.inv2p[k]      =  0.5 * inv_p;
            spp.chi[k]        = chi;
            spp.alpha[k]      = alpha;
            spp.beta[k]       = beta;
            spp.expmChiAB2[k] = emc;
            spp.sss[k]        = pi_over_p * std::sqrt(pi_over_p) * emc;

            const double Px = (alpha * Ax + beta * Bx) * inv_p;
            const double Py = (alpha * Ay + beta * By) * inv_p;
            const double Pz = (alpha * Az + beta * Bz) * inv_p;
            spp.Px[k] = Px;      spp.Py[k] = Py;      spp.Pz[k] = Pz;
            spp.PAx[k] = Px - Ax;
            spp.PAy[k] = Py - Ay;
            spp.PAz[k] = Pz - Az;

            // Per-axis Hermite seeds E^0_{0,0} = exp(-chi * (A_axis - B_axis)^2).
            spp.e0x[k] = std::exp(-chi * spp.ABx * spp.ABx);
            spp.e0y[k] = std::exp(-chi * spp.ABy * spp.ABy);
            spp.e0z[k] = std::exp(-chi * spp.ABz * spp.ABz);
            spp.neg2pi_over_p[k] = -2.0 * pi_over_p;

            // Nuclear s-s monopole estimate (F_0 <= 1). For s-s components the
            // contribution of this primitive pair is bounded by
            // |neg2pi_over_p| * exp(-chi AB^2) * max|c_a| * max|c_b| * Z_sum.
            spp.screen[k] = (2.0 * pi_over_p) * emc *
                            maxca[static_cast<std::size_t>(ia)] *
                            maxcb[static_cast<std::size_t>(ib)];
        }
    }
    return spp;
}

void cache_shell_pair_nucleus_hermite(
    const ShellGroup& group_a, const ShellGroup& group_b, ShellPairPrim& spp) {
    const int la = group_a.L, lb = group_b.L;
    if (la + lb == 0) return;
    const std::size_t axis_size = static_cast<std::size_t> (la + 1) * (lb + 1) * (la + lb + 1);
    const std::size_t npp = static_cast<std::size_t>(spp.n_prim_a)*spp.n_prim_b;
    spp.nuclear_hermite.resize(3 * axis_size * npp);
    for (std::size_t k = 0; k < npp; ++k) {
        double* dst = spp.nuclear_hermite.data() + 3 * axis_size * k;
        build_full_hermite_table_1d_seed(la, lb, spp.ABx, spp.alpha[k], spp.beta[k], 
            spp.p[k], spp.inv2p[k], spp.e0x[k], dst);
        build_full_hermite_table_1d_seed(la, lb, spp.ABy, spp.alpha[k], spp.beta[k], 
            spp.p[k], spp.inv2p[k], spp.e0y[k], dst + axis_size);
        build_full_hermite_table_1d_seed(la, lb, spp.ABz, spp.alpha[k], spp.beta[k], 
            spp.p[k], spp.inv2p[k], spp.e0z[k], dst + 2 * axis_size);
    }
}

// Angular-momentum-specialized overlap + kinetic kernel.  Same compile-time
// specialization idea as the nuclear kernel: with la_max/lb_max as template
// constants the shape-table builds and the OS contraction unroll per (la,lb).
// LA_MAX_T = LB_MAX_T = -1 keeps the original runtime bounds (generic fallback).
template <int LA_MAX_T, int LB_MAX_T>
static void overlap_kinetic_impl(
    const ShellGroup& group_a,
    const ShellGroup& group_b,
    const ShellPairPrim& spp,
    double* overlap_block,
    double* kinetic_block,
    double min_abs_overlap) {

    const int n_a = static_cast<int>(group_a.ang.size());
    const int n_b = static_cast<int>(group_b.ang.size());
    const int la_max = (LA_MAX_T < 0) ? group_a.L : LA_MAX_T;
    const int lb_max = (LB_MAX_T < 0) ? group_b.L : LB_MAX_T;

    // Extended ket dimension needed for the kinetic lb + 2 term.
    const int lb_ext = lb_max + 2;
    const int ncol = lb_ext + 1;   // columns in each shape table

    const double AB2 = spp.AB2;
    const double ABx = spp.ABx;
    const double ABy = spp.ABy;
    const double ABz = spp.ABz;

    // Shape tables: Sx[ia*(lb_ext+1) + ib] = S1D_x(ia, ib).  Sized for the
    // direct-write recursion, which uses bra rows up to la_max + lb_ext.
    // thread_local avoids per-group-pair heap allocation; sized to max needed on first use.
    thread_local std::vector<double> Sx, Sy, Sz;
    const std::size_t table_size = static_cast<std::size_t>(la_max + lb_ext + 1) * static_cast<std::size_t>(ncol);
    if (Sx.size() < table_size) { Sx.resize(table_size); }
    if (Sy.size() < table_size) { Sy.resize(table_size); }
    if (Sz.size() < table_size) { Sz.resize(table_size); }

    const int n_prim_a = spp.n_prim_a;
    const int n_prim_b = spp.n_prim_b;

    // Fast path for pure s-type shell groups.
    // This removes the 1D-table build and the tensor-product contraction.
    if (la_max == 0 && lb_max == 0) {
        for (int ia_prim = 0; ia_prim < n_prim_a; ++ia_prim) {
            for (int ib_prim = 0; ib_prim < n_prim_b; ++ib_prim) {
                const std::size_t k = 
                    static_cast<std::size_t>(ia_prim) * static_cast<std::size_t>(n_prim_b) +
                    static_cast<std::size_t>(ib_prim);
                const double overlap_prim = spp.sss[k];

                if (min_abs_overlap > 0.0 && std::abs(overlap_prim) < min_abs_overlap) {
                    continue;
                }

                const double chi = spp.chi[k];
                const double kinetic_prim = overlap_prim * chi * (3.0 - 2.0 * chi * AB2);
                const double w_base = overlap_prim;

                for (int ca = 0; ca < n_a; ++ca) {
                    const double d_a = group_a.coeffs[static_cast<std::size_t>(ca)][static_cast<std::size_t>(ia_prim)];
                    for (int cb = 0; cb < n_b; ++cb) {
                        const double d_b = group_b.coeffs[static_cast<std::size_t>(cb)][static_cast<std::size_t>(ib_prim)];
                        const std::size_t block_index = static_cast<std::size_t>(ca * n_b + cb);
                        const double w = d_a * d_b * w_base;
                        overlap_block[block_index] += w;
                        kinetic_block[block_index] += d_a * d_b * kinetic_prim;
                    }
                }
            }
        }
        return;
    }

    for (int ia_prim = 0; ia_prim < n_prim_a; ++ia_prim) {
        for (int ib_prim = 0; ib_prim < n_prim_b; ++ib_prim) {
            const std::size_t k = 
                static_cast<std::size_t>(ia_prim) * static_cast<std::size_t>(n_prim_b) + 
                static_cast<std::size_t>(ib_prim);
            const double beta     = spp.beta[k]; 
            const double inv2p    = spp.inv2p[k];
            const double sss_base = spp.sss[k];

            // Optional screening: skip pair if the ss overlap (all-component bound) is tiny.
            if (min_abs_overlap > 0.0 && std::abs(sss_base) < min_abs_overlap) {
                continue;
            }

            // Build shape tables for each axis (VRR + HRR), extended to lb+2.
            build_shape_table_1d(la_max, lb_ext, spp.PAx[k], ABx, inv2p, Sx.data());
            build_shape_table_1d(la_max, lb_ext, spp.PAy[k], ABy, inv2p, Sy.data());
            build_shape_table_1d(la_max, lb_ext, spp.PAz[k], ABz, inv2p, Sz.data());

            // Accumulate into the block for all component pairs.
            for (int ca = 0; ca < n_a; ++ca) {
                const double d_a = group_a.coeffs[static_cast<std::size_t>(ca)][static_cast<std::size_t>(ia_prim)];
                const int la = group_a.ang[static_cast<std::size_t>(ca)][0];
                const int ma = group_a.ang[static_cast<std::size_t>(ca)][1];
                const int na = group_a.ang[static_cast<std::size_t>(ca)][2];

                for (int cb = 0; cb < n_b; ++cb) {
                    const double d_b = group_b.coeffs[static_cast<std::size_t>(cb)][static_cast<std::size_t>(ib_prim)];
                    const double w   = d_a * d_b * sss_base;
                    const int lb = group_b.ang[static_cast<std::size_t>(cb)][0];
                    const int mb = group_b.ang[static_cast<std::size_t>(cb)][1];
                    const int nb = group_b.ang[static_cast<std::size_t>(cb)][2];

                    const double sx = Sx[static_cast<std::size_t>(la * ncol + lb)];
                    const double sy = Sy[static_cast<std::size_t>(ma * ncol + mb)];
                    const double sz = Sz[static_cast<std::size_t>(na * ncol + nb)];

                    overlap_block[static_cast<std::size_t>(ca * n_b + cb)] += w * sx * sy * sz;

                    // Kinetic 1-D factors via OS formula:
                    // T_x(la,lb) = β(2lb+1)S(la,lb) - 2β²S(la,lb+2) - ½lb(lb-1)S(la,lb-2)
                    const double sx_p2 = Sx[static_cast<std::size_t>(la * ncol + (lb + 2))];
                    const double sx_m2 = (lb >= 2) ? Sx[static_cast<std::size_t>(la * ncol + (lb - 2))] : 0.0;
                    const double tx = beta * (2.0 * lb + 1.0) * sx
                                    - 2.0 * beta * beta * sx_p2
                                    - 0.5 * static_cast<double>(lb * (lb - 1)) * sx_m2;

                    const double sy_p2 = Sy[static_cast<std::size_t>(ma * ncol + (mb + 2))];
                    const double sy_m2 = (mb >= 2) ? Sy[static_cast<std::size_t>(ma * ncol + (mb - 2))] : 0.0;
                    const double ty = beta * (2.0 * mb + 1.0) * sy
                                    - 2.0 * beta * beta * sy_p2
                                    - 0.5 * static_cast<double>(mb * (mb - 1)) * sy_m2;

                    const double sz_p2 = Sz[static_cast<std::size_t>(na * ncol + (nb + 2))];
                    const double sz_m2 = (nb >= 2) ? Sz[static_cast<std::size_t>(na * ncol + (nb - 2))] : 0.0;
                    const double tz = beta * (2.0 * nb + 1.0) * sz
                                    - 2.0 * beta * beta * sz_p2
                                    - 0.5 * static_cast<double>(nb * (nb - 1)) * sz_m2;

                    kinetic_block[static_cast<std::size_t>(ca * n_b + cb)] +=
                        w * (tx * sy * sz + sx * ty * sz + sx * sy * tz);
                }
            }
        }
    }
}

// Public entry: dispatch to the angular-momentum-specialized overlap+kinetic
// instantiation (s,p,d,f); higher L uses the runtime generic instantiation.
void contracted_shell_block_overlap_kinetic(
    const ShellGroup& group_a,
    const ShellGroup& group_b,
    const ShellPairPrim& spp,
    double* overlap_block, 
    double* kinetic_block, 
    double min_abs_overlap) {

    const int la_max = group_a.L;
    const int lb_max = group_b.L;

#define TLNO_ST_CASE(LA, LB) \ 
        case (LA) * 4 + (LB): \
            overlap_kinetic_impl<LA, LB>(group_a, group_b, spp, overlap_block, kinetic_block, min_abs_overlap); \
            return;

        if (la_max <= 3 && lb_max <= 3) {
            switch (la_max * 4 + lb_max) {
                TLNO_ST_CASE(0, 0) TLNO_ST_CASE(0, 1) TLNO_ST_CASE(0, 2) TLNO_ST_CASE(0, 3)
                TLNO_ST_CASE(1, 0) TLNO_ST_CASE(1, 1) TLNO_ST_CASE(1, 2) TLNO_ST_CASE(1, 3)
                TLNO_ST_CASE(2, 0) TLNO_ST_CASE(2, 1) TLNO_ST_CASE(2, 2) TLNO_ST_CASE(2, 3)
                TLNO_ST_CASE(3, 0) TLNO_ST_CASE(3, 1) TLNO_ST_CASE(3, 2) TLNO_ST_CASE(3, 3)
                default: break;
            }
        }
    #undef TLNO_ST_CASE

        overlap_kinetic_impl<-1, -1>(group_a, group_b, spp, overlap_block, kinetic_block, min_abs_overlap);
}

// ── Shell-block nuclear attraction ───────────────────────────────────────────
// Like contracted_shell_block_overlap_kinetic but for V.
// Key optimisation: for each (primitive pair, nucleus), the Boys function and
// R_{tuv}^{(n)} table are computed ONCE and distributed to all (n_a * n_b)
// component combinations — eliminating the per-AO-pair Boys redundancy.
//
// boys_r000_all   : computes all needed (-2p)^n F_n(x) in one erf + sqrt call.
// build_full_hermite_table_1d : Hermite E^t_{ia,ib} for all (ia,ib,t).
// Bounded MD recursion        : skips R_{tuv}^{(n)} entries never used.

// Nuclear primitive-pair screening tolerance. A primitive pair is skipped when
// its s-s monopole bound/estimate (spp.screen[k] * Z_sum) falls below this value.
// It is rigorous for s-s components; higher-L safety is established by the
// no-screen matrix validation. Set 6+ orders of magnitude below the 2e-8
// validation tolerance so accumulated dropped contributions remain negligible
// over the validated suite.
// Runtime-adjustable (default kNuclearScreenEps) via set_onee_nuclear_screen_eps()
// so that benchmarks can set it to 0 to disable nuclear screening and isolate the
// raw kernel cost from the work that screening saves.
double g_onee_nuclear_screen_eps = kNuclearScreenEps;

// Angular-momentum-specialized nuclear kernel.  When LA_MAX_T/LB_MAX_T >= 0 the
// shell angular momenta are compile-time constants, so ltot/max_n and every
// loop bound and table stride below fold to constants and the Hermite build,
// the R_{tuv} MD recursion and the final contraction unroll per (la,lb).  Passing
// LA_MAX_T = LB_MAX_T = -1 keeps the original runtime bounds (generic fallback).
template <int LA_MAX_T, int LB_MAX_T>
static void nuclear_general_impl(
    const ShellGroup& group_a,
    const ShellGroup& group_b,
    const ShellPairPrim& spp,
    double* nuclear_block,
    const std::vector<int>& nuclear_charges,
    const std::vector<std::array<double, 3>>& nuclear_centers) {

    const int n_a    = static_cast<int>(group_a.ang.size());
    const int n_b    = static_cast<int>(group_b.ang.size());
    const int la_max = (LA_MAX_T < 0) ? group_a.L : LA_MAX_T;
    const int lb_max = (LB_MAX_T < 0) ? group_b.L : LB_MAX_T;
    const int ltot   = la_max + lb_max;   // max Hermite index per axis
    // For Cartesian shell groups, la+ma+na = L_a for every component, so
    // tx+ty+tz = (la+lb)+(ma+mb)+(na+nb) = L_a+L_b = ltot for all (ca,cb).
    const int max_n  = ltot;
    const int n_K    = static_cast<int>(nuclear_centers.size());

    // Hermite table layout: out[(ia*(lb_max+1)+ib)*(ltot+1)+t]
    const int Et_nt  = ltot + 1;
    const int Et_col = lb_max + 1;
    const int Et_sz  = (la_max + 1) * Et_col * Et_nt;

    thread_local std::vector<double> Etx_tl, Ety_tl, Etz_tl;
    if (static_cast<int>(Etx_tl.size()) < Et_sz) {
        Etx_tl.resize(Et_sz); Ety_tl.resize(Et_sz); Etz_tl.resize(Et_sz);
    }
    const double* Etx = Etx_tl.data();
    const double* Ety = Ety_tl.data();
    const double* Etz = Etz_tl.data();

    // R_{tuv}^{(n)} has KL independent charge-center lanes sharing one
    // primitive pair. Explicit packets keep the recurrence at the selected
    // ISA width even when the compiler prefers narrower automatic vectors.
    constexpr int KL = TLNO_NUCLEAR_LANES;
    static_assert(KL == 1 || KL == 2 || KL == 4 || KL == 8, "invalid nuclear width");
    using V = tlno::core::simd::Packet<KL>;
    // Only t+u+v <= L is live. At order n every dependency belongs to
    // n+1, so two compact tetrahedral slices suffice. This transfers the
    // bounded radial-state layout used by the improved two-electron producer.
    const int Rsum_sz = (ltot+1)*(ltot+2)*(ltot+3)/6;
    const auto ridx = [=](int t, int u, int v) {
        const int m = ltot-t+1;
        return Rsum_sz - m*(m+1)*(m+2)/6 + u*m-u*(u-1)/2+v;
    };
    const auto roffset = [=](int t, int u, int v, int n) -> std::size_t {
        return static_cast<std::size_t>((n&1)*Rsum_sz+ridx(t,u,v))*KL;
    };
    thread_local std::vector<double> R_tl, boys_tl, Rsum_tl;
    if (static_cast<int>(R_tl.size()) < 2*Rsum_sz*KL) R_tl.resize(2*Rsum_sz*KL);
    if (static_cast<int>(boys_tl.size()) < (max_n+1)*KL) boys_tl.resize((max_n+1)*KL);
    if (static_cast<int>(Rsum_tl.size()) < Rsum_sz) Rsum_tl.resize(Rsum_sz);
    double* R_buf = R_tl.data();
    double* boys_lane = boys_tl.data();
    // Charge contraction still precedes the component-pair Hermite projection.
    double* R_sum = Rsum_tl.data();

    const double ABx = spp.ABx;
    const double ABy = spp.ABy;
    const double ABz = spp.ABz;

    const int n_prim_a = spp.n_prim_a;
    const int n_prim_b = spp.n_prim_b;

    // Primitive-pair screening threshold: skip pair k when spp.screen[k] < screen_thr,
    // i.e. when screen[k] * Z_sum < kNuclearScreenEps.  Z_sum is molecule-global, so
    // fold it into the threshold once here (one division) instead of per pair.
    double z_sum = 0.0;
    for (int K = 0; K < n_K; ++K) z_sum += std::abs(static_cast<double>(nuclear_charges[static_cast<std::size_t>(K)]));
    const double screen_thr = (z_sum > 0.0) ? (g_onee_nuclear_screen_eps / z_sum) : 0.0;

    // Fast path for pure s-type shell groups.
    // For s-s shells the nuclear integral reduces to a simple Boys-F0 sum.
    if (la_max == 0 && lb_max == 0) {
        for (int ia_prim = 0; ia_prim < n_prim_a; ++ia_prim) {
            for (int ib_prim = 0; ib_prim < n_prim_b; ++ib_prim) {
                const std::size_t k =
                    static_cast<std::size_t>(ia_prim) * static_cast<std::size_t>(n_prim_b) +
                    static_cast<std::size_t>(ib_prim);
                if (spp.screen[k] < screen_thr) continue;
                const double p = spp.p[k];
                const double pref = spp.neg2pi_over_p[k] * spp.expmChiAB2[k];

                const double Px = spp.Px[k];
                const double Py = spp.Py[k];
                const double Pz = spp.Pz[k];

                double nuc_sum = 0.0;
#if defined(__AVX512F__) && defined(__FMA__)
                if constexpr (KL == 8) {
                    auto process_packet = [&](auto width_tag, int K0) {
                        constexpr int PW = decltype(width_tag)::value;
                        double Zl[PW], xs[PW], f0[PW];unsigned active_mask=0;
                        for (int l = 0; l < PW; ++l) {
                            const int K = K0 + l;
                            if (K < n_K && nuclear_charges[static_cast<std::size_t>(K)] != 0) {
                                active_mask|=1u<<l;
                                Zl[l] = static_cast<double>(nuclear_charges[static_cast<std::size_t>(K)]);
                                const double dx = Px - nuclear_centers[static_cast<std::size_t>(K)][0];
                                const double dy = Py - nuclear_centers[static_cast<std::size_t>(K)][1];
                                const double dz = Pz - nuclear_centers[static_cast<std::size_t>(K)][2];
                                xs[l] = p * (dx * dx + dy * dy + dz * dz);
                            } else {
                                Zl[l] = 0.0;
                                xs[l] = 1.0;
                            }
                        }
                        math::boys_r000_all_batch_masked<PW>(0, xs, p, f0,active_mask);
                        for (int l = 0; l < PW; ++l) nuc_sum += Zl[l] * f0[l];
                    };
                    const int tail = n_K % KL;
                    const int wide_end = n_K - (tail <= 4 ? tail : 0);
                    int K0 = 0;
                    for (; K0 < wide_end; K0 += KL)
                        process_packet(NuclearPacketWidth<8>{}, K0);
                    if (K0 < n_K)
                        process_packet(NuclearPacketWidth<4>{}, K0);
                } else {
#endif
                for (int K0 = 0; K0 < n_K; K0 += KL) {
                    double Zl[KL], xs[KL], f0[KL];unsigned active_mask=0;
                    for (int l = 0; l < KL; ++l) {
                        const int K = K0 + l;
                        if (K < n_K && nuclear_charges[static_cast<std::size_t>(K)] != 0) {
                            active_mask|=1u<<l;
                            Zl[l] = static_cast<double>(nuclear_charges[static_cast<std::size_t>(K)]);
                            const double dx = Px - nuclear_centers[static_cast<std::size_t>(K)][0];
                            const double dy = Py - nuclear_centers[static_cast<std::size_t>(K)][1];
                            const double dz = Pz - nuclear_centers[static_cast<std::size_t>(K)][2];
                            xs[l] = p * (dx * dx + dy * dy + dz * dz);
                        } else {
                            Zl[l] = 0.0;
                            xs[l] = 1.0;
                        }
                    }
                    math::boys_r000_all_batch_masked<KL>(0, xs, p, f0,active_mask);
                    for (int l = 0; l < KL; ++l) nuc_sum += Zl[l] * f0[l];
                }
#if defined(__AVX512F__) && defined(__FMA__)
                }
#endif

                const double v_prim = pref * nuc_sum;
                for (int ca = 0; ca < n_a; ++ca) {
                    const double d_a = group_a.coeffs[static_cast<std::size_t>(ca)][static_cast<std::size_t>(ia_prim)];
                    for (int cb = 0; cb < n_b; ++cb) {
                        const double d_b = group_b.coeffs[static_cast<std::size_t>(cb)][static_cast<std::size_t>(ib_prim)];
                        const std::size_t block_index = static_cast<std::size_t>(ca * n_b + cb);
                        nuclear_block[block_index] += d_a * d_b * v_prim;
                    }
                }
            }
        }
        return;
    }

    for (int ia_prim = 0; ia_prim < n_prim_a; ++ia_prim) {
        for (int ib_prim = 0; ib_prim < n_prim_b; ++ib_prim) {
            const std::size_t k =
                static_cast<std::size_t>(ia_prim) * static_cast<std::size_t>(n_prim_b) +
                static_cast<std::size_t>(ib_prim);
            if (spp.screen[k] < screen_thr) continue;
            const double alpha = spp.alpha[k];
            const double beta  = spp.beta[k];
            const double p     = spp.p[k];
            const double inv2p = spp.inv2p[k];

            const double Px = spp.Px[k];
            const double Py = spp.Py[k];
            const double Pz = spp.Pz[k];
            const double neg2pi_over_p = spp.neg2pi_over_p[k];

            // Reuse the plan-owned projectors when this pair fits its bounded
            // cache budget. Otherwise retain the same seeded construction.
            if (!spp.nuclear_hermite.empty()) {
                Etx = spp.nuclear_hermite.data()+3*k*Et_sz;
                Ety = Etx+Et_sz;
                Etz = Ety+Et_sz;
            } else {
                build_full_hermite_table_1d_seed(la_max, lb_max, ABx, alpha, beta, p, inv2p, spp.e0x[k], Etx_tl.data());
                build_full_hermite_table_1d_seed(la_max, lb_max, ABy, alpha, beta, p, inv2p, spp.e0y[k], Ety_tl.data());
                build_full_hermite_table_1d_seed(la_max, lb_max, ABz, alpha, beta, p, inv2p, spp.e0z[k], Etz_tl.data());
            }

            // ── Pass 1: build R_sum = Σ_K Z_K · R^{nuc,K}_{tuv}^{(0)} ─────────
            // Every entry of this compact buffer is used by this angular class;
            // clear it for the next primitive pair (including after screening).
            std::fill(R_sum, R_sum + Rsum_sz, 0.0);
            // Blocked nucleus loop: KL lanes share the primitive pair, so the
            // MD recursion bounds are identical per lane and the fixed-KL
            // innermost loops vectorize (AVX2: one 4-wide FMA per recursion
            // step instead of four scalar ones). Trailing and zero-charge lanes
            // have zero Boys seeds; their finite dummy offsets keep the SIMD
            // recurrence valid without evaluating their Boys arguments.
#if defined(__AVX512F__) && defined(__FMA__)
            if constexpr (KL == 8) {
                auto process_packet = [&](auto width_tag, int K0) {
                    constexpr int PW = decltype(width_tag)::value;
                    using PV = tlno::core::simd::Packet<PW>;
                    double Zl[PW], pcx[PW], pcy[PW], pcz[PW];
                    unsigned active_mask=0;
                    for (int l = 0; l < PW; ++l) {
                        const int K = K0 + l;
                        if (K < n_K && nuclear_charges[static_cast<std::size_t>(K)] != 0) {
                            Zl[l]  = static_cast<double>(nuclear_charges[static_cast<std::size_t>(K)]);
                            pcx[l] = Px - nuclear_centers[static_cast<std::size_t>(K)][0];
                            pcy[l] = Py - nuclear_centers[static_cast<std::size_t>(K)][1];
                            pcz[l] = Pz - nuclear_centers[static_cast<std::size_t>(K)][2];
                            active_mask|=1u<<l;
                        } else {
                            Zl[l] = 0.0; pcx[l] = 1.0; pcy[l] = 0.0; pcz[l] = 0.0;
                        }
                    }
                    if (!active_mask) return;

                    // Boys seeds use packed PW stride; roffset retains the allocated
                    // KL stride so a W4 tail touches only the lower four R lanes.
                    // Boys seeds, PW lanes fused (vector interpolation + lane-fused
                    // downward recursion; layout boys_lane[n*PW + l]).
                    double xs[PW];
                    for (int l = 0; l < PW; ++l) {
                        xs[l] = p * (pcx[l] * pcx[l] + pcy[l] * pcy[l] + pcz[l] * pcz[l]);
                    }
                    math::boys_r000_all_batch_masked<PW>(max_n, xs, p, boys_lane,active_mask);

                    // Bounded MD recursion for R_{tuv}^{(n)}, PW lanes fused.
                    // At level n, only fill t+u+v <= max_n - n (no out-of-range reads).
                    for (int n = max_n; n >= 0; --n) {
                        PV::load(boys_lane+n*PW).store(R_buf+roffset(0,0,0,n));
                        const int max_tuv = max_n - n;
                        for (int v = 1; v <= std::min(ltot, max_tuv); ++v) {
                            const std::size_t o  = roffset(0,0,v,n);
                            const std::size_t o1 = roffset(0,0,v-1,n+1);
                            const std::size_t o2 = roffset(0,0,v-2,n+1);
                            const double cv = static_cast<double>(v - 1);
                            PV val = PV::load(pcz) * PV::load(R_buf+o1);
                            if (v > 1) val = PV::fma(PV(cv),PV::load(R_buf+o2),val);
                            val.store(R_buf+o);
                        }
                        for (int u = 1; u <= std::min(ltot, max_tuv); ++u) {
                            const double cu = static_cast<double>(u - 1);
                            for (int v = 0; v <= std::min(ltot, max_tuv - u); ++v) {
                                const std::size_t o  = roffset(0,u,v,n);
                                const std::size_t o1 = roffset(0,u-1,v,n+1);
                                const std::size_t o2 = roffset(0,u-2,v,n+1);
                                PV val = PV::load(pcy) * PV::load(R_buf+o1);
                                if (u > 1) val = PV::fma(PV(cu),PV::load(R_buf+o2),val);
                                val.store(R_buf+o);
                            }
                        }
                        for (int t = 1; t <= std::min(ltot, max_tuv); ++t) {
                            const double ct = static_cast<double>(t - 1);
                            for (int u = 0; u <= std::min(ltot, max_tuv - t); ++u) {
                                for (int v = 0; v <= std::min(ltot, max_tuv - t - u); ++v) {
                                    const std::size_t o  = roffset(t,u,v,n);
                                    const std::size_t o1 = roffset(t-1,u,v,n+1);
                                    const std::size_t o2 = roffset(t-2,u,v,n+1);
                                    PV val = PV::load(pcx) * PV::load(R_buf+o1);
                                    if (t > 1) val = PV::fma(PV(ct),PV::load(R_buf+o2),val);
                                    val.store(R_buf+o);
                                }
                            }
                        }
                    }

                    // Accumulate Σ_lane Z_l · R^{(0)}_{tuv} into R_sum.  Only the
                    // n=0 slice with t+u+v <= ltot is populated by the recurrence
                    // above, and only that range is read in the contraction below.
                    for (int t = 0; t <= ltot; ++t) {
                        for (int u = 0; u <= ltot - t; ++u) {
                            for (int v = 0; v <= ltot - t - u; ++v) {
                                const double* r =
                                    &R_buf[roffset(t,u,v,0)];
                                double charge_sum = 0.0;
                                for (int l = 0; l < PW; ++l) charge_sum += Zl[l]*r[l];
                                R_sum[ridx(t,u,v)] += charge_sum;
                            }
                        }
                    }
                };
                const int tail = n_K % KL;
                const int wide_end = n_K - (tail <= 4 ? tail : 0);
                int K0 = 0;
                for (; K0 < wide_end; K0 += KL)
                    process_packet(NuclearPacketWidth<8>{}, K0);
                if (K0 < n_K)
                    process_packet(NuclearPacketWidth<4>{}, K0);
            } else {
#endif
            for (int K0 = 0; K0 < n_K; K0 += KL) {
                double Zl[KL], pcx[KL], pcy[KL], pcz[KL];
                unsigned active_mask=0;
                for (int l = 0; l < KL; ++l) {
                    const int K = K0 + l;
                    if (K < n_K && nuclear_charges[static_cast<std::size_t>(K)] != 0) {
                        Zl[l]  = static_cast<double>(nuclear_charges[static_cast<std::size_t>(K)]);
                        pcx[l] = Px - nuclear_centers[static_cast<std::size_t>(K)][0];
                        pcy[l] = Py - nuclear_centers[static_cast<std::size_t>(K)][1];
                        pcz[l] = Pz - nuclear_centers[static_cast<std::size_t>(K)][2];
                        active_mask|=1u<<l;
                    } else {
                        Zl[l] = 0.0; pcx[l] = 1.0; pcy[l] = 0.0; pcz[l] = 0.0;
                    }
                }
                if (!active_mask) continue;

                // Boys seeds, KL lanes fused (vector interpolation + lane-fused
                // downward recursion; layout boys_lane[n*KL + l]).
                double xs[KL];
                for (int l = 0; l < KL; ++l) {
                    xs[l] = p * (pcx[l] * pcx[l] + pcy[l] * pcy[l] + pcz[l] * pcz[l]);
                }
                math::boys_r000_all_batch_masked<KL>(max_n, xs, p, boys_lane,active_mask);

                // Bounded MD recursion for R_{tuv}^{(n)}, KL lanes fused.
                // At level n, only fill t+u+v <= max_n - n (no out-of-range reads).
                for (int n = max_n; n >= 0; --n) {
                    V::load(boys_lane+n*KL).store(R_buf+roffset(0,0,0,n));
                    const int max_tuv = max_n - n;
                    for (int v = 1; v <= std::min(ltot, max_tuv); ++v) {
                        const std::size_t o  = roffset(0,0,v,n);
                        const std::size_t o1 = roffset(0,0,v-1,n+1);
                        const std::size_t o2 = roffset(0,0,v-2,n+1);
                        const double cv = static_cast<double>(v - 1);
                        V val = V::load(pcz) * V::load(R_buf+o1);
                        if (v > 1) val = V::fma(V(cv),V::load(R_buf+o2),val);
                        val.store(R_buf+o);
                    }
                    for (int u = 1; u <= std::min(ltot, max_tuv); ++u) {
                        const double cu = static_cast<double>(u - 1);
                        for (int v = 0; v <= std::min(ltot, max_tuv - u); ++v) {
                            const std::size_t o  = roffset(0,u,v,n);
                            const std::size_t o1 = roffset(0,u-1,v,n+1);
                            const std::size_t o2 = roffset(0,u-2,v,n+1);
                            V val = V::load(pcy) * V::load(R_buf+o1);
                            if (u > 1) val = V::fma(V(cu),V::load(R_buf+o2),val);
                            val.store(R_buf+o);
                        }
                    }
                    for (int t = 1; t <= std::min(ltot, max_tuv); ++t) {
                        const double ct = static_cast<double>(t - 1);
                        for (int u = 0; u <= std::min(ltot, max_tuv - t); ++u) {
                            for (int v = 0; v <= std::min(ltot, max_tuv - t - u); ++v) {
                                const std::size_t o  = roffset(t,u,v,n);
                                const std::size_t o1 = roffset(t-1,u,v,n+1);
                                const std::size_t o2 = roffset(t-2,u,v,n+1);
                                V val = V::load(pcx) * V::load(R_buf+o1);
                                if (t > 1) val = V::fma(V(ct),V::load(R_buf+o2),val);
                                val.store(R_buf+o);
                            }
                        }
                    }
                }

                // Accumulate Σ_lane Z_l · R^{(0)}_{tuv} into R_sum.  Only the
                // n=0 slice with t+u+v <= ltot is populated by the recurrence
                // above, and only that range is read in the contraction below.
                for (int t = 0; t <= ltot; ++t) {
                    for (int u = 0; u <= ltot - t; ++u) {
                        for (int v = 0; v <= ltot - t - u; ++v) {
                            const double* r =
                                &R_buf[roffset(t,u,v,0)];
                            double charge_sum = 0.0;
                            for (int l = 0; l < KL; ++l) charge_sum += Zl[l]*r[l];
                            R_sum[ridx(t,u,v)] += charge_sum;
                        }
                    }
                }
            }  // blocked nucleus loop
#if defined(__AVX512F__) && defined(__FMA__)
            }
#endif


            // ── Pass 2: single E·R_sum contraction per primitive pair ────────
            // Replaces the n_K-fold contraction in the original layout.
            for (int ca = 0; ca < n_a; ++ca) {
                const double d_a = group_a.coeffs[static_cast<std::size_t>(ca)][static_cast<std::size_t>(ia_prim)];
                const int la = group_a.ang[static_cast<std::size_t>(ca)][0];
                const int ma = group_a.ang[static_cast<std::size_t>(ca)][1];
                const int na = group_a.ang[static_cast<std::size_t>(ca)][2];

                for (int cb = 0; cb < n_b; ++cb) {
                    const double d_b = group_b.coeffs[static_cast<std::size_t>(cb)][static_cast<std::size_t>(ib_prim)];
                    const int lb = group_b.ang[static_cast<std::size_t>(cb)][0];
                    const int mb = group_b.ang[static_cast<std::size_t>(cb)][1];
                    const int nb = group_b.ang[static_cast<std::size_t>(cb)][2];
                    const double w = d_a * d_b * neg2pi_over_p;

                    double sum = 0.0;
                    for (int t = 0; t <= la + lb; ++t) {
                        const double ex = Etx[static_cast<std::size_t>((la * Et_col + lb) * Et_nt + t)];
                        for (int u = 0; u <= ma + mb; ++u) {
                            const double ey = Ety[static_cast<std::size_t>((ma * Et_col + mb) * Et_nt + u)];
                            const double exy = ex * ey;
                            for (int v = 0; v <= na + nb; ++v) {
                                sum += exy *
                                       Etz[static_cast<std::size_t>((na * Et_col + nb) * Et_nt + v)] *
                                       R_sum[ridx(t,u,v)];
                            }
                        }
                    }
                    nuclear_block[static_cast<std::size_t>(ca * n_b + cb)] += w * sum;
                }
            }
        }  // ib_prim
    }  // ia_prim
}

// Public entry: dispatch to the angular-momentum-specialized instantiation so
// the recursion/contraction unroll with compile-time bounds.  Covers s,p,d,f
// (la_max,lb_max <= 3 — every basis in the suite); higher L uses the runtime
// generic instantiation <-1,-1>.
void contracted_shell_block_nuclear(
    const ShellGroup& group_a, 
    const ShellGroup& group_b, 
    const ShellPairPrim& spp, 
    double* nuclear_block, 
    const std::vector<int>& nuclear_charges, 
    const std::vector<std::array<double, 3>>& nuclear_centers) {

    const int la_max = group_a.L;
    const int lb_max = group_b.L;

#define TLNO_NUC_CASE(LA, LB) \
    case (LA) * 4 + (LB): \ 
        nuclear_general_impl<LA, LB>(group_a, group_b, spp, nuclear_block, nuclear_charges, nuclear_centers); \ 
        return;

    if (la_max <= 3 && lb_max <= 3) {
        switch (la_max * 4 + lb_max) {
            TLNO_NUC_CASE(0, 0) TLNO_NUC_CASE(0, 1) TLNO_NUC_CASE(0, 2) TLNO_NUC_CASE(0, 3)
            TLNO_NUC_CASE(1, 0) TLNO_NUC_CASE(1, 1) TLNO_NUC_CASE(1, 2) TLNO_NUC_CASE(1, 3)
            TLNO_NUC_CASE(2, 0) TLNO_NUC_CASE(2, 1) TLNO_NUC_CASE(2, 2) TLNO_NUC_CASE(2, 3)
            TLNO_NUC_CASE(3, 0) TLNO_NUC_CASE(3, 1) TLNO_NUC_CASE(3, 2) TLNO_NUC_CASE(3, 3)
            default: break;
        }
    }
#undef TLNO_NUC_CASE

    nuclear_general_impl<-1, -1>(group_a, group_b, spp, nuclear_block, nuclear_charges, nuclear_centers);
}

void set_onee_nuclear_screen_eps(double eps) { g_onee_nuclear_screen_eps = eps; }
double get_onee_nuclear_screen_eps() { return g_onee_nuclear_screen_eps; }

} // namespace tlno::integrals::onee
