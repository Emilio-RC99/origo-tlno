#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_ROOTS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_ROOTS_IPP

// native_roots.ipp — SIMD-batched Rys root+weight table interpolation
// ─────────────────────────────────────────────────────────────────────────────
// Must be included from rys_engine.ipp *after* RysRootWeightTable (g_rys_tbl)
// and rys_roots_weights_fixed<N> are defined.
//
// Private helpers:
//
//   template<int N, int W>
//   void native_roots_weights(const double* targets,
//                             double* roots, double* weights) noexcept
//     Computes N Rys roots and weights for W independent quartets.
//     targets[W]  – T arguments, one per quartet, W contiguous doubles.
//     roots  [N*W] – output, root-major: roots  [n*W + l] = root n, lane l.
//     weights[N*W] – output, root-major: weights[n*W + l] = weight n, lane l.
//     W ∈ {2,4,8};  N ∈ [1,9] (RysRootWeightTable::NMIN..NMAX).
//     Ordinary lanes (T < tmax[N]): batched gather from the uniform-grid table
//     + five-pint interpolation (native_root_interpolation.ipp) via Packet<W>, no scalar per-lane work.
//     All-asymptotic packets (every T ≥ tmax[N]): SIMD evaluation of the same
//     Hermite asymptotic table, before any gather or interpolation.
//     W8 mixed packets repair only the asymptotic root-major slots using
//     masked Hermite arithmetic. W2/W4 retain their scalar repair.
//
//   template<int W>
//   void native_weight1(const double* targets, double* weights) noexcept
//     N=1 weight-only path for ss|ss quartets (no root computed or stored).
//     weights[W] – output, one weight per lane.
//
// ISA coverage:
//   W=8 – AVX-512F + FMA: _mm512_i32gather_pd with 8×32-bit lane indices.
//   W=4 – AVX2    + FMA: _mm256_i32gather_pd with 4×32-bit lane indices.
//   W=2 – SSE2  (baseline): _mm_set_pd for 2-wide gather.
//
// Limitations:
//   • W2/W4 mixed batches retain a scalar fixup loop for asymptotic lanes.
//     W8 uses masked arithmetic and preserves table slots with masked stores.
//   • A scalar T histogram does not determine how often an entire packet is
//     asymptotic; that depends on the quartet schedule.
//   • N outside [1,9] triggers static_assert; the caller should route those
//     to rys_roots_weights_accurate directly.

// Included in the private engine scope; Packet is already declared.
using ::tlno::core::simd::Packet;

// Same per-lane formula and constants as rys_hermite_asymptotic_table<N>.
// Call only after every target meets the tmax[N] threshold.
// Weight-only instantiation neither reads root constants nor computes roots.
template<int N, int W, bool WithRoots = true>
inline void native_hermite_asymptotic_all(Packet<W> targets,
                                          double* roots,
                                          double* weights) noexcept {
    static_assert(N >= 1 && N <= RysHermiteAsymptoticTable::NMAX,
                  "native Hermite asymptotic order out of range");
    static_assert(W == 2 || W == 4 || W == 8,
                  "native Hermite asymptotic width must be 2, 4, or 8");
    const Packet<W> invT = Packet<W>(1.0) / targets;
    const Packet<W> invSqrtT = Packet<W>::sqrt(invT);
    for (int n = 0; n < N; ++n) {
        if constexpr (WithRoots) {
            (Packet<W>(g_rys_hermite_asymp.roots_sq[N][n]) * invT)
                .store(roots + n * W);
        }
        (Packet<W>(g_rys_hermite_asymp.weights[N][n]) * invSqrtT)
            .store(weights + n * W);
    }
}

// ── 4th-order Newton-forward stencil, generic Packet<W> ──────────────────────
// f0..f4: table values at grid rows ki, ki+1, ..., ki+4 for one root across W lanes.
// t: fractional offsets in [0,1).  tm1=t-1, tm2=t-2, tm3=t-3.
//
// Finite-difference coefficients d1..d4 match the scalar rys_roots_weights_table
// loop exactly.  Horner nesting matches rys_newton5v arithmetic association.
// Packet<W>::fma(a,b,c)=a*b+c; for W=4/8 this emits a real FMA instruction.
template<int W>
[[nodiscard]] inline Packet<W>
newton5_pkt(Packet<W> f0, Packet<W> f1, Packet<W> f2,
            Packet<W> f3, Packet<W> f4,
            Packet<W> t,  Packet<W> tm1,
            Packet<W> tm2, Packet<W> tm3) noexcept {
    // d1 = f1 - f0
    const Packet<W> d1 = f1 - f0;
    // d2 = 0.5*(f2 - 2*f1 + f0)  =  fma(-2, f1, f2+f0)*0.5
    // d2 = (f2 - 2*f1 + f0)/2; f1 + f1 is exact, so d2 matches the scalar loop bit for bit.
    const Packet<W> d2 = ((f2 - (f1 + f1)) + f0) * Packet<W>(0.5);
    // d3 = (f3 - 3*f2 + 3*f1 - f0)/6  =  (fma(3,f1,f3) - fma(3,f2,f0))/6
    const Packet<W> d3 = (Packet<W>::fma(Packet<W>(3.0), f1, f3)
                        - Packet<W>::fma(Packet<W>(3.0), f2, f0))
                         * Packet<W>(1.0 / 6.0);
    // d4 = (f4 - 4*f3 + 6*f2 - 4*f1 + f0)/24
    //     = (fma(6,f2, f4+f0) + (-4)*(f3+f1))/24
    const Packet<W> d4 = (Packet<W>::fma(Packet<W>(6.0), f2, f4 + f0)
                        + Packet<W>(-4.0) * (f3 + f1))
                         * Packet<W>(1.0 / 24.0);
    // Horner: f0 + t*(d1 + tm1*(d2 + tm2*(d3 + tm3*d4)))
    return Packet<W>::fma(t,
               Packet<W>::fma(tm1,
                   Packet<W>::fma(tm2,
                       Packet<W>::fma(tm3, d4, d3),
                   d2),
               d1),
           f0);
}

#include "native_root_interpolation.ipp"
// ── Scalar lane fix-up for out-of-table (asymptotic) lanes ───────────────────
// Called only when one or more lanes exceed tmax[N].  Overwrites only the
// affected root-major slots; SIMD-computed slots for other lanes are preserved.
template<int N, int W>
inline void fixup_lane(int lane, double target,
                       double* roots, double* weights) noexcept {
    double lr[N], lw[N];
    rys_roots_weights_fixed<N>(target, lr, lw);
    for (int n = 0; n < N; ++n) {
        roots[n * W + lane]   = lr[n];
        weights[n * W + lane] = lw[n];
    }
}

// ── AVX-512 (W=8): _mm512_i32gather_pd with 8×int32 lane index vector ────────
#if defined(__AVX512F__) && defined(__FMA__)

// Gather 8 doubles from tbl at byte-positions tbl + (base32[l]+off)*8.
inline Packet<8> rys_gather8(const double* tbl,
                              __m256i base32, int off) noexcept {
    __m256i idx = _mm256_add_epi32(base32, _mm256_set1_epi32(off));
    return Packet<8>(_mm512_i32gather_pd(idx, tbl, 8));
}

// Mixed packets use exactly the same Hermite constants as the scalar repair.
// Mask the division itself: discarded table lanes can contain T=0. Masked
// stores preserve the already interpolated table values without reloading them.
template<int N, bool WithRoots = true>
inline void native_hermite_asymptotic_masked8(__m512d targets,__mmask8 mask,
                                              double* roots, double* weights) noexcept {
    const __m512d inverse = _mm512_mask_div_pd(_mm512_setzero_pd(), mask,
                                            _mm512_set1_pd(1.0), targets);
    const __m512d inverse_sqrt = _mm512_sqrt_pd(inverse);
    for(int n = 0; n < N; ++n) {
        if constexpr(WithRoots)
            _mm512_mask_storeu_pd(roots + n * 8, mask,
                _mm512_mul_pd(_mm512_set1_pd(g_rys_hermite_asymp.roots_sq[N][n]), inverse));
        _mm512_mask_storeu_pd(weights + n * 8, mask,
            _mm512_mul_pd(_mm512_set1_pd(g_rys_hermite_asymp.weights[N][n]), inverse_sqrt));
    }
}

template<int N>
inline void native_roots_weights_w8(const double* __restrict targets,
                                    double* __restrict roots,
                                    double* __restrict weights) noexcept {
    static_assert(N >= RysRootWeightTable::NMIN && N <= RysRootWeightTable::NMAX,
                  "native_roots_weights_w8: N out of table range [1,9]");
    const double  tm   = g_rys_tbl.tmax[N];
    const double* rtbl = g_rys_tbl.roots_tbl[N].data();
    const double* wtbl = g_rys_tbl.wts_tbl[N].data();

    const __m512d input = _mm512_loadu_pd(targets);
    const int mask = _mm512_cmp_pd_mask(input, _mm512_set1_pd(tm), _CMP_GE_OQ);
    if (mask == 255) {
        native_hermite_asymptotic_all<N, 8>(Packet<8>(input), roots, weights);
        return;
    }
    const __m512d safe = _mm512_mask_mov_pd(input, (__mmask8)mask, _mm512_set1_pd(tm - 0.1));
    const __m512d scaled = _mm512_mul_pd(safe, _mm512_set1_pd(RysRootWeightTable::INV_DX));
    const __m256i index = _mm512_cvttpd_epi32(scaled);
    const __m256i bidx = _mm256_mullo_epi32(index, _mm256_set1_epi32(N));
    const Packet<8> t(_mm512_sub_pd(scaled, _mm512_cvtepi32_pd(index)));
    const NativeRootInterpolation<8> interpolation(t);

    // One five-sample stencil per output across all 8 lanes. The
    // Lagrange route shares its coefficients across all roots and weights.
    // Both routes retain 10 gathers + 2 stores per root and the same bidx.
    for (int n = 0; n < N; ++n) {
        interpolation.apply(rys_gather8(rtbl, bidx, n),
                       rys_gather8(rtbl, bidx, n +     N),
                       rys_gather8(rtbl, bidx, n + 2 * N),
                       rys_gather8(rtbl, bidx, n + 3 * N),
                       rys_gather8(rtbl, bidx, n + 4 * N)).store(roots   + n * 8);
        interpolation.apply(rys_gather8(wtbl, bidx, n),
                       rys_gather8(wtbl, bidx, n +     N),
                       rys_gather8(wtbl, bidx, n + 2 * N),
                       rys_gather8(wtbl, bidx, n + 3 * N),
                       rys_gather8(wtbl, bidx, n + 4 * N)).store(weights + n * 8);
    }
    if (mask) {
        native_hermite_asymptotic_masked8<N>(input, (__mmask8)mask, roots, weights);
    }
}

// N=1 weight-only: skips root gather/store entirely.
inline void native_weight1_w8(const double* __restrict targets,
                               double* __restrict weights) noexcept {
    const double  tm   = g_rys_tbl.tmax[1];
    const double* wtbl = g_rys_tbl.wts_tbl[1].data();

    const __m512d input = _mm512_loadu_pd(targets);
    const int mask = _mm512_cmp_pd_mask(input, _mm512_set1_pd(tm), _CMP_GE_OQ);
    if (mask == 255) {
        native_hermite_asymptotic_all<1, 8, false>(Packet<8>(input), nullptr, weights);
        return;
    }
    const __m512d safe = _mm512_mask_mov_pd(input, (__mmask8)mask, _mm512_set1_pd(tm - 0.1));
    const __m512d scaled = _mm512_mul_pd(safe, _mm512_set1_pd(RysRootWeightTable::INV_DX));
    const __m256i index = _mm512_cvttpd_epi32(scaled);
    const __m256i bidx = _mm256_mullo_epi32(index, _mm256_set1_epi32(1));
    const Packet<8> t(_mm512_sub_pd(scaled, _mm512_cvtepi32_pd(index)));
    const Packet<8> tm1 = t - Packet<8>(1.0);
    const Packet<8> tm2 = t - Packet<8>(2.0);
    const Packet<8> tm3 = t - Packet<8>(3.0);

    newton5_pkt<8>(rys_gather8(wtbl, bidx, 0),
                   rys_gather8(wtbl, bidx, 1),
                   rys_gather8(wtbl, bidx, 2),
                   rys_gather8(wtbl, bidx, 3),
                   rys_gather8(wtbl, bidx, 4),
                   t, tm1, tm2, tm3).store(weights);
    if (mask) {
        native_hermite_asymptotic_masked8<1,false>(input, (__mmask8)mask, nullptr, weights);
    }
}

#endif // __AVX512F__ && __FMA__

// ── AVX2 (W=4): _mm256_i32gather_pd with 4×int32 lane index vector ───────────
// Note: argument order differs from AVX-512 gather — (base_addr, vindex, scale).
#if defined(__AVX2__) && defined(__FMA__)

inline Packet<4> rys_gather4(const double* tbl,
                              __m128i base32, int off) noexcept {
    __m128i idx = _mm_add_epi32(base32, _mm_set1_epi32(off));
    return Packet<4>(_mm256_i32gather_pd(tbl, idx, 8));
}

template<int N>
inline void native_roots_weights_w4(const double* __restrict targets,
                                    double* __restrict roots,
                                    double* __restrict weights) noexcept {
    static_assert(N >= RysRootWeightTable::NMIN && N <= RysRootWeightTable::NMAX,
                  "native_roots_weights_w4: N out of table range [1,9]");
    const double  tm   = g_rys_tbl.tmax[N];
    const double* rtbl = g_rys_tbl.roots_tbl[N].data();
    const double* wtbl = g_rys_tbl.wts_tbl[N].data();

    const __m256d input = _mm256_loadu_pd(targets);
    const __m256d asym = _mm256_cmp_pd(input, _mm256_set1_pd(tm), _CMP_GE_OQ);
    const int mask = _mm256_movemask_pd(asym);
    if (mask == 15) {
        native_hermite_asymptotic_all<N, 4>(Packet<4>(input), roots, weights);
        return;
    }
    const __m256d safe = _mm256_blendv_pd(input, _mm256_set1_pd(tm - 0.1), asym);
    const __m256d scaled = _mm256_mul_pd(safe, _mm256_set1_pd(RysRootWeightTable::INV_DX));
    const __m128i index = _mm256_cvttpd_epi32(scaled);
    const __m128i bidx = _mm_mullo_epi32(index, _mm_set1_epi32(N));
    const Packet<4> t(_mm256_sub_pd(scaled, _mm256_cvtepi32_pd(index)));
    const NativeRootInterpolation<4> interpolation(t);

    for (int n = 0; n < N; ++n) {
        interpolation.apply(rys_gather4(rtbl, bidx, n),
                       rys_gather4(rtbl, bidx, n +     N),
                       rys_gather4(rtbl, bidx, n + 2 * N),
                       rys_gather4(rtbl, bidx, n + 3 * N),
                       rys_gather4(rtbl, bidx, n + 4 * N)).store(roots   + n * 4);
        interpolation.apply(rys_gather4(wtbl, bidx, n),
                       rys_gather4(wtbl, bidx, n +     N),
                       rys_gather4(wtbl, bidx, n + 2 * N),
                       rys_gather4(wtbl, bidx, n + 3 * N),
                       rys_gather4(wtbl, bidx, n + 4 * N)).store(weights + n * 4);
    }
    if (mask) {
        for (int l = 0; l < 4; ++l)
            if (mask & (1 << l))
                fixup_lane<N, 4>(l, targets[l], roots, weights);
    }
}

inline void native_weight1_w4(const double* __restrict targets,
                               double* __restrict weights) noexcept {
    const double  tm   = g_rys_tbl.tmax[1];
    const double* wtbl = g_rys_tbl.wts_tbl[1].data();

    const __m256d input = _mm256_loadu_pd(targets);
    const __m256d asym = _mm256_cmp_pd(input, _mm256_set1_pd(tm), _CMP_GE_OQ);
    const int mask = _mm256_movemask_pd(asym);
    if (mask == 15) {
        native_hermite_asymptotic_all<1, 4, false>(Packet<4>(input), nullptr, weights);
        return;
    }
    const __m256d safe = _mm256_blendv_pd(input, _mm256_set1_pd(tm - 0.1), asym);
    const __m256d scaled = _mm256_mul_pd(safe, _mm256_set1_pd(RysRootWeightTable::INV_DX));
    const __m128i index = _mm256_cvttpd_epi32(scaled);
    const __m128i bidx = _mm_mullo_epi32(index, _mm_set1_epi32(1));
    const Packet<4> t(_mm256_sub_pd(scaled, _mm256_cvtepi32_pd(index)));
    const Packet<4> tm1 = t - Packet<4>(1.0);
    const Packet<4> tm2 = t - Packet<4>(2.0);
    const Packet<4> tm3 = t - Packet<4>(3.0);

    newton5_pkt<4>(rys_gather4(wtbl, bidx, 0),
                   rys_gather4(wtbl, bidx, 1),
                   rys_gather4(wtbl, bidx, 2),
                   rys_gather4(wtbl, bidx, 3),
                   rys_gather4(wtbl, bidx, 4),
                   t, tm1, tm2, tm3).store(weights);
    if (mask) {
        for (int l = 0; l < 4; ++l) {
            if (!(mask & (1 << l))) continue;
            double dummy_root[1], lw[1];
            rys_roots_weights_fixed<1>(targets[l], dummy_root, lw);
            weights[l] = lw[0];
        }
    }
}

#endif // __AVX2__ && __FMA__

// ── SSE2 baseline (W=2): _mm_set_pd for 2-wide gather ────────────────────────
// Always compiled.  _mm_set_pd(high, low) so element ordering matches storeu:
// p[0]=lane0=low, p[1]=lane1=high.

template<int N>
inline void native_roots_weights_w2(const double* __restrict targets,
                                    double* __restrict roots,
                                    double* __restrict weights) noexcept {
    static_assert(N >= RysRootWeightTable::NMIN && N <= RysRootWeightTable::NMAX,
                  "native_roots_weights_w2: N out of table range [1,9]");
    const double  tm   = g_rys_tbl.tmax[N];
    if (targets[0] >= tm && targets[1] >= tm) {
        native_hermite_asymptotic_all<N, 2>(Packet<2>::load(targets), roots, weights);
        return;
    }
    const double* rtbl = g_rys_tbl.roots_tbl[N].data();
    const double* wtbl = g_rys_tbl.wts_tbl[N].data();

    int    base[2], mask = 0;
    double tf[2];
    for (int l = 0; l < 2; ++l) {
        const bool asym = (targets[l] >= tm);
        if (asym) mask |= (1 << l);
        const double safe = asym ? (tm - 0.1) : targets[l];
        const double fi   = safe * RysRootWeightTable::INV_DX;
        const int    ki   = static_cast<int>(fi);
        base[l] = ki * N;
        tf[l]   = fi - static_cast<double>(ki);
    }

    const Packet<2> t  (_mm_loadu_pd(tf));
    const NativeRootInterpolation<2> interpolation(t);

    for (int n = 0; n < N; ++n) {
        // 2-wide "gather" — set_pd(lane1, lane0) = set_pd(high, low)
        auto gr = [&](int off) noexcept {
            return Packet<2>(_mm_set_pd(rtbl[base[1] + off], rtbl[base[0] + off]));
        };
        auto gw = [&](int off) noexcept {
            return Packet<2>(_mm_set_pd(wtbl[base[1] + off], wtbl[base[0] + off]));
        };
        interpolation.apply(gr(n), gr(n+N), gr(n+2*N), gr(n+3*N), gr(n+4*N)).store(roots   + n * 2);
        interpolation.apply(gw(n), gw(n+N), gw(n+2*N), gw(n+3*N), gw(n+4*N)).store(weights + n * 2);
    }
    if (mask) {
        for (int l = 0; l < 2; ++l)
            if (mask & (1 << l))
                fixup_lane<N, 2>(l, targets[l], roots, weights);
    }
}

inline void native_weight1_w2(const double* __restrict targets,
                               double* __restrict weights) noexcept {
    const double  tm   = g_rys_tbl.tmax[1];
    if (targets[0] >= tm && targets[1] >= tm) {
        native_hermite_asymptotic_all<1, 2, false>(Packet<2>::load(targets), nullptr, weights);
        return;
    }
    const double* wtbl = g_rys_tbl.wts_tbl[1].data();

    int    base[2], mask = 0;
    double tf[2];
    for (int l = 0; l < 2; ++l) {
        const bool asym = (targets[l] >= tm);
        if (asym) mask |= (1 << l);
        const double safe = asym ? (tm - 0.1) : targets[l];
        const double fi   = safe * RysRootWeightTable::INV_DX;
        const int    ki   = static_cast<int>(fi);
        base[l] = ki;
        tf[l]   = fi - static_cast<double>(ki);
    }

    const Packet<2> t  (_mm_loadu_pd(tf));
    const Packet<2> tm1(_mm_sub_pd(t.value, _mm_set1_pd(1.0)));
    const Packet<2> tm2(_mm_sub_pd(t.value, _mm_set1_pd(2.0)));
    const Packet<2> tm3(_mm_sub_pd(t.value, _mm_set1_pd(3.0)));

    auto gw = [&](int off) noexcept {
        return Packet<2>(_mm_set_pd(wtbl[base[1] + off], wtbl[base[0] + off]));
    };
    newton5_pkt<2>(gw(0), gw(1), gw(2), gw(3), gw(4),
                   t, tm1, tm2, tm3).store(weights);
    if (mask) {
        for (int l = 0; l < 2; ++l) {
            if (!(mask & (1 << l))) continue;
            double dummy_root[1], lw[1];
            rys_roots_weights_fixed<1>(targets[l], dummy_root, lw);
            weights[l] = lw[0];
        }
    }
}

// ── Public API: dispatch by W to ISA-specific implementation ─────────────────

template<int N, int W>
inline void native_roots_weights(const double* targets,
                                 double* roots,
                                 double* weights) noexcept {
    static_assert(N >= RysRootWeightTable::NMIN && N <= RysRootWeightTable::NMAX,
                  "native_roots_weights: N out of table range [1,9]");
    static_assert(W == 2 || W == 4 || W == 8,
                  "native_roots_weights: W must be 2, 4, or 8");
    if constexpr (W == 8) {
#if defined(__AVX512F__) && defined(__FMA__)
        native_roots_weights_w8<N>(targets, roots, weights);
#endif
    } else if constexpr (W == 4) {
#if defined(__AVX2__) && defined(__FMA__)
        native_roots_weights_w4<N>(targets, roots, weights);
#endif
    } else {
        native_roots_weights_w2<N>(targets, roots, weights);
    }
}

template<int W>
inline void native_weight1(const double* targets, double* weights) noexcept {
    static_assert(W == 2 || W == 4 || W == 8,
                  "native_weight1: W must be 2, 4, or 8");
    if constexpr (W == 8) {
#if defined(__AVX512F__) && defined(__FMA__)
        native_weight1_w8(targets, weights);
#endif
    } else if constexpr (W == 4) {
#if defined(__AVX2__) && defined(__FMA__)
        native_weight1_w4(targets, weights);
#endif
    } else {
        native_weight1_w2(targets, weights);
    }
}

#endif
