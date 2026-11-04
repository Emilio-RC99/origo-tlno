#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SP_BOYS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SP_BOYS_IPP

// Bounded Boys-moment paths from the legacy AVX2 s/p portfolio. Coefficients
// and domains are shared with rys_engine.ipp, not refit for the native width.
// Each Packet lane is one independent primitive quartet.
#if defined(__AVX512F__) && defined(__FMA__)
template<int Last, class V>
inline void native_boys_global(V target, V* f) noexcept {
    static_assert(Last >= 0 && Last <= 2);
    const V x = target * V(1.0 / kRysSpBoysGlobalMaxT);
    V den(kRysSpBoysGlobalDenominator.back());
    f[0] = V(kRysSpBoysGlobal0Numerator.back());
    if constexpr (Last >= 1) f[1] = V(kRysSpBoysGlobal1Numerator.back());
    if constexpr (Last >= 2) f[2] = V(kRysSpBoysGlobal2Numerator.back());
#pragma GCC unroll 10
    for (int k = 9; k >= 0; --k) {
        f[0] = V::fma(f[0], x, V(kRysSpBoysGlobal0Numerator[k]));
        if constexpr (Last >= 1) f[1] = V::fma(f[1], x, V(kRysSpBoysGlobal1Numerator[k]));
        if constexpr (Last >= 2) f[2] = V::fma(f[2], x, V(kRysSpBoysGlobal2Numerator[k]));
        den = V::fma(den, x, V(kRysSpBoysGlobalDenominator[k]));
    }
    const V inv = V(1.0) / den;
    for (int m = 0; m <= Last; ++m) f[m] = f[m] * inv;
}

template<int Last, class V>
inline void native_boys_medium(V target, V* f) noexcept {
    static_assert(Last >= 0 && Last <= 2);
    const V z = V::fma(target, V(2.0 / kRysSpBoysMediumMaxT), V(-1.0));
    if constexpr (Last == 0) {
        V p(kRysSsssMediumNumerator.back()), q(kRysSsssMediumDenominator.back());
        q = V::fma(q, z, V(kRysSsssMediumDenominator[5]));
#pragma GCC unroll 5
        for (int k = 4; k >= 0; --k) {
            p = V::fma(p, z, V(kRysSsssMediumNumerator[k]));
            q = V::fma(q, z, V(kRysSsssMediumDenominator[k]));
        }
        f[0] = p / q;
    } else {
        const V zz = z + z;
        V p0, p02, p1, p12, p2, p22, q, q2;
#pragma GCC unroll 8
        for (int k = 8; k >= 1; --k) {
            const V a = V::fma(zz, p0, V() - p02) + V(kRysPsSsBoys0Numerator[k]);
            const V b = V::fma(zz, p1, V() - p12) + V(kRysPsSsBoys1Numerator[k]);
            const V d = V::fma(zz, q, V() - q2) + V(kRysPsSsBoysDenominator[k]);
            p02 = p0; p0 = a; p12 = p1; p1 = b; q2 = q; q = d;
            if constexpr (Last == 2) {
                const V c = V::fma(zz, p2, V() - p22) + V(kRysPsPsBoys2Numerator[k]);
                p22 = p2; p2 = c;
            }
        }
        const V den = V::fma(z, q, V() - q2) + V(kRysPsSsBoysDenominator[0]);
        const V inv = V(1.0) / den;
        f[0] = (V::fma(z, p0, V() - p02) + V(kRysPsSsBoys0Numerator[0])) * inv;
        f[1] = (V::fma(z, p1, V() - p12) + V(kRysPsSsBoys1Numerator[0])) * inv;
        if constexpr (Last == 2) f[2] = (V::fma(z, p2, V() - p22) + V(kRysPsPsBoys2Numerator[0])) * inv;
    }
}

template<int Last>
inline bool native_boys_bounded8(Packet<8> target, Packet<8>* f) noexcept {
    if (_mm512_cmp_pd_mask(target.value, _mm512_setzero_pd(), _CMP_GE_OQ) != 255 ||
        _mm512_cmp_pd_mask(target.value, _mm512_set1_pd(kRysSpBoysGlobalMaxT), _CMP_LE_OQ) != 255)
        return false;
    if (_mm512_cmp_pd_mask(target.value, _mm512_set1_pd(kRysSpBoysMediumMaxT), _CMP_LE_OQ) == 255)
        native_boys_medium<Last>(target, f);
    else native_boys_global<Last>(target, f);
    return true;
}

template<int A, int B, int C, int D>
inline bool native_sp_boys_contract8(Packet<8> target, const Packet<8>* pa,
        const Packet<8>* qc, const Packet<8>* pq, const Packet<8>* ab,
        Packet<8> qo, Packet<8> po, Packet<8> inv2p, Packet<8> inv2pq,
        Packet<8> pref, Packet<8>* __restrict output) noexcept {
    using V = Packet<8>;
    constexpr int Degree = A + B + C + D;
    static_assert((A == 0 && Degree == 0) || (A == 1 && B == 0 && C <= 1 && D == 0) ||
                  (A == 1 && B == 1 && C == 0 && D == 0));
    V f[Degree + 1];
    if (!native_boys_bounded8<Degree>(target, f)) return false;
    if constexpr (Degree == 0) output[0] = V::fma(pref, f[0], output[0]);
    else if constexpr (Degree == 1) {
        const V scale = qo * f[1];
        for (int i = 0; i < 3; ++i)
            output[i] = V::fma(pref, V::fma(V() - scale, pq[i], f[0] * pa[i]), output[i]);
    } else {
        const V c1[3] = {V() - qo * pq[0], V() - qo * pq[1], V() - qo * pq[2]};
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            V d0, d1, first, constant;
            if constexpr (B == 0) {
                d0 = qc[j]; d1 = po * pq[j];
                constant = pa[i] * d0;
                first = V::fma(pa[i], d1, c1[i] * d0);
                if (i == j) first = first + inv2pq;
            } else {
                d0 = pa[j] + ab[j]; d1 = c1[j];
                constant = pa[i] * d0;
                first = V::fma(pa[i], d1, c1[i] * d0);
                if (i == j) { constant = constant + inv2p; first = first - inv2p * qo; }
            }
            const V value = V::fma(f[2], c1[i] * d1, V::fma(f[1], first, f[0] * constant));
            output[3 * i + j] = V::fma(pref, value, output[3 * i + j]);
        }
    }
    return true;
}
#endif // __AVX512F__ && __FMA__
#endif
