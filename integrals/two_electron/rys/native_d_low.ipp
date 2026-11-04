#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_D_LOW_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_D_LOW_IPP

// Exact spherical ds/dp against ss/ps, using the same Gaussian moment
// identities as the tuned AVX2 producer. No radial Cartesian intermediate.
template<int W, bool PairP, bool OtherP>
inline void native_d_low_contract(const tlno::core::simd::Packet<W> c[3],
    const tlno::core::simd::Packet<W> d[3], const tlno::core::simd::Packet<W> ab[3],
    tlno::core::simd::Packet<W> b00, tlno::core::simd::Packet<W> b10,
    tlno::core::simd::Packet<W> weight, tlno::core::simd::Packet<W>* acc) noexcept {
    using V = tlno::core::simd::Packet<W>;
    V h[5], dh[5][3]; native_d_solid_harmonics<W>(c, h, dh);
    for (int a = 0; a < 5; ++a) {
        if constexpr (!PairP) {
            if constexpr (!OtherP) acc[a] = V::fma(weight, h[a], acc[a]);
            else for (int j = 0; j < 3; ++j) {
                const V value = V::fma(b00, dh[a][j], h[a] * d[j]);
                acc[3 * a + j] = V::fma(weight, value, acc[3 * a + j]);
            }
        } else for (int i = 0; i < 3; ++i) {
            const V shifted = c[i] + ab[i];
            const V f = V::fma(b10, dh[a][i], h[a] * shifted);
            if constexpr (!OtherP) acc[3 * a + i] = V::fma(weight, f, acc[3 * a + i]);
            else for (int j = 0; j < 3; ++j) {
                V df = V::fma(b10, V(kNativeDSolidHessian[a][i][j]), dh[a][j] * shifted);
                if (i == j) df = df + h[a];
                const V value = V::fma(b00, df, f * d[j]);
                const int o = 3 * (3 * a + i) + j;
                acc[o] = V::fma(weight, value, acc[o]);
            }
        }
    }
}
#endif
