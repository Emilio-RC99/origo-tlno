#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_D_DIRECT_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_D_DIRECT_IPP

// Exact spherical ds|ds and ds|pp root contractions, ported from the AVX2
// rys_d_solid_harmonics_v / accumulate_rys_ds{ds,pp}_spherical_shell_bin_pair
// in rys_engine.ipp. Include in the same namespace as native_specialized.ipp.
// Each Packet lane remains one independent shell quartet. The caller owns
// masks, primitive/root ordering, prefactors, zeroing, and final emission.
// Root geometry is C = P-A-q/(p+q)*root*(P-Q), D = Q-Ccenter+
// p/(p+q)*root*(P-Q). No Cartesian block or later C2S is needed here.

// Identical binary64 literal to kRysDSolidInvSqrt3 in the legacy producer.
// A local name permits this header to parse even in a build excluding AVX2.
inline constexpr double kNativeDSolidInvSqrt3 =
    0.57735026918962576451;

inline constexpr double kNativeDSolidHessian[5][3][3] = {
    {{0.0, 1.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 0.0, 0.0}},
    {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}},
    {{-kNativeDSolidInvSqrt3, 0.0, 0.0},
     {0.0, -kNativeDSolidInvSqrt3, 0.0},
     {0.0, 0.0, 2.0 * kNativeDSolidInvSqrt3}},
    {{0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}},
    {{1.0, 0.0, 0.0}, {0.0, -1.0, 0.0}, {0.0, 0.0, 0.0}},
};

template<int W>
inline void native_d_solid_harmonics(
    const tlno::core::simd::Packet<W> X[3],
    tlno::core::simd::Packet<W> H[5],
    tlno::core::simd::Packet<W> dH[5][3]) noexcept {
    using V = tlno::core::simd::Packet<W>;
    const V zero, half(0.5), two(2.0), invsqrt3(kNativeDSolidInvSqrt3);
    const V x = X[0], y = X[1], z = X[2];
    const V x2 = x * x, y2 = y * y, z2 = z * z;

    H[0] = x * y;
    H[1] = y * z;
    H[2] = invsqrt3 * (z2 - half * (x2 + y2));
    H[3] = x * z;
    H[4] = half * (x2 - y2);

    dH[0][0] = y;    dH[0][1] = x;    dH[0][2] = zero;
    dH[1][0] = zero; dH[1][1] = z;    dH[1][2] = y;
    dH[2][0] = zero - invsqrt3 * x;
    dH[2][1] = zero - invsqrt3 * y;
    dH[2][2] = (two * invsqrt3) * z;
    dH[3][0] = z;    dH[3][1] = zero; dH[3][2] = x;
    dH[4][0] = x;    dH[4][1] = zero - y; dH[4][2] = zero;
}

// acc[5*a+b], with a,b in the engine's pure d order:
// xy, yz, (z*z-(x*x+y*y)/2)/sqrt(3), xz, (x*x-y*y)/2.
// weight already contains the primitive prefactor and this root's weight.
template<int W>
inline void native_dsds_contract(
    const tlno::core::simd::Packet<W> C[3],
    const tlno::core::simd::Packet<W> D[3],
    tlno::core::simd::Packet<W> b00,
    tlno::core::simd::Packet<W> weight,
    tlno::core::simd::Packet<W>* acc) noexcept {
    using V = tlno::core::simd::Packet<W>;
    const V b00sq = b00 * b00;
    V HC[5], HD[5], dHC[5][3], dHD[5][3];
    native_d_solid_harmonics<W>(C, HC, dHC);
    native_d_solid_harmonics<W>(D, HD, dHD);
    for (int a = 0; a < 5; ++a) {
        for (int b = 0; b < 5; ++b) {
            V value = HC[a] * HD[b];
            V dot = dHC[a][0] * dHD[b][0];
            dot = V::fma(dHC[a][1], dHD[b][1], dot);
            dot = V::fma(dHC[a][2], dHD[b][2], dot);
            value = V::fma(b00, dot, value);
            if (a == b) value = value + b00sq;
            const int component = 5 * a + b;
            acc[component] = V::fma(weight, value, acc[component]);
        }
    }
}

// Canonical native orientation only: bra=(d,s), ket=(p,p).
// cd is the ket center displacement Ccenter-Dcenter, i.e. native cd[] and
// legacy ket_R[]. b01 is the same-side ket covariance, not bra b10.
// acc[9*a+3*i+j] is already the final pure-spherical ds|pp row/column order.
template<int W>
inline void native_dspp_contract(
    const tlno::core::simd::Packet<W> C[3],
    const tlno::core::simd::Packet<W> D[3],
    const tlno::core::simd::Packet<W> cd[3],
    tlno::core::simd::Packet<W> b00,
    tlno::core::simd::Packet<W> b01,
    tlno::core::simd::Packet<W> weight,
    tlno::core::simd::Packet<W>* acc) noexcept {
    using V = tlno::core::simd::Packet<W>;
    const V b00sq = b00 * b00;
    V H[5], dH[5][3];
    native_d_solid_harmonics<W>(C, H, dH);
    for (int a = 0; a < 5; ++a) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
                const V shifted = D[j] + cd[j];
                V pp = D[i] * shifted;
                if (i == j) pp = pp + b01;
                V first = dH[a][i] * shifted;
                first = V::fma(dH[a][j], D[i], first);
                V value = V::fma(b00, first, H[a] * pp);
                value = V::fma(b00sq, V(kNativeDSolidHessian[a][i][j]), value);
                const int component = 9 * a + 3 * i + j;
                acc[component] = V::fma(weight, value, acc[component]);
            }
        }
    }
}

#endif
