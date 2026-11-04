#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_DP_FRAGMENT_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_DP_FRAGMENT_IPP

// Exact dp|ds and dp|pp spherical fragments from the tuned AVX2 producer.
// Each lane is an independent quartet. Root records are reused by five small
// output fragments; only one packet's primitive/root geometry is retained.
template<int W>
struct NativeDPMatrixRootRecord {
    using V = tlno::core::simd::Packet<W>;
    V weight, xdp[3], xother[3], b00, b00sq, bdp, bother;
};

template<int Component, int W>
inline void native_d_solid_component(const tlno::core::simd::Packet<W> x[3],
    tlno::core::simd::Packet<W>& h, tlno::core::simd::Packet<W> dh[3]) noexcept {
    using V = tlno::core::simd::Packet<W>;
    static_assert(Component >= 0 && Component < 5);
    if constexpr (Component == 0) {
        h = x[0] * x[1]; dh[0] = x[1]; dh[1] = x[0]; dh[2] = V();
    } else if constexpr (Component == 1) {
        h = x[1] * x[2]; dh[0] = V(); dh[1] = x[2]; dh[2] = x[1];
    } else if constexpr (Component == 2) {
        const V k(kNativeDSolidInvSqrt3);
        h = k * (x[2] * x[2] - V(.5) * (x[0] * x[0] + x[1] * x[1]));
        dh[0] = V() - k * x[0]; dh[1] = V() - k * x[1]; dh[2] = V(2.0 * kNativeDSolidInvSqrt3) * x[2];
    } else if constexpr (Component == 3) {
        h = x[0] * x[2]; dh[0] = x[2]; dh[1] = V(); dh[2] = x[0];
    } else {
        h = V(.5) * (x[0] * x[0] - x[1] * x[1]); dh[0] = x[0]; dh[1] = V() - x[1]; dh[2] = V();
    }
}

template<int DComponent, bool OtherPP, int W>
inline void native_dp_matrix_fragment(
    const std::vector<NativeDPMatrixRootRecord<W>>& records,
    const tlno::core::simd::Packet<W> rdp[3],
    const tlno::core::simd::Packet<W> rother[3],
    tlno::core::simd::Packet<W>* __restrict output) noexcept {
    using V = tlno::core::simd::Packet<W>;
    constexpr int NOther = OtherPP ? 9 : 5;
    V acc[3 * NOther]{};
    for (const auto& record : records) {
        V hd, dhd[3];
        native_d_solid_component<DComponent, W>(record.xdp, hd, dhd);
        V hother[5], dhother[5][3];
        if constexpr (!OtherPP) native_d_solid_harmonics<W>(record.xother, hother, dhother);
        for (int pair_p = 0; pair_p < 3; ++pair_p) {
            const V shifted_dp = record.xdp[pair_p] + rdp[pair_p];
            const V f = V::fma(record.bdp, dhd[pair_p], hd * shifted_dp);
            V df[3];
            for (int axis = 0; axis < 3; ++axis) {
                df[axis] = V::fma(record.bdp, V(kNativeDSolidHessian[DComponent][pair_p][axis]), dhd[axis] * shifted_dp);
                if (axis == pair_p) df[axis] = df[axis] + hd;
            }
            if constexpr (!OtherPP) {
                for (int other_d = 0; other_d < 5; ++other_d) {
                    V dot = df[0] * dhother[other_d][0];
                    dot = V::fma(df[1], dhother[other_d][1], dot);
                    dot = V::fma(df[2], dhother[other_d][2], dot);
                    V second = DComponent == other_d ? shifted_dp : V();
                    for (int axis = 0; axis < 3; ++axis) {
                        const double coefficient = kNativeDSolidHessian[other_d][axis][pair_p];
                        if (coefficient != 0.0) second = V::fma(V(coefficient), dhd[axis], second);
                    }
                    V value = V::fma(record.b00, dot, f * hother[other_d]);
                    value = V::fma(record.b00sq, second, value);
                    const int local = pair_p * NOther + other_d;
                    acc[local] = V::fma(record.weight, value, acc[local]);
                }
            } else {
                for (int pi = 0; pi < 3; ++pi) for (int pj = 0; pj < 3; ++pj) {
                    const V shifted_other = record.xother[pj] + rother[pj];
                    V pp = record.xother[pi] * shifted_other;
                    if (pi == pj) pp = pp + record.bother;
                    V first = df[pi] * shifted_other;
                    first = V::fma(df[pj], record.xother[pi], first);
                    V ddf = V(kNativeDSolidHessian[DComponent][pi][pj]) * shifted_dp;
                    if (pj == pair_p) ddf = ddf + dhd[pi];
                    if (pi == pair_p) ddf = ddf + dhd[pj];
                    V value = V::fma(record.b00, first, f * pp);
                    value = V::fma(record.b00sq, ddf, value);
                    const int local = pair_p * NOther + 3 * pi + pj;
                    acc[local] = V::fma(record.weight, value, acc[local]);
                }
            }
        }
    }
    for (int pair_p = 0; pair_p < 3; ++pair_p) for (int other = 0; other < NOther; ++other)
        output[(3 * DComponent + pair_p) * NOther + other] = acc[pair_p * NOther + other];
}
#endif
