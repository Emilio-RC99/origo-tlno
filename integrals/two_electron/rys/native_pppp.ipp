#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_PPPP_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_PPPP_IPP

// The AVX2 pp|pp producer widened through Packet<W>. Each lane remains an
// independent quartet. Contract the 9x9 pre-HRR radial basis first; apply
// center shifts once per contracted packet. Twelve root descriptors bound
// scratch and let each nine-component row remain in registers between roots.
// This is a candidate until numerical and paired timing gates pass.
template<class V> inline V native_pppp_mul(V a, V b) { return a * b; }
template<class V>
inline void native_pppp_axis(V C, V D,
                             V B00, V B10, V B01,
                             V G[3][3]) noexcept {
    const V one = V(1.0);
    const V two = V(2.0);
    G[0][0] = one;
    G[0][1] = D;
    G[0][2] = V::fma(D, D, B01);
    G[1][0] = C;
    G[1][1] = V::fma(C, D, B00);
    G[1][2] = V::fma(
        D, G[1][1],
        V::fma(B00, D, native_pppp_mul(B01, C)));
    G[2][0] = V::fma(C, C, B10);
    G[2][1] = V::fma(
        D, G[2][0], native_pppp_mul(native_pppp_mul(two, B00), C));
    G[2][2] = V::fma(
        D, G[2][1],
        V::fma(native_pppp_mul(two, B00), G[1][1],
               native_pppp_mul(B01, G[2][0])));
}

constexpr int native_pppp_degree[9][3] = {
    {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
    {2, 0, 0}, {1, 1, 0}, {1, 0, 1},
    {0, 2, 0}, {0, 1, 1}, {0, 0, 2}};

// Match the established twelve-descriptor blocking policy. W8 descriptors
// occupy 21 KiB in total; the 81 radial accumulators occupy about 5 KiB.
// Row contraction reuses nine accumulators over the block before storing.
constexpr int native_pppp_capacity = 12;
template<class V>
struct NativePPPPDescriptor {
    V weight;
    V axis[3][3][3];
};

template<class V>
__attribute__((noinline)) void native_pppp_accumulate_row(
    V* acc_row,
    int bra,
    const NativePPPPDescriptor<V>* descriptors,
    int descriptor_count) noexcept {
    V row[9];
#pragma GCC unroll 9
    for (int ket = 0; ket < 9; ++ket) row[ket] = acc_row[ket];
    const int bra_x = native_pppp_degree[bra][0];
    const int bra_y = native_pppp_degree[bra][1];
    const int bra_z = native_pppp_degree[bra][2];
    for (int descriptor = 0; descriptor < descriptor_count; ++descriptor) {
        const NativePPPPDescriptor<V>& radial = descriptors[descriptor];
        const V wz0 = native_pppp_mul(
            radial.weight, radial.axis[2][bra_z][0]);
        const V wz1 = native_pppp_mul(
            radial.weight, radial.axis[2][bra_z][1]);
        const V wz2 = native_pppp_mul(
            radial.weight, radial.axis[2][bra_z][2]);
        V xy = native_pppp_mul(
            radial.axis[0][bra_x][1], radial.axis[1][bra_y][0]);
        row[0] = V::fma(xy, wz0, row[0]);
        row[5] = V::fma(xy, wz1, row[5]);
        xy = native_pppp_mul(
            radial.axis[0][bra_x][0], radial.axis[1][bra_y][1]);
        row[1] = V::fma(xy, wz0, row[1]);
        row[7] = V::fma(xy, wz1, row[7]);
        xy = native_pppp_mul(
            radial.axis[0][bra_x][0], radial.axis[1][bra_y][0]);
        row[2] = V::fma(xy, wz1, row[2]);
        row[8] = V::fma(xy, wz2, row[8]);
        xy = native_pppp_mul(
            radial.axis[0][bra_x][2], radial.axis[1][bra_y][0]);
        row[3] = V::fma(xy, wz0, row[3]);
        xy = native_pppp_mul(
            radial.axis[0][bra_x][1], radial.axis[1][bra_y][1]);
        row[4] = V::fma(xy, wz0, row[4]);
        xy = native_pppp_mul(
            radial.axis[0][bra_x][0], radial.axis[1][bra_y][2]);
        row[6] = V::fma(xy, wz0, row[6]);
    }
#pragma GCC unroll 9
    for (int ket = 0; ket < 9; ++ket) acc_row[ket] = row[ket];
}

template<class V>
inline void native_pppp_accumulate_block(
    V* acc,
    const NativePPPPDescriptor<V>* descriptors,
    int descriptor_count) noexcept {
    for (int bra = 0; bra < 9; ++bra) {
        native_pppp_accumulate_row(
            acc + 9 * bra, bra, descriptors, descriptor_count);
    }
}


template<class V>
inline void native_pppp_finish(const V* radial, const V* ab, const V* cd, V* cart) {
    constexpr int unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    for (int ij = 0; ij < 9; ++ij) {
        const int i = ij / 3, j = ij % 3, hi_bra = 3 + unique[ij];
        for (int kl = 0; kl < 9; ++kl) {
            const int k = kl / 3, l = kl % 3, hi_ket = 3 + unique[kl];
            const V hi = V::fma(ab[j], radial[9 * i + hi_ket], radial[9 * hi_bra + hi_ket]);
            const V lo = V::fma(ab[j], radial[9 * i + k], radial[9 * hi_bra + k]);
            cart[9 * ij + kl] = V::fma(cd[l], lo, hi);
        }
    }
}
#endif
