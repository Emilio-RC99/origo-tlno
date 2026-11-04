#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SPECIALIZED_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SPECIALIZED_IPP

// Constant-angular-momentum native-width kernels. The recurrence is the
// established shell4 VRR/HRR recurrence expressed with Packet<W> operations.
// s/p classes omit sparse projection loops; ss|ss evaluates only Boys F0.
// Larger classes retain exact angular bounds and the existing C2S projectors.
#include "native_roots.ipp"
#include "native_sp_boys.ipp"
#include "native_d_direct.ipp"
#include "native_d_low.ipp"
#include "native_dp_fragment.ipp"
#include "native_pppp.ipp"

#ifndef TLNO_NATIVE_COMPACT_KET_DP
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_NATIVE_COMPACT_KET_DP 1
#else
#define TLNO_NATIVE_COMPACT_KET_DP 0
#endif
#endif
#if TLNO_NATIVE_COMPACT_KET_DP != 0 && TLNO_NATIVE_COMPACT_KET_DP != 1
#error "TLNO_NATIVE_COMPACT_KET_DP must be 0 or 1"
#endif
#ifndef TLNO_NATIVE_COMPACT_KET_FS
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_NATIVE_COMPACT_KET_FS 1
#else
#define TLNO_NATIVE_COMPACT_KET_FS 0
#endif
#endif
#if TLNO_NATIVE_COMPACT_KET_FS != 0 && TLNO_NATIVE_COMPACT_KET_FS != 1
#error "TLNO_NATIVE_COMPACT_KET_FS must be 0 or 1"
#endif

// W8 layouts retained with shared root interpolation after the production-path
// four-arm campaign: GM 1.044113x, nine cases, 35/36 paired wins on Xeon 8573C.
// This is a bundle result, not an isolated layout result.
// W2/W4 keep their explicit opt-in.
// A dp ket needs degrees 2..3; an fs ket needs only degree 3. Their lower
// cumulative Cartesian columns (four and ten respectively) are unused.
// Keep the full per-axis VRR (including its lower-degree dependencies); only
// omit these columns from the three-axis contraction and subsequent storage.
template<int La, int Lb, int Lc, int Ld>
struct NativeRadialKetLayout {
    static constexpr bool compact_dp = TLNO_NATIVE_COMPACT_KET_DP &&
        La == 2 && (Lb == 1 || Lb == 2) && Lc == 2 && Ld == 1;
    static constexpr bool compact_fs = TLNO_NATIVE_COMPACT_KET_FS &&
        La == 3 && (Lb == 2 || Lb == 3) && Lc == 3 && Ld == 0;
    static constexpr bool compact = compact_dp || compact_fs;
    static constexpr int offset = compact ? offset_for_L(Lc) : 0;
    static constexpr int columns = n_cart_cum(Lc + Ld) - offset;
    static constexpr int full_index(int local) { return offset + local; }
    static constexpr int local_index(int full) { return full - offset; }
};

template<int L>
constexpr std::array<int, 3> native_cartesian_power(int index) {
    int k = 0;
    for (int x = L; x >= 0; --x) for (int y = L - x; y >= 0; --y)
        if (k++ == index) return {x, y, L - x - y};
    return {0, 0, 0};
}
template<class F, std::size_t... I>
inline void native_static_for(F&& f, std::index_sequence<I...>) {
    (f(std::integral_constant<std::size_t, I>{}), ...);
}

// Closed low-L Gaussian moments, as in the tuned s/p portfolio. Each Packet
// component still belongs to a distinct quartet; no recurrence scratch is
// constructed for ps|ss, ps|ps, pp|ss or pp|ps.
template<int A, int B, int C, int D, class V>
inline void native_sp_contract(const V* c, const V* d, const V* ab,
    V b00, V b10, V weight, V* __restrict output) {
    constexpr int Count = n_cart(A) * n_cart(B) * n_cart(C) * n_cart(D);
    V bra[9];
    if constexpr (B == 1) native_static_for([&](auto index) {
        constexpr int i = decltype(index)::value / 3, j = decltype(index)::value % 3;
        V v = V::fma(c[i], c[j] + ab[j], V());
        if constexpr (i == j) v = v + b10;
        bra[i * 3 + j] = v;
    }, std::make_index_sequence<9>{});
    native_static_for([&](auto index) {
        constexpr int o = decltype(index)::value;
        V value;
        if constexpr (A == 0) value = V(1.0);
        else if constexpr (B == 0 && C == 0) value = c[o];
        else if constexpr (B == 0 && C == 1) {
            constexpr int i = o / 3, k = o % 3;
            value = c[i] * d[k]; if constexpr (i == k) value = value + b00;
        } else if constexpr (B == 1 && C == 0) value = bra[o];
        else {
            constexpr int i = o / 9, j = (o / 3) % 3, k = o % 3;
            value = bra[i * 3 + j] * d[k];
            if constexpr (i == k) value = V::fma(b00, c[j] + ab[j], value);
            if constexpr (j == k) value = V::fma(b00, c[i], value);
        }
        output[o] = V::fma(weight, value, output[o]);
    }, std::make_index_sequence<Count>{});
}

template<int A, int B, int C, int D>
struct NativeStaticShape {
    static constexpr int la = A, lb = B, lc = C, ld = D, lab = A + B, lcd = C + D, gk = C + D + 1;
    static constexpr int roots = (A + B + C + D) / 2 + 1;
    static constexpr int axis_size = (A + 1) * (B + 1) * (C + D + 1) * (D + 1);
    static constexpr int lhs_cart = n_cart(A) * n_cart(B), rhs_cart = n_cart(C) * n_cart(D);
    std::array<int, lhs_cart * rhs_cart> ix{}, iy{}, iz{};
    constexpr NativeStaticShape() {
        int o = 0;
        for (int a = 0; a < n_cart(A); ++a) for (int b = 0; b < n_cart(B); ++b)
          for (int c = 0; c < n_cart(C); ++c) for (int d = 0; d < n_cart(D); ++d, ++o) {
            const auto av = native_cartesian_power<A>(a), bv = native_cartesian_power<B>(b);
            const auto cv = native_cartesian_power<C>(c), dv = native_cartesian_power<D>(d);
            ix[o] = ((av[0] * (B + 1) + bv[0]) * gk + cv[0]) * (D + 1) + dv[0];
            iy[o] = ((av[1] * (B + 1) + bv[1]) * gk + cv[1]) * (D + 1) + dv[1];
            iz[o] = ((av[2] * (B + 1) + bv[2]) * gk + cv[2]) * (D + 1) + dv[2];
          }
    }
};

// Compact radial VRR from the tuned AVX2 producer. Horizontal transfer and
// spherical conversion are fused into the pair projectors after contraction.
template<int LAB, int LCD, class V>
inline void native_radial_axis(V c, V d, V b00, V b10, V b01, V seed, V* g) {
    constexpr int K = LCD + 1;
    g[0] = seed;
    for (int i = 0; i < LAB; ++i) {
        V v = c * g[i * K];
        if (i) v = V::fma(V(double(i)) * b10, g[(i - 1) * K], v);
        g[(i + 1) * K] = v;
    }
    for (int k = 0; k < LCD; ++k) for (int i = 0; i <= LAB; ++i) {
        V v = d * g[i * K + k];
        if (i) v = V::fma(V(double(i)) * b00, g[(i - 1) * K + k], v);
        if (k) v = V::fma(V(double(k)) * b01, g[i * K + k - 1], v);
        g[i * K + k + 1] = v;
    }
}

// Retained singleton ps|ss producer: broadcast one bra and load consecutive
// kets into eight independent lanes before forming primitive intermediates.
// The structural guard preserves the screened fallback and avoids the gather
// cost of the rejected general packet producer. Native C/E: water 1.104x,
// nine-case GM 1.012x; the other eight cases were approximately neutral.
// The singleton algorithm and its invariants are described in native_singleton_psss.ipp.
#ifndef TLNO_NATIVE_SINGLETON_PSSS
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_NATIVE_SINGLETON_PSSS 1
#else
#define TLNO_NATIVE_SINGLETON_PSSS 0
#endif
#endif
#if TLNO_NATIVE_SINGLETON_PSSS != 0 && TLNO_NATIVE_SINGLETON_PSSS != 1
#error "TLNO_NATIVE_SINGLETON_PSSS must be 0 or 1"
#endif
#if TLNO_NATIVE_SINGLETON_PSSS && defined(TLNO_NATIVE_FLATTENED_PSSS) && TLNO_NATIVE_FLATTENED_PSSS
#error "Singleton producer requires TLNO_NATIVE_FLATTENED_PSSS=0"
#endif
#if TLNO_NATIVE_SINGLETON_PSSS && defined(__AVX512F__) && defined(__FMA__)
#include "native_singleton_psss.ipp"
#endif
#if TLNO_NATIVE_SSSS_KET_TILE && defined(__AVX512F__) && defined(__FMA__)
#include "native_ssss_wide.ipp"
#endif
#ifndef TLNO_NATIVE_SPARSE_TRANSPOSED_J
// Production AVX-512 direct J keeps exact-zero original bra packets local
// across ket rows. Smaller native widths retain their existing producer.
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_NATIVE_SPARSE_TRANSPOSED_J 1
#else
#define TLNO_NATIVE_SPARSE_TRANSPOSED_J 0
#endif
#endif
#if TLNO_NATIVE_SPARSE_TRANSPOSED_J != 0 && TLNO_NATIVE_SPARSE_TRANSPOSED_J != 1
#error "TLNO_NATIVE_SPARSE_TRANSPOSED_J must be 0 or 1"
#endif
#if TLNO_NATIVE_SPARSE_TRANSPOSED_J && defined(__AVX512F__) && defined(__FMA__)
#include "native_sparse_transposed_j.ipp"
#endif
template<int W, int La, int Lb, int Lc, int Ld, class Emit, bool DPFragment = false>
void specialized_quartet_bin_pair(const BasisDataHGP& bd, const NativePairBin& lhs,
                                  const NativePairBin& rhs, const NativeQuartetShape& shape,
                                  bool same_bin, const std::vector<double>* metric,
                                  double threshold, double primitive_squared,
                                  std::vector<double>& values) {
    using V = tlno::core::simd::Packet<W>;
#if TLNO_NATIVE_SSSS_KET_TILE && defined(__AVX512F__) && defined(__FMA__)
    if constexpr (W == 8 && La == 0 && Lb == 0 && Lc == 0 && Ld == 0) {
        if (native_ssss_wide_bin_pair<TLNO_NATIVE_SSSS_KET_TILE, Emit>(
                bd, lhs, rhs, same_bin, metric, threshold, primitive_squared, values)) return;
    }
#endif
#if TLNO_NATIVE_SINGLETON_PSSS && defined(__AVX512F__) && defined(__FMA__)
    if constexpr (W == 8 && La == 1 && Lb == 0 && Lc == 0 && Ld == 0) {
        if (native_singleton_psss_bin_pair<Emit>(bd, lhs, rhs, same_bin, metric,
                threshold, primitive_squared, values)) return;
    }
#endif
    // For equal angular shapes either pair orientation uses this exact
    // specialization. Choose the orientation with fewer padded W8 packets;
    // the emitter receives the corresponding pair references in that order.
    // Same-bin triangular enumeration retains its original ordering.
    if constexpr (W == 8 && La == Lc && Lb == Ld) {
        if (!same_bin && ((rhs.ids.size() + W - 1) / W) * lhs.ids.size() <
                         ((lhs.ids.size() + W - 1) / W) * rhs.ids.size()) {
            specialized_quartet_bin_pair<W, La, Lb, Lc, Ld, Emit>(
                bd, rhs, lhs, shape, false,
                metric, threshold, primitive_squared, values);
            return;
        }
    }
    using Shape = NativeStaticShape<La, Lb, Lc, Ld>;
    static constexpr Shape s{};
    constexpr int N = Shape::roots;
    constexpr int Key = 10 * (La * (La + 1) / 2 + Lb) + (Lc * (Lc + 1) / 2 + Ld);
    constexpr bool PPPP = W == 8 && La == 1 && Lb == 1 && Lc == 1 && Ld == 1;
    constexpr bool DirectDSDS = W == 8 && La == 2 && Lb == 0 && Lc == 2 && Ld == 0;
    constexpr bool DirectDSPP = W == 8 && La == 2 && Lb == 0 && Lc == 1 && Ld == 1;
    constexpr bool DirectDLow = W == 8 && La == 2 && Lb <= 1 && Lc <= 1 && Ld == 0;
    constexpr bool DirectD = DirectDSDS || DirectDSPP || DPFragment || DirectDLow;
    static_assert(!DPFragment || (W == 8 && La == 2 && Lb == 1 &&
        ((Lc == 2 && Ld == 0) || (Lc == 1 && Ld == 1))));
    constexpr bool Radial = !DirectD && (La >= 2 || Lc >= 2) && Key != 53 && Key != 63 && Key != 64 && Key != 66 && Key != 73;
    using KetLayout = NativeRadialKetLayout<La, Lb, Lc, Ld>;
    constexpr int Br = n_cart_cum(La + Lb) - offset_for_L(La), Kr = KetLayout::columns;
    constexpr int Axis = (La + Lb + 1) * (Lc + Ld + 1);
    constexpr int NOut = Radial ? Br * Kr : Shape::lhs_cart * Shape::rhs_cart;
    struct RadialMap {
        std::array<int, Radial ? Br * Kr : 1> x{}, y{}, z{};
        RadialMap() { if constexpr (Radial) for (int a = 0; a < Br; ++a) for (int c = 0; c < Kr; ++c) {
            const auto& av = g_cart.at(offset_for_L(La) + a);
            const auto& cv = g_cart.at(KetLayout::full_index(c));
            x[a * Kr + c] = av[0] * (Lc + Ld + 1) + cv[0];
            y[a * Kr + c] = av[1] * (Lc + Ld + 1) + cv[1];
            z[a * Kr + c] = av[2] * (Lc + Ld + 1) + cv[2];
        }}
    };
    static const RadialMap rmap;
    struct Scratch { std::vector<V> gx, gy, gz, g, bra, cart, mid, sph, projector, j_delta; std::vector<int> offsets, columns; std::vector<unsigned char> j_zero_packets; std::vector<NativeDPMatrixRootRecord<W>> records; std::array<NativePPPPDescriptor<V>, PPPP ? native_pppp_capacity : 1> pp_descriptors; std::array<V, PPPP ? 81 : 1> pp_radial; };
    thread_local Scratch scratch;
    auto& t = scratch;
    if constexpr (DPFragment) t.records.reserve(lhs.primitives * rhs.primitives * N);
    if constexpr (!DirectD) {
        t.gx.resize(Radial ? N * Axis : s.axis_size); t.gy.resize(Radial ? N * Axis : s.axis_size); t.gz.resize(Radial ? N * Axis : s.axis_size);
        t.g.resize((s.lab + 1) * s.gk); t.bra.resize((s.lab + 1) * (s.lb + 1) * s.gk);
        t.cart.resize(NOut);
    }
    const auto& lp = get_final_sph_pair_projector(s.la, s.lb);
    const auto& rp = get_final_sph_pair_projector(s.lc, s.ld);
    if constexpr (!DirectD) t.mid.resize(lp.sph_rows * (Radial ? Kr : s.rhs_cart));
    t.sph.resize(lp.sph_rows * rp.sph_rows);
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
    static_assert(W == tlno::core::simd::lanes, "radial cache packet width must match schedule width");
    if constexpr (Radial) {
        if (!lhs.radial_projector_cache || lhs.radial_projector_cache->rows != lp.sph_rows ||
           lhs.radial_projector_cache->br != Br)
            throw std::logic_error("native radial cache missing or incompatible with specialization");
    }
#else
    if constexpr (Radial) { t.projector.resize(lp.sph_rows * Br); t.offsets.resize(lp.sph_rows + 1); }
#endif
    const auto* density = g_rys_density_screen;
    const auto* secondary = g_native_secondary_shell_screen;
    const auto* compact_bra = native_j_compact_bin(lhs);
    const auto* compact_positions = native_j_compact_positions(lhs);
    const auto* full_bounds = native_shell_packet_bounds(lhs);
    const auto* compact_bounds = compact_bra ? native_shell_packet_bounds(lhs, true) : nullptr;
    const double secondary_threshold = secondary ? secondary->threshold : 0.0;
#if TLNO_NATIVE_SPARSE_TRANSPOSED_J
    const auto* transpose_context = rys_jk_context_slot();
#endif
    // Both traversal orders call this exact packet body. Root evaluation,
    // primitive rejection, recurrence and spherical projection are shared.
    auto produce_packet = [&](const NativePairBin& bra, std::uint32_t ket_id,
                            std::size_t start, int valid, bool transposed)->bool {
        std::uint32_t ids[W]; bool active[W]; bool any = false;
        for (int lane = 0; lane < W; ++lane) {
            ids[lane] = bra.ids[start + (lane < valid ? lane : 0)];
            const double q = metric ? (*metric)[ids[lane]] * (*metric)[ket_id] : 0.0;
            active[lane] = lane < valid && (!metric || (q >= threshold &&
                native_secondary_shell_live(secondary, ids[lane], ket_id))) &&
                (!density || native_density_live(density, ids[lane], ket_id, q));
            any = any || active[lane];
        }
        if (!any) return false;
        const auto [gc, gd] = bd.pair_groups[ket_id];
        const auto& C = bd.groups[gc].center; const auto& D = bd.groups[gd].center;
        const auto& kets = bd.rys_pair_primitives[ket_id];
        const V av[3] = {V::load(bra.ax.data() + start), V::load(bra.ay.data() + start), V::load(bra.az.data() + start)};
        const V ab[3] = {V::load(bra.abx.data() + start), V::load(bra.aby.data() + start), V::load(bra.abz.data() + start)};
        const V cd[3] = {V(C[0] - D[0]), V(C[1] - D[1]), V(C[2] - D[2])};
        double mask_data[W]; for (int lane = 0; lane < W; ++lane) mask_data[lane] = active[lane] ? 1.0 : 0.0;
        const V mask = V::load(mask_data);
#if !TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
        if constexpr (Radial) {
            std::fill(t.projector.begin(), t.projector.end(), V());
            // Lane-major geometry is converted once per contracted packet;
            // coefficients are then consumed in component-major SIMD order.
            thread_local std::vector<double> coeff;
            coeff.assign(std::size_t(lp.sph_rows) * Br * W, 0.0);
            for (int lane = 0; lane < W; ++lane) {
                const auto& rows = bd.rys_radial_pair_projectors[ids[lane]].bra_rows;
                for (int row = 0; row < lp.sph_rows; ++row) for (const auto& term : rows[row])
                    coeff[(row * Br + term.index) * W + lane] = term.coeff;
            }
            t.columns.clear(); t.offsets[0] = 0;
            for (int row = 0; row < lp.sph_rows; ++row) {
                for (int a = 0; a < Br; ++a) {
                    const double* c = coeff.data() + (row * Br + a) * W;
                    bool live = false; for (int lane = 0; lane < valid; ++lane) live = live || c[lane] != 0.0;
                    if (live) { t.columns.push_back(a); t.projector[row * Br + a] = V::load(c); }
                }
                t.offsets[row + 1] = static_cast<int>(t.columns.size());
            }
        }
#endif
        int pp_count = 0;
        if constexpr (PPPP) std::fill(t.pp_radial.begin(), t.pp_radial.end(), V());
        if constexpr (DPFragment) t.records.clear();
        else if constexpr (DirectD) std::fill(t.sph.begin(), t.sph.end(), V());
        else if constexpr (!PPPP) std::fill(t.cart.begin(), t.cart.end(), V());
        for (std::size_t ip = 0; ip < bra.primitives; ++ip) {
          const std::size_t off = ip * bra.stride + start;
          const V p = V::load(bra.exponent.data() + off);
          const V inv2p = V(.5) * V::load(bra.inverse.data() + off);
          const V pv[3] = {V::load(bra.px.data() + off), V::load(bra.py.data() + off), V::load(bra.pz.data() + off)};
          const V bw = mask * V::load(bra.weight.data() + off);
          for (const auto& kp : kets) {
            if (primitive_squared > 0) {
                bool primitive_live = false;
                for (int l = 0; l < valid; ++l) primitive_live = primitive_live ||
                    (active[l] && bra.schwarz[off + l] * kp.fast_schwarz >= primitive_squared);
                if (!primitive_live) continue;
            }
            const V q(kp.s), sum = p + q, invsum = V(1.0) / sum;
            const V qo = q * invsum, po = p * invsum, inv2pq = V(.5) * invsum, inv2q(.5 * kp.invs);
            const V qv[3] = {V(kp.Wx), V(kp.Wy), V(kp.Wz)};
            const V pq[3] = {pv[0] - qv[0], pv[1] - qv[1], pv[2] - qv[2]};
            const V targ = p * q * invsum * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
            const V pref = V(kTwoPi52 * kp.w) * bw * V::sqrt(invsum);
#if defined(__AVX512F__) && defined(__FMA__)
            // Integrate the low-degree moment polynomial directly with the
            // established bounded Boys approximants. Outside their domain,
            // retain the original Rys root/weight recurrence below.
            if constexpr (W == 8 && ((La == 0 && Lb == 0 && Lc == 0 && Ld == 0) ||
                         (La == 1 && Lb == 0 && Lc <= 1 && Ld == 0) ||
                         (La == 1 && Lb == 1 && Lc == 0 && Ld == 0))) {
                const V pa[3] = {pv[0] - av[0], pv[1] - av[1], pv[2] - av[2]};
                const V qc[3] = {qv[0] - V(C[0]), qv[1] - V(C[1]), qv[2] - V(C[2])};
                if (native_sp_boys_contract8<La, Lb, Lc, Ld>(targ, pa, qc, pq, ab,
                        qo, po, inv2p, inv2pq, pref, t.cart.data())) continue;
            }
#endif
            alignas(64) double ts[W], roots[N * W], weights[N * W]; targ.store(ts);
            if constexpr (La + Lb + Lc + Ld == 0) {
                native_weight1<W>(ts, weights);
                t.cart[0] = V::fma(pref, V::load(weights), t.cart[0]);
                continue;
            } else native_roots_weights<N, W>(ts, roots, weights);
            for (int r = 0; r < N; ++r) {
                const V root = V::load(roots + r * W);
                const V b00 = inv2pq * root, b10 = inv2p * (V(1.0) - qo * root), b01 = inv2q * (V(1.0) - po * root);
                V* axes[3] = {t.gx.data(), t.gy.data(), t.gz.data()};
                if constexpr (DirectD) {
                    const V c[3] = {pv[0] - av[0] - qo * pq[0] * root, pv[1] - av[1] - qo * pq[1] * root, pv[2] - av[2] - qo * pq[2] * root};
                    const V d[3] = {qv[0] - V(C[0]) + po * pq[0] * root, qv[1] - V(C[1]) + po * pq[1] * root, qv[2] - V(C[2]) + po * pq[2] * root};
                    const V weight = pref * V::load(weights + r * W);
                    if constexpr (DPFragment) t.records.push_back({weight, {c[0], c[1], c[2]},
                        {d[0], d[1], d[2]}, b00, b00 * b00, b10, b01});
                    else if constexpr (DirectDLow) native_d_low_contract<W, Lb == 1, Lc == 1>(c, d, ab, b00, b10, weight, t.sph.data());
                    else if constexpr (DirectDSDS) native_dsds_contract<W>(c, d, b00, weight, t.sph.data());
                    else native_dspp_contract<W>(c, d, cd, b00, b01, weight, t.sph.data());
                } else if constexpr (PPPP) {
                    auto& descriptor = t.pp_descriptors[pp_count++];
                    descriptor.weight = pref * V::load(weights + r * W);
                    for (int axis = 0; axis < 3; ++axis)
                        native_pppp_axis(pv[axis] - av[axis] - qo * pq[axis] * root,
                            qv[axis] - V(C[axis]) + po * pq[axis] * root, b00, b10, b01,
                            descriptor.axis[axis]);
                    if (pp_count == native_pppp_capacity) {
                        native_pppp_accumulate_block(t.pp_radial.data(), t.pp_descriptors.data(), pp_count);
                        pp_count = 0;
                    }
                } else if constexpr (La <= 1 && Lb <= 1 && Lc <= 1 && Ld <= 1 && La + Lb + Lc + Ld <= 3) {
                    const V c[3] = {pv[0] - av[0] - qo * pq[0] * root, pv[1] - av[1] - qo * pq[1] * root, pv[2] - av[2] - qo * pq[2] * root};
                    const V d[3] = {qv[0] - V(C[0]) + po * pq[0] * root, qv[1] - V(C[1]) + po * pq[1] * root, qv[2] - V(C[2]) + po * pq[2] * root};
                    native_sp_contract<La, Lb, Lc, Ld>(c, d, ab, b00, b10, pref * V::load(weights + r * W), t.cart.data());
                } else if constexpr (Radial) {
                    for (int axis = 0; axis < 3; ++axis)
                        native_radial_axis<La + Lb, Lc + Ld>(pv[axis] - av[axis] - qo * pq[axis] * root,
                            qv[axis] - V(C[axis]) + po * pq[axis] * root, b00, b10, b01,
                            axis == 2 ? pref * V::load(weights + r * W) : V(1.0), axes[axis] + r * Axis);
                } else {
                    for (int axis = 0; axis < 3; ++axis)
                        native_axis(s, pv[axis] - av[axis] - qo * pq[axis] * root,
                            qv[axis] - V(C[axis]) + po * pq[axis] * root, b00, b10, b01, ab[axis], cd[axis],
                            axis == 2 ? pref * V::load(weights + r * W) : V(1.0), axes[axis], t.g.data(), t.bra.data());
                    for (int o = 0; o < NOut; ++o)
                        t.cart[o] = V::fma(t.gx[s.ix[o]] * t.gy[s.iy[o]], t.gz[s.iz[o]], t.cart[o]);
                }
            }
            if constexpr (Radial) for (int o = 0; o < NOut; ++o) {
                V value = t.cart[o];
                for (int r = 0; r < N; ++r) value = V::fma(t.gx[r * Axis + rmap.x[o]] * t.gy[r * Axis + rmap.y[o]], t.gz[r * Axis + rmap.z[o]], value);
                t.cart[o] = value;
            }
          }
        }
        if constexpr (PPPP) {
            if (pp_count) native_pppp_accumulate_block(t.pp_radial.data(), t.pp_descriptors.data(), pp_count);
            native_pppp_finish(t.pp_radial.data(), ab, cd, t.cart.data());
        }
        if constexpr (DPFragment) {
            native_dp_matrix_fragment<0, Lc == 1, W>(t.records, ab, cd, t.sph.data());
            native_dp_matrix_fragment<1, Lc == 1, W>(t.records, ab, cd, t.sph.data());
            native_dp_matrix_fragment<2, Lc == 1, W>(t.records, ab, cd, t.sph.data());
            native_dp_matrix_fragment<3, Lc == 1, W>(t.records, ab, cd, t.sph.data());
            native_dp_matrix_fragment<4, Lc == 1, W>(t.records, ab, cd, t.sph.data());
        } else if constexpr (DirectD) {
            // Already normalized and in final spherical component order.
        } else if constexpr (Radial) {
            const auto& ket_rows = bd.rys_radial_pair_projectors[ket_id].ket_rows;
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
            const auto& cached = bra.radial_projector_cache->packets[start / W];
            const auto& prefix = cached.prefixes[cached.prefix_index[valid - 1]];
            const auto& projector = cached.projector;
            const auto& offsets = prefix.offsets;
            const auto& columns = prefix.columns;
            // Valid lanes have exactly the original coefficients and sparse
            // term order. Cached lanes beyond this triangular/tail prefix are
            // independent SIMD lanes and are never presented to the emitter.
#endif
            for (int i = 0; i < lp.sph_rows; ++i) {
                for (int j = 0; j < Kr; ++j) {
                    V v;
#if TLNO_NATIVE_RADIAL_PROJECTOR_CACHE
                    for (int n = offsets[i]; n < offsets[i + 1]; ++n) {
                        const int a = columns[n]; v = V::fma(projector[i * Br + a], t.cart[a * Kr + j], v);
                    }
#else
                    for (int n = t.offsets[i]; n < t.offsets[i + 1]; ++n) {
                        const int a = t.columns[n]; v = V::fma(t.projector[i * Br + a], t.cart[a * Kr + j], v);
                    }
#endif
                    t.mid[i * Kr + j] = v;
                }
                for (int j = 0; j < rp.sph_rows; ++j) {
                    V v; for (const auto& term : ket_rows[j]) v = V::fma(V(term.coeff), t.mid[i * Kr + KetLayout::local_index(term.index)], v);
                    t.sph[i * rp.sph_rows + j] = v;
                }
            }
        } else if constexpr (La <= 1 && Lb <= 1 && Lc <= 1 && Ld <= 1) {
            // s/p projectors are permutations with one coefficient per row.
            for (int i = 0; i < lp.sph_rows; ++i) for (int j = 0; j < rp.sph_rows; ++j) {
                const auto& a = lp.rows[i][0]; const auto& b = rp.rows[j][0];
                t.sph[i * rp.sph_rows + j] = V(a.coeff * b.coeff) * t.cart[a.index * s.rhs_cart + b.index];
            }
        } else for (int i = 0; i < lp.sph_rows; ++i) {
            for (int j = 0; j < s.rhs_cart; ++j) {
                V v;
                for (const auto& term : lp.rows[i]) v = V::fma(V(term.coeff), t.cart[term.index * s.rhs_cart + j], v);
                t.mid[i * s.rhs_cart + j] = v;
            }
            for (int j = 0; j < rp.sph_rows; ++j) {
                V v;
                for (const auto& term : rp.rows[j]) v = V::fma(V(term.coeff), t.mid[i * s.rhs_cart + term.index], v);
                t.sph[i * rp.sph_rows + j] = v;
            }
        }
#if TLNO_NATIVE_SPARSE_TRANSPOSED_J && defined(__AVX512F__) && defined(__FMA__)
        if constexpr (W == 8) if (transposed) {
            const auto& refs = bd.packed_spherical_pair_components[ket_id];
            const auto* rho = transpose_context->j_pair_density;
            __mmask8 live_mask = 0;
            for (int lane = 0; lane < valid; ++lane)
                if (active[lane]) live_mask |= static_cast<__mmask8>(1u << lane);
            for (int i = 0; i < lp.sph_rows; ++i) {
                V delta = t.j_delta[i];
                for (const auto& ref : refs)
                    // mask3 preserves delta in inactive original lanes,
                    // even when an unobserved inactive ERI is non-finite.
                    delta = V(_mm512_mask3_fmadd_pd(
                        t.sph[std::size_t(i) * rp.sph_rows + ref.component_index].value,
                        _mm512_set1_pd(rho[ref.pair_index]), delta.value, live_mask));
                t.j_delta[i] = delta;
            }
#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || defined(TLNO_JK_COVERAGE_CHECKSUM)
            thread_local std::vector<double> audit_scalar;
            audit_scalar.resize(t.sph.size() * W);
            for (std::size_t i = 0; i < t.sph.size(); ++i) t.sph[i].store(audit_scalar.data() + i * W);
            for (int lane = 0; lane < valid; ++lane) if (active[lane]) {
                for (const auto& l : bd.packed_spherical_pair_components[ids[lane]]) {
                    for (const auto& r : refs) {
                        const auto index = (std::size_t(l.component_index) * rp.sph_rows + r.component_index) * W + lane;
                        rys_j_speed_probe_record_scalar(*transpose_context, l.pair_index, r.pair_index, audit_scalar[index]);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
                    }
                }
            }
#endif
            return true;
        }
#else
        (void)transposed;
#endif
        // Explicit stores avoid aliasing a SIMD wrapper as a scalar array.
        thread_local std::vector<double> scalar;
        scalar.resize(t.sph.size() * W);
        for (std::size_t i = 0; i < t.sph.size(); ++i) t.sph[i].store(scalar.data() + i * W);
        for (int lane = 0; lane < valid; ++lane) {
            if (!active[lane]) continue;
            rys_emit_spherical_lane_block<Emit>(values,
                bd.packed_spherical_pair_components[ids[lane]],
                bd.packed_spherical_pair_components[ket_id], ids[lane] == ket_id,
                [&](const PackedPairComponentH& l, const PackedPairComponentH& r) {
                    return scalar[(std::size_t(l.component_index) * rp.sph_rows + r.component_index) * W + lane];
                });
        }
        return true;
    };
#if TLNO_NATIVE_SPARSE_TRANSPOSED_J && defined(__AVX512F__) && defined(__FMA__)
    if constexpr (W == 8 && std::is_same_v<Emit, RysDirectJCoulombEmit>) {
        t.j_delta.resize(lp.sph_rows);
        if (native_transposed_zero_bra_j<W>(bd, lhs, rhs, same_bin, metric, threshold,
                t.j_delta, t.j_zero_packets, [&](std::uint32_t ket_id, std::size_t start, int valid) {
                    return produce_packet(lhs, ket_id, start, valid, true);
                })) return;
    }
#endif
    const auto rows = native_weighted_row_range(lhs, rhs, same_bin, W);
    for (std::size_t ki = rows.begin; ki < rows.end; ki += rows.step) {
      const auto ket_id = rhs.ids[ki];
      const bool ket_zero = density && density->j_only &&
          density->shell_rho[ket_id] == 0.0;
      const auto& bra = ket_zero && compact_bra ? *compact_bra : lhs;
      const auto* bounds = ket_zero && compact_bra ? compact_bounds : full_bounds;
      const double primary_other = metric ? (*metric)[ket_id] : 0.0;
      const double secondary_other = secondary ? (*secondary->metric)[ket_id] : 0.0;
      if (bounds && !bounds->row_live(primary_other, threshold,
                                    secondary_other, secondary_threshold)) continue;
      const std::size_t end = same_bin
          ? (ket_zero && compact_positions
                 ? static_cast<std::size_t>(std::upper_bound(
                       compact_positions->begin(), compact_positions->end(), ki)
                       -compact_positions->begin())
                 : ki + 1)
          : bra.ids.size();
      for (std::size_t start = 0; start < end; start += W) {
#if TLNO_NATIVE_SPARSE_TRANSPOSED_J && TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J && defined(__AVX512F__) && defined(__FMA__)
        if constexpr (W == 8 && std::is_same_v<Emit, RysDirectJCoulombEmit>) {
            // Skip only original full-bra packets handled above. Compact
            // packets for zero ket blocks retain their complete old path.
            if (!ket_zero && !t.j_zero_packets.empty() && t.j_zero_packets[start / W]) continue;
        }
#endif
        if (bounds && !bounds->packet_live(start, primary_other, threshold,
                                         secondary_other, secondary_threshold)) continue;
        const int valid = static_cast<int>(std::min<std::size_t>(W, end - start));
        produce_packet(bra, ket_id, start, valid, false);
      }
    }
}

template<int W, class Emit>
bool dispatch_native_specialized(const BasisDataHGP& bd, const NativePairBin& lhs,
    const NativePairBin& rhs, const NativeQuartetShape& shape, bool same_bin,
    const std::vector<double>* metric, double threshold, double primitive_squared,
    std::vector<double>& values) {
    // Bin ordering and pair orientation establish A>=B, C>=D, AB>=CD.
    // Every s-through-f class has an exact shape instantiation; g stays in
    // the independently validated residual producer.
    switch (native_shape_key(lhs.la, lhs.lb, rhs.la, rhs.lb)) {
    case 0: specialized_quartet_bin_pair<W, 0, 0, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 125: specialized_quartet_bin_pair<W, 1, 0, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 130: specialized_quartet_bin_pair<W, 1, 0, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 150: specialized_quartet_bin_pair<W, 1, 1, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 155: specialized_quartet_bin_pair<W, 1, 1, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 156: specialized_quartet_bin_pair<W, 1, 1, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 250: specialized_quartet_bin_pair<W, 2, 0, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 255: specialized_quartet_bin_pair<W, 2, 0, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 256: specialized_quartet_bin_pair<W, 2, 0, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 260: specialized_quartet_bin_pair<W, 2, 0, 2, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 275: specialized_quartet_bin_pair<W, 2, 1, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 280: specialized_quartet_bin_pair<W, 2, 1, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 281:
        if constexpr (W == 8) {
            if (lhs.ids.size() * rhs.ids.size() >= 256) {
                specialized_quartet_bin_pair<W, 2, 1, 1, 1, Emit, true>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
            }
        }
        specialized_quartet_bin_pair<W, 2, 1, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 285:
        if constexpr (W == 8) {
            if (lhs.ids.size() * rhs.ids.size() >= 256) {
                specialized_quartet_bin_pair<W, 2, 1, 2, 0, Emit, true>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
            }
        }
        specialized_quartet_bin_pair<W, 2, 1, 2, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 286: specialized_quartet_bin_pair<W, 2, 1, 2, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 300: specialized_quartet_bin_pair<W, 2, 2, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 305: specialized_quartet_bin_pair<W, 2, 2, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 306: specialized_quartet_bin_pair<W, 2, 2, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 310: specialized_quartet_bin_pair<W, 2, 2, 2, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 311: specialized_quartet_bin_pair<W, 2, 2, 2, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 312: specialized_quartet_bin_pair<W, 2, 2, 2, 2, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 375: specialized_quartet_bin_pair<W, 3, 0, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 380: specialized_quartet_bin_pair<W, 3, 0, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 381: specialized_quartet_bin_pair<W, 3, 0, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 385: specialized_quartet_bin_pair<W, 3, 0, 2, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 386: specialized_quartet_bin_pair<W, 3, 0, 2, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 387: specialized_quartet_bin_pair<W, 3, 0, 2, 2, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 390: specialized_quartet_bin_pair<W, 3, 0, 3, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 400: specialized_quartet_bin_pair<W, 3, 1, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 405: specialized_quartet_bin_pair<W, 3, 1, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 406: specialized_quartet_bin_pair<W, 3, 1, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 410: specialized_quartet_bin_pair<W, 3, 1, 2, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 411: specialized_quartet_bin_pair<W, 3, 1, 2, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 412: specialized_quartet_bin_pair<W, 3, 1, 2, 2, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 415: specialized_quartet_bin_pair<W, 3, 1, 3, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 416: specialized_quartet_bin_pair<W, 3, 1, 3, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 425: specialized_quartet_bin_pair<W, 3, 2, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 430: specialized_quartet_bin_pair<W, 3, 2, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 431: specialized_quartet_bin_pair<W, 3, 2, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 435: specialized_quartet_bin_pair<W, 3, 2, 2, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 436: specialized_quartet_bin_pair<W, 3, 2, 2, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 437: specialized_quartet_bin_pair<W, 3, 2, 2, 2, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 440: specialized_quartet_bin_pair<W, 3, 2, 3, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 441: specialized_quartet_bin_pair<W, 3, 2, 3, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 442: specialized_quartet_bin_pair<W, 3, 2, 3, 2, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 450: specialized_quartet_bin_pair<W, 3, 3, 0, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 455: specialized_quartet_bin_pair<W, 3, 3, 1, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 456: specialized_quartet_bin_pair<W, 3, 3, 1, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 460: specialized_quartet_bin_pair<W, 3, 3, 2, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 461: specialized_quartet_bin_pair<W, 3, 3, 2, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 462: specialized_quartet_bin_pair<W, 3, 3, 2, 2, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 465: specialized_quartet_bin_pair<W, 3, 3, 3, 0, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 466: specialized_quartet_bin_pair<W, 3, 3, 3, 1, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 467: specialized_quartet_bin_pair<W, 3, 3, 3, 2, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    case 468: specialized_quartet_bin_pair<W, 3, 3, 3, 3, Emit>(bd, lhs, rhs, shape, same_bin, metric, threshold, primitive_squared, values); return true;
    default: return false;
    }
}
#endif
