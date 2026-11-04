#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SSSS_WIDE_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SSSS_WIDE_IPP

// ss|ss producer transferred from the tuned AVX2 1x16 schedule to native W8.
// Retained at 16 kets after native numerical gates and paired timings;
// the 32-ket ablation stays explicit (heterogeneous deep contractions lose).
// A fixed bra feeds two/four independent ZMM accumulators (16/32 ket pairs).
// Shared exponent factors are hoisted only after an exact plan-time check.
// Full tiles, then 16/8/tail tiles, keep capacity identical to W8 row packing:
// unrolling does not create extra padding or claim to fill triangular tails.
// Primitive-packet screening retains the original producer/packet semantics.

struct NativeSsssFactor { double rho, square_root; };

template<int Groups, bool Shared, class Emit>
inline void native_ssss_ket_tile(
    const BasisDataHGP& bd, const NativePairBin& bra, const NativePairBin& ket,
    std::size_t bi, std::size_t start, int count, const std::vector<double>* metric,
    double threshold, const NativeSsssFactor* factors, std::vector<double>& values) {
    using V = tlno::core::simd::Packet<8>;
    constexpr int W = 8, Capacity = Groups * W;
    const auto bra_id = bra.ids[bi];
    const auto* density = g_rys_density_screen;
    const auto* secondary = g_native_secondary_shell_screen;
    const auto* bounds = native_shell_packet_bounds(ket);
    const double primary_other = metric ? (*metric)[bra_id] : 0.0;
    const double secondary_other = secondary ? (*secondary->metric)[bra_id] : 0.0;
    const double secondary_threshold = secondary ? secondary->threshold : 0.0;
    __mmask8 valid_masks[Groups], live_masks[Groups];
    V masks[Groups], acc[Groups];
    bool any = false;
    native_static_for([&](auto tag) {
        constexpr int g = decltype(tag)::value;
        const int valid = std::min(W, count - g * W);
        valid_masks[g] = static_cast<__mmask8>((1u << valid) - 1u);
        unsigned live = 0;
        const bool packet_live = !bounds || bounds->packet_live(start + g * W,
            primary_other, threshold, secondary_other, secondary_threshold);
        for (int lane = 0; packet_live && lane < valid; ++lane) {
            const auto ket_id = ket.ids[start + g * W + lane];
            const double bound = metric ? (*metric)[bra_id] * (*metric)[ket_id] : 0.;
            if ((!metric || (bound >= threshold &&
                native_secondary_shell_live(secondary, bra_id, ket_id))) &&
                (!density || native_density_live(density, bra_id, ket_id, bound)))
                live |= 1u << lane;
        }
        live_masks[g] = static_cast<__mmask8>(live); any = any || live;
        masks[g] = V(_mm512_maskz_mov_pd(live_masks[g], _mm512_set1_pd(1.)));
    }, std::make_index_sequence<Groups>{});
    if (!any) return;
    for (std::size_t ip = 0; ip < bra.primitives; ++ip) {
        const auto boff = ip * bra.stride + bi;
        const V p(bra.exponent[boff]), px(bra.px[boff]), py(bra.py[boff]), pz(bra.pz[boff]);
        const V bw(bra.weight[boff]);
        for (std::size_t iq = 0; iq < ket.primitives; ++iq) {
            V target[Groups], pref[Groups];
            V shared_rho, shared_sqrt;
            if constexpr (Shared) {
                const auto& f = factors[ip * ket.primitives + iq];
                shared_rho = V(f.rho); shared_sqrt = V(f.square_root);
            }
            native_static_for([&](auto tag) {
                constexpr int g = decltype(tag)::value;
                const auto off = iq * ket.stride + start + g * W;
                // A partial vector repeats its first valid ket. Inactive lanes
                // remain finite and cannot read beyond this primitive rank.
                const auto load = [&](const std::vector<double>& a) {
                    if constexpr (Groups > 1) return V::load(a.data() + off);
                    else return V(_mm512_mask_loadu_pd(_mm512_set1_pd(a[off]), valid_masks[g], a.data() + off));
                };
                const V dx = px - load(ket.px), dy = py - load(ket.py), dz = pz - load(ket.pz);
                V rho, root;
                if constexpr (Shared) { rho = shared_rho; root = shared_sqrt; }
                else { const V q = load(ket.exponent), inv = V(1.) / (p + q); rho = p * q * inv; root = V::sqrt(inv); }
                target[g] = rho * (dx * dx + dy * dy + dz * dz);
                // Preserve the native prefactor association; hoist only sqrt.
                pref[g] = (V(kTwoPi52) * load(ket.weight)) * (masks[g] * bw) * root;
            }, std::make_index_sequence<Groups>{});

            bool medium = true;
            native_static_for([&](auto tag) {
                constexpr int g = decltype(tag)::value;
                medium = medium && _mm512_cmp_pd_mask(target[g].value, _mm512_setzero_pd(), _CMP_GE_OQ) == 255 &&
                    _mm512_cmp_pd_mask(target[g].value, _mm512_set1_pd(kRysSpBoysMediumMaxT), _CMP_LE_OQ) == 255;
            }, std::make_index_sequence<Groups>{});
            if (medium) {
                // Interleave Horner chains across independent quartets. This
                // shares coefficient broadcasts without changing coefficients.
                V z[Groups], num[Groups], den[Groups];
                native_static_for([&](auto tag) {
                    constexpr int g = decltype(tag)::value;
                    z[g] = V::fma(target[g], V(2. / kRysSpBoysMediumMaxT), V(-1.));
                    num[g] = V(kRysSsssMediumNumerator.back());
                    den[g] = V::fma(V(kRysSsssMediumDenominator.back()), z[g], V(kRysSsssMediumDenominator[5]));
                }, std::make_index_sequence<Groups>{});
#pragma GCC unroll 5
                for (int k = 4; k >= 0; --k) {
                    const V a(kRysSsssMediumNumerator[k]), b(kRysSsssMediumDenominator[k]);
                    native_static_for([&](auto tag) {
                        constexpr int g = decltype(tag)::value;
                        num[g] = V::fma(num[g], z[g], a); den[g] = V::fma(den[g], z[g], b);
                    }, std::make_index_sequence<Groups>{});
                }
                native_static_for([&](auto tag) {
                    constexpr int g = decltype(tag)::value;
                    acc[g] = V::fma(pref[g], num[g] / den[g], acc[g]);
                }, std::make_index_sequence<Groups>{});
            } else {
                // Mixed tiles retain per-W8 Boys domains. One distant quartet
                // must not force all 32 kets through the table fallback.
                native_static_for([&](auto tag) {
                    constexpr int g = decltype(tag)::value;
                    V f[1];
                    if (!native_boys_bounded8<0>(target[g], f)) {
                        alignas(64) double ts[W], weights[W]; target[g].store(ts);
                        native_weight1<W>(ts, weights); f[0] = V::load(weights);
                    }
                    acc[g] = V::fma(pref[g], f[0], acc[g]);
                }, std::make_index_sequence<Groups>{});
            }
        }
    }
    alignas(64) double scalar[Capacity];
    const auto& proj = get_final_sph_pair_projector(0, 0);
    const V coefficient(proj.rows[0][0].coeff * proj.rows[0][0].coeff);
    native_static_for([&](auto tag) {
        constexpr int g = decltype(tag)::value; (coefficient * acc[g]).store(scalar + g * W);
    }, std::make_index_sequence<Groups>{});
    for (int lane = 0; lane < count; ++lane) {
        if (!(live_masks[lane / W] & (1u << (lane % W)))) continue;
        const auto ket_id = ket.ids[start + lane];
        rys_emit_spherical_lane_block<Emit>(values,
            bd.packed_spherical_pair_components[bra_id], bd.packed_spherical_pair_components[ket_id], bra_id == ket_id,
            [&](const PackedPairComponentH&, const PackedPairComponentH&) { return scalar[lane]; });
    }
}

template<int Tile, bool Shared, class Emit>
void native_ssss_wide_impl(const BasisDataHGP& bd, const NativePairBin& bra,
    const NativePairBin& ket, bool same_bin, const std::vector<double>* metric,
    double threshold, std::vector<double>& values) {
    thread_local std::vector<NativeSsssFactor> scratch;
    if constexpr (Shared) {
        scratch.resize(bra.primitives * ket.primitives);
        for (std::size_t ip = 0; ip < bra.primitives; ++ip) for (std::size_t iq = 0; iq < ket.primitives; ++iq) {
            const double p = bra.exponent[ip * bra.stride], q = ket.exponent[iq * ket.stride], inv = 1. / (p + q);
            scratch[ip * ket.primitives + iq] = {p * q * inv, std::sqrt(inv)};
        }
    }
    const auto* secondary = g_native_secondary_shell_screen;
    const auto* bounds = native_shell_packet_bounds(ket);
    const double secondary_threshold = secondary ? secondary->threshold : 0.0;
    // This producer vectorizes ket rows and does not use compact bra bins.
    const auto rows = native_weighted_row_range(ket, bra, same_bin, 8, false);
    for (std::size_t bi = rows.begin; bi < rows.end; bi += rows.step) {
        const auto bra_id = bra.ids[bi];
        const double primary_other = metric ? (*metric)[bra_id] : 0.0;
        const double secondary_other = secondary ? (*secondary->metric)[bra_id] : 0.0;
        if (bounds && !bounds->row_live(primary_other, threshold,
                                        secondary_other, secondary_threshold)) continue;
        const auto end = same_bin ? bi + 1 : ket.ids.size();
        std::size_t start = 0;
        if constexpr (Tile == 32) for (; end - start >= 32; start += 32)
            native_ssss_ket_tile<4, Shared, Emit>(bd, bra, ket, bi, start, 32, metric, threshold, scratch.data(), values);
        for (; end - start >= 16; start += 16)
            native_ssss_ket_tile<2, Shared, Emit>(bd, bra, ket, bi, start, 16, metric, threshold, scratch.data(), values);
        for (; start < end; start += 8)
            native_ssss_ket_tile<1, Shared, Emit>(bd, bra, ket, bi, start, static_cast<int>(std::min<std::size_t>(8, end - start)), metric, threshold, scratch.data(), values);
    }
}

template<int Tile, class Emit>
bool native_ssss_wide_bin_pair(const BasisDataHGP& bd, const NativePairBin& lhs,
    const NativePairBin& rhs, bool same_bin, const std::vector<double>* metric,
    double threshold, double primitive_squared, std::vector<double>& values) {
    if (primitive_squared != 0. || lhs.ids.empty() || rhs.ids.empty()) return false;
    // Minimize padded W8 capacity first; prefer longer ket rows on a tie.
    const NativePairBin* bra = &lhs; const NativePairBin* ket = &rhs;
    const auto a = lhs.ids.size(), b = rhs.ids.size();
    if (!same_bin && (b * ((a + 7) / 8) < a * ((b + 7) / 8) ||
        (b * ((a + 7) / 8) == a * ((b + 7) / 8) && a > b))) { bra = &rhs; ket = &lhs; }
    if (bra->ssss_shared_exponents && ket->ssss_shared_exponents)
        native_ssss_wide_impl<Tile, true, Emit>(bd, *bra, *ket, same_bin, metric, threshold, values);
    else native_ssss_wide_impl<Tile, false, Emit>(bd, *bra, *ket, same_bin, metric, threshold, values);
    return true;
}
#endif
