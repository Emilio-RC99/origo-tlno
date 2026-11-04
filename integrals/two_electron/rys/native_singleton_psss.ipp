#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SINGLETON_PSSS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SINGLETON_PSSS_IPP

// Singleton-bra ps|ss producer, AVX512F+FMA and W8 only; native_specialized.ipp
// enables it by default (TLNO_NATIVE_SINGLETON_PSSS). One fixed bra is broadcast;
// consecutive ket lanes use masked contiguous loads, never gather indices.
// No quartet list is cached; identifiers and temporary storage are O(8).
template<class Emit>
bool native_singleton_psss_bin_pair(
    const BasisDataHGP& bd, const NativePairBin& lhs, const NativePairBin& rhs,
    bool same_bin, const std::vector<double>* metric, double threshold,
    double primitive_squared, std::vector<double>& values) {
    using V = tlno::core::simd::Packet<8>;
    constexpr int W = 8;
    // Preserve the original primitive-packet rejection when it is active.
    // Contracted and density masks remain per quartet, as in the general path.
    if (same_bin || lhs.ids.size() != 1 || rhs.ids.size() <= 1 ||
        primitive_squared != 0.0) return false;
    if (rhs.ids.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
        return false;
    const auto& lp = get_final_sph_pair_projector(1, 0);
    const auto& rp = get_final_sph_pair_projector(0, 0);
    const auto* density = g_rys_density_screen;
    const auto* secondary = g_native_secondary_shell_screen;
    const auto* bounds = native_shell_packet_bounds(rhs);
    const double primary_other = metric ? (*metric)[lhs.ids[0]] : 0.0;
    const double secondary_other = secondary ? (*secondary->metric)[lhs.ids[0]] : 0.0;
    const double secondary_threshold = secondary ? secondary->threshold : 0.0;
    if (bounds && !bounds->row_live(primary_other, threshold,
                                    secondary_other, secondary_threshold)) return true;
    const auto rows = native_uniform_row_range((rhs.ids.size() + W - 1) / W);
    std::size_t ket_pos = W * rows.begin;
    const std::size_t ket_end = std::min(rhs.ids.size(), W * rows.end);
    while (ket_pos < ket_end) {
        const std::size_t start = ket_pos;
        const int valid = static_cast<int>(std::min<std::size_t>(W, rhs.ids.size() - start));
        // Advance even when the complete contracted packet is rejected.
        ket_pos += W * rows.step;
        if (bounds && !bounds->packet_live(start, primary_other, threshold,
                                           secondary_other, secondary_threshold)) continue;
        const __mmask8 valid_mask = static_cast<__mmask8>((1u << valid) - 1u);
        std::uint32_t bra_id[W], ket_id[W];
        bool active[W];
        for (int lane = 0; lane < W; ++lane) {
            bra_id[lane] = lhs.ids[0];
            ket_id[lane] = rhs.ids[start + (lane < valid ? static_cast<std::size_t>(lane) : 0u)];
        }
        bool any = false;
        alignas(64) double mask_data[W];
        for (int lane = 0; lane < W; ++lane) {
            const double bound = metric ? (*metric)[bra_id[lane]] * (*metric)[ket_id[lane]] : 0.0;
            active[lane] = lane < valid && (!metric || (bound >= threshold &&
                native_secondary_shell_live(secondary, bra_id[lane], ket_id[lane]))) &&
                (!density || native_density_live(density, bra_id[lane], ket_id[lane], bound));
            any = any || active[lane]; mask_data[lane] = active[lane] ? 1.0 : 0.0;
        }
        if (!any) continue;
        const auto broadcast_bra = [](const double* base) { return V(base[0]); };
        // start is a valid ket in every primitive rank. Masked-off lanes do
        // not read memory and replicate this packet's first ket, matching D's
        // padding rather than NativePairBin's whole-bin padding.
        const auto load_ket = [&](const double* base) {
            return V(_mm512_mask_loadu_pd(_mm512_set1_pd(base[start]), valid_mask, base + start));
        };
        const V av[3] = {broadcast_bra(lhs.ax.data()), broadcast_bra(lhs.ay.data()), broadcast_bra(lhs.az.data())};
        const V mask = V::load(mask_data);
        V cart[3];
        const V unused[3] = {V(), V(), V()};
        for (std::size_t ip = 0; ip < lhs.primitives; ++ip) {
            const std::size_t boff = ip * lhs.stride;
            const V p = broadcast_bra(lhs.exponent.data() + boff);
            const V pv[3] = {broadcast_bra(lhs.px.data() + boff), broadcast_bra(lhs.py.data() + boff), broadcast_bra(lhs.pz.data() + boff)};
            const V bw = mask * broadcast_bra(lhs.weight.data() + boff);
            const V pa[3] = {pv[0] - av[0], pv[1] - av[1], pv[2] - av[2]};
            for (std::size_t iq = 0; iq < rhs.primitives; ++iq) {
                const std::size_t koff = iq * rhs.stride;
                const V q = load_ket(rhs.exponent.data() + koff);
                const V sum = p + q, invsum = V(1.0) / sum, qo = q * invsum;
                const V qv[3] = {load_ket(rhs.px.data() + koff), load_ket(rhs.py.data() + koff), load_ket(rhs.pz.data() + koff)};
                const V pq[3] = {pv[0] - qv[0], pv[1] - qv[1], pv[2] - qv[2]};
                const V targ = p * q * invsum * (pq[0] * pq[0] + pq[1] * pq[1] + pq[2] * pq[2]);
                // Same multiplication association as V(kTwoPi52*kp.w)*bw*sqrt.
                // The scalar ket factor is now computed independently per lane.
                const V pref = (V(kTwoPi52) * load_ket(rhs.weight.data() + koff)) * bw * V::sqrt(invsum);
                // Reuse the exact retained Boys implementation. Degree one
                // does not consume qc, ab, po, inv2p or inv2pq.
                if (native_sp_boys_contract8<1, 0, 0, 0>(targ, pa, unused, pq, unused,
                        qo, V(), V(), V(), pref, cart)) continue;
                alignas(64) double targets[W], roots[W], weights[W];
                targ.store(targets);
                native_roots_weights<1, W>(targets, roots, weights);
                const V root = V::load(roots);
                const V c[3] = {pv[0] - av[0] - qo * pq[0] * root,
                    pv[1] - av[1] - qo * pq[1] * root, pv[2] - av[2] - qo * pq[2] * root};
                native_sp_contract<1, 0, 0, 0>(c, unused, unused, V(), V(),
                    pref * V::load(weights), cart);
            }
        }
        alignas(64) double scalar[3 * W];
        // Retain final spherical permutation/coefficient multiplication and
        // the canonical emitter. s/p projectors have one term per row.
        for (int i = 0; i < lp.sph_rows; ++i) for (int j = 0; j < rp.sph_rows; ++j) {
            const auto& a = lp.rows[i][0]; const auto& b = rp.rows[j][0];
            (V(a.coeff * b.coeff) * cart[a.index + b.index]).store(scalar + (i * rp.sph_rows + j) * W);
        }
        for (int lane = 0; lane < valid; ++lane) {
            if (!active[lane]) continue;
            rys_emit_spherical_lane_block<Emit>(values,
                bd.packed_spherical_pair_components[bra_id[lane]],
                bd.packed_spherical_pair_components[ket_id[lane]], bra_id[lane] == ket_id[lane],
                [&](const PackedPairComponentH& l, const PackedPairComponentH& r) {
                    return scalar[(std::size_t(l.component_index) * rp.sph_rows + r.component_index) * W + lane];
                });
        }
    }
    return true;
}
#endif
