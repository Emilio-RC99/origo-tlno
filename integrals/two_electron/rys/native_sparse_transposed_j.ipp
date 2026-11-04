#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SPARSE_TRANSPOSED_J_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_SPARSE_TRANSPOSED_J_IPP

// Generic one-sided J traversal. The caller supplies its existing exact
// specialized packet producer and vector density contraction. This helper
// changes neither recurrence physics nor any original bra packet's members.
#ifndef TLNO_NATIVE_J_TRANSPOSED_CYCLIC_PACKETS
#define TLNO_NATIVE_J_TRANSPOSED_CYCLIC_PACKETS 1
#endif
#if TLNO_NATIVE_J_TRANSPOSED_CYCLIC_PACKETS != 0 && TLNO_NATIVE_J_TRANSPOSED_CYCLIC_PACKETS != 1
#error "TLNO_NATIVE_J_TRANSPOSED_CYCLIC_PACKETS must be 0 or 1"
#endif
#ifndef TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J
// Production also handles zero packets within mixed bins and retains their
// original triangular prefixes. Zero remains available for timing ablations.
#define TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J 1
#endif
#if TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J != 0 && TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J != 1
#error "TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J must be 0 or 1"
#endif

template<int W, class Produce>
bool native_transposed_zero_bra_j(
    const BasisDataHGP& bd, const NativePairBin& lhs, const NativePairBin& rhs,
    bool same_bin, const std::vector<double>* metric, double threshold,
    std::vector<tlno::core::simd::Packet<W>>& deltas,
    std::vector<unsigned char>& selected_packets, Produce&& produce) {
    selected_packets.clear();
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
    using V = tlno::core::simd::Packet<W>;
    if (lhs.ids.empty() || rhs.ids.empty()) return false;
#if !TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J
    if (same_bin) return false;
#endif
    const auto* density = g_rys_density_screen;
    const auto* context = rys_jk_context_slot();
    if (!density || !density->j_only || !density->shell_rho ||
        !context || !context->j_pair || !context->j_pair_density) return false;
    const auto* compact_bra = native_j_compact_bin(lhs);
    bool whole_bin_zero = true;
#if TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J
    const std::size_t packets = (lhs.ids.size() + W - 1) / W;
    selected_packets.assign(packets, 1u);
    if (compact_bra) {
        // The call's existing sparse index lists active positions in the
        // immutable original bra order. Summarize once per bin call; the
        // fallback never rescans shell densities in each ket row.
        for (const auto position : *native_j_compact_positions(lhs)) {
            selected_packets[position / W] = 0u;
            whole_bin_zero = false;
        }
    } else {
        for (std::size_t i = 0; i < lhs.ids.size(); ++i) {
            if (density->shell_rho[lhs.ids[i]] != 0.0) {
                selected_packets[i / W] = 0u;
                whole_bin_zero = false;
            }
        }
    }
    if (std::none_of(selected_packets.begin(), selected_packets.end(),
                     [](unsigned char zero) { return zero != 0; })) {
        selected_packets.clear();
        return false;
    }
#else
    if (compact_bra) {
        if (!compact_bra->ids.empty()) return false;
    } else {
        for (const auto id : lhs.ids) if (density->shell_rho[id] != 0.0) return false;
    }
#endif
    const auto* active_kets = native_j_compact_bin(rhs)
        ? native_j_compact_positions(rhs) : nullptr;
    const std::size_t ket_count = active_kets ? active_kets->size() : rhs.ids.size();
    const auto* secondary = g_native_secondary_shell_screen;
    const double secondary_threshold = secondary ? secondary->threshold : 0.0;
    const auto* bra_bounds = native_shell_packet_bounds(lhs);
    const auto* ket_bounds = native_shell_packet_bounds(rhs, active_kets != nullptr);
#if !TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J
    const std::size_t packets = (lhs.ids.size() + W - 1) / W;
#endif
#if TLNO_NATIVE_J_TRANSPOSED_CYCLIC_PACKETS
    // Nearby original packets may share costly geometry. Cyclic ownership
    // disperses those packets across workers without splitting any packet.
    const auto part = g_native_row_partition;
    const NativeRowRange rows{part.rank, packets, part.count};
#else
    const auto rows = native_uniform_row_range(packets);
#endif
    thread_local std::vector<double> scalar;
    scalar.resize(deltas.size() * W);
    for (std::size_t packet = rows.begin; packet < rows.end; packet += rows.step) {
#if TLNO_NATIVE_SPARSE_TRANSPOSED_MIXED_J
        if (!selected_packets[packet]) continue;
#endif
        const std::size_t start = packet * W;
        const int valid = static_cast<int>(std::min<std::size_t>(W, lhs.ids.size() - start));
        if (bra_bounds && ket_bounds && !bra_bounds->packet_live(start,
                ket_bounds->primary_max, threshold,
                ket_bounds->secondary_max, secondary_threshold)) continue;
        std::fill(deltas.begin(), deltas.end(), V());
        bool any = false;
        for (std::size_t row = 0; row < ket_count; ++row) {
            const std::size_t ki = active_kets ? (*active_kets)[row] : row;
            const auto ket_id = rhs.ids[ki];
            // Original zero-ket rows use compact active-bra packets, which
            // contain no selected zero-bra pair. The fallback owns that entire
            // path, including its distinct primitive-packet membership.
            if (density->shell_rho[ket_id] == 0.0) continue;
            const std::size_t end = same_bin ? std::min(lhs.ids.size(), ki + 1) : lhs.ids.size();
            if (end <= start) continue;
            const int row_valid = static_cast<int>(std::min<std::size_t>(W, end - start));
            const double primary_other = metric ? (*metric)[ket_id] : 0.0;
            const double secondary_other = secondary ? (*secondary->metric)[ket_id] : 0.0;
            if (bra_bounds && !bra_bounds->packet_live(start, primary_other, threshold,
                                                       secondary_other, secondary_threshold)) continue;
            // Do not short-circuit a later callback after the first live row.
            const bool live = produce(ket_id, start, row_valid);
            any = any || live;
        }
        if (!any) continue;
        for (std::size_t i = 0; i < deltas.size(); ++i) deltas[i].store(scalar.data() + i * W);
        for (int lane = 0; lane < valid; ++lane) {
            const auto id = lhs.ids[start + lane];
            for (const auto& ref : bd.packed_spherical_pair_components[id])
                context->j_pair[ref.pair_index] +=
                    scalar[std::size_t(ref.component_index) * W + lane];
        }
    }
    // The original ket-outer traversal still owns every unselected full-bra
    // packet and every compact zero-ket packet in a mixed bin.
    return whole_bin_zero;
#else
    (void)bd; (void)lhs; (void)rhs; (void)same_bin; (void)metric;
    (void)threshold; (void)deltas; (void)produce;
    return false;
#endif
}
#endif
