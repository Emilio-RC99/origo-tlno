#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SSDS_RECEIVERS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SSDS_RECEIVERS_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

template <bool BraDS, int Lane, int Component>
__attribute__((always_inline)) inline void
rys_j_ssds_common_preemit_component(
    const RysJkEmitContext& context,
    const std::vector<PackedPairComponentH>& moving_refs,
    const std::vector<PackedPairComponentH>& fixed_refs,
    __m256d contracted,
    double& moving_delta,
    double (&fixed_delta)[5]) noexcept {
    static_assert(Lane >= 0 && Lane < 4);
    static_assert(Component >= 0 && Component < 5);
    const PackedPairComponentH& lhs = BraDS
        ? moving_refs[static_cast<std::size_t>(Component)]
        : moving_refs.front();
    const PackedPairComponentH& rhs = BraDS
        ? fixed_refs.front()
        : fixed_refs[static_cast<std::size_t>(Component)];
    // The microkernel has already transformed and normalized the six Cartesian
    // d contractions into this pure-spherical value; unlike low-L ordered
    // components, no additional PairComponent norm belongs at the receiver.
    const double value = rys_j_extract_lane<Lane>(contracted);
    rys_j_speed_probe_record_candidate_ssds(
        context, lhs.pair_index, rhs.pair_index, value);

    double* const __restrict j_pair = context.j_pair;
    const double* const __restrict rho = context.j_pair_density;
    if constexpr (BraDS) {
        j_pair[lhs.pair_index] += value * rho[rhs.pair_index];
        fixed_delta[0] += value * rho[lhs.pair_index];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
        ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
    } else {
        moving_delta += value * rho[rhs.pair_index];
        fixed_delta[Component] += value * rho[lhs.pair_index];
    }
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
}

template <bool BraDS, int Lane>
__attribute__((always_inline)) inline void
rys_j_ssds_common_preemit_lane(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    const std::vector<PackedPairComponentH>& fixed_refs,
    __m256d e0,
    __m256d e1,
    __m256d e2,
    __m256d e3,
    __m256d e4,
    double (&fixed_delta)[5]) noexcept {
    const std::uint32_t moving_group =
        bra_bin.pair_group_ids[bra_pos + static_cast<std::size_t>(Lane)];
    const auto& moving_refs = bd.packed_spherical_pair_components[
        static_cast<std::size_t>(moving_group)];
    double moving_delta = 0.0;
    rys_j_ssds_common_preemit_component<BraDS, Lane, 0>(
        context, moving_refs, fixed_refs, e0, moving_delta, fixed_delta);
    rys_j_ssds_common_preemit_component<BraDS, Lane, 1>(
        context, moving_refs, fixed_refs, e1, moving_delta, fixed_delta);
    rys_j_ssds_common_preemit_component<BraDS, Lane, 2>(
        context, moving_refs, fixed_refs, e2, moving_delta, fixed_delta);
    rys_j_ssds_common_preemit_component<BraDS, Lane, 3>(
        context, moving_refs, fixed_refs, e3, moving_delta, fixed_delta);
    rys_j_ssds_common_preemit_component<BraDS, Lane, 4>(
        context, moving_refs, fixed_refs, e4, moving_delta, fixed_delta);
    if constexpr (!BraDS) {
        context.j_pair[moving_refs.front().pair_index] += moving_delta;
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
        ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
    }
}

template <bool BraDS>
__attribute__((always_inline)) inline void
rys_j_ssds_common_preemit_packet(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    const std::vector<PackedPairComponentH>& fixed_refs,
    unsigned active_bits,
    __m256d e0,
    __m256d e1,
    __m256d e2,
    __m256d e3,
    __m256d e4) noexcept {
    double fixed_delta[5] = {};
    if ((active_bits & 1u) != 0u)
        rys_j_ssds_common_preemit_lane<BraDS, 0>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, e3, e4, fixed_delta);
    if ((active_bits & 2u) != 0u)
        rys_j_ssds_common_preemit_lane<BraDS, 1>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, e3, e4, fixed_delta);
    if ((active_bits & 4u) != 0u)
        rys_j_ssds_common_preemit_lane<BraDS, 2>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, e3, e4, fixed_delta);
    if ((active_bits & 8u) != 0u)
        rys_j_ssds_common_preemit_lane<BraDS, 3>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, e3, e4, fixed_delta);

    constexpr int fixed_components = BraDS ? 1 : 5;
    for (int component = 0; component < fixed_components; ++component) {
        context.j_pair[fixed_refs[static_cast<std::size_t>(component)].pair_index] +=
            fixed_delta[component];
    }
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    g_rys_direct_j_speed_probe_audit.candidate_physical_commits +=
        static_cast<std::uint64_t>(fixed_components);
#endif
}
#endif
