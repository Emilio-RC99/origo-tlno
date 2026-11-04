#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SSSS_RECEIVERS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SSSS_RECEIVERS_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

struct RysJSsssPairView {
    std::uint32_t pair = 0;
    std::size_t matrix_index = 0;
    double density = 0.0;
    double scale = 1.0;
    double norm = 1.0;
};

// WHAT: resolve the one spherical component of an ss shell pair once at the
// output-stationary boundary.
// WHY: scalar emission looked up the same pair metadata for both Coulomb
// terms of every ERI; ss|ss has exactly one component and can reuse it across
// all partners in a bra or ket tile.
// PURPOSE: the J probe accumulates final matrix deltas by destination without
// materializing or rereading an ERI block.
template <class Emit>
__attribute__((always_inline)) inline RysJSsssPairView
rys_j_ssss_pair_view(const RysJkEmitContext& context,
                     const BasisDataHGP& bd,
                     std::uint32_t group) noexcept {
    const PackedPairComponentH& ref = low_batch_pair_components(bd)[
        static_cast<std::size_t>(group)].front();
    if constexpr (rys_emit_uses_pairspace_preemit_ssss_j_v<Emit>) {
        return {ref.pair_index, static_cast<std::size_t>(ref.pair_index),
                context.j_pair_density[ref.pair_index], 1.0, ref.norm};
    }
    const RysJkPairInfo& info = context.pair_info[ref.pair_index];
    const std::size_t index =
        info.first_row + static_cast<std::size_t>(info.second);
    return {ref.pair_index, index, context.p[index],
            static_cast<double>(info.scale), ref.norm};
}

template <int Lane>
__attribute__((always_inline)) inline double rys_j_ssss_extract_lane(
    __m256d value) noexcept {
    static_assert(Lane >= 0 && Lane < 4);
    if constexpr (Lane < 2) {
        const __m128d half = _mm256_castpd256_pd128(value);
        if constexpr (Lane == 0) {
            return _mm_cvtsd_f64(half);
        } else {
            return _mm_cvtsd_f64(_mm_unpackhi_pd(half, half));
        }
    } else {
        const __m128d half = _mm256_extractf128_pd(value, 1);
        if constexpr (Lane == 2) {
            return _mm_cvtsd_f64(half);
        } else {
            return _mm_cvtsd_f64(_mm_unpackhi_pd(half, half));
        }
    }
}

template <class Emit, bool FixedIsLhs>
__attribute__((always_inline)) inline void
rys_j_ssss_accumulate_one(
    const RysJkEmitContext& context,
    const RysJSsssPairView& fixed,
    const RysJSsssPairView& moving,
    double raw,
    double& moving_delta,
    double& fixed_delta) noexcept {
    const double value = raw * fixed.norm * moving.norm;
    const std::uint32_t lhs_pair = FixedIsLhs ? fixed.pair : moving.pair;
    const std::uint32_t rhs_pair = FixedIsLhs ? moving.pair : fixed.pair;
    rys_j_speed_probe_record_candidate_ssss(
        context, lhs_pair, rhs_pair, value);
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    if constexpr (rys_emit_uses_preemit_ssss_exchange_v<Emit>) {
        // The value is already normalized and the canonical pair orientation is
        // resolved above.  This helper contributes only -K/2; J remains in the
        // output-stationary pair-space deltas below.
        accumulate_exchange_half_symmetric_pair_fast(
            context, lhs_pair, rhs_pair, value);
    }
    if constexpr (rys_emit_uses_preemit_ssss_positive_k_v<Emit>) {
        // K-only owns no Coulomb destination: close all four positive exchange
        // terms here and deliberately leave the J deltas below untouched.
        accumulate_k_symmetric_pair_fast(
            context, lhs_pair, rhs_pair, value);
    }
#endif
    if constexpr (rys_emit_uses_preemit_ssss_positive_k_v<Emit>) {
        (void)moving_delta;
        (void)fixed_delta;
    } else if constexpr (rys_emit_uses_pairspace_preemit_ssss_j_v<Emit>) {
        if (fixed.pair == moving.pair) {
            // Split the sole diagonal pair-space contribution between the two
            // live destinations.  Their commits target the same packed pair;
            // the halves preserve both the exact value and the two-term ledger.
            const double half = 0.5 * value * fixed.density;
            moving_delta += half;
            fixed_delta += half;
        } else {
            moving_delta += value * fixed.density;
            fixed_delta += value * moving.density;
        }
    } else {
        double u4 = 4.0 * value * fixed.scale * moving.scale;
        if (fixed.pair == moving.pair) u4 *= 0.5;
        moving_delta += u4 * fixed.density;
        fixed_delta += u4 * moving.density;
    }
}

template <class Emit, bool FixedIsLhs, int Lane>
__attribute__((always_inline)) inline void
rys_j_ssss_accumulate_lane(
    const RysJkEmitContext& context,
    const RysJSsssPairView& fixed,
    const RysJSsssPairView (&moving)[4],
    __m256d values,
    double (&moving_delta)[4],
    std::uint64_t (&moving_terms)[4],
    double& fixed_delta,
    std::uint64_t& fixed_terms) noexcept {
    rys_j_ssss_accumulate_one<Emit, FixedIsLhs>(
        context, fixed, moving[Lane],
        rys_j_ssss_extract_lane<Lane>(values),
        moving_delta[Lane], fixed_delta);
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    if constexpr (rys_emit_uses_preemit_ssss_positive_k_v<Emit>) {
        // No output-stationary J destination exists in K-only mode.  The four
        // exchange writes were committed by accumulate_one above.
        return;
    }
#endif
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++moving_terms[Lane];
    ++fixed_terms;
#else
    // WHAT/WHY/PURPOSE: release needs only an "ever live" bit for the final
    // output commit; exact logical multiplicities belong exclusively to the
    // untimed diagnostic build and must not add an integer reduction per ERI.
    moving_terms[Lane] = 1u;
    fixed_terms = 1u;
#endif
}

template <class Emit, bool FixedIsLhs>
__attribute__((always_inline)) inline void
rys_j_ssss_accumulate_packet(
    const RysJkEmitContext& context,
    const RysJSsssPairView& fixed,
    const RysJSsssPairView (&moving)[4],
    unsigned active_bits,
    __m256d values,
    double (&moving_delta)[4],
    std::uint64_t (&moving_terms)[4],
    double& fixed_delta,
    std::uint64_t& fixed_terms) noexcept {
    if ((active_bits & 1u) != 0u)
        rys_j_ssss_accumulate_lane<Emit, FixedIsLhs, 0>(
            context, fixed, moving, values, moving_delta, moving_terms,
            fixed_delta, fixed_terms);
    if ((active_bits & 2u) != 0u)
        rys_j_ssss_accumulate_lane<Emit, FixedIsLhs, 1>(
            context, fixed, moving, values, moving_delta, moving_terms,
            fixed_delta, fixed_terms);
    if ((active_bits & 4u) != 0u)
        rys_j_ssss_accumulate_lane<Emit, FixedIsLhs, 2>(
            context, fixed, moving, values, moving_delta, moving_terms,
            fixed_delta, fixed_terms);
    if ((active_bits & 8u) != 0u)
        rys_j_ssss_accumulate_lane<Emit, FixedIsLhs, 3>(
            context, fixed, moving, values, moving_delta, moving_terms,
            fixed_delta, fixed_terms);
}

template <class Emit>
__attribute__((always_inline)) inline void rys_j_ssss_commit(
    const RysJkEmitContext& context,
    std::size_t matrix_index,
    double delta,
    std::uint64_t logical_terms) noexcept {
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    if constexpr (rys_emit_uses_preemit_ssss_positive_k_v<Emit>) {
        (void)context;
        (void)matrix_index;
        (void)delta;
        (void)logical_terms;
        return;
    }
#endif
    if (logical_terms == 0u) return;
    if constexpr (rys_emit_uses_pairspace_preemit_ssss_j_v<Emit>) {
        context.j_pair[matrix_index] += delta;
    } else {
        context.g[matrix_index] += delta;
    }
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
    g_rys_direct_j_speed_probe_audit.logical_j_terms_committed +=
        logical_terms;
#else
    (void)logical_terms;
#endif
}

template <class Emit, bool FixedIsLhs>
__attribute__((always_inline)) inline void
rys_j_ssss_consume_moving_packet(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    unsigned active_bits,
    __m256d values,
    const RysJSsssPairView& fixed,
    double& fixed_delta,
    std::uint64_t& fixed_terms) noexcept {
    RysJSsssPairView moving[4] = {};
    double moving_delta[4] = {};
    std::uint64_t moving_terms[4] = {};
    for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) != 0u) {
            moving[lane] = rys_j_ssss_pair_view<Emit>(
                context, bd,
                moving_bin.pair_group_ids[moving_pos + lane]);
        }
    }
    rys_j_ssss_accumulate_packet<Emit, FixedIsLhs>(
        context, fixed, moving, active_bits, values,
        moving_delta, moving_terms, fixed_delta, fixed_terms);
    for (std::size_t lane = 0; lane < 4; ++lane) {
        if (moving_terms[lane] != 0u) {
            rys_j_ssss_commit<Emit>(
                context, moving[lane].matrix_index,
                moving_delta[lane], moving_terms[lane]);
        }
    }
}
#endif
