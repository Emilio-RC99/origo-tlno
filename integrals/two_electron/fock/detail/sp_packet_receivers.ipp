#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SP_PACKET_RECEIVERS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SP_PACKET_RECEIVERS_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

template <int Lane>
__attribute__((always_inline)) inline double rys_j_extract_lane(
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

__attribute__((always_inline)) inline __m256d rys_j_active_lane_mask(
    unsigned active_bits) noexcept {
    return _mm256_castsi256_pd(_mm256_set_epi64x(
        (active_bits & 8u) != 0u ? -1ll : 0ll,
        (active_bits & 4u) != 0u ? -1ll : 0ll,
        (active_bits & 2u) != 0u ? -1ll : 0ll,
        (active_bits & 1u) != 0u ? -1ll : 0ll));
}

__attribute__((always_inline)) inline double rys_j_horizontal_sum(
    __m256d value) noexcept {
    const __m128d lo = _mm256_castpd256_pd128(value);
    const __m128d hi = _mm256_extractf128_pd(value, 1);
    const __m128d pair = _mm_add_pd(lo, hi);
    return _mm_cvtsd_f64(_mm_hadd_pd(pair, pair));
}

// Stateful pre-emission J tile for one low-L shell-pair packet.  SIMD lanes
// enumerate independent moving shell pairs and consume one final component
// vector at a time.  Thus ps|ps, pp|ps and pp|pp never need a second
// E[component][lane] block just to reach the receiver.
//
// Each canonical ERI contributes two logical Coulomb terms.  Both destination
// families remain local until finish(), which writes every touched pair once.
// If the moving and fixed groups coincide, consume() enforces the same s8
// component triangle as rys_emit_block.  A diagonal value is split between the
// two local families so the two commits sum to one J contribution.
template <int LhsComponentCount, int RhsComponentCount>
class RysJSpPreemitPacket {
public:
    static_assert(LhsComponentCount >= 1 && LhsComponentCount <= 9);
    static_assert(RhsComponentCount >= 1 && RhsComponentCount <= 9);

    __attribute__((noinline)) RysJSpPreemitPacket(
        const RysJkEmitContext& context,
        const BasisDataHGP& bd,
        const RysLowPairBinH& moving_bin,
        std::size_t moving_pos,
        std::uint32_t fixed_group,
        const std::vector<PackedPairComponentH>& fixed_refs,
        unsigned active_bits) noexcept
        : context_(context), fixed_group_(fixed_group),
          active_bits_(active_bits) {
        for (const PackedPairComponentH& ref : fixed_refs) {
            const int component = static_cast<int>(ref.component_index);
            if (component >= 0 && component < RhsComponentCount) {
                fixed_by_component_[component] = &ref;
            }
        }
        for (std::size_t lane = 0; lane < 4; ++lane) {
            if ((active_bits_ & (1u << lane)) == 0u) continue;
            moving_group_[lane] =
                moving_bin.pair_group_ids[moving_pos + lane];
            const auto& refs = low_batch_pair_components(bd)[
                static_cast<std::size_t>(moving_group_[lane])];
            for (const PackedPairComponentH& ref : refs) {
                const int component = static_cast<int>(ref.component_index);
                if (component >= 0 && component < LhsComponentCount) {
                    moving_by_component_[lane][component] = &ref;
                }
            }
        }
    }

    __attribute__((noinline)) void consume(
        int lhs_component,
        int rhs_component,
        __m256d values) noexcept {
        const PackedPairComponentH* const rhs =
            fixed_by_component_[rhs_component];
        if (rhs == nullptr) return;
        bool component_live = false;
        for (std::size_t lane = 0; lane < 4; ++lane) {
            component_live = component_live ||
                moving_by_component_[lane][lhs_component] != nullptr;
        }
        if (!component_live) return;

        alignas(32) double lane_value[4];
        _mm256_store_pd(lane_value, values);
        const double* const __restrict rho = context_.j_pair_density;
        for (std::size_t lane = 0; lane < 4; ++lane) {
            const PackedPairComponentH* const lhs =
                moving_by_component_[lane][lhs_component];
            if (lhs == nullptr) continue;
            if (moving_group_[lane] == fixed_group_ &&
                rhs->pair_index > lhs->pair_index) {
                continue;
            }

            const double value = lane_value[lane] * lhs->norm * rhs->norm;
            rys_j_speed_probe_record_candidate_sp_block(
                context_, lhs->pair_index, rhs->pair_index, value);
            if (lhs->pair_index == rhs->pair_index) {
                const double half = 0.5 * value * rho[lhs->pair_index];
                moving_delta_[lane][lhs_component] += half;
                fixed_delta_[rhs_component] += half;
            } else {
                moving_delta_[lane][lhs_component] +=
                    value * rho[rhs->pair_index];
                fixed_delta_[rhs_component] +=
                    value * rho[lhs->pair_index];
            }
            moving_touched_[lane][lhs_component] = true;
            fixed_touched_[rhs_component] = true;
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
            g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
        }
    }

    __attribute__((noinline)) void finish() noexcept {
        double* const __restrict j_pair = context_.j_pair;
        for (std::size_t lane = 0; lane < 4; ++lane) {
            for (int component = 0;
                 component < LhsComponentCount; ++component) {
                if (!moving_touched_[lane][component]) continue;
                j_pair[moving_by_component_[lane][component]->pair_index] +=
                    moving_delta_[lane][component];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
            }
        }
        for (int component = 0; component < RhsComponentCount; ++component) {
            if (!fixed_touched_[component]) continue;
            j_pair[fixed_by_component_[component]->pair_index] +=
                fixed_delta_[component];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
            ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
        }
    }

private:
    const RysJkEmitContext& context_;
    std::uint32_t fixed_group_ = 0;
    unsigned active_bits_ = 0u;
    const PackedPairComponentH*
        fixed_by_component_[RhsComponentCount] = {};
    const PackedPairComponentH*
        moving_by_component_[4][LhsComponentCount] = {};
    std::uint32_t moving_group_[4] = {};
    double moving_delta_[4][LhsComponentCount] = {};
    bool moving_touched_[4][LhsComponentCount] = {};
    double fixed_delta_[RhsComponentCount] = {};
    bool fixed_touched_[RhsComponentCount] = {};
};

#ifdef TLNO_RYS_DIRECT_EXCHANGE
__attribute__((noinline, noclone)) void rys_k_pppp_preemit_packet(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    unsigned active_bits,
    const __m256d (&bra_r)[3],
    const __m256d (&ket_r)[3],
    const __m256d (&acc)[81]) noexcept;

__attribute__((noinline, noclone, hot)) void rys_k_psps_preemit_packet_v2(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    unsigned active_bits,
    __m256d e00,
    __m256d e01,
    __m256d e02,
    __m256d e10,
    __m256d e11,
    __m256d e12,
    __m256d e20,
    __m256d e21,
    __m256d e22) noexcept {
    alignas(32) double raw[9][4];
    _mm256_store_pd(raw[0], e00);
    _mm256_store_pd(raw[1], e01);
    _mm256_store_pd(raw[2], e02);
    _mm256_store_pd(raw[3], e10);
    _mm256_store_pd(raw[4], e11);
    _mm256_store_pd(raw[5], e12);
    _mm256_store_pd(raw[6], e20);
    _mm256_store_pd(raw[7], e21);
    _mm256_store_pd(raw[8], e22);

    const auto [fixed_p_shell, fixed_s_shell] =
        bd.pair_groups[static_cast<std::size_t>(fixed_group)];
#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
    assert(bd.groups[static_cast<std::size_t>(fixed_p_shell)].L == 1);
    assert(bd.groups[static_cast<std::size_t>(fixed_s_shell)].L == 0);
#endif
    const int fixed_p0 = bd.spherical_group_bases[
        static_cast<std::size_t>(fixed_p_shell)];
    const int fixed_s = bd.spherical_group_bases[
        static_cast<std::size_t>(fixed_s_shell)];
    std::uint32_t fixed_pair[3];
    for (int component = 0; component < 3; ++component) {
        fixed_pair[component] = EriPackedGAccessor::pair_index(
            fixed_p0 + component, fixed_s);
    }

    for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t moving_group = moving_bin.pair_group_ids[
            moving_pos + lane];
        const auto [moving_p_shell, moving_s_shell] =
            bd.pair_groups[static_cast<std::size_t>(moving_group)];
#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
        assert(bd.groups[static_cast<std::size_t>(moving_p_shell)].L == 1);
        assert(bd.groups[static_cast<std::size_t>(moving_s_shell)].L == 0);
#endif
        const int moving_p0 = bd.spherical_group_bases[
            static_cast<std::size_t>(moving_p_shell)];
        const int moving_s = bd.spherical_group_bases[
            static_cast<std::size_t>(moving_s_shell)];
        std::uint32_t moving_pair[3];
        for (int component = 0; component < 3; ++component) {
            moving_pair[component] = EriPackedGAccessor::pair_index(
                moving_p0 + component, moving_s);
        }
        const bool same_pair_group = moving_group == fixed_group;

        // Work in the logical (p,s) orientation instead of the canonical AO
        // ordering stored in PairInfo.  Swapping the two endpoints of either
        // pair merely permutes the same four K updates.  This exposes the exact
        // ps|ps topology without a dynamic destination lookup: nine p-p cells
        // are touched once, while the three p-s cells, three s-p cells and the
        // single s-s cell stay live across the complete tile.  Different pair
        // groups may share their p shell or their s shell, but a p AO can never
        // equal an s AO, so the four destination families remain disjoint.
        const std::size_t M = context.M;
        const double* const p = context.p;
        double* const k = context.g;
        const std::size_t moving_s_row =
            static_cast<std::size_t>(moving_s) * M;
        const double p_ss = p[moving_s_row +
                              static_cast<std::size_t>(fixed_s)];
        double moving_p_fixed_s[3];
        double moving_s_fixed_p[3];
        for (int component = 0; component < 3; ++component) {
            moving_p_fixed_s[component] =
                k[(static_cast<std::size_t>(moving_p0 + component) * M) +
                  static_cast<std::size_t>(fixed_s)];
            moving_s_fixed_p[component] =
                k[moving_s_row +
                  static_cast<std::size_t>(fixed_p0 + component)];
        }
        double moving_s_fixed_s =
            k[moving_s_row + static_cast<std::size_t>(fixed_s)];

        for (int moving_component = 0; moving_component < 3;
             ++moving_component) {
            const int fixed_limit = same_pair_group
                ? moving_component + 1
                : 3;
            for (int fixed_component = 0;
                 fixed_component < fixed_limit; ++fixed_component) {
                const int component =
                    3 * moving_component + fixed_component;
                const double value = raw[component][lane];
                const std::uint32_t lhs_pair =
                    moving_pair[moving_component];
                const std::uint32_t rhs_pair =
                    fixed_pair[fixed_component];
                rys_jk_record_one(context, lhs_pair, rhs_pair, value);

#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
                assert(context.pair_info[lhs_pair].scale == 1.0f);
                assert(context.pair_info[rhs_pair].scale == 1.0f);
                assert((lhs_pair == rhs_pair) ==
                       (same_pair_group &&
                        moving_component == fixed_component));
#endif
                const double factor =
                    (same_pair_group &&
                     moving_component == fixed_component)
                        ? value
                        : 2.0 * value;
                const int moving_p = moving_p0 + moving_component;
                const int fixed_p = fixed_p0 + fixed_component;
                const std::size_t moving_p_row =
                    static_cast<std::size_t>(moving_p) * M;

                // These are the canonical reducer's four exchange terms in
                // logical p/s order.  The p-p destination is unique to this
                // component pair; the other seven destinations are cached.
                k[moving_p_row + static_cast<std::size_t>(fixed_p)] +=
                    factor * p_ss;
                moving_p_fixed_s[moving_component] +=
                    factor * p[moving_s_row +
                               static_cast<std::size_t>(fixed_p)];
                moving_s_fixed_p[fixed_component] +=
                    factor * p[moving_p_row +
                               static_cast<std::size_t>(fixed_s)];
                moving_s_fixed_s +=
                    factor * p[moving_p_row +
                               static_cast<std::size_t>(fixed_p)];
            }
        }
        for (int component = 0; component < 3; ++component) {
            k[(static_cast<std::size_t>(moving_p0 + component) * M) +
              static_cast<std::size_t>(fixed_s)] =
                moving_p_fixed_s[component];
            k[moving_s_row +
              static_cast<std::size_t>(fixed_p0 + component)] =
                moving_s_fixed_p[component];
        }
        k[moving_s_row + static_cast<std::size_t>(fixed_s)] =
            moving_s_fixed_s;
    }
}
#endif

// ps|ps has a regular 3x3 component tile.  Resolve its pair metadata before
// the nine contracted YMM accumulators become live, then consume those values
// one row at a time.  The moving normalizations are kept as SoA vectors so a
// complete J row is formed with three FMAs; only its four final destinations
// are extracted.  The three fixed destinations stay in vector accumulators
// until the packet closes.
struct RysJPspsRowSidecar {
    const RysJkEmitContext* context;
    unsigned active_bits;
    unsigned distinct_bits;
    int alias_lane;
    std::uint32_t fixed_pair[3];
    double fixed_density[3];
    std::uint32_t moving_pair[3][4];
    alignas(32) double moving_density[3][4];
    int alias_fixed_slot[3];
};

__attribute__((always_inline)) inline void rys_j_psps_prepare_row_sidecar(
    RysJPspsRowSidecar& sidecar,
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    const std::vector<PackedPairComponentH>& fixed_refs,
    unsigned active_bits) noexcept {
    sidecar.context = &context;
    sidecar.active_bits = active_bits;
    sidecar.alias_lane = -1;
    sidecar.alias_fixed_slot[0] = -1;
    sidecar.alias_fixed_slot[1] = -1;
    sidecar.alias_fixed_slot[2] = -1;

    const PackedPairComponentH* fixed_by_component[3] = {};
    for (const PackedPairComponentH& ref : fixed_refs) {
        const int component = static_cast<int>(ref.component_index);
        if (component >= 0 && component < 3) {
            fixed_by_component[component] = &ref;
        }
    }
    for (int component = 0; component < 3; ++component) {
        assert(fixed_by_component[component] != nullptr);
        const PackedPairComponentH& ref = *fixed_by_component[component];
        sidecar.fixed_pair[component] = ref.pair_index;
        sidecar.fixed_density[component] =
            context.j_pair_density[ref.pair_index];
    }

    for (int lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t moving_group =
            moving_bin.pair_group_ids[
                moving_pos + static_cast<std::size_t>(lane)];
        const auto& moving_refs = low_batch_pair_components(bd)[
            static_cast<std::size_t>(moving_group)];
        const PackedPairComponentH* moving_by_component[3] = {};
        for (const PackedPairComponentH& ref : moving_refs) {
            const int component = static_cast<int>(ref.component_index);
            if (component >= 0 && component < 3) {
                moving_by_component[component] = &ref;
            }
        }
        for (int component = 0; component < 3; ++component) {
            assert(moving_by_component[component] != nullptr);
            const PackedPairComponentH& ref =
                *moving_by_component[component];
            sidecar.moving_pair[component][lane] = ref.pair_index;
            sidecar.moving_density[component][lane] =
                context.j_pair_density[ref.pair_index];
        }
        if (moving_group == fixed_group) {
            assert(sidecar.alias_lane < 0);
            sidecar.alias_lane = lane;
            for (int component = 0; component < 3; ++component) {
                for (int fixed_component = 0;
                     fixed_component < 3; ++fixed_component) {
                    if (sidecar.moving_pair[component][lane] ==
                        sidecar.fixed_pair[fixed_component]) {
                        sidecar.alias_fixed_slot[component] = fixed_component;
                        break;
                    }
                }
                assert(sidecar.alias_fixed_slot[component] >= 0);
            }
        }
    }
    sidecar.distinct_bits = active_bits;
    if (sidecar.alias_lane >= 0) {
        sidecar.distinct_bits &=
            ~(1u << static_cast<unsigned>(sidecar.alias_lane));
    }
}

template <int Row, int Lane>
__attribute__((always_inline)) inline void rys_j_psps_commit_moving_lane(
    const RysJPspsRowSidecar& sidecar,
    __m256d delta) noexcept {
    static_assert(Row >= 0 && Row < 3);
    static_assert(Lane >= 0 && Lane < 4);
    if ((sidecar.distinct_bits & (1u << Lane)) == 0u) return;
    sidecar.context->j_pair[sidecar.moving_pair[Row][Lane]] +=
        rys_j_extract_lane<Lane>(delta);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
}

template <int Lane>
__attribute__((always_inline)) inline void rys_j_psps_alias_row_values(
    __m256d e0,
    __m256d e1,
    __m256d e2,
    double (&values)[3]) noexcept {
    values[0] = rys_j_extract_lane<Lane>(e0);
    values[1] = rys_j_extract_lane<Lane>(e1);
    values[2] = rys_j_extract_lane<Lane>(e2);
}

template <bool FirstRow, int Row>
__attribute__((always_inline)) inline void rys_j_psps_digest_row(
    const RysJPspsRowSidecar& sidecar,
    __m256d e0,
    __m256d e1,
    __m256d e2,
    __m256d& fixed_delta0,
    __m256d& fixed_delta1,
    __m256d& fixed_delta2) noexcept {
    static_assert(Row >= 0 && Row < 3);
    const __m256d distinct_mask =
        rys_j_active_lane_mask(sidecar.distinct_bits);
    const __m256d moving_density = _mm256_maskload_pd(
        sidecar.moving_density[Row],
        _mm256_castpd_si256(distinct_mask));

    const __m256d moving_sum = _mm256_fmadd_pd(
        e2, _mm256_set1_pd(sidecar.fixed_density[2]),
        _mm256_fmadd_pd(
            e1, _mm256_set1_pd(sidecar.fixed_density[1]),
            _mm256_mul_pd(
                e0, _mm256_set1_pd(sidecar.fixed_density[0]))));
    if constexpr (FirstRow) {
        // Assign, rather than add to three zeros, while E9 is live.  Once
        // these products are formed e00/e01/e02 die and the peak falls from
        // E9+J3 to E6+J3 before any scattered output write.
        fixed_delta0 = _mm256_mul_pd(e0, moving_density);
        fixed_delta1 = _mm256_mul_pd(e1, moving_density);
        fixed_delta2 = _mm256_mul_pd(e2, moving_density);
    } else {
        fixed_delta0 = _mm256_fmadd_pd(
            e0, moving_density, fixed_delta0);
        fixed_delta1 = _mm256_fmadd_pd(
            e1, moving_density, fixed_delta1);
        fixed_delta2 = _mm256_fmadd_pd(
            e2, moving_density, fixed_delta2);
    }

    rys_j_psps_commit_moving_lane<Row, 0>(sidecar, moving_sum);
    rys_j_psps_commit_moving_lane<Row, 1>(sidecar, moving_sum);
    rys_j_psps_commit_moving_lane<Row, 2>(sidecar, moving_sum);
    rys_j_psps_commit_moving_lane<Row, 3>(sidecar, moving_sum);

#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM) || \
    defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT)
    alignas(32) double row_values[3][4];
    _mm256_store_pd(row_values[0], e0);
    _mm256_store_pd(row_values[1], e1);
    _mm256_store_pd(row_values[2], e2);
    for (int lane = 0; lane < 4; ++lane) {
        if ((sidecar.active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t lhs_pair = sidecar.moving_pair[Row][lane];
        const bool alias = lane == sidecar.alias_lane;
        for (int component = 0; component < 3; ++component) {
            const std::uint32_t rhs_pair = sidecar.fixed_pair[component];
            if (alias && rhs_pair > lhs_pair) continue;
            const double value = row_values[component][lane];
            rys_j_speed_probe_record_candidate_sp_block(
                *sidecar.context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
            g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
        }
    }
#endif
}

template <int Row>
__attribute__((always_inline)) inline void rys_j_psps_digest_alias_row(
    const RysJPspsRowSidecar& sidecar,
    __m256d e0,
    __m256d e1,
    __m256d e2,
    double (&alias_delta)[3]) noexcept {
    static_assert(Row >= 0 && Row < 3);
    assert(sidecar.alias_lane >= 0);
    double alias_values[3];
    switch (sidecar.alias_lane) {
        case 0:
            rys_j_psps_alias_row_values<0>(e0, e1, e2, alias_values);
            break;
        case 1:
            rys_j_psps_alias_row_values<1>(e0, e1, e2, alias_values);
            break;
        case 2:
            rys_j_psps_alias_row_values<2>(e0, e1, e2, alias_values);
            break;
        default:
            rys_j_psps_alias_row_values<3>(e0, e1, e2, alias_values);
            break;
    }
    const int lane = sidecar.alias_lane;
    const std::uint32_t lhs_pair = sidecar.moving_pair[Row][lane];
    const int lhs_slot = sidecar.alias_fixed_slot[Row];
    const double* const __restrict rho = sidecar.context->j_pair_density;
    for (int component = 0; component < 3; ++component) {
        const std::uint32_t rhs_pair = sidecar.fixed_pair[component];
        if (rhs_pair > lhs_pair) continue;
        const double value = alias_values[component];
        if (lhs_pair == rhs_pair) {
            alias_delta[lhs_slot] += value * rho[lhs_pair];
        } else {
            alias_delta[lhs_slot] += value * rho[rhs_pair];
            alias_delta[component] += value * rho[lhs_pair];
        }
    }
}

__attribute__((always_inline)) inline void rys_j_psps_finish_rows(
    const RysJPspsRowSidecar& sidecar,
    __m256d fixed_delta0,
    __m256d fixed_delta1,
    __m256d fixed_delta2,
    const double (&alias_delta)[3]) noexcept {
    double* const __restrict j_pair = sidecar.context->j_pair;
    j_pair[sidecar.fixed_pair[0]] +=
        rys_j_horizontal_sum(fixed_delta0) + alias_delta[0];
    j_pair[sidecar.fixed_pair[1]] +=
        rys_j_horizontal_sum(fixed_delta1) + alias_delta[1];
    j_pair[sidecar.fixed_pair[2]] +=
        rys_j_horizontal_sum(fixed_delta2) + alias_delta[2];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    g_rys_direct_j_speed_probe_audit.candidate_physical_commits += 3u;
#endif
}

// pp|ss and ss|pp are rectangular output-stationary tiles.  The pp side may
// expose either nine Cartesian pair components or the six lower-triangular
// components of a same-shell pair; every lookup is therefore keyed by the
// explicit component_index rather than by vector position.
template <bool BraPP>
struct RysJSsppRowSidecar {
    static constexpr int moving_component_count = BraPP ? 9 : 1;
    static constexpr int fixed_component_count = BraPP ? 1 : 9;
    const RysJkEmitContext* context;
    unsigned active_bits;
    unsigned fixed_component_bits;
    unsigned moving_component_bits[moving_component_count];
    std::uint32_t fixed_pair[fixed_component_count];
    double fixed_density[fixed_component_count];
    std::uint32_t moving_pair[moving_component_count][4];
    alignas(32) double moving_density[moving_component_count][4];
};

template <bool BraPP>
__attribute__((always_inline)) inline void rys_j_sspp_prepare_row_sidecar(
    RysJSsppRowSidecar<BraPP>& sidecar,
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    const std::vector<PackedPairComponentH>& fixed_refs,
    unsigned active_bits) noexcept {
    sidecar.context = &context;
    sidecar.active_bits = active_bits;
    sidecar.fixed_component_bits = 0u;
    for (int component = 0;
         component < RysJSsppRowSidecar<BraPP>::moving_component_count;
         ++component) {
        sidecar.moving_component_bits[component] = 0u;
    }
    for (const PackedPairComponentH& ref : fixed_refs) {
        const int component = static_cast<int>(ref.component_index);
        if (component < 0 ||
            component >= RysJSsppRowSidecar<BraPP>::fixed_component_count) {
            continue;
        }
        sidecar.fixed_component_bits |= 1u << component;
        sidecar.fixed_pair[component] = ref.pair_index;
        sidecar.fixed_density[component] =
            context.j_pair_density[ref.pair_index];
    }
    for (int lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t moving_group =
            moving_bin.pair_group_ids[
                moving_pos + static_cast<std::size_t>(lane)];
        assert(moving_group != fixed_group);
        const auto& moving_refs = low_batch_pair_components(bd)[
            static_cast<std::size_t>(moving_group)];
        for (const PackedPairComponentH& ref : moving_refs) {
            const int component = static_cast<int>(ref.component_index);
            if (component < 0 ||
                component >=
                    RysJSsppRowSidecar<BraPP>::moving_component_count) {
                continue;
            }
            sidecar.moving_component_bits[component] |= 1u << lane;
            sidecar.moving_pair[component][lane] = ref.pair_index;
            sidecar.moving_density[component][lane] =
                context.j_pair_density[ref.pair_index];
        }
    }
    if constexpr (BraPP) {
        assert(sidecar.fixed_component_bits == 1u);
    } else {
        assert(sidecar.moving_component_bits[0] == active_bits);
    }
}

template <bool BraPP, int Component>
__attribute__((always_inline)) inline void rys_j_sspp_record_component(
    const RysJSsppRowSidecar<BraPP>& sidecar,
    unsigned live_bits,
    __m256d raw) noexcept {
#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM) || \
    defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT)
    alignas(32) double values[4];
    _mm256_store_pd(values, raw);
    constexpr int moving_component = BraPP ? Component : 0;
    constexpr int fixed_component = BraPP ? 0 : Component;
    for (int lane = 0; lane < 4; ++lane) {
        if ((live_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t lhs_pair =
            sidecar.moving_pair[moving_component][lane];
        const std::uint32_t rhs_pair =
            sidecar.fixed_pair[fixed_component];
        const double value = values[lane];
        rys_j_speed_probe_record_candidate_sp_block(
            *sidecar.context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
        g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
    }
#else
    (void)sidecar;
    (void)live_bits;
    (void)raw;
#endif
}

template <int Component, int Lane>
__attribute__((always_inline)) inline void
rys_j_sspp_commit_moving_pp_lane(
    const RysJSsppRowSidecar<true>& sidecar,
    unsigned live_bits,
    __m256d delta) noexcept {
    if ((live_bits & (1u << Lane)) == 0u) return;
    sidecar.context->j_pair[sidecar.moving_pair[Component][Lane]] +=
        rys_j_extract_lane<Lane>(delta);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
}

template <int Component>
__attribute__((always_inline)) inline __m256d rys_j_sspp_final_value(
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[3],
    const __m256d (&acc_hi)[6]) noexcept {
    static_assert(Component >= 0 && Component < 9);
    constexpr int pp_unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    constexpr int first_axis = Component / 3;
    constexpr int second_axis = Component - 3 * first_axis;
    return _mm256_fmadd_pd(
        vR[second_axis], acc_lo[first_axis], acc_hi[pp_unique[Component]]);
}

#ifdef TLNO_RYS_DIRECT_EXCHANGE
__attribute__((always_inline)) inline void rys_k_sspp_preemit_lane(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    bool bra_pp,
    std::size_t lane,
    const double (&raw)[9][4]) noexcept {
    const std::uint32_t moving_group = moving_bin.pair_group_ids[
        moving_pos + lane];
    const std::uint32_t pp_group = bra_pp ? moving_group : fixed_group;
    const std::uint32_t ss_group = bra_pp ? fixed_group : moving_group;
    const auto [p_shell0, p_shell1] =
        bd.pair_groups[static_cast<std::size_t>(pp_group)];
    const auto [ss_shell0, ss_shell1] =
        bd.pair_groups[static_cast<std::size_t>(ss_group)];
#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
    assert(bd.groups[static_cast<std::size_t>(p_shell0)].L == 1);
    assert(bd.groups[static_cast<std::size_t>(p_shell1)].L == 1);
    assert(bd.groups[static_cast<std::size_t>(ss_shell0)].L == 0);
    assert(bd.groups[static_cast<std::size_t>(ss_shell1)].L == 0);
#endif
    const int a0 = bd.spherical_group_bases[
        static_cast<std::size_t>(p_shell0)];
    const int b0 = bd.spherical_group_bases[
        static_cast<std::size_t>(p_shell1)];
    const int x = bd.spherical_group_bases[
        static_cast<std::size_t>(ss_shell0)];
    const int y = bd.spherical_group_bases[
        static_cast<std::size_t>(ss_shell1)];
    const double e00 = raw[0][lane];
    const double e10 = raw[3][lane];
    const double e11 = raw[4][lane];
    const double e20 = raw[6][lane];
    const double e21 = raw[7][lane];
    const double e22 = raw[8][lane];
    double e01 = e10;
    double e02 = e20;
    double e12 = e21;
    if (p_shell0 != p_shell1) {
        e01 = raw[1][lane];
        e02 = raw[2][lane];
        e12 = raw[5][lane];
    }

    const std::uint32_t ss_pair = EriPackedGAccessor::pair_index(x, y);
    const auto record = [&](int ai, int bj, double value) noexcept {
        const std::uint32_t pp_pair = EriPackedGAccessor::pair_index(
            a0 + ai, b0 + bj);
        if (bra_pp) {
            rys_jk_record_one(context, pp_pair, ss_pair, value);
        } else {
            rys_jk_record_one(context, ss_pair, pp_pair, value);
        }
    };
    if (p_shell0 == p_shell1) {
        record(0, 0, e00);
        record(1, 0, e10);
        record(1, 1, e11);
        record(2, 0, e20);
        record(2, 1, e21);
        record(2, 2, e22);
    } else {
        record(0, 0, e00); record(0, 1, e01); record(0, 2, e02);
        record(1, 0, e10); record(1, 1, e11); record(1, 2, e12);
        record(2, 0, e20); record(2, 1, e21); record(2, 2, e22);
    }

    const std::size_t M = context.M;
    const double* const p = context.p;
    double* const k = context.g;
    const std::size_t xr = static_cast<std::size_t>(x) * M;
    const std::size_t yr = static_cast<std::size_t>(y) * M;
    const double scale = x == y ? 1.0 : 2.0;
    if (p_shell0 == p_shell1) {
        const double py0 = p[yr + static_cast<std::size_t>(a0)];
        const double py1 = p[yr + static_cast<std::size_t>(a0 + 1)];
        const double py2 = p[yr + static_cast<std::size_t>(a0 + 2)];
        const double px0 = p[xr + static_cast<std::size_t>(a0)];
        const double px1 = p[xr + static_cast<std::size_t>(a0 + 1)];
        const double px2 = p[xr + static_cast<std::size_t>(a0 + 2)];
        k[xr + static_cast<std::size_t>(a0)] += scale *
            (e00 * py0 + e01 * py1 + e02 * py2);
        k[xr + static_cast<std::size_t>(a0 + 1)] += scale *
            (e10 * py0 + e11 * py1 + e12 * py2);
        k[xr + static_cast<std::size_t>(a0 + 2)] += scale *
            (e20 * py0 + e21 * py1 + e22 * py2);
        k[yr + static_cast<std::size_t>(a0)] += scale *
            (e00 * px0 + e01 * px1 + e02 * px2);
        k[yr + static_cast<std::size_t>(a0 + 1)] += scale *
            (e10 * px0 + e11 * px1 + e12 * px2);
        k[yr + static_cast<std::size_t>(a0 + 2)] += scale *
            (e20 * px0 + e21 * px1 + e22 * px2);
        return;
    }

    const double pyb0 = p[yr + static_cast<std::size_t>(b0)];
    const double pyb1 = p[yr + static_cast<std::size_t>(b0 + 1)];
    const double pyb2 = p[yr + static_cast<std::size_t>(b0 + 2)];
    const double pxb0 = p[xr + static_cast<std::size_t>(b0)];
    const double pxb1 = p[xr + static_cast<std::size_t>(b0 + 1)];
    const double pxb2 = p[xr + static_cast<std::size_t>(b0 + 2)];
    const double pya0 = p[yr + static_cast<std::size_t>(a0)];
    const double pya1 = p[yr + static_cast<std::size_t>(a0 + 1)];
    const double pya2 = p[yr + static_cast<std::size_t>(a0 + 2)];
    const double pxa0 = p[xr + static_cast<std::size_t>(a0)];
    const double pxa1 = p[xr + static_cast<std::size_t>(a0 + 1)];
    const double pxa2 = p[xr + static_cast<std::size_t>(a0 + 2)];
    const double xa0 = e00 * pyb0 + e01 * pyb1 + e02 * pyb2;
    const double xa1 = e10 * pyb0 + e11 * pyb1 + e12 * pyb2;
    const double xa2 = e20 * pyb0 + e21 * pyb1 + e22 * pyb2;
    const double ya0 = e00 * pxb0 + e01 * pxb1 + e02 * pxb2;
    const double ya1 = e10 * pxb0 + e11 * pxb1 + e12 * pxb2;
    const double ya2 = e20 * pxb0 + e21 * pxb1 + e22 * pxb2;
    const double xb0 = e00 * pya0 + e10 * pya1 + e20 * pya2;
    const double xb1 = e01 * pya0 + e11 * pya1 + e21 * pya2;
    const double xb2 = e02 * pya0 + e12 * pya1 + e22 * pya2;
    const double yb0 = e00 * pxa0 + e10 * pxa1 + e20 * pxa2;
    const double yb1 = e01 * pxa0 + e11 * pxa1 + e21 * pxa2;
    const double yb2 = e02 * pxa0 + e12 * pxa1 + e22 * pxa2;
    k[xr + static_cast<std::size_t>(a0)] += scale * xa0;
    k[xr + static_cast<std::size_t>(a0 + 1)] += scale * xa1;
    k[xr + static_cast<std::size_t>(a0 + 2)] += scale * xa2;
    k[yr + static_cast<std::size_t>(a0)] += scale * ya0;
    k[yr + static_cast<std::size_t>(a0 + 1)] += scale * ya1;
    k[yr + static_cast<std::size_t>(a0 + 2)] += scale * ya2;
    k[xr + static_cast<std::size_t>(b0)] += scale * xb0;
    k[xr + static_cast<std::size_t>(b0 + 1)] += scale * xb1;
    k[xr + static_cast<std::size_t>(b0 + 2)] += scale * xb2;
    k[yr + static_cast<std::size_t>(b0)] += scale * yb0;
    k[yr + static_cast<std::size_t>(b0 + 1)] += scale * yb1;
    k[yr + static_cast<std::size_t>(b0 + 2)] += scale * yb2;
}

__attribute__((noinline, noclone, hot)) void rys_k_sspp_preemit_packet(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    bool bra_pp,
    unsigned active_bits,
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[3],
    const __m256d (&acc_hi)[6]) noexcept {
    alignas(32) double raw[9][4];
    _mm256_store_pd(raw[0], rys_j_sspp_final_value<0>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[1], rys_j_sspp_final_value<1>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[2], rys_j_sspp_final_value<2>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[3], rys_j_sspp_final_value<3>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[4], rys_j_sspp_final_value<4>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[5], rys_j_sspp_final_value<5>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[6], rys_j_sspp_final_value<6>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[7], rys_j_sspp_final_value<7>(vR, acc_lo, acc_hi));
    _mm256_store_pd(raw[8], rys_j_sspp_final_value<8>(vR, acc_lo, acc_hi));
    for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        rys_k_sspp_preemit_lane(
            context, bd, moving_bin, moving_pos, fixed_group, bra_pp,
            lane, raw);
    }
}
#endif

template <int Component>
__attribute__((always_inline)) inline void
rys_j_sspp_digest_moving_pp_component(
    const RysJSsppRowSidecar<true>& sidecar,
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[3],
    const __m256d (&acc_hi)[6],
    __m256d& fixed_delta) noexcept {
    const unsigned live_bits = sidecar.moving_component_bits[Component] &
                               sidecar.active_bits;
    if (live_bits == 0u) return;
    const __m256d raw =
        rys_j_sspp_final_value<Component>(vR, acc_lo, acc_hi);
    const __m256d mask = rys_j_active_lane_mask(live_bits);
    const __m256d moving_density = _mm256_maskload_pd(
        sidecar.moving_density[Component], _mm256_castpd_si256(mask));
    const __m256d live_raw = _mm256_and_pd(raw, mask);
    fixed_delta = _mm256_fmadd_pd(
        live_raw, moving_density, fixed_delta);
    const __m256d moving_delta = _mm256_mul_pd(
        live_raw, _mm256_set1_pd(sidecar.fixed_density[0]));
    rys_j_sspp_commit_moving_pp_lane<Component, 0>(
        sidecar, live_bits, moving_delta);
    rys_j_sspp_commit_moving_pp_lane<Component, 1>(
        sidecar, live_bits, moving_delta);
    rys_j_sspp_commit_moving_pp_lane<Component, 2>(
        sidecar, live_bits, moving_delta);
    rys_j_sspp_commit_moving_pp_lane<Component, 3>(
        sidecar, live_bits, moving_delta);
    rys_j_sspp_record_component<true, Component>(
        sidecar, live_bits, raw);
}

template <int Component>
__attribute__((always_inline)) inline void
rys_j_sspp_digest_fixed_pp_component(
    const RysJSsppRowSidecar<false>& sidecar,
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[3],
    const __m256d (&acc_hi)[6],
    __m256d& moving_delta) noexcept {
    if ((sidecar.fixed_component_bits & (1u << Component)) == 0u) return;
    const __m256d raw =
        rys_j_sspp_final_value<Component>(vR, acc_lo, acc_hi);
    const unsigned live_bits = sidecar.moving_component_bits[0] &
                               sidecar.active_bits;
    const __m256d mask = rys_j_active_lane_mask(live_bits);
    const __m256d moving_density = _mm256_maskload_pd(
        sidecar.moving_density[0], _mm256_castpd_si256(mask));
    const __m256d live_raw = _mm256_and_pd(raw, mask);
    moving_delta = _mm256_fmadd_pd(
        live_raw,
        _mm256_set1_pd(sidecar.fixed_density[Component]),
        moving_delta);
    sidecar.context->j_pair[sidecar.fixed_pair[Component]] +=
        rys_j_horizontal_sum(_mm256_mul_pd(live_raw, moving_density));
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
    rys_j_sspp_record_component<false, Component>(
        sidecar, live_bits, raw);
}

template <int Lane>
__attribute__((always_inline)) inline void rys_j_sspp_commit_moving_ss_lane(
    const RysJSsppRowSidecar<false>& sidecar,
    __m256d delta) noexcept {
    if ((sidecar.active_bits & (1u << Lane)) == 0u) return;
    sidecar.context->j_pair[sidecar.moving_pair[0][Lane]] +=
        rys_j_extract_lane<Lane>(delta);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
}

__attribute__((always_inline)) inline void
rys_j_sspp_finish_moving_pp(
    const RysJSsppRowSidecar<true>& sidecar,
    __m256d fixed_delta) noexcept {
    sidecar.context->j_pair[sidecar.fixed_pair[0]] +=
        rys_j_horizontal_sum(fixed_delta);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
}

__attribute__((always_inline)) inline void
rys_j_sspp_finish_moving_ss(
    const RysJSsppRowSidecar<false>& sidecar,
    __m256d moving_delta) noexcept {
    rys_j_sspp_commit_moving_ss_lane<0>(sidecar, moving_delta);
    rys_j_sspp_commit_moving_ss_lane<1>(sidecar, moving_delta);
    rys_j_sspp_commit_moving_ss_lane<2>(sidecar, moving_delta);
    rys_j_sspp_commit_moving_ss_lane<3>(sidecar, moving_delta);
}

template <bool BraPP>
struct RysJPsppRowSidecar {
    static constexpr int moving_component_count = BraPP ? 9 : 3;
    static constexpr int fixed_component_count = BraPP ? 3 : 9;
    const RysJkEmitContext* context;
    unsigned active_bits;
    unsigned fixed_component_bits;
    unsigned moving_component_bits[moving_component_count];
    std::uint32_t fixed_pair[fixed_component_count];
    double fixed_density[fixed_component_count];
    std::uint32_t moving_pair[moving_component_count][4];
    alignas(32) double moving_density[moving_component_count][4];
};

template <int PpComponent, int PsComponent>
__attribute__((always_inline)) inline __m256d rys_j_pspp_final_value(
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[9],
    const __m256d (&acc_hi)[18]) noexcept {
    static_assert(PpComponent >= 0 && PpComponent < 9);
    static_assert(PsComponent >= 0 && PsComponent < 3);
    constexpr int pp_unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    constexpr int first_axis = PpComponent / 3;
    constexpr int second_axis = PpComponent - 3 * first_axis;
    return _mm256_fmadd_pd(
        vR[second_axis], acc_lo[3 * PsComponent + first_axis],
        acc_hi[6 * PsComponent + pp_unique[PpComponent]]);
}

#ifdef TLNO_RYS_DIRECT_EXCHANGE
template <bool BraPP>
__attribute__((noinline, noclone)) void rys_k_pspp_packet_v3(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    unsigned active_bits,
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[9],
    const __m256d (&acc_hi)[18]) noexcept;
#endif

template <bool BraPP>
__attribute__((always_inline)) inline void rys_j_pspp_prepare_row_sidecar(
    RysJPsppRowSidecar<BraPP>& sidecar,
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    const std::vector<PackedPairComponentH>& fixed_refs,
    unsigned active_bits) noexcept {
    sidecar.context = &context;
    sidecar.active_bits = active_bits;
    sidecar.fixed_component_bits = 0u;
    for (int component = 0;
         component < RysJPsppRowSidecar<BraPP>::moving_component_count;
         ++component) {
        sidecar.moving_component_bits[component] = 0u;
    }
    for (const PackedPairComponentH& ref : fixed_refs) {
        const int component = static_cast<int>(ref.component_index);
        if (component < 0 ||
            component >= RysJPsppRowSidecar<BraPP>::fixed_component_count) {
            continue;
        }
        sidecar.fixed_component_bits |= 1u << component;
        sidecar.fixed_pair[component] = ref.pair_index;
        sidecar.fixed_density[component] =
            context.j_pair_density[ref.pair_index];
    }
    for (int lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t moving_group =
            moving_bin.pair_group_ids[
                moving_pos + static_cast<std::size_t>(lane)];
        assert(moving_group != fixed_group);
        const auto& moving_refs = low_batch_pair_components(bd)[
            static_cast<std::size_t>(moving_group)];
        for (const PackedPairComponentH& ref : moving_refs) {
            const int component = static_cast<int>(ref.component_index);
            if (component < 0 ||
                component >=
                    RysJPsppRowSidecar<BraPP>::moving_component_count) {
                continue;
            }
            sidecar.moving_component_bits[component] |= 1u << lane;
            sidecar.moving_pair[component][lane] = ref.pair_index;
            sidecar.moving_density[component][lane] =
                context.j_pair_density[ref.pair_index];
        }
    }
    if constexpr (BraPP) {
        assert(sidecar.fixed_component_bits == 0x7u);
    } else {
        assert(sidecar.moving_component_bits[0] == active_bits);
        assert(sidecar.moving_component_bits[1] == active_bits);
        assert(sidecar.moving_component_bits[2] == active_bits);
    }
}

template <bool BraPP, int MovingComponent, int FixedComponent>
__attribute__((always_inline)) inline void rys_j_pspp_record_cell(
    const RysJPsppRowSidecar<BraPP>& sidecar,
    unsigned live_bits,
    __m256d raw) noexcept {
#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM) || \
    defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT)
    alignas(32) double values[4];
    _mm256_store_pd(values, raw);
    for (int lane = 0; lane < 4; ++lane) {
        if ((live_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t lhs_pair =
            sidecar.moving_pair[MovingComponent][lane];
        const std::uint32_t rhs_pair =
            sidecar.fixed_pair[FixedComponent];
        const double value = values[lane];
        rys_j_speed_probe_record_candidate_sp_block(
            *sidecar.context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
        g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
    }
#else
    (void)sidecar;
    (void)live_bits;
    (void)raw;
#endif
}

template <int Component, int Lane>
__attribute__((always_inline)) inline void
rys_j_pspp_commit_moving_pp_lane(
    const RysJPsppRowSidecar<true>& sidecar,
    unsigned live_bits,
    __m256d delta) noexcept {
    if ((live_bits & (1u << Lane)) == 0u) return;
    sidecar.context->j_pair[sidecar.moving_pair[Component][Lane]] +=
        rys_j_extract_lane<Lane>(delta);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
}

template <int Component>
__attribute__((always_inline)) inline void rys_j_pspp_digest_moving_pp_row(
    const RysJPsppRowSidecar<true>& sidecar,
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[9],
    const __m256d (&acc_hi)[18],
    __m256d& fixed_delta0,
    __m256d& fixed_delta1,
    __m256d& fixed_delta2) noexcept {
    const unsigned live_bits = sidecar.moving_component_bits[Component] &
                               sidecar.active_bits;
    if (live_bits == 0u) return;
    const __m256d raw0 =
        rys_j_pspp_final_value<Component, 0>(vR, acc_lo, acc_hi);
    const __m256d raw1 =
        rys_j_pspp_final_value<Component, 1>(vR, acc_lo, acc_hi);
    const __m256d raw2 =
        rys_j_pspp_final_value<Component, 2>(vR, acc_lo, acc_hi);
    const __m256d mask = rys_j_active_lane_mask(live_bits);
    const __m256d moving_density = _mm256_maskload_pd(
        sidecar.moving_density[Component], _mm256_castpd_si256(mask));
    const __m256d live_raw0 = _mm256_and_pd(raw0, mask);
    const __m256d live_raw1 = _mm256_and_pd(raw1, mask);
    const __m256d live_raw2 = _mm256_and_pd(raw2, mask);
    fixed_delta0 = _mm256_fmadd_pd(
        live_raw0, moving_density, fixed_delta0);
    fixed_delta1 = _mm256_fmadd_pd(
        live_raw1, moving_density, fixed_delta1);
    fixed_delta2 = _mm256_fmadd_pd(
        live_raw2, moving_density, fixed_delta2);
    const __m256d moving_delta = _mm256_fmadd_pd(
        live_raw2, _mm256_set1_pd(sidecar.fixed_density[2]),
        _mm256_fmadd_pd(
            live_raw1, _mm256_set1_pd(sidecar.fixed_density[1]),
            _mm256_mul_pd(
                live_raw0,
                _mm256_set1_pd(sidecar.fixed_density[0]))));
    rys_j_pspp_commit_moving_pp_lane<Component, 0>(
        sidecar, live_bits, moving_delta);
    rys_j_pspp_commit_moving_pp_lane<Component, 1>(
        sidecar, live_bits, moving_delta);
    rys_j_pspp_commit_moving_pp_lane<Component, 2>(
        sidecar, live_bits, moving_delta);
    rys_j_pspp_commit_moving_pp_lane<Component, 3>(
        sidecar, live_bits, moving_delta);
    rys_j_pspp_record_cell<true, Component, 0>(
        sidecar, live_bits, raw0);
    rys_j_pspp_record_cell<true, Component, 1>(
        sidecar, live_bits, raw1);
    rys_j_pspp_record_cell<true, Component, 2>(
        sidecar, live_bits, raw2);
}

template <int Component>
__attribute__((always_inline)) inline void rys_j_pspp_digest_fixed_pp_column(
    const RysJPsppRowSidecar<false>& sidecar,
    const __m256d (&vR)[3],
    const __m256d (&acc_lo)[9],
    const __m256d (&acc_hi)[18],
    __m256d& moving_delta0,
    __m256d& moving_delta1,
    __m256d& moving_delta2) noexcept {
    if ((sidecar.fixed_component_bits & (1u << Component)) == 0u) return;
    const __m256d raw0 =
        rys_j_pspp_final_value<Component, 0>(vR, acc_lo, acc_hi);
    const __m256d raw1 =
        rys_j_pspp_final_value<Component, 1>(vR, acc_lo, acc_hi);
    const __m256d raw2 =
        rys_j_pspp_final_value<Component, 2>(vR, acc_lo, acc_hi);
    const unsigned live_bits = sidecar.active_bits;
    const __m256d mask = rys_j_active_lane_mask(live_bits);
    const __m256d live_raw0 = _mm256_and_pd(raw0, mask);
    const __m256d live_raw1 = _mm256_and_pd(raw1, mask);
    const __m256d live_raw2 = _mm256_and_pd(raw2, mask);
    const __m256d fixed_density =
        _mm256_set1_pd(sidecar.fixed_density[Component]);
    moving_delta0 = _mm256_fmadd_pd(
        live_raw0, fixed_density, moving_delta0);
    moving_delta1 = _mm256_fmadd_pd(
        live_raw1, fixed_density, moving_delta1);
    moving_delta2 = _mm256_fmadd_pd(
        live_raw2, fixed_density, moving_delta2);
    const __m256d fixed_lane_delta = _mm256_fmadd_pd(
        live_raw2,
        _mm256_maskload_pd(
            sidecar.moving_density[2], _mm256_castpd_si256(mask)),
        _mm256_fmadd_pd(
            live_raw1,
            _mm256_maskload_pd(
                sidecar.moving_density[1], _mm256_castpd_si256(mask)),
            _mm256_mul_pd(
                live_raw0,
                _mm256_maskload_pd(
                    sidecar.moving_density[0],
                    _mm256_castpd_si256(mask)))));
    sidecar.context->j_pair[sidecar.fixed_pair[Component]] +=
        rys_j_horizontal_sum(fixed_lane_delta);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
    rys_j_pspp_record_cell<false, 0, Component>(
        sidecar, live_bits, raw0);
    rys_j_pspp_record_cell<false, 1, Component>(
        sidecar, live_bits, raw1);
    rys_j_pspp_record_cell<false, 2, Component>(
        sidecar, live_bits, raw2);
}

template <int Component, int Lane>
__attribute__((always_inline)) inline void rys_j_pspp_commit_moving_ps_lane(
    const RysJPsppRowSidecar<false>& sidecar,
    __m256d delta) noexcept {
    if ((sidecar.active_bits & (1u << Lane)) == 0u) return;
    sidecar.context->j_pair[sidecar.moving_pair[Component][Lane]] +=
        rys_j_extract_lane<Lane>(delta);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
}

__attribute__((always_inline)) inline void rys_j_pspp_finish_moving_pp(
    const RysJPsppRowSidecar<true>& sidecar,
    __m256d fixed_delta0,
    __m256d fixed_delta1,
    __m256d fixed_delta2) noexcept {
    sidecar.context->j_pair[sidecar.fixed_pair[0]] +=
        rys_j_horizontal_sum(fixed_delta0);
    sidecar.context->j_pair[sidecar.fixed_pair[1]] +=
        rys_j_horizontal_sum(fixed_delta1);
    sidecar.context->j_pair[sidecar.fixed_pair[2]] +=
        rys_j_horizontal_sum(fixed_delta2);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    g_rys_direct_j_speed_probe_audit.candidate_physical_commits += 3u;
#endif
}

__attribute__((always_inline)) inline void rys_j_pspp_finish_moving_ps(
    const RysJPsppRowSidecar<false>& sidecar,
    __m256d moving_delta0,
    __m256d moving_delta1,
    __m256d moving_delta2) noexcept {
    rys_j_pspp_commit_moving_ps_lane<0, 0>(sidecar, moving_delta0);
    rys_j_pspp_commit_moving_ps_lane<0, 1>(sidecar, moving_delta0);
    rys_j_pspp_commit_moving_ps_lane<0, 2>(sidecar, moving_delta0);
    rys_j_pspp_commit_moving_ps_lane<0, 3>(sidecar, moving_delta0);
    rys_j_pspp_commit_moving_ps_lane<1, 0>(sidecar, moving_delta1);
    rys_j_pspp_commit_moving_ps_lane<1, 1>(sidecar, moving_delta1);
    rys_j_pspp_commit_moving_ps_lane<1, 2>(sidecar, moving_delta1);
    rys_j_pspp_commit_moving_ps_lane<1, 3>(sidecar, moving_delta1);
    rys_j_pspp_commit_moving_ps_lane<2, 0>(sidecar, moving_delta2);
    rys_j_pspp_commit_moving_ps_lane<2, 1>(sidecar, moving_delta2);
    rys_j_pspp_commit_moving_ps_lane<2, 2>(sidecar, moving_delta2);
    rys_j_pspp_commit_moving_ps_lane<2, 3>(sidecar, moving_delta2);
}

template <bool BraP, int Lane>
__attribute__((always_inline)) inline void
rys_j_ssps_record_lane(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    const std::vector<PackedPairComponentH>& fixed_refs,
    __m256d e0,
    __m256d e1,
    __m256d e2) noexcept {
#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM) || \
    defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT)
    const std::uint32_t bra_group =
        bra_bin.pair_group_ids[bra_pos + static_cast<std::size_t>(Lane)];
    const auto& bra_refs = low_batch_pair_components(bd)[
        static_cast<std::size_t>(bra_group)];
    const __m256d values[3] = {e0, e1, e2};
    for (int component = 0; component < 3; ++component) {
        const PackedPairComponentH& lhs = BraP
            ? bra_refs[static_cast<std::size_t>(component)]
            : bra_refs.front();
        const PackedPairComponentH& rhs = BraP
            ? fixed_refs.front()
            : fixed_refs[static_cast<std::size_t>(component)];
        rys_j_speed_probe_record_candidate_ssps(
            context, lhs.pair_index, rhs.pair_index,
            rys_j_extract_lane<Lane>(values[component]) *
                lhs.norm * rhs.norm);
    }
#else
    (void)context;
    (void)bd;
    (void)bra_bin;
    (void)bra_pos;
    (void)fixed_refs;
    (void)e0;
    (void)e1;
    (void)e2;
#endif
}

template <int Lane, std::uint64_t LogicalTerms = 1u>
__attribute__((always_inline)) inline void
rys_j_ssps_commit_distinct(
    const RysJkEmitContext& context,
    const std::size_t (&indices)[4],
    __m256d values) noexcept {
    context.g[indices[Lane]] += rys_j_extract_lane<Lane>(values);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
    g_rys_direct_j_speed_probe_audit.logical_j_terms_committed +=
        LogicalTerms;
#endif
}

template <int Component>
__attribute__((always_inline)) inline void
rys_j_ssps_commit_brap_component(
    const RysJkEmitContext& context,
    unsigned shell_live,
    const std::size_t (&moving_indices)[3][4],
    const double (&moving_density)[3][4],
    const double (&moving_scale)[3][4],
    double common,
    double fixed_density,
    __m256d value,
    __m256d& fixed_sum) noexcept {
    static_assert(Component >= 0 && Component < 3);
    const __m256d scale = _mm256_set_pd(
        moving_scale[Component][3], moving_scale[Component][2],
        moving_scale[Component][1], moving_scale[Component][0]);
    const __m256d density = _mm256_set_pd(
        moving_density[Component][3], moving_density[Component][2],
        moving_density[Component][1], moving_density[Component][0]);
    const __m256d scaled = _mm256_mul_pd(
        value, _mm256_mul_pd(scale, _mm256_set1_pd(common)));
    const __m256d distinct = _mm256_mul_pd(
        scaled, _mm256_set1_pd(fixed_density));
    if ((shell_live & 1u) != 0u)
        rys_j_ssps_commit_distinct<0>(
            context, moving_indices[Component], distinct);
    if ((shell_live & 2u) != 0u)
        rys_j_ssps_commit_distinct<1>(
            context, moving_indices[Component], distinct);
    if ((shell_live & 4u) != 0u)
        rys_j_ssps_commit_distinct<2>(
            context, moving_indices[Component], distinct);
    if ((shell_live & 8u) != 0u)
        rys_j_ssps_commit_distinct<3>(
            context, moving_indices[Component], distinct);
    fixed_sum = _mm256_fmadd_pd(scaled, density, fixed_sum);
}

template <int Component>
__attribute__((always_inline)) inline void
rys_j_ssps_commit_ketp_component(
    const RysJkEmitContext& context,
    const std::vector<PackedPairComponentH>& fixed_refs,
    unsigned shell_live,
    const __m256d lhs_scale,
    const __m256d lhs_density,
    __m256d value,
    __m256d& lhs_sum) noexcept {
    static_assert(Component >= 0 && Component < 3);
    const PackedPairComponentH& fixed =
        fixed_refs[static_cast<std::size_t>(Component)];
    const RysJkPairInfo& fi = context.pair_info[fixed.pair_index];
    const std::size_t fixed_index =
        fi.first_row + static_cast<std::size_t>(fi.second);
    const __m256d scaled = _mm256_mul_pd(
        value,
        _mm256_mul_pd(
            lhs_scale,
            _mm256_set1_pd(4.0 * static_cast<double>(fi.scale))));
    lhs_sum = _mm256_fmadd_pd(
        scaled, _mm256_set1_pd(context.p[fixed_index]), lhs_sum);
    context.g[fixed_index] += rys_j_horizontal_sum(
        _mm256_mul_pd(scaled, lhs_density));
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
    // WHAT: this one physical horizontal write represents one Coulomb term
    // from every live lane for the selected p component.
    // WHY: physical stores are deliberately fewer than logical ERI terms, so
    // deriving the latter from the store count would under-certify the work.
    // PURPOSE: the diagnostic ledger proves both terms of every owned ERI
    // were committed even though the fast path merges destinations.
    g_rys_direct_j_speed_probe_audit.logical_j_terms_committed +=
        static_cast<std::uint64_t>(
            __builtin_popcount(shell_live & 0x0fu));
#endif
}
#endif
