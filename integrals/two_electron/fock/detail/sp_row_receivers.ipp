#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SP_ROW_RECEIVERS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_SP_ROW_RECEIVERS_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

template <int Lane>
__attribute__((always_inline)) inline double
rys_j_ssps_extract_lane(__m256d value) noexcept {
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

#ifdef TLNO_RYS_DIRECT_EXCHANGE
// Close one active ss|ps lane directly into positive K.  A ps pair has one
// fixed s endpoint and three varying p endpoints.  Across its three ERIs, two
// K destinations per ss endpoint are output-stationary.  The other six logical
// destinations are transposed into two contiguous p triplets: the public K is
// S(K)=0.5*(K+K^T), so S(C)==S(C^T) for each individual contribution.  Thus an
// off-diagonal ss pair closes with two scalar RMWs and two SSE+scalar triplet
// RMWs; a diagonal ss pair needs only one of each.  No ERI reaches `raw`.
//
// BraP preserves the producer's canonical orientation.  When true, ps is lhs
// and its AO endpoints select K rows; otherwise ss is lhs.  Do not canonicalize
// the resulting K destinations: the public endpoint deliberately averages the
// two directed triangles only after the complete canonical s8 traversal.
template <bool BraP, int Lane>
__attribute__((always_inline)) inline void
rys_k_ssps_preemit_lane(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    __m256d e0,
    __m256d e1,
    __m256d e2) noexcept {
    static_assert(Lane >= 0 && Lane < 4);
    const std::uint32_t moving_group =
        moving_bin.pair_group_ids[
            moving_pos + static_cast<std::size_t>(Lane)];
    const std::uint32_t ps_group = BraP ? moving_group : fixed_group;
    const std::uint32_t ss_group = BraP ? fixed_group : moving_group;
    const auto [p_shell, ps_s_shell] =
        bd.pair_groups[static_cast<std::size_t>(ps_group)];
    const auto [ss_shell0, ss_shell1] =
        bd.pair_groups[static_cast<std::size_t>(ss_group)];
#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
    assert(bd.groups[static_cast<std::size_t>(p_shell)].L == 1);
    assert(bd.groups[static_cast<std::size_t>(ps_s_shell)].L == 0);
    assert(bd.groups[static_cast<std::size_t>(ss_shell0)].L == 0);
    assert(bd.groups[static_cast<std::size_t>(ss_shell1)].L == 0);
#endif

    // The direct endpoint is already in the pure-spherical stream.  Its s/p
    // component norms are exactly one, ps pairs are necessarily off-diagonal,
    // and the only remaining s8 scale is 1/2 for a diagonal ss pair.  Derive
    // those facts from the immutable shell groups instead of chasing four
    // PairInfo records and two component vectors for every active lane.
    const int v0 = bd.spherical_group_bases[
        static_cast<std::size_t>(p_shell)];
    const int v1 = v0 + 1;
    const int v2 = v0 + 2;
    const int z = bd.spherical_group_bases[
        static_cast<std::size_t>(ps_s_shell)];
    const int x = bd.spherical_group_bases[
        static_cast<std::size_t>(ss_shell0)];
    const int y = bd.spherical_group_bases[
        static_cast<std::size_t>(ss_shell1)];
    const double value0 = rys_j_ssps_extract_lane<Lane>(e0);
    const double value1 = rys_j_ssps_extract_lane<Lane>(e1);
    const double value2 = rys_j_ssps_extract_lane<Lane>(e2);
    const std::uint32_t ss_pair =
        EriPackedGAccessor::pair_index(x, y);
    const std::uint32_t ps_pair0 =
        EriPackedGAccessor::pair_index(v0, z);
    const std::uint32_t ps_pair1 =
        EriPackedGAccessor::pair_index(v1, z);
    const std::uint32_t ps_pair2 =
        EriPackedGAccessor::pair_index(v2, z);
    if constexpr (BraP) {
        rys_jk_record_one(context, ps_pair0, ss_pair, value0);
        rys_jk_record_one(context, ps_pair1, ss_pair, value1);
        rys_jk_record_one(context, ps_pair2, ss_pair, value2);
    } else {
        rys_jk_record_one(context, ss_pair, ps_pair0, value0);
        rys_jk_record_one(context, ss_pair, ps_pair1, value1);
        rys_jk_record_one(context, ss_pair, ps_pair2, value2);
    }

    const double ss_scale = x == y ? 0.5 : 1.0;
    const double u0 = 2.0 * value0 * ss_scale;
    const double u1 = 2.0 * value1 * ss_scale;
    const double u2 = 2.0 * value2 * ss_scale;
    const std::size_t M = context.M;
    const std::size_t z_row = static_cast<std::size_t>(z) * M;
    const std::size_t v0_row = static_cast<std::size_t>(v0) * M;
    const std::size_t v1_row = static_cast<std::size_t>(v1) * M;
    const std::size_t v2_row = static_cast<std::size_t>(v2) * M;
    const std::size_t x_row = static_cast<std::size_t>(x) * M;
    const std::size_t y_row = static_cast<std::size_t>(y) * M;
    double* const k = context.g;
    const double* const p = context.p;

    // Two p components travel in one 128-bit RMW and the third remains scalar.
    // This is deliberately SSE+scalar rather than AVX2 maskload/maskstore: on
    // the Haswell-family acceptance host maskmov costs enough uops to erase the
    // benefit of grouping three hot doubles.  `v0..v2` are consecutive by the
    // pure-spherical p-shell contract proved above.
    const __m128d u01 = _mm_set_pd(u1, u0);
    const auto add_triplet = [u01, u2](double* destination,
                                       double density) noexcept {
        const __m128d old01 = _mm_loadu_pd(destination);
        _mm_storeu_pd(
            destination,
            _mm_fmadd_pd(u01, _mm_set1_pd(density), old01));
        destination[2] += u2 * density;
    };
    const auto add_triplet_twice = [u01, u2](double* destination,
                                             double density) noexcept {
        const __m128d vrho = _mm_set1_pd(density);
        __m128d updated01 = _mm_loadu_pd(destination);
        updated01 = _mm_fmadd_pd(u01, vrho, updated01);
        updated01 = _mm_fmadd_pd(u01, vrho, updated01);
        _mm_storeu_pd(destination, updated01);
        destination[2] += u2 * density;
        destination[2] += u2 * density;
    };

    if constexpr (BraP) {
        const double fixed_x =
            u0 * p[v0_row + static_cast<std::size_t>(y)] +
            u1 * p[v1_row + static_cast<std::size_t>(y)] +
            u2 * p[v2_row + static_cast<std::size_t>(y)];
        const double fixed_y =
            u0 * p[v0_row + static_cast<std::size_t>(x)] +
            u1 * p[v1_row + static_cast<std::size_t>(x)] +
            u2 * p[v2_row + static_cast<std::size_t>(x)];
        const double p_zy = p[z_row + static_cast<std::size_t>(y)];
        const double p_zx = p[z_row + static_cast<std::size_t>(x)];
        if (x == y) {
            k[z_row + static_cast<std::size_t>(x)] += fixed_x;
            k[z_row + static_cast<std::size_t>(x)] += fixed_x;
            add_triplet_twice(
                k + x_row + static_cast<std::size_t>(v0), p_zx);
        } else {
            k[z_row + static_cast<std::size_t>(x)] += fixed_x;
            k[z_row + static_cast<std::size_t>(y)] += fixed_y;
            add_triplet(
                k + x_row + static_cast<std::size_t>(v0), p_zy);
            add_triplet(
                k + y_row + static_cast<std::size_t>(v0), p_zx);
        }
    } else {
        const double fixed_x =
            u0 * p[y_row + static_cast<std::size_t>(v0)] +
            u1 * p[y_row + static_cast<std::size_t>(v1)] +
            u2 * p[y_row + static_cast<std::size_t>(v2)];
        const double fixed_y =
            u0 * p[x_row + static_cast<std::size_t>(v0)] +
            u1 * p[x_row + static_cast<std::size_t>(v1)] +
            u2 * p[x_row + static_cast<std::size_t>(v2)];
        const double p_yz = p[y_row + static_cast<std::size_t>(z)];
        const double p_xz = p[x_row + static_cast<std::size_t>(z)];
        if (x == y) {
            k[x_row + static_cast<std::size_t>(z)] += fixed_x;
            k[x_row + static_cast<std::size_t>(z)] += fixed_x;
            add_triplet_twice(
                k + x_row + static_cast<std::size_t>(v0), p_xz);
        } else {
            k[x_row + static_cast<std::size_t>(z)] += fixed_x;
            k[y_row + static_cast<std::size_t>(z)] += fixed_y;
            add_triplet(
                k + x_row + static_cast<std::size_t>(v0), p_yz);
            add_triplet(
                k + y_row + static_cast<std::size_t>(v0), p_xz);
        }
    }
}

template <bool BraP>
__attribute__((noinline, noclone, hot)) void
rys_k_ssps_preemit_packet(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    unsigned active_bits,
    __m256d e0,
    __m256d e1,
    __m256d e2) noexcept {
    if ((active_bits & 1u) != 0u) {
        rys_k_ssps_preemit_lane<BraP, 0>(
            context, bd, moving_bin, moving_pos, fixed_group, e0, e1, e2);
    }
    if ((active_bits & 2u) != 0u) {
        rys_k_ssps_preemit_lane<BraP, 1>(
            context, bd, moving_bin, moving_pos, fixed_group, e0, e1, e2);
    }
    if ((active_bits & 4u) != 0u) {
        rys_k_ssps_preemit_lane<BraP, 2>(
            context, bd, moving_bin, moving_pos, fixed_group, e0, e1, e2);
    }
    if ((active_bits & 8u) != 0u) {
        rys_k_ssps_preemit_lane<BraP, 3>(
            context, bd, moving_bin, moving_pos, fixed_group, e0, e1, e2);
    }
}
#endif

template <bool BraP, int Lane, int Component>
__attribute__((always_inline)) inline void
rys_j_ssps_common_preemit_component(
    const RysJkEmitContext& context,
    const std::vector<PackedPairComponentH>& moving_refs,
    const std::vector<PackedPairComponentH>& fixed_refs,
    __m256d contracted,
    double& moving_delta,
    double (&fixed_delta)[3]) noexcept {
    static_assert(Lane >= 0 && Lane < 4);
    static_assert(Component >= 0 && Component < 3);
    const PackedPairComponentH& lhs = BraP
        ? moving_refs[static_cast<std::size_t>(Component)]
        : moving_refs.front();
    const PackedPairComponentH& rhs = BraP
        ? fixed_refs.front()
        : fixed_refs[static_cast<std::size_t>(Component)];
    const double value = rys_j_ssps_extract_lane<Lane>(contracted) *
                         lhs.norm * rhs.norm;
    rys_j_speed_probe_record_candidate_ssps(
        context, lhs.pair_index, rhs.pair_index, value);

    double* const __restrict j_pair = context.j_pair;
    const double* const __restrict rho = context.j_pair_density;
    if constexpr (BraP) {
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

template <int Lane>
__attribute__((always_inline)) inline double rys_j_extract_lane(
    __m256d value) noexcept;
__attribute__((always_inline)) inline __m256d rys_j_active_lane_mask(
    unsigned active_bits) noexcept;
__attribute__((always_inline)) inline double rys_j_horizontal_sum(
    __m256d value) noexcept;

struct RysJPpppRowSidecar {
    const RysJkEmitContext* context;
    unsigned active_bits;
    unsigned distinct_bits;
    unsigned fixed_component_bits;
    unsigned moving_component_bits[9];
    int alias_lane;
    int alias_fixed_slot[9];
    std::uint32_t fixed_pair[9];
    double fixed_density[9];
    std::uint32_t moving_pair[9][4];
    alignas(32) double moving_density[9][4];
};

__attribute__((always_inline)) inline void rys_j_pppp_prepare_row_sidecar(
    RysJPpppRowSidecar& sidecar,
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
    sidecar.alias_lane = -1;
    for (int component = 0; component < 9; ++component) {
        sidecar.moving_component_bits[component] = 0u;
        sidecar.alias_fixed_slot[component] = -1;
    }
    for (const PackedPairComponentH& ref : fixed_refs) {
        const int component = static_cast<int>(ref.component_index);
        if (component < 0 || component >= 9) continue;
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
        const auto& moving_refs = low_batch_pair_components(bd)[
            static_cast<std::size_t>(moving_group)];
        for (const PackedPairComponentH& ref : moving_refs) {
            const int component = static_cast<int>(ref.component_index);
            if (component < 0 || component >= 9) continue;
            sidecar.moving_component_bits[component] |= 1u << lane;
            sidecar.moving_pair[component][lane] = ref.pair_index;
            sidecar.moving_density[component][lane] =
                context.j_pair_density[ref.pair_index];
        }
        if (moving_group == fixed_group) {
            assert(sidecar.alias_lane < 0);
            sidecar.alias_lane = lane;
            for (const PackedPairComponentH& moving_ref : moving_refs) {
                const int moving_component =
                    static_cast<int>(moving_ref.component_index);
                if (moving_component < 0 || moving_component >= 9) continue;
                for (const PackedPairComponentH& fixed_ref : fixed_refs) {
                    if (fixed_ref.pair_index == moving_ref.pair_index) {
                        sidecar.alias_fixed_slot[moving_component] =
                            static_cast<int>(fixed_ref.component_index);
                        break;
                    }
                }
                assert(sidecar.alias_fixed_slot[moving_component] >= 0);
            }
        }
    }
    sidecar.distinct_bits = active_bits;
    if (sidecar.alias_lane >= 0) {
        sidecar.distinct_bits &=
            ~(1u << static_cast<unsigned>(sidecar.alias_lane));
    }
}

__attribute__((always_inline)) inline __m256d rys_j_pppp_final_value(
    int ab,
    int cd,
    const __m256d (&v_bra_r)[3],
    const __m256d (&v_ket_r)[3],
    const __m256d (&acc)[81]) noexcept {
    static constexpr int pp_unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    const int a = ab / 3;
    const int b = ab - 3 * a;
    const int c = cd / 3;
    const int d = cd - 3 * c;
    const int bra_hi = 3 + pp_unique[ab];
    const int ket_hi = 3 + pp_unique[cd];
    const __m256d hi = _mm256_fmadd_pd(
        v_bra_r[b], acc[9 * a + ket_hi],
        acc[9 * bra_hi + ket_hi]);
    const __m256d lo = _mm256_fmadd_pd(
        v_bra_r[b], acc[9 * a + c], acc[9 * bra_hi + c]);
    return _mm256_fmadd_pd(v_ket_r[d], lo, hi);
}

__attribute__((always_inline)) inline double rys_j_extract_lane_runtime(
    __m256d value,
    int lane) noexcept {
    switch (lane) {
        case 0: return rys_j_extract_lane<0>(value);
        case 1: return rys_j_extract_lane<1>(value);
        case 2: return rys_j_extract_lane<2>(value);
        default: return rys_j_extract_lane<3>(value);
    }
}

__attribute__((always_inline)) inline void rys_j_pppp_record_distinct_cell(
    const RysJPpppRowSidecar& sidecar,
    int ab,
    int cd,
    unsigned distinct_live,
    __m256d raw) noexcept {
#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM) || \
    defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT)
    alignas(32) double raw_lanes[4];
    _mm256_store_pd(raw_lanes, raw);
    for (int lane = 0; lane < 4; ++lane) {
        if ((distinct_live & (1u << lane)) == 0u) continue;
        rys_j_speed_probe_record_candidate_sp_block(
            *sidecar.context,
            sidecar.moving_pair[ab][lane],
            sidecar.fixed_pair[cd],
            raw_lanes[lane]);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
        g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
    }
#else
    (void)sidecar;
    (void)ab;
    (void)cd;
    (void)distinct_live;
    (void)raw;
#endif
}

__attribute__((always_inline)) inline void rys_j_pppp_record_alias_cell(
    const RysJPpppRowSidecar& sidecar,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM) || \
    defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT)
    rys_j_speed_probe_record_candidate_sp_block(
        *sidecar.context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
#else
    (void)sidecar;
    (void)lhs_pair;
    (void)rhs_pair;
    (void)value;
#endif
}

__attribute__((always_inline)) inline void
rys_j_pppp_record_fixed_commit() noexcept {
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
}

__attribute__((always_inline)) inline void rys_j_pppp_digest_rows(
    const RysJPpppRowSidecar& sidecar,
    const __m256d (&v_bra_r)[3],
    const __m256d (&v_ket_r)[3],
    const __m256d (&acc)[81]) noexcept {
    // Named accumulators are deliberate.  An indexed fixed_delta[9] made GCC
    // materialize a second 9-vector tile and perform a load/FMA/store for
    // every cell.  Nine named loop-carried values can remain in YMM registers
    // and need only nine horizontal reductions when the packet closes.
    __m256d fixed_delta0 = _mm256_setzero_pd();
    __m256d fixed_delta1 = _mm256_setzero_pd();
    __m256d fixed_delta2 = _mm256_setzero_pd();
    __m256d fixed_delta3 = _mm256_setzero_pd();
    __m256d fixed_delta4 = _mm256_setzero_pd();
    __m256d fixed_delta5 = _mm256_setzero_pd();
    __m256d fixed_delta6 = _mm256_setzero_pd();
    __m256d fixed_delta7 = _mm256_setzero_pd();
    __m256d fixed_delta8 = _mm256_setzero_pd();
    double alias_delta[9] = {};
    const double* const __restrict rho =
        sidecar.context->j_pair_density;
    for (int ab = 0; ab < 9; ++ab) {
        const unsigned row_live =
            sidecar.moving_component_bits[ab] & sidecar.active_bits;
        if (row_live == 0u) continue;
        const unsigned distinct_live =
            row_live & sidecar.distinct_bits;
        if (distinct_live != 0u) {
            const __m256d distinct_mask =
                rys_j_active_lane_mask(distinct_live);
            const __m256d moving_density = _mm256_maskload_pd(
                sidecar.moving_density[ab],
                _mm256_castpd_si256(distinct_mask));
            __m256d moving_delta = _mm256_setzero_pd();
#define TLNO_RYS_J_PPPP_DISTINCT_CELL(CD, FIXED_DELTA) do { \
            if ((sidecar.fixed_component_bits & (1u << (CD))) != 0u) { \
                const __m256d raw = rys_j_pppp_final_value( \
                    ab, (CD), v_bra_r, v_ket_r, acc); \
                moving_delta = _mm256_fmadd_pd( \
                    raw, _mm256_set1_pd(sidecar.fixed_density[(CD)]), \
                    moving_delta); \
                (FIXED_DELTA) = _mm256_fmadd_pd( \
                    raw, moving_density, (FIXED_DELTA)); \
                rys_j_pppp_record_distinct_cell( \
                    sidecar, ab, (CD), distinct_live, raw); \
            } \
        } while (false)
            TLNO_RYS_J_PPPP_DISTINCT_CELL(0, fixed_delta0);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(1, fixed_delta1);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(2, fixed_delta2);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(3, fixed_delta3);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(4, fixed_delta4);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(5, fixed_delta5);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(6, fixed_delta6);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(7, fixed_delta7);
            TLNO_RYS_J_PPPP_DISTINCT_CELL(8, fixed_delta8);
#undef TLNO_RYS_J_PPPP_DISTINCT_CELL

            alignas(32) double moving_lanes[4];
            _mm256_store_pd(moving_lanes, moving_delta);
            for (int lane = 0; lane < 4; ++lane) {
                if ((distinct_live & (1u << lane)) == 0u) continue;
                sidecar.context->j_pair[
                    sidecar.moving_pair[ab][lane]] += moving_lanes[lane];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                ++g_rys_direct_j_speed_probe_audit
                    .candidate_physical_commits;
#endif
            }
        }
    }

    // A same-group lane is uncommon and obeys the canonical pair triangle.
    // Handle it after the vector bulk so ordinary packets pay no alias branch
    // in any of their 81 cells.  Recomputing terminal HRR for the one alias
    // lane is cheaper than perturbing every distinct packet.
    if (sidecar.alias_lane >= 0) {
        const int lane = sidecar.alias_lane;
        const unsigned alias_bit = 1u << static_cast<unsigned>(lane);
        for (int ab = 0; ab < 9; ++ab) {
            if ((sidecar.moving_component_bits[ab] & alias_bit) == 0u) {
                continue;
            }
            const std::uint32_t lhs_pair = sidecar.moving_pair[ab][lane];
            const int lhs_slot = sidecar.alias_fixed_slot[ab];
#define TLNO_RYS_J_PPPP_ALIAS_CELL(CD) do { \
            if ((sidecar.fixed_component_bits & (1u << (CD))) != 0u) { \
                const std::uint32_t rhs_pair = sidecar.fixed_pair[(CD)]; \
                if (rhs_pair <= lhs_pair) { \
                    const __m256d raw = rys_j_pppp_final_value( \
                        ab, (CD), v_bra_r, v_ket_r, acc); \
                    const double value = \
                        rys_j_extract_lane_runtime(raw, lane); \
                    if (lhs_pair == rhs_pair) { \
                        alias_delta[lhs_slot] += value * rho[lhs_pair]; \
                    } else { \
                        alias_delta[lhs_slot] += value * rho[rhs_pair]; \
                        alias_delta[(CD)] += value * rho[lhs_pair]; \
                    } \
                    rys_j_pppp_record_alias_cell( \
                        sidecar, lhs_pair, rhs_pair, value); \
                } \
            } \
        } while (false)
            TLNO_RYS_J_PPPP_ALIAS_CELL(0);
            TLNO_RYS_J_PPPP_ALIAS_CELL(1);
            TLNO_RYS_J_PPPP_ALIAS_CELL(2);
            TLNO_RYS_J_PPPP_ALIAS_CELL(3);
            TLNO_RYS_J_PPPP_ALIAS_CELL(4);
            TLNO_RYS_J_PPPP_ALIAS_CELL(5);
            TLNO_RYS_J_PPPP_ALIAS_CELL(6);
            TLNO_RYS_J_PPPP_ALIAS_CELL(7);
            TLNO_RYS_J_PPPP_ALIAS_CELL(8);
#undef TLNO_RYS_J_PPPP_ALIAS_CELL
        }
    }

#define TLNO_RYS_J_PPPP_COMMIT_FIXED(CD, FIXED_DELTA) do { \
    if ((sidecar.fixed_component_bits & (1u << (CD))) != 0u) { \
        sidecar.context->j_pair[sidecar.fixed_pair[(CD)]] += \
            rys_j_horizontal_sum((FIXED_DELTA)) + alias_delta[(CD)]; \
        rys_j_pppp_record_fixed_commit(); \
    } \
} while (false)
    TLNO_RYS_J_PPPP_COMMIT_FIXED(0, fixed_delta0);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(1, fixed_delta1);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(2, fixed_delta2);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(3, fixed_delta3);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(4, fixed_delta4);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(5, fixed_delta5);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(6, fixed_delta6);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(7, fixed_delta7);
    TLNO_RYS_J_PPPP_COMMIT_FIXED(8, fixed_delta8);
#undef TLNO_RYS_J_PPPP_COMMIT_FIXED
}

#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    !defined(TLNO_BUILD_RYS_VALIDATION) || \
    defined(TLNO_VALIDATE_RYS_SPHERICAL_PACKED)
// WHAT: close pp|pp in two output-stationary strips, four fixed columns and
// then five, while carrying only the nine moving-row partials between strips.
// WHY: the nine simultaneously-live fixed vectors in the Opt2 baseline force
// GCC to clone a large terminal body into every pp|pp producer instantiation.
// Four plus five fits the AVX2 register budget and this single non-cloned host
// bounds both register pressure and text growth.
// PURPOSE: retain the row digest's nine final horizontal sums without
// materializing an 81-cell ERI packet or returning to per-cell J updates.
__attribute__((noinline, noclone, hot)) void rys_j_pppp_digest_tiled45(
    const RysJPpppRowSidecar& sidecar,
    const __m256d (&v_bra_r)[3],
    const __m256d (&v_ket_r)[3],
    const __m256d (&acc)[81]) noexcept {
    double alias_delta[9] = {};
    const double* const __restrict rho =
        sidecar.context->j_pair_density;

    // A same-group lane follows the canonical pair triangle and contributes
    // only to fixed-group destinations.  Resolve it once, outside both strips,
    // so the ordinary distinct-lane loops contain no alias branch.
    if (sidecar.alias_lane >= 0) {
        const int lane = sidecar.alias_lane;
        const unsigned alias_bit = 1u << static_cast<unsigned>(lane);
        for (int ab = 0; ab < 9; ++ab) {
            if ((sidecar.moving_component_bits[ab] & alias_bit) == 0u) {
                continue;
            }
            const std::uint32_t lhs_pair = sidecar.moving_pair[ab][lane];
            const int lhs_slot = sidecar.alias_fixed_slot[ab];
#define TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(CD) do { \
            if ((sidecar.fixed_component_bits & (1u << (CD))) != 0u) { \
                const std::uint32_t rhs_pair = sidecar.fixed_pair[(CD)]; \
                if (rhs_pair <= lhs_pair) { \
                    const __m256d raw = rys_j_pppp_final_value( \
                        ab, (CD), v_bra_r, v_ket_r, acc); \
                    const double value = \
                        rys_j_extract_lane_runtime(raw, lane); \
                    if (lhs_pair == rhs_pair) { \
                        alias_delta[lhs_slot] += value * rho[lhs_pair]; \
                    } else { \
                        alias_delta[lhs_slot] += value * rho[rhs_pair]; \
                        alias_delta[(CD)] += value * rho[lhs_pair]; \
                    } \
                    rys_j_pppp_record_alias_cell( \
                        sidecar, lhs_pair, rhs_pair, value); \
                } \
            } \
        } while (false)
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(0);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(1);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(2);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(3);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(4);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(5);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(6);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(7);
            TLNO_RYS_J_PPPP_TILED_ALIAS_CELL(8);
#undef TLNO_RYS_J_PPPP_TILED_ALIAS_CELL
        }
    }

    // Strip 0: four fixed destinations remain in YMM registers.  The moving
    // row is stored once so strip 1 can finish it without recomputing HRR.
    alignas(32) double moving_partial[9][4] = {};
    __m256d fixed_delta0 = _mm256_setzero_pd();
    __m256d fixed_delta1 = _mm256_setzero_pd();
    __m256d fixed_delta2 = _mm256_setzero_pd();
    __m256d fixed_delta3 = _mm256_setzero_pd();
    for (int ab = 0; ab < 9; ++ab) {
        const unsigned distinct_live =
            sidecar.moving_component_bits[ab] & sidecar.distinct_bits;
        if (distinct_live == 0u) continue;
        const __m256d distinct_mask =
            rys_j_active_lane_mask(distinct_live);
        const __m256d moving_density = _mm256_maskload_pd(
            sidecar.moving_density[ab],
            _mm256_castpd_si256(distinct_mask));
        __m256d moving_delta = _mm256_setzero_pd();
#define TLNO_RYS_J_PPPP_TILED_CELL(CD, FIXED_DELTA) do { \
        if ((sidecar.fixed_component_bits & (1u << (CD))) != 0u) { \
            const __m256d raw = rys_j_pppp_final_value( \
                ab, (CD), v_bra_r, v_ket_r, acc); \
            moving_delta = _mm256_fmadd_pd( \
                raw, _mm256_set1_pd(sidecar.fixed_density[(CD)]), \
                moving_delta); \
            (FIXED_DELTA) = _mm256_fmadd_pd( \
                raw, moving_density, (FIXED_DELTA)); \
            rys_j_pppp_record_distinct_cell( \
                sidecar, ab, (CD), distinct_live, raw); \
        } \
    } while (false)
        TLNO_RYS_J_PPPP_TILED_CELL(0, fixed_delta0);
        TLNO_RYS_J_PPPP_TILED_CELL(1, fixed_delta1);
        TLNO_RYS_J_PPPP_TILED_CELL(2, fixed_delta2);
        TLNO_RYS_J_PPPP_TILED_CELL(3, fixed_delta3);
#undef TLNO_RYS_J_PPPP_TILED_CELL
        _mm256_store_pd(moving_partial[ab], moving_delta);
    }

#define TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(CD, FIXED_DELTA) do { \
    if ((sidecar.fixed_component_bits & (1u << (CD))) != 0u) { \
        sidecar.context->j_pair[sidecar.fixed_pair[(CD)]] += \
            rys_j_horizontal_sum((FIXED_DELTA)) + alias_delta[(CD)]; \
        rys_j_pppp_record_fixed_commit(); \
    } \
} while (false)
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(0, fixed_delta0);
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(1, fixed_delta1);
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(2, fixed_delta2);
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(3, fixed_delta3);

    // Strip 1: finish each moving row, commit it once, then reduce the five
    // remaining fixed destinations.  Total cross-strip traffic is bounded at
    // nine aligned stores plus nine aligned loads per pp|pp packet.
    __m256d fixed_delta4 = _mm256_setzero_pd();
    __m256d fixed_delta5 = _mm256_setzero_pd();
    __m256d fixed_delta6 = _mm256_setzero_pd();
    __m256d fixed_delta7 = _mm256_setzero_pd();
    __m256d fixed_delta8 = _mm256_setzero_pd();
    for (int ab = 0; ab < 9; ++ab) {
        const unsigned distinct_live =
            sidecar.moving_component_bits[ab] & sidecar.distinct_bits;
        if (distinct_live == 0u) continue;
        const __m256d distinct_mask =
            rys_j_active_lane_mask(distinct_live);
        const __m256d moving_density = _mm256_maskload_pd(
            sidecar.moving_density[ab],
            _mm256_castpd_si256(distinct_mask));
        __m256d moving_delta = _mm256_load_pd(moving_partial[ab]);
#define TLNO_RYS_J_PPPP_TILED_CELL(CD, FIXED_DELTA) do { \
        if ((sidecar.fixed_component_bits & (1u << (CD))) != 0u) { \
            const __m256d raw = rys_j_pppp_final_value( \
                ab, (CD), v_bra_r, v_ket_r, acc); \
            moving_delta = _mm256_fmadd_pd( \
                raw, _mm256_set1_pd(sidecar.fixed_density[(CD)]), \
                moving_delta); \
            (FIXED_DELTA) = _mm256_fmadd_pd( \
                raw, moving_density, (FIXED_DELTA)); \
            rys_j_pppp_record_distinct_cell( \
                sidecar, ab, (CD), distinct_live, raw); \
        } \
    } while (false)
        TLNO_RYS_J_PPPP_TILED_CELL(4, fixed_delta4);
        TLNO_RYS_J_PPPP_TILED_CELL(5, fixed_delta5);
        TLNO_RYS_J_PPPP_TILED_CELL(6, fixed_delta6);
        TLNO_RYS_J_PPPP_TILED_CELL(7, fixed_delta7);
        TLNO_RYS_J_PPPP_TILED_CELL(8, fixed_delta8);
#undef TLNO_RYS_J_PPPP_TILED_CELL

        alignas(32) double moving_lanes[4];
        _mm256_store_pd(moving_lanes, moving_delta);
        for (int lane = 0; lane < 4; ++lane) {
            if ((distinct_live & (1u << lane)) == 0u) continue;
            sidecar.context->j_pair[
                sidecar.moving_pair[ab][lane]] += moving_lanes[lane];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
            ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
        }
    }

    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(4, fixed_delta4);
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(5, fixed_delta5);
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(6, fixed_delta6);
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(7, fixed_delta7);
    TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED(8, fixed_delta8);
#undef TLNO_RYS_J_PPPP_TILED_COMMIT_FIXED
}
#endif

template <bool BraP, int Lane>
__attribute__((always_inline)) inline void
rys_j_ssps_common_preemit_lane(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    const std::vector<PackedPairComponentH>& fixed_refs,
    __m256d e0,
    __m256d e1,
    __m256d e2,
    double (&fixed_delta)[3]) noexcept {
    const std::uint32_t moving_group =
        bra_bin.pair_group_ids[bra_pos + static_cast<std::size_t>(Lane)];
    const auto& moving_refs = low_batch_pair_components(bd)[
        static_cast<std::size_t>(moving_group)];
    double moving_delta = 0.0;
    rys_j_ssps_common_preemit_component<BraP, Lane, 0>(
        context, moving_refs, fixed_refs, e0,
        moving_delta, fixed_delta);
    rys_j_ssps_common_preemit_component<BraP, Lane, 1>(
        context, moving_refs, fixed_refs, e1,
        moving_delta, fixed_delta);
    rys_j_ssps_common_preemit_component<BraP, Lane, 2>(
        context, moving_refs, fixed_refs, e2,
        moving_delta, fixed_delta);
    if constexpr (!BraP) {
        context.j_pair[moving_refs.front().pair_index] += moving_delta;
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
        ++g_rys_direct_j_speed_probe_audit.candidate_physical_commits;
#endif
    }
}

template <bool BraP>
__attribute__((always_inline)) inline void
rys_j_ssps_common_preemit_packet(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& bra_bin,
    std::size_t bra_pos,
    const std::vector<PackedPairComponentH>& fixed_refs,
    unsigned active_bits,
    __m256d e0,
    __m256d e1,
    __m256d e2) noexcept {
    double fixed_delta[3] = {};
    if ((active_bits & 1u) != 0u)
        rys_j_ssps_common_preemit_lane<BraP, 0>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, fixed_delta);
    if ((active_bits & 2u) != 0u)
        rys_j_ssps_common_preemit_lane<BraP, 1>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, fixed_delta);
    if ((active_bits & 4u) != 0u)
        rys_j_ssps_common_preemit_lane<BraP, 2>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, fixed_delta);
    if ((active_bits & 8u) != 0u)
        rys_j_ssps_common_preemit_lane<BraP, 3>(
            context, bd, bra_bin, bra_pos, fixed_refs,
            e0, e1, e2, fixed_delta);

    constexpr int fixed_components = BraP ? 1 : 3;
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
