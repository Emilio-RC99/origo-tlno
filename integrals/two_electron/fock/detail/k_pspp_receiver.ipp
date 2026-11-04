#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_K_PSPP_RECEIVER_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_K_PSPP_RECEIVER_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

#ifdef TLNO_RYS_DIRECT_EXCHANGE
// Close pp|pp while its radial accumulators are available.  Project one row
// at a time and retain the four crossed 3x3 K tiles until the quartet closes.
// Each ERI is read once; the output matrix receives at most 36 additions per
// quartet instead of four additions for each of its up to 81 ERIs.  The four
// tiles stay separate even when shells coincide, so their final additions
// preserve every repeated-index contribution without aliasing local carries.
__attribute__((noinline, noclone)) void rys_k_pppp_preemit_packet(
    const RysJkEmitContext& context,
    const BasisDataHGP& bd,
    const RysLowPairBinH& moving_bin,
    std::size_t moving_pos,
    std::uint32_t fixed_group,
    unsigned active_bits,
    const __m256d (&bra_r)[3],
    const __m256d (&ket_r)[3],
    const __m256d (&acc)[81]) noexcept {
    constexpr int unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    constexpr int output_row[4] = {0, 0, 1, 1};
    constexpr int output_col[4] = {2, 3, 2, 3};
    constexpr int density_row[4] = {1, 1, 0, 0};
    constexpr int density_col[4] = {3, 2, 3, 2};
    struct Lane {
        int base[4] = {};
        const PackedPairComponentH* lhs[9] = {};
        bool same_group = false;
        bool rectangular = false;
        double rho[4][9] = {};
        double delta[4][9] = {};
    };
    Lane lanes[4];
    const PackedPairComponentH* rhs[9] = {};
    for (const auto& ref : low_batch_pair_components(bd)[fixed_group]) {
        rhs[ref.component_index] = &ref;
    }
    const auto [cs, ds] = bd.pair_groups[fixed_group];
    const std::size_t M = context.M;
    for (unsigned lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        auto& state = lanes[lane];
        const auto group = moving_bin.pair_group_ids[moving_pos + lane];
        const auto [as, bs] = bd.pair_groups[group];
        state.base[0] = bd.spherical_group_bases[as];
        state.base[1] = bd.spherical_group_bases[bs];
        state.base[2] = bd.spherical_group_bases[cs];
        state.base[3] = bd.spherical_group_bases[ds];
        state.same_group = group == fixed_group;
        // Distinct shells within each pair have nine unit-normalized p/p
        // components and no diagonal AO pairs.  Distinct pair groups also
        // exclude pair equality, so every coefficient is exactly 2*ERI.
        // Crossed shell coincidences remain safe: the four carries below
        // are independent and their final additions preserve all aliases.
        state.rectangular = as != bs && cs != ds && !state.same_group;
        for (const auto& ref : low_batch_pair_components(bd)[group]) {
            state.lhs[ref.component_index] = &ref;
        }
        for (int family = 0; family < 4; ++family) {
            for (int i = 0; i < 3; ++i) {
                const auto row = static_cast<std::size_t>(
                    state.base[density_row[family]] + i) * M;
                for (int j = 0; j < 3; ++j) {
                    state.rho[family][3 * i + j] = context.p[
                        row + state.base[density_col[family]] + j];
                }
            }
        }
    }
    alignas(32) double projected_row[9][4];
    for (int ab = 0; ab < 9; ++ab) {
        const int a = ab / 3;
        const int b = ab % 3;
        const int bra_hi = 3 + unique[ab];
        for (int cd = 0; cd < 9; ++cd) {
            if (rhs[cd] == nullptr) continue;
            const int c = cd / 3;
            const int d = cd % 3;
            const int ket_hi = 3 + unique[cd];
            const auto hi = _mm256_fmadd_pd(
                bra_r[b], acc[9 * a + ket_hi], acc[9 * bra_hi + ket_hi]);
            const auto lo = _mm256_fmadd_pd(
                bra_r[b], acc[9 * a + c], acc[9 * bra_hi + c]);
            _mm256_store_pd(projected_row[cd],
                _mm256_fmadd_pd(ket_r[d], lo, hi));
        }
        for (unsigned lane = 0; lane < 4; ++lane) {
            if ((active_bits & (1u << lane)) == 0u) continue;
            auto& state = lanes[lane];
            if (state.rectangular) {
                const double* const bdensity = state.rho[0] + 3 * b;
                const double* const adensity = state.rho[2] + 3 * a;
                double* const adelta = state.delta[1] + 3 * a;
                double* const bdelta = state.delta[3] + 3 * b;
                for (int c = 0; c < 3; ++c) {
                    const double u0 = 2.0 * projected_row[3 * c][lane];
                    const double u1 = 2.0 * projected_row[3 * c + 1][lane];
                    const double u2 = 2.0 * projected_row[3 * c + 2][lane];
                    // Two three-term dot products close the contracted d
                    // direction before touching the AC and BC output cells.
                    state.delta[0][3 * a + c] +=
                        u0 * bdensity[0] + u1 * bdensity[1] + u2 * bdensity[2];
                    state.delta[2][3 * b + c] +=
                        u0 * adensity[0] + u1 * adensity[1] + u2 * adensity[2];
                    const double bc = state.rho[1][3 * b + c];
                    const double ac = state.rho[3][3 * a + c];
                    adelta[0] += u0 * bc;
                    adelta[1] += u1 * bc;
                    adelta[2] += u2 * bc;
                    bdelta[0] += u0 * ac;
                    bdelta[1] += u1 * ac;
                    bdelta[2] += u2 * ac;
#if defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM)
                    for (int d = 0; d < 3; ++d) {
                        assert(state.lhs[ab] && state.lhs[ab]->norm == 1.0);
                        assert(rhs[3 * c + d] && rhs[3 * c + d]->norm == 1.0);
                        rys_jk_record_one(context, state.lhs[ab]->pair_index,
                            rhs[3 * c + d]->pair_index,
                            projected_row[3 * c + d][lane]);
                    }
#endif
                }
                continue;
            }
            const auto* lhs = state.lhs[ab];
            if (lhs == nullptr) continue;
            const double lhs_scale = context.pair_info[lhs->pair_index].scale;
            for (int cd = 0; cd < 9; ++cd) {
                const auto* right = rhs[cd];
                if (right == nullptr || (state.same_group &&
                    right->pair_index > lhs->pair_index)) continue;
                const int c = cd / 3;
                const int d = cd % 3;
                const double value = projected_row[cd][lane] *
                                     lhs->norm * right->norm;
                rys_jk_record_one(context, lhs->pair_index,
                                  right->pair_index, value);
                double u = 2.0 * value * lhs_scale *
                           context.pair_info[right->pair_index].scale;
                if (lhs->pair_index == right->pair_index) u *= 0.5;
                state.delta[0][3 * a + c] += u * state.rho[0][3 * b + d];
                state.delta[1][3 * a + d] += u * state.rho[1][3 * b + c];
                state.delta[2][3 * b + c] += u * state.rho[2][3 * a + d];
                state.delta[3][3 * b + d] += u * state.rho[3][3 * a + c];
            }
        }
    }
    for (unsigned lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const auto& state = lanes[lane];
        for (int family = 0; family < 4; ++family) {
            for (int i = 0; i < 3; ++i) {
                const auto row = static_cast<std::size_t>(
                    state.base[output_row[family]] + i) * M;
                for (int j = 0; j < 3; ++j) {
                    context.g[row + state.base[output_col[family]] + j] +=
                        state.delta[family][3 * i + j];
                }
            }
        }
    }
}

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
    const __m256d (&acc_hi)[18]) noexcept {
    struct LaneMeta {
        int a0 = 0, b0 = 0, c0 = 0, z = 0;
        bool same_pp = false;
    };
    struct LaneCarry {
        double block[3][3] = {};
        double vec[3] = {};
    };
    constexpr int pp_unique[9] = {0, 1, 2, 1, 3, 4, 2, 4, 5};
    LaneMeta meta[4];
    LaneCarry carry[4];
    unsigned rectangular_bits = 0u;
    for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const std::uint32_t moving_group = moving_bin.pair_group_ids[
            moving_pos + lane];
        const std::uint32_t pp_group = BraPP ? moving_group : fixed_group;
        const std::uint32_t ps_group = BraPP ? fixed_group : moving_group;
        const auto [a_shell, b_shell] =
            bd.pair_groups[static_cast<std::size_t>(pp_group)];
        const auto [c_shell, z_shell] =
            bd.pair_groups[static_cast<std::size_t>(ps_group)];
        meta[lane].a0 = bd.spherical_group_bases[
            static_cast<std::size_t>(a_shell)];
        meta[lane].b0 = bd.spherical_group_bases[
            static_cast<std::size_t>(b_shell)];
        meta[lane].c0 = bd.spherical_group_bases[
            static_cast<std::size_t>(c_shell)];
        meta[lane].z = bd.spherical_group_bases[
            static_cast<std::size_t>(z_shell)];
        meta[lane].same_pp = a_shell == b_shell;
        if (!meta[lane].same_pp) rectangular_bits |= 1u << lane;
    }

    const std::size_t M = context.M;
    const double* const p = context.p;
    double* const k_matrix = context.g;
    alignas(32) double row[3][3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            if (j > i && rectangular_bits == 0u) continue;
            const int pp = 3 * i + j;
            for (int ps = 0; ps < 3; ++ps) {
                _mm256_store_pd(
                    row[j][ps],
                    _mm256_fmadd_pd(
                        vR[j], acc_lo[3 * ps + i],
                        acc_hi[6 * ps + pp_unique[pp]]));
            }
        }
        for (std::size_t lane = 0; lane < 4; ++lane) {
            if ((active_bits & (1u << lane)) == 0u) continue;
            const LaneMeta& m = meta[lane];
            double urow[3] = {};
            double vrow = 0.0;
            const int j_limit = m.same_pp ? i + 1 : 3;
            for (int j = 0; j < j_limit; ++j) {
                const std::uint32_t pp_pair = EriPackedGAccessor::pair_index(
                    m.a0 + i, m.b0 + j);
                for (int ps = 0; ps < 3; ++ps) {
                    const double value = row[j][ps][lane];
                    const std::uint32_t ps_pair =
                        EriPackedGAccessor::pair_index(m.c0 + ps, m.z);
                    if constexpr (BraPP) {
                        rys_jk_record_one(context, pp_pair, ps_pair, value);
                    } else {
                        rys_jk_record_one(context, ps_pair, pp_pair, value);
                    }
                    const double q =
                        (m.same_pp && i == j ? 1.0 : 2.0) * value;
                    const double p_bz = BraPP
                        ? p[static_cast<std::size_t>(m.b0 + j) * M + m.z]
                        : p[static_cast<std::size_t>(m.z) * M + m.b0 + j];
                    const double p_bc = BraPP
                        ? p[static_cast<std::size_t>(m.b0 + j) * M + m.c0 + ps]
                        : p[static_cast<std::size_t>(m.c0 + ps) * M + m.b0 + j];
                    const double p_az = BraPP
                        ? p[static_cast<std::size_t>(m.a0 + i) * M + m.z]
                        : p[static_cast<std::size_t>(m.z) * M + m.a0 + i];
                    const double p_ac = BraPP
                        ? p[static_cast<std::size_t>(m.a0 + i) * M + m.c0 + ps]
                        : p[static_cast<std::size_t>(m.c0 + ps) * M + m.a0 + i];
                    if (m.same_pp) {
                        carry[lane].block[i][ps] += q * p_bz;
                        carry[lane].vec[i] += q * p_bc;
                        carry[lane].block[j][ps] += q * p_az;
                        carry[lane].vec[j] += q * p_ac;
                    } else {
                        urow[ps] += q * p_bz;
                        vrow += q * p_bc;
                        carry[lane].block[j][ps] += q * p_az;
                        carry[lane].vec[j] += q * p_ac;
                    }
                }
            }
            if (!m.same_pp) {
                for (int ps = 0; ps < 3; ++ps) {
                    if constexpr (BraPP) {
                        k_matrix[static_cast<std::size_t>(m.a0 + i) * M +
                                 m.c0 + ps] += urow[ps];
                    } else {
                        k_matrix[static_cast<std::size_t>(m.c0 + ps) * M +
                                 m.a0 + i] += urow[ps];
                    }
                }
                if constexpr (BraPP) {
                    k_matrix[static_cast<std::size_t>(m.a0 + i) * M + m.z] +=
                        vrow;
                } else {
                    k_matrix[static_cast<std::size_t>(m.z) * M + m.a0 + i] +=
                        vrow;
                }
            }
        }
    }
    for (std::size_t lane = 0; lane < 4; ++lane) {
        if ((active_bits & (1u << lane)) == 0u) continue;
        const LaneMeta& m = meta[lane];
        const int base = m.same_pp ? m.a0 : m.b0;
        for (int component = 0; component < 3; ++component) {
            for (int ps = 0; ps < 3; ++ps) {
                if constexpr (BraPP) {
                    k_matrix[static_cast<std::size_t>(base + component) * M +
                             m.c0 + ps] += carry[lane].block[component][ps];
                } else {
                    k_matrix[static_cast<std::size_t>(m.c0 + ps) * M +
                             base + component] +=
                        carry[lane].block[component][ps];
                }
            }
            if constexpr (BraPP) {
                k_matrix[static_cast<std::size_t>(base + component) * M + m.z] +=
                    carry[lane].vec[component];
            } else {
                k_matrix[static_cast<std::size_t>(m.z) * M + base + component] +=
                    carry[lane].vec[component];
            }
        }
    }
}
#endif
#endif
