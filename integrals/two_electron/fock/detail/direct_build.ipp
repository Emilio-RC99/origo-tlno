#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_DIRECT_BUILD_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_DIRECT_BUILD_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

template <class DirectEmit>
static EriDirectJKResult build_jk_direct_impl(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options) {
    auto bd = packed_plan_basis_data(plan);
    if (!bd) {
        throw std::runtime_error("eri_ao_tensor_hgp: null packed ERI plan.");
    }
    const int M = bd->spherical_function_count;
    const std::size_t m = static_cast<std::size_t>(M);
    if (density.size() != m * m) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: density must be spherical M x M row-major.");
    }
    const std::size_t pair_count = m * (m + 1u) / 2u;

    // WHAT: validate a symmetric pair of P entries and return the larger
    // magnitude of its two accepted orientations. WHY/PURPOSE: the same read
    // serves both the s8 contract and the conservative CVHF bound, without
    // choosing an orientation that may be slightly smaller by rounding.
    constexpr double symmetry_tol =
        128.0 * std::numeric_limits<double>::epsilon();
    const auto checked_density_pair_abs_max =
        [&](const std::size_t i, const std::size_t j) -> double {
        const double pij = density[i * m + j];
        if (!tlno::core::math::is_finite_binary64(pij)) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: density contains a non-finite value.");
        }
        if (i == j) {
            return std::abs(pij);
        }
        const double pji = density[j * m + i];
        if (!tlno::core::math::is_finite_binary64(pji)) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: density contains a non-finite value.");
        }
        const double abs_pij = std::abs(pij);
        const double abs_pji = std::abs(pji);
        const double scale = std::max({1.0, abs_pij, abs_pji});
        if (std::abs(pij - pji) > symmetry_tol * scale) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: density must be symmetric.");
        }
        return std::max(abs_pij, abs_pji);
    };

    // WHAT: fuse the validation of P with the table of maxima only on the
    // density route that will actually build it. WHY/PURPOSE: removes the
    // duplicated O(M^2) pass without changing the error order of options that
    // cannot enter that route. The macro keeps both passes alive for an A/B
    // comparison compiled from exactly this same source.
#if defined(TLNO_JK_FORCE_SEPARATE_DENSITY_VALIDATION)
    constexpr bool allow_fused_density_validation = false;
#else
    constexpr bool allow_fused_density_validation = true;
#endif
    const bool fuse_density_validation =
        allow_fused_density_validation &&
        tlno::core::math::is_finite_binary64(options.density_threshold) &&
        options.density_threshold > 0.0 && options.decay_threshold == 0.0 &&
        bd->jk_pair_info.size() == pair_count;
    if (!fuse_density_validation) {
        // WHAT: keep the independent validation when there will be no density
        // tables or when the A/B control asks for it. WHY/PURPOSE: P still
        // fails early, and the control reproduces the previous cost.
        for (std::size_t i = 0; i < m; ++i) {
            (void)checked_density_pair_abs_max(i, i);
            for (std::size_t j = 0; j < i; ++j) {
                (void)checked_density_pair_abs_max(i, j);
            }
        }
    }
    if (!tlno::core::math::is_finite_binary64(options.density_threshold) ||
        options.density_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: density_threshold must be finite and >= 0.");
    }
    if (options.density_threshold > 0.0 &&
        options.decay_threshold > 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: density screening requires Schwarz bounds and "
            "cannot be combined with the heuristic decay metric.");
    }
    if (bd->jk_pair_info.size() != pair_count) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incomplete direct J/K pair metadata.");
    }
    const std::vector<RysJkPairInfo>& pair_info = bd->jk_pair_info;

    // Per-group density tables, O(ng^2), rebuilt once per call because P
    // changes in every SCF iteration.
    //
    // drow[g] = max_h D[g][h] and dpair[pg] = max(drow[left], drow[right]) give
    // an UPPER bound of the density maximum over the quartet's four groups:
    // every crossed term D[ga][gc] is dominated by the row of either of its
    // two indices.  See the RysBatchScreen comment for why the exact maximum
    // over the 10 pairs is not used.
    // Copy of the options: the effective Schwarz threshold is raised with the
    // density (see below) without mutating what the caller asked for.
    EriPackedBuildOptions effective_options = options;
    std::vector<double> shell_rho;
    std::vector<double> pair_group_density_upper;
    std::vector<double> group_pair_density_max;
    RysDensityScreenContext density_context;
    constexpr bool j_only_endpoint =
        rys_emit_is_j_only_v<DirectEmit>;
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    constexpr bool k_only_endpoint =
        rys_emit_is_k_only_v<DirectEmit>;
#else
    constexpr bool k_only_endpoint = false;
#endif
    if (options.density_threshold > 0.0) {
        const std::size_t ng = bd->groups.size();
        const std::size_t npg = bd->pair_groups.size();

        // Max |P| over the AO block of each pair of groups.
        group_pair_density_max.assign(ng * ng, 0.0);
        // WHAT: omit the group-row exchange summary in the J-only endpoint.
        // WHY: only K consults cross-group density blocks; constructing its
        // O(ng^2) summary before every timed J call is pure unrelated work.
        // PURPOSE: J setup contains only the block maxima needed by its two
        // Coulomb terms, while the combined G endpoint remains unchanged.
        std::vector<double> density_row_max;
        if constexpr (!j_only_endpoint) density_row_max.assign(ng, 0.0);
        for (std::size_t gi = 0; gi < ng; ++gi) {
            const int i_begin = bd->spherical_group_bases[gi];
            const int i_end = i_begin + n_sph(bd->groups[gi].L);
            for (std::size_t gj = gi; gj < ng; ++gj) {
                const int j_begin = bd->spherical_group_bases[gj];
                const int j_end = j_begin + n_sph(bd->groups[gj].L);
                double max_abs = 0.0;
                if (fuse_density_validation) {
                    // WHAT: diagonal blocks traverse only their triangle;
                    // rectangular blocks read each pair once and the lambda
                    // validates both orientations. WHY/PURPOSE: all of P is
                    // validated and summarized with exactly M^2 loads, keeping
                    // the symmetric maximum a conservative rejection needs.
                    for (int i2 = i_begin; i2 < i_end; ++i2) {
                        const int fused_j_end =
                            gi == gj ? i2 + 1 : j_end;
                        for (int j2 = j_begin; j2 < fused_j_end; ++j2) {
                            max_abs = std::max(
                                max_abs,
                                checked_density_pair_abs_max(
                                    static_cast<std::size_t>(i2),
                                    static_cast<std::size_t>(j2)));
                        }
                    }
                } else {
                    // WHAT: the control keeps the rectangular sweep and the
                    // two original orientations. WHY/PURPOSE: it isolates the
                    // value of the fusion, not a rewrite of the baseline.
                    for (int i2 = i_begin; i2 < i_end; ++i2) {
                        const std::size_t row =
                            static_cast<std::size_t>(i2) * m;
                        for (int j2 = j_begin; j2 < j_end; ++j2) {
                            const std::size_t column =
                                static_cast<std::size_t>(j2);
                            max_abs = std::max({
                                max_abs,
                                std::abs(density[row + column]),
                                std::abs(density[
                                    column * m +
                                    static_cast<std::size_t>(i2)])});
                        }
                    }
                }
                group_pair_density_max[gi * ng + gj] = max_abs;
                group_pair_density_max[gj * ng + gi] = max_abs;
                // WHAT: accumulate drow while materializing D by blocks.
                // WHY: the task bound must not introduce another O(ng^2)
                // sweep per SCF iteration.
                if constexpr (!j_only_endpoint) {
                    density_row_max[gi] = std::max(
                        density_row_max[gi], max_abs);
                    density_row_max[gj] = std::max(
                        density_row_max[gj], max_abs);
                }
            }
        }

        // rho_ab is exactly the block maximum already computed above.  An
        // earlier implementation rescanned packed_spherical_pair_components
        // here to recover the same AO block maximum, duplicating work on every
        // P/dP update.  Pair-group endpoints are immutable plan metadata too.
        shell_rho.assign(npg, 0.0);
        if constexpr (!j_only_endpoint) {
            pair_group_density_upper.assign(npg, 0.0);
        }
        if (bd->pair_group_left.size() != npg ||
            bd->pair_group_right.size() != npg) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: incomplete density-screen pair metadata.");
        }
        for (std::size_t pg = 0; pg < npg; ++pg) {
            const auto [left, right] = bd->pair_groups[pg];
            shell_rho[pg] = group_pair_density_max[
                static_cast<std::size_t>(left) * ng +
                static_cast<std::size_t>(right)];
            // WHAT: dpair dominates any exchange block that connects this
            // pair group with another. WHY: min(dpair_ab,dpair_cd) is a safe,
            // factorizable upper bound of the four crossed D blocks.
            if constexpr (!j_only_endpoint) {
                pair_group_density_upper[pg] = std::max(
                    density_row_max[static_cast<std::size_t>(left)],
                    density_row_max[static_cast<std::size_t>(right)]);
            }
        }

        density_context.shell_rho = shell_rho.data();
        density_context.pair_group_density_upper =
            j_only_endpoint ? nullptr : pair_group_density_upper.data();
        density_context.group_pair_density_max = group_pair_density_max.data();
        density_context.ng = static_cast<int>(ng);
        density_context.pair_group_left = bd->pair_group_left.data();
        density_context.pair_group_right = bd->pair_group_right.data();
        density_context.density_break = options.density_threshold;
        density_context.j_only = j_only_endpoint;
#ifdef TLNO_RYS_DIRECT_EXCHANGE
        density_context.k_only = k_only_endpoint;
#endif

        // rho_global = 4 * max(shell_rho), and q_break = max(thresh_int,
        // thresh_den / rho_global).  This is THE piece that saves: it raises
        // the global Schwarz cutoff, so the screened sweep -- which already
        // sorts and truncates by Schwarz -- skips the whole tail of the pair
        // list instead of discarding individual quartets.
        double rho_global = 0.0;
        for (const double value : shell_rho) {
            rho_global = std::max(rho_global, value);
        }
        // CVHF's vj cap owns a factor four; its vk cap does not.  The combined
        // endpoint uses the larger Coulomb envelope.  K-only therefore raises
        // the global Q cutoff with max|P|, not 4*max|P|.
        if constexpr (!k_only_endpoint) rho_global *= 4.0;
        if (rho_global > 0.0) {
            effective_options.schwarz_threshold =
                std::max(effective_options.schwarz_threshold,
                         options.density_threshold / rho_global);
        }
    }

    // A zero density pair contributes nothing to direct J when its opposite
    // pair is also zero. The production SAD density has many exact-zero AO
    // blocks, so retain this lossless test even when approximate density
    // screening is disabled (density_threshold == 0). It is independent of
    // the Schwarz, decay, and primitive FASTSCHWARZ thresholds.
#if defined(TLNO_RYS_J_DISABLE_EXACT_ZERO_SCREEN)
    constexpr bool allow_exact_zero_j_screen = false;
#else
    constexpr bool allow_exact_zero_j_screen = true;
#endif
    const bool exact_zero_j_screen =
        allow_exact_zero_j_screen && j_only_endpoint &&
        tlno::integrals::twoe::portable_kernel && options.density_threshold == 0.0;
    if (exact_zero_j_screen) {
        const std::size_t npg = bd->pair_groups.size();
        shell_rho.assign(npg, 0.0);
        for (std::size_t pg = 0; pg < npg; ++pg) {
            const auto [left, right] = bd->pair_groups[pg];
            const int i_begin = bd->spherical_group_bases[left];
            const int j_begin = bd->spherical_group_bases[right];
            const int i_end = i_begin + n_sph(bd->groups[left].L);
            const int j_end = j_begin + n_sph(bd->groups[right].L);
            double max_abs = 0.0;
            for (int i = i_begin; i < i_end; ++i) {
                const std::size_t row = static_cast<std::size_t>(i) * m;
                for (int j = j_begin; j < j_end; ++j) {
                    const std::size_t column = static_cast<std::size_t>(j);
                    max_abs = std::max({
                        max_abs, std::abs(density[row + column]),
                        std::abs(density[column * m + static_cast<std::size_t>(i)])});
                }
            }
            shell_rho[pg] = max_abs;
        }
        density_context.shell_rho = shell_rho.data();
        density_context.j_only = true;
    }
    const RysDensityScreenGuard density_guard(
        (options.density_threshold > 0.0 || exact_zero_j_screen)
            ? &density_context : nullptr);

    EriDirectJKResult out;
    out.M = M;
    out.g.assign(m * m, 0.0);

    std::vector<double> j_pair;
    std::vector<double> j_pair_density;
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
    if constexpr (rys_emit_needs_j_pair_scratch_v<DirectEmit>) {
        j_pair.assign(pair_count, 0.0);
        j_pair_density.resize(pair_count);
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const RysJkPairInfo& info = pair_info[pair];
            const std::size_t matrix_index =
                info.first_row + static_cast<std::size_t>(info.second);
            // PairInfo::scale is 1/2 for a diagonal AO pair and 1 otherwise;
            // twice that value is precisely the packed-density multiplicity.
            j_pair_density[pair] =
                2.0 * static_cast<double>(info.scale) *
                density[matrix_index];
        }
    }
#endif

    RysJkEmitContext context;
    context.g = out.g.data();
    context.p = density.data();
    context.pair_info = pair_info.data();
    context.j_pair = j_pair.empty() ? nullptr : j_pair.data();
    context.j_pair_density =
        j_pair_density.empty() ? nullptr : j_pair_density.data();
    context.M = m;
#ifdef TLNO_JK_COVERAGE_CHECKSUM
    double coverage_checksum = 0.0;
    context.coverage_checksum = &coverage_checksum;
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    long double moment_1 = 0.0L, moment_2 = 0.0L, moment_3 = 0.0L;
    unsigned long long canonical_count = 0;
    context.indexed_moment_1 = &moment_1;
    context.indexed_moment_2 = &moment_2;
    context.indexed_moment_3 = &moment_3;
    context.canonical_count = &canonical_count;
#endif
    RysJkEmitContextGuard context_guard(&context);

#ifdef TLNO_J_COMMON_PRODUCER_GTFOCK_GROUP_CACHE
    RysJGtfockGroupCache gtfock_cache(
        context.j_pair, context.j_pair_density);
    RysJGtfockCacheGuard gtfock_cache_guard(
        std::is_same_v<DirectEmit, RysDirectJCommonProducerEmit>
            ? &gtfock_cache
            : nullptr);
#endif
#ifdef TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST
    RysJVeloxTaskDigest velox_digest(
        pair_count, context.j_pair, context.j_pair_density);
    RysJVeloxDigestGuard velox_digest_guard(
        std::is_same_v<DirectEmit, RysDirectJCommonProducerEmit>
            ? &velox_digest
            : nullptr);
#endif

#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    if constexpr (rys_emit_is_j_only_v<DirectEmit>) {
        g_rys_direct_j_speed_probe_audit = {};
    }
#endif

    // Run the exact canonical scalar consumer through the shared Rys traversal.
    const EriPackedGData ignored =
        build_eri_packed_for_G_rys_impl<DirectEmit>(bd, effective_options);
    (void)ignored;

#ifdef TLNO_J_COMMON_PRODUCER_GTFOCK_GROUP_CACHE
    if constexpr (std::is_same_v <
                      DirectEmit, RysDirectJCommonProducerEmit > ) {
        gtfock_cache.flush_all();
    }
#endif
#ifdef TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST
    if constexpr (std::is_same_v <
                      DirectEmit, RysDirectJCommonProducerEmit > ) {
        velox_digest.finish();
    }
#endif

#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    if constexpr (rys_emit_is_j_only_v<DirectEmit>) {
        // WHAT: traverse the same screened workload a second time with the
        // ordinary producer dispatch and an index-only sink.
        // WHY: counters maintained by the candidate cannot reveal a quartet
        // that its own specialized dispatcher accidentally omitted.  The
        // reference policy is a distinct Emit type, so ss|ss and ss|ps take
        // their normal producers rather than the speed-probe kernels.
        // PURPOSE: an untimed diagnostic binary certifies the full multiset
        // of canonical ERIs independently of numerical accuracy or J values.
        const EriPackedGData reference_ignored =
            build_eri_packed_for_G_rys_impl<RysDirectJReferenceCountEmit>(
                bd, effective_options);
        (void)reference_ignored;

        const RysDirectJSpeedProbeAudit& audit =
            g_rys_direct_j_speed_probe_audit;
        const std::uint64_t partitioned_eris =
            audit.candidate_ssss_eris + audit.candidate_ssps_eris +
            audit.candidate_sp_block_eris +
            audit.candidate_ssds_eris + audit.scalar_other_eris;
        const bool complete =
            partitioned_eris == audit.total_canonical_eris &&
            audit.reference_canonical_eris == audit.total_canonical_eris &&
            audit.reference_index_xor == audit.actual_index_xor &&
            audit.reference_index_sum == audit.actual_index_sum &&
            audit.total_canonical_eris <=
                std::numeric_limits<std::uint64_t>::max() / 2u &&
            audit.logical_j_terms_committed ==
                2u * audit.total_canonical_eris;
        if (!complete) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: J speed-probe structural ledger "
                "mismatch.");
        }
        // WHAT: emit the structural ledger only in a separately compiled
        // diagnostic binary, after both complete stream traversals.
        // WHY: logical terms are counted at their actual write sites, while
        // count+XOR+sum compare candidate ownership against an independent
        // canonical producer walk.
        // PURPOSE: a fast result cannot pass by omitting a class, duplicating
        // an ERI, or dropping either Coulomb family; release timings still pay
        // none of these counters, the second traversal, validation or I/O.
        std::fprintf(
            stderr,
            "j_speed_probe_ledger total_canonical_eris=%llu "
            "candidate_ssss_eris=%llu candidate_ssps_eris=%llu "
            "candidate_sp_block_eris=%llu "
            "candidate_ssds_eris=%llu "
            "scalar_other_eris=%llu "
            "logical_j_terms=%llu candidate_physical_commits=%llu "
            "reference_canonical_eris=%llu "
            "actual_index_xor=%016llx actual_index_sum=%016llx "
            "reference_index_xor=%016llx reference_index_sum=%016llx\n",
            static_cast<unsigned long long>(audit.total_canonical_eris),
            static_cast<unsigned long long>(audit.candidate_ssss_eris),
            static_cast<unsigned long long>(audit.candidate_ssps_eris),
            static_cast<unsigned long long>(audit.candidate_sp_block_eris),
            static_cast<unsigned long long>(audit.candidate_ssds_eris),
            static_cast<unsigned long long>(audit.scalar_other_eris),
            static_cast<unsigned long long>(
                audit.logical_j_terms_committed),
            static_cast<unsigned long long>(audit.candidate_physical_commits),
            static_cast<unsigned long long>(
                audit.reference_canonical_eris),
            static_cast<unsigned long long>(audit.actual_index_xor),
            static_cast<unsigned long long>(audit.actual_index_sum),
            static_cast<unsigned long long>(audit.reference_index_xor),
            static_cast<unsigned long long>(audit.reference_index_sum));
    }
#endif

#ifdef TLNO_JK_DIGEST_PROBE
    // WHAT: observe the sink once, outside the hot loop.
    // WHY/PURPOSE: it forces the probe's loads/arithmetic to be kept without
    // reintroducing the per-integral volatile store that biased the previous
    // probe. This block does not exist in release builds without the macro.
    const volatile double jk_probe_observation = g_jk_probe_sink;
    (void)jk_probe_observation;
#endif

#ifdef TLNO_JK_COVERAGE_CHECKSUM
    out.coverage_checksum = coverage_checksum;
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    out.indexed_moment_1 = static_cast<double>(moment_1);
    out.indexed_moment_2 = static_cast<double>(moment_2);
    out.indexed_moment_3 = static_cast<double>(moment_3);
    out.canonical_count = canonical_count;
#endif

    // Canonical s8 owns one directed triangle; finish the public symmetric G.
    for (int i = 0; i < M; ++i) {
        for (int j = i + 1; j < M; ++j) {
            const std::size_t ij = static_cast<std::size_t>(i) * m +
                                   static_cast<std::size_t>(j);
            const std::size_t ji = static_cast<std::size_t>(j) * m +
                                   static_cast<std::size_t>(i);
            const double v = 0.5 * (out.g[ij] + out.g[ji]);
            out.g[ij] = v;
            out.g[ji] = v;
        }
    }
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
    if constexpr (rys_emit_needs_j_pair_scratch_v<DirectEmit>) {
        // The common-producer endpoint accumulates every class here.  The
        // retained legacy marker may already have closed producer-fused
        // classes in matrix space, so += remains correct for both policies.
        // Copy each canonical result to both AO orientations.
        for (std::size_t pair = 0; pair < pair_count; ++pair) {
            const RysJkPairInfo& info = pair_info[pair];
            const std::size_t lower =
                info.first_row + static_cast<std::size_t>(info.second);
            out.g[lower] += j_pair[pair];
            if (info.first != info.second) {
                const std::size_t upper =
                    static_cast<std::size_t>(info.second) * m +
                    static_cast<std::size_t>(info.first);
                out.g[upper] += j_pair[pair];
            }
        }
    }
#endif
    return out;
}

EriDirectJKResult build_jk_direct_for_G_rys(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options) {
    // The compatibility and incremental endpoints use the same single-pass
    // RHF consumer as the public matrix-builder API.  The independent scalar
    // control remains available below for numerical and performance audits.
    return build_jk_direct_impl<RysDirectJKFusedEmit>(plan, density, options);
}

namespace jk_bridge {

EriDirectJKResult build_direct_scalar(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options) {
    return build_jk_direct_impl<RysDirectJKEmit>(plan, density, options);
}

#ifdef TLNO_RYS_DIRECT_EXCHANGE
EriDirectJKResult build_direct_fused(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options) {
    // One policy instantiation means one mathematical ERI traversal.  ss|ss
    // closes J and -K/2 at its producer terminal; all other classes use the
    // exact combined scalar fallback selected by the policy traits above.
    return build_jk_direct_impl<RysDirectJKFusedEmit>(
        plan, density, options);
}

EriDirectJKResult build_exchange_k(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options) {
    // Exactly one integral traversal.  Specialized s/p and d/p terminals
    // close positive K before emission; residual classes execute the four
    // exchange contractions through the lhs-hoisted projected-block receiver.
    return build_jk_direct_impl<RysDirectKExchangeEmit>(
        plan, density, options);
}
#endif

#if !defined(TLNO_BUILD_RYS_VALIDATION) || \
    defined(TLNO_VALIDATE_RYS_SPHERICAL_PACKED)
std::vector<double> build_coulomb_j(
    const EriPackedPlan& plan,
    const std::vector<double>& density,
    const EriPackedBuildOptions& options) {
    return build_jk_direct_impl<RysDirectJCoulombEmit>(
        plan, density, options).g;
}
#endif

}  // namespace jk_bridge

EriDirectJKIncrementalState::EriDirectJKIncrementalState(
    const EriPackedPlan& plan,
    const EriDirectJKIncrementalOptions& options)
    : plan_(plan), options_(options) {
    const auto bd = packed_plan_basis_data(plan_);
    if (!bd) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: null incremental direct J/K plan.");
    }
    if (plan_.profile.M != bd->spherical_function_count) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental direct J/K plan/profile mismatch.");
    }
    M_ = bd->spherical_function_count;

    if (options_.exact_rebuild_interval == 0u) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental exact_rebuild_interval must be "
            ">= 1.");
    }
    const EriPackedBuildOptions& base = options_.base_options;
    if (!tlno::core::math::is_finite_binary64(base.schwarz_threshold) ||
        base.schwarz_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental schwarz_threshold must be finite "
            "and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(
            base.primitive_fastschwarz_threshold) ||
        base.primitive_fastschwarz_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental primitive_fastschwarz_threshold "
            "must be finite and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(base.decay_threshold) ||
        base.decay_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental decay_threshold must be finite "
            "and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(base.density_threshold) ||
        base.density_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental base density_threshold must be "
            "finite and >= 0.");
    }
    if (base.schwarz_threshold != 0.0 &&
        base.decay_threshold != 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental base options cannot combine "
            "Schwarz and decay screening.");
    }
    if (!tlno::core::math::is_finite_binary64(
            options_.delta_density_threshold) ||
        options_.delta_density_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental delta_density_threshold must be "
            "finite and >= 0.");
    }
    if (!tlno::core::math::is_finite_binary64(
            options_.first_delta_density_threshold) ||
        options_.first_delta_density_threshold < 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental "
            "first_delta_density_threshold must be finite and >= 0.");
    }
    if ((options_.delta_density_threshold > 0.0 ||
         (options_.use_first_delta_density_threshold &&
          options_.first_delta_density_threshold > 0.0)) &&
        base.decay_threshold > 0.0) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental delta-density screening cannot "
            "be combined with the heuristic decay metric.");
    }

    // A full-density threshold would make a purported exact refresh depend on
    // P.  Keep one canonical effective base configuration and inject the
    // delta threshold only into intervening dP builds.
    options_.base_options.density_threshold = 0.0;
}

EriDirectJKUpdateStatus EriDirectJKIncrementalState::update(
    const std::vector<double>& density,
    bool force_exact_rebuild) {
    const std::size_t m = static_cast<std::size_t>(M_);
    const std::size_t matrix_size = m * m;
    if (density.size() != matrix_size) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental density must be spherical M x M "
            "row-major.");
    }

    std::vector<double> delta;
    if (initialized_) {
        delta.resize(matrix_size);
    }
    double max_abs_delta = 0.0;
    bool exactly_unchanged = initialized_;
    for (std::size_t i = 0; i < matrix_size; ++i) {
        const double value = density[i];
        if (!tlno::core::math::is_finite_binary64(value)) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: incremental density contains a "
                "non-finite value.");
        }
        const double d = initialized_ ? value - density_[i] : value;
        if (!tlno::core::math::is_finite_binary64(d)) {
            throw std::runtime_error(
                "eri_ao_tensor_hgp: incremental density delta overflowed.");
        }
        if (initialized_) {
            delta[i] = d;
            exactly_unchanged = exactly_unchanged && (d == 0.0);
        }
        max_abs_delta = std::max(max_abs_delta, std::abs(d));
    }

    // An explicit force is stronger than the zero-delta fast path: callers
    // can use it to discard accumulated screening/roundoff error even when P
    // itself did not move.  Ordinary unchanged calls do not advance cadence.
    if (exactly_unchanged && !force_exact_rebuild) {
        ++counters_.updates;
        ++counters_.skipped_updates;
        last_status_ = EriDirectJKUpdateStatus::skipped_unchanged;
        last_max_abs_density_delta_ = 0.0;
        last_effective_density_threshold_ = 0.0;
        return last_status_;
    }

    const bool periodic_exact =
        initialized_ &&
        updates_since_exact_rebuild_ >=
            options_.exact_rebuild_interval - 1u;
    const bool exact =
        !initialized_ || force_exact_rebuild || periodic_exact;

    // Copy before the expensive build so allocation failure cannot leave G
    // paired with the wrong cached density.
    std::vector<double> next_density(density);
    EriPackedBuildOptions build_options = options_.base_options;
    double effective_density_threshold = 0.0;
    if (!exact) {
        const bool first_incremental_after_exact =
            updates_since_exact_rebuild_ == 0u;
        effective_density_threshold =
            options_.use_first_delta_density_threshold &&
                    first_incremental_after_exact
                ? options_.first_delta_density_threshold
                : options_.delta_density_threshold;
        build_options.density_threshold = effective_density_threshold;
    }
    EriDirectJKResult built = build_jk_direct_for_G_rys(
        plan_, exact ? density : delta, build_options);
    if (built.M != M_ || built.g.size() != matrix_size) {
        throw std::runtime_error(
            "eri_ao_tensor_hgp: incremental direct J/K builder returned an "
            "inconsistent matrix shape.");
    }

    std::vector<double> next_g;
    if (exact) {
        next_g = std::move(built.g);
        for (const double value : next_g) {
            if (!tlno::core::math::is_finite_binary64(value)) {
                throw std::runtime_error(
                    "eri_ao_tensor_hgp: exact direct J/K rebuild produced a "
                    "non-finite value.");
            }
        }
    } else {
        next_g = g_;
        for (std::size_t i = 0; i < matrix_size; ++i) {
            if (!tlno::core::math::is_finite_binary64(built.g[i])) {
                throw std::runtime_error(
                    "eri_ao_tensor_hgp: incremental dG build produced a "
                    "non-finite value.");
            }
            next_g[i] += built.g[i];
            if (!tlno::core::math::is_finite_binary64(next_g[i])) {
                throw std::runtime_error(
                    "eri_ao_tensor_hgp: incremental cached G overflowed.");
            }
        }
    }

    density_.swap(next_density);
    g_.swap(next_g);
    initialized_ = true;
    ++counters_.updates;
    last_max_abs_density_delta_ = max_abs_delta;
    last_effective_density_threshold_ = effective_density_threshold;
    if (exact) {
        ++counters_.exact_rebuilds;
        updates_since_exact_rebuild_ = 0;
        last_status_ = EriDirectJKUpdateStatus::exact_rebuild;
    } else {
        ++counters_.incremental_updates;
        ++updates_since_exact_rebuild_;
        last_status_ = EriDirectJKUpdateStatus::incremental_update;
    }
    return last_status_;
}

void EriDirectJKIncrementalState::reset() noexcept {
    density_.clear();
    g_.clear();
    counters_ = {};
    last_status_ = EriDirectJKUpdateStatus::uninitialized;
    last_max_abs_density_delta_ = 0.0;
    last_effective_density_threshold_ = 0.0;
    updates_since_exact_rebuild_ = 0;
    initialized_ = false;
}
#endif
