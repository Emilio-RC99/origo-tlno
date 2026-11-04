#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_PARALLEL_J_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_PARALLEL_J_IPP

// Direct Coulomb J only. Shared immutable geometry, density and projectors;
// private receivers and scratch. No ERI tensor or molecular quartet list.
#ifndef TLNO_NATIVE_J_PAIR_ONLY_STORAGE
#define TLNO_NATIVE_J_PAIR_ONLY_STORAGE 1
#endif
#if TLNO_NATIVE_J_PAIR_ONLY_STORAGE != 0 && TLNO_NATIVE_J_PAIR_ONLY_STORAGE != 1
#error "TLNO_NATIVE_J_PAIR_ONLY_STORAGE must be 0 or 1"
#endif
inline void accumulate_native_parallel_j(
    const BasisDataHGP& bd, const NativeQuartetSchedule& schedule,
    const std::vector<double>* metric, double threshold, double primitive_squared,
    const NativeSecondaryShellScreen* secondary) {
#ifdef _OPENMP
    const int requested = omp_get_max_threads();
#else
    const int requested = 1;
#endif
    const auto* parent = rys_jk_context_slot();
    if (!parent) throw std::runtime_error("native parallel J requires a receiver");
    const std::size_t matrix_size = parent->M * parent->M;
    const std::size_t pairs = parent->M * (parent->M + 1) / 2;
    const auto* density = g_rys_density_screen;
    const auto* compact_pairs = g_native_j_compact_pairs;
    const auto* packet_screen = g_native_shell_packet_screen;
    // All dispatched producers below use RysDirectJCoulombEmit. With the
    // canonical pair accumulator enabled, its receiver writes only j_pair;
    // matrix-space storage is needed only for builds using the other policy.
    bool pair_only = false;
#if defined(TLNO_RYS_J_PAIR_SPACE_ACCUMULATION) && TLNO_NATIVE_J_PAIR_ONLY_STORAGE
    pair_only = parent->j_pair != nullptr;
#endif
    std::vector<std::vector<double>> matrices(requested), pair_values(requested);
    std::vector<std::exception_ptr> errors(requested);
#ifdef TLNO_JK_COVERAGE_CHECKSUM
    std::vector<double> coverage(requested, 0.);
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    std::vector<long double> moment1(requested, 0.), moment2(requested, 0.), moment3(requested, 0.);
    std::vector<unsigned long long> counts(requested, 0);
#endif
    int actual = 1;
#ifdef _OPENMP
#pragma omp parallel num_threads(requested)
#endif
    {
#ifdef _OPENMP
        const int rank = omp_get_thread_num(), count = omp_get_num_threads();
#pragma omp single
        actual = count;
#else
        const int rank = 0, count = 1;
#endif
        try {
            if (!pair_only) matrices[rank].assign(matrix_size, 0.);
            if (parent->j_pair) pair_values[rank].assign(pairs, 0.);
            auto context = *parent;
            context.g = pair_only ? nullptr : matrices[rank].data();
            context.j_pair = pair_values[rank].empty() ? nullptr : pair_values[rank].data();
#ifdef TLNO_JK_COVERAGE_CHECKSUM
            context.coverage_checksum = &coverage[rank];
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
            context.indexed_moment_1 = &moment1[rank];
            context.indexed_moment_2 = &moment2[rank];
            context.indexed_moment_3 = &moment3[rank];
            context.canonical_count = &counts[rank];
#endif
            const NativeRowPartitionGuard partition(rank, count, TLNO_NATIVE_J_WEIGHTED_ROWS != 0);
            const RysJkEmitContextGuard receiver(&context);
            const RysDensityScreenGuard density_guard(density);
            const NativeJCompactPairGuard compact_guard(compact_pairs);
            const NativeShellPacketScreenGuard packet_screen_guard(packet_screen);
            const NativeSecondaryShellScreenGuard secondary_guard(secondary);
            EriStreamAccumulator accumulator;
            const EriStreamModeGuard stream(true, &bd.packed_spherical_pair_components, &accumulator);
            std::vector<double> unused;
            for (std::size_t i = 0; i < schedule.bins.size(); ++i) for (std::size_t j = 0; j <= i; ++j) {
                const auto& lhs = schedule.bins[i]; const auto& rhs = schedule.bins[j];
                const auto& shape = schedule.shapes.at(native_shape_key(lhs.la, lhs.lb, rhs.la, rhs.lb));
#if TLNO_NATIVE_SPECIALIZED
                if (dispatch_native_specialized<tlno::core::simd::lanes, RysDirectJCoulombEmit>(
                    bd, lhs, rhs, shape, i == j, metric, threshold, primitive_squared, unused)) continue;
#endif
                native_quartet_bin_pair<tlno::core::simd::lanes, RysDirectJCoulombEmit>(
                    bd, lhs, rhs, shape, i == j, metric, threshold, primitive_squared, unused);
            }
        } catch (...) {
            // C++ exceptions cannot cross an OpenMP region. Restore all scoped
            // contexts, join the team, then report the first failing partition.
            errors[rank] = std::current_exception();
        }
    }
    for (int rank = 0; rank < actual; ++rank) if (errors[rank]) std::rethrow_exception(errors[rank]);
    // Fixed rank order per matrix element gives repeatable sums at a fixed
    // thread count. Different thread counts are checked numerically against 1T.
    for (int rank = 0; rank < actual; ++rank) {
        if (!pair_only) for (std::size_t i = 0; i < matrix_size; ++i) parent->g[i] += matrices[rank][i];
        if (parent->j_pair) for (std::size_t i = 0; i < pairs; ++i) parent->j_pair[i] += pair_values[rank][i];
#ifdef TLNO_JK_COVERAGE_CHECKSUM
        if (parent->coverage_checksum) * parent->coverage_checksum += coverage[rank];
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
        if (parent->indexed_moment_1) {
            *parent->indexed_moment_1 += moment1[rank];
            *parent->indexed_moment_2 += moment2[rank];
            *parent->indexed_moment_3 += moment3[rank];
            *parent->canonical_count += counts[rank];
        }
#endif
    }
}
#endif
