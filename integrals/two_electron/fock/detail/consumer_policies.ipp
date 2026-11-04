#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_CONSUMER_POLICIES_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_CONSUMER_POLICIES_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

// ---------------------------------------------------------------------------
// Direct J/K policy: digest each integral into G without materializing the tensor.
//
// WHAT: RysDirectJKEmit receives each value through the same Emit seam that
// RysPackedTensorEmit uses and contracts it at once against the density P.
// The O(M^4) tensor is never allocated or written: the pure-spherical sweep
// already runs in stream mode, so changing the policy is enough to turn it
// into a direct J/K build with O(M^2) memory.
//
// WHY at COMPILE time: a run-time choice would have to test the destination
// ONCE PER INTEGRAL. Here the policy type selects that path without a branch
// and keeps the exact scalar policy.
//
// The context (pointers to G, to P and to the inverse tables) DOES travel
// through a file-scope pointer. The scalar Emit seam carries (lhs, rhs, val),
// and the context is protected by RAII.
// The five values the digestion needs do not enlarge
// PackedPairComponentH, because every stream kernel reads that record. The
// J/K metadata live in a compact, immutable SIDE array, indexed by pair_index
// and touched only by the J/K path. It replaces five reads from five
// different arrays (five cache lines) with a single
// contiguous read.
// WHAT: use the neutral J/K records at the shared producer boundary.
// WHY: the ERI stream does not own density rows, G destinations or packet
// consumers; keeping those definitions here made every J/K change an HGP
// architecture change.
// PURPOSE: new consumers compile independently while the scalar migration
// control retains the exact same field layout and arithmetic.
using RysJkPairInfo = tlno::integrals::twoe::fock::detail::PairInfo;
using RysJkEmitContext = tlno::integrals::twoe::fock::detail::BuildContext;

// Do NOT reintroduce a bra-side memo indexed by lhs_pair. It was tried on
// 2026-08-02 and measured 0.898x (gly/def2svp) and 0.990x (isoleucine/def2svp),
// that is, a REGRESSION: the pair tables are small and live in L1, so looking
// them up is cheaper than the memo's thread_local access plus its branch.
//
// The diagnosis that motivated it was flawed: the TLNO_JK_DIGEST_PROBE=1 probe
// used a `volatile` so the compiler would not remove the lookups, and that
// forces one store per integral that the real code does not do. That store
// inflated the apparent cost of the lookups. Any future breakdown of the
// digestion cost needs a probe that adds no writes.
#ifdef TLNO_JK_DIGEST_PROBE
// Probe sink: it is observed only once, after the stream, so the compiler
// cannot remove the arithmetic, yet no observable store is paid per
// integral.
inline double g_jk_probe_sink = 0.0;
#endif

inline const RysJkEmitContext * & rys_jk_context_slot() noexcept {
    static thread_local const RysJkEmitContext* context = nullptr;
    return context;
}

// RAII guard: it restores the previous value instead of setting nullptr, so
// nesting (e.g. a validation inside a sweep) does not break the outer one.
class RysJkEmitContextGuard {
public:
    explicit RysJkEmitContextGuard(const RysJkEmitContext* context) noexcept
        : previous_(rys_jk_context_slot()) {
        rys_jk_context_slot() = context;
    }
    ~RysJkEmitContextGuard() noexcept { rys_jk_context_slot() = previous_; }
    RysJkEmitContextGuard(const RysJkEmitContextGuard&) = delete;
    RysJkEmitContextGuard& operator = (const RysJkEmitContextGuard&) = delete;

private:
    const RysJkEmitContext* previous_;
};

// Symmetric digestion of one unique value of the s8 tensor into J and K.
//
// Goal: the RHF G = J - 0.5*K, NOT J and K separately.
//
// The value arrives once per unique quartet (ab >= cd, a >= b, c >= d). The
// permutation multiplicity is absorbed into u through halves: 0.5 if a == b,
// 0.5 if c == d and 0.5 if lhs == rhs -- which is what
// pair_symmetry_scale encodes. Six writes remain, and symmetrize_g_matrix
// completes the triangle average at the end of the sweep; without that final
// call G is only half built and the result is wrong.
inline void accumulate_jk_symmetric_pair_fast(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    double* __restrict g = context.g;
    const double* __restrict p = context.p;
    const RysJkPairInfo& li = context.pair_info[lhs_pair];
    const RysJkPairInfo& ri = context.pair_info[rhs_pair];
    const int b = static_cast<int>(li.second);
    const int c = static_cast<int>(ri.first);
    const int d = static_cast<int>(ri.second);
    double u = value * li.scale * ri.scale;
    if (lhs_pair == rhs_pair) u *= 0.5;

    const std::size_t ar = li.first_row;
    const std::size_t br = li.second_row;
    const std::size_t cr = ri.first_row;
    // These rows may coincide (a == b), and the central G write can alias a
    // row write for repeated quartet indices.  They are all derived from the
    // single restricted base above; giving the derivatives independent
    // restrict contracts would be undefined for exactly those valid cases.
    double* ga = g + ar;
    double* gb = g + br;
    const double* pa = p + ar;
    const double* pb = p + br;
    const double u4 = 4.0 * u;
#ifdef TLNO_JK_DIGEST_PROBE
    // CORRECTED probe. The previous one used a `volatile` that forced a store
    // per integral -- work the real code does not do -- and inflated the
    // apparent cost of the lookups. This one does EVERYTHING except writing G:
    // the same lookups, the same reads of P, the same arithmetic, accumulated
    // into a file-scope double consumed outside the loop. The difference from
    // probe=0 is then exactly the cost of the six scattered writes.
    {
        static const int probe = []() {
            const char* v = std::getenv("TLNO_JK_DIGEST_PROBE");
            return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 0;
        }();
        if (probe == 1) {
            const std::size_t cd_idx = cr + static_cast<std::size_t>(d);
            g_jk_probe_sink +=
                u4 * p[cd_idx] + u4 * pa[b] -
                u * pb[d] - u * pb[c] - u * pa[d] - u * pa[c];
            return;
        }
    }
#endif
    // [OPT] Precompute the repeated cd offset to save address arithmetic.
    // Reading p[cd_idx] and pa[b] into local consts forces the compiler
    // to schedule the loads earlier, avoiding pointer aliasing pessimism.
    // This explicit unrolling helps hold and outperform relative accelerations
    // against Simint and Libcint J/K building paths.
    const std::size_t cd_idx = cr + static_cast<std::size_t>(d);
    const double p_cd = p[cd_idx];
    const double pa_b = pa[b];

    ga[b] += u4 * p_cd;
    g[cd_idx] += u4 * pa_b;

    // Explicitly cache exchange lookups
    const double upb_d = u * pb[d];
    const double upb_c = u * pb[c];
    const double upa_d = u * pa[d];
    const double upa_c = u * pa[c];

    ga[c] -= upb_d;
    ga[d] -= upb_c;
    gb[c] -= upa_d;
    gb[d] -= upa_c;
}

#ifdef TLNO_RYS_DIRECT_EXCHANGE
// Exchange-only half of the canonical s8 contraction used by the fused
// RHF consumer.  The ss|ss terminal already owns the two Coulomb terms in
// pair space, so calling the complete scalar reducer here would count J twice.
// Keep the four writes separate: valid repeated-index quartets can alias their
// destinations, and their sequential additions are part of the exact contract.
__attribute__((always_inline)) inline void
accumulate_exchange_half_symmetric_pair_fast(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    double* g = context.g;
    const double* p = context.p;
    const RysJkPairInfo& li = context.pair_info[lhs_pair];
    const RysJkPairInfo& ri = context.pair_info[rhs_pair];
    const int c = static_cast<int>(ri.first);
    const int d = static_cast<int>(ri.second);
    double u = value * static_cast<double>(li.scale) *
               static_cast<double>(ri.scale);
    if (lhs_pair == rhs_pair) u *= 0.5;

    double* const ga = g + li.first_row;
    double* const gb = g + li.second_row;
    const double* const pa = p + li.first_row;
    const double* const pb = p + li.second_row;
    ga[c] -= u * pb[d];
    ga[d] -= u * pb[c];
    gb[c] -= u * pa[d];
    gb[d] -= u * pa[c];
}

// Positive K-only contraction for the isolated exchange endpoint.  The
// canonical s8 reducer above contributes -K/2 to G with four writes of -u;
// therefore the exact positive exchange matrix uses the same destinations
// with +2u.  Final triangular symmetrization remains shared with every direct
// endpoint.  Keep the writes sequential because repeated AO indices may alias.
__attribute__((always_inline)) inline void
accumulate_k_symmetric_pair_fast(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    double* k = context.g;
    const double* p = context.p;
    const RysJkPairInfo& li = context.pair_info[lhs_pair];
    const RysJkPairInfo& ri = context.pair_info[rhs_pair];
    const int c = static_cast<int>(ri.first);
    const int d = static_cast<int>(ri.second);
    double u2 = 2.0 * value * static_cast<double>(li.scale) *
                static_cast<double>(ri.scale);
    if (lhs_pair == rhs_pair) u2 *= 0.5;

    double* const ka = k + li.first_row;
    double* const kb = k + li.second_row;
    const double* const pa = p + li.first_row;
    const double* const pb = p + li.second_row;
    ka[c] += u2 * pb[d];
    ka[d] += u2 * pb[c];
    kb[c] += u2 * pa[d];
    kb[d] += u2 * pa[c];
}
#endif

// Digestion of one value with the context ALREADY resolved. It exists so the
// block route can look the context up once without duplicating the gates.
// The bra side is extracted ONCE per block row.
//
// WHY. `accumulate_jk_symmetric_pair_fast` receives pair indices and from them
// re-derives li, b, ar, br, ga, gb, pa, pb and li.scale. Inside the double
// loop of `rys_emit_block` the lhs is constant over the whole inner loop, so
// all of that is loop-invariant. Extracting it explicitly keeps that
// property even if the compiler's inlining or aliasing decisions
// change.
struct RysJkLhsView {
    double* ga = nullptr;        // g + a*M; may equal gb
    double* gb = nullptr;        // g + b*M
    const double* pa = nullptr;  // p + a*M; may equal pb
    const double* pb = nullptr;  // p + b*M
    // WHAT: retain the canonical lhs pair after hoisting its row metadata.
    // WHY/PURPOSE: equality scaling and coverage/fingerprint recording still
    // need the exact pair id without looking up pair_info again per ERI.
    std::uint32_t pair = 0;
    int b = 0;
    double scale = 1.0;
};

inline RysJkLhsView rys_jk_lhs_view(const RysJkEmitContext& context,
                                    std::uint32_t lhs_pair) noexcept {
    const RysJkPairInfo& li = context.pair_info[lhs_pair];
    RysJkLhsView v;
    v.ga = context.g + li.first_row;
    v.gb = context.g + li.second_row;
    v.pa = context.p + li.first_row;
    v.pb = context.p + li.second_row;
    v.pair = lhs_pair;
    v.b = static_cast<int>(li.second);
    v.scale = li.scale;
    return v;
}

// Digestion with the bra side already extracted. Same body as
// accumulate_jk_symmetric_pair_fast, except that it does not read pair_info[lhs] again.
inline void
accumulate_jk_symmetric_pair_lhs(
    const RysJkEmitContext& context,
    const RysJkLhsView& lv,
    std::uint32_t rhs_pair,
    double value) noexcept {
    // lv already carries pointers derived from context.g/context.p outside
    // this function.  A fresh restrict association here would conflict with
    // those aliases, so this hoisted path deliberately uses ordinary pointers.
    double* g = context.g;
    const double* p = context.p;
    const RysJkPairInfo& ri = context.pair_info[rhs_pair];
    const int b = lv.b;
    const int c = static_cast<int>(ri.first);
    const int d = static_cast<int>(ri.second);
    double u = value * lv.scale * ri.scale;
    if (lv.pair == rhs_pair) u *= 0.5;

    const std::size_t cr = ri.first_row;
    double* ga = lv.ga;
    double* gb = lv.gb;
    const double* pa = lv.pa;
    const double* pb = lv.pb;
    const double u4 = 4.0 * u;

#ifdef TLNO_JK_DIGEST_PROBE
    // Keep the block-digestion probe identical to the scalar digestion probe:
    // execute all lookups, density loads, and arithmetic, but suppress the six
    // scattered G writes when the runtime probe is enabled.
    {
        static const int probe = []() {
            const char* v = std::getenv("TLNO_JK_DIGEST_PROBE");
            return (v != nullptr && v[0] != '\0') ? std::atoi(v) : 0;
        }();
        if (probe == 1) {
            const std::size_t cd_idx = cr + static_cast<std::size_t>(d);
            g_jk_probe_sink +=
                u4 * p[cd_idx] + u4 * pa[b] -
                u * pb[d] - u * pb[c] - u * pa[d] - u * pa[c];
            return;
        }
    }
#endif
    // [OPT] Precompute the repeated cd offset to save address arithmetic.
    // Reading p[cd_idx] and pa[b] into local consts forces the compiler
    // to schedule the loads earlier, avoiding pointer aliasing pessimism.
    // This explicit unrolling helps hold and outperform relative accelerations
    // against Simint and Libcint J/K building paths.
    const std::size_t cd_idx = cr + static_cast<std::size_t>(d);
    const double p_cd = p[cd_idx];
    const double pa_b = pa[b];

    ga[b] += u4 * p_cd;
    g[cd_idx] += u4 * pa_b;

    // Explicitly cache exchange lookups
    const double upb_d = u * pb[d];
    const double upb_c = u * pb[c];
    const double upa_d = u * pa[d];
    const double upa_c = u * pa[c];

    ga[c] -= upb_d;
    ga[d] -= upb_c;
    gb[c] -= upa_d;
    gb[d] -= upa_c;
}

// Gates shared by the scalar and lhs-hoisted digestion paths. Keeping them
// separate from the G update lets both paths expose every canonical
// (pair index, value) to the coverage and fingerprint audits.
inline void rys_jk_record_one(const RysJkEmitContext& context,
                              std::uint32_t lhs_pair,
                              std::uint32_t rhs_pair,
                              double val) noexcept {
#ifdef TLNO_JK_COVERAGE_CHECKSUM
    if (context.coverage_checksum) * context.coverage_checksum += val;
#endif
#ifdef TLNO_VALIDATE_RYS_STREAM_FINGERPRINT
    const std::uint64_t idx = static_cast<std::uint64_t>(
        EriPackedGAccessor::quartet_index(lhs_pair, rhs_pair));
    if (context.indexed_moment_1) {
        const long double x = static_cast<long double>(val);
        *context.indexed_moment_1 +=
            x * eri_fp_weight(idx, 0x243f6a8885a308d3ull);
        *context.indexed_moment_2 +=
            x * eri_fp_weight(idx, 0x13198a2e03707344ull);
        *context.indexed_moment_3 +=
            x * eri_fp_weight(idx, 0xa4093822299f31d0ull);
        ++*context.canonical_count;
    }
#else
    (void)context;
    (void)lhs_pair;
    (void)rhs_pair;
    (void)val;
#endif
}

inline void
rys_jk_digest_one_lhs(const RysJkEmitContext& context,
                      const RysJkLhsView& lv,
                      std::uint32_t rhs_pair,
                      double val) noexcept {
    rys_jk_record_one(context, lv.pair, rhs_pair, val);
    accumulate_jk_symmetric_pair_lhs(context, lv, rhs_pair, val);
}

// Generic projected d/f/g blocks reuse the lhs density and destination rows
// across the complete rhs row, just as the combined consumer does.  K has
// four crossed updates; no Coulomb density loads or pair-space scratch.
inline void rys_k_digest_one_lhs(const RysJkEmitContext& context,
                                 const RysJkLhsView& lhs,
                                 std::uint32_t rhs_pair,
                                 double value) noexcept {
    rys_jk_record_one(context, lhs.pair, rhs_pair, value);
    const auto& rhs = context.pair_info[rhs_pair];
    double u = 2.0 * value * lhs.scale * rhs.scale;
    if (lhs.pair == rhs_pair) u *= 0.5;
    const int c = rhs.first;
    const int d = rhs.second;
    // Repeated AO indices can alias: these updates must remain sequential.
    lhs.ga[c] += u * lhs.pb[d];
    lhs.ga[d] += u * lhs.pb[c];
    lhs.gb[c] += u * lhs.pa[d];
    lhs.gb[d] += u * lhs.pa[c];
}

inline void rys_jk_digest_one(const RysJkEmitContext& context,
                              std::uint32_t lhs_pair,
                              std::uint32_t rhs_pair,
                              double val) noexcept {
    rys_jk_record_one(context, lhs_pair, rhs_pair, val);
    accumulate_jk_symmetric_pair_fast(context, lhs_pair, rhs_pair, val);
}

struct RysDirectJKEmit {
    static inline void emit(std::vector<double>& /*vals*/,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double val) noexcept {
        rys_jk_digest_one(*rys_jk_context_slot(), lhs_pair, rhs_pair, val);
    }
};

// WHAT: keep the experimental J-only accounting behind a compile-time gate.
// WHY: the speed binary must not pay counters, atomics, or formatting in its
// hot path, while an untimed diagnostic build still has to prove that the
// candidate consumes every canonical ERI and exactly two logical J terms.
// PURPOSE: timing and structural certification remain separate experiments;
// never enable this macro in the benchmark binary.
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
struct RysDirectJSpeedProbeAudit {
    std::uint64_t total_canonical_eris = 0;
    std::uint64_t candidate_ssss_eris = 0;
    std::uint64_t candidate_ssps_eris = 0;
    std::uint64_t candidate_sp_block_eris = 0;
    std::uint64_t candidate_ssds_eris = 0;
    std::uint64_t scalar_other_eris = 0;
    std::uint64_t logical_j_terms_committed = 0;
    std::uint64_t candidate_physical_commits = 0;
    std::uint64_t actual_index_xor = 0;
    std::uint64_t actual_index_sum = 0;
    std::uint64_t reference_canonical_eris = 0;
    std::uint64_t reference_index_xor = 0;
    std::uint64_t reference_index_sum = 0;
};
thread_local RysDirectJSpeedProbeAudit g_rys_direct_j_speed_probe_audit;

inline std::uint64_t rys_j_speed_probe_index_hash(
    std::uint32_t lhs_pair, std::uint32_t rhs_pair) noexcept {
    std::uint64_t value =
        (static_cast<std::uint64_t>(lhs_pair) << 32u) |
        static_cast<std::uint64_t>(rhs_pair);
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27u)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31u);
}

inline void rys_j_speed_probe_record_actual_index(
    std::uint32_t lhs_pair, std::uint32_t rhs_pair) noexcept {
    const std::uint64_t hash =
        rys_j_speed_probe_index_hash(lhs_pair, rhs_pair);
    g_rys_direct_j_speed_probe_audit.actual_index_xor ^= hash;
    g_rys_direct_j_speed_probe_audit.actual_index_sum += hash;
}
#endif

inline void rys_j_speed_probe_record_scalar(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    rys_jk_record_one(context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.total_canonical_eris;
    ++g_rys_direct_j_speed_probe_audit.scalar_other_eris;
    rys_j_speed_probe_record_actual_index(lhs_pair, rhs_pair);
#endif
}

inline void rys_j_speed_probe_record_candidate_ssps(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    rys_jk_record_one(context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.total_canonical_eris;
    ++g_rys_direct_j_speed_probe_audit.candidate_ssps_eris;
    rys_j_speed_probe_record_actual_index(lhs_pair, rhs_pair);
#endif
}

inline void rys_j_speed_probe_record_candidate_ssss(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    rys_jk_record_one(context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.total_canonical_eris;
    ++g_rys_direct_j_speed_probe_audit.candidate_ssss_eris;
    rys_j_speed_probe_record_actual_index(lhs_pair, rhs_pair);
#endif
}

inline void rys_j_speed_probe_record_candidate_ssds(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    rys_jk_record_one(context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.total_canonical_eris;
    ++g_rys_direct_j_speed_probe_audit.candidate_ssds_eris;
    rys_j_speed_probe_record_actual_index(lhs_pair, rhs_pair);
#endif
}

inline void rys_j_speed_probe_record_candidate_sp_block(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
    rys_jk_record_one(context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    ++g_rys_direct_j_speed_probe_audit.total_canonical_eris;
    ++g_rys_direct_j_speed_probe_audit.candidate_sp_block_eris;
    rys_j_speed_probe_record_actual_index(lhs_pair, rhs_pair);
#endif
}

// WHAT: contract one canonical ERI into J only: the (ab) and (cd) Coulomb
// destinations are the two, and only two, logical terms owned by that ERI.
// WHY: reusing the old combined J/K reducer would still execute four exchange
// density loads and four exchange RMWs, invalidating a J-only speed result.
// PURPOSE: every class outside the producer-owned J slices remains a
// complete, honest J workload while those slices are isolated.
inline void accumulate_j_symmetric_pair_fast(
    const RysJkEmitContext& context,
    std::uint32_t lhs_pair,
    std::uint32_t rhs_pair,
    double value) noexcept {
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
    // A canonical ERI (p|q) contributes rho[q]*(p|q) to J[p] and, for
    // distinct pair indices, rho[p]*(p|q) to J[q], where rho already contains
    // the off-diagonal density multiplicity.  Pair-space accumulation removes
    // both PairInfo lookups and both AO-matrix address calculations from the
    // overwhelmingly dominant scalar receiver.
    double* const __restrict j_pair = context.j_pair;
    const double* const __restrict rho = context.j_pair_density;
    if (lhs_pair == rhs_pair) {
        j_pair[lhs_pair] += value * rho[lhs_pair];
    } else {
        j_pair[lhs_pair] += value * rho[rhs_pair];
        j_pair[rhs_pair] += value * rho[lhs_pair];
    }
#else
    const RysJkPairInfo& li = context.pair_info[lhs_pair];
    const RysJkPairInfo& ri = context.pair_info[rhs_pair];
    const std::size_t lhs_index =
        li.first_row + static_cast<std::size_t>(li.second);
    const std::size_t rhs_index =
        ri.first_row + static_cast<std::size_t>(ri.second);
    double u4 = 4.0 * value * static_cast<double>(li.scale) *
                static_cast<double>(ri.scale);
    if (lhs_pair == rhs_pair) u4 *= 0.5;
    context.g[lhs_index] += u4 * context.p[rhs_index];
    context.g[rhs_index] += u4 * context.p[lhs_index];
#endif
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
}

// MEASURED AND DISCARDED (2026-08-15): materializing the rhs side once per
// block in a thread_local vector -- so the inner loop would not chase
// pair_info[rhs] for every (lhs, rhs) pair -- worsened j_over_eri by +5% to +8%
// in the six cases tested. In small blocks, which are exactly the s/p ones
// where J loses, the clear/reserve/push_back and the thread_local access cost
// more than the dependent loads they save. Do not reintroduce without measuring.

struct RysJLhsView {
    std::uint32_t pair = 0;
    std::size_t matrix_index = 0;
    double density = 0.0;
    double scale = 1.0;
};

inline RysJLhsView rys_j_lhs_view(const RysJkEmitContext& context,
                                  std::uint32_t lhs_pair) noexcept {
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
    return {lhs_pair, static_cast<std::size_t>(lhs_pair),
            context.j_pair_density[lhs_pair], 1.0};
#else
    const RysJkPairInfo& li = context.pair_info[lhs_pair];
    const std::size_t index =
        li.first_row + static_cast<std::size_t>(li.second);
    return {lhs_pair, index, context.p[index],
            static_cast<double>(li.scale)};
#endif
}

// WHAT: reuse one resolved lhs pair for every rhs component in an emitted
// block, but execute only the two Coulomb updates.
// WHY: generic d/f projection already materializes its block, so its available
// gain is removing repeated pair-table and density lookups, not pretending the
// block itself can be removed at this receiver boundary.
// PURPOSE: all residual classes feed the J-only probe without inheriting the
// six-update combined G reducer.
inline void rys_j_digest_one_lhs(
    const RysJkEmitContext& context,
    const RysJLhsView& lhs,
    std::uint32_t rhs_pair,
    double value) noexcept {
    rys_j_speed_probe_record_scalar(
        context, lhs.pair, rhs_pair, value);
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
    double* const __restrict j_pair = context.j_pair;
    const double* const __restrict rho = context.j_pair_density;
    if (lhs.pair == rhs_pair) {
        j_pair[lhs.pair] += value * lhs.density;
    } else {
        j_pair[lhs.pair] += value * rho[rhs_pair];
        j_pair[rhs_pair] += value * lhs.density;
    }
#else
    const RysJkPairInfo& ri = context.pair_info[rhs_pair];
    const std::size_t rhs_index =
        ri.first_row + static_cast<std::size_t>(ri.second);
    double u4 = 4.0 * value * lhs.scale * static_cast<double>(ri.scale);
    if (lhs.pair == rhs_pair) u4 *= 0.5;
    context.g[lhs.matrix_index] += u4 * context.p[rhs_index];
    context.g[rhs_index] += u4 * lhs.density;
#endif
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    g_rys_direct_j_speed_probe_audit.logical_j_terms_committed += 2u;
#endif
}

inline void rys_j_digest_one(const RysJkEmitContext& context,
                             std::uint32_t lhs_pair,
                             std::uint32_t rhs_pair,
                             double value) noexcept {
    rys_j_speed_probe_record_scalar(
        context, lhs_pair, rhs_pair, value);
    accumulate_j_symmetric_pair_fast(
        context, lhs_pair, rhs_pair, value);
}

#ifdef TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST
// VeloxChem-like task-local J receiver.  The producer still emits the same
// canonical values in the same order; only their final pair-space destination
// remains private for the lifetime of one homogeneous low-L task.
class RysJVeloxTaskDigest {
public:
    static constexpr std::size_t kMaxPairs =
        static_cast<std::size_t>(TLNO_J_VELOX_TASK_MAX_PAIRS);
    static constexpr std::uint16_t kNoSlot =
        std::numeric_limits<std::uint16_t>::max();
    static_assert(TLNO_J_VELOX_TASK_MAX_PAIRS > 0,
                  "Velox task-local digest needs a positive pair cap");
    static_assert(TLNO_J_VELOX_TASK_MAX_PAIRS <
                      static_cast<int>(kNoSlot),
                  "Velox task-local pair cap must fit uint16_t");

    RysJVeloxTaskDigest(std::size_t pair_count,
                        double* global_j,
                        const double* global_rho)
        : slot_of_pair_(pair_count, kNoSlot),
          active_pairs_(kMaxPairs),
          rho_(kMaxPairs),
          j_(kMaxPairs),
          global_j_(global_j),
          global_rho_(global_rho) {}

    void begin() noexcept {
        // A previous task must have closed through its RAII scope.  Be robust
        // to early exits in experimental code without carrying stale output.
        finish();
        collecting_ = true;
        active_ = false;
        count_ = 0u;
    }

    bool register_pair(std::uint32_t pair) noexcept {
        if (!collecting_) return false;
        if (pair >= slot_of_pair_.size()) {
            abandon_setup();
            return false;
        }
        if (slot_of_pair_[pair] != kNoSlot) return true;
        if (count_ == kMaxPairs) {
            abandon_setup();
            return false;
        }
        const std::uint16_t slot = static_cast<std::uint16_t>(count_++);
        slot_of_pair_[pair] = slot;
        active_pairs_[slot] = pair;
        rho_[slot] = global_rho_[pair];
        j_[slot] = 0.0;
        return true;
    }

    void activate() noexcept {
        active_ = collecting_ && count_ != 0u;
        collecting_ = false;
    }

    bool active() const noexcept { return active_; }

    bool add(std::uint32_t lhs_pair,
             std::uint32_t rhs_pair,
             double value) noexcept {
        if (!active_ || lhs_pair >= slot_of_pair_.size() ||
            rhs_pair >= slot_of_pair_.size()) return false;
        const std::uint16_t lhs_slot = slot_of_pair_[lhs_pair];
        const std::uint16_t rhs_slot = slot_of_pair_[rhs_pair];
        if (lhs_slot == kNoSlot || rhs_slot == kNoSlot) return false;
        if (lhs_pair == rhs_pair) {
            j_[lhs_slot] += value * rho_[lhs_slot];
        } else {
            j_[lhs_slot] += value * rho_[rhs_slot];
            j_[rhs_slot] += value * rho_[lhs_slot];
        }
        return true;
    }

    void finish() noexcept {
        if (active_) {
            for (std::size_t slot = 0; slot < count_; ++slot) {
                global_j_[active_pairs_[slot]] += j_[slot];
            }
        }
        reset_map();
        collecting_ = false;
        active_ = false;
    }

private:
    void abandon_setup() noexcept {
        reset_map();
        collecting_ = false;
        active_ = false;
    }

    void reset_map() noexcept {
        for (std::size_t slot = 0; slot < count_; ++slot) {
            slot_of_pair_[active_pairs_[slot]] = kNoSlot;
        }
        count_ = 0u;
    }

    std::vector<std::uint16_t> slot_of_pair_;
    std::vector<std::uint32_t> active_pairs_;
    std::vector<double> rho_;
    std::vector<double> j_;
    double* global_j_ = nullptr;
    const double* global_rho_ = nullptr;
    std::size_t count_ = 0u;
    bool collecting_ = false;
    bool active_ = false;
};

inline RysJVeloxTaskDigest * & rys_j_velox_digest_slot() noexcept {
    static thread_local RysJVeloxTaskDigest* state = nullptr;
    return state;
}

class RysJVeloxDigestGuard {
public:
    explicit RysJVeloxDigestGuard(RysJVeloxTaskDigest* state) noexcept
        : previous_(rys_j_velox_digest_slot()) {
        rys_j_velox_digest_slot() = state;
    }
    ~RysJVeloxDigestGuard() noexcept {
        rys_j_velox_digest_slot() = previous_;
    }
    RysJVeloxDigestGuard(const RysJVeloxDigestGuard&) = delete;
    RysJVeloxDigestGuard& operator = (const RysJVeloxDigestGuard&) = delete;

private:
    RysJVeloxTaskDigest* previous_ = nullptr;
};
#endif

// Marker policy for the J-only speed experiment.  It intentionally has a
// complete scalar implementation: every class outside a dedicated producer
// slice still performs both Coulomb terms and never calls the G control.
struct RysDirectJSpeedProbeEmit {
    static inline void emit(std::vector<double>& /*vals*/,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double value) noexcept {
        rys_j_digest_one(
            *rys_jk_context_slot(), lhs_pair, rhs_pair, value);
    }
};

// Productive Coulomb-only policy.  Its producer selection is encoded by the
// type traits below rather than by an experiment build flag: all six s/p
// families contract before scalar emission, pp|pp uses the 4+5 tiled closer,
// and every residual d/f value retains the exact pair-space scalar receiver.
struct RysDirectJCoulombEmit {
    static inline void emit(std::vector<double>& /*vals*/,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double value) noexcept {
        rys_j_digest_one(
            *rys_jk_context_slot(), lhs_pair, rhs_pair, value);
    }
};

#ifdef TLNO_RYS_DIRECT_EXCHANGE
// Combined RHF consumer: ss|ss closes J and exchange before scalar emission;
// other classes use the fast ERI0 producer and the exact combined fallback.
struct RysDirectJKFusedEmit {
    static inline void emit(std::vector<double>& /*vals*/,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double value) noexcept {
        rys_jk_digest_one(
            *rys_jk_context_slot(), lhs_pair, rhs_pair, value);
    }
};

// Positive exchange-only policy.  The s/p terminals and selected d/p terminals
// contract before scalar emission; residual classes use the exact four-update
// K fallback.  No Coulomb term is formed by this endpoint.
inline void rys_k_digest_one(const RysJkEmitContext& context,
                             std::uint32_t lhs_pair,
                             std::uint32_t rhs_pair,
                             double value) noexcept {
    rys_jk_record_one(context, lhs_pair, rhs_pair, value);
    accumulate_k_symmetric_pair_fast(
        context, lhs_pair, rhs_pair, value);
}

struct RysDirectKExchangeEmit {
    static inline void emit(std::vector<double>& /*vals*/,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double value) noexcept {
        rys_k_digest_one(
            *rys_jk_context_slot(), lhs_pair, rhs_pair, value);
    }
};

#endif

// Marker policy for a strict producer-common J experiment.  Unlike the older
// speed-probe marker, no producer dispatcher recognizes this type specially:
// roots, recurrences, shell kernels and projection therefore follow the same
// route as the ordinary ERI stream.  Only the terminal receiver identifies it
// and contracts the already-produced values into J.
struct RysDirectJCommonProducerEmit {
    static inline void emit(std::vector<double>& /*vals*/,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double value) noexcept {
#ifdef TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST
        RysJVeloxTaskDigest* const state = rys_j_velox_digest_slot();
        if (state != nullptr && state->add(lhs_pair, rhs_pair, value)) {
            const RysJkEmitContext& context = *rys_jk_context_slot();
            rys_j_speed_probe_record_scalar(
                context, lhs_pair, rhs_pair, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
            g_rys_direct_j_speed_probe_audit
                .logical_j_terms_committed += 2u;
#endif
            return;
        }
#endif
        rys_j_digest_one(
            *rys_jk_context_slot(), lhs_pair, rhs_pair, value);
    }
};

#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
// WHAT: replay the same screened producer through an index-only policy in the
// untimed diagnostic binary.
// WHY: counters updated by the J consumer cannot independently detect an ERI
// that the consumer itself never saw; this policy follows the ordinary producer
// rather than the specialized ss|ps dispatch and hashes no numerical values.
// PURPOSE: count+XOR+modular-sum certify the complete multiset of canonical
// pair-pair indices without reintroducing numerical-accuracy requirements.
struct RysDirectJReferenceCountEmit {
    static inline void emit(std::vector<double>& /*vals*/,
                            std::uint32_t lhs_pair,
                            std::uint32_t rhs_pair,
                            double /*value*/) noexcept {
        RysDirectJSpeedProbeAudit& audit =
            g_rys_direct_j_speed_probe_audit;
        const std::uint64_t hash =
            rys_j_speed_probe_index_hash(lhs_pair, rhs_pair);
        ++audit.reference_canonical_eris;
        audit.reference_index_xor ^= hash;
        audit.reference_index_sum += hash;
    }
};
#endif

// These properties are deliberately independent.  A mixed J/K candidate needs
// the pair-space J scratch, but must retain the combined J-union-K density
// predicate and the exact combined fallback for every class it has not fused.
template <class Emit>
inline constexpr bool rys_emit_is_j_only_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit> ||
    std::is_same_v<Emit, RysDirectJSpeedProbeEmit> ||
    std::is_same_v<Emit, RysDirectJCommonProducerEmit>;

template <class Emit>
inline constexpr bool rys_emit_is_k_only_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_needs_j_pair_scratch_v =
    rys_emit_is_j_only_v<Emit>
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    || std::is_same_v<Emit, RysDirectJKFusedEmit>
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_combined_jk_fallback_v =
    std::is_same_v<Emit, RysDirectJKEmit>
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    || std::is_same_v<Emit, RysDirectJKFusedEmit>
#endif
    ;

// WHAT: opt only the exact common-producer J endpoint into the already proven
// ss|ss terminal contraction seam.
// WHY: the common endpoint must keep its ordinary root evaluator, primitive
// recurrence, screening predicates and traversal schedule; only the final
// destination changes from ERI emission to the two Coulomb accumulations.
// PURPOSE: this A/B measures pre-emission digest cost without accelerating or
// otherwise changing the ERI0 producer used as the denominator.
template <class Emit>
inline constexpr bool rys_emit_uses_preemit_ssss_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit> ||
    std::is_same_v<Emit, RysDirectJSpeedProbeEmit>
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    || std::is_same_v<Emit, RysDirectJKFusedEmit>
    || std::is_same_v<Emit, RysDirectKExchangeEmit>
#endif
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSSS) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_ssss_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    || std::is_same_v<Emit, RysDirectJKFusedEmit>
#endif
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSSS) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#else
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_ssss_exchange_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectJKFusedEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_ssss_positive_k_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

// First K-only producer closer beyond ss|ss.  Keep it independent from the J
// trait: K owns four crossed AO destinations and must never acquire j_pair
// scratch or the two Coulomb reductions merely because it shares this producer.
template <class Emit>
inline constexpr bool rys_emit_uses_preemit_ssps_positive_k_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_sspp_positive_k_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_psps_positive_k_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_pspp_positive_k_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_dpps_positive_k_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_pppp_positive_k_v =
    std::is_same_v<Emit, RysDirectKExchangeEmit>;

template <class Emit>
inline constexpr bool rys_emit_uses_preemit_dppp_positive_k_v =
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    std::is_same_v<Emit, RysDirectKExchangeEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_ssps_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPS) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#else
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_ssds_j_v =
#ifdef TLNO_J_COMMON_PRODUCER_PREEMIT_SSDS
    std::is_same_v<Emit, RysDirectJCommonProducerEmit>;
#else
    false;
#endif

// WHAT: let the isolated J host select the exact same bounded-moment and
// arbitrary-contraction s/p producers as ERI0 without changing its receiver.
// WHY: a late-vs-preemit digest A/B is causal only when both arms inherit the
// current ERI producer; otherwise a producer regression is mislabeled as J
// digestion cost.
// PURPOSE: TLNO_J_COMMON_PRODUCER_FAST_SP is the late-receiver control;
// PREEMIT_SP retains the provisional generic packet A/B, and PREEMIT_SP_ROWS
// selects the class-shaped row reducers.  Keeping the latter two disjoint is
// essential: otherwise the nominal generic control silently executes rows.
// None of these switches can enter the production Rys object (guard above).
template <class Emit>
inline constexpr bool rys_emit_uses_fast_sp_producer_v =
    std::is_same_v<Emit, RysPackedTensorEmit> ||
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#ifdef TLNO_RYS_DIRECT_EXCHANGE
    || std::is_same_v<Emit, RysDirectJKFusedEmit>
    || std::is_same_v<Emit, RysDirectKExchangeEmit>
#endif
#if defined(TLNO_J_COMMON_PRODUCER_FAST_SP) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPS_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPP_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPP_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PPPP_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_sp_j_v =
#ifdef TLNO_J_COMMON_PRODUCER_PREEMIT_SP
    std::is_same_v<Emit, RysDirectJCommonProducerEmit>;
#else
    false;
#endif

template <class Emit>
inline constexpr bool rys_emit_uses_j_density_screen_v =
    rys_emit_is_j_only_v<Emit>
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
    || std::is_same_v<Emit, RysDirectJReferenceCountEmit>
#endif
    ;

// WHAT: isolate the first register-topology replacement from the provisional
// generic s/p packet so it can be benchmarked against the same fast producer.
// WHY: ps|ps naturally exposes three output-stationary rows; selecting it
// alone tells us whether row digestion retains ERI0 speed before the wider
// pp tiles are rewritten.
// PURPOSE: this experiment changes only the terminal destination of ps|ps.
template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_psps_row_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPS_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#else
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_sspp_row_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPP_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#else
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_pspp_row_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPP_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#else
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_pppp_row_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#if defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PPPP_ROW) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
    defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS)
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#else
#endif
    ;

template <class Emit>
inline constexpr bool rys_emit_uses_pairspace_preemit_pppp_tiled45_j_v =
    std::is_same_v<Emit, RysDirectJCoulombEmit>
#ifdef TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45
    || std::is_same_v<Emit, RysDirectJCommonProducerEmit>
#else
#endif
    ;

static_assert(
    rys_emit_uses_fast_sp_producer_v<RysDirectJCoulombEmit> &&
    rys_emit_uses_pairspace_preemit_ssss_j_v<RysDirectJCoulombEmit> &&
    rys_emit_uses_pairspace_preemit_ssps_j_v<RysDirectJCoulombEmit> &&
    rys_emit_uses_pairspace_preemit_psps_row_j_v<RysDirectJCoulombEmit> &&
    rys_emit_uses_pairspace_preemit_sspp_row_j_v<RysDirectJCoulombEmit> &&
    rys_emit_uses_pairspace_preemit_pspp_row_j_v<RysDirectJCoulombEmit> &&
    rys_emit_uses_pairspace_preemit_pppp_row_j_v<RysDirectJCoulombEmit> &&
    rys_emit_uses_pairspace_preemit_pppp_tiled45_j_v <
        RysDirectJCoulombEmit > &&
    !rys_emit_uses_pairspace_preemit_sp_j_v<RysDirectJCoulombEmit>);

#ifdef TLNO_RYS_DIRECT_EXCHANGE
static_assert(
    rys_emit_needs_j_pair_scratch_v<RysDirectJKFusedEmit> &&
    !rys_emit_is_j_only_v<RysDirectJKFusedEmit> &&
    rys_emit_uses_combined_jk_fallback_v<RysDirectJKFusedEmit> &&
    rys_emit_uses_fast_sp_producer_v<RysDirectJKFusedEmit> &&
    rys_emit_uses_pairspace_preemit_ssss_j_v <
        RysDirectJKFusedEmit > &&
    rys_emit_uses_preemit_ssss_exchange_v <
        RysDirectJKFusedEmit > &&
    !rys_emit_uses_pairspace_preemit_ssps_j_v <
        RysDirectJKFusedEmit > &&
    !rys_emit_uses_pairspace_preemit_psps_row_j_v <
        RysDirectJKFusedEmit > &&
    !rys_emit_uses_pairspace_preemit_sspp_row_j_v <
        RysDirectJKFusedEmit > &&
    !rys_emit_uses_pairspace_preemit_pspp_row_j_v <
        RysDirectJKFusedEmit > &&
    !rys_emit_uses_pairspace_preemit_pppp_row_j_v <
        RysDirectJKFusedEmit > );

static_assert(
    rys_emit_is_k_only_v<RysDirectKExchangeEmit> &&
    !rys_emit_is_j_only_v<RysDirectKExchangeEmit> &&
    !rys_emit_needs_j_pair_scratch_v<RysDirectKExchangeEmit> &&
    !rys_emit_uses_combined_jk_fallback_v<RysDirectKExchangeEmit> &&
    rys_emit_uses_fast_sp_producer_v<RysDirectKExchangeEmit> &&
    rys_emit_uses_preemit_ssss_j_v<RysDirectKExchangeEmit> &&
    rys_emit_uses_preemit_ssss_positive_k_v <
        RysDirectKExchangeEmit > &&
    rys_emit_uses_preemit_ssps_positive_k_v <
        RysDirectKExchangeEmit > &&
    rys_emit_uses_preemit_sspp_positive_k_v <
        RysDirectKExchangeEmit > &&
    rys_emit_uses_preemit_psps_positive_k_v <
        RysDirectKExchangeEmit > &&
    rys_emit_uses_preemit_pspp_positive_k_v <
        RysDirectKExchangeEmit > &&
    rys_emit_uses_preemit_dpps_positive_k_v <
        RysDirectKExchangeEmit > &&
    rys_emit_uses_preemit_dppp_positive_k_v <
        RysDirectKExchangeEmit > &&
    !rys_emit_uses_pairspace_preemit_ssss_j_v <
        RysDirectKExchangeEmit > );
#endif

#ifdef TLNO_J_COMMON_PRODUCER_PREEMIT_SP
static_assert(
    rys_emit_uses_pairspace_preemit_sp_j_v<RysDirectJCommonProducerEmit>);
static_assert(
    !rys_emit_uses_pairspace_preemit_psps_row_j_v <
        RysDirectJCommonProducerEmit > &&
    !rys_emit_uses_pairspace_preemit_sspp_row_j_v <
        RysDirectJCommonProducerEmit > &&
    !rys_emit_uses_pairspace_preemit_pspp_row_j_v <
        RysDirectJCommonProducerEmit > &&
    !rys_emit_uses_pairspace_preemit_pppp_row_j_v <
        RysDirectJCommonProducerEmit > );
#endif

#ifdef TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS
static_assert(
    !rys_emit_uses_pairspace_preemit_sp_j_v<RysDirectJCommonProducerEmit>);
static_assert(
    rys_emit_uses_pairspace_preemit_psps_row_j_v <
        RysDirectJCommonProducerEmit > &&
    rys_emit_uses_pairspace_preemit_sspp_row_j_v <
        RysDirectJCommonProducerEmit > &&
    rys_emit_uses_pairspace_preemit_pspp_row_j_v <
        RysDirectJCommonProducerEmit > &&
    rys_emit_uses_pairspace_preemit_pppp_row_j_v <
        RysDirectJCommonProducerEmit > );
static_assert(
    !rys_emit_uses_pairspace_preemit_pppp_tiled45_j_v <
        RysDirectJCommonProducerEmit > );
#endif

#ifdef TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45
static_assert(
    !rys_emit_uses_pairspace_preemit_sp_j_v<RysDirectJCommonProducerEmit>);
static_assert(
    rys_emit_uses_pairspace_preemit_psps_row_j_v <
        RysDirectJCommonProducerEmit > &&
    rys_emit_uses_pairspace_preemit_sspp_row_j_v <
        RysDirectJCommonProducerEmit > &&
    rys_emit_uses_pairspace_preemit_pspp_row_j_v <
        RysDirectJCommonProducerEmit > &&
    rys_emit_uses_pairspace_preemit_pppp_row_j_v <
        RysDirectJCommonProducerEmit > &&
    rys_emit_uses_pairspace_preemit_pppp_tiled45_j_v <
        RysDirectJCommonProducerEmit > );
#endif
#endif
