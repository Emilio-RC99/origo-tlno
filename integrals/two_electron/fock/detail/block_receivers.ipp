#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_BLOCK_RECEIVERS_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_BLOCK_RECEIVERS_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

#ifdef TLNO_J_COMMON_PRODUCER_GTFOCK_GROUP_CACHE
// GTFock-like persistent pair-group cache.  Unlike a per-quartet scratch, a
// slot survives across producer blocks and tasks until LRU eviction, so its
// density pack and output tile can be reused without altering traversal.
// MEASURED/REJECTED (2026-08-21): median candidate/control was 1.121x for
// benzene/6-31G(d,p) and 1.043x for benzene/def2-SVP.  Eight delayed tiles
// add lookup/eviction traffic while the complete 52-KiB pair accumulator for
// M=114 already fits in cache.  Keep this macro only for reproduction.
class RysJGtfockGroupCache {
public:
    static constexpr std::size_t kMaxComponents = 81u;
    static constexpr std::size_t kSlotCount =
        static_cast<std::size_t>(TLNO_J_GTFOCK_CACHE_SLOTS);
    static_assert(TLNO_J_GTFOCK_CACHE_SLOTS >= 2,
                  "GTFock digest needs at least two protected cache slots");

    RysJGtfockGroupCache(double* global_j,
                         const double* global_rho) noexcept
        : global_j_(global_j), global_rho_(global_rho) {}

    template <class ValueFn>
    bool consume(
        const std::vector<PackedPairComponentH>& lhs_refs,
        const std::vector<PackedPairComponentH>& rhs_refs,
        bool same_pair_group,
        ValueFn&& value_of,
        const RysJkEmitContext& context) noexcept {
        if (lhs_refs.empty() || rhs_refs.empty() ||
            lhs_refs.size() > kMaxComponents ||
            rhs_refs.size() > kMaxComponents ||
            lhs_refs.size() * rhs_refs.size() <
                static_cast<std::size_t>(TLNO_J_GTFOCK_MIN_BLOCK_ERIS)) {
            return false;
        }
        const int lhs_slot = acquire(lhs_refs, -1);
        if (lhs_slot < 0) return false;
        const int rhs_slot = same_pair_group
            ? lhs_slot
            : acquire(rhs_refs, lhs_slot);
        if (rhs_slot < 0) return false;

        Slot& lhs = slots_[static_cast<std::size_t>(lhs_slot)];
        Slot& rhs = slots_[static_cast<std::size_t>(rhs_slot)];
        lhs.dirty = true;
        rhs.dirty = true;
        for (std::size_t li = 0; li < lhs_refs.size(); ++li) {
            const std::uint32_t lhs_pair = lhs_refs[li].pair_index;
            const double lhs_density = lhs.rho[li];
            double lhs_acc = lhs.acc[li];
            for (std::size_t ri = 0; ri < rhs_refs.size(); ++ri) {
                const std::uint32_t rhs_pair = rhs_refs[ri].pair_index;
                if (same_pair_group && rhs_pair > lhs_pair) continue;
                const double value = value_of(lhs_refs[li], rhs_refs[ri]);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                rys_j_speed_probe_record_scalar(
                    context, lhs_pair, rhs_pair, value);
#endif
                if (lhs_pair == rhs_pair) {
                    lhs_acc += value * lhs_density;
                } else {
                    lhs_acc += value * rhs.rho[ri];
                    rhs.acc[ri] += value * lhs_density;
                }
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                g_rys_direct_j_speed_probe_audit
                    .logical_j_terms_committed += 2u;
#endif
            }
            lhs.acc[li] = lhs_acc;
        }
        return true;
    }

    void flush_all() noexcept {
        for (Slot& slot : slots_) flush(slot);
    }

private:
    struct Slot {
        const PackedPairComponentH* tag = nullptr;
        std::size_t count = 0u;
        std::uint64_t stamp = 0u;
        bool dirty = false;
        alignas(64) std::array<double, kMaxComponents> acc{};
        alignas(64) std::array<double, kMaxComponents> rho{};
    };

    int acquire(const std::vector<PackedPairComponentH>& refs,
                int protected_slot) noexcept {
        const PackedPairComponentH* const tag = refs.data();
        ++clock_;
        for (std::size_t i = 0; i < kSlotCount; ++i) {
            Slot& slot = slots_[i];
            if (slot.tag == tag && slot.count == refs.size()) {
                slot.stamp = clock_;
                return static_cast<int>(i);
            }
        }

        int victim = -1;
        std::uint64_t oldest = std::numeric_limits<std::uint64_t>::max();
        for (std::size_t i = 0; i < kSlotCount; ++i) {
            if (static_cast<int>(i) == protected_slot) continue;
            const Slot& slot = slots_[i];
            if (slot.tag == nullptr) {
                victim = static_cast<int>(i);
                break;
            }
            if (slot.stamp < oldest) {
                oldest = slot.stamp;
                victim = static_cast<int>(i);
            }
        }
        if (victim < 0) return -1;
        Slot& slot = slots_[static_cast<std::size_t>(victim)];
        flush(slot);
        slot.tag = tag;
        slot.count = refs.size();
        slot.stamp = clock_;
        slot.dirty = false;
        std::fill_n(slot.acc.data(), slot.count, 0.0);
        for (std::size_t i = 0; i < slot.count; ++i) {
            slot.rho[i] = global_rho_[refs[i].pair_index];
        }
        return victim;
    }

    void flush(Slot& slot) noexcept {
        if (slot.tag == nullptr || !slot.dirty) return;
        for (std::size_t i = 0; i < slot.count; ++i) {
            global_j_[slot.tag[i].pair_index] += slot.acc[i];
            slot.acc[i] = 0.0;
        }
        slot.dirty = false;
    }

    std::array<Slot, kSlotCount> slots_{};
    double* global_j_ = nullptr;
    const double* global_rho_ = nullptr;
    std::uint64_t clock_ = 0u;
};

inline RysJGtfockGroupCache * & rys_j_gtfock_cache_slot() noexcept {
    static thread_local RysJGtfockGroupCache* cache = nullptr;
    return cache;
}

class RysJGtfockCacheGuard {
public:
    explicit RysJGtfockCacheGuard(RysJGtfockGroupCache* cache) noexcept
        : previous_(rys_j_gtfock_cache_slot()) {
        rys_j_gtfock_cache_slot() = cache;
    }
    ~RysJGtfockCacheGuard() noexcept {
        rys_j_gtfock_cache_slot() = previous_;
    }
    RysJGtfockCacheGuard(const RysJGtfockCacheGuard&) = delete;
    RysJGtfockCacheGuard& operator = (const RysJGtfockCacheGuard&) = delete;

private:
    RysJGtfockGroupCache* previous_ = nullptr;
};
#endif

// BLOCK-wise digestion of the stream directly into the J/K accumulator.
//
// WHY it exists: the per-value seam forces resolving the J/K context -- a
// thread_local dereference -- ONCE PER INTEGRAL. With millions of integrals
// that shows, and it was already seen up close: the bra-side memo lost
// precisely by paying one extra TLS access per integral. Here the context is
// resolved once per block and the inner loop keeps g and p in
// registers.
//
// The packed policy does NOT change behavior: it walks the block calling its
// per-value emit, exactly as before.
template <class Emit, class ValueFn>
inline void rys_emit_block(
    std::vector<double>& vals,
    const std::vector<PackedPairComponentH>& lhs_refs,
    const std::vector<PackedPairComponentH>& rhs_refs,
    bool same_pair_group,
    ValueFn&& value_of) noexcept {
    if constexpr (rys_emit_uses_combined_jk_fallback_v<Emit>) {
        const RysJkEmitContext& context = *rys_jk_context_slot();
        // WHAT: keep only the streaming lhs-hoisted consumer at this seam; the
        // persistent M^2 remap, the reducer over tiles and the contraction with
        // a scratch traversed six times were removed.
        // WHY: the fresh paired J/K accelerations regressed: moving the
        // six RMWs to tiles does not remove them, and six passes reread each ERI.
        // PURPOSE: the next block consumer must fuse with the projected
        // producer, keep family accumulators in registers and write each output
        // once, with no per-ERI RMW, local or global.
#ifdef TLNO_JK_NO_LHS_HOIST
        // Control branch of the A/B. It exists so that "with extraction" and
        // "without extraction" are the SAME source one macro apart: measuring
        // against a saved older binary is how before/after comparisons get lost,
        // and it also invites comparing builds that differ in everything
        // else. Do not compile this way in production.
        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (same_pair_group &&
                    rhs_ref.pair_index > lhs_ref.pair_index) continue;
                rys_jk_digest_one(context, lhs_ref.pair_index,
                                  rhs_ref.pair_index,
                                  value_of(lhs_ref, rhs_ref));
            }
        }
#else
        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            // A single pair_info lookup per block row instead of one
            // per integral; see RysJkLhsView for why the compiler
            // cannot do it alone.
            const RysJkLhsView lv =
                rys_jk_lhs_view(context, lhs_ref.pair_index);
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (same_pair_group &&
                    rhs_ref.pair_index > lhs_ref.pair_index) continue;
                rys_jk_digest_one_lhs(context, lv, rhs_ref.pair_index,
                                      value_of(lhs_ref, rhs_ref));
            }
        }
#endif
    } else if constexpr (rys_emit_is_k_only_v<Emit>) {
        const RysJkEmitContext& context = *rys_jk_context_slot();
        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            const RysJkLhsView lhs =
                rys_jk_lhs_view(context, lhs_ref.pair_index);
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (same_pair_group &&
                    rhs_ref.pair_index > lhs_ref.pair_index) continue;
                rys_k_digest_one_lhs(context, lhs, rhs_ref.pair_index,
                                     value_of(lhs_ref, rhs_ref));
            }
        }
    } else if constexpr (rys_emit_is_j_only_v<Emit>) {
        const RysJkEmitContext& context = *rys_jk_context_slot();
        // WHAT: hoist the lhs J metadata across the rhs projection row.
        // WHY: this receiver is still necessary for already-materialized d/f
        // blocks, but J needs only one output offset, one P value and one
        // symmetry scale from the lhs side.
        // PURPOSE: the fallback classes pay the minimum honest scalar J
        // digest while ss|ps bypasses this receiver altogether below.
        //
        // MEASURED AND DISCARDED (2026-08-15): accumulating the rhs side of the
        // block in a STACK buffer -- resolving (index, scale, P) once per block
        // and writing each g[rhs] once instead of once per integral --
        // worsened t_J/t_ERI by 7.0% on caffeine/631g and 2.0% on
        // tyrosine/631g, with J identical to rounding (JFP rel 6.7e-16 and
        // 3.4e-15). A/B of 7 passes x 3 repetitions, both binaries from the
        // same source one macro apart, alternating the order, machine
        // at loadavg1=0.26.
        //
        // This is the THIRD measured rejection of the same family: the bra-side
        // memo and the materialization of rhs in a thread_local vector
        // (+5% to +8%) already failed. That one was blamed on the per-integral
        // TLS access; with an automatic buffer there is no TLS and it still
        // loses, so that was not the cause: in these classes the blocks are so
        // small that the per-block precomputation does not amortize over the few
        // lhs rows that consume it, and the four auxiliary tables evict L1.
        //
        // Do not reintroduce any variant of "hoisting the rhs side" without measuring.
        // The routes that DO win in s/p (ssss, ssps) do not hoist the rhs: they fuse
        // the kernel with the digestion and keep the deltas of BOTH sides in
        // registers inside the same vectorized microkernel. That is the
        // path for sspp/psps/pspp/pppp, not this one.
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
        double* const __restrict j_pair = context.j_pair;
        const double* const __restrict rho = context.j_pair_density;
#endif
#ifndef TLNO_RYS_J_EXACT_ZERO_BLOCK_DIGEST
#define TLNO_RYS_J_EXACT_ZERO_BLOCK_DIGEST 1
#endif
#if defined(TLNO_RYS_J_PAIR_SPACE_ACCUMULATION) && TLNO_RYS_J_EXACT_ZERO_BLOCK_DIGEST
        if constexpr (std::is_same_v<Emit, RysDirectJCoulombEmit>) {
            // Exact zeros select the direction of the Coulomb contraction.
            // This leaves the producer's packets, roots and primitive gates
            // untouched. In particular, it is not a small-density cutoff.
            bool lhs_zero = true, rhs_zero = true;
            for (const auto& ref : lhs_refs) {
                if (rho[ref.pair_index] != 0.0) {
                    lhs_zero = false;
                    break;
                }
            }
            for (const auto& ref : rhs_refs) {
                if (rho[ref.pair_index] != 0.0) {
                    rhs_zero = false;
                    break;
                }
            }
            if (lhs_zero && rhs_zero) {
                // Keep the canonical stream visible to structural audits,
                // even though neither J destination has a nonzero term.
#if defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT) || defined(TLNO_JK_COVERAGE_CHECKSUM)
                for (const auto& lhs_ref : lhs_refs) {
                    for (const auto& rhs_ref : rhs_refs) {
                        if (same_pair_group &&
                            rhs_ref.pair_index > lhs_ref.pair_index) continue;
                        const double value = value_of(lhs_ref, rhs_ref);
                        rys_j_speed_probe_record_scalar(
                            context, lhs_ref.pair_index,
                            rhs_ref.pair_index, value);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        g_rys_direct_j_speed_probe_audit
                            .logical_j_terms_committed += 2u;
#endif
                    }
                }
#endif
                return;
            }
            if (!same_pair_group && lhs_zero) {
                // Only J[lhs] receives a term. Preserve the contiguous rhs
                // row and commit once, removing all zero rhs RMWs.
                for (const auto& lhs_ref : lhs_refs) {
                    double acc = 0.0;
                    for (const auto& rhs_ref : rhs_refs) {
                        const double value = value_of(lhs_ref, rhs_ref);
                        rys_j_speed_probe_record_scalar(
                            context, lhs_ref.pair_index,
                            rhs_ref.pair_index, value);
                        acc += value * rho[rhs_ref.pair_index];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        g_rys_direct_j_speed_probe_audit
                            .logical_j_terms_committed += 2u;
#endif
                    }
                    j_pair[lhs_ref.pair_index] += acc;
                }
                return;
            }
            if (!same_pair_group && rhs_zero) {
                // Only J[rhs] receives a term. Hold that output in a register
                // across the lhs dot product, so a sparse SAD block no longer
                // writes the large global pair accumulator once per ERI.
                // Every term is retained; the local sum can round differently
                // from repeated additions to an already nonzero destination.
                for (const auto& rhs_ref : rhs_refs) {
                    double acc = 0.0;
                    for (const auto& lhs_ref : lhs_refs) {
                        const double value = value_of(lhs_ref, rhs_ref);
                        rys_j_speed_probe_record_scalar(
                            context, lhs_ref.pair_index,
                            rhs_ref.pair_index, value);
                        acc += value * rho[lhs_ref.pair_index];
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        g_rys_direct_j_speed_probe_audit
                            .logical_j_terms_committed += 2u;
#endif
                    }
                    j_pair[rhs_ref.pair_index] += acc;
                }
                return;
            }
        }
#endif
#if defined(TLNO_J_COMMON_PRODUCER_GTFOCK_GROUP_CACHE) && \
    defined(TLNO_RYS_J_PAIR_SPACE_ACCUMULATION)
        if constexpr (std::is_same_v <
                          Emit, RysDirectJCommonProducerEmit > ) {
            RysJGtfockGroupCache* const cache =
                rys_j_gtfock_cache_slot();
            if (cache != nullptr &&
                cache->consume(lhs_refs, rhs_refs, same_pair_group,
                               value_of, context)) {
                return;
            }
        }
#endif
#if defined(TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST) && \
    defined(TLNO_RYS_J_PAIR_SPACE_ACCUMULATION)
        // MEASURED/REJECTED (2026-08-21): task-local accumulation was 1.065x
        // the control for benzene/6-31G(d,p), and a seven-pair follow-up for
        // def2-SVP settled at 1.021x after two misleading slow controls.  The
        // slot-map setup/scatter cost was not recovered by low-L locality.
        if constexpr (std::is_same_v <
                          Emit, RysDirectJCommonProducerEmit > ) {
            RysJVeloxTaskDigest* const state =
                rys_j_velox_digest_slot();
            if (state != nullptr && state->active()) {
                for (const PackedPairComponentH& lhs_ref : lhs_refs) {
                    for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                        if (same_pair_group &&
                            rhs_ref.pair_index > lhs_ref.pair_index) continue;
                        const double value = value_of(lhs_ref, rhs_ref);
                        if (state->add(lhs_ref.pair_index,
                                       rhs_ref.pair_index, value)) {
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                            rys_j_speed_probe_record_scalar(
                                context, lhs_ref.pair_index,
                                rhs_ref.pair_index, value);
                            g_rys_direct_j_speed_probe_audit
                                .logical_j_terms_committed += 2u;
#endif
                        } else {
                            // A task setup that did not cover this component
                            // remains exact by falling back per value.
                            rys_j_digest_one(
                                context, lhs_ref.pair_index,
                                rhs_ref.pair_index, value);
                        }
                    }
                }
                return;
            }
        }
#endif
#if defined(TLNO_J_COMMON_PRODUCER_PSI4_SHELL_JT) && \
    defined(TLNO_RYS_J_PAIR_SPACE_ACCUMULATION)
        if constexpr (std::is_same_v <
                          Emit, RysDirectJCommonProducerEmit > ) {
            // Psi4 DirectJK keeps two JT destinations local for a shell task.
            // At TLNO's producer-common seam the literal one-block analogue
            // is a contiguous lhs traversal plus one rhs scratch and two
            // flushes.  This is deliberately a macro-only control: the true
            // multi-quartet/atomic design matters only after MT is enabled.
            // MEASURED (2026-08-21): it improved the local paired median by
            // 3.1% for benzene/6-31G(d,p), but regressed def2-SVP by 8.7% and
            // the cross-engine retention campaign still failed at 0.9464x.
            // It therefore remains experimental and is off by default.
            constexpr std::size_t kMaxPurePairComponents = 81u;
            if (rhs_refs.size() <= kMaxPurePairComponents) {
                std::array<double, kMaxPurePairComponents> rhs_acc;
                std::fill_n(rhs_acc.data(), rhs_refs.size(), 0.0);
                for (const PackedPairComponentH& lhs_ref : lhs_refs) {
                    const std::uint32_t lhs_pair = lhs_ref.pair_index;
                    const double lhs_density = rho[lhs_pair];
                    double lhs_acc = 0.0;
                    for (std::size_t ri = 0; ri < rhs_refs.size(); ++ri) {
                        const PackedPairComponentH& rhs_ref = rhs_refs[ri];
                        const std::uint32_t rhs_pair = rhs_ref.pair_index;
                        if (same_pair_group && rhs_pair > lhs_pair) continue;
                        const double value = value_of(lhs_ref, rhs_ref);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        rys_j_speed_probe_record_scalar(
                            context, lhs_pair, rhs_pair, value);
#endif
                        if (lhs_pair == rhs_pair) {
                            lhs_acc += value * lhs_density;
                        } else {
                            lhs_acc += value * rho[rhs_pair];
                            rhs_acc[ri] += value * lhs_density;
                        }
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        g_rys_direct_j_speed_probe_audit
                            .logical_j_terms_committed += 2u;
#endif
                    }
                    j_pair[lhs_pair] += lhs_acc;
                }
                for (std::size_t ri = 0; ri < rhs_refs.size(); ++ri) {
                    j_pair[rhs_refs[ri].pair_index] += rhs_acc[ri];
                }
                return;
            }
        }
#endif
#if defined(TLNO_J_COMMON_PRODUCER_CVHF_BLOCK_DIGEST) && \
    defined(TLNO_RYS_J_PAIR_SPACE_ACCUMULATION)
        // MEASURED/REJECTED (2026-08-21): this exact CVHF-style topology
        // regressed the paired J median by 4.31% for benzene/6-31G(d,p) and
        // 10.14% for benzene/def2-SVP (7 alternating pairs, 3 inner repeats,
        // all certified screens active).  Keep it only as a reproducible A/B;
        // never enable the macro in the default build.  CVHF stores lhs
        // contiguously, whereas TLNO's projected block stores rhs contiguously,
        // so this rhs-major traversal turns its loads into strided accesses.
        if constexpr (std::is_same_v <
                          Emit, RysDirectJCommonProducerEmit > ) {
            // WHAT: reproduce the output-stationary contraction topology of
            // PySCF CVHFnrs8_ji_s2kl at TLNO's already-projected block seam.
            // Reference: PySCF 2.12.1 lib/vhf/nr_direct_dot.c, Apache-2.0.
            // One contiguous lhs scratch is updated while each rhs output is
            // held in a scalar; both sides are written to global pair space
            // once per block instead of once per canonical ERI.
            //
            // WHY: this macro changes only terminal digestion.  value_of is
            // still invoked exactly once for every canonical value selected
            // by the unchanged TLNO scheduler and screening gates.  Roots,
            // recurrence, contraction and Cartesian-to-spherical projection
            // therefore remain byte-identical to the control build.
            //
            // PURPOSE: test the one architectural feature of CVHF not covered
            // by the rejected lhs-major RHS buffer: rhs-major traversal with
            // a dense lhs scratch and scalar rhs reduction.  The fixed bound
            // is the largest pure-spherical shell-pair block for L<=4 (9*9).
            constexpr std::size_t kMaxPurePairComponents = 81u;
            const std::size_t lhs_count = lhs_refs.size();
            if (lhs_count <= kMaxPurePairComponents) {
                std::array<double, kMaxPurePairComponents> lhs_acc;
                std::array<double, kMaxPurePairComponents> lhs_density;
                std::fill_n(lhs_acc.data(), lhs_count, 0.0);
                for (std::size_t lhs_index = 0;
                     lhs_index < lhs_count; ++lhs_index) {
                    lhs_density[lhs_index] = rho[
                        lhs_refs[lhs_index].pair_index];
                }

                for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                    const std::uint32_t rhs_pair = rhs_ref.pair_index;
                    const double rhs_density = rho[rhs_pair];
                    double rhs_acc = 0.0;
                    for (std::size_t lhs_index = 0;
                         lhs_index < lhs_count; ++lhs_index) {
                        const PackedPairComponentH& lhs_ref =
                            lhs_refs[lhs_index];
                        const std::uint32_t lhs_pair = lhs_ref.pair_index;
                        if (same_pair_group && rhs_pair > lhs_pair) continue;
                        const double value = value_of(lhs_ref, rhs_ref);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        rys_j_speed_probe_record_scalar(
                            context, lhs_pair, rhs_pair, value);
#endif
                        if (lhs_pair == rhs_pair) {
                            // The canonical diagonal owns one Coulomb term;
                            // placing it on lhs avoids a duplicate rhs update.
                            lhs_acc[lhs_index] +=
                                value * lhs_density[lhs_index];
                        } else {
                            lhs_acc[lhs_index] += value * rhs_density;
                            rhs_acc += value * lhs_density[lhs_index];
                        }
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                        g_rys_direct_j_speed_probe_audit
                            .logical_j_terms_committed += 2u;
#endif
                    }
                    j_pair[rhs_pair] += rhs_acc;
                }
                for (std::size_t lhs_index = 0;
                     lhs_index < lhs_count; ++lhs_index) {
                    j_pair[lhs_refs[lhs_index].pair_index] +=
                        lhs_acc[lhs_index];
                }
                return;
            }
        }
#endif
        // MEASURED AND DISCARDED (2026-08-20): two output-stationary passes
        // restricted to d blocks (size 5/15/25) removed the rhs RMW,
        // but rereading the block raised the median 0.281625 -> 0.291994 s
        // (+3.68%) on benzene/def2-SVP, 7 alternating A/B runs. Do not reintroduce.
        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            const RysJLhsView lhs =
                rys_j_lhs_view(context, lhs_ref.pair_index);
            // WHAT: accumulate the lhs side in a register and write it once.
            // WHY: g[lhs.matrix_index] is the SAME address over the whole
            // rhs loop, but the compiler cannot hoist it because `g` and `p`
            // are both double* and may alias; without this rewrite it pays a
            // scattered read-modify-write per integral that is not needed.
            // PURPOSE: it removes half of the scattered write traffic in J,
            // which is where s/p loses against the ERI stream -- the
            // blocks there are tiny and amortize nothing.
            //
            // The multiset of summed terms is identical; only the reduction
            // order changes, so JFP matches to rounding and JHASH need not.
            // It is validated with a tolerance, not bit for bit.
            double lhs_acc = 0.0;
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (same_pair_group &&
                    rhs_ref.pair_index > lhs_ref.pair_index) continue;
                const std::uint32_t rhs_pair = rhs_ref.pair_index;
                const double value = value_of(lhs_ref, rhs_ref);
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                // This block bypasses Emit::emit, so the diagnostic traversal
                // must explicitly account for its canonical ERI identity.
                rys_j_speed_probe_record_scalar(
                    context, lhs.pair, rhs_pair, value);
#endif
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
                if (lhs.pair == rhs_pair) {
                    lhs_acc += value * lhs.density;
                } else {
                    lhs_acc += value * rho[rhs_pair];
                    j_pair[rhs_pair] += value * lhs.density;
                }
#else
                const RysJkPairInfo& ri = context.pair_info[rhs_pair];
                const std::size_t rhs_index =
                    ri.first_row + static_cast<std::size_t>(ri.second);
                double u4 = 4.0 * value * lhs.scale *
                            static_cast<double>(ri.scale);
                if (lhs.pair == rhs_pair) u4 *= 0.5;
                lhs_acc += u4 * context.p[rhs_index];
                context.g[rhs_index] += u4 * lhs.density;
#endif
#ifdef TLNO_J_SPEED_PROBE_DIAGNOSTICS
                g_rys_direct_j_speed_probe_audit
                    .logical_j_terms_committed += 2u;
#endif
            }
#ifdef TLNO_RYS_J_PAIR_SPACE_ACCUMULATION
            j_pair[lhs.pair] += lhs_acc;
#else
            context.g[lhs.matrix_index] += lhs_acc;
#endif
        }
    } else {
        for (const PackedPairComponentH& lhs_ref : lhs_refs) {
            for (const PackedPairComponentH& rhs_ref : rhs_refs) {
                if (same_pair_group &&
                    rhs_ref.pair_index > lhs_ref.pair_index) continue;
                Emit::emit(vals, lhs_ref.pair_index, rhs_ref.pair_index,
                           value_of(lhs_ref, rhs_ref));
            }
        }
    }
}
#endif
