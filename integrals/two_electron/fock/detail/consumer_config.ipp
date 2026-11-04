#ifndef TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_CONSUMER_CONFIG_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_FOCK_DETAIL_CONSUMER_CONFIG_IPP

// Private matrix-builder implementation fragment.
// Included inside the ERI producer translation unit at its original scope and
// declaration position, preserving compile-time specialization and inlining.

// Exact direct exchange and fused RHF consumers are part of this backend.
// This private capability flag enables their producer terminals and K-only
// density masks without enabling any of the independent experiment switches.
#define TLNO_RYS_DIRECT_EXCHANGE 1

#if !defined(TLNO_BUILD_RYS_AUDIT) && \
    (defined(TLNO_J_SPEED_PROBE_PAIR_ACCUMULATOR) || \
     defined(TLNO_J_SPEED_PROBE_DISABLE_PAIR_ACCUMULATOR) || \
     defined(TLNO_J_SPEED_PROBE_DIAGNOSTICS) || \
     defined(TLNO_J_SPEED_PROBE_EXACT_ROOTS) || \
     defined(TLNO_J_SPEED_PROBE_EXACT_N1_LAGRANGE) || \
     defined(TLNO_J_SPEED_PROBE_EXACT_N2_LAGRANGE) || \
     defined(TLNO_J_SPEED_PROBE_LEGACY_ROOT1) || \
     defined(TLNO_J_COMMON_PRODUCER_CVHF_BLOCK_DIGEST) || \
     defined(TLNO_J_COMMON_PRODUCER_GTFOCK_GROUP_CACHE) || \
     defined(TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST) || \
     defined(TLNO_J_COMMON_PRODUCER_PSI4_SHELL_JT) || \
     defined(TLNO_J_COMMON_PRODUCER_FAST_SP) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSSS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSDS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPS_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPP_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPP_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PPPP_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP))
#error "J speed-probe switches require the experiment-only Rys object"
#endif

// Canonical pair-space accumulation is an implementation property of the
// productive Coulomb receiver, not an experiment selector.  The experiment
// host may still disable it for the retained matrix-space A/B; production and
// validation never observe that experimental switch.
#if !defined(TLNO_BUILD_RYS_AUDIT) || \
    !defined(TLNO_J_SPEED_PROBE_DISABLE_PAIR_ACCUMULATOR)
#define TLNO_RYS_J_PAIR_SPACE_ACCUMULATION 1
#endif

// Common-producer experiments are intentionally mutually exclusive.  Their
// purpose is to compare one controlled producer or output receiver at a time
// while compiling the scheduler and screening predicates from the same source.
#if (defined(TLNO_J_COMMON_PRODUCER_CVHF_BLOCK_DIGEST) + \
     defined(TLNO_J_COMMON_PRODUCER_GTFOCK_GROUP_CACHE) + \
     defined(TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST) + \
     defined(TLNO_J_COMMON_PRODUCER_PSI4_SHELL_JT) + \
     defined(TLNO_J_COMMON_PRODUCER_FAST_SP) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSSS) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPS) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSDS) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPS_ROW) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPP_ROW) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPP_ROW) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PPPP_ROW) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS) + \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP)) > 1
#error "select at most one TLNO common-producer J experiment"
#endif

#if (defined(TLNO_J_COMMON_PRODUCER_GTFOCK_GROUP_CACHE) || \
     defined(TLNO_J_COMMON_PRODUCER_VELOX_TASK_LOCAL_DIGEST) || \
     defined(TLNO_J_COMMON_PRODUCER_PSI4_SHELL_JT) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSSS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSDS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPS_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SSPP_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PSPP_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_PPPP_ROW) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS_TILED45) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP_ROWS) || \
     defined(TLNO_J_COMMON_PRODUCER_PREEMIT_SP)) && \
    !defined(TLNO_RYS_J_PAIR_SPACE_ACCUMULATION)
#error "experimental J digests require canonical pair-space accumulation"
#endif

#ifndef TLNO_J_VELOX_TASK_MAX_PAIRS
#define TLNO_J_VELOX_TASK_MAX_PAIRS 512
#endif

#ifndef TLNO_J_GTFOCK_CACHE_SLOTS
#define TLNO_J_GTFOCK_CACHE_SLOTS 8
#endif

#ifndef TLNO_J_GTFOCK_MIN_BLOCK_ERIS
#define TLNO_J_GTFOCK_MIN_BLOCK_ERIS 16
#endif
#endif
