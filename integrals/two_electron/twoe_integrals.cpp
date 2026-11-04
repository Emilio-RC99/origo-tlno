// Sole production host for the Rys/HGP implementation.
//
// Validation and experiment builds have their own translation-unit hosts.
// This file must therefore never be repurposed through role macros: it always
// instantiates exactly the production packed-stream and direct-J/K policies.

#if defined(TLNO_BUILD_RYS_VALIDATION) || \
    defined(TLNO_BUILD_RYS_AUDIT) || \
    defined(TLNO_VALIDATE_RYS_SPHERICAL_PACKED) || \
    defined(TLNO_JK_COVERAGE_CHECKSUM) || \
    defined(TLNO_VALIDATE_RYS_STREAM_FINGERPRINT)
#error "twoe_integrals.cpp is the production-only Rys host"
#endif

#include "integrals/two_electron/rys/rys_engine.ipp"
