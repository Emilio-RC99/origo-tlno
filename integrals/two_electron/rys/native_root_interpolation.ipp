#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_ROOT_INTERPOLATION_IPP
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_ROOT_INTERPOLATION_IPP

#ifndef TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_LAGRANGE_ROOTS
// Retained for the native W8 build with compact DP/FS. The complete-stream
// gain belongs to that measured bundle; W2/W4 remain explicit experiments.
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_LAGRANGE_ROOTS 1
#else
#define TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_LAGRANGE_ROOTS 0
#endif
#endif
#if TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_LAGRANGE_ROOTS != 0 && \
    TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_LAGRANGE_ROOTS != 1
#error "TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_LAGRANGE_ROOTS must be 0 or 1"
#endif
// Included after Packet<W> and newton5_pkt<W> in native_roots.ipp.
// WHAT: construct the five interpolation coefficients once per quartet packet,
// then reuse them for every root and every weight at the same T coordinates.
// WHY: Newton differences depend on five table samples and are rebuilt for
// each output. Cardinal coefficients depend only on T and can be shared by
// all 2*N outputs. The table, stencil rows, boundaries and asymptotic repair
// are unchanged. No larger coefficient table or extra gather is introduced.
// The formula changes floating-point association. The retained W8 bundle
// passed full-value gates and native timing; source FLOP savings alone do not
// establish a wall-time gain.

// Additional experiment: On AVX512 measure speedup and accuracy for 3 and 7 roots.

template<int W>
struct NativeRootInterpolation {
    using V = Packet<W>;
#if TLNO_INTEGRALS_TWO_ELECTRON_RYS_NATIVE_LAGRANGE_ROOTS
    V l0, l1, l2, l3, l4;

    explicit NativeRootInterpolation(V t) noexcept {
        const V tm1 = t - V(1.0), tm2 = t - V(2.0);
        const V tm3 = t - V(3.0), tm4 = t - V(4.0);
        const V t01 = t * tm1, t34 = tm3 * tm4, t012 = t01 * tm2;
        l0 = (tm1 * tm2) * t34 * V(1.0/24.0);
        l1 = (t * tm2) * t34 * V(-1.0/6.0);
        l2 = t01 * t34 * V(1.0/4.0);
        l3 = t012 * tm4 * V(-1.0/6.0);
        l4 = t012 * tm3 * V(1.0/24.0);
    }

    V apply(V f0, V f1, V f2, V f3, V f4) const noexcept {
        return V::fma(l4, f4, V::fma(l3, f3, V::fma(l2, f2,
               V::fma(l1, f1, l0 * f0))));
    }
#else
    V t, tm1, tm2, tm3;

    explicit NativeRootInterpolation(V x) noexcept
        : t(x), tm1(x - V(1.0)), tm2(x - V(2.0)), tm3(x - V(3.0)) {}

    V apply(V f0, V f1, V f2, V f3, V f4) const noexcept {
        return newton5_pkt<W>(f0, f1, f2, f3, f4, t, tm1, tm2, tm3);
    }
#endif
};

#endif
