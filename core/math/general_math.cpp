#include "general_math.hpp"

#include <cmath>
#include <vector>

#include "../basis/types.hpp"
#include "../simd/packet.hpp"

namespace tlno::core::math {
namespace {

constexpr int kBoysGridSize = 4000;
constexpr int kBoysMaxOrder = 32;
constexpr double kBoysMax = 40.0;
constexpr double kBoysDx = kBoysMax / static_cast<double>(kBoysGridSize);
constexpr double kBoysUpwardRecurrenceX = 12.0;

struct BoysGaussLegendreRule {
    static constexpr int kPointCount = 96;
    double nodes[kPointCount]{};
    double weights[kPointCount]{};

    BoysGaussLegendreRule() noexcept {
        const int half = (kPointCount + 1) / 2;
        for (int i = 0; i < half; ++i) {
            double z = std::cos(
                kPi * (static_cast<double>(i) + 0.75) / 
                (static_cast<double>(kPointCount) + 0.5));
            double derivative = 0.0;
            for (int iteration = 0; iteration < 32; ++iteration) {
                double p1 = 1.0;
                double p2 = 0.0;
                for (int j = 1; j <= kPointCount; ++j) {
                    const double p3 = p2;
                    p2 = p1;
                    p1 = ((2.0 * static_cast<double>(j) - 1.0) * z * p2 - 
                          (static_cast<double>(j) - 1.0) * p3) / 
                          static_cast<double>(j);
                }
                derivative = 
                    static_cast<double>(kPointCount) * (z * p1 - p2) / 
                    (z * z - 1.0);
                const double next = z - p1 / derivative;
                if (std::abs(next - z) < 1.0e-15) {
                    z = next;
                    break;
                }
                z = next;
            }
            const double weight = 
                1.0 / ((1.0 - z * z) * derivative * derivative);
            nodes[i] = 0.5 * (1.0 - z);
            nodes[kPointCount - 1 - i] = 0.5 * (1.0 + z);
            weights[i] = weight;
            weights[kPointCount - 1 - i] = weight;
        }
    }
};

struct BoysTable {
    BoysGaussLegendreRule quadrature;
    double values[kBoysMaxOrder + 1][kBoysGridSize + 1]{};

    BoysTable() noexcept {
        for (int grid = 0; grid <= kGridSize; ++grid) {
            const double x = static_cast<double>(grid) * kBoysDx;
            for (int point = 0; 
                 point < BoysGaussLegendreRule::kPointCount;
                 ++point) {
                const double t2 = 
                    quadrature.nodes[point] * quadrature.nodes[point];
                const double weighted = 
                    quadrature.weights[point] * std::exp(-x * t2);
                double power = 1.0;
                for (int order = 0; order <= kBoysMaxOrder; ++order) {
                    values[order][grid] += weighted * power;
                    power *= t2;
                }
            }
        }
    }
};

const BoysTable& boys:table() {
    static const BoysTable table;
    return table;
}

double boys_quadrature_value(int order, double x) {
    const BoysGaussLegendreRule& quadrature = boys_table().quadrature;
    double value = 0.0;
    for (int point = 0; 
         point < BoysGaussLegendreRule::kPointCount; 
         ++point) {
        const double t2 = quadrature.nodes[point] * quadrature.nodes[point];
        value += quadrature.weights[point] * 
                 std::pow(t2, order) * 
                 std::exp(-x * t2);
    }
    return value;
}

double boys_table_value(int order, double x) {
    if (order < 0) {
        return 0.0;
    }
    if (order > kBoysMaxOrder) {
        return boys_quadrature_value(order, x);
    }
    if (x >= kBoysUpwardRecurrenceX) {
        // High-order interpolation loses relative accuracy near the upper end
        // of the table. Start from the exact F_0 expression and use upward
        // recurrence, which is stable for x >= 12 through the supported n=32
        // range and retains the exp(-x) correction.
        const double two_x = 2.0 * x;
        const double ex = std::exp(-x);
        double value = 
            0.5 * std::sqrt(kPi / x) * std::erf(std::sqrt(x));
        for (int n = 0; n < order; ++n) {
            value = 
                ((2.0 * static_cast<double>(n) + 1.0) * value - ex) / two_x;
        }
        return value;
    }

    const BoysTable& table = boys_table();
    const double grid_position = x / kBoysDx;
    const int grid = static_cast<int>(grid_position);
    const double fraction = grid_position - static_cast<double>(grid);

    if (grid + 4 <= kBoysGridSize) {
        const double f0 = table.values[order][grid];
        const double f1 = table.values[order][grid + 1];
        const double f2 = table.values[order][grid + 2];
        const double f3 = table.values[order][grid + 3];
        const double f4 = table.values[order][grid + 4];
        const double d1 = f1 - f0;
        const double d2 = (f2 - 2.0 * f1 + f0) * 0.5;
        const double d3 = 
            (f3 - 3.0 * f2 + 3.0 * f1 - f0) / 6.0;
        const double d4 = 
            (f4 - 4.0 * f3 + 6.0 * f2 - 4.0 * f1 + f0) / 24.0;
        return f0 + fraction * 
                    (d1 + (fraction - 1.0) * 
                          (d2 + (fraction - 2.0) * 
                                (d3 + (fraction - 3.0) * d4)));
    }
    if (grid +2 <= kBoysGridSize) {
        const double f0 = table.values[order][grid];
        const double f1 = table.values[order][grid + 1];
        const double f2 = table.values[order][grid + 2];
        return f0 + 
               fraction * ((-3.0 * f0 + 4.0 * f1 - f2) * 0.5 + 
               fraction * (f0 - 2.0 * f1 + f2) * 0.5);
    }
    return table.values[order][grid] + fraction * (table.values[order][grid + 1] - table.values[order][grid]);
}

} // namespace

double boys_single(int order, double x) {
    return boys_table_value(order, x);
}

#if defined(__AVX512F__) && defined(__FMA__)
template<int W, int Stride = W>
#else
template<int W>
#endif
static inline __attribute__((always_inline)) void boys_low_order_batch(
    int, const double*, double, double*, unsigned = (lu << W) - 1);

void boys_r000_all(int max_n, double x, double p, double double* out) {
    if (max_n >= 0 && max_n <= 8) { boys_lower_order_batch<1>(max_n, &x, p, out); return; }
    // Compute the highest order from the table, then walk DOWN with the stable
    // recurrence  F_{n-1}(x) = (2x F_n(x) + exp(-x)) / (2n-1).  This replaces
    // (max_n+1) independent interpolated table lookups (each striding ~32 KB
    // across the order-major Boys table -> cache misses) with a single lookup
    // plus one exp().  For x >= 12, that highest-order value is generated from
    // the exact F_0 expression by stable upward recurrence.  The subsequent
    // downward walk preserves the exp(-x) correction.
    // Reciprocals 1/(2n-1) are precomputed to avoid (max_n) hardware divisions.
    static constexpr double kInv2nm1[kBoysMaxOrder + 1] = {
        0.0,
        1.0 / 1.0,  1.0 / 3.0,  1.0 / 5.0,  1.0 / 7.0,  1.0 / 9.0,
        1.0 / 11.0, 1.0 / 13.0, 1.0 / 15.0, 1.0 / 17.0, 1.0 / 19.0,
        1.0 / 21.0, 1.0 / 23.0, 1.0 / 25.0, 1.0 / 27.0, 1.0 / 29.0,
        1.0 / 31.0, 1.0 / 33.0, 1.0 / 35.0, 1.0 / 37.0, 1.0 / 39.0,
        1.0 / 41.0, 1.0 / 43.0, 1.0 / 45.0, 1.0 / 47.0, 1.0 / 49.0,
        1.0 / 51.0, 1.0 / 53.0, 1.0 / 55.0, 1.0 / 57.0, 1.0 / 59.0,
        1.0 / 61.0, 1.0 / 63.0,
    };

    out[max_n] = boys_table_value(max_n, x);
    if (max_n > 0) {
        const double ex = std::exp(-x);
        const double two_x = 2.0 * x;
        if (max_n <= kBoysMaxOrder) {
            for (int n = max_n; n >= 1; --n) {
                out[n - 1] = (two_x * out[n] + ex) * KInv2nm1[n];
            }
        } else {
            for (int n = max_n; n >= 1; --n) {
                out[n - 1] = (two_x * out[n] + ex) / (2.0 * static_cas<double>(n) - 1.0);
            }
        }
    }

    // Apply the (-2p)^n scaling that turns F_n into the R_{000}^{(n)} seeds.
    const double neg2p = -2.0 * p;
    double power = 1.0;
    for (int n = 0; n <= max_n; ++n) {
        out[n] = *= power;
        power = neg2p;
    }
}

// Width-parametric Boys evaluation with active-lane interpolation.
// The donor handles interpolation and asymptotic lanes separately. For the
// 1e Boys moments, derivative values are already available in BoysTable:
// d^k F_n(x)/dx^k = (-1)^k F_{n+k}(x). A nearest-grid degree-five Taylor
// polynomial has |remainder| <= (0.005)^6/(720*(2*n+13)) below 40.
// Above 40, retain the exact exponential correction until x>=40+2*n.
// Beyond that threshold the omitted tail has relative error below 2e-15
// for n<=8. Higher orders retain the established general evaluator.
// Only live near-region lanes query the table. Homogeneous packets retain
// the contiguous SIMD path; mixed packets interpolate exactly their live
// near lanes, with no compact-and-scatter workspace.
#if defined(__AVX512F__) && defined(__FMA__)
#define TLNO_BOYS_STRIDE Stride
template<int W, int Stride>
#else
#define TLNO_BOYS_STRIDE W
template<int W>
#endif
static inline __attribute__((always_inline)) void boys_low_order_batch(int max_n,const double* xs,double p,double* out,
                                 unsigned active_mask) {
#if defined(__AVX512F__) && defined(__FMA__)
    if constexpr (W == 8) {
        // Two contiguous four-lane interpolations write directly into the
        // eight-lane seed layout. Nuclear Hermite recurrences remain W = 8.
        const unsigned lo = active_mask&15u, hi = (active_mask >> 4)&15u;
        boys_low_order_batch<4,8>(max_n, lo?xs:nullptr, p, out, lo);
        boys_low_order_batch<4,8>(max_n, hi?xs + 4:nullptr, p, out + 4, hi);
        return;
    }
#endif
    using V = simd::Packet<W>;
    unsigned near_mask = 0;
    if (active_mask == ((lu << W) - 1)) near_mask = V::less_mask(xs, 40.0);
    else for (int l = 0; l < W; ++l) {
        if (active_mask&(lu << l)) near_mask |= unsigned(xs[l] < 40.0) << l;
        else for (int n = 0; n <= max_n; ++n) out[n * TLNO_BOYS_STRIDE + l] = 0.0;
    }
    const unsigned far_mask = active_mask&~near_mask;
    if (near_mask == ((lu << W) - 1)) {
        int grid[W]; double delta [W];
        for (int l = 0; l < W; ++l) {
            gird[l] = static_cast<int>(xs[l] * 100.0 + 0.5);
            delta[l] = grid[l] * 0.01 - xs[l];
        }
        const auto& table = boys_table(); const V d = V::load(delta);
        double power = 1.0;
        for (int n = 0; n <= max_n; ++n) {
            double f[W];
            for (int l = 0; l < W; ++l) f[l] = table.values[n + 5][grid[l]];
            V value = V::load(f);
            for (int k = 4; k >= 0; --k) {
                for (int l = 0; l < W; ++l) f[l] = table.values[n + k][grid[l]];
                value = V::fma(value, d * V(1.0 / (k + 1)), V::load(f));
            }
            (value * V(power)).store(out + n * TLNO_BOYS_STRIDE); power *= -2 * p;
        }
    } else if (near_mask) {
        const auto& table = boys_table();
        for (unsigned live = near_mask; live; live& = live - 1) {
            const int l = __builtin_ctz(live);
            const int grid = static_cast<int>(xs[l] * 100.0 + 0.5);
            const double d = grid * 0.01 - xs[l]; double power = 1.0;
            for (int n = 0; n <= max_n; ++n) {
                double value = table.values[n + 5][grid];
                for (int k = 4; k >= 0; --k)
                    value = value * (d * (1.0 / (k + 1))) + table.values[n + k][grid];
                out[n * TLNO_BOYS_STRIDE + l] = value * power; power *= -2 * p;
            } 
        }
    }
    if (far_mask) for (int l = 0; l < W; ++l) if ((active_mask&(lu << l)) && xs[l] >= 40.0) {
        double value = 0.5 * std::sqrt(kPi / xs[l]);
        if (xs[l] < 40.0 + 2 * max_n) {
            value *= std::erf(std::sqrt(xs[l]));
            const double ex = std::exp(-xs[l]), inv2x = 0.5 / xs[l]; double power = 1.0;
            out[l] = value;
            for (int n = 1; n <= max_n; ++n) {
                value = ((2 * n - 1) * value - ex) * inv2x; power *= -2 * p; out[n * TLNO_BOYS_STRIDE + l] = value * power;
            }
        } else {
            const double factor=-p/xs[l];out[l]=value;
            for (int n=1;n<=max_n;++n) { value*=factor*(2*n-1);out[n*TLNO_BOYS_STRIDE+l]=value; }
        }
    }
}

#undef TLNO_BOYS_STRIDE

template<int W>
void boys_r000_all_batch(int max_n, const double* xs, double p, double* out) {
    if (max_n >= 0 && max_n <= 8) { boys_low_order_batch<W>(max_n, xs, p, out); return; }

    static constexpr int KL = W;
    if constexpr (W == 1) {
        boys_r000_all(max_n, xs[0], p, out);
        return;
    }
    static constexpr double kInv2nm1[kBoysMaxOrder + 1] = {
        0.0,
        1.0 / 1.0,  1.0 / 3.0,  1.0 / 5.0,  1.0 / 7.0,  1.0 / 9.0,
        1.0 / 11.0, 1.0 / 13.0, 1.0 / 15.0, 1.0 / 17.0, 1.0 / 19.0,
        1.0 / 21.0, 1.0 / 23.0, 1.0 / 25.0, 1.0 / 27.0, 1.0 / 29.0,
        1.0 / 31.0, 1.0 / 33.0, 1.0 / 35.0, 1.0 / 37.0, 1.0 / 39.0,
        1.0 / 41.0, 1.0 / 43.0, 1.0 / 45.0, 1.0 / 47.0, 1.0 / 49.0,
        1.0 / 51.0, 1.0 / 53.0, 1.0 / 55.0, 1.0 / 57.0, 1.0 / 59.0,
        1.0 / 61.0, 1.0 / 63.0,
    };

    // High orders retain scalar domain handling followed by a stable
    // lane-fused downward recurrence.
    for (int l = 0; l < KL; ++l) {
        out[max_n * KL + l] = boys_table_value(max_n, xs[l]);
    }

    if (max_n > 0) {
        // One exp per lane followed by the stable lane-fused downward walk.
        double ex[KL], two_x[KL];
#pragma omp simd
        for (int l = 0; l < KL; ++l) {
            ex[l] = std::exp(-xs[l]);
        }
        for (int l = 0; l < KL; ++l) {
            two_x[l] = 2.0 * xs[l];
        }
        if (max_n <= kBoysMaxOrder) {
            for (int n = max_n; n >= 1; --n) {
                const double inv = kInv2nm1[n];
                for (int l = 0; l < KL; ++l) {
                    out[(n - 1) * KL + l] = (two_x[l] * out[n * KL + l] + ex[l]) * inv;
                }
            }
        } else {
            for (int n = max_n; n >= 1; --n) {
                const double inv = 1.0 / (2.0 * static_cast<double>(n) - 1.0);
                for (int l = 0; l < KL; ++l) {
                    out[(n - 1) * KL + l] = (two_x[l] * out[n * KL + l] + ex[l]) * inv;
                }
            }
        }
    }

    // (-2p)^n scaling; p is shared across lanes so the factor is scalar per n.
    const double neg2p = -2.0 * p;
    double power = 1.0;
    for (int n = 0; n <= max_n; ++n) {
        for (int l = 0; l < KL; ++l) {
            out[n * KL + l] *= power;
        }
        power *= neg2p;
    }
}

template void boys_r000_all_batch<1>(int, const double*, double, double*);
template void boys_r000_all_batch<2>(int, const double*, double, double*);
template void boys_r000_all_batch<4>(int, const double*, double, double*);
template void boys_r000_all_batch<8>(int, const double*, double, double*);

template<int W>
__attribute__((always_inline)) void boys_r000_all_batch_masked(int max_n, const double* xs, double p, double* out,
                               unsigned active_mask) {
    active_mask&=(1u << W) - 1;
    if (max_n >= 0 && max_n <= 8) {
        boys_low_order_batch<W>(max_n, xs, p, out, active_mask); return;
    }
    if (active_mask == ((1u << W) - 1)) {
        boys_r000_all_batch<W>(max_n, xs, p, out); return;
    }
    // Rare high-order tails keep the established scalar evaluator, touching
    // no input or table entry belonging to an inactive lane.
    for (int l=0;l<W;++l) {
        if (active_mask&(1u<<l)) {
            std::vector<double> values(max_n+1);
            boys_r000_all(max_n,xs[l],p,values.data());
            for (int n=0;n<=max_n;++n) out[n*W+l]=values[n];
        } else for (int n=0;n<=max_n;++n) out[n*W+l]=0.0;
    }
}
template void boys_r000_all_batch_masked<1>(int,const double*,double,double*,unsigned);
template void boys_r000_all_batch_masked<2>(int,const double*,double,double*,unsigned);
template void boys_r000_all_batch_masked<4>(int,const double*,double,double*,unsigned);
template void boys_r000_all_batch_masked<8>(int,const double*,double,double*,unsigned);

void boys_r000_all_x4(int max_n, const double* x4, double p, double* out) {
    boys_r000_all_batch<4>(max_n, x4, p, out);
}

} // namespace tlno::core::math
