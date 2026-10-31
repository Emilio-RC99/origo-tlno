#ifndef TLNO_CORE_MATH_GENERAL_MATH_HPP
#define TLNO_CORE_MATH_GENERAL_MATH_HPP

namespace tlno::core::math {

double boys_single(int order, double x);
void boys_r000_all(int max_n, double x, double p, double* out);
// Width-parametric seeds, n-major/lane-minor; W = 1, 2, 4 or 8.
// W=1 uses the scalar reference; wider packets preserve its table and domains.
template<int W>
void boys_r000_all_batch(int max_n, const double* xs, double p, double* out);
// Inactive lanes are not evaluated and all their output moments are zero.
// Bits 0..W-1 select physical, nonzero-charge centers in a nuclear packet.
template<int W>
void boys_r000_all_batch_masked(int max_n, const double* xs, double p, 
                                double* out, unsigned active_mask);

// Four-lane variant for the blocked nucleus loop: computes the scaled
// R_000^(n) seeds for 4 x-values sharing one p.  Output layout is n-major,
// lane-minor: out[n*4 + lane], so the per-n recursion and scaling loops
// vectorize (one exp / one FMA per 4 lanes instead of 4 scalar passes).
void boys_r000_all_x4(int max_n, const double* x4, double p, double* out);

} // namespace tlno::core::math

#endif
