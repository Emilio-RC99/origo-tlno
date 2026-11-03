// Inactive inputs may be invalid; every inactive output must be overwritten.
#include "core/math/general_math.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

// A bit test remains meaningful when the production evaluator uses fast-math.
bool finite_bits(double x) {
    std::uint64_t bits;std::memcpy(&bits, &x, sizeof bits);
    return (bits & UINT64_C(0x7ff0000000000000)) != UINT64_C(0x7ff0000000000000);
}
template<int W> std::size_t check() {
    std::size_t checked=0;
    const double cases[]={0., .005, 39.999999999, 40., 40.000000001, 43.99999999, 56., 1000.};
    for (int n : {0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 32}) for (int rotate = 0; rotate < 8; ++rotate)
        for (unsigned mask = 0; mask < (1u << W); ++mask) {
            double xs[W], safe[W];
            for (int l = 0; l < W; ++l) {
                safe[l] = cases[(l + rotate) % 8];
                xs[l] = (mask&(1u << l)) ? safe[l] : std::numeric_limits<double>::quiet_NaN();
            }
            std::vector<double> out((n + 1) * W + 2, 12345.), ref((n + 1) * W);
            tlno::core::math::boys_r000_all_batch<W>(n, safe, .75, ref.data());
            tlno::core::math::boys_r000_all_batch_masked<W>(n, mask ? xs : nullptr, .75, out.data() + 1, mask);
            if(out.front() != 12345. || out.back() != 12345.) throw std::runtime_error("Output boundary");
            for(int m = 0; m <= n; ++m) for(int l = 0; l < W; ++l) {
                double got = out[1 + m * W + l], expected = (mask&(1u << l)) ? ref[m * W + l] : 0.;
                if(!finite_bits(got) || std::abs(got - expected) > 2e-13*(1 + std::abs(expected)))
                    throw std::runtime_error("Masked Boys moment differs");
                if(!(mask&(1u << l)) && got != 0.) throw std::runtime_error("Inactive output not zero");
                ++checked;
            }
        }
    return checked;
}
int main() {
    try {
        auto count = check<1>() + check<2>() + check<4>() + check<8>();
        std::cout << "PASS: " << count << " masked moments; all masks, inactive NaNs/null, bounds, orders through 32\n";
    } catch(const std::exception& e) {std::cerr << "FAIL: " << e.what() << '\n';return 1;}
}
