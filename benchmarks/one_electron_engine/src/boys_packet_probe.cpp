// Text protocol for independent high-precision validation of packet seeds.
#include "core/math/general_math.hpp"
#include <iomanip>
#include <iostream>
#include <vector>

template<int W> void evaluate(int n, double p, const double* x) {
    std::vector<double> out((n + 1) * W);
    tlno::core::math::boys_r000_all_batch<W>(n, x, p, out.data());
    for (double value:out) std::cout << value << ' ';
    std::cout << '\n';
}

int main() {
    std::cout << std::setprecision(17);
    int width, n; double p;
    while (std::cin >> width >> n >> p) {
        if (n < 0 || n > 32 || (width != 1 && width != 2 && width != 4 && width != 8)) return 2;
        double x[8];
        for (int i = 0; i < width; ++i) if (!(std::cin >> x[i]) || x[i] < 0) return 2;
        switch (width) {
            case 1: evaluate<1>(n, p, x);break;
            case 2: evaluate<2>(n, p, x);break;
            case 4: evaluate<4>(n, p, x);break;
            case 8: evaluate<8>(n, p, x);break;
        }
    }
    return std::cin.eof() ? 0:2;
}
