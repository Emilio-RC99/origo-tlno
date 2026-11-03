#include "core/math/general_math.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>

int main(int argc, char** argv) {
    std::cout << std::setprecision(17);
    if (argc == 3) {
        const int order = std::atoi(argv[1]);
        const double x = std::atof(argv[2]);
        std::cout << tlno::core::math::boys_single(order, x) << '\n';
        return 0;
    }
    if (argc == 1) {
        int order = 0;
        double x = 0.0;
        while (std::cin >> order >> x) {
            std::cout << tlno::core::math::boys_single(order, x) << '\n';
        }
        return std::cin.eof() ? 0 : 2;
    }
    return 2;
}
