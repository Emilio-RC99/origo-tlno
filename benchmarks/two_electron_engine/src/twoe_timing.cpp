#include "benchmarks/one_electron_engine/src/molecule.io.hpp"
#include "core/basis/basis_general.hpp"

#include <omp.h>
#include <chrono>
#include <exception>
#include <iomanip>
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: twoe_timing INPUT.inp BASIS\n";
        return 1;
    }

    namespace basis = tlno::core::basis;
    namespace input = tlno::benchmarks::onee;
    namespace twoe = tlno::integrals::twoe;
    namespace 
}