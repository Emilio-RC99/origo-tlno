#include "benchmarks/common/molecule_io.hpp"
#include "core/basis/basis_general.hpp"
#include "integrals/two_electron/twoe_integrals.hpp"

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
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::duration<double, std::milli>;

    omp_set_dynamic(0);
    omp_set_num_threads(1);

    try {
        const auto molecule = input::read_inp_file(argv[1]);
        const auto basis_kind = basis::parse_basis_kind(argv[2]);
        const auto ao_basis =
            basis::build_basis_general(input::molecule_spec(molecule), basis_kind);

        twoe::EriPackedBuildOptions options;
        options.schwarz_threshold = 0.0;

        const auto plan_start = Clock::now();
        const auto plan =
            twoe::build_eri_packed_plan_for_G_rys(ao_basis.basis, options);
        const auto plan_end = Clock::now();

        const auto eri_start = Clock::now();
        const auto result = twoe::build_eri_packed_for_G_rys(plan, options);
        const auto eri_end = Clock::now();

        if (result.values.size() != 1) {
            std::cerr << "Expected one ERI checksum.\n";
            return 1;
        }

        std::cout << std::setprecision(17)
                  << "spherical_aos = " << plan.profile.M << '\n'
                  << "plan_ms = " << Milliseconds(plan_end - plan_start).count() << '\n'
                  << "eri_ms = " << Milliseconds(eri_end - eri_start).count() << '\n'
                  << "eri_checksum = " << result.values.front() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}