#ifndef TLNO_BENCHMARKS_ONE_ELECTRON_MOLECULE_IO_HPP
#define TLNO_BENCHMARKS_ONE_ELECTRON_MOLECULE_IO_HPP

#include <array>
#include <string>
#include <vector>

#include "core/molecule/molecule_spec.hpp"

namespace tlno::benchmarks::onee {

struct MoleculeInput {
    std::vector<std::string> symbols;
    std::vector<std::array<double, 3>> coords_bohr;
    int charge = 0;
    int multiplicity = 1;
};

MoleculeInput read_xyz_file(
    const std::string& path,
    int charge = 0,
    int multiplicity = 1);

MoleculeInput read_inp_file(const std::string& path);

inline tlno::core::molecule::MoleculeSpec molecule_spec(const MoleculeInput& input) {
    return {input.symbols, input.coords_bohr, input.charge};
}

} // namespace tlno::benchmarks::onee

#endif
