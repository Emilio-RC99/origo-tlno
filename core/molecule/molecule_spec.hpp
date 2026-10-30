#ifndef TLNO_CORE_MOLECULE_SPEC_HPP
#define TLNO_CORE_MOLECULE_SPEC_HPP

#include <array>
#include <string>
#include <vector>

namespace tlno::core::molecule {

struct MoleculeSpec {
    std::vector<std::string> symbols;
    std::vector<std::array<double, 3>> coords;
    int charge = 0;
};

} // namespace tlno::core::molecule

#endif
