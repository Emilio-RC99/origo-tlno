#ifndef TLNO_CORE_MOLECULE_READER_HPP
#define TLNO_CORE_MOLECULE_READER_HPP

#include <string>

#include "molecule_spec.hpp"

namespace tlno::core::molecule {

MoleculeSpec read_inp_xyz_block(const std::string& path);

} // namespace tlno::core::molecule

#endif
