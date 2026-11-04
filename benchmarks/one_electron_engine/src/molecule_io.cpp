#include "molecule_io.hpp"
#include "core/math/binary64.hpp"

#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

using tlno::core::math::is_finite_binary64;

namespace tlno::benchmarks::onee {
namespace {

std::string trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string lowercase(std::string value) {
    for (char& c : value) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return value;
}

void append_atom_line(
    MoleculeInput& molecule, 
    const std::string& line, 
    const std::string& path, 
    int line_number) {
    std::istringstream input(line);
    std::string symbol;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    if (!(input >> symbol >> x >> y >> z) ||
        !is_finite_binary64(x) || !is_finite_binary64(y) || !is_finite_binary64(z)) {
        throw std::runtime_error(
            "bad coordinate line " + std::to_string(line_number) + ": " + path);
    }
    molecule.symbols.push_back(std::move(symbol));
    molecule.coords_bohr.push_back({x, y, z});
}

} // namespace

MoleculeInput read_xyz_file(
    const std::string& path, 
    int charge, 
    int multiplicity) {
    if (multiplicity < 1) {
        throw std::runtime_error("XYZ multiplicity must be at least one.");
    }

    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open XYZ file: " + path);
    }
    
    std::string line;
    if (!std::getline(input, line)) {
        throw std::runtime_error("empty XYZ file: " + path);
    }

    int atom_count = 0;
    try {
        std::size_t parsed = 0;
        const std::string count_text = trim(line);
        atom_count = std::stoi(count_text, &parsed);
        if (parsed != count_text.size()) {
            throw std::invalid_argument("trailing text");
        }
    } catch (...) {
        throw std::runtime_error("XYZ line 1 must contain the atom count: " + path);
    }
    if (atom_count < 1) {
        throw std::runtime_error("XYZ atom count must be at least one: " + path);
    }
    if (!std::getline(input, line)) {
        throw std::runtime_error("XYZ file is missing its comment line: " + path);
    }

    MoleculeInput molecule;
    molecule.charge = charge;
    molecule.multiplicity = multiplicity;
    molecule.symbols.reserve(static_cast<std::size_t>(atom_count));
    molecule.coords_bohr.reserve(static_cast<std::size_t>(atom_count));
    for (int atom = 0; atom < atom_count; ++atom) {
        if (!std::getline(input, line)) {
            throw std::runtime_error(
                "XYZ file ended after " + std::to_string(atom) + " atoms: " + path);
        }
        append_atom_line(molecule, line, path, atom + 3);
    }
    return molecule;
}

MoleculeInput read_inp_file(const std::string & path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot open input file: " + path);
    }
    
    MoleculeInput molecule;
    bool found_xyz_block = false;
    bool closed_xyz_block = false;
    int line_number = 0;
    std::string line;
    while (std::getline(input, line)) {
        ++line_number;
        const std::string text = trim(line);
        if (text.empty()) {
            continue;
        }

        if (!found_xyz_block) {
            if (text.front() != '*') {
                continue;
            }
            std::istringstream header(text);
            char star = '\0';
            std::string keyword;
            int charge = 0;
            int multiplicity = 0;
            if (header >> star >> keyword >> charge >> multiplicity;
                star == '*' && lowercase(keyword) == "xyz") {
                if (multiplicity < 1) {
                    throw std::runtime_error(
                        "xyz multiplicity must be at least one: " + path);
                }
                molecule.charge = charge;
                molecule.multiplicity = multiplicity;
                found_xyz_block = true;
            }
            continue;
        }

        if (text.front() == '*') {
            closed_xyz_block = true;
            break;
        }
        if (text.front() == '#' || text.front() == '!') {
            continue;
        }
        append_atom_line(molecule, text, path, line_number);
    }

    if (!found_xyz_block) {
        throw std::runtime_error(
            "Missing '* xyz charge multiplicity' block in input: " + path);
    }
    if (!closed_xyz_block) {
        throw std::runtime_error("Unterminated xyz block: " + path);
    }
    if (molecule.symbols.empty()) {
        throw std::runtime_error("Empty xyz block: " + path);
    }
    return molecule;
}

} // tlno::benchmarks::onee
