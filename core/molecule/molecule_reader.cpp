#include "molecule_reader.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace tlno::core::molecule {
namespace {

std::string to_lower_copy(std::string text) {
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return text;
}

} // namespace

MoleculeSpec read_inp_xyz_block(const std::string& path) {
    std::ifstream input_file(path);
    if (!input_file) {
        throw std::runtime_error(
            "Cannot open input file: " + path);
    }

    MoleculeSpec molecule;
    bool found_xyz = false;
    std::string text;
    while (std::getline(input_file, text)) {
        while (!text.empty() && 
                std::isspace(static_cast<unsigned char>(text.front()))) {
            text.erase(text.begin());
        }
        while (!text.empty() && 
                std::isspace(static_cast<unsigned char>(text.back()))) {
            text.pop_back();
        }
        if (text.empty()) continue;

        if (!found_xyz) {
            if (text[0] != '*') continue;
            std::istringstream fields(text);
            char star = '\0';
            std::string key;
            if (!(fields >> star >> key)) continue;
            if (star == '*' && to_lower_copy(key) == "xyz") {
                int multiplicity = 0;
                if (!(fields >> molecule.charge >> multiplicity) ||
                    multiplicity <= 0) {
                    throw std::runtime_error(
                        "Invalid '* xyz charge multiplicity' header in "
                        "input: " + path);
                }
                found_xyz = true;
            }
            continue;
        }

        if (text[0] == '*') {
            break;
        }
        if (text[0] == '#' || text[0] == '!') continue;

        std::istringstream fields(text);
        std::string symbol;
        double x = 0.0;
        double y = 0.0;
        double z = 0.0;
        if (!(fields >> symbol >> x >> y >> z)) {
            throw std::runtime_error(
                "Bad atom line in xyz block: " + path);
        }
        molecule.symbols.push_back(symbol);
        molecule.coords.push_back({x, y, z});
    }

    if (!found_xyz) {
        throw std::runtime_error(
            "Missing '* xyz charge mult' block in input: " + path);
    }
    if (molecule.symbols.empty()) {
        throw std::runtime_error(
            "Empty xyz block in input: " + path);
    }
    return molecule;
}

} // namespace tlno::core::molecule
