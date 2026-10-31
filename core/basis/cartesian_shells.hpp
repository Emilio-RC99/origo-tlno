#ifndef TLNO_CORE_BASIS_CARTESIAN_SHELLS_HPP
#define TLNO_CORE_BASIS_CARTESIAN_SHELLS_HPP

#include <array>
#include <cstddef>
#include <initializer_list>
#include <stdexcept>
#include <string>

#include "types.hpp"

namespace tlno::core::basis::detail {

using CartesianTuple = std::array<int, 3>;

// Generate each table at compile time, in descending x then y order.
template<int L>
constexpr auto make_cartesian_tuples() {
    static_assert(L >= 0, "Angular momentum must be nonnegative.");
    std::array<CartesianTuple, (L + 1) * (L + 2) / 2> tuples{};
    std::size_t index = 0;
    for (int x = L; x >= 0; --x) {
        for (int y = L - x; y >= 0; --y) {
            tuples[index++] = {x, y, L - x - y};
        }
    }
    return tuples;
}

template<int L>
inline constexpr auto cartesian_tuples = make_cartesian_tuples<L>();

// A read-only view of static storage; iteration allocates nothing.
struct CartesianTupleView {
    const CartesianTuple* data;
    std::size_t size;

    constexpr const CartesianTuple* begin() const { return data; }
    constexpr const CartesianTuple* end() const { return data + size; }
};

template<int L>
constexpr CartesianTupleView cartesian_tuple_view() {
    return {cartesian_tuples<L>.data(), cartesian_tuples<L>.size()};
}

// The shared runtime interface supports s through g.
inline CartesianTupleView cartesian_tuples_for_total_L(int total_L) {
    switch (total_L) {
        case 0: return cartesian_tuple_view<0>();
        case 1: return cartesian_tuple_view<1>();
        case 2: return cartesian_tuple_view<2>();
        case 3: return cartesian_tuple_view<3>();
        case 4: return cartesian_tuple_view<4>();
        default: throw std::out_of_range("Cartesian shell helper supports angular momenta 0-4.");
    }
}

// Literal lists avoid temporary vectors; each RawShell owns its copied data.
// The caller can reserve atom.shells once for the complete atom.
inline void add_cartesian_shells(RawAtomBasis& atom, int L, int shell_index,
                                 std::initializer_list<double> exps,
                                 std::initializer_list<double> coeffs) {
    const auto tuples = cartesian_tuples_for_total_L(L);
    constexpr const char* prefixes[] = {"s", "p", "d", "f", "g"};
    const std::string suffix = std::to_string(shell_index);
    for (const auto& ang : tuples) {
        const std::string label = L == 0
            ? std::string(prefixes[L]) + suffix
            : std::string(prefixes[L]) + "_" + std::to_string(ang[0])
                + std::to_string(ang[1]) + std::to_string(ang[2]) + "_" + suffix;
        atom.shells.push_back({label, ang, exps, coeffs});
    }
}

} // namespace tlno::core::basis::detail

#endif
