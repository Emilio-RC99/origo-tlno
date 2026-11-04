#!/usr/bin/env python3
"""Code generator for the Rys right-01 scatter maps.

For the six pure s/p classes (ss|ss), (ps|ss), (ps|ps), (ss|ps), (ss|pp) and
(pp|ss), the horizontal transfer moves at most one unit of angular momentum
from a to b and from c to d, so every output integral is a short closed-form
sum, per Cartesian direction:

    (a b|c d) = sum_u sum_v C(b, u) C(d, v) AB^(b-u) CD^(d-v) (a+u 0|c+v 0)

Each entry lists its terms as {t_index, scale, ab_x, ab_y, ab_z, cd_x, cd_y,
cd_z}: the radial intermediate, the binomial factor and the powers of AB and
CD.  The engine uses these tables instead of its block-HRR code for exactly
these classes, on the legacy four-lane route.

The map emitter below is copied unchanged from TLNOv1's
generate_rys_right01_extra_maps.py (commit ae4f2d1b); this script only
restricts it to the production classes and writes the header directly.

Run from anywhere; writes rys_right01_maps.hpp next to this script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

GUARD = "TLNO_INTEGRALS_TWO_ELECTRON_RYS_RYS_RIGHT01_MAPS_HPP"

# Production classes, in the order of the existing header.
CLASSES = [
    (0, 0, 0, 0),  # ss|ss
    (1, 0, 0, 0),  # ps|ss
    (1, 0, 1, 0),  # ps|ps
    (0, 0, 1, 0),  # ss|ps
    (0, 0, 1, 1),  # ss|pp
    (1, 1, 0, 0),  # pp|ss
]


@dataclass(frozen=True)
class Term:
    t_index: int
    scale: int
    ab_x: int
    ab_y: int
    ab_z: int
    cd_x: int
    cd_y: int
    cd_z: int


def n_cart(L: int) -> int:
    return (L + 1) * (L + 2) // 2


def n_cart_cum(L: int) -> int:
    return (L + 1) * (L + 2) * (L + 3) // 6


def offset_for_L(L: int) -> int:
    return 0 if L == 0 else L * (L + 1) * (L + 2) // 6


def triple_idx_within(L: int, lx: int, ly: int) -> int:
    dx = L - lx
    return dx * (dx + 1) // 2 + (dx - ly)


def cart_flat_idx(lx: int, ly: int, lz: int) -> int:
    L = lx + ly + lz
    return offset_for_L(L) + triple_idx_within(L, lx, ly)


def triples(L: int) -> list[tuple[int, int, int]]:
    out: list[tuple[int, int, int]] = []
    for lx in range(L, -1, -1):
        for ly in range(L - lx, -1, -1):
            lz = L - lx - ly
            out.append((lx, ly, lz))
    return out


def binom_small(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if n == 2 and k == 1:
        return 2
    if n == 3:
        return 3
    raise ValueError(f"Unexpected binomial n={n} k={k}")


def emit_map(La: int, Lb: int, Lc: int, Ld: int) -> str:
    a_tr = triples(La)
    b_tr = triples(Lb)
    c_tr = triples(Lc)
    d_tr = triples(Ld)
    nb = n_cart(Lb)
    nc = n_cart(Lc)
    nd = n_cart(Ld)
    lhs_count = len(a_tr) * len(b_tr)
    rhs_count = len(c_tr) * len(d_tr)
    entry_count = lhs_count * rhs_count
    a_offset = offset_for_L(La)
    nc_full = n_cart_cum(Lc + Ld)

    lines = [
        "template <>",
        f"struct RysRight01GeneratedMap<{La}, {Lb}, {Lc}, {Ld}> {{",
        "    static constexpr bool available = true;",
        f"    static constexpr int entry_count = {entry_count};",
        "    inline static constexpr RysGeneratedScatterEntry entries[entry_count] = {",
    ]

    for a_local, a in enumerate(a_tr):
        for b_local, b in enumerate(b_tr):
            lhs_component = a_local * nb + b_local
            assert lhs_component < lhs_count
            for c_local, c in enumerate(c_tr):
                for d_local, d in enumerate(d_tr):
                    terms: list[Term] = []
                    for ux in range(b[0] + 1):
                        bx = binom_small(b[0], ux)
                        for uy in range(b[1] + 1):
                            bxy = bx * binom_small(b[1], uy)
                            for uz in range(b[2] + 1):
                                bcoef = bxy * binom_small(b[2], uz)
                                arow = (
                                    cart_flat_idx(a[0] + ux, a[1] + uy, a[2] + uz)
                                    - a_offset
                                )
                                for vx in range(d[0] + 1):
                                    dx = binom_small(d[0], vx)
                                    for vy in range(d[1] + 1):
                                        dxy = dx * binom_small(d[1], vy)
                                        for vz in range(d[2] + 1):
                                            dcoef = dxy * binom_small(d[2], vz)
                                            cflat = cart_flat_idx(
                                                c[0] + vx, c[1] + vy, c[2] + vz
                                            )
                                            terms.append(
                                                Term(
                                                    arow * nc_full + cflat,
                                                    bcoef * dcoef,
                                                    b[0] - ux,
                                                    b[1] - uy,
                                                    b[2] - uz,
                                                    d[0] - vx,
                                                    d[1] - vy,
                                                    d[2] - vz,
                                                )
                                            )
                    rhs_component = c_local * nd + d_local
                    assert lhs_component * rhs_count + rhs_component < entry_count
                    term_text = ", ".join(
                        "{{{}, {}, {}, {}, {}, {}, {}, {}}}".format(
                            t.t_index,
                            t.scale,
                            t.ab_x,
                            t.ab_y,
                            t.ab_z,
                            t.cd_x,
                            t.cd_y,
                            t.cd_z,
                        )
                        for t in terms
                    )
                    lines.append(f"        {{{len(terms)}, {{{term_text}}}}},")

    lines.extend(["    };", "};", ""])
    return "\n".join(lines)


def render() -> str:
    head = [
        "// rys_right01_maps.hpp — GENERATED by gen_rys_right01_maps.py; DO NOT EDIT.",
        "// Compact generated scatter maps used by the six low-s/p classes that do not",
        "// take the block-HRR route.",
        f"#ifndef {GUARD}",
        f"#define {GUARD}",
        "",
    ]
    blocks = [emit_map(*cls) for cls in CLASSES]
    return "\n".join(head) + "\n" + "\n".join(blocks) + "\n#endif\n"


def main() -> None:
    out = Path(__file__).resolve().parent / "rys_right01_maps.hpp"
    out.write_text(render())
    entries = sum(
        n_cart(la) * n_cart(lb) * n_cart(lc) * n_cart(ld)
        for la, lb, lc, ld in CLASSES
    )
    print(f"wrote {out} ({len(CLASSES)} classes, {entries} entries)")


if __name__ == "__main__":
    main()
