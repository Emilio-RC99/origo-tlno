#!/usr/bin/env python3
"""Validate the local one-electron engine against independent Boys/PySCF values."""

from __future__ import annotations

import argparse
import csv
import math
import random
import subprocess
from pathlib import Path

import numpy as np

from benchmark_molecule_suite import (
    BASIS_PAIRS,
    BENCH,
    DEFAULT_LIB,
    load_tlno,
    make_pyscf_molecule,
    pin_threads,
    pyscf_basis_spec,
    read_geometry,
    resolve_molecules,
    tlno_copy,
)


ROOT = Path(__file__).resolve().parents[3]


def boys_reference(order : int, x: float) -> float:
    if x < 0.5: 
        total = 0.0
        power_over_factorial = 1.0
        for k in range(256):
            term = power_over_factorial / float(2 * order + 2 * k + 1)
            total += term
            if k > 0 and abs(term) <= 1.0e-18 * abs(total):
                break
            power_over_factorial *= -x / float(k + 1)
        return total
    from scipy.special import gamma, gammainc

    return (
        0.5
        * gamma(order + 0.5)
        * gammainc(order + 0.5, x)
        / x ** (order + 0.5)
    )


def boys_x_grid(seed: int) -> list[float]:
    fixed = [
        0.0, 1.0e-16, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8,
        1.0e-6, 1.0e-4, 1.0e-3, 1.0e-2, 0.1, 1.0, 10.0,
        39.99, 40.0, 40.01, 100.0, 1000.0, 10000.0,
    ]
    logarithmic = np.logspace(-16, 4, 201).tolist()
    transition = np.linspace(39.5, 40.5, 201).tolist()
    rng = random.Random(seed)
    randomized = [10.0 ** rng.uniform(-16.0, 4.0) for _ in range(512)]
    return sorted(set(fixed + logarithmic + transition + randomized))


def validate_boys(
    probe: Path, *, seed: int, out: Path
) -> tuple[float, tuple[int, float, float, float]]:
    cases = [
        (order, x)
        for order in range(33)
        for x in boys_x_grid(seed)
    ]
    payload = "".join(f"{order} {x:.17g}\n" for order, x in cases)
    completed = subprocess.run(
        [str(probe)],
        input = payload,
        text = True,
        capture_output = True,
        check = True,
    )
    values = [float(line) for line in completed.stdout.splitlines()]
    if len(values) != len(cases):
        raise RuntimeError("Boys probe returned an incomplete batch")

    rows = []
    worst = 0.0
    worst_case = (0, 0.0, 0.0, 0.0)
    for (order, x), got in zip(cases, values):
        if not math.isfinite(got):
            raise RuntimeError(f"Nonfinite Boys value: n = {order}, x = {x}, got = {got}")
        reference = boys_reference(order, x)
        if not math.isfinite(reference) or reference <= 0:
            raise RuntimeError(f"Invalid Boys reference: n = {order}, x = {x}, value = {reference}")
        absolute = abs(got - reference)
        relative = absolute / abs(reference)
        if not math.isfinite(absolute) or not math.isfinite(relative):
            raise RuntimeError(f"Nonfinite Boys error: n = {order}, x = {x}")
        rows.append({
            "order": order,
            "x": x,
            "tlno": got,
            "reference": reference,
            "absolute_error": absolute,
            "relative_error": relative,
        })
        if relative > worst:
            worst = relative
            worst_case = (order, x, got, reference)
    out.parent.mkdir(parents = True, exist_ok = True)
    with out.open("w", newline = "") as stream:
        writer = csv.DictWriter(stream, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Boys: {len(rows)} points, max relative error {worst:.3e} "
        f"at n = {worst_case[0]}, x = {worst_case[1]:.17g}"
    )
    return worst, worst_case


def normalized_pyscf_matrices(
    charge: int,
    multiplicity: int,
    atoms: tuple[str, ...],
    basis,
    *,
    spherical: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from pyscf import lib

    lib.num_threads(1)
    molecule = make_pyscf_molecule(atoms, basis, charge, multiplicity,
                                   spherical = spherical)
    suffix = "sph" if spherical else "cart"
    overlap = molecule.intor(f"int1e_ovlp_{suffix}", hermi = 1)
    scale = 1.0 / np.sqrt(np.diag(overlap))
    return tuple(
        scale[:, None] * molecule.intor(name, hermi = 1) * scale[None, :]
        for name in (f"int1e_ovlp_{suffix}", f"int1e_kin_{suffix}", f"int1e_nuc_{suffix}")
    )


def matrix_errors(
    lhs: tuple[np.ndarray, ...], rhs: tuple[np.ndarray, ...]
) -> tuple[list[float], list[float]]:
    if len(lhs) != 3 or len(rhs) != 3:
        raise RuntimeError("Expected three S/T/V matrices")
    if any(left.ndim != 2 or right.ndim != 2 or left.shape != right.shape
           or left.shape[0] == 0 or left.shape[0] != left.shape[1]
           for left, right in zip(lhs, rhs)):
        raise RuntimeError("Matrix shape mismatch")
    if any(not np.isfinite(matrix).all() for matrix in (*lhs, *rhs)):
        raise RuntimeError("Nonfinite matrix value")
    absolute = [
        float(np.max(np.abs(left - right)))
        for left, right in zip(lhs, rhs)
    ]
    relative_frobenius = [
        float(np.linalg.norm(left - right) / max(np.linalg.norm(right), 1.0e-300))
        for left, right in zip(lhs, rhs)
    ]
    if not all(math.isfinite(value) for value in (*absolute, *relative_frobenius)):
        raise RuntimeError("Nonfinite matrix error")
    return absolute, relative_frobenius


def validate_matrices(
    lib_path: Path,
    *,
    molecules: str,
    bases: list[str],
    out: Path,
) -> tuple[float, float]:
    pin_threads(1)
    lib = load_tlno(lib_path)
    rows = []
    worst_external = 0.0
    worst_screen = 0.0
    for path in resolve_molecules(molecules):
        charge, multiplicity, atoms = read_geometry(path)
        for basis in bases:
            reference = normalized_pyscf_matrices(
                charge, multiplicity, atoms, pyscf_basis_spec(basis, lib = lib, atoms = atoms)
            )
            handle = lib.tlno_onee_create_from_inp(
                str(path).encode(), basis.encode()
            )
            if not handle:
                raise RuntimeError(f"TLNO handle failed: {path} {basis}")
            nbf = lib.tlno_onee_nbf(handle)

            try:
                lib.tlno_onee_set_screening(0)
                lib.tlno_onee_eval_full(handle)
                unscreened = tuple(tlno_copy(lib, handle, nbf))
                lib.tlno_onee_set_screening(1)
                lib.tlno_onee_eval_full(handle)
                screened = tuple(tlno_copy(lib, handle, nbf))
            finally:
                lib.tlno_onee_destroy(handle)
                lib.tlno_onee_set_screening(1)

            uns_abs, uns_rel = matrix_errors(unscreened, reference)
            scr_abs, scr_rel = matrix_errors(screened, reference)
            delta_abs, delta_rel = matrix_errors(screened, unscreened)
            worst_external = max(worst_external, *uns_abs, *scr_abs)
            worst_screen = max(worst_screen, *delta_abs)
            row = {
                "molecule": path.stem,
                "basis": basis,
                "ao_count": nbf,
            }
            for index, label in enumerate(("S", "T", "V")):
                row.update({
                    f"unscreened_{label}_max_abs": uns_abs[index],
                    f"unscreened_{label}_rel_fro": uns_rel[index],
                    f"screened_{label}_max_abs": scr_abs[index],
                    f"screened_{label}_rel_fro": scr_rel[index],
                    f"screening_delta_{label}_max_abs": delta_abs[index],
                    f"screening_delta_{label}_rel_fro": delta_rel[index],
                })
            rows.append(row)
            print(
                f"{path.stem:15s} {basis:9s} N = {nbf:4d} "
                f"external={max(*uns_abs, *scr_abs):.3e} "
                f"screen={max(delta_abs):.3e}",
                flush=True,
            )
    lib.tlno_onee_set_screening(1)
    out.parent.mkdir(parents = True, exist_ok = True)
    if not rows:
        raise RuntimeError("No molecule/basis cases selected")
    with out.open("w", newline = "") as stream:
        writer = csv.DictWriter(stream, fieldnames = list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"Matrices: {len(rows)} cases, worst external max error "
        f"{worst_external:.3e}, worst screening delta {worst_screen:.3e}"
    )
    return worst_external, worst_screen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--lib", type = Path, default = DEFAULT_LIB)
    parser.add_argument(
        "--boys-probe", type = Path, default = BENCH / "build" / "boys_probe"
    )
    parser.add_argument("--molecules", default = "all")
    parser.add_argument(
        "--basis",
        default = "sto3g,431g,631g,631gd,631gdp,def2svp,def2tzvp",
    )
    parser.add_argument("--matrix-tolerance", type = float, default = 2.0e-8)
    parser.add_argument("--boys-tolerance", type = float, default = 1.0e-8)
    parser.add_argument("--seed", type = int, default = 20260615)
    parser.add_argument(
        "--out-dir",
        type = Path,
        default = BENCH / "results" / "validation",
    )
    args = parser.parse_args()
    if not all(math.isfinite(value) and value > 0
               for value in (args.matrix_tolerance, args.boys_tolerance)):
        parser.error("validation tolerances must be finite and positive")
    return args


def main() -> None:
    args = parse_args()
    if not args.lib.is_file() or not args.boys_probe.is_file():
        raise SystemExit("Missing benchmark binaries; run scripts/build.sh")
    bases = [item.strip() for item in args.basis.split(",") if item.strip()]
    unknown = sorted(set(bases) - set(BASIS_PAIRS))
    if unknown:
        raise SystemExit(f"Unknown basis names: {unknown}")
    boys_error, _ = validate_boys(
        args.boys_probe,
        seed = args.seed,
        out = args.out_dir / "boys_validation.csv",
    )
    matrix_error, screening_error = validate_matrices(
        args.lib,
        molecules = args.molecules,
        bases = bases,
        out = args.out_dir / "matrix_validation.csv",
    )
    if boys_error > args.boys_tolerance:
        raise SystemExit(
            f"Boys validation failed: {boys_error:.3e} > "
            f"{args.boys_tolerance:.3e}"
        )
    if matrix_error > args.matrix_tolerance:
        raise SystemExit(
            f"Matrix validation failed: {matrix_error:.3e} > "
            f"{args.matrix_tolerance:.3e}"
        )
    if screening_error > args.matrix_tolerance:
        raise SystemExit(
            f"Screening validation failed: {screening_error:.3e} > "
            f"{args.matrix_tolerance:.3e}"
        )
    print("PASS: Boys and selected molecule/basis matrix validation")


if __name__ == "__main__":
    main()
