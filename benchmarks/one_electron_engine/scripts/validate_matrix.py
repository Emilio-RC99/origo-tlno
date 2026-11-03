#!/usr/bin/env python3
"""Validate C ABI representation/scope switches, PySCF values and optional parity.

Production representation-switch validator. The current
library can be checked independently; --previous enables an additional explicit
before/after check and requires compatible basis parameters in both libraries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from benchmark_molecule_suite import (
    BENCH,
    DEFAULT_LIB,
    TEST_MOLECULES,
    load_tlno,
    pin_threads,
    pyscf_basis_spec,
    read_geometry,
    tlno_copy,
)


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--lib", type = Path, default = DEFAULT_LIB)
    parser.add_argument("--previous", type = Path,
                        help = "optional compatible reference C ABI library")
    parser.add_argument("--threads", type = int, default = 1)
    parser.add_argument("--cases", default = "h2o:631gdp,gly:def2tzvp,benzene:def2tzvp",
                        help = "comma-separated molecule:basis pairs")
    parser.add_argument("--molecule-dir", type = Path, default = TEST_MOLECULES)
    parser.add_argument("--matrix-tolerance", type = float, default = 2e-8)
    parser.add_argument("--out", type = Path,
                        default = BENCH / "results" / "validation" / "matrix_api.json")
    args = parser.parse_args()
    if not 1 <= args.threads <= 3:
        parser.error("threads must be 1..3")
    if not math.isfinite(args.matrix_tolerance) or args.matrix_tolerance <= 0:
        parser.error("Matrix tolerance must be finite and positive")
    try:
        cases = [tuple(case.strip().split(":")) for case in args.cases.split(",")]
        if not cases or any(len(case) != 2 or not all(case) for case in cases):
            raise ValueError
    except ValueError:
        parser.error("Cases must be comma-separated molecule:basis pairs")
    pin_threads(args.threads)

    import numpy as np
    from pyscf import lib as pyscf_lib
    from validate import matrix_errors, normalized_pyscf_matrices

    current = load_tlno(args.lib.resolve())
    previous = load_tlno(args.previous.resolve()) if args.previous else None
    libraries = [current] if previous is None else [current, previous]
    rows = []
    try: 
        for molecule, basis in cases:
            path = args.molecule_dir / (molecule + ".inp")
            charge, multiplicity, atoms = read_geometry(path)
            exact_basis = pyscf_basis_spec(basis, lib = current, atoms = atoms)
            independent = {
                spherical: normalized_pyscf_matrices(
                    charge, multiplicity, atoms, exact_basis, spherical = spherical
                )
                for spherical in (False, True)
            }
            # PySCF and the C ABI can share an OpenMP runtime. Restore the
            # requested count after the independent serial reference calls.
            pyscf_lib.num_threads(args.threads)
            for screened in (False, True):
                handles = []
                try:
                    for library in libraries:
                        library.tlno_onee_set_screening(int(screened))
                        handle = library.tlno_onee_create_from_inp(
                            str(path).encode(), basis.encode()
                        )
                        if not handle:
                            raise RuntimeError(f"Handle construction failed: {path}/{basis}")
                        handles.append(handle)

                    handle = handles[0]
                    full_reference = {}
                    for spherical in (False, True):
                        suffix = "_sph" if spherical else ""
                        getattr(current, "tlno_onee_eval_full" + suffix)(handle)
                        n = getattr(current, "tlno_onee_nbf" + suffix)(handle)
                        full_reference[spherical] = tuple(tlno_copy(
                            current, handle, n,
                            getattr(current, "tlno_onee_copy" + suffix),
                        ))
                    for library, prepared_handle in zip(libraries, handles):
                        library.tlno_onee_prepare_plan(prepared_handle)

                    # Switch directly from Cartesian preparation to spherical,
                    # then back after both cached and freshly built plans.
                    for scope, spherical in (
                        ("cached", True), ("cached", False), ("full", True),
                        ("full", False), ("cached", True), ("cached", False),
                    ):
                        suffix = "_sph" if spherical else ""
                        values = []
                        for library, prepared_handle in zip(libraries, handles):
                            getattr(library, "tlno_onee_eval_" + scope + suffix)(prepared_handle)
                            n = getattr(library, "tlno_onee_nbf" + suffix)(prepared_handle)
                            values.append(tuple(tlno_copy(
                                library, prepared_handle, n,
                                getattr(library, "tlno_onee_copy" + suffix),
                            )))
                        got = values[0]
                        external, _ = matrix_errors(got, independent[spherical])
                        scope_error, _ = matrix_errors(got, full_reference[spherical])
                        if max(external) > args.matrix_tolerance:
                            raise RuntimeError(("PySCF mismatch", molecule, basis,
                                                screened, scope, spherical, external))
                        for actual, reference in zip(got, full_reference[spherical]):
                            np.testing.assert_allclose(actual, reference, atol = 2e-10, rtol = 2e-11)
                            if not np.array_equal(actual, actual.T):
                                raise RuntimeError("Asymmetric matrix")
                        row = dict(molecule = molecule, basis = basis, screened = screened,
                                   scope = scope, spherical = spherical,
                                   max_abs_stv = scope_error, pyscf_max_abs_stv = external)
                        if previous is not None:
                            parity_error, _ = matrix_errors(got, values[1])
                            for actual, reference in zip(got, values[1]):
                                np.testing.assert_allclose(actual, reference,
                                                           atol = 2e-10, rtol = 2e-11)
                            row["previous_max_abs_stv"] = parity_error
                        rows.append(row)
                finally:
                    for library, handle in zip(libraries, handles):
                        library.tlno_onee_destroy(handle)
    finally:
        for library in libraries:
            library.tlno_onee_set_screening(1)
        
    binaries = [args.lib] + ([args.previous] if args.previous else [])
    result = dict(
        status = "PASS", threads = args.threads, checks = rows,
        basis_reference = "Actual current provider parameters exported outside evaluation",
        binaries = {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in binaries},
    )
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(result, indent = 2) + "\n")
    worst = max(max(row["pyscf_max_abs_stv"]) for row in rows)
    print(f"PASS: {len(rows)} C ABI representation/scope/screening switches, "
          f"{args.threads} threads; max PySCF difference {worst:.3e}")


if __name__ == "__main__":
    main()
