#!/usr/bin/env python3
"""Independent Libcint oracle for contracted s-through-g transfer fixtures."""
import argparse
import hashlib
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from pyscf import gto, lib

BENCH = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("--contract", type = Path, default = BENCH/'build'/'transfer_contract')
    parser.add_argument("--out", type = Path, default = BENCH/'results'/'validation'/'transfer_fixtures.json')
    args = parser.parse_args()
    lib.num_threads(1)
    dump = subprocess.check_output([str(args.contract.resolve()), "--dump"], text = True)
    data = np.loadtxt(io.StringIO(dump))
    if data.shape != (1225, 7) or not np.isfinite(data).all():
        raise RuntimeError("Expected 1,225 finite ordered s-through-g fixture rows")
    rows = []
    for la in range(5):
        for lb in range(5):
            def basis(L,depth):
                return [[L, *[[0.18+0.7*k,-0.2 if k%2 else 0.7] for k in range(depth)]]]
            mol=gto.M(atom = "H1 0 0 0; H2 0.3 -0.8 1.1", unit = "Bohr", cart = True,
                      basis = {"H1":basis(la, 2), "H2":basis(lb, 3)}, verbose = 0)
            s = mol.intor("int1e_ovlp_cart")
            t = mol.intor("int1e_kin_cart")
            v = np.zeros_like(s)
            for k in range(17):
                charge = 0 if k % 5 == 1 else (1 if k % 3 else -2)
                with mol.with_rinv_origin((0.1 * k, 0.7 * k, -0.2 * k)):
                    v -= charge * mol.intor("int1e_rinv_cart")
            scale = 1.0 / np.sqrt(np.diag(s))
            na = (la + 1) * (la + 2) // 2
            selected = data[(data[:, 0] == la) & (data[:, 1] == lb)]
            nb = (lb + 1) * (lb + 2) // 2
            expected = np.array([(i,j) for i in range(na) for j in range(nb)])
            np.testing.assert_array_equal(selected[:,2:4],expected)
            errors=[]
            for column, ref in zip((4,5,6),(s,t,v)):
                normalized = (scale[:,None]*ref*scale[None,:])[:na,na:]
                got = selected[:,column].reshape(normalized.shape)
                np.testing.assert_allclose(got,normalized, atol=2e-10, rtol=2e-11)
                errors.append(float(np.max(np.abs(got - normalized))))
            rows.append(dict(la = la, lb = lb, max_abs_stv = errors))
    args.out.parent.mkdir(parents = True, exist_ok = True)
    args.out.write_text(json.dumps(dict(status = "PASS", pyscf = __import__("pyscf").__version__,
                                      values = int(data.shape[0] * 3), cases = rows,
                                      contract_sha256 = hashlib.sha256(args.contract.read_bytes()).hexdigest()), indent = 2) + "\n")
    print(f"PASS: {data.shape[0]*3} S/T/V values, all 25 ordered s-through-g classes; "
          f"max error {max(max(r['max_abs_stv']) for r in rows):.3e}")


if __name__ == "__main__":
    main()
