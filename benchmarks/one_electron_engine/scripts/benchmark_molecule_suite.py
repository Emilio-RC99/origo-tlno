#!/usr/bin/env python3
"""Shared C ABI and Libcint helpers.
The command-line entry point is benchmark_onee.py. Geometry, basis export,
reference preparation and result copies happen outside the timed region.
"""

from __future__ import annotations

import ctypes
import json
import os

from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / "benchmarks" / "one_electron_engine"
DEFAULT_LIB = BENCH / "build" / "libtlno_onee.so"
TEST_MOLECULES = ROOT / "test_molecules"

# TLNO basis name -> PySCF basis name.
BASIS_PAIRS = {
    "sto3g": "sto-3g",
    "431g": "4-31g",
    "631g": "6-31g",
    "631gd": "6-31g*",
    "631gdp": "6-31g**",
    "def2svp": "def2-svp",
    "def2tzvp": "def2-tzvp",
}

def pyscf_basis_spec(basis: str, *, lib = None, atoms = None):

    """Use the tested library's literal basis when lib and atoms are supplied.

    Migrated callers provide both keywords so basis changes cannot masquerade
    as integral-engine errors.
    """

    if lib is not None or atoms is not None:
        if lib is None or atoms is None:
            raise ValueError("Exact basis export requires both lib and atoms.")
        try:
            export = lib.tlno_onee_basis_json
        except AttributeError as exc:
            raise RuntimeError("Library lacks tlno_onee_basis_json; rebuild the benchmark C ABI") from exc
        export.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        export.restype = ctypes.c_char_p
        result = {}
        for atom in atoms:
            raw_symbol = atom.split()[0]
            symbol = raw_symbol[0].upper() + raw_symbol[1:].lower()
            if symbol in result:
                continue
            payload = export(basis.encode(), symbol.encode())
            if not payload:
                raise RuntimeError(f"Basis export failed: {symbol}/{basis}")
            data = json.loads(payload)
            result[symbol] = data["shells"]
        if not result:
            raise ValueError("No atoms supplied for basis export")
        return result
    return BASIS_PAIRS[basis]

BLAS_KEYS = (
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

# A small molecule -> large molecule spread (every name has a .inp).
DEFAULT_MOLECULES = (
    "heh,co,f2,methane,butadiene,benzene,ala_ala,gly_pro,"
    "tryptophan,carnosine,met_enkephalin,glutathione"
)


def pin_threads(threads: int) -> None:
    """Pin OpenMP + BLAS threads before pyscf or the .so initialise OpenMP."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OMP_DYNAMIC"] = "FALSE"
    os.environ["MKL_DYNAMIC"] = "FALSE"
    for key in BLAS_KEYS:
        os.environ[key] = "1"

def read_geometry(path: Path) -> tuple[int, int, tuple[str, ...]]:
    charge, multiplicity = 0, 1
    atoms: list[str] = []
    in_geometry = False
    for raw in path.read_text().splitlines():
        text = raw.strip()
        if not in_geometry:
            low = text.lower()
            if low.startswith("*") and "xyz" in low:
                fields = text.split()
                charge, multiplicity = int(fields[2]), int(fields[3])
                in_geometry = True
            continue
        if text.startswith("*"):
            break
        if not text or text.startswith(("#", "!")):
            continue
        fields = text.split()
        symbol = fields[0]
        x, y, z = (float(v) for v in fields[1:4])
        atoms.append(f"{symbol} {x:.15g} {y:.15g} {z:.15g}")
    if not atoms:
        raise ValueError(f"Missing xyz geometry: {path}")
    return charge, multiplicity, tuple(atoms)

def resolve_molecules(spec: str) -> list[Path]:
    if spec.strip().lower() == "all":
        suite = json.loads((BENCH / "suite.json").read_text())
        paths = [TEST_MOLECULES / (name + ".inp") for name in suite["molecules"]]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise ValueError("Missing production molecule inputs: " + ", ".join(missing))
        return paths
    paths = []
    for name in spec.split(","):
        item = name.strip()
        if not item:
            continue
        requested = Path(item).expanduser()
        path = requested if requested.is_file() else (TEST_MOLECULES / item).with_suffix(".inp")
        if not path.is_file():
            raise ValueError(f"Missing molecule input: {path}")
        if path.resolve() not in paths:
            paths.append(path.resolve())
    return paths

# --------------------------------------------------------------------------
# TLNO via ctypes
# --------------------------------------------------------------------------

def load_tlno(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    c_int_p = ctypes.POINTER(ctypes.c_int)
    lib.tlno_onee_create_from_inp.restype = ctypes.c_void_p
    lib.tlno_onee_create_from_inp.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.tlno_onee_nbf.restype = ctypes.c_int
    lib.tlno_onee_nbf.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_set_screening.argtypes = [ctypes.c_int]
    lib.tlno_onee_screening_enabled.restype = ctypes.c_int
    lib.tlno_onee_prepare_plan.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_eval_full.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_eval_cached.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_copy.argtypes = [ctypes.c_void_p] * 4
    # Spherical-harmonic output path (cart kernels + per-block cart->sph).
    lib.tlno_onee_nbf_sph.restype = ctypes.c_int
    lib.tlno_onee_nbf_sph.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_eval_full_sph.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_eval_cached_sph.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_copy_sph.argtypes = [ctypes.c_void_p] * 4
    lib.tlno_onee_pair_stats.argtypes = [ctypes.c_void_p, c_int_p, c_int_p, c_int_p]
    lib.tlno_onee_noop.argtypes = [ctypes.c_void_p]
    lib.tlno_onee_destroy.argtypes = [ctypes.c_void_p]
    for name in ("tlno_onee_set_screening", 
                 "tlno_onee_prepare_plan",
                 "tlno_onee_eval_full", 
                 "tlno_onee_eval_cached", 
                 "tlno_onee_copy",
                 "tlno_onee_eval_full_sph", 
                 "tlno_onee_eval_cached_sph", 
                 "tlno_onee_copy_sph",
                 "tlno_onee_pair_stats", 
                 "tlno_onee_noop", "tlno_onee_destroy"):
        getattr(lib, name).restype = None
    for name in ("tlno_onee_set_group_pair_decay_eps", "tlno_onee_set_nuclear_screen_eps"):
        getattr(lib, name).argtypes = [ctypes.c_double]
        getattr(lib, name).restype = None
    for name in ("tlno_onee_group_pair_decay_eps", "tlno_onee_nuclear_screen_eps"):
        getattr(lib, name).argtypes = []
        getattr(lib, name).restype = ctypes.c_double
    return lib

def tlno_pair_stats(lib: ctypes.CDLL, handle: int) -> tuple[int, int, int]:
    g, total, kept = ctypes.c_int(), ctypes.c_int(), ctypes.c_int()
    lib.tlno_onee_pair_stats(
        handle, ctypes.byref(g), ctypes.byref(total), ctypes.byref(kept)
    )
    return g.value, total.value, kept.value


def tlno_copy(lib: ctypes.CDLL, handle: int, nbf: int, copy_fn = None):

    import numpy as np
    
    out = [np.empty((nbf, nbf), dtype=np.float64) for _ in range(3)]
    (copy_fn or lib.tlno_onee_copy)(
        handle,
        out[0].ctypes.data_as(ctypes.c_void_p),
        out[1].ctypes.data_as(ctypes.c_void_p),
        out[2].ctypes.data_as(ctypes.c_void_p),
    )
    return out


# --------------------------------------------------------------------------
# libcint via the same ctypes path, output pre-allocated outside the loop
# --------------------------------------------------------------------------

class _NativeCintOpt(ctypes.c_void_p):
    """CINTOpt pointer whose destructor uses the selected libcgto."""

    deleter = None

    def __del__(self):
        if self.deleter is not None and self.value:
            self.deleter(ctypes.byref(self))


def _load_libcgto(path: Path | None):

    from pyscf.gto import moleintor

    if path is None:
        return moleintor.libcgto
    mode = os.RTLD_LOCAL | os.RTLD_NOW
    mode |= getattr(os, "RTLD_DEEPBIND", 0)
    return ctypes.CDLL(str(path.resolve()), mode=mode)


def _make_cintopt(libcgto, atm, bas, env, intor):
    name = intor.replace("_sph", "").replace("_cart", "").replace("_spinor", "")
    opt = _NativeCintOpt()
    opt.deleter = libcgto.CINTdel_optimizer
    getattr(libcgto, name + "_optimizer")(
        ctypes.byref(opt),
        atm.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(atm.shape[0]),
        bas.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_int(bas.shape[0]),
        env.ctypes.data_as(ctypes.c_void_p),
    )
    return opt


def make_pyscf_molecule(atoms, basis_spec, charge, multiplicity, *, spherical = False):
    """Construct a reference with the exported contraction-shell order.

    PySCF sorts an explicit element basis by angular momentum. The engine keeps
    provider order (which can interleave s and p contractions), so reorder basis
    rows once at setup. Their exponent/coefficient pointers remain unchanged.
    """
    import numpy as np
    from pyscf import gto
    from collections import defaultdict, deque

    mol = gto.M(atom="; ".join(atoms), basis=basis_spec, charge=charge,
                spin=multiplicity - 1, cart=not spherical, unit="Bohr", verbose=0)
    if isinstance(basis_spec, dict):
        order = []
        for atom in range(mol.natm):
            symbol = mol.atom_pure_symbol(atom)
            shells = basis_spec.get(symbol)
            if shells is None:
                raise ValueError(f"Missing exported basis for {symbol}")
            by_l = defaultdict(deque)
            for row, values in enumerate(mol._bas):
                if int(values[gto.ATOM_OF]) == atom:
                    by_l[int(values[gto.ANG_OF])].append(row)
            for shell in shells:
                angular = int(shell[0])
                if not by_l[angular]:
                    raise ValueError(f"Reference contraction-shell mismatch: {symbol}, L={angular}")
                row = by_l[angular].popleft()
                if int(mol._bas[row, gto.NCTR_OF]) != 1:
                    raise ValueError("Exported reference requires one contraction per shell")
                order.append(row)
            if any(by_l.values()):
                raise ValueError(f"Extra reference contraction shells: {symbol}")
        mol._bas = np.asarray(mol._bas[order], dtype=np.int32, order="C")
    return mol


def make_libcint(
    atoms, basis, charge, multiplicity, threads, libcgto_path: Path | None = None,
    libcint_expcutoff: float = 0.0, spherical: bool = False,
):

    import numpy as np
    from pyscf import gto, lib
    from pyscf.gto import moleintor

    lib.num_threads(threads)
    mol = make_pyscf_molecule(atoms, basis, charge, multiplicity, 
                              spherical = spherical)
    # Override libcint's primitive-pair screening (env[PTR_EXPCUTOFF]). 0 keeps
    # PySCF's default (libcint EXPCUTOFF=60); a large value effectively disables
    # both libcint screening levels for the unscreened-vs-unscreened study.
    if libcint_expcutoff:
        mol._env[gto.PTR_EXPCUTOFF] = libcint_expcutoff
    atm = np.asarray(mol._atm, dtype = np.int32, order = "C")
    bas = np.asarray(mol._bas, dtype = np.int32, order = "C")
    env = np.asarray(mol._env, dtype = np.float64, order = "C")
    nbas = bas.shape[0]
    # Spherical drives int1e_*_sph (Cartesian recurrence + per-block c2s inside
    # libcint's CINT1e_drv); Cartesian drives int1e_*_cart (no transform).
    suffix = "_sph" if spherical else "_cart"
    nao = mol.nao_nr() if spherical else mol.nao_cart()
    shls = (ctypes.c_int * 4)(0, nbas, 0, nbas)
    libcgto = _load_libcgto(libcgto_path)
    driver = libcgto.GTOint2c
    names = (f"int1e_ovlp{suffix}", f"int1e_kin{suffix}", f"int1e_nuc{suffix}")

    def make_setups():
        setups = []
        for name in names:
            loc = moleintor.make_loc(bas, name)
            opt = _make_cintopt(libcgto, atm, bas, env, name)
            setups.append((getattr(libcgto, name), loc, opt))
        return setups

    atm_ptr = atm.ctypes.data_as(ctypes.c_void_p)
    bas_ptr = bas.ctypes.data_as(ctypes.c_void_p)
    env_ptr = env.ctypes.data_as(ctypes.c_void_p)
    natm = atm.shape[0]
    # Pre-allocate the three output matrices ONCE, outside the timed loop.
    outputs = [np.empty((nao, nao), dtype=np.float64, order="F") for _ in range(3)]
    out_ptrs = [m.ctypes.data_as(ctypes.c_void_p) for m in outputs]

    def one_pass(setups):
        for (integral, loc, opt), mat_ptr in zip(setups, out_ptrs):
            driver(
                integral,
                mat_ptr,
                ctypes.c_int(1),
                ctypes.c_int(1),  # hermi=1: one shell triangle.
                shls,
                loc.ctypes.data_as(ctypes.c_void_p),
                opt,
                atm_ptr,
                ctypes.c_int(natm),
                bas_ptr,
                ctypes.c_int(nbas),
                env_ptr,
            )

    return nao, make_setups, one_pass, outputs


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
def main() -> None:
    # Retain the established filename as an entry point and import location.
    from benchmark_onee import main as run_benchmark
    run_benchmark()


if __name__ == "__main__":
    main()
