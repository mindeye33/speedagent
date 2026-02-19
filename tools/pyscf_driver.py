"""More general PySCF helpers.

This module factors out common patterns:
  - building a PySCF molecule from XYZ
  - constructing an SCF object from a 'driver' string and optional XC

It is intended to be reused by single-point, gradients, and geometry
optimization tools.
"""

from __future__ import annotations

from typing import Literal

from .chemio import parse_xyz, to_pyscf_atom_block


SCFDriver = Literal["RHF", "UHF", "RKS", "UKS"]


def build_mol_from_xyz(
    *,
    xyz: str,
    charge: int = 0,
    spin: int = 0,
    basis: str = "6-31g",
    unit: str = "Angstrom",
    verbose: int = 0,
):
    """Build a PySCF gto.M molecule from an XYZ string."""

    from pyscf import gto

    atoms = parse_xyz(xyz)
    mol = gto.M(
        atom=to_pyscf_atom_block(atoms),
        basis=basis,
        unit=unit,
        charge=int(charge),
        spin=int(spin),
        verbose=int(verbose),
    )
    return mol


def make_scf(
    mol,
    *,
    driver: SCFDriver = "RKS",
    xc: str = "B3LYP",
):
    """Create a PySCF SCF/KS object from a driver string."""

    from pyscf import scf

    drv = str(driver).upper().strip()
    if drv == "RHF":
        mf = scf.RHF(mol)
    elif drv == "UHF":
        mf = scf.UHF(mol)
    elif drv == "RKS":
        mf = scf.RKS(mol)
        mf.xc = xc
    elif drv == "UKS":
        mf = scf.UKS(mol)
        mf.xc = xc
    else:
        raise ValueError(f"Unsupported driver: {driver}")
    return mf, drv
