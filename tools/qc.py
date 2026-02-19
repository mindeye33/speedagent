"""General quantum-chemistry (QC) tools built on PySCF.

This module aims to be *more abstract* than the previous geomopt.py by:
  - providing a single 'run' entry point parameterized by task
  - centralizing parsing, molecule building, and SCF construction
  - returning a consistent results schema

Currently supports:
  - single point energies
  - geometry optimization via geomeTRIC

In the future this can host gradients, Hessians, properties, etc.
"""

from __future__ import annotations

from typing import Literal

from .chemio import Atom, format_xyz
from .pyscf_driver import SCFDriver, build_mol_from_xyz, make_scf


QCTask = Literal["single_point", "optimize_geometry"]


def pyscf_run(
    *,
    task: QCTask,
    xyz: str,
    charge: int = 0,
    spin: int = 0,
    basis: str = "6-31g",
    driver: SCFDriver = "RKS",
    xc: str = "B3LYP",
    maxsteps: int = 50,
    verbose: int = 0,
) -> dict:
    """Run a QC task with PySCF.

    Parameters
    ----------
    task:
        "single_point" or "optimize_geometry".

    Returns
    -------
    dict containing at least:
      - task
      - e_hartree
      - natoms
      - settings

    For optimize_geometry also includes:
      - xyz_optimized
    """

    task2 = str(task).strip().lower()

    mol = build_mol_from_xyz(
        xyz=xyz,
        charge=charge,
        spin=spin,
        basis=basis,
        unit="Angstrom",
        verbose=verbose,
    )

    mf, drv = make_scf(mol, driver=driver, xc=xc)

    if task2 == "single_point":
        mf.run(verbose=verbose)
        e = float(mf.e_tot)
        return {
            "task": "single_point",
            "e_hartree": e,
            "natoms": int(mol.natm),
            "settings": {
                "basis": basis,
                "driver": drv,
                "xc": xc,
                "charge": int(charge),
                "spin": int(spin),
            },
        }

    if task2 in ("optimize_geometry", "opt", "geomopt"):
        from pyscf import geomopt

        opt_mol = geomopt.optimize(mf, maxsteps=int(maxsteps))

        # single point at optimized geometry
        mf2, _drv2 = make_scf(opt_mol, driver=drv, xc=xc)
        mf2.run(verbose=verbose)
        eopt = float(mf2.e_tot)

        coords = opt_mol.atom_coords(unit="Angstrom")
        syms = [opt_mol.atom_symbol(i) for i in range(opt_mol.natm)]
        atoms_opt = [
            Atom(s, float(c[0]), float(c[1]), float(c[2]))
            for s, c in zip(syms, coords)
        ]

        comment = (
            f"E={eopt:.12f} Hartree  driver={drv} xc={xc} basis={basis} "
            f"charge={int(charge)} spin={int(spin)}"
        )

        return {
            "task": "optimize_geometry",
            "e_hartree": eopt,
            "natoms": int(opt_mol.natm),
            "xyz_optimized": format_xyz(atoms_opt, comment=comment),
            "settings": {
                "basis": basis,
                "driver": drv,
                "xc": xc,
                "charge": int(charge),
                "spin": int(spin),
                "maxsteps": int(maxsteps),
            },
        }

    raise ValueError(f"Unsupported task: {task}")
