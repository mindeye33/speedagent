from __future__ import annotations

"""Legacy geometry optimization and single-point tools (PySCF-based).

This module originally implemented its own XYZ parsing and SCF driver selection.
The repository now has a more abstract entry point in :mod:`tools.qc`.

We keep these functions for backwards compatibility, but implement them as thin
wrappers around :func:`tools.qc.pyscf_run`.

Notes
-----
- Requires PySCF.
- Geometry optimization requires geomeTRIC (installed as `geometric`).
"""

from typing import Literal

from .qc import pyscf_run


def pyscf_optimize_geometry(
    *,
    xyz: str,
    charge: int = 0,
    spin: int = 0,
    basis: str = "6-31g",
    driver: Literal["RHF", "UHF", "RKS", "UKS"] = "RKS",
    xc: str = "B3LYP",
    maxsteps: int = 50,
    verbose: int = 0,
) -> dict:
    """Optimize geometry using PySCF + geomeTRIC.

    Backwards-compatible wrapper around :func:`tools.qc.pyscf_run`.

    Returns
    -------
    dict with keys:
      - xyz_optimized
      - e_hartree
      - natoms
      - settings
    """

    res = pyscf_run(
        task="optimize_geometry",
        xyz=xyz,
        charge=charge,
        spin=spin,
        basis=basis,
        driver=driver,
        xc=xc,
        maxsteps=maxsteps,
        verbose=verbose,
    )

    return {
        "xyz_optimized": res["xyz_optimized"],
        "e_hartree": res["e_hartree"],
        "natoms": res["natoms"],
        "settings": res["settings"],
        "notes": {"deprecated": "Use tools.qc.pyscf_run(task='optimize_geometry', ...)"},
    }


def pyscf_single_point(
    *,
    xyz: str,
    charge: int = 0,
    spin: int = 0,
    basis: str = "6-31g",
    driver: Literal["RHF", "UHF", "RKS", "UKS"] = "RKS",
    xc: str = "B3LYP",
    verbose: int = 0,
) -> dict:
    """Compute a single-point energy using PySCF.

    Backwards-compatible wrapper around :func:`tools.qc.pyscf_run`.
    """

    res = pyscf_run(
        task="single_point",
        xyz=xyz,
        charge=charge,
        spin=spin,
        basis=basis,
        driver=driver,
        xc=xc,
        verbose=verbose,
    )

    return {
        "e_hartree": res["e_hartree"],
        "natoms": res["natoms"],
        "settings": res["settings"],
        "notes": {"deprecated": "Use tools.qc.pyscf_run(task='single_point', ...)"},
    }
