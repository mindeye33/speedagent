"""General, engine-agnostic molecular geometry utilities.

This module fills a gap between low-level XYZ parsing (:mod:`tools.chemio`)
and heavy QC/MD engines.

Design goals
------------
- **Abstract**: operate on generic "atom lists" (symbol + xyz) independent of
  any downstream engine.
- **General**: common transformations (translate, rotate, Kabsch alignment),
  distance matrices, and RMSD.
- **JSON-friendly**: return lists/dicts with floats (no numpy objects).

Notes
-----
We implement a minimal Kabsch alignment (with optional mass/weighting) so that
RMSD can be computed in a way consistent with most chemistry tooling.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .chemio import Atom, parse_xyz
from .schema import _as_list


def _atoms_to_arrays(atoms: list[Atom]) -> tuple[list[str], np.ndarray]:
    syms = [a.symbol for a in atoms]
    xyz = np.array([[a.x, a.y, a.z] for a in atoms], dtype=float)
    return syms, xyz


def geometry_basic(
    *,
    xyz: str,
    center: Literal["com", "centroid", "none"] = "centroid",
    masses: dict[str, float] | None = None,
) -> dict:
    """Compute basic geometry descriptors from an XYZ string.

    Returns
    -------
    dict with keys:
      - natoms
      - symbols
      - centroid_angstrom
      - com_angstrom (if masses provided or center='com')
      - distance_matrix_angstrom
    """

    atoms = parse_xyz(xyz)
    syms, X = _atoms_to_arrays(atoms)

    centroid = X.mean(axis=0)

    com = None
    if center == "com" or masses is not None:
        if masses is None:
            raise ValueError("masses must be provided when center='com'")
        w = np.array([float(masses[s]) for s in syms], dtype=float)
        if np.any(w <= 0):
            raise ValueError("All masses must be positive")
        com = (w[:, None] * X).sum(axis=0) / w.sum()

    d = X[:, None, :] - X[None, :, :]
    dist = np.sqrt((d * d).sum(axis=2))

    out = {
        "natoms": int(len(atoms)),
        "symbols": syms,
        "centroid_angstrom": _as_list(centroid),
        "distance_matrix_angstrom": _as_list(dist),
        "settings": {"center": center},
    }
    if com is not None:
        out["com_angstrom"] = _as_list(com)
    return out


def kabsch_align(
    *,
    xyz_ref: str,
    xyz_mob: str,
    weights: list[float] | None = None,
    allow_reflection: bool = False,
) -> dict:
    """Align a mobile geometry onto a reference using the Kabsch algorithm.

    Returns
    -------
    dict with keys:
      - rotation (3x3)
      - translation (3,) such that: X_mob_aligned = X_mob @ R + t
      - rmsd_angstrom (after alignment)
    """

    ref = parse_xyz(xyz_ref)
    mob = parse_xyz(xyz_mob)
    if len(ref) != len(mob):
        raise ValueError("xyz_ref and xyz_mob must have same number of atoms")

    syms_r, Xr = _atoms_to_arrays(ref)
    syms_m, Xm = _atoms_to_arrays(mob)
    if syms_r != syms_m:
        raise ValueError("Atom symbols/order differ between reference and mobile")

    n = Xr.shape[0]
    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        if len(weights) != n:
            raise ValueError("weights must have length natoms")
        w = np.asarray(weights, dtype=float)
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")

    wsum = float(w.sum())
    if wsum == 0.0:
        raise ValueError("Sum of weights is zero")

    cr = (w[:, None] * Xr).sum(axis=0) / wsum
    cm = (w[:, None] * Xm).sum(axis=0) / wsum
    Xrc = Xr - cr
    Xmc = Xm - cm

    H = (w[:, None] * Xmc).T @ Xrc
    U, _S, Vt = np.linalg.svd(H)
    R = U @ Vt
    if not allow_reflection and np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    t = cr - cm @ R
    Xma = Xm @ R + t
    diff = Xma - Xr
    rmsd_val = np.sqrt(((w[:, None] * diff * diff).sum() / wsum))

    return {
        "rotation": _as_list(R),
        "translation": _as_list(t),
        "rmsd_angstrom": float(rmsd_val),
        "settings": {
            "allow_reflection": bool(allow_reflection),
            "weighted": weights is not None,
        },
    }


def rmsd(
    *,
    xyz_ref: str,
    xyz_mob: str,
    align: bool = True,
    weights: list[float] | None = None,
) -> dict:
    """Compute RMSD between two XYZ geometries.

    If `align=True`, performs Kabsch alignment first.
    """

    if align:
        res = kabsch_align(xyz_ref=xyz_ref, xyz_mob=xyz_mob, weights=weights)
        return {
            "rmsd_angstrom": float(res["rmsd_angstrom"]),
            "aligned": True,
            "settings": {"align": True, "weighted": weights is not None},
        }

    ref = parse_xyz(xyz_ref)
    mob = parse_xyz(xyz_mob)
    if len(ref) != len(mob):
        raise ValueError("xyz_ref and xyz_mob must have same number of atoms")

    _, Xr = _atoms_to_arrays(ref)
    _, Xm = _atoms_to_arrays(mob)
    diff = Xm - Xr

    if weights is None:
        rmsd_val = np.sqrt((diff * diff).sum() / Xr.shape[0])
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != Xr.shape[0]:
            raise ValueError("weights must have length natoms")
        wsum = float(w.sum())
        if wsum == 0.0:
            raise ValueError("Sum of weights is zero")
        rmsd_val = np.sqrt(((w[:, None] * diff * diff).sum() / wsum))

    return {
        "rmsd_angstrom": float(rmsd_val),
        "aligned": False,
        "settings": {"align": False, "weighted": weights is not None},
    }
