
"""RDKit-based utilities for small-molecule workflows.

Scope
-----
- SMILES -> 3D conformers (ETKDG)
- optional MMFF/UFF minimization
- RMSD computation
- clustering / pruning
- export conformers as XYZ strings

Design choices
--------------
- We default to RDKit's Butina clustering for labels, because it's a standard,
  reproducible clustering algorithm commonly used in conformer workflows.
- We compute RMSD on the *final returned set* by default (smaller, more useful).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "RDKit is required for rdkit_tools but is not available. "
            "Install `rdkit` and retry."
        ) from e


def _mol_to_xyz_block(mol: Any, conformer_id: int, comment: str = "") -> str:
    """Convert an RDKit mol+conformer into an XYZ string."""
    _require_rdkit()

    conf = mol.GetConformer(conformer_id)
    lines: list[str] = []
    nat = mol.GetNumAtoms()
    lines.append(str(nat))
    lines.append(comment)
    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        sym = atom.GetSymbol()
        lines.append(f"{sym:2s} {pos.x: .8f} {pos.y: .8f} {pos.z: .8f}")
    return "\n".join(lines) + "\n"


def _embed_conformers(
    mol: Any,
    n_confs: int,
    seed: int,
    use_small_ring_torsions: bool = True,
    use_basic_knowledge: bool = True,
    prune_rmsd: float | None = 0.5,
    etkdg_version: Literal["v3", "v2"] = "v3",
) -> list[int]:
    _require_rdkit()
    from rdkit.Chem import AllChem

    if etkdg_version == "v3":
        params = AllChem.ETKDGv3()
    elif etkdg_version == "v2":
        params = AllChem.ETKDGv2()
    else:
        raise ValueError("etkdg_version must be 'v3' or 'v2'")

    params.randomSeed = int(seed)
    params.useSmallRingTorsions = bool(use_small_ring_torsions)
    params.useBasicKnowledge = bool(use_basic_knowledge)
    if prune_rmsd is not None:
        params.pruneRmsThresh = float(prune_rmsd)

    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params))
    return ids


def _minimize_conformers(
    mol: Any,
    conformer_ids: list[int],
    ff: Literal["mmff", "uff", "none"] = "mmff",
    max_iters: int = 200,
) -> dict[int, float | None]:
    """Return energies (kcal/mol) for each conformer id."""
    _require_rdkit()
    from rdkit.Chem import AllChem

    energies: dict[int, float | None] = {cid: None for cid in conformer_ids}
    if ff == "none":
        return energies

    if ff == "mmff":
        props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
        if props is None:
            ff = "uff"  # fallback
        else:
            for cid in conformer_ids:
                f = AllChem.MMFFGetMoleculeForceField(mol, props, confId=cid)
                if f is None:
                    energies[cid] = None
                    continue
                f.Minimize(maxIts=int(max_iters))
                energies[cid] = float(f.CalcEnergy())
            return energies

    if ff == "uff":
        for cid in conformer_ids:
            f = AllChem.UFFGetMoleculeForceField(mol, confId=cid)
            if f is None:
                energies[cid] = None
                continue
            f.Minimize(maxIts=int(max_iters))
            energies[cid] = float(f.CalcEnergy())
        return energies

    raise ValueError("ff must be one of: 'mmff', 'uff', 'none'")


def _pairwise_rmsd(mol: Any, conformer_ids: list[int]) -> np.ndarray:
    """Compute symmetric RMSD matrix (Angstrom) for the given conformer ids."""
    _require_rdkit()
    from rdkit.Chem import AllChem

    n = len(conformer_ids)
    rmsd = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            v = AllChem.GetBestRMS(
                mol, mol, prbId=conformer_ids[i], refId=conformer_ids[j]
            )
            rmsd[i, j] = rmsd[j, i] = float(v)
    return rmsd


def _butina_cluster_labels_from_rmsd(rmsd: np.ndarray, thresh: float) -> list[int]:
    """Cluster conformers using RDKit Butina on an RMSD matrix.

    Returns
    -------
    labels : list[int]
        Cluster label for each point, aligned with the order used to build `rmsd`.

    Notes
    -----
    Butina is a simple and common clustering algorithm in cheminformatics.
    It groups points using a distance threshold and returns clusters ordered by size.
    """

    _require_rdkit()
    from rdkit.ML.Cluster import Butina

    n = int(rmsd.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [0]

    # Upper triangle distances as a flat list (RDKit convention)
    dists: list[float] = []
    for i in range(1, n):
        for j in range(i):
            dists.append(float(rmsd[i, j]))

    clusters = Butina.ClusterData(
        dists, nPts=n, distThresh=float(thresh), isDistData=True
    )

    labels = [-1] * n
    for k, clust in enumerate(clusters):
        for idx in clust:
            labels[int(idx)] = int(k)

    # Safety: if any unassigned (shouldn't happen), give singleton clusters
    next_k = len(clusters)
    for i, lab in enumerate(labels):
        if lab == -1:
            labels[i] = int(next_k)
            next_k += 1

    return labels


def _greedy_prune_by_rmsd(
    mol: Any,
    conformer_ids: list[int],
    energies_kcal: dict[int, float | None],
    rmsd_thresh: float,
) -> list[int]:
    """Pick a diverse subset by greedy selection in ascending energy order."""

    if len(conformer_ids) <= 1:
        return conformer_ids

    def key(cid: int):
        e = energies_kcal.get(cid, None)
        return (e is None, 1e30 if e is None else float(e))

    ordered = sorted(conformer_ids, key=key)
    rmsd = _pairwise_rmsd(mol, ordered)

    chosen_idx: list[int] = []
    for i in range(len(ordered)):
        if not chosen_idx:
            chosen_idx.append(i)
            continue
        if all(rmsd[i, j] >= rmsd_thresh for j in chosen_idx):
            chosen_idx.append(i)

    return [ordered[i] for i in chosen_idx]


def rdkit_conformers_from_smiles(
    smiles: str,
    n_confs: int = 50,
    seed: int = 0,
    add_hs: bool = True,
    embed_prune_rmsd: float | None = 0.5,
    etkdg_version: Literal["v3", "v2"] = "v3",
    minimize_ff: Literal["mmff", "uff", "none"] = "mmff",
    minimize_max_iters: int = 200,
    cluster_rmsd_thresh: float | None = 0.75,
    max_return: int | None = 20,
    # New in this iteration:
    return_rmsd_matrix: bool = True,
    return_cluster_labels: bool = True,
    cluster_method: Literal["butina", "greedy"] = "butina",
) -> dict:
    """Generate 3D conformers, optionally minimize, prune/cluster, export XYZ.

    What you get
    -----------
    - `conformers`: list of conformer dicts (id, energy, xyz)
    - optionally `rmsd_matrix`: NxN RMSD matrix (Angstrom) for the returned conformers
    - optionally `cluster_labels`: length-N list of ints for returned conformers

    Defaults
    --------
    We compute RMSD/clusters on the *returned* conformer set (after pruning/cap),
    which keeps the matrix small and immediately useful.
    """

    _require_rdkit()
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Could not parse SMILES")
    if add_hs:
        mol = Chem.AddHs(mol)

    conf_ids = _embed_conformers(
        mol,
        n_confs=n_confs,
        seed=seed,
        prune_rmsd=embed_prune_rmsd,
        etkdg_version=etkdg_version,
    )
    if len(conf_ids) == 0:
        raise RuntimeError("RDKit failed to embed any conformers")

    energies = _minimize_conformers(
        mol, conf_ids, ff=minimize_ff, max_iters=minimize_max_iters
    )

    # Step 1: reduce conformers for downstream use (diversity pruning)
    keep = conf_ids
    if cluster_rmsd_thresh is not None:
        keep = _greedy_prune_by_rmsd(
            mol, conf_ids, energies_kcal=energies, rmsd_thresh=float(cluster_rmsd_thresh)
        )

    if max_return is not None:
        keep = keep[: int(max_return)]

    # Step 2: compute RMSD and cluster labels on the final set
    rmsd_matrix = None
    cluster_labels = None
    if return_rmsd_matrix or return_cluster_labels:
        rmsd_matrix = _pairwise_rmsd(mol, keep)

    if return_cluster_labels:
        if cluster_method == "butina":
            cluster_labels = _butina_cluster_labels_from_rmsd(
                rmsd_matrix, thresh=float(cluster_rmsd_thresh or 0.75)
            )
        elif cluster_method == "greedy":
            # Greedy pruning doesn't naturally yield multi-member clusters.
            # Provide a trivial label per conformer.
            cluster_labels = list(range(len(keep)))
        else:
            raise ValueError("cluster_method must be 'butina' or 'greedy'")

    # Step 3: package conformers
    conformers: list[dict[str, Any]] = []
    for i, cid in enumerate(keep):
        e = energies.get(cid, None)
        comment = f"smiles={smiles} confId={cid}"
        if e is not None:
            comment += f" energy_kcal_mol={e:.6f}"
        if cluster_labels is not None:
            comment += f" cluster={cluster_labels[i]}"

        conformers.append(
            {
                "conformer_id": int(cid),
                "energy_kcal_mol": None if e is None else float(e),
                "cluster": None if cluster_labels is None else int(cluster_labels[i]),
                "xyz": _mol_to_xyz_block(mol, cid, comment=comment),
            }
        )

    out = {
        "smiles": smiles,
        "n_embedded": int(len(conf_ids)),
        "n_returned": int(len(conformers)),
        "conformers": conformers,
        "settings": {
            "n_confs": int(n_confs),
            "seed": int(seed),
            "add_hs": bool(add_hs),
            "embed_prune_rmsd": None if embed_prune_rmsd is None else float(embed_prune_rmsd),
            "etkdg_version": etkdg_version,
            "minimize_ff": minimize_ff,
            "minimize_max_iters": int(minimize_max_iters),
            "cluster_rmsd_thresh": None
            if cluster_rmsd_thresh is None
            else float(cluster_rmsd_thresh),
            "max_return": None if max_return is None else int(max_return),
            "return_rmsd_matrix": bool(return_rmsd_matrix),
            "return_cluster_labels": bool(return_cluster_labels),
            "cluster_method": cluster_method,
        },
    }

    if return_rmsd_matrix:
        out["rmsd_matrix"] = None if rmsd_matrix is None else rmsd_matrix.tolist()
    if return_cluster_labels:
        out["cluster_labels"] = cluster_labels

    return out
