"""Modular OpenMM MD utilities.

This module provides:
  - building a periodic mixture system + topology + initial positions
  - running short MD and returning sampled velocities (and optionally positions)
  - a backwards-compatible analysis helper (VACF -> FFT spectrum)

Design note
-----------
The core analysis is now factored into :mod:`tools.md_analysis` and
:mod:`tools.signal` so it can be reused with trajectories produced by other
engines.

This is still a lightweight workflow (toy force field); the abstractions are
intended to make it easy to swap in a real force field later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_openmm_and_scipy():
    try:
        import openmm as mm  # noqa: F401
        import openmm.app as app  # noqa: F401
        from openmm import unit  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError("OpenMM is required. Install openmm>=8 and retry.") from e

    # SciPy is required for some analysis paths; kept for backward-compat.
    try:
        from scipy.signal import windows  # noqa: F401
        from scipy.signal import find_peaks  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SciPy is required (scipy.signal.windows, scipy.signal.find_peaks). Install scipy and retry."
        ) from e


def _make_molecule_positions_nm(name: str) -> np.ndarray:
    """Return a crude gas-phase geometry in nm for small molecules.

    Supported: 'water', 'methanol', 'acetone'
    """

    name = name.lower().strip()
    if name in ("water", "h2o"):
        O = np.array([0.0, 0.0, 0.0])
        rOH = 0.09572
        angleHOH = np.deg2rad(104.52)
        H1 = np.array([rOH, 0.0, 0.0])
        H2 = np.array([rOH * np.cos(angleHOH), rOH * np.sin(angleHOH), 0.0])
        return np.vstack([O, H1, H2])

    if name in ("methanol", "meoh", "ch3oh"):
        # Atom order: C, O, H(O), H1, H2, H3
        C = np.array([0.000, 0.000, 0.000])
        O = np.array([0.143, 0.000, 0.000])
        H_O = np.array([0.239, 0.000, 0.000])
        H1 = np.array([-0.036, 0.1030, 0.0000])
        H2 = np.array([-0.036, -0.0515, 0.0892])
        H3 = np.array([-0.036, -0.0515, -0.0892])
        return np.vstack([C, O, H_O, H1, H2, H3])


    if name in ("acetone", "propanone"):
        # Very rough geometry (nm). Atom order: C0 (carbonyl), O, C1 (methyl),
        # C2 (methyl), then 6 H (3 on each methyl).
        # Distances: C=O ~1.22 Å, C-C ~1.51 Å.
        C0 = np.array([0.000, 0.000, 0.000])
        O = np.array([0.122, 0.000, 0.000])
        C1 = np.array([-0.151, 0.110, 0.000])
        C2 = np.array([-0.151, -0.110, 0.000])

        # Place methyl hydrogens in a crude tetrahedral-ish arrangement
        H1a = C1 + np.array([-0.036, 0.103, 0.000])
        H1b = C1 + np.array([-0.036, -0.0515, 0.0892])
        H1c = C1 + np.array([-0.036, -0.0515, -0.0892])
        H2a = C2 + np.array([-0.036, -0.103, 0.000])
        H2b = C2 + np.array([-0.036, 0.0515, 0.0892])
        H2c = C2 + np.array([-0.036, 0.0515, -0.0892])

        return np.vstack([C0, O, C1, C2, H1a, H1b, H1c, H2a, H2b, H2c])

    raise ValueError(f"Unsupported molecule name: {name}")


@dataclass
class BuiltSystem:
    topology: Any
    system: Any
    positions_nm: np.ndarray
    box_nm: float


def openmm_build_periodic_mixture(
    *,
    components: list[dict],
    box_nm: float = 2.0,
    seed: int = 0,
    cutoff_nm: float = 0.9,
) -> dict:
    """Build a periodic mixture from simple templates.

    components: list of dicts like {"name": "water", "count": 20}

    Force field: toy LJ + Coulomb + harmonic bonds/angles.

    Returns a dict with OpenMM objects plus a plain numpy positions array in nm.
    (Tools should return JSONable objects; OpenMM objects are provided under
    private keys for internal chaining.)
    """

    _require_openmm_and_scipy()

    import openmm as mm
    import openmm.app as app
    from openmm import unit

    rng = np.random.default_rng(seed)

    top = app.Topology()
    chain = top.addChain("A")
    positions: list[np.ndarray] = []

    elem = {
        "H": app.element.hydrogen,
        "O": app.element.oxygen,
        "C": app.element.carbon,
    }

    def add_water(res):
        aO = top.addAtom("O", elem["O"], res)
        aH1 = top.addAtom("H1", elem["H"], res)
        aH2 = top.addAtom("H2", elem["H"], res)
        top.addBond(aO, aH1)
        top.addBond(aO, aH2)

    def add_methanol(res):
        aC = top.addAtom("C", elem["C"], res)
        aO = top.addAtom("O", elem["O"], res)
        aHO = top.addAtom("HO", elem["H"], res)
        aH1 = top.addAtom("H1", elem["H"], res)
        aH2 = top.addAtom("H2", elem["H"], res)
        aH3 = top.addAtom("H3", elem["H"], res)
        top.addBond(aC, aO)
        top.addBond(aO, aHO)
        top.addBond(aC, aH1)
        top.addBond(aC, aH2)
        top.addBond(aC, aH3)


    def add_acetone(res):
        """Add acetone (propanone) atoms and bonds.

        Atom order matches _make_molecule_positions_nm('acetone').
        """

        aC0 = top.addAtom("C0", elem["C"], res)
        aO = top.addAtom("O", elem["O"], res)
        aC1 = top.addAtom("C1", elem["C"], res)
        aC2 = top.addAtom("C2", elem["C"], res)
        hs = [top.addAtom(f"H{i+1}", elem["H"], res) for i in range(6)]

        top.addBond(aC0, aO)
        top.addBond(aC0, aC1)
        top.addBond(aC0, aC2)

        for h in hs[:3]:
            top.addBond(aC1, h)
        for h in hs[3:]:
            top.addBond(aC2, h)


    # Place molecules
    for comp in components:
        name = comp["name"].lower().strip()
        count = int(comp.get("count", 0))
        if count <= 0:
            continue

        template = _make_molecule_positions_nm(name)
        for _ in range(count):
            resname = "HOH" if name in ("water", "h2o") else "MOL"
            res = top.addResidue(resname, chain)

            if name in ("water", "h2o"):
                add_water(res)
            elif name in ("methanol", "meoh", "ch3oh"):
                add_methanol(res)
            elif name in ("acetone", "propanone"):
                add_acetone(res)
            else:
                raise ValueError(f"Unsupported component: {name}")

            shift = rng.uniform(0, box_nm, size=3)
            positions += [p + shift for p in template]

    # periodic vectors
    a = mm.Vec3(box_nm, 0, 0)
    b = mm.Vec3(0, box_nm, 0)
    c = mm.Vec3(0, 0, box_nm)
    top.setPeriodicBoxVectors((a, b, c))

    positions_nm = np.array(positions, dtype=float)
    positions_q = unit.Quantity(positions_nm, unit.nanometer)

    # System and masses
    system = mm.System()
    for atom in top.atoms():
        s = atom.element.symbol
        if s == "H":
            system.addParticle(1.008 * unit.amu)
        elif s == "O":
            system.addParticle(15.999 * unit.amu)
        elif s == "C":
            system.addParticle(12.011 * unit.amu)
        else:
            system.addParticle(12.0 * unit.amu)

    # Nonbonded: toy params
    nb = mm.NonbondedForce()
    nb.setNonbondedMethod(mm.NonbondedForce.CutoffPeriodic)
    nb.setCutoffDistance(float(cutoff_nm) * unit.nanometer)
    nb.setUseDispersionCorrection(False)

    params = {
        "H": (0.0, 0.10, 0.00),
        "O": (-0.4, 0.315, 0.65),
        "C": (0.2, 0.34, 0.45),
    }
    for atom in top.atoms():
        q, sig, eps = params[atom.element.symbol]
        nb.addParticle(q, sig * unit.nanometer, eps * unit.kilojoule_per_mole)

    # Add exceptions for bonded pairs (avoid strongest intramolecular interactions)
    atoms = list(top.atoms())
    idx = {a: i for i, a in enumerate(atoms)}
    for a1, a2 in top.bonds():
        i, j = idx[a1], idx[a2]
        nb.addException(i, j, 0.0, 0.1 * unit.nanometer, 0.0)

    system.addForce(nb)

    # Bonds/angles: harmonic
    bond = mm.HarmonicBondForce()
    ang = mm.HarmonicAngleForce()

    k_bond = 200000.0 * unit.kilojoule_per_mole / unit.nanometer**2
    k_angle = 300.0 * unit.kilojoule_per_mole / unit.radian**2

    for bnd in top.bonds():
        i, j = idx[bnd[0]], idx[bnd[1]]
        pair = {bnd[0].element.symbol, bnd[1].element.symbol}
        if pair == {"O", "H"}:
            r0 = 0.096
        elif pair == {"C", "H"}:
            r0 = 0.109
        elif pair == {"C", "O"}:
            r0 = 0.143
        else:
            r0 = 0.14
        bond.addBond(i, j, r0 * unit.nanometer, k_bond)

    for res in top.residues():
        res_atoms = list(res.atoms())
        adj = {a: set() for a in res_atoms}
        for bnd in top.bonds():
            if bnd[0].residue is res and bnd[1].residue is res:
                adj[bnd[0]].add(bnd[1])
                adj[bnd[1]].add(bnd[0])

        for center in res_atoms:
            neigh = list(adj[center])
            for ii in range(len(neigh)):
                for kk in range(ii + 1, len(neigh)):
                    a1, a3 = neigh[ii], neigh[kk]
                    i, j, k = idx[a1], idx[center], idx[a3]
                    if (
                        center.element.symbol == "O"
                        and a1.element.symbol == "H"
                        and a3.element.symbol == "H"
                    ):
                        theta0 = np.deg2rad(104.52)
                    else:
                        theta0 = np.deg2rad(109.5)
                    ang.addAngle(i, j, k, theta0 * unit.radian, k_angle)

    system.addForce(bond)
    system.addForce(ang)

    system.setDefaultPeriodicBoxVectors(
        a * unit.nanometer, b * unit.nanometer, c * unit.nanometer
    )

    return {
        "n_atoms": int(len(positions_nm)),
        "box_nm": float(box_nm),
        "positions_nm": positions_nm.tolist(),
        "components": components,
        "_topology": top,
        "_system": system,
        "_positions_quantity": positions_q,
    }


def openmm_run_md(
    *,
    built: dict,
    temperature_k: float = 300.0,
    friction_per_ps: float = 5.0,
    dt_fs: float = 0.5,
    nsteps: int = 2000,
    report_interval: int = 5,
    platform: str = "Reference",
    minimize_max_iterations: int = 200,
    collect_positions: bool = False,
) -> dict:
    """Run MD given a built system from `openmm_build_periodic_mixture`.

    Returns velocities as a numpy-like list with shape (T, N, 3) in nm/ps.
    Optionally returns positions (T, N, 3) in nm.
    """

    _require_openmm_and_scipy()

    import openmm as mm
    import openmm.app as app
    from openmm import unit

    top = built.get("_topology")
    system = built.get("_system")
    positions_q = built.get("_positions_quantity")
    if top is None or system is None or positions_q is None:
        raise ValueError(
            "built must be the output of openmm_build_periodic_mixture (missing OpenMM objects)"
        )

    integrator = mm.LangevinMiddleIntegrator(
        float(temperature_k) * unit.kelvin,
        float(friction_per_ps) / unit.picosecond,
        float(dt_fs) * unit.femtosecond,
    )

    # Platform fallback
    try:
        plat = mm.Platform.getPlatformByName(platform)
    except Exception:
        plat = mm.Platform.getPlatformByName("Reference")
        platform = "Reference"

    sim = app.Simulation(top, system, integrator, plat)
    sim.context.setPositions(positions_q)

    sim.minimizeEnergy(maxIterations=int(minimize_max_iterations))
    sim.context.setVelocitiesToTemperature(float(temperature_k) * unit.kelvin)

    vel_frames: list[np.ndarray] = []
    pos_frames: list[np.ndarray] = []

    t0 = time.time()
    nblocks = max(1, int(nsteps) // int(report_interval))
    for _ in range(nblocks):
        sim.step(int(report_interval))
        st = sim.context.getState(getVelocities=True, getPositions=collect_positions)
        v = st.getVelocities(asNumpy=True).value_in_unit(
            unit.nanometer / unit.picosecond
        )
        vel_frames.append(np.asarray(v, dtype=float))
        if collect_positions:
            p = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
            pos_frames.append(np.asarray(p, dtype=float))

    elapsed = time.time() - t0

    vel = np.asarray(vel_frames, dtype=float)

    out: dict[str, Any] = {
        "elapsed_s": float(elapsed),
        "n_atoms": int(vel.shape[1]) if vel.ndim == 3 else int(built.get("n_atoms", 0)),
        "n_frames": int(vel.shape[0]) if vel.ndim == 3 else 0,
        "dt_fs": float(dt_fs),
        "report_interval": int(report_interval),
        "velocities_nm_per_ps": vel.tolist(),
        "settings": {
            "temperature_k": float(temperature_k),
            "friction_per_ps": float(friction_per_ps),
            "dt_fs": float(dt_fs),
            "nsteps": int(nsteps),
            "report_interval": int(report_interval),
            "platform": platform,
            "minimize_max_iterations": int(minimize_max_iterations),
            "collect_positions": bool(collect_positions),
        },
    }

    if collect_positions:
        pos = np.asarray(pos_frames, dtype=float)
        out["positions_nm"] = pos.tolist()

    return out


def analyze_vacf_spectrum(
    *,
    velocities_nm_per_ps: list,
    dt_between_frames_fs: float,
    window: str = "hann",
    peak_max_cm1: float = 4000.0,
    max_peaks: int = 8,
    peak_prominence: float | None = None,
) -> dict:
    """Analyze velocity frames into a VACF-FFT spectrum and peak list.

    Backwards-compatible wrapper.

    New code should prefer:
      - tools.md_analysis.vacf_from_velocities
      - tools.md_analysis.spectrum_from_vacf

    Returns
    -------
    dict compatible with the previous schema:
      - wavenumber_cm1, intensity, vacf, peaks_cm1, peak_intensities
    """

    from .md_analysis import vacf_from_velocities, spectrum_from_vacf

    vres = vacf_from_velocities(velocities=velocities_nm_per_ps, remove_com=True, normalize="v0")
    vacf = vres["vacf"]

    dt_s = float(dt_between_frames_fs) * 1e-15
    spec = spectrum_from_vacf(
        vacf=vacf,
        dt_s=dt_s,
        x_units="cm-1",
        window="hann" if str(window).lower() == "hann" else "none",
        peak_max_x=float(peak_max_cm1),
        max_peaks=int(max_peaks),
        peak_prominence=peak_prominence,
    )

    return {
        "wavenumber_cm1": spec["x"],
        "intensity": spec["y"],
        "vacf": vacf,
        "peaks_cm1": [float(x) for x in spec["peaks_x"]],
        "peak_intensities": [float(y) for y in spec["peaks_y"]],
        "settings": {
            "dt_between_frames_fs": float(dt_between_frames_fs),
            "window": window,
            "peak_max_cm1": float(peak_max_cm1),
            "max_peaks": int(max_peaks),
            "peak_prominence": peak_prominence,
        },
        "notes": {
            "spectrum": "FFT of a simple velocity autocorrelation (VACF), approximate",
            "analysis_backend": "tools.md_analysis + tools.signal",
        },
    }


def openmm_md_workflow_vacf_spectrum(
    *,
    components: list[dict] | None = None,
    box_nm: float = 2.0,
    seed: int = 0,
    cutoff_nm: float = 0.9,
    temperature_k: float = 300.0,
    friction_per_ps: float = 5.0,
    dt_fs: float = 0.5,
    nsteps: int = 2000,
    report_interval: int = 5,
    platform: str = "Reference",
    window: str = "hann",
    peak_max_cm1: float = 4000.0,
    max_peaks: int = 8,
    peak_prominence: float | None = None,
) -> dict:
    """Backwards-compatible convenience wrapper: build + run + analyze."""

    if components is None:
        components = [{"name": "water", "count": 20}, {"name": "methanol", "count": 5}]

    built = openmm_build_periodic_mixture(
        components=components, box_nm=box_nm, seed=seed, cutoff_nm=cutoff_nm
    )
    traj = openmm_run_md(
        built=built,
        temperature_k=temperature_k,
        friction_per_ps=friction_per_ps,
        dt_fs=dt_fs,
        nsteps=nsteps,
        report_interval=report_interval,
        platform=platform,
        collect_positions=False,
    )

    dt_between_frames_fs = float(dt_fs) * float(report_interval)
    spec = analyze_vacf_spectrum(
        velocities_nm_per_ps=traj["velocities_nm_per_ps"],
        dt_between_frames_fs=dt_between_frames_fs,
        window=window,
        peak_max_cm1=peak_max_cm1,
        max_peaks=max_peaks,
        peak_prominence=peak_prominence,
    )

    return {
        "elapsed_s": traj["elapsed_s"],
        "n_atoms": traj["n_atoms"],
        "n_frames": traj["n_frames"],
        "dt_fs": traj["dt_fs"],
        "report_interval": traj["report_interval"],
        "peaks_cm1": spec["peaks_cm1"],
        "peak_intensities": spec["peak_intensities"],
        "settings": {
            "components": components,
            "box_nm": float(box_nm),
            "seed": int(seed),
            "cutoff_nm": float(cutoff_nm),
            "temperature_k": float(temperature_k),
            "friction_per_ps": float(friction_per_ps),
            "dt_fs": float(dt_fs),
            "nsteps": int(nsteps),
            "report_interval": int(report_interval),
            "platform": platform,
            "window": window,
            "peak_max_cm1": float(peak_max_cm1),
            "max_peaks": int(max_peaks),
            "peak_prominence": peak_prominence,
        },
        "notes": {
            "force_field": "toy LJ+Coulomb + harmonic bonds/angles + bonded-pair NB exceptions",
            "spectrum": spec["notes"]["spectrum"],
            "analysis_backend": spec["notes"].get("analysis_backend", "unknown"),
        },
    }
