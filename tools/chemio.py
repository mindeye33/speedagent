"""General chemistry I/O helpers.

These are intentionally dependency-light utilities used by multiple tools.
The goal is to normalize geometry parsing/formatting so computational tools
can focus on the method rather than string wrangling.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Atom:
    symbol: str
    x: float
    y: float
    z: float


def parse_xyz(xyz: str) -> list[Atom]:
    """Parse XYZ (Angstrom) with or without the 2-line XYZ header.

    Accepts either:
      - Standard XYZ: first line natoms (int), second line comment
      - Headerless XYZ: each line is "Sym x y z"

    Header validation
    -----------------
    If the first line is an integer, we interpret it as a standard XYZ header.
    In that case we *validate* that the number of coordinate lines matches
    natoms, raising ValueError if it does not.

    This avoids a common footgun where malformed XYZ strings silently drop
    atoms (e.g., natoms present but fewer coordinate lines than natoms).
    """

    lines = [ln.strip() for ln in str(xyz).strip().splitlines() if ln.strip()]
    if not lines:
        raise ValueError("Empty XYZ")

    start = 0
    natoms_header: int | None = None
    try:
        natoms_header = int(lines[0])
        start = 2
    except Exception:
        start = 0
        natoms_header = None

    coord_lines = lines[start:]

    if natoms_header is not None:
        if natoms_header < 0:
            raise ValueError("XYZ natoms header must be nonnegative")
        if len(coord_lines) != natoms_header:
            raise ValueError(
                f"XYZ natoms header says {natoms_header} atoms but found {len(coord_lines)} coordinate lines"
            )

    atoms: list[Atom] = []
    for ln in coord_lines:
        parts = ln.split()
        if len(parts) < 4:
            raise ValueError(f"Bad XYZ line: {ln!r}")
        sym = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(Atom(sym, float(x), float(y), float(z)))

    if not atoms:
        raise ValueError("No atoms parsed from XYZ")
    return atoms


def format_xyz(atoms: list[Atom], comment: str = "") -> str:
    """Format atoms as a standard XYZ string (Angstrom)."""

    lines = [str(len(atoms)), comment]
    for a in atoms:
        lines.append(f"{a.symbol:2s} {a.x: .10f} {a.y: .10f} {a.z: .10f}")
    return "\n".join(lines)


def to_pyscf_atom_block(atoms: list[Atom]) -> str:
    """Convert atoms to a PySCF 'atom' block."""

    return "\n".join(f"{a.symbol} {a.x} {a.y} {a.z}" for a in atoms)
