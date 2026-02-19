import numpy as np

from tools.chemgeom import rmsd, kabsch_align, geometry_basic


def test_rmsd_zero_after_alignment_translation_only():
    # Proper XYZ header with explicit comment line.
    xyz1 = """3
comment
O 0.0 0.0 0.0
H 0.0 0.0 1.0
H 0.0 1.0 0.0
"""
    xyz2 = """3
comment
O 1.0 2.0 3.0
H 1.0 2.0 4.0
H 1.0 3.0 3.0
"""
    out = rmsd(xyz_ref=xyz1, xyz_mob=xyz2, align=True)
    assert out["aligned"] is True
    assert out["rmsd_angstrom"] == pytest_approx(0.0, 1e-10)


def test_kabsch_align_rotation_only():
    # Proper XYZ header with explicit comment line.
    xyz_ref = """2
comment
H 1.0 0.0 0.0
H 0.0 1.0 0.0
"""
    xyz_mob = """2
comment
H 0.0 1.0 0.0
H -1.0 0.0 0.0
"""
    res = kabsch_align(xyz_ref=xyz_ref, xyz_mob=xyz_mob)
    assert res["rmsd_angstrom"] == pytest_approx(0.0, 1e-10)
    R = np.array(res["rotation"], dtype=float)
    assert R.shape == (3, 3)


def test_geometry_basic_distance_matrix_symmetry():
    # Headerless XYZ still supported.
    xyz = """He 0 0 0
He 0 0 1
"""
    g = geometry_basic(xyz=xyz)
    d = np.array(g["distance_matrix_angstrom"], dtype=float)
    assert d.shape == (2, 2)
    assert d[0, 1] == pytest_approx(1.0, 1e-12)
    assert d[1, 0] == pytest_approx(1.0, 1e-12)


def pytest_approx(x, tol):
    # lightweight local approx to avoid importing pytest directly in this sandbox
    return Approx(x, tol)


class Approx:
    def __init__(self, x, tol):
        self.x = x
        self.tol = tol

    def __eq__(self, other):
        return abs(float(other) - float(self.x)) <= float(self.tol)
