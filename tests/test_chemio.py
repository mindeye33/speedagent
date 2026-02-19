from tools.chemio import parse_xyz, format_xyz


def test_parse_xyz_headerless():
    xyz = """\
    O 0.0 0.0 0.0
    H 0.0 0.0 1.0
    H 0.0 1.0 0.0
    """
    atoms = parse_xyz(xyz)
    assert len(atoms) == 3
    assert atoms[0].symbol == "O"


def test_parse_xyz_with_header():
    xyz = """\
    2
    comment
    H 0 0 0
    H 0 0 0.74
    """
    atoms = parse_xyz(xyz)
    assert len(atoms) == 2
    assert atoms[1].z == 0.74


def test_format_xyz_roundtrip_count():
    xyz = """\
    1
    x
    He 0 0 0
    """
    atoms = parse_xyz(xyz)
    out = format_xyz(atoms, comment="y")
    assert out.splitlines()[0].strip() == "1"
