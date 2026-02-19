

def test_rdkit_conformers_from_smiles_basic_with_rmsd_and_labels():
    from tools.rdkit_tools import rdkit_conformers_from_smiles

    res = rdkit_conformers_from_smiles(
        "CCO",  # ethanol
        n_confs=25,
        seed=1,
        embed_prune_rmsd=0.2,
        cluster_rmsd_thresh=0.5,
        max_return=10,
        return_rmsd_matrix=True,
        return_cluster_labels=True,
        cluster_method="butina",
    )

    assert res["smiles"] == "CCO"
    assert res["n_embedded"] >= 1
    assert 1 <= res["n_returned"] <= 10

    # RMSD matrix
    rmsd = res["rmsd_matrix"]
    assert isinstance(rmsd, list)
    n = res["n_returned"]
    assert len(rmsd) == n
    assert all(len(row) == n for row in rmsd)
    assert all(abs(rmsd[i][i]) < 1e-12 for i in range(n))

    # labels
    labels = res["cluster_labels"]
    assert isinstance(labels, list)
    assert len(labels) == n
    assert all(isinstance(x, int) for x in labels)

    # conformer records contain xyz and per-conformer cluster
    c0 = res["conformers"][0]
    assert "xyz" in c0 and isinstance(c0["xyz"], str)
    xyz_lines = c0["xyz"].strip().splitlines()
    assert int(xyz_lines[0]) > 0
    assert len(xyz_lines) == int(xyz_lines[0]) + 2
    assert c0["cluster"] == labels[0]
