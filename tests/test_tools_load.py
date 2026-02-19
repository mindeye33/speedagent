import pytest


def test_load_tools_smoke():
    # pydantic_ai may not be available in the pytest interpreter on some setups.
    tools = pytest.importorskip("tools")
    try:
        ts = tools.load_tools(name_style="module.func", best_effort=True)
    except RuntimeError:
        pytest.skip("pydantic_ai Tool wrapper not available in this environment")

    names = [t.name for t in ts]
    assert "arithmetic_add" in names
    assert "qc_pyscf_run" in names
