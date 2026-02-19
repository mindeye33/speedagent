import pytest

from agent import Router, QCSubagent, MDSubagent


def test_router_routes_qc_by_default():
    r = Router()
    assert r.route(user_text="Please run a DFT single point") == "qc"


def test_router_routes_md_when_md_keywords_present():
    r = Router()
    assert r.route(user_text="Run an OpenMM MD trajectory and VACF") == "md"


def test_qc_subagent_plan_normalizes_defaults():
    qc = QCSubagent()
    plan = qc.plan({"xyz": "2\n\nH 0 0 0\nH 0 0 0.74\n"})
    assert plan["subagent"] == "qc"
    assert plan["tool"] == "qc.qc_pyscf_run"
    assert plan["args"]["task"] == "single_point"
    assert plan["args"]["basis"] == "6-31g"
    assert plan["args"]["xc"] == "B3LYP"


def test_md_subagent_plan_provides_minimal_components_default():
    md = MDSubagent()
    plan = md.plan({})
    assert plan["subagent"] == "md"
    assert plan["tool"] == "md.md_openmm_md_workflow_vacf_spectrum"
    assert isinstance(plan["args"]["components"], list)
    assert plan["args"]["components"][0]["name"] == "water"


def test_qc_subagent_requires_xyz():
    qc = QCSubagent()
    with pytest.raises(ValueError):
        qc.plan({"task": "single_point"})
