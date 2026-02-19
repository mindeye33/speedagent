import dotenv
dotenv.load_dotenv()

from pydantic_ai import Agent
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional

from langchain_experimental.tools.python.tool import PythonREPLTool
from pydantic_ai.ext.langchain import tool_from_langchain
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai import ModelRetry
import tools
import subprocess

import logfire
logfire.configure()
logfire.instrument_pydantic_ai()
logfire.instrument_httpx(capture_all=True)

python_tool = tool_from_langchain(PythonREPLTool())


# ---------------------------
# Subagent framework (QC / MD)
# ---------------------------

TaskType = Literal["qc", "md", "auto"]


@dataclass
class QCDefaults:
    basis: str = "6-31g"
    xc: str = "B3LYP"
    driver_closed_shell: str = "RKS"
    driver_open_shell: str = "UKS"
    charge: int = 0
    spin: int = 0


@dataclass
class MDDefaults:
    # Build
    box_nm: float = 2.0
    cutoff_nm: float = 0.9
    seed: int = 0
    # Dynamics
    temperature_k: float = 300.0
    friction_per_ps: float = 5.0
    dt_fs: float = 0.5
    nsteps: int = 2000
    report_interval: int = 5
    platform: str = "Reference"
    # Analysis
    window: str = "hann"
    peak_max_cm1: float = 4000.0
    max_peaks: int = 8
    peak_prominence: Optional[float] = None


class QCSubagent:
    """Quantum chemistry specialist.

    Focuses on electronic structure computations via tools.qc.qc_pyscf_run.
    Provides light normalization and sensible defaults.
    """

    def __init__(self, defaults: Optional[QCDefaults] = None):
        self.defaults = defaults or QCDefaults()

    def normalize_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        if "xyz" not in task or not isinstance(task["xyz"], str) or not task["xyz"].strip():
            raise ValueError("QC task must include non-empty 'xyz' string")

        qc_task = task.get("task", "single_point")
        if qc_task not in {"single_point", "optimize_geometry"}:
            raise ValueError("QC task 'task' must be 'single_point' or 'optimize_geometry'")

        charge = int(task.get("charge", self.defaults.charge))
        spin = int(task.get("spin", self.defaults.spin))
        driver = task.get(
            "driver",
            self.defaults.driver_open_shell if spin != 0 else self.defaults.driver_closed_shell,
        )

        return {
            "task": qc_task,
            "xyz": task["xyz"],
            "charge": charge,
            "spin": spin,
            "basis": task.get("basis", self.defaults.basis),
            "driver": driver,
            "xc": task.get("xc", self.defaults.xc),
            "maxsteps": int(task.get("maxsteps", 50)),
            "verbose": int(task.get("verbose", 0)),
        }

    def tool_name(self) -> str:
        # tools.load_tools uses name_style="module.func"; qc runner is tools/qc.py: qc_pyscf_run
        return "qc.qc_pyscf_run"

    def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_task(task)
        return {"tool": self.tool_name(), "args": normalized, "subagent": "qc"}


class MDSubagent:
    """Molecular dynamics specialist.

    Focuses on OpenMM toy MD workflows and spectral analysis.
    Uses the one-shot md_openmm_md_workflow_vacf_spectrum by default.
    """

    def __init__(self, defaults: Optional[MDDefaults] = None):
        self.defaults = defaults or MDDefaults()

    def normalize_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        components = task.get("components")
        if not components or not isinstance(components, list):
            # Minimal default mixture if user forgot
            components = [{"name": "water", "count": 64}]

        return {
            "components": components,
            "box_nm": float(task.get("box_nm", self.defaults.box_nm)),
            "seed": int(task.get("seed", self.defaults.seed)),
            "cutoff_nm": float(task.get("cutoff_nm", self.defaults.cutoff_nm)),
            "temperature_k": float(task.get("temperature_k", self.defaults.temperature_k)),
            "friction_per_ps": float(task.get("friction_per_ps", self.defaults.friction_per_ps)),
            "dt_fs": float(task.get("dt_fs", self.defaults.dt_fs)),
            "nsteps": int(task.get("nsteps", self.defaults.nsteps)),
            "report_interval": int(task.get("report_interval", self.defaults.report_interval)),
            "platform": task.get("platform", self.defaults.platform),
            "window": task.get("window", self.defaults.window),
            "peak_max_cm1": float(task.get("peak_max_cm1", self.defaults.peak_max_cm1)),
            "max_peaks": int(task.get("max_peaks", self.defaults.max_peaks)),
            "peak_prominence": task.get("peak_prominence", self.defaults.peak_prominence),
        }

    def tool_name(self) -> str:
        return "md.md_openmm_md_workflow_vacf_spectrum"

    def plan(self, task: Dict[str, Any]) -> Dict[str, Any]:
        normalized = self.normalize_task(task)
        return {"tool": self.tool_name(), "args": normalized, "subagent": "md"}


class Router:
    """Small, debuggable router between QC and MD subagents."""

    QC_HINTS = (
        "dft",
        "basis",
        "functional",
        "xc",
        "scf",
        "pyscf",
        "optimize geometry",
        "geometry optimization",
        "single point",
        "homo",
        "lumo",
        "orbitals",
    )
    MD_HINTS = (
        "md",
        "openmm",
        "trajectory",
        "langevin",
        "vacf",
        "rdf",
        "diffusion",
        "equilibr",
        "npt",
        "nvt",
        "thermostat",
    )

    def __init__(self, qc: Optional[QCSubagent] = None, md: Optional[MDSubagent] = None):
        self.qc = qc or QCSubagent()
        self.md = md or MDSubagent()

    def route(self, user_text: str, task_type: TaskType = "auto") -> TaskType:
        if task_type in {"qc", "md"}:
            return task_type
        t = (user_text or "").lower()
        qc_score = sum(1 for h in self.QC_HINTS if h in t)
        md_score = sum(1 for h in self.MD_HINTS if h in t)
        if md_score > qc_score:
            return "md"
        return "qc"


router = Router()


agent = Agent(
    "openai-responses:gpt-5.2",
    instructions="""
You're a computational chemistry agent that interprets user intent and executes computational chemistry calculations by writing your own python functions and tools. You can:
* execute your Python code via the `Python_REPL` tool.
* use web search, for example: for code examples of pyscf on https://github.com/pyscf/pyscf/tree/master/examples.
* execute bash commands via the `bash` tool, e.g. to install missing Python packages with "uv" like: "uv add numpy", etc.
* add new tools to yourself, by adding them to the `tools` directory as top-level functions. After adding any new tool, make a unit test in the tests directory, and make sure it passes the tests.
YOUR OWN SOURCE CODE is in the `agent.py` file, so you can modify your own instructions.
Only save new results/figures in "work_dir".

Internal structure note:
- You have two specialized subagents: QC (quantum chemistry) and MD (molecular dynamics).
- Use them to normalize parameters and select appropriate tools.
    """,
    tools=[python_tool] + tools.load_tools(name_style="module.func"),
    builtin_tools=[WebSearchTool()],
    retries=5,
)


@agent.tool_plain(retries=5)
def bash(command: str) -> str:
    print(f"Ran bash command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(
            f"Ran bash command: {command}\nReturn code: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
        )
        return result.stdout + result.stderr
    except Exception as e:
        raise ModelRetry(f"Error running bash command: {e}") from e


@agent.tool_plain
def plan_subagent_task(
    user_text: str,
    task: Dict[str, Any],
    task_type: TaskType = "auto",
) -> Dict[str, Any]:
    """Return a tool-call plan using the QC or MD subagent.

    This does not execute the plan; it returns a dict with:
      - subagent: 'qc' | 'md'
      - tool: module.func tool name
      - args: normalized tool arguments

    Use-case: debugging / reproducible orchestration.
    """
    chosen = router.route(user_text=user_text, task_type=task_type)
    try:
        if chosen == "md":
            return router.md.plan(task)
        return router.qc.plan(task)
    except Exception as e:
        raise ModelRetry(f"Failed to plan subagent task: {e}") from e


app = agent.to_web(
    models={"GPT 5": "openai-responses:gpt-5.2"},
)
