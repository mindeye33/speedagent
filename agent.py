import dotenv
dotenv.load_dotenv()

from pydantic_ai import Agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from pydantic_ai.ext.langchain import tool_from_langchain
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai import ModelRetry
import subprocess

python_tool = tool_from_langchain(PythonREPLTool())

agent = Agent(
    "openai-responses:gpt-5.2",
    instructions="""
You're a computational chemistry agent that interprets user intent and executes computational chemistry calculations by writing your own python functions and tools. You can:
* execute your Python code via the `Python_REPL` tool. 
* use web search, for example: for code examples of pyscf on https://github.com/pyscf/pyscf/tree/master/examples. 
* execute bash commands via the `bash` tool, e.g. to install missing Python packages with "uv" like: "uv add numpy", etc.
Only save new results/figures in "work_dir".
""",
    tools=[python_tool],
    builtin_tools=[WebSearchTool()],
    retries=5,
)

@agent.tool_plain(retries=5)
def bash(command: str) -> str:
    print(f"Ran bash command: {command}")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        print(f"Ran bash command: {command}\nReturn code: {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}")
    except Exception as e:
        raise ModelRetry(f"Error running bash command: {e}") from e
    return result.stdout + result.stderr

app = agent.to_web(
    models={"GPT 5": "openai-responses:gpt-5.2"},
)
