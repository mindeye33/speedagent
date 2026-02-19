
# speedAgent

Minimal `pydantic_ai` chat agent with a Python REPL tool.

## Security warning (USE AT YOUR OWN RISK)

This project intentionally exposes powerful tools to the model, including:

- an unrestricted Python REPL (arbitrary Python execution)
- an unrestricted `bash` command runner (arbitrary shell command execution)

That means prompts (and model/tool output) can:

- read/modify/delete files accessible to your user account
- execute arbitrary programs and install packages
- access environment variables and other secrets available on the machine
- make network requests (directly or via installed tools)

Only run this on a machine/environment you trust and are willing to risk. Do not deploy it to the public internet. Prefer a sandboxed, disposable environment (e.g., a container/VM) with minimal permissions and no sensitive credentials.

## Setup

Sync dependencies:

```bash
uv sync
```

## Run

Set your model API key (OpenAI backend by default):

```bash
export OPENAI_API_KEY=... 
```

Or create a `.env` file in the project root:

```bash
OPENAI_API_KEY=...
```

Start the CLI:

```bash
uv run uvicorn agent:app --reload
```

If `OPENAI_API_KEY` is not set, the app will show the UI, but not interact with the model.

## Example prompt

Ask it to run a small PySCF calculation via the `python_repl` tool, e.g.:

"Compute RHF energy for H2 at 0.74 Å in STO-3G and report the total energy."


