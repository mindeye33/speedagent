
# speedAgent

Minimal `pydantic_ai` chat agent with a Python REPL tool.

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

Optional model override:

```bash
export SPEEDAGENT_MODEL=gpt-4.1-mini
```

Start the CLI:

```bash
uv run uvicorn agent:app --reload
```

If `OPENAI_API_KEY` is not set, the app will show the UI, but not interact with the model.

## Example prompt

Ask it to run a small PySCF calculation via the `python_repl` tool, e.g.:

"Compute RHF energy for H2 at 0.74 Å in STO-3G and report the total energy."


