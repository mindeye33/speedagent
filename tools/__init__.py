from __future__ import annotations

from functools import wraps
import inspect
import warnings
from importlib import import_module
from pkgutil import iter_modules
from typing import Iterable, Any


def load_tools(
    *,
    include_modules: Iterable[str] | None = None,
    exclude_modules: Iterable[str] = (),
    name_style: str = "module.func",  # safer than "func"
    best_effort: bool = True,
) -> list[Any]:
    """Discover and wrap eligible functions from all modules in the `tools` package.

    This returns a list of pydantic_ai.Tool objects *if pydantic_ai is installed*.
    In test/minimal environments where pydantic_ai isn't available, this will
    raise on use; importing `tools` remains possible.

    Eligibility:
      - top-level functions defined in the module (not imported)
      - name does not start with "_"

    Parameters
    ----------
    include_modules:
        Optional allow-list of module names.
    exclude_modules:
        Block-list of module names.
    name_style:
        "module.func" -> tool name is "{module}_{func}".
        "func" -> tool name is just "{func}" (risk of collisions).
    best_effort:
        If True, skip modules that fail to import (e.g., optional deps), with a warning.
        If False, raise.
    """

    try:
        from pydantic_ai import ModelRetry, Tool  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "pydantic_ai is required to load tools (Tool wrapper missing)."
        ) from e

    package = __name__  # "tools"
    pkg = import_module(package)
    pkg_path = pkg.__path__  # type: ignore[attr-defined]

    include_set = set(include_modules) if include_modules else None
    exclude_set = set(exclude_modules)

    out: list[Any] = []
    seen: set[str] = set()

    mods = [m for m in iter_modules(pkg_path) if not m.ispkg]
    mods.sort(key=lambda m: m.name)

    for m in mods:
        mod_name = m.name
        if mod_name.startswith("_") or mod_name in exclude_set:
            continue
        if include_set is not None and mod_name not in include_set:
            continue

        try:
            mod = import_module(f"{package}.{mod_name}")
        except Exception as e:
            if best_effort:
                warnings.warn(
                    f"Skipping tools module '{mod_name}' due to import error: {e}",
                    RuntimeWarning,
                )
                continue
            raise
        fns = [
            (fn_name, fn)
            for fn_name, fn in inspect.getmembers(mod, inspect.isfunction)
        ]
        fns.sort(key=lambda t: t[0])

        def with_model_retry(func):
            """Wrap a tool function so PydanticAI can retry at the model level.

            We intentionally *do not* locally retry here. Instead, we raise
            `ModelRetry` so the LLM can see the error and potentially adjust
            inputs and retry.
            """

            @wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    return func(*args, **kwargs)
                except ModelRetry:
                    raise
                except Exception as e:  # pragma: no cover
                    raise ModelRetry(
                        f"Tool '{getattr(func, '__name__', 'tool')}' failed: {e}"
                    ) from e

            return wrapper

        fns = [(name, with_model_retry(fn)) for name, fn in fns]

        for fn_name, fn in fns:
            if fn_name.startswith("_"):
                continue
            if fn.__module__ != mod.__name__:
                continue

            tool_name = f"{mod_name}_{fn_name}" if name_style == "module.func" else fn_name

            if tool_name in seen:
                raise RuntimeError(
                    f"Tool name collision: '{tool_name}'. "
                    f"Use name_style='module.func' or rename functions."
                )
            seen.add(tool_name)

            # PydanticAI has used both `retries` and `max_retries` across versions.
            tool_kwargs: dict[str, Any] = {"name": tool_name}
            tool_sig = inspect.signature(Tool)
            if "retries" in tool_sig.parameters:
                tool_kwargs["retries"] = 5
            elif "max_retries" in tool_sig.parameters:
                tool_kwargs["max_retries"] = 5
            out.append(Tool(fn, **tool_kwargs))

    return out
