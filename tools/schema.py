"""Lightweight schema/validation utilities shared across tools.

The tool functions in this repository intentionally return JSON-serializable
structures (dict/list/float/int/str/bool/None). In practice, each tool ends up
re-implementing a bit of the same glue:

  - validating array shapes
  - coercing numpy arrays to lists
  - emitting consistent metadata blocks

This module centralizes those patterns so the *domain* tools (QC/MD/RDKit) can
stay focused and the results are more uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


def _as_list(x: Any) -> Any:
    """Recursively convert numpy scalars/arrays (and similar) to JSONable lists.

    - numpy arrays -> list
    - numpy scalars -> Python scalars
    - dict/list/tuple -> recursive

    For unknown objects, returns as-is.
    """

    # numpy is optional; import lazily
    try:  # pragma: no cover
        import numpy as np

        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer, np.bool_)):
            return x.item()
    except Exception:
        pass

    if isinstance(x, dict):
        return {k: _as_list(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_as_list(v) for v in x]
    return x


def _validate_array_shape(arr: Any, shape: Sequence[int | None], name: str) -> None:
    """Validate that `arr` (array-like) matches a given shape.

    Parameters
    ----------
    arr:
        Any object convertible to a nested list/ndarray.
    shape:
        Expected shape, with None as a wildcard dimension.
        Example: (None, None, 3) for (T, N, 3)
    name:
        Name used in error messages.
    """

    try:
        import numpy as np

        a = np.asarray(arr)
        actual = a.shape
    except Exception:
        actual = _infer_nested_shape(arr)

    if len(actual) != len(shape):
        raise ValueError(f"{name} must have {len(shape)} dims; got shape {actual}")

    for i, (got, exp) in enumerate(zip(actual, shape)):
        if exp is None:
            continue
        if int(got) != int(exp):
            raise ValueError(f"{name} dim {i} must be {exp}; got shape {actual}")


def _infer_nested_shape(x: Any) -> tuple[int, ...]:
    if not isinstance(x, list):
        return ()
    if len(x) == 0:
        return (0,)
    return (len(x),) + _infer_nested_shape(x[0])


@dataclass(frozen=True)
class ResultMeta:
    """A tiny metadata block that tools can embed in outputs."""

    tool: str
    units: dict[str, str] | None = None
    notes: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"tool": self.tool}
        if self.units:
            d["units"] = dict(self.units)
        if self.notes:
            d["notes"] = dict(self.notes)
        return d
