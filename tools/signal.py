"""General signal-processing utilities.

This module is intentionally domain-agnostic: it operates on time-series arrays
and returns spectra + peak picks. MD-specific tools can reuse it for VACF or
other correlation functions.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .schema import _as_list


def fft_spectrum(
    *,
    y: list[float] | Any,
    dt_s: float,
    window: Literal["hann", "none"] = "hann",
    detrend: Literal["none", "mean"] = "mean",
) -> dict:
    """Compute a one-sided FFT power-like spectrum from a real signal.

    Parameters
    ----------
    y:
        Real-valued time series.
    dt_s:
        Sampling interval in seconds.
    window:
        Window function to apply.
    detrend:
        Remove the mean before FFT.

    Returns
    -------
    dict with keys:
      - freq_hz
      - spectrum
      - settings
    """

    v = np.asarray(y, dtype=float).reshape(-1)
    if v.size < 2:
        raise ValueError("y must have at least 2 samples")

    if detrend == "mean":
        v = v - v.mean()
    elif detrend != "none":
        raise ValueError("detrend must be 'none' or 'mean'")

    if window == "hann":
        try:
            from scipy.signal import windows

            w = windows.hann(len(v))
        except Exception as e:  # pragma: no cover
            raise RuntimeError("SciPy required for hann window; install scipy.") from e
    elif window == "none":
        w = np.ones_like(v)
    else:
        raise ValueError("window must be 'hann' or 'none'")

    vw = v * w
    spec = np.real(np.fft.rfft(vw))
    freq = np.fft.rfftfreq(len(vw), d=float(dt_s))

    # Shift to nonnegative baseline for convenience
    spec2 = spec - float(spec.min())

    return {
        "freq_hz": _as_list(freq),
        "spectrum": _as_list(spec2),
        "settings": {
            "dt_s": float(dt_s),
            "window": window,
            "detrend": detrend,
            "n": int(len(v)),
        },
    }


def pick_peaks(
    *,
    x: list[float] | Any,
    y: list[float] | Any,
    max_peaks: int = 8,
    prominence: float | None = None,
) -> dict:
    """Pick peaks in (x,y) data.

    Uses scipy.signal.find_peaks when available.
    If no peaks are found, falls back to top-k by intensity.
    """

    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    if xv.shape != yv.shape:
        raise ValueError("x and y must have the same shape")

    try:
        from scipy.signal import find_peaks

        idx, _props = find_peaks(yv, prominence=prominence)
        idx = np.asarray(idx, dtype=int)
    except Exception:
        idx = np.array([], dtype=int)

    if idx.size == 0:
        k = min(int(max_peaks), int(yv.size))
        sel = np.argsort(yv)[-k:][::-1]
    else:
        k = min(int(max_peaks), int(idx.size))
        order = np.argsort(yv[idx])[-k:][::-1]
        sel = idx[order]

    return {
        "peaks_x": _as_list(xv[sel]),
        "peaks_y": _as_list(yv[sel]),
        "settings": {"max_peaks": int(max_peaks), "prominence": prominence},
    }
