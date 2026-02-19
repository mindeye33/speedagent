"""MD analysis utilities.

These functions are designed to be reusable beyond the toy OpenMM builder.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from .schema import _as_list, _validate_array_shape
from .signal import fft_spectrum, pick_peaks


def vacf_from_velocities(
    *,
    velocities: list | Any,
    remove_com: bool = True,
    normalize: Literal["v0", "none"] = "v0",
) -> dict:
    """Compute a simple velocity autocorrelation function (VACF).

    Parameters
    ----------
    velocities:
        Array-like of shape (T, N, 3) in any velocity units.
    remove_com:
        Subtract center-of-mass/mean velocity per frame.
    normalize:
        - 'v0': divide by v(0)·v(0)
        - 'none': raw dot products

    Returns
    -------
    dict with keys: vacf
    """

    _validate_array_shape(velocities, (None, None, 3), name="velocities")
    v = np.asarray(velocities, dtype=float)

    if remove_com:
        v = v - v.mean(axis=1, keepdims=True)

    v0 = v[0].reshape(-1)
    vt = v.reshape(v.shape[0], -1)
    vacf = vt @ v0

    if normalize == "v0":
        denom = float(v0 @ v0)
        if denom == 0.0:
            raise ValueError("Zero initial velocity norm; cannot normalize VACF")
        vacf = vacf / denom
    elif normalize != "none":
        raise ValueError("normalize must be 'v0' or 'none'")

    return {
        "vacf": _as_list(vacf),
        "settings": {"remove_com": bool(remove_com), "normalize": normalize},
    }


def spectrum_from_vacf(
    *,
    vacf: list[float] | Any,
    dt_s: float,
    x_units: Literal["hz", "cm-1"] = "cm-1",
    window: Literal["hann", "none"] = "hann",
    peak_max_x: float | None = 4000.0,
    max_peaks: int = 8,
    peak_prominence: float | None = None,
) -> dict:
    """Compute an FFT spectrum from a VACF and pick peaks.

    Parameters
    ----------
    vacf:
        1D VACF.
    dt_s:
        sampling interval in seconds.
    x_units:
        return x-axis as 'hz' or 'cm-1'.
    peak_max_x:
        optional upper bound for peak picking in chosen x units.
    """

    spec = fft_spectrum(y=vacf, dt_s=dt_s, window=window, detrend="none")
    freq = np.asarray(spec["freq_hz"], dtype=float)
    inten = np.asarray(spec["spectrum"], dtype=float)

    if x_units == "hz":
        x = freq
    elif x_units in ("cm-1", "cm1"):
        x = freq / 2.99792458e10
        x_units = "cm-1"
    else:
        raise ValueError("x_units must be 'hz' or 'cm-1'")

    # Apply peak window mask
    if peak_max_x is not None:
        mask = (x > 0) & (x < float(peak_max_x))
        x2 = x[mask]
        y2 = inten[mask]
    else:
        x2, y2 = x, inten

    peaks = pick_peaks(x=x2, y=y2, max_peaks=max_peaks, prominence=peak_prominence)

    return {
        "x": _as_list(x2),
        "y": _as_list(y2),
        "x_units": x_units,
        "peaks_x": peaks["peaks_x"],
        "peaks_y": peaks["peaks_y"],
        "settings": {
            "dt_s": float(dt_s),
            "window": window,
            "x_units": x_units,
            "peak_max_x": peak_max_x,
            "max_peaks": int(max_peaks),
            "peak_prominence": peak_prominence,
        },
    }
