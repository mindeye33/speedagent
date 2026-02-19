import numpy as np

from tools.md import analyze_vacf_spectrum


def test_analyze_vacf_spectrum_shapes_and_peaks():
    # Synthetic velocity: sinusoidal mode at 1000 cm^-1
    # Build per-atom signals with different phases so COM subtraction doesn't kill everything.
    c_cm_s = 2.99792458e10
    wn = 1000.0
    f_hz = wn * c_cm_s

    dt_fs = 1.0
    dt_s = dt_fs * 1e-15
    T = 4096

    t = np.arange(T) * dt_s
    N = 6
    phases = np.linspace(0.0, 2 * np.pi, N, endpoint=False)

    v = np.zeros((T, N, 3), dtype=float)
    for i, ph in enumerate(phases):
        v[:, i, 0] = np.sin(2 * np.pi * f_hz * t + ph)

    res = analyze_vacf_spectrum(
        velocities_nm_per_ps=v.tolist(),
        dt_between_frames_fs=dt_fs,
        window="hann",
        peak_max_cm1=2000.0,
        max_peaks=5,
        peak_prominence=None,
    )

    assert "peaks_cm1" in res
    assert len(res["peaks_cm1"]) >= 1
    assert any(abs(p - wn) < 50 for p in res["peaks_cm1"])  # coarse tolerance
