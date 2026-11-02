"""Equations (1)-(4) remove known offset, amplitude and phase drift."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drift_correction import subtract_row_median, normalize_amplitude, phase_drift, correct_series
from make_test_data import make_frames


def test_row_median_removes_a_row_dependent_offset():
    rng = np.random.default_rng(0)
    signal = rng.normal(size=(50, 60)) + 1j * rng.normal(size=(50, 60))
    offset = (np.linspace(0, 5, 50) + 3j)[:, None]          # different in every row
    a = subtract_row_median(signal + offset, smooth_rows=1)
    b = subtract_row_median(signal, smooth_rows=1)
    assert np.allclose(a, b)


def test_normalized_frame_has_mean_amplitude_one():
    e = 7.3 * np.exp(1j * np.random.default_rng(0).uniform(0, 6, (30, 30)))
    assert np.isclose(np.mean(np.abs(normalize_amplitude(e))), 1)


def test_phase_drift_is_found():
    e = np.exp(1j * np.linspace(0, 20, 900)).reshape(30, 30)
    assert np.isclose(phase_drift(e, e * np.exp(-1j * 0.8)), 0.8)


def test_corrected_frames_agree_without_noise():
    # Frames with different offset, amplitude and phase drift, shifts undone exactly
    _, ez, truth = make_frames(n_frames=6, noise=0.0, seed=3)
    frames = []
    for m in range(6):
        sx, sy = map(int, truth["shift"][m])
        frames.append(np.roll(ez[m], (sy, sx), axis=(0, 1)))
    frames = np.array(frames)[:, 10:-10, 10:-10]            # drop the rolled border
    corrected, dphi = correct_series(frames)

    # Phase drift recovered (Eq. 3): dphi_m = phase_1 - phase_m
    expected = np.angle(np.exp(1j * (truth["phase"][0] - np.array(truth["phase"]))))
    assert np.allclose(np.angle(np.exp(1j * (dphi - expected))), 0, atol=0.05)

    # Frames agree after correction. Not exactly: the row median of Eq. (1) is
    # taken separately for Re and Im, which depends slightly on each frame's
    # global phase, so a residual of a few percent of the signal remains.
    spread = np.median(np.abs(corrected - corrected[0])) / np.median(np.abs(corrected[0]))
    assert spread < 0.1, f"frames differ by {spread:.3f} of the signal after correction"


if __name__ == "__main__":
    # Run the tests of this file only
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS ", name)
