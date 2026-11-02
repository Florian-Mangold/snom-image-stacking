"""The complete workflow on synthetic and on real example data."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from load_gsf import load_series
from align import align_series
from drift_correction import correct_series, subtract_row_median
from stack import stack, mean_stack
from test_stack import figure_of_merit
from make_test_data import make_frames


def similarity(a, b):
    """|<a, b>| / (|a| |b|): 1 for identical images up to a global complex factor."""
    return np.abs(np.vdot(a.ravel(), b.ravel())) / np.linalg.norm(a) / np.linalg.norm(b)


def test_synthetic_stack_recovers_the_signal():
    afm, ez, truth = make_frames(n_frames=16, noise=1.0, outlier_frame=5, seed=4)
    afm, ez, _ = align_series(afm, ez)
    ez, _ = correct_series(ez)
    inner = (slice(10, -10), slice(10, -10))
    signal = subtract_row_median(truth["signal"])[inner]   # what Eq. (1) leaves of it
    single = similarity(signal, ez[0][inner])
    plain = similarity(signal, mean_stack(ez)[inner])
    trimmed = similarity(signal, stack(ez)[inner])
    assert trimmed > 0.85 and trimmed > single + 0.2, (single, trimmed)
    assert trimmed > plain                                 # outlier frame 5 is rejected


def test_example_data_noise_drops_with_stacking():
    root = os.path.join(os.path.dirname(__file__), "..", "example_data")
    folders = [os.path.join(root, f) for f in sorted(os.listdir(root))]
    afm, ez, _ = load_series(folders, 4)
    afm, ez, _ = align_series(afm, ez)
    ez, _ = correct_series(ez)
    ratio = figure_of_merit(stack(ez)) / figure_of_merit(ez[0])
    assert ratio < 0.5, f"FOM(8)/FOM(1) = {ratio:.2f}"     # 1/sqrt(8) = 0.35


if __name__ == "__main__":
    # Run the tests of this file only
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS ", name)
