"""Stacking reduces noise as 1/sqrt(n) and rejects transient outliers."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from stack import stack, mean_stack, trimmed_mean


def figure_of_merit(image):
    """Main text Eq. (6): std of the difference between adjacent scan rows (real part).

    Neighbouring rows carry almost the same signal, so their difference is
    dominated by noise.
    """
    real = np.real(image)
    return np.nanstd(real[1:, :] - real[:-1, :])


def noise_frames(n, seed=0, shape=(64, 64)):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n,) + shape) + 1j * rng.normal(size=(n,) + shape)


def test_mean_noise_falls_as_one_over_sqrt_n():
    frames = noise_frames(64)
    for n in (4, 16, 64):
        ratio = np.std(mean_stack(frames[:n]).real) / np.std(frames[0].real)
        assert abs(ratio * np.sqrt(n) - 1) < 0.05, f"n={n}: {ratio:.3f} vs {1/np.sqrt(n):.3f}"


def test_trimmed_mean_costs_little_on_pure_noise():
    frames = noise_frames(64)
    ratio = np.std(stack(frames).real) / np.std(mean_stack(frames).real)
    assert ratio < 1.06                                    # about 3% expected


def test_trimmed_mean_rejects_an_outlier_frame():
    frames = noise_frames(32) * 0.1
    frames[7, 20:30, 20:30] += 50 + 50j                    # one disturbed frame
    patch = (slice(20, 30), slice(20, 30))
    assert np.abs(mean_stack(frames)[patch]).mean() > 1
    assert np.abs(stack(frames)[patch]).mean() < 0.1


def test_number_of_discarded_values():
    # k = int(0.1 n) at each end, at least 1 from n = 6 on, none for n <= 5
    for n, k in [(3, 0), (5, 0), (6, 1), (19, 1), (20, 2), (29, 2), (128, 12)]:
        values = np.arange(n, dtype=float)[:, None, None]
        values[-1] = 1000.0                                # outlier at the top end
        expected = np.mean(np.sort(values[:, 0, 0])[k:n - k])
        assert np.isclose(trimmed_mean(values)[0, 0], expected), (n, k)


def test_nan_border_pixels_are_ignored():
    frames = noise_frames(16)
    frames[3, :, :5] = complex(np.nan, np.nan)             # shifted frame: NaN border (as align.py)
    assert not np.isnan(stack(frames)).any()
    # the border pixels are the trimmed mean of the 15 frames that cover them
    expected = stack(np.delete(frames, 3, axis=0))[:, :5]
    assert np.allclose(stack(frames)[:, :5], expected)


def test_pixels_covered_by_two_frames_are_not_lost():
    frames = noise_frames(16)
    frames[2:, :, :3] = complex(np.nan, np.nan)            # only frames 1, 2 cover the edge
    assert np.allclose(stack(frames)[:, :3], frames[:2, :, :3].mean(axis=0))


def test_figure_of_merit_measures_noise_not_signal():
    y, x = np.mgrid[0:64, 0:64]
    fringes = np.cos(x / 3.0) + 0j                          # constant along y
    assert figure_of_merit(fringes) < 1e-12
    noise = noise_frames(1)[0]
    assert abs(figure_of_merit(fringes + noise) / np.sqrt(2) - 1) < 0.05


if __name__ == "__main__":
    # Run the tests of this file only
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS ", name)
