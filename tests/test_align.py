"""Alignment recovers known shifts."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from align import find_shift, shift_image, align_series
from make_test_data import make_frames


def test_known_shifts_are_recovered():
    afm, ez, truth = make_frames(n_frames=8, noise=0.5, seed=2)
    shifts = [find_shift(afm[0], a) for a in afm]
    error = np.abs(np.array(shifts) - np.array(truth["shift"]))
    assert error.max() <= 0.5, f"shift error up to {error.max():.2f} px"


def test_shift_image_moves_content_and_marks_border_nan():
    image = np.zeros((20, 20)); image[10, 10] = 1
    moved = shift_image(image, 3, -2)
    assert moved[8, 13] == 1                       # moved right by 3, up by 2
    assert np.isnan(moved[:, :3]).all()            # new left border is NaN
    assert not np.isnan(shift_image(image, 0, 0)).any()


def test_complex_image_is_shifted_as_real_and_imaginary_part():
    image = np.exp(1j * np.linspace(-np.pi, np.pi, 400)).reshape(20, 20)
    moved = shift_image(image, 0.5, 0)
    expected = (image[:, :-1] + image[:, 1:]) / 2  # bilinear half-pixel shift
    assert np.allclose(moved[:, 1:], expected)


def test_example_data_shifts_match_original_code():
    # Shifts of the 8 example measurements as found by the code used for the paper
    paper = [(0, 0), (1.0, -0.35), (2.16, -1.6), (3.6, -2.64),
             (5.0, -2.64), (7.0, -3.0), (8.0, -3.0), (9.5, -3.6)]
    from load_gsf import load_series
    root = os.path.join(os.path.dirname(__file__), "..", "example_data")
    folders = [os.path.join(root, f) for f in sorted(os.listdir(root))]
    afm, ez, _ = load_series(folders, 4)
    shifts = [find_shift(afm[0], a) for a in afm]          # before rounding
    assert np.allclose(shifts, paper, atol=0.01)


def test_default_shifts_are_whole_pixels():
    afm, ez, _ = make_frames(n_frames=4, seed=5)
    _, _, shifts = align_series(afm, ez)
    assert all(float(dx).is_integer() and float(dy).is_integer() for dx, dy in shifts)


if __name__ == "__main__":
    # Run the tests of this file only
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS ", name)
