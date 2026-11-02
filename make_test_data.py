"""
Create synthetic s-SNOM measurements with known drift, for testing.

Each synthetic measurement is written as a folder of .gsf files, exactly like
a Neaspec measurement (M1A, O4A, O4P), so it runs through the same code.
Every frame gets, with known values:

    a position shift   (dx, dy) in pixels
    a complex offset   that drifts linearly from the top to the bottom row
    an amplitude factor and a global phase
    complex Gaussian noise
and optionally one frame gets a transient outlier (a disturbed patch).

Run   python make_test_data.py   to write the folder "test_data".
"""
import os
import numpy as np
from load_gsf import write_gsf

SIZE = 128          # pixels per side of a measurement
PAD = 16            # margin, so shifted frames can be cut from a larger canvas


def afm_canvas(rng, size):
    """AFM-like image: a flake edge plus randomly placed round particles."""
    y, x = np.mgrid[0:size, 0:size]
    afm = 1.0 * (x > 0.8 * size)                                  # flake edge
    for _ in range(40):                                           # particles
        x0, y0 = rng.uniform(0, size, 2)
        r = rng.uniform(1.5, 4)
        afm += rng.uniform(0.5, 2) * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * r ** 2))
    return afm


def near_field(size):
    """Noise-free complex near field: a damped wave launched at the flake edge."""
    y, x = np.mgrid[0:size, 0:size]
    distance = 0.8 * size - x                                     # from the edge
    wave = np.exp(1j * 2 * np.pi * distance / 20) * np.exp(-distance / 60)
    return np.where(distance > 0, wave, 0.3)


def make_frames(n_frames=16, noise=0.5, outlier_frame=None, seed=0):
    """Return a dict with the frames and the ground truth used to create them."""
    rng = np.random.default_rng(seed)
    big = SIZE + 2 * PAD
    afm_big = afm_canvas(rng, big)
    field_big = near_field(big)

    truth = {"shift": [], "phase": [], "amplitude": [], "signal": field_big[PAD:-PAD, PAD:-PAD]}
    afm_frames, ez_frames = [], []
    rows = np.arange(SIZE)[:, None] / SIZE
    for m in range(n_frames):
        sx, sy = (0, 0) if m == 0 else rng.integers(-6, 7, 2)      # integer shift in px
        cut = (slice(PAD + sy, PAD + sy + SIZE), slice(PAD + sx, PAD + sx + SIZE))
        amplitude = rng.uniform(0.5, 2)
        phase = rng.uniform(-np.pi, np.pi)
        offset = rng.normal(0, 2, 2) @ [1, 1j] + rng.normal(0, 1) * rows   # drifts along y

        ez = amplitude * np.exp(1j * phase) * field_big[cut] + offset
        ez = ez + noise * (rng.normal(size=ez.shape) + 1j * rng.normal(size=ez.shape))
        if m == outlier_frame:
            ez[40:55, 30:60] += 30                                # a disturbed patch

        afm_frames.append(afm_big[cut] + 0.02 * rng.normal(size=(SIZE, SIZE)))
        ez_frames.append(ez)
        # (dx, dy) that moves this frame back onto the first one
        truth["shift"].append((float(sx), float(sy)))
        truth["phase"].append(phase)
        truth["amplitude"].append(amplitude)
    return np.array(afm_frames), np.array(ez_frames), truth


def write_frames(folder, afm, ez, pixel_size_m=100e-9):
    """Write the frames as Neaspec-style measurement folders."""
    size_m = afm.shape[-1] * pixel_size_m
    for m in range(len(afm)):
        name = f"frame_{m:03d}"
        path = os.path.join(folder, name)
        os.makedirs(path, exist_ok=True)
        write_gsf(os.path.join(path, f"{name} M1A raw.gsf"), afm[m], size_m, size_m)
        write_gsf(os.path.join(path, f"{name} O4A raw.gsf"), np.abs(ez[m]), size_m, size_m)
        write_gsf(os.path.join(path, f"{name} O4P raw.gsf"), np.angle(ez[m]), size_m, size_m)


if __name__ == "__main__":
    afm, ez, truth = make_frames(n_frames=16, outlier_frame=5)
    write_frames(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data"), afm, ez)
    print("16 synthetic measurements written to test_data/ (frame 5 contains an outlier patch)")
