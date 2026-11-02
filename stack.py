"""
Stacking: pixel-wise averaging of the aligned, drift-corrected frames
(main text Equation (5)).

The frames are averaged with a trimmed mean, which protects the stack against
transient outliers (single frames with a disturbance). At every pixel, the
real and the imaginary part are sorted separately over the n frames, and
k = int(0.1 n) values are discarded at each end before averaging the rest.
At least one value per end is discarded as soon as there are six or more
values; for five or fewer, the plain mean is used. Pixels that are not
covered by every frame (the NaN borders created by the alignment) use only
the frames that cover them.

For purely Gaussian noise, this costs about 3% in signal-to-noise ratio
compared with the plain mean.
"""
import warnings
import numpy as np

TRIM = 0.1     # fraction of the values discarded at each end
N_MIN = 6      # from this number of values on, at least one is discarded at each end


def mean_stack(frames):
    """Eq. (5) without trimming: plain pixel-wise mean of the complex frames (N, ny, nx)."""
    return np.nanmean(frames, axis=0)


def trimmed_mean(values):
    """Trimmed mean along axis 0 of a real array, ignoring NaN.

    At every pixel the n valid values are sorted and k values are dropped at
    each end: k = int(TRIM * n), at least 1 for n >= N_MIN, none below.
    """
    n = np.sum(~np.isnan(values), axis=0)                               # valid values per pixel
    k = np.where(n >= N_MIN, np.maximum((n * TRIM).astype(int), 1), 0)  # dropped at each end
    rank = np.arange(values.shape[0]).reshape((-1,) + (1,) * (values.ndim - 1))
    ordered = np.sort(values, axis=0)                                   # NaN sorts last
    kept = np.where((rank >= k) & (rank < n - k), ordered, np.nan)
    with warnings.catch_warnings():                                     # pixels without any frame
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(kept, axis=0)


def stack(frames):
    """Eq. (5): trimmed mean of the complex frames (N, ny, nx), real and imaginary part separately."""
    frames = np.asarray(frames)
    return trimmed_mean(frames.real) + 1j * trimmed_mean(frames.imag)


if __name__ == "__main__":
    # Quick check: stack the example measurements and compare with a single
    # frame and with the plain mean (same color scale for all panels)
    import matplotlib.pyplot as plt
    from load_gsf import load_series, example_folders
    from align import align_series
    from drift_correction import correct_series

    afm, Ez, size_um = load_series(example_folders(), order=4)
    afm, Ez, _ = align_series(afm, Ez)
    Ez, _ = correct_series(Ez)

    panels = [("single frame", Ez[0]),
              (f"plain mean, n = {len(Ez)}", mean_stack(Ez)),
              (f"trimmed mean, n = {len(Ez)}", stack(Ez))]
    limit = np.nanpercentile(np.abs(panels[1][1].real), 90) * 1.1
    extent = (0, size_um[0], size_um[1], 0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.5), constrained_layout=True)
    for a, (title, image) in zip(ax, panels):
        a.imshow(image.real, cmap="seismic", vmin=-limit, vmax=limit, extent=extent)
        a.set_title(f"Re($E_z$), {title}")
    plt.show()
