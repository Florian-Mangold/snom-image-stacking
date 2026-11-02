"""
Signal drift correction, main text Equations (1) to (4).

Every measurement m is corrected on its own, except for the phase, which is
referenced to the first measurement:

    Eq. (1)  remove the complex background offset row by row (row median)
    Eq. (2)  normalize the amplitude:   E / <|E|>
    Eq. (3)  phase drift:               dPhi_m = arg < E_1 * conj(E_m) >
    Eq. (4)  apply it:                  E_m * exp(i dPhi_m)

<...> is the mean over all pixels of the frame. NaN pixels (the borders
created by the alignment) are ignored everywhere.
"""
import warnings
import numpy as np


def subtract_row_median(Ez, smooth_rows=3):
    """Eq. (1): subtract the median of every scan row, separately for Re and Im.

    The row medians are smoothed over `smooth_rows` neighbouring rows to make
    the estimate less noisy (Supplementary Note 2). Rows that are entirely NaN
    get the offset of their neighbours (linear interpolation).
    """
    def corrected(part):
        with warnings.catch_warnings():                 # all-NaN rows are expected
            warnings.simplefilter("ignore", RuntimeWarning)
            offset = np.nanmedian(part, axis=1)          # one value per row
        rows = np.arange(len(offset))
        ok = ~np.isnan(offset)
        offset = np.interp(rows, rows[ok], offset[ok])   # fill all-NaN rows
        if smooth_rows > 1:                              # running mean over rows
            k = smooth_rows // 2
            padded = np.pad(offset, k, mode="symmetric")
            offset = np.convolve(padded, np.ones(smooth_rows) / smooth_rows, mode="valid")
        return part - offset[:, None]

    return corrected(Ez.real) + 1j * corrected(Ez.imag)


def normalize_amplitude(Ez):
    """Eq. (2): divide by the mean amplitude, so every frame has <|E|> = 1."""
    return Ez / np.nanmean(np.abs(Ez))


def phase_drift(Ez_ref, Ez):
    """Eq. (3): amplitude-weighted mean phase difference to the reference frame."""
    return np.angle(np.nanmean(Ez_ref * np.conj(Ez)))


def correct_series(Ez):
    """Apply Eqs. (1) to (4) to a series of frames (N, ny, nx).

    Returns the corrected frames and the phase corrections dPhi_m (radians).
    """
    frames = [normalize_amplitude(subtract_row_median(e)) for e in Ez]   # Eqs. (1), (2)
    reference = frames[0]
    dphi = [phase_drift(reference, f) for f in frames]                  # Eq. (3)
    corrected = np.array([f * np.exp(1j * p) for f, p in zip(frames, dphi)])  # Eq. (4)
    return corrected, np.array(dphi)


if __name__ == "__main__":
    # Quick check: correct the example measurements and compare the real part
    # of one frame before and after the correction
    import matplotlib.pyplot as plt
    from load_gsf import load_series, example_folders
    from align import align_series

    afm, Ez, size_um = load_series(example_folders(), order=4)
    afm, Ez, _ = align_series(afm, Ez)
    corrected, dphi = correct_series(Ez)
    for i, p in enumerate(dphi):
        print(f"measurement {i + 1}: phase correction {np.degrees(p):+7.1f}°")

    m = len(Ez) - 1                                     # last measurement
    extent = (0, size_um[0], size_um[1], 0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    for a, image, title in [(ax[0], Ez[m], "before correction"),
                            (ax[1], corrected[m], "after Eqs. (1)-(4)")]:
        limit = np.nanpercentile(np.abs(image.real), 90) * 1.1
        a.imshow(image.real, cmap="seismic", vmin=-limit, vmax=limit, extent=extent)
        a.set_title(f"Re($E_z$), measurement {m + 1}, {title}")
    plt.show()
