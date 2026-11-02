# Topography-guided image stacking for s-SNOM

Minimal reference implementation of the stacking workflow described in

> F. Mangold, F. Aghashirinov, J. Schwab, E. Baù, B. Frank, A. Tittl, H. Giessen,
> *Topography-Guided Image Stacking: Enhancing SNOM Measurements Without Additional Hardware* (2026, submitted).

Version 1.0.0: the code as used for the paper.

Many short s-SNOM measurements of the same area are aligned with the
simultaneously recorded AFM channel, corrected for near-field signal drift,
and averaged. The noise of the stack falls as 1/√n with the number n of
measurements.

## Quick start

```
pip install -r requirements.txt
python run_stacking.py
```

This stacks the eight example measurements in `example_data/` (4Ω signal) and
writes to `output/`:

| file | content |
|---|---|
| `stack_O4_n8_complex.npy` | stacked complex near field E_z (load with `numpy.load`) |
| `stack_O4_n8_real.tif` | real part of the stack, 32-bit float TIFF |
| `stack_AFM_n8.tif` | stacked AFM channel, to check the alignment (no doubled features) |
| `overview_O4_n8.png` | AFM, one single frame and the stack side by side |

## The files

| file | what it does | paper |
|---|---|---|
| `load_gsf.py` | reads Neaspec `.gsf` files, builds E_z = A·e^{iφ} | Methods |
| `align.py` | ORB keypoints on the AFM channel, shift = median keypoint displacement (whole pixels), applied to E_z | Section 2, Supplementary Note 2 |
| `drift_correction.py` | row-wise median offset, amplitude normalization, phase drift | Eqs. (1)–(4) |
| `stack.py` | pixel-wise trimmed mean (real and imaginary part sorted separately, lowest and highest 10% discarded) | Eq. (5) |
| `run_stacking.py` | the complete workflow; settings at the top | |
| `make_test_data.py` | synthetic measurements with known drift, noise and an outlier | |
| `tests/` | checks of every step | |

Every file can also be run on its own (e.g. `python align.py`) for a quick
check of that step on the example data.

Each step is a few plain functions and can be used on its own, e.g.

```python
from load_gsf import load_series
from align import align_series
from drift_correction import correct_series
from stack import stack

afm, Ez, size_um = load_series(list_of_measurement_folders, order=4)
afm, Ez, shifts = align_series(afm, Ez)
Ez, phase_corrections = correct_series(Ez)
result = stack(Ez)
```

## Your own data

Put one folder per measurement into a folder and set `DATA_FOLDER` in
`run_stacking.py`; `ORDER` selects the demodulation order n of the nΩ signal.
Each measurement folder must contain the Neaspec export files
`<name> M1A raw.gsf`, `<name> O<n>A raw.gsf` and `<name> O<n>P raw.gsf`,
where `<name>` is the folder name. Folders are processed in alphabetical
order, which for Neaspec names (date and time first) is the acquisition order.
The first measurement is the reference for position and phase.

## Tests

```
python tests/run_tests.py
```

or `pytest`. The tests check that known shifts, offsets, amplitudes and phases
are recovered from synthetic data, that the noise of a stack falls as 1/√n
(also with the figure of merit of main-text Eq. (6)), that the trimmed mean
discards the right number of values and rejects an outlier frame, and that the example data give the alignment shifts
found by the code used for the paper.

## Example data

`example_data/` holds the first eight 10-min measurements of the series used in
the paper (MoOCl₂ flake, λ = 3.2 µm, 20 × 20 µm², 220 × 220 pixels), with the
AFM channel and the 2Ω to 5Ω near-field channels. The complete data set (all 136
10-min measurements, the ten 60-min measurements and the 600-min measurement)
is available on DaRUS: https://doi.org/10.18419/DARUS-6675

## Requirements

Python 3.10 or newer with numpy, opencv-python, matplotlib and tifffile
(tested with Python 3.13, numpy 2.5, opencv-python 5.0, matplotlib 3.11 and
tifffile 2026.10).

## About this code

This code was extracted, with the help of AI, from the original, much larger
analysis program used for the paper. It focuses on simplicity: it is short,
easy to read, and implements exactly the steps described in the paper, applied
to the whole image. It is deliberately not built for integration into a
graphical user interface, for offering as many options as possible, for short
calculation times, for automatic comparisons between different options, or for
the automatic generation of complex plots. For the same reason, the
region-wise offset correction (the row-wise offset of Eq. (1) computed
separately on either side of the topographic step, Supplementary Note 4), the Fourier analysis of main-text
Figure 4 and the plotting scripts of the paper figures are not included.

The complete program is planned to be made available in the future,
including a graphical user interface for easy handling and many more options
than described in the paper (for example, different alignment algorithms).

## How to cite

If you use this code, please cite the paper above (the full reference will be
added after publication). `CITATION.cff` contains the citation information;
GitHub shows it under "Cite this repository".

## License

The code is released under the MIT License (see `LICENSE`). The example
measurements in `example_data/` are released under the Creative Commons
Attribution 4.0 International License (CC BY 4.0).
