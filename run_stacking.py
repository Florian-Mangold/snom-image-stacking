"""
Topography-guided image stacking of s-SNOM measurements: complete workflow.

    1. load      every measurement folder in DATA_FOLDER (Neaspec .gsf files)
    2. align     all frames to the first one, using the AFM channel   (align.py)
    3. correct   the near-field signal drift, Eqs. (1)-(4)            (drift_correction.py)
    4. stack     the frames, Eq. (5) with a trimmed mean              (stack.py)
    5. save      the complex stack, its real part, and an overview figure

Edit the settings below and run:   python run_stacking.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import tifffile

from load_gsf import load_series, AFM_CHANNEL
from align import align_series
from drift_correction import correct_series
from stack import stack

# ----------------------------------------------------------------- settings
DATA_FOLDER = "example_data"   # one subfolder per measurement, sorted by name = time
ORDER = 4                      # demodulation order n of the nΩ signal
OUTPUT_FOLDER = "output"
# --------------------------------------------------------------------------


def color_limits(image):
    """Symmetric color limits from the 10th/90th percentile, Supplementary Eqs. (S1)-(S3)."""
    p10, p90 = np.nanpercentile(image, (10, 90))
    vmin = p10 * (1 - 0.1 * np.sign(p10))
    vmax = p90 * (1 + 0.1 * np.sign(p90))
    limit = max(abs(vmin), abs(vmax))
    return -limit, limit


def main():
    here = os.path.dirname(os.path.abspath(__file__))     # relative paths start here
    data_folder = os.path.join(here, DATA_FOLDER)
    output_folder = os.path.join(here, OUTPUT_FOLDER)
    folders = sorted(os.path.join(data_folder, f) for f in os.listdir(data_folder)
                     if os.path.isdir(os.path.join(data_folder, f)))
    print(f"{len(folders)} measurements found in {data_folder}")

    afm, Ez, size_um = load_series(folders, ORDER)                              # 1
    afm, Ez, shifts = align_series(afm, Ez)                                     # 2
    for folder, (dx, dy) in zip(folders, shifts):
        print(f"  shift {dx:+3d} px, {dy:+3d} px   {os.path.basename(folder)}")
    Ez, dphi = correct_series(Ez)                                               # 3
    stacked = stack(Ez)                                                         # 4
    stacked_afm = np.nanmean(afm, axis=0)

    os.makedirs(output_folder, exist_ok=True)                                   # 5
    name = f"O{ORDER}_n{len(folders)}"
    np.save(os.path.join(output_folder, f"stack_{name}_complex.npy"), stacked)
    tifffile.imwrite(os.path.join(output_folder, f"stack_{name}_real.tif"),
                     stacked.real.astype(np.float32))
    tifffile.imwrite(os.path.join(output_folder, f"stack_AFM_n{len(folders)}.tif"),
                     stacked_afm.astype(np.float32))

    # Overview: AFM, first frame and stack (real part, same color scale)
    extent = (0, size_um[0], size_um[1], 0)
    vmin, vmax = color_limits(stacked.real)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ax[0].imshow(stacked_afm, cmap="gray", extent=extent)
    ax[0].set_title(f"AFM ({AFM_CHANNEL}), stacked")
    ax[1].imshow(Ez[0].real, cmap="seismic", vmin=vmin, vmax=vmax, extent=extent)
    ax[1].set_title(f"Re($E_z$), {ORDER}Ω, single frame")
    im = ax[2].imshow(stacked.real, cmap="seismic", vmin=vmin, vmax=vmax, extent=extent)
    ax[2].set_title(f"Re($E_z$), {ORDER}Ω, stack of {len(Ez)}")
    for a in ax:
        a.set_xlabel("x (µm)")
    ax[0].set_ylabel("y (µm)")
    fig.colorbar(im, ax=ax[1:], label="Re($E_z$) (a.u.)")
    fig.savefig(os.path.join(output_folder, f"overview_{name}.png"), dpi=150)
    print(f"\nResults written to {output_folder}")


if __name__ == "__main__":
    main()
