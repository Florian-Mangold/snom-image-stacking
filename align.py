"""
Positional alignment using the simultaneously recorded AFM channel
(main text Section 2, Supplementary Note 2, Figure S3).

ORB keypoints are detected in the AFM image of every measurement and matched
to the keypoints of the reference (the first measurement). The shift of a
measurement is the median displacement of its matched keypoints, so a few
wrong matches do not matter. The same shift is then applied to the
near-field data.

Shifts are rounded to whole pixels. A sub-pixel shift would interpolate
between neighbouring pixels, which averages and therefore lowers the noise
of the shifted frame and slightly blurs it. The alignment is accurate to
about +-2 px anyway.
"""
import cv2
import numpy as np


def to_uint8(image):
    """Scale a float image to 0..255 (1st to 99th percentile), as ORB needs 8-bit input."""
    lo, hi = np.nanpercentile(image, (1, 99))
    scaled = (np.clip(image, lo, hi) - lo) / (hi - lo) * 255
    return np.nan_to_num(scaled).astype(np.uint8)


def find_shift(reference, image, n_features=5000, keep_best=0.9):
    """Shift (dx, dy) in pixels that moves `image` onto `reference` (not rounded)."""
    orb = cv2.ORB_create(n_features)
    kp_ref, des_ref = orb.detectAndCompute(to_uint8(reference), None)
    kp_img, des_img = orb.detectAndCompute(to_uint8(image), None)
    if des_ref is None or des_img is None:
        raise RuntimeError("No ORB keypoints found; is the AFM image featureless?")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_ref, des_img), key=lambda m: m.distance)
    matches = matches[:int(len(matches) * keep_best)]     # drop the 10% worst matches
    if len(matches) < 4:
        raise RuntimeError("Fewer than 4 keypoint matches; alignment not possible.")

    p_ref = np.array([kp_ref[m.queryIdx].pt for m in matches])
    p_img = np.array([kp_img[m.trainIdx].pt for m in matches])
    dx, dy = np.median(p_ref - p_img, axis=0)
    return dx, dy


def shift_image(image, dx, dy):
    """Shift a real or complex image by (dx, dy) pixels.

    Pixels that move in from outside the image are set to NaN, so they are
    ignored by all later steps (nanmean, nanmedian, ...).
    Complex images are shifted as real and imaginary part, never as amplitude
    and phase, because interpolating a wrapped phase gives wrong values.
    """
    if np.iscomplexobj(image):
        return shift_image(image.real, dx, dy) + 1j * shift_image(image.imag, dx, dy)
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    h, w = image.shape

    def warp(a):
        return cv2.warpAffine(a.astype(np.float32), M, (w, h),
                              flags=cv2.INTER_LINEAR, borderValue=0).astype(float)

    shifted = warp(image)
    inside = warp(np.ones_like(image))          # 1 where only real pixels were used
    shifted[inside < 0.99] = np.nan
    return shifted


def align_series(afm, Ez):
    """Align all measurements to the first one, by whole-pixel shifts.

    afm, Ez : arrays (N, ny, nx), AFM channel and complex near-field
    Returns aligned afm, aligned Ez and the list of shifts (dx, dy) applied.
    """
    shifts = [find_shift(afm[0], a) for a in afm]
    shifts = [(round(dx), round(dy)) for dx, dy in shifts]
    afm_aligned = np.array([shift_image(a, dx, dy) for a, (dx, dy) in zip(afm, shifts)])
    Ez_aligned = np.array([shift_image(e, dx, dy) for e, (dx, dy) in zip(Ez, shifts)])
    return afm_aligned, Ez_aligned, shifts


if __name__ == "__main__":
    # Quick check: align the example measurements and compare the averaged AFM
    # image without and with alignment (without: features are smeared/doubled)
    import matplotlib.pyplot as plt
    from load_gsf import load_series, example_folders

    afm, Ez, size_um = load_series(example_folders(), order=4)
    afm_aligned, _, shifts = align_series(afm, Ez)
    for i, (dx, dy) in enumerate(shifts):
        print(f"measurement {i + 1}: shift {dx:+d} px, {dy:+d} px")

    extent = (0, size_um[0], size_um[1], 0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.5), constrained_layout=True)
    ax[0].imshow(np.mean(afm, axis=0), cmap="gray", extent=extent)
    ax[0].set_title("AFM, averaged without alignment")
    ax[1].imshow(np.nanmean(afm_aligned, axis=0), cmap="gray", extent=extent)
    ax[1].set_title("AFM, averaged after alignment")
    plt.show()
