"""
Read Neaspec s-SNOM measurements stored as Gwyddion Simple Field (.gsf) files.

A Neaspec measurement folder contains one .gsf file per channel, e.g.

    <name> M1A raw.gsf      AFM channel used for alignment (mechanical amplitude)
    <name> O4A raw.gsf      near-field amplitude, 4th demodulation order
    <name> O4P raw.gsf      near-field phase,     4th demodulation order

The folder name and the file prefix <name> are identical, so a measurement is
addressed by its folder alone.
"""
import os
import numpy as np

AFM_CHANNEL = "M1A"     # mechanical amplitude: shows the topographic features most sharply


def read_gsf(path):
    """Read one .gsf file. Returns (image, metadata).

    Format: http://gwyddion.net/documentation/user-guide-en/gsf.html
    Text header of "key=value" lines, terminated by NUL bytes that pad the
    header to a multiple of 4 bytes, followed by little-endian float32 data.
    """
    with open(path, "rb") as f:
        raw = f.read()

    end = raw.index(b"\x00")                       # end of the text header
    header = raw[:end].decode("utf-8", errors="replace").splitlines()
    if header[0].strip() != "Gwyddion Simple Field 1.0":
        raise ValueError(f"{path} is not a Gwyddion Simple Field file")

    meta = {}
    for line in header[1:]:
        if "=" in line:
            key, value = line.split("=", 1)
            meta[key.strip()] = value.strip()

    nx, ny = int(meta["XRes"]), int(meta["YRes"])
    start = end + (4 - end % 4)                    # data starts after the padding
    image = np.frombuffer(raw[start:start + 4 * nx * ny], dtype="<f4").reshape(ny, nx)
    return image.astype(float), meta


def write_gsf(path, image, x_size_m=1e-6, y_size_m=1e-6):
    """Write a 2D real array as .gsf (opens in Gwyddion). Used for the test data."""
    ny, nx = image.shape
    header = (f"Gwyddion Simple Field 1.0\nXRes={nx}\nYRes={ny}\n"
              f"XReal={x_size_m}\nYReal={y_size_m}\nXYUnits=m\n").encode()
    header += b"\x00" * (4 - len(header) % 4)
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.asarray(image, dtype="<f4").tobytes())


def load_measurement(folder, order=4):
    """Load one measurement folder.

    Returns
        afm      2D real array, the AFM channel used for alignment
        Ez       2D complex array, A * exp(i * phi) of the chosen demodulation order
        size_um  (x, y) scan size in micrometres
    """
    name = os.path.basename(os.path.normpath(folder))

    def channel(ch):
        return read_gsf(os.path.join(folder, f"{name} {ch} raw.gsf"))

    afm, meta = channel(AFM_CHANNEL)
    amplitude, _ = channel(f"O{order}A")
    phase, _ = channel(f"O{order}P")

    Ez = amplitude * np.exp(1j * phase)
    size_um = (float(meta["XReal"]) * 1e6, float(meta["YReal"]) * 1e6)
    return afm, Ez, size_um


def load_series(folders, order=4):
    """Load several measurements. Returns arrays afm (N, ny, nx), Ez (N, ny, nx), size_um."""
    afm, Ez = [], []
    for folder in folders:
        a, e, size_um = load_measurement(folder, order)
        afm.append(a)
        Ez.append(e)
    return np.array(afm), np.array(Ez), size_um


def example_folders():
    """The measurement folders in example_data/, in acquisition order."""
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_data")
    return [os.path.join(root, f) for f in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, f))]


if __name__ == "__main__":
    # Quick check: load the first example measurement and show it
    import matplotlib.pyplot as plt

    afm, Ez, size_um = load_measurement(example_folders()[0], order=4)
    print(f"image size {afm.shape[1]} x {afm.shape[0]} px, scan size {size_um[0]:.1f} x {size_um[1]:.1f} µm")

    extent = (0, size_um[0], size_um[1], 0)
    fig, ax = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    ax[0].imshow(afm, cmap="gray", extent=extent)
    ax[0].set_title(f"AFM ({AFM_CHANNEL})")
    ax[1].imshow(np.abs(Ez), cmap="viridis", extent=extent)
    ax[1].set_title("amplitude, 4Ω")
    ax[2].imshow(np.angle(Ez), cmap="twilight", extent=extent)
    ax[2].set_title("phase, 4Ω")
    plt.show()
