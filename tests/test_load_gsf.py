"""Reading and writing .gsf files."""
import os, sys, tempfile
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from load_gsf import read_gsf, write_gsf, load_measurement


def test_write_then_read_gives_same_image():
    image = np.random.default_rng(1).normal(size=(37, 52))     # odd, non-square size
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "test.gsf")
        write_gsf(path, image, 5e-6, 3e-6)
        back, meta = read_gsf(path)
    assert back.shape == (37, 52)
    assert np.allclose(back, image.astype(np.float32))
    assert float(meta["XReal"]) == 5e-6


def test_example_measurement_loads():
    root = os.path.join(os.path.dirname(__file__), "..", "example_data")
    folder = os.path.join(root, sorted(os.listdir(root))[0])
    afm, Ez, size_um = load_measurement(folder, order=4)
    assert afm.shape == Ez.shape == (220, 220)
    assert np.iscomplexobj(Ez)
    assert np.allclose(size_um, (20, 20))


if __name__ == "__main__":
    # Run the tests of this file only
    for name, test in list(globals().items()):
        if name.startswith("test_"):
            test()
            print("PASS ", name)
