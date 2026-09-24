import numpy as np
import pytest

from assistlabel.io.dataset import file_hash, out_paths, scan_images
from assistlabel.io.depth_io import (
    dequantize_uint16_to_m,
    pseudocolor,
    quantize_m_to_uint16,
    read_depth_png,
    save_depth_png,
)


@pytest.fixture
def images(tmp_path):
    (tmp_path / "sub").mkdir()
    for name in ["a.jpg", "b.PNG", "c.txt", "sub/d.jpeg"]:
        path = tmp_path / name
        if path.suffix == ".txt":
            path.write_text("not an image")
        else:
            path.write_bytes(b"\xff\xd8fake")
    return tmp_path


def test_scan_images_default_suffixes(images):
    found = [p.name for p in scan_images(images)]
    assert found == ["a.jpg", "b.PNG", "d.jpeg"]


def test_scan_images_patterns(images):
    found = [p.name for p in scan_images(images, ["*.jpg"])]
    assert found == ["a.jpg"]


def test_scan_images_missing_dir(tmp_path):
    with pytest.raises(NotADirectoryError):
        scan_images(tmp_path / "nope")


def test_file_hash_stable_and_content_sensitive(images):
    f = images / "a.jpg"
    h1 = file_hash(f)
    assert h1 == file_hash(f)
    f.write_bytes(b"\xff\xd8fake2")
    assert h1 != file_hash(f)


def test_quantize_roundtrip_and_invalid():
    depth = np.array([[0.0, 1.5], [np.nan, 100.0]], dtype=np.float32)
    u16 = quantize_m_to_uint16(depth)
    assert u16.dtype == np.uint16
    assert u16[0, 0] == 0          # zero stays invalid
    assert u16[1, 1] == 65535      # clipped beyond uint16 mm range
    assert u16[0, 1] == 1500
    back = dequantize_uint16_to_m(u16)
    assert abs(back[0, 1] - 1.5) < 1e-3


def test_depth_png_save_read_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.5, 30.0, size=(40, 60)).astype(np.float32)
    path = tmp_path / "d.png"
    save_depth_png(depth, path)
    loaded = read_depth_png(path)
    assert loaded.shape == (40, 60)
    assert np.abs(loaded - depth).max() < 1e-3  # 1 mm quantization
    assert loaded.dtype == np.float32


def test_pseudocolor_shape_and_valid_handling():
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[2:8, 2:8] = np.linspace(1.0, 20.0, 36).reshape(6, 6)
    img = pseudocolor(depth)
    assert img.shape == (10, 10, 3) and img.dtype == np.uint8
