from pathlib import Path

import numpy as np
import pytest

from sunpy.map import Map

DATA_DIR = Path(__file__).resolve().parent / "data" / "aia"
WAVES = [94, 131, 171, 193, 211, 335]
TIME_TAG = "2014-01-01T00-00-00"


def _files():
    files = [DATA_DIR / f"aia_synoptic_{TIME_TAG}_{w:03d}.fits" for w in WAVES]
    if not all(p.exists() for p in files):
        pytest.skip("AIA synoptic data not present. Run scripts/fetch_aia_cutouts.py")
    return files


def test_aia_cutout_shapes():
    maps = [Map(p) for p in _files()]
    for m in maps:
        assert m.data.shape == (1024, 1024)
        assert np.isfinite(m.data).all()


def test_aia_cutout_ordering():
    maps = [Map(p) for p in _files()]
    waves = [round(m.wavelength.to_value()) for m in maps]
    assert waves == sorted(waves)
