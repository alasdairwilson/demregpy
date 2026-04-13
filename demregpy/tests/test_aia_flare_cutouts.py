from pathlib import Path

import numpy as np
import pytest

from sunpy.map import Map

DATA_DIR = Path(__file__).resolve().parent / "data" / "aia_flare"
WAVES = [94, 131, 171, 193, 211, 335]
TIME_SLOTS = [
    "2011-02-15T01-44",
    "2011-02-15T01-46",
    "2011-02-15T01-48",
    "2011-02-15T01-50",
    "2011-02-15T01-52",
    "2011-02-15T01-54",
    "2011-02-15T01-56",
    "2011-02-15T01-58",
    "2011-02-15T02-00",
    "2011-02-15T02-02",
]


def _files():
    files = []
    for time_slot in TIME_SLOTS:
        for wave in WAVES:
            matches = sorted(DATA_DIR.glob(f"aia_flare_{time_slot}-*_{wave:03d}.fits"))
            if len(matches) != 1:
                pytest.skip("AIA flare data not present. Run scripts/fetch_aia_flare_cutouts.py")
            files.extend(matches)
    if not files:
        pytest.skip("AIA flare data not present. Run scripts/fetch_aia_flare_cutouts.py")
    return files


def test_aia_flare_cutout_shapes():
    maps = [Map(path) for path in _files()]
    shapes = {m.data.shape for m in maps}

    assert len(shapes) == 1
    shape = shapes.pop()
    assert shape[0] < 100
    assert shape[1] < 100
    for amap in maps:
        assert np.isfinite(amap.data).all()


def test_aia_flare_cutout_grouping():
    for time_slot in TIME_SLOTS:
        row = [Map(sorted(DATA_DIR.glob(f"aia_flare_{time_slot}-*_{wave:03d}.fits"))[0]) for wave in WAVES]
        assert [round(amap.wavelength.to_value()) for amap in row] == WAVES
