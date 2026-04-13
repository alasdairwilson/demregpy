"""
Simple loaders for the local AIA example datasets.
"""

from pathlib import Path

from sunpy.map import Map

AIA_WAVELENGTHS = [94, 131, 171, 193, 211, 335]
AIA_FLARE_SLOT_TAGS = [
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

_DATA_DIR = Path(__file__).resolve().parent / "data"


def load_aia_full_disk_maps():
    """
    Load the local full-disk AIA test maps used by the examples.
    """
    files = [
        _DATA_DIR / "aia" / f"aia_synoptic_2014-01-01T00-00-00_{wave:03d}.fits"
        for wave in AIA_WAVELENGTHS
    ]
    return [Map(path) for path in files]


def load_aia_flare_timeseries():
    """
    Load the local AIA flare cutouts grouped by time step.
    """
    data_dir = _DATA_DIR / "aia_flare"
    map_rows = []
    for slot_tag in AIA_FLARE_SLOT_TAGS:
        row = [
            Map(sorted(data_dir.glob(f"aia_flare_{slot_tag}-*_{wave:03d}.fits"))[0])
            for wave in AIA_WAVELENGTHS
        ]
        map_rows.append(row)
    return map_rows
