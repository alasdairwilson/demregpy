"""
Small data loaders shared by the documentation examples.
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


def _example_data_dir(name):
    return Path(__file__).resolve().parents[1] / "demregpy" / "tests" / "data" / name


def load_aia_test_maps():
    """
    Load the local AIA synoptic FITS files used by the examples and tests.
    """
    data_dir = _example_data_dir("aia")
    files = [data_dir / f"aia_synoptic_2014-01-01T00-00-00_{w:03d}.fits" for w in AIA_WAVELENGTHS]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Local AIA example data are missing. Expected files: "
            f"{joined}. Run scripts/fetch_aia_cutouts.py if needed."
        )
    maps = [Map(path) for path in files]
    return sorted(maps, key=lambda amap: round(amap.wavelength.to_value()))


def load_aia_flare_maps():
    """
    Load the local AIA flare cutout sequence used by the DEMogram example.
    """
    data_dir = _example_data_dir("aia_flare")
    map_rows = []
    missing = []

    for slot_tag in AIA_FLARE_SLOT_TAGS:
        row = []
        for wave in AIA_WAVELENGTHS:
            matches = sorted(data_dir.glob(f"aia_flare_{slot_tag}-*_{wave:03d}.fits"))
            if len(matches) != 1:
                missing.append(f"{slot_tag} {wave:03d}")
                continue
            row.append(Map(matches[0]))
        if row:
            map_rows.append(row)

    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(
            "Local AIA flare example data are missing or incomplete. Expected one file per "
            "time slot and wavelength. Missing entries: "
            f"{joined}. Run scripts/fetch_aia_flare_cutouts.py if needed."
        )

    return map_rows
