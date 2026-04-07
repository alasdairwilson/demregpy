#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download six AIA synoptic images and save them as test data.

Requires network access and SunPy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from astropy import time as atime
from astropy import units as u
from sunpy.map import Map
from sunpy.net import Fido, attrs

TIME_STR = "2014-01-01T00:00:00"
WAVELENGTHS = [94, 131, 171, 193, 211, 335] * u.angstrom
LEVEL = "1.5s"
SEARCH_SECONDS = 60


def _wave_value(value) -> int:
    if hasattr(value, "to_value"):
        return int(round(value.to_value(u.angstrom)))
    return int(value)


def fetch_aia_maps(download_dir: Path):
    t = atime.Time(TIME_STR, scale="utc")
    td = atime.TimeDelta(SEARCH_SECONDS, format="sec")
    query = Fido.search(
        attrs.Time(t, t + td),
        attrs.Instrument.aia,
        attrs.Level(LEVEL),
    )
    if len(query) == 0:
        raise RuntimeError(f"No AIA synoptic files found in {TIME_STR}..{(t + td).isot}")
    results = query[0]
    target_waves = {int(w.to_value(u.angstrom)) for w in WAVELENGTHS}
    first_start = min(row["Start Time"] for row in results)
    selected = [
        row
        for row in results
        if row["Start Time"] == first_start and _wave_value(row["Wavelength"]) in target_waves
    ]
    found = sorted(_wave_value(row["Wavelength"]) for row in selected)
    if found != sorted(target_waves):
        raise RuntimeError(f"Expected wavelengths {sorted(target_waves)}, found {found} at {first_start.isot}")
    files = Fido.fetch(*selected, path=str(download_dir / "{file}"))
    if len(files) != len(target_waves):
        raise RuntimeError(f"Expected {len(target_waves)} files, fetched {len(files)}")
    maps = [Map(f) for f in files]
    maps = sorted(maps, key=lambda x: x.wavelength)
    return maps


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch AIA synoptic test data.")
    parser.add_argument(
        "--output-dir",
        default="demregpy/tests/data/aia",
        help="Directory for saved synoptic files.",
    )
    parser.add_argument(
        "--download-dir",
        default="/tmp/demregpy-aia-downloads",
        help="Directory for downloaded AIA files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    maps = fetch_aia_maps(download_dir)

    for m in maps:
        w = int(round(m.wavelength.to_value(u.angstrom)))
        out = output_dir / f"aia_synoptic_{TIME_STR.replace(':', '-')}_{w:03d}.fits"
        m.save(out, overwrite=True)
        print(f"saved {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
