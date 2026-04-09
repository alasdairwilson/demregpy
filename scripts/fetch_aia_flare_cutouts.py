#!/usr/bin/env python3
"""
Download a small AIA flare cutout time series and save it as local example data.

Requires network access, SunPy, and a registered JSOC email address.
"""

import argparse
from pathlib import Path

from astropy import time as atime
from astropy import units as u
from astropy.coordinates import SkyCoord

from sunpy.coordinates import frames
from sunpy.map import Map
from sunpy.net import Fido, attrs

TIME_START = "2011-02-15T01:44:00"
TIME_END = "2011-02-15T02:02:00"
TIME_SLOTS = [
    "2011-02-15T01:44:00",
    "2011-02-15T01:46:00",
    "2011-02-15T01:48:00",
    "2011-02-15T01:50:00",
    "2011-02-15T01:52:00",
    "2011-02-15T01:54:00",
    "2011-02-15T01:56:00",
    "2011-02-15T01:58:00",
    "2011-02-15T02:00:00",
    "2011-02-15T02:02:00",
]
SAMPLE = 2 * u.minute
WAVELENGTHS = [94, 131, 171, 193, 211, 335] * u.angstrom
CUTOUT_CENTER = (192.4, -220.8) * u.arcsec
CUTOUT_WIDTH = 40 * u.arcsec
CUTOUT_HEIGHT = 40 * u.arcsec
SERIES = "aia.lev1_euv_12s"


def _cutout_attr(obstime):
    bottom_left = SkyCoord(
        CUTOUT_CENTER[0] - CUTOUT_WIDTH / 2,
        CUTOUT_CENTER[1] - CUTOUT_HEIGHT / 2,
        frame=frames.Helioprojective,
        obstime=obstime,
        observer="earth",
    )
    return attrs.jsoc.Cutout(
        bottom_left,
        width=CUTOUT_WIDTH,
        height=CUTOUT_HEIGHT,
        tracking=True,
    )


def fetch_flare_maps(download_dir: Path, email: str):
    t0 = atime.Time(TIME_START, scale="utc")
    t1 = atime.Time(TIME_END, scale="utc")
    cutout = _cutout_attr(t0)
    fetched = []

    for wavelength in WAVELENGTHS:
        query = Fido.search(
            attrs.Time(t0, t1),
            attrs.Sample(SAMPLE),
            attrs.jsoc.Series(SERIES),
            attrs.jsoc.Notify(email),
            attrs.Wavelength(wavelength),
            attrs.jsoc.Segment("image"),
            cutout,
        )
        files = Fido.fetch(
            query,
            path=str(download_dir / "{file}"),
            overwrite=True,
            progress=False,
            max_conn=1,
        )
        fetched.extend(str(path) for path in files)

    return [Map(path) for path in fetched]


def fetch_slot_map(download_dir: Path, email: str, slot_time: str, wavelength):
    t0 = atime.Time(slot_time, scale="utc")
    t1 = t0 + 40 * u.second
    query = Fido.search(
        attrs.Time(t0, t1),
        attrs.jsoc.Series(SERIES),
        attrs.jsoc.Notify(email),
        attrs.Wavelength(wavelength),
        attrs.jsoc.Segment("image"),
        _cutout_attr(t0),
    )
    files = Fido.fetch(
        query,
        path=str(download_dir / "{file}"),
        overwrite=True,
        progress=False,
        max_conn=1,
    )
    return [Map(path) for path in files]


def save_maps(maps, output_dir: Path):
    for amap in sorted(maps, key=lambda item: (item.date.utc.isot, round(item.wavelength.to_value()))):
        wave = round(amap.wavelength.to_value(u.angstrom))
        time_tag = amap.date.strftime("%Y-%m-%dT%H-%M-%S")
        out = output_dir / f"aia_flare_{time_tag}_{wave:03d}.fits"
        amap.save(out, overwrite=True)
        print(f"saved {out}")


def missing_slots(output_dir: Path):
    missing = []
    for slot_time in TIME_SLOTS:
        slot_tag = slot_time[:16].replace(":", "-")
        for wavelength in WAVELENGTHS:
            wave = round(wavelength.to_value(u.angstrom))
            matches = list(output_dir.glob(f"aia_flare_{slot_tag}-*_{wave:03d}.fits"))
            if len(matches) != 1:
                missing.append((slot_time, wavelength))
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch AIA flare cutout test data.")
    parser.add_argument(
        "--email",
        required=True,
        help="Registered JSOC email address used for export requests.",
    )
    parser.add_argument(
        "--output-dir",
        default="demregpy/tests/data/aia_flare",
        help="Directory for saved flare cutout files.",
    )
    parser.add_argument(
        "--download-dir",
        default="/tmp/demregpy-aia-flare-downloads",
        help="Directory for downloaded AIA cutout files.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    maps = fetch_flare_maps(download_dir, args.email)
    save_maps(maps, output_dir)

    for slot_time, wavelength in missing_slots(output_dir):
        save_maps(fetch_slot_map(download_dir, args.email, slot_time, wavelength), output_dir)

    unresolved = missing_slots(output_dir)
    if unresolved:
        missing_desc = ", ".join(
            f"{slot_time} {round(wavelength.to_value(u.angstrom)):03d}" for slot_time, wavelength in unresolved
        )
        raise RuntimeError(f"Missing flare cutout records after refill: {missing_desc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
