"""
Utilities shared by the documentation examples.
"""

from pathlib import Path

import numpy as np

from sunpy.map import Map

from demregpy.tresp import load_aia_response

AIA_WAVELENGTHS = [94, 131, 171, 193, 211, 335]


def load_aia_test_maps():
    """
    Load the local AIA synoptic FITS files used by the test suite.
    """
    data_dir = Path(__file__).resolve().parents[1] / "demregpy" / "tests" / "data" / "aia"
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


def make_synthetic_case(
    centers=None,
    sigma=0.08,
    dem_peaks=None,
):
    """
    Build a compact synthetic DEM inversion problem.
    """
    tresp_logt = np.linspace(5.7, 6.3, 7)
    if centers is None:
        centers = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.25])
    centers = np.array(centers)
    nf = len(centers)
    trmatrix = np.zeros((len(tresp_logt), nf))
    for i, center in enumerate(centers):
        trmatrix[:, i] = np.exp(-((tresp_logt - center) ** 2) / (2 * sigma ** 2))

    if dem_peaks is None:
        dem_peaks = [
            {"m": 6.0, "s": 0.12, "d": 4e22},
        ]

    root2pi = (2.0 * np.pi) ** 0.5
    dem_mod = np.zeros_like(tresp_logt)
    for peak in dem_peaks:
        dem_mod += (peak["d"] / (root2pi * peak["s"])) * np.exp(
            -((tresp_logt - peak["m"]) ** 2) / (2 * peak["s"] ** 2)
        )

    dlogt = np.full(len(tresp_logt), tresp_logt[1] - tresp_logt[0])
    tc_full = np.zeros((len(tresp_logt), nf))
    for i in range(nf):
        tc_full[:, i] = dem_mod * trmatrix[:, i] * 10 ** tresp_logt * np.log(10 ** dlogt)

    dn_in = np.sum(tc_full, axis=0)
    edn_in = 0.1 * dn_in
    logtemps = np.linspace(tresp_logt.min(), tresp_logt.max(), len(tresp_logt) + 1)
    temps = 10 ** logtemps
    mlogt = np.array(
        [
            np.mean([np.log10(temps[i]), np.log10(temps[i + 1])])
            for i in range(len(temps) - 1)
        ]
    )
    return {
        "dn_in": dn_in,
        "edn_in": edn_in,
        "trmatrix": trmatrix,
        "tresp_logt": tresp_logt,
        "temps": temps,
        "dem_mod": dem_mod,
        "mlogt": mlogt,
    }


def make_user_weight(mlogt, tresp_logt, dem_mod):
    """
    Construct a normalized user weighting curve from a DEM model.
    """
    demwght0 = 10 ** np.interp(mlogt, tresp_logt, np.log10(dem_mod))
    return demwght0 / np.max(demwght0)


def make_synthetic_time_series(n_steps=8):
    """
    Build a small synthetic time series suitable for a DEMogram example.
    """
    base = make_synthetic_case()
    nf = base["dn_in"].shape[0]
    dn = np.zeros((n_steps, nf))
    edn = np.zeros((n_steps, nf))

    for i in range(n_steps):
        scale = 0.8 + 0.08 * i
        warm_boost = 1.0 + 0.25 * np.sin(i / max(n_steps - 1, 1) * np.pi)
        dn[i, :] = base["dn_in"] * scale
        dn[i, -2:] *= warm_boost
        edn[i, :] = 0.1 * dn[i, :] + 1e-8

    return base | {"dn_in": dn, "edn_in": edn}


def make_aia_pixel_problem(x=None, y=None):
    """
    Build a single-pixel AIA inversion problem from the local FITS fixtures.
    """
    maps = load_aia_test_maps()
    channels, tresp_logt, trmatrix = load_aia_response()
    if x is None:
        x = maps[0].data.shape[0] // 2
    if y is None:
        y = maps[0].data.shape[1] // 2
    dn = np.array([amap.data[x, y] for amap in maps], dtype=float)
    edn = 0.1 * dn + 1e-8
    temps = 10 ** np.linspace(5.7, 7.1, num=17)
    mlogt = np.array(
        [
            np.mean([np.log10(temps[i]), np.log10(temps[i + 1])])
            for i in range(len(temps) - 1)
        ]
    )
    return {
        "maps": maps,
        "channels": channels,
        "tresp_logt": tresp_logt,
        "trmatrix": trmatrix,
        "temps": temps,
        "mlogt": mlogt,
        "dn_in": dn,
        "edn_in": edn,
        "x": x,
        "y": y,
    }


def make_aia_patch_problem(width=10, height=10):
    """
    Build a small AIA patch inversion problem from the local FITS fixtures.
    """
    pixel = make_aia_pixel_problem()
    maps = pixel["maps"]
    nx, ny = maps[0].data.shape
    x0 = nx // 2 - width // 2
    y0 = ny // 2 - height // 2
    x1 = x0 + width
    y1 = y0 + height

    data = np.stack([amap.data[x0:x1, y0:y1] for amap in maps], axis=-1).astype(float)
    edata = 0.1 * data + 1e-8
    return pixel | {
        "dn_in": data,
        "edn_in": edata,
        "x_slice": slice(x0, x1),
        "y_slice": slice(y0, y1),
    }
