from pathlib import Path

import numpy as np
import pytest
import scipy.io as io
from sunpy.map import Map

from demregpy import dn2dem
from demregpy.demmap import dem_inv_gsvd, dem_pix, dem_reg_map
from demregpy.tresp import aia_tresp


def _synthetic_dem_pix_inputs():
    centers = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.25])
    tresp_logt = np.linspace(5.7, 6.3, 7)
    nt = len(tresp_logt)
    nf = len(centers)
    trmatrix = np.zeros((nt, nf))
    for i, c in enumerate(centers):
        trmatrix[:, i] = np.exp(-((tresp_logt - c) ** 2) / (2 * 0.08 ** 2))

    root2pi = (2.0 * np.pi) ** 0.5
    dem_mod = (4e22 / (root2pi * 0.12)) * np.exp(-((tresp_logt - 6.0) ** 2) / (2 * 0.12 ** 2))
    dlogt = np.full(nt, tresp_logt[1] - tresp_logt[0])
    tc_full = np.zeros((nt, nf))
    for i in range(nf):
        tc_full[:, i] = dem_mod * trmatrix[:, i] * 10 ** tresp_logt * np.log(10 ** dlogt)

    dnin = np.sum(tc_full, axis=0)
    ednin = 0.1 * dnin
    glc = np.zeros(nf)
    dem_norm0 = np.ones(nt)
    return dnin, ednin, trmatrix, tresp_logt, dlogt, glc, dem_norm0


def _aia_files():
    data_dir = Path(__file__).resolve().parent / "data" / "aia"
    waves = [94, 131, 171, 193, 211, 335]
    files = [data_dir / f"aia_synoptic_2014-01-01T00-00-00_{w:03d}.fits" for w in waves]
    if not all(p.exists() for p in files):
        pytest.skip("AIA synoptic data not present. Run scripts/fetch_aia_cutouts.py")
    return files


def test_dem_pix_golden_outputs():
    dnin, ednin, trmatrix, logt, dlogt, glc, dem_norm0 = _synthetic_dem_pix_inputs()
    dem, edem, elogt, chisq, dn_reg = dem_pix(
        dnin, ednin, trmatrix, logt, dlogt, glc, dem_norm0=dem_norm0, warn=False
    )

    expected_dem = np.array([
        4.9001405218412504e26,
        5.0269204612449675e27,
        1.7172112149704068e28,
        2.7844235246882650e28,
        2.5160765211326454e28,
        1.2582381066073235e28,
        2.5298774405784752e27,
    ])
    expected_edem = np.array([
        1.4036303366534813e27,
        1.2945698688876086e27,
        2.5558592985833515e27,
        3.8720925742199345e27,
        3.3393402865501804e27,
        1.8814658004051443e27,
        2.0553788098382618e27,
    ])
    expected_elogt = np.array([
        6.4705882352964342e-02,
        7.6470588235310162e-02,
        5.8823529411777071e-02,
        5.8823529411777071e-02,
        5.8823529411777071e-02,
        8.2352941176496494e-02,
        5.8823529411777071e-02,
    ])
    expected_dn_reg = np.array([
        7.7116460996933640e27,
        2.3337351427268551e28,
        4.2333655763949478e28,
        4.8788324123894541e28,
        3.6414260250620058e28,
        1.6981402517644043e28,
    ])

    np.testing.assert_allclose(dem, expected_dem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(edem, expected_edem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(elogt, expected_elogt, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(dn_reg, expected_dn_reg, rtol=1e-8, atol=0.0)
    assert abs(chisq - 0.1907488958648144) < 1e-12


def test_dem_reg_map_golden_output():
    sigmaa = np.array([1.0, 0.5])
    sigmab = np.array([0.2, 0.3])
    U = np.eye(2)
    W = np.eye(2)
    data = np.array([1.0, 1.0])
    err = np.array([1.0, 1.0])

    lamb = dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak=1.0, nmu=10)
    assert abs(lamb - 4.9999999999999964) < 1e-12


def test_dem_inv_gsvd_golden_output():
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    B = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    alpha, beta, U, V, W = dem_inv_gsvd(A, B)

    expected_alpha = np.array([0.9919565989236012, 0.6079123653201358, 0.0])
    expected_beta = np.array([0.1265784065592676, 0.7940041322334759, 1.0])
    expected_U = np.array([
        [-0.3933981805934430, -0.9193681850127032],
        [-0.9193681850127032, 0.3933981805934430],
        [0.0, 0.0],
    ])
    expected_V = np.array([
        [-0.5194626249289980, 0.8544931656480434, 0.0],
        [0.0, 0.0, 1.0],
        [-0.8544931656480434, -0.5194626249289980, 0.0],
    ])
    expected_W = np.array([
        [-0.0657527477233592, 0.6784711015356956, 0.0],
        [0.0, 0.0, 0.0],
        [-0.1081603876567152, -0.4124554660896158, 0.0],
    ])

    np.testing.assert_allclose(alpha, expected_alpha, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(beta, expected_beta, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(U, expected_U, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(V, expected_V, rtol=1e-7, atol=0.0)
    np.testing.assert_allclose(W, expected_W, rtol=1e-7, atol=0.0)


def test_aia_synoptic_central_pixel_golden():
    maps = [Map(str(p)) for p in _aia_files()]
    maps = sorted(maps, key=lambda x: x.wavelength)

    trin = io.readsav(aia_tresp)
    tresp_logt = np.array(trin["logt"])
    nf = len(trin["tr"][:])
    trmatrix = np.zeros((len(tresp_logt), nf))
    for i in range(nf):
        trmatrix[:, i] = trin["tr"][i]

    cx = maps[0].data.shape[0] // 2
    cy = maps[0].data.shape[1] // 2
    dn = np.array([m.data[cx, cy] for m in maps], dtype=float)
    edn = 0.1 * dn + 1e-8
    temps = 10 ** np.linspace(5.7, 7.1, num=17)

    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn, edn, trmatrix, tresp_logt, temps, nmu=40, warn=False
    )

    expected_dn = np.array([1.9375, 9.0, 155.0, 220.9375, 125.125, 3.4375])
    expected_dem = np.array([
        3.9249346697138215e19,
        1.0031267124265219e20,
        1.6881475736966655e20,
        2.3727228286862746e20,
        2.8144321204964336e20,
        3.9845459297879724e20,
        5.5482596897758349e20,
        3.2746623175662107e20,
        6.7855283830401194e19,
        7.0125008938609777e18,
        3.0972172322849398e18,
        1.3534675170245511e19,
        4.6906697525057544e19,
        7.1413410716315234e19,
        9.5373917692145242e19,
        9.1634708501010489e19,
    ])
    expected_dn_reg = np.array([
        1.8319994597570531,
        8.5043537661176800,
        1.4334401152774559e02,
        2.2201041934342710e02,
        8.1062243858595260e01,
        3.5156718223827933,
    ])

    np.testing.assert_allclose(dn, expected_dn, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dem, expected_dem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(dn_reg, expected_dn_reg, rtol=1e-8, atol=0.0)
    assert abs(float(chisq) - 2.270053708011263) < 1e-8
