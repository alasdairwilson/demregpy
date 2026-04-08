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


def _synthetic_dem_pix_inputs_high_nf():
    nt = 40
    nf = 18
    centers = np.linspace(5.8, 7.0, nf)
    logt = np.linspace(5.7, 7.2, nt)
    dlogt = np.full(nt, logt[1] - logt[0])
    trmatrix = np.zeros((nt, nf))
    for i, c in enumerate(centers):
        trmatrix[:, i] = np.exp(-((logt - c) ** 2) / (2 * 0.09 ** 2))

    root2pi = np.sqrt(2.0 * np.pi)
    dem_mod = (
        (2.5e22 / (root2pi * 0.10)) * np.exp(-((logt - 6.1) ** 2) / (2 * 0.10 ** 2))
        + (1.6e22 / (root2pi * 0.07)) * np.exp(-((logt - 6.65) ** 2) / (2 * 0.07 ** 2))
    )
    tc_full = np.zeros((nt, nf))
    for i in range(nf):
        tc_full[:, i] = dem_mod * trmatrix[:, i] * 10 ** logt * np.log(10 ** dlogt)

    dnin = np.sum(tc_full, axis=0)
    ednin = 0.1 * dnin
    glc = np.zeros(nf)
    dem_norm0 = np.ones(nt)
    return dnin, ednin, trmatrix, logt, dlogt, glc, dem_norm0


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


def test_dem_pix_golden_outputs_high_nf():
    dnin, ednin, trmatrix, logt, dlogt, glc, dem_norm0 = _synthetic_dem_pix_inputs_high_nf()
    dem, edem, elogt, chisq, dn_reg = dem_pix(
        dnin, ednin, trmatrix, logt, dlogt, glc, dem_norm0=dem_norm0, warn=False
    )

    expected_dem = np.array([
        1.4123955486279001e25, 2.2148313862698478e25, 3.5894530195887170e25,
        1.2677339502056880e26, 3.7208683611344642e26, 8.2610344958579857e26,
        1.3684265105109065e27, 1.8309071712728653e27, 2.1179163430278924e27,
        2.2479251536628070e27, 2.2877020793751224e27, 2.2905108601190346e27,
        2.2674567624865182e27, 2.1797467784993441e27, 1.9495173161263182e27,
        1.5133888895060704e27, 1.0159972705392316e27, 5.3447399086103907e26,
        3.4939238385211275e26, 6.8918086517997882e26, 1.3953905557019004e27,
        2.0407403493028786e27, 2.4636463968545674e27, 2.6134702241650741e27,
        2.5967221194747111e27, 2.5859860233510754e27, 2.6623943436123942e27,
        2.7655325661773710e27, 2.6956955805256520e27, 2.2743673672504379e27,
        1.5414741888044745e27, 7.2721924504662063e26, 1.7384487391623236e26,
        5.2625614579888752e25, 8.4448318162297639e25, 7.4189570337642581e25,
        4.5502327057338231e25, 2.0062514012196167e25, 5.5445887787500045e24,
        9.7962933046103014e23,
    ])
    expected_edem = np.array([
        2.5100382512501505e24, 3.5965916951703670e24, 5.1486495155315894e24,
        1.5373600081670576e25, 3.6260480832727146e25, 6.2158593510453671e25,
        8.0914166493747028e25, 9.4261868847213907e25, 1.0424613454057976e26,
        1.0829389672969209e26, 1.0805386725833002e26, 1.0731624810413983e26,
        1.0783515138567236e26, 1.0833279102545370e26, 1.0386490505236621e26,
        8.7435407970031703e25, 6.3112220995008984e25, 3.4717842895451321e25,
        2.2875917351064857e25, 4.4021424552421416e25, 8.4904040847705267e25,
        1.1586569111678321e26, 1.2793797226668021e26, 1.2308635700389908e26,
        1.1289413720766312e26, 1.0853047335843834e26, 1.1419362808267897e26,
        1.2748288113572611e26, 1.3673144102816669e26, 1.2573560594821142e26,
        9.2681419509178569e25, 5.2342244332610240e25, 1.8106583189015192e25,
        8.8020082093388902e24, 2.2622993285384818e25, 3.0708277537856737e25,
        2.8021669792613484e25, 1.7776575341895101e25, 6.8604578587863120e24,
        1.6456849248098388e24,
    ])
    expected_elogt = np.array([
        5.8823529411764498e-02, 5.8823529411764498e-02, 4.4117647058823380e-02,
        5.8823529411764498e-02, 5.8823529411764498e-02, 5.8823529411764498e-02,
        7.3529411764705564e-02, 7.3529411764705564e-02, 7.3529411764706050e-02,
        8.8235294117647194e-02, 8.8235294117647194e-02, 1.0294117647058831e-01,
        1.0294117647058831e-01, 8.8235294117646708e-02, 8.8235294117646708e-02,
        7.3529411764705564e-02, 1.7647058823529387e-01, 1.7647058823529387e-01,
        1.7647058823529436e-01, 1.6176470588235328e-01, 5.8823529411764942e-02,
        7.3529411764706050e-02, 8.8235294117647194e-02, 8.8235294117647194e-02,
        1.0294117647058831e-01, 1.1764705882352943e-01, 1.0294117647058790e-01,
        8.8235294117646708e-02, 8.8235294117647194e-02, 7.3529411764706050e-02,
        5.8823529411764942e-02, 5.8823529411764942e-02, 5.8823529411764942e-02,
        4.4117647058823845e-02, 4.4117647058823845e-02, 1.1764705882352992e-01,
        1.1764705882352992e-01, 1.3235294117647098e-01, 1.3235294117647098e-01,
        1.3235294117647098e-01,
    ])
    expected_dn_reg = np.array([
        1.9938914536573061e27, 4.5968825951960177e27, 7.9531419216301888e27,
        1.0823325726659797e28, 1.2286266232484853e28, 1.2126742760221538e28,
        1.0446721447554426e28, 7.9464695751592733e27, 6.4055812356838101e27,
        7.4151912979716462e27, 1.0394425304900580e28, 1.3236121016309702e28,
        1.4656974689512103e28, 1.4420909769982523e28, 1.2213534136229142e28,
        8.2493485602484588e27, 4.1717398088257394e27, 1.5901190829237344e27,
    ])

    np.testing.assert_allclose(dem, expected_dem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(edem, expected_edem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(elogt, expected_elogt, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(dn_reg, expected_dn_reg, rtol=1e-8, atol=0.0)
    assert abs(chisq - 46.37409968570018) < 1e-10


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

    dem, _edem, _elogt, chisq, dn_reg = dn2dem(
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
