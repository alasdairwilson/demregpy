from pathlib import Path

import numpy as np
import pytest

from sunpy.map import Map

from demregpy import dn2dem
from demregpy.demmap import dem_pix
from demregpy.tresp import load_aia_response


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
        3.9946257705148112e26,
        5.2569731605668519e27,
        1.6674786377718149e28,
        2.4760991998726412e28,
        2.2708483024572261e28,
        1.2472243085750785e28,
        2.8742723050118840e27,
    ])
    expected_edem = np.array([
        9.1042036819147464e26,
        8.1551097489343939e26,
        2.0619394987489058e27,
        2.6646289329851594e27,
        2.4626189852584232e27,
        1.3121985158878014e27,
        1.3294283664289035e27,
    ])
    expected_elogt = np.array([
        6.4705882352940947e-02,
        7.6470588235294290e-02,
        6.4705882352941391e-02,
        7.0588235294117396e-02,
        7.0588235294117840e-02,
        8.2352941176470740e-02,
        5.8823529411764497e-02,
    ])
    expected_dn_reg = np.array([
        7.7171181960892595e27,
        2.2551682448873074e28,
        3.9003703368380743e28,
        4.4134530767982199e28,
        3.3830482708874761e28,
        1.6727878130152810e28,
    ])

    np.testing.assert_allclose(dem, expected_dem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(edem, expected_edem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(elogt, expected_elogt, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(dn_reg, expected_dn_reg, rtol=1e-8, atol=0.0)
    assert abs(chisq - 0.9999999999997423) < 1e-12


def test_dem_pix_golden_outputs_high_nf():
    dnin, ednin, trmatrix, logt, dlogt, glc, dem_norm0 = _synthetic_dem_pix_inputs_high_nf()
    dem, edem, elogt, chisq, dn_reg = dem_pix(
        dnin, ednin, trmatrix, logt, dlogt, glc, dem_norm0=dem_norm0, warn=False
    )

    expected_dem = np.array([
        1.4740227441924264e25,
        2.3589926169077172e25,
        3.5192047463393497e25,
        1.3065117326154325e26,
        4.0002132745327483e26,
        9.2221205728641349e26,
        1.5772043340610200e27,
        2.1821945570717943e27,
        2.6003838481390101e27,
        2.8251728154278154e27,
        2.9194954829597226e27,
        2.9435245334320708e27,
        2.9110068828207709e27,
        2.7749896998488814e27,
        2.4453296483285472e27,
        1.8736007781510827e27,
        1.2401322800185790e27,
        6.4064921401392215e26,
        4.4339835037612813e26,
        8.8837153753850656e26,
        1.7674367691488677e27,
        2.5873091413631460e27,
        3.1567680554614268e27,
        3.3838981305855190e27,
        3.3954595347870344e27,
        3.3960144984102865e27,
        3.4802935455656127e27,
        3.5649233052596901e27,
        3.3976315479785414e27,
        2.7703639997791207e27,
        1.7965394675165539e27,
        7.9909828392766633e26,
        1.7681373464599680e26,
        4.0723593501216724e25,
        4.6652178580532981e25,
        1.7000599773596846e25,
        -9.6889741988951430e24,
        -1.5995569662724536e25,
        -8.7328003978372578e24,
        -2.5129311228611365e24,
    ])
    expected_edem = np.array([
        3.4323081639002880e24,
        4.9131040556693808e24,
        6.2842694704355842e24,
        1.8928913694721576e25,
        4.4085449804129748e25,
        7.3698337094483073e25,
        9.5336658778450525e25,
        1.1613532859852180e26,
        1.3411996793513494e26,
        1.4218189287731740e26,
        1.4318366970778460e26,
        1.4273341405136993e26,
        1.4306960360221380e26,
        1.4259723719677078e26,
        1.3531741413416142e26,
        1.1345370152562103e26,
        8.1564030568063851e25,
        4.4328961921174002e25,
        3.0894054100564595e25,
        5.9954848641146223e25,
        1.1245008840158914e26,
        1.5193548581282820e26,
        1.6804304392085949e26,
        1.6266165050018245e26,
        1.5068761525973083e26,
        1.4557762163015441e26,
        1.5218778691455523e26,
        1.6743252838525923e26,
        1.7736145899333163e26,
        1.6064232100519927e26,
        1.1378886427202259e26,
        5.9420960544383621e25,
        2.0485886304736044e25,
        9.6083992924349407e24,
        2.6509727196541224e25,
        3.7617209652917168e25,
        3.5257950545494864e25,
        2.2715309270142153e25,
        8.9938709144911510e24,
        2.1981318233103811e24,
    ])
    expected_elogt = np.array([
        5.8823529411764497e-02,
        5.8823529411764497e-02,
        5.8823529411764497e-02,
        4.4117647058823373e-02,
        5.8823529411764497e-02,
        5.8823529411764497e-02,
        7.3529411764705621e-02,
        7.3529411764705621e-02,
        8.8235294117647189e-02,
        8.8235294117647189e-02,
        8.8235294117647189e-02,
        1.0294117647058831e-01,
        1.0294117647058831e-01,
        8.8235294117646745e-02,
        7.3529411764705621e-02,
        7.3529411764705621e-02,
        1.7647058823529393e-01,
        1.7647058823529393e-01,
        1.7647058823529438e-01,
        1.6176470588235325e-01,
        5.8823529411764941e-02,
        7.3529411764706065e-02,
        8.8235294117647189e-02,
        8.8235294117647189e-02,
        1.0294117647058831e-01,
        1.1764705882352944e-01,
        1.0294117647058787e-01,
        8.8235294117646745e-02,
        8.8235294117647189e-02,
        7.3529411764706065e-02,
        5.8823529411764941e-02,
        5.8823529411764941e-02,
        5.8823529411764941e-02,
        4.4117647058823817e-02,
        1.3235294117647101e-01,
        1.3235294117647101e-01,
        1.3235294117647101e-01,
        1.3235294117647101e-01,
        1.1764705882352988e-01,
        1.1764705882352988e-01,
    ])
    expected_dn_reg = np.array([
        2.2641743092666683e27,
        5.3724673597707682e27,
        9.5627966050132451e27,
        1.3354999093597919e28,
        1.5447865288916157e28,
        1.5363542224067560e28,
        1.3195352962343526e28,
        9.9634946375741266e27,
        8.0323552836642616e27,
        9.4107629081757911e27,
        1.3331723708067197e28,
        1.7094535691344940e28,
        1.8953424352165094e28,
        1.8477071428426570e28,
        1.5336469511548993e28,
        1.0061768197327194e28,
        4.8600946464432453e27,
        1.6526323543867828e27,
    ])

    np.testing.assert_allclose(dem, expected_dem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(edem, expected_edem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(elogt, expected_elogt, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(dn_reg, expected_dn_reg, rtol=1e-8, atol=0.0)
    assert abs(chisq - 38.443359375000085) < 1e-10


def test_aia_synoptic_central_pixel_golden():
    maps = [Map(str(p)) for p in _aia_files()]
    maps = sorted(maps, key=lambda x: x.wavelength)

    _channels, tresp_logt, trmatrix = load_aia_response()

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
        3.9295340769657381e19,
        1.0049171654541651e20,
        1.6913271270047144e20,
        2.3751527707734796e20,
        2.8088812666852085e20,
        3.9819226653221139e20,
        5.5709891355696405e20,
        3.2855571653156530e20,
        6.7051822186842907e19,
        6.5080590488457165e18,
        3.0224235082084613e18,
        1.3477484675817300e19,
        4.6901421168426729e19,
        7.1490917858443182e19,
        9.5452066433491337e19,
        9.1658180121378341e19,
    ])
    expected_dn_reg = np.array([
        1.8331657208887395e00,
        8.5100995119648033e00,
        1.4344730658702397e02,
        2.2223320970771047e02,
        8.1241131219980801e01,
        3.5175924957129192e00,
    ])

    np.testing.assert_allclose(dn, expected_dn, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(dem, expected_dem, rtol=1e-8, atol=0.0)
    np.testing.assert_allclose(dn_reg, expected_dn_reg, rtol=1e-8, atol=0.0)
    assert abs(float(chisq) - 2.2499999999997167) < 1e-8
