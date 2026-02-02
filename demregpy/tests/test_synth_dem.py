import numpy as np
import pytest

from demregpy.dn2dem import dn2dem


def _synthetic_case(centers=None, sigma=0.08, d1=4e22, m1=6.0, s1=0.12):
    # Response grid in logT (matches temps bins)
    tresp_logt = np.linspace(5.7, 6.3, 7)
    nt = len(tresp_logt)
    if centers is None:
        centers = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.25])
    centers = np.array(centers)
    nf = len(centers)

    # Simple gaussian response curves
    trmatrix = np.zeros((nt, nf))
    for i, c in enumerate(centers):
        trmatrix[:, i] = np.exp(-((tresp_logt - c) ** 2) / (2 * sigma ** 2))

    # Synthetic DEM model
    root2pi = (2.0 * np.pi) ** 0.5
    dem_mod = (d1 / (root2pi * s1)) * np.exp(-((tresp_logt - m1) ** 2) / (2 * s1 ** 2))

    # Build DN from DEM and response (mirrors example)
    step = tresp_logt[1] - tresp_logt[0]
    dlogt = np.full(nt, step)
    tc_full = np.zeros((nt, nf))
    for i in range(nf):
        tc_full[:, i] = dem_mod * trmatrix[:, i] * 10 ** tresp_logt * np.log(10 ** dlogt)
    dn_in = np.sum(tc_full, axis=0)
    edn_in = 0.1 * dn_in

    # Temps for dn2dem are bin edges
    logtemps = np.linspace(tresp_logt.min(), tresp_logt.max(), nt + 1)
    temps = 10 ** logtemps

    return dn_in, edn_in, trmatrix, tresp_logt, temps


@pytest.mark.parametrize(
    "centers",
    [
        [5.75, 5.85, 5.95, 6.05, 6.15, 6.25],
        [5.72, 5.84, 5.96, 6.08, 6.20, 6.28],
        [5.74, 5.86, 5.98, 6.02, 6.14, 6.26],
    ],
)
def test_synth_dn_ratio_close(centers):
    dn_in, edn_in, trmatrix, tresp_logt, temps = _synthetic_case(centers=centers)
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_in, edn_in, trmatrix, tresp_logt, temps, nmu=50, warn=False
    )
    ratio = dn_reg / dn_in
    print(f"DN_reg/DN_in ratio (centers={centers}):", ratio)
    assert np.all((ratio > 0.85) & (ratio < 1.10))


@pytest.mark.parametrize(
    "centers",
    [
        [5.75, 5.85, 5.95, 6.05, 6.15, 6.25],
        [5.72, 5.84, 5.96, 6.08, 6.20, 6.28],
        [5.74, 5.86, 5.98, 6.02, 6.14, 6.26],
    ],
)
def test_synth_chisq_near_unity(centers):
    dn_in, edn_in, trmatrix, tresp_logt, temps = _synthetic_case(centers=centers)
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_in, edn_in, trmatrix, tresp_logt, temps, nmu=50, warn=False
    )
    assert 0.5 < chisq < 1.5


def test_synth_2d_shapes():
    dn_in, edn_in, trmatrix, tresp_logt, temps = _synthetic_case()
    nx, ny = 2, 3
    nf = dn_in.shape[0]
    dn = np.zeros((nx, ny, nf))
    edn = np.zeros((nx, ny, nf))
    for x in range(nx):
        for y in range(ny):
            scale = 1.0 + 0.1 * x + 0.05 * y
            dn[x, y, :] = dn_in * scale
            edn[x, y, :] = 0.1 * dn[x, y, :]

    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn, edn, trmatrix, tresp_logt, temps, nmu=50, warn=False
    )

    nt = len(temps) - 1
    assert dem.shape == (nx, ny, nt)
    assert edem.shape == (nx, ny, nt)
    assert elogt.shape == (nx, ny, nt)
    assert chisq.shape == (nx, ny)
    assert dn_reg.shape == (nx, ny, nf)


def test_synth_golden_outputs():
    dn_in, edn_in, trmatrix, tresp_logt, temps = _synthetic_case()
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_in, edn_in, trmatrix, tresp_logt, temps, nmu=50, warn=False
    )

    expected_dem = np.array([
        1.8998512156840871e22,
        5.6440134829154069e22,
        9.1930471180977148e22,
        1.1650968982247284e23,
        9.2088301149635700e22,
        5.6544958483669910e22,
        1.8878719745010950e22,
    ])
    expected_dn_reg = np.array([
        7.477884135223385e27,
        2.090509339898088e28,
        3.833124954797410e28,
        4.513470586328764e28,
        3.460360957796536e28,
        1.687714518984836e28,
    ])
    expected_chisq = 1.1057804956332273

    np.testing.assert_allclose(dem, expected_dem, rtol=1e-5, atol=0.0)
    np.testing.assert_allclose(dn_reg, expected_dn_reg, rtol=1e-5, atol=0.0)
    assert abs(chisq - expected_chisq) < 1e-6
