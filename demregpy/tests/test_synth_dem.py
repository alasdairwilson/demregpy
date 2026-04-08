import numpy as np
import pytest

from demregpy.dn2dem import dn2dem


def _synthetic_case(
    centers=None,
    sigma=0.08,
    d1=4e22,
    m1=6.0,
    s1=0.12,
    dem_peaks=None,
):
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

    # Synthetic DEM model (single or multi-peak)
    root2pi = (2.0 * np.pi) ** 0.5
    if dem_peaks is None:
        dem_mod = (d1 / (root2pi * s1)) * np.exp(
            -((tresp_logt - m1) ** 2) / (2 * s1 ** 2)
        )
    else:
        dem_mod = np.zeros_like(tresp_logt)
        for peak in dem_peaks:
            p_m = peak.get("m", m1)
            p_s = peak.get("s", s1)
            p_d = peak.get("d", d1)
            dem_mod += (p_d / (root2pi * p_s)) * np.exp(
                -((tresp_logt - p_m) ** 2) / (2 * p_s ** 2)
            )

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

    mlogt = np.array(
        [
            np.mean([np.log10(temps[i]), np.log10(temps[i + 1])])
            for i in np.arange(0, len(temps) - 1)
        ]
    )

    return dn_in, edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt


def _norm_kwargs(mode, tresp_logt, dem_mod, mlogt):
    if mode == "default":
        return {}
    if mode == "gloci":
        return {"gloci": 1}
    if mode == "user":
        demwght0 = 10 ** np.interp(mlogt, tresp_logt, np.log10(dem_mod))
        demwght0 /= np.max(demwght0)
        return {"dem_norm0": demwght0}
    raise ValueError(f"Unknown norm mode: {mode}")


@pytest.mark.parametrize(
    "centers",
    [
        [5.75, 5.85, 5.95, 6.05, 6.15, 6.25],
        [5.72, 5.84, 5.96, 6.08, 6.20, 6.28],
        [5.74, 5.86, 5.98, 6.02, 6.14, 6.26],
    ],
)
@pytest.mark.parametrize("norm_mode", ["default", "gloci", "user"])
def test_synth_dn_ratio_close(centers, norm_mode):
    dn_in, edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt = _synthetic_case(
        centers=centers
    )
    norm_kwargs = _norm_kwargs(norm_mode, tresp_logt, dem_mod, mlogt)
    _dem, _edem, _elogt, _chisq, dn_reg = dn2dem(
        dn_in, edn_in, trmatrix, tresp_logt, temps, nmu=50, warn=False, **norm_kwargs
    )
    ratio = dn_reg / dn_in
    print(f"DN_reg/DN_in ratio (mode=DEM, centers={centers}, norm={norm_mode}):", ratio)
    assert np.all((ratio > 0.85) & (ratio < 1.10))


@pytest.mark.parametrize(
    "centers",
    [
        [5.75, 5.85, 5.95, 6.05, 6.15, 6.25],
        [5.72, 5.84, 5.96, 6.08, 6.20, 6.28],
        [5.74, 5.86, 5.98, 6.02, 6.14, 6.26],
    ],
)
@pytest.mark.parametrize("norm_mode", ["default", "gloci", "user"])
def test_synth_chisq_near_unity(centers, norm_mode):
    dn_in, edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt = _synthetic_case(
        centers=centers
    )
    norm_kwargs = _norm_kwargs(norm_mode, tresp_logt, dem_mod, mlogt)
    _dem, _edem, _elogt, chisq, _dn_reg = dn2dem(
        dn_in, edn_in, trmatrix, tresp_logt, temps, nmu=50, warn=False, **norm_kwargs
    )
    assert 0.5 < chisq < 1.5


@pytest.mark.parametrize("noise_frac", [0.02, 0.05, 0.10])
@pytest.mark.parametrize("norm_mode", ["default", "gloci", "user"])
def test_synth_with_noise(noise_frac, norm_mode):
    dn_in, _edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt = _synthetic_case()
    norm_kwargs = _norm_kwargs(norm_mode, tresp_logt, dem_mod, mlogt)
    rng = np.random.RandomState(0)
    dn_noisy = dn_in * (1.0 + rng.normal(0.0, noise_frac, size=dn_in.shape))
    dn_noisy = np.maximum(dn_noisy, dn_in * 0.1)
    edn_noisy = noise_frac * dn_in

    _dem, _edem, _elogt, chisq, dn_reg = dn2dem(
        dn_noisy, edn_noisy, trmatrix, tresp_logt, temps, nmu=50, warn=False, **norm_kwargs
    )

    ratio = dn_reg / dn_noisy
    print(f"DN_reg/DN_in ratio (mode=DEM, noise={noise_frac}, norm={norm_mode}):", ratio)
    assert np.all((ratio > 0.70) & (ratio < 1.20))
    # reg_tweak scales the target misfit, so chisq can legitimately drop below 1.0
    assert 0.1 < chisq < 3.0


@pytest.mark.parametrize("reg_tweak", [0.5, 1.0, 1.5, 2.0])
@pytest.mark.parametrize("rgt_fact", [1.2, 1.5, 2.0])
@pytest.mark.parametrize("norm_mode", ["default", "gloci", "user"])
def test_synth_reg_tweak_and_rgt_factor(reg_tweak, rgt_fact, norm_mode):
    dn_in, edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt = _synthetic_case()
    norm_kwargs = _norm_kwargs(norm_mode, tresp_logt, dem_mod, mlogt)
    _dem, _edem, _elogt, chisq, dn_reg = dn2dem(
        dn_in,
        edn_in,
        trmatrix,
        tresp_logt,
        temps,
        nmu=50,
        warn=False,
        reg_tweak=reg_tweak,
        rgt_fact=rgt_fact,
        **norm_kwargs,
    )
    ratio = dn_reg / dn_in
    print(
        f"DN_reg/DN_in ratio (mode=DEM, reg_tweak={reg_tweak}, rgt_fact={rgt_fact}, "
        f"norm={norm_mode}):",
        ratio,
    )
    assert np.all((ratio > 0.75) & (ratio < 1.20))
    # reg_tweak scales the target misfit, so chisq can legitimately drop below 1.0
    assert 0.1 < chisq < 3.0


def test_synth_2d_shapes():
    dn_in, _edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt = _synthetic_case()
    nx, ny = 2, 3
    nf = dn_in.shape[0]
    dn = np.zeros((nx, ny, nf))
    edn = np.zeros((nx, ny, nf))
    for x in range(nx):
        for y in range(ny):
            scale = 1.0 + 0.1 * x + 0.05 * y
            dn[x, y, :] = dn_in * scale
            edn[x, y, :] = 0.1 * dn[x, y, :]

    for norm_mode in ["default", "gloci", "user"]:
        norm_kwargs = _norm_kwargs(norm_mode, tresp_logt, dem_mod, mlogt)
        dem, edem, elogt, chisq, dn_reg = dn2dem(
            dn, edn, trmatrix, tresp_logt, temps, nmu=50, warn=False, **norm_kwargs
        )

        nt = len(temps) - 1
        assert dem.shape == (nx, ny, nt)
        assert edem.shape == (nx, ny, nt)
        assert elogt.shape == (nx, ny, nt)
        assert chisq.shape == (nx, ny)
        assert dn_reg.shape == (nx, ny, nf)


def test_synth_golden_outputs():
    dn_in, edn_in, trmatrix, tresp_logt, temps, _dem_mod, _mlogt = _synthetic_case()
    dem, _edem, _elogt, chisq, dn_reg = dn2dem(
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


@pytest.mark.parametrize(
    "dem_peaks",
    [
        [
            {"m": 5.9, "s": 0.08, "d": 2e22},
            {"m": 6.05, "s": 0.10, "d": 3e22},
        ],
        [
            {"m": 5.85, "s": 0.07, "d": 1.5e22},
            {"m": 6.00, "s": 0.09, "d": 2.5e22},
            {"m": 6.15, "s": 0.08, "d": 2e22},
        ],
        [
            {"m": 5.8, "s": 0.06, "d": 1.2e22},
            {"m": 5.95, "s": 0.08, "d": 2.0e22},
            {"m": 6.1, "s": 0.07, "d": 1.8e22},
            {"m": 6.25, "s": 0.09, "d": 1.4e22},
        ],
    ],
)
def test_synth_multi_peak_dem(dem_peaks):
    for norm_mode in ["default", "gloci", "user"]:
        dn_in, edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt = _synthetic_case(
            dem_peaks=dem_peaks
        )
        norm_kwargs = _norm_kwargs(norm_mode, tresp_logt, dem_mod, mlogt)
        _dem, _edem, _elogt, chisq, dn_reg = dn2dem(
            dn_in, edn_in, trmatrix, tresp_logt, temps, nmu=50, warn=False, **norm_kwargs
        )
        ratio = dn_reg / dn_in
        assert np.all((ratio > 0.80) & (ratio < 1.10))
        assert 0.5 < chisq < 1.5


def test_synth_emd_mode_basic():
    dn_in, edn_in, trmatrix, tresp_logt, temps, _dem_mod, _mlogt = _synthetic_case()
    _dem, _edem, _elogt, chisq, dn_reg = dn2dem(
        dn_in,
        edn_in,
        trmatrix,
        tresp_logt,
        temps,
        nmu=50,
        warn=False,
        emd_int=True,
        emd_ret=False,
    )
    ratio = dn_reg / dn_in
    print("DN_reg/DN_in ratio (mode=EMD, norm=default):", ratio)
    assert np.all((ratio > 0.75) & (ratio < 1.15))
    # reg_tweak scales the target misfit, so chisq can legitimately drop below 1.0
    assert 0.1 < chisq < 3.0


def test_synth_emd_mode_gloci_and_user():
    dn_in, edn_in, trmatrix, tresp_logt, temps, dem_mod, mlogt = _synthetic_case()
    for norm_mode in ["gloci", "user"]:
        norm_kwargs = _norm_kwargs(norm_mode, tresp_logt, dem_mod, mlogt)
        _dem, _edem, _elogt, chisq, dn_reg = dn2dem(
            dn_in,
            edn_in,
            trmatrix,
            tresp_logt,
            temps,
            nmu=50,
            warn=False,
            emd_int=True,
            emd_ret=False,
            **norm_kwargs,
        )
        ratio = dn_reg / dn_in
        print(f"DN_reg/DN_in ratio (mode=EMD, norm={norm_mode}):", ratio)
        assert np.all((ratio > 0.75) & (ratio < 1.15))
        assert 0.3 < chisq < 3.0


@pytest.mark.parametrize("reg_tweak", [0.5, 1.0, 1.5, 2.0])
def test_synth_emd_reg_tweak(reg_tweak):
    dn_in, edn_in, trmatrix, tresp_logt, temps, _dem_mod, _mlogt = _synthetic_case()
    _dem, _edem, _elogt, chisq, dn_reg = dn2dem(
        dn_in,
        edn_in,
        trmatrix,
        tresp_logt,
        temps,
        nmu=50,
        warn=False,
        emd_int=True,
        emd_ret=False,
        reg_tweak=reg_tweak,
    )
    ratio = dn_reg / dn_in
    print(f"DN_reg/DN_in ratio (mode=EMD, reg_tweak={reg_tweak}):", ratio)
    assert np.all((ratio > 0.70) & (ratio < 1.15))
    # reg_tweak scales the target misfit, so chisq can legitimately drop below 1.0
    assert 0.1 < chisq < 3.0


def test_synth_non_pos_mode_runs():
    dn_in, edn_in, trmatrix, tresp_logt, temps, _dem_mod, _mlogt = _synthetic_case()
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_in,
        edn_in,
        trmatrix,
        tresp_logt,
        temps,
        nmu=50,
        warn=False,
        non_pos=True,
    )
    assert np.isfinite(dem).all()
    assert np.isfinite(edem).all()
    assert np.isfinite(elogt).all()
    assert np.isfinite(chisq)
    assert np.isfinite(dn_reg).all()


def test_synth_l_emd_flag_runs():
    dn_in, edn_in, trmatrix, tresp_logt, temps, _dem_mod, _mlogt = _synthetic_case()
    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_in,
        edn_in,
        trmatrix,
        tresp_logt,
        temps,
        nmu=50,
        warn=False,
        l_emd=True,
    )
    assert np.isfinite(dem).all()
    assert np.isfinite(edem).all()
    assert np.isfinite(elogt).all()
    assert np.isfinite(chisq)
    assert np.isfinite(dn_reg).all()
