from importlib import import_module

import numpy as np
import pytest

from demregpy.dn2dem import dn2dem


def _basic_dn2dem_inputs():
    dn = np.array([1.0, 2.0])
    edn = np.array([0.1, 0.2])
    tresp = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0],
            [13.0, 14.0],
        ]
    )
    tresp_logt = np.linspace(5.0, 5.6, tresp.shape[0])
    temps = 10 ** np.linspace(5.0, 5.6, tresp.shape[0] + 1)
    return dn, edn, tresp, tresp_logt, temps


def test_dn2dem_rejects_partially_nonfinite_dem_norm0():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()
    dem_norm0 = np.array([1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match=r"dem_norm0.*finite"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, dem_norm0=dem_norm0, warn=False)


def test_dn2dem_handles_non_positive_tresp_entries():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()
    tresp[0, 0] = -1.0

    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn,
        edn,
        tresp,
        tresp_logt,
        temps,
        warn=False,
        non_pos=True,
    )

    assert np.isfinite(dem).all()
    assert np.isfinite(edem).all()
    assert np.isfinite(elogt).all()
    assert np.isfinite(chisq)
    assert np.isfinite(dn_reg).all()


def test_dn2dem_rejects_filter_with_no_positive_tresp_values():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()
    tresp[:, 0] = -1.0

    with pytest.raises(ValueError, match="positive"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, warn=False)


def test_dn2dem_rejects_too_short_temperature_grid():
    dn = np.array([1.0, 2.0])
    edn = np.array([0.1, 0.2])
    tresp = np.array(
        [
            [1.0, 3.0],
            [2.0, 4.0],
            [5.0, 6.0],
        ]
    )
    tresp_logt = np.linspace(5.0, 5.2, tresp.shape[0])
    temps = 10 ** np.linspace(5.0, 5.2, 3)

    with pytest.raises(ValueError, match="at least"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, warn=False)


def test_dn2dem_rejects_tresp_filter_count_mismatch():
    dn = np.array([1.0, 2.0, 3.0])
    edn = np.array([0.1, 0.2, 0.3])
    tresp = np.ones((7, 2))
    tresp_logt = np.linspace(5.0, 5.6, tresp.shape[0])
    temps = 10 ** np.linspace(5.0, 5.6, tresp.shape[0] + 1)

    with pytest.raises(ValueError, match="same number of filters"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, warn=False)


def test_dn2dem_rejects_mismatched_count_shapes():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()

    with pytest.raises(ValueError, match="same shape"):
        dn2dem(dn, edn[np.newaxis, :], tresp, tresp_logt, temps, warn=False)


def test_dn2dem_rejects_more_than_three_leading_dimensions():
    dn = np.ones((1, 2, 3, 4, 2))
    edn = np.ones_like(dn) * 0.1
    tresp = np.ones((7, 2))
    tresp_logt = np.linspace(5.0, 5.6, tresp.shape[0])
    temps = 10 ** np.linspace(5.0, 5.6, tresp.shape[0] + 1)

    with pytest.raises(ValueError, match=r"dn_in must have shape"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, warn=False)


def test_dn2dem_accepts_gloci_filter_mask(monkeypatch):
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()
    captured = {}
    dn2dem_module = import_module("demregpy.dn2dem")

    def fake_demmap(dd, ed, rmatrix, logt, dlogt, glc, **kwargs):
        captured["glc"] = np.array(glc, copy=True)
        na = dd.shape[0]
        nt = logt.shape[0]
        nf = dd.shape[1]
        return (
            np.zeros((na, nt)),
            np.zeros((na, nt)),
            np.zeros((na, nt)),
            np.zeros(na),
            np.zeros((na, nf)),
        )

    monkeypatch.setattr(dn2dem_module, "demmap", fake_demmap)

    dn2dem(dn, edn, tresp, tresp_logt, temps, gloci=np.array([1, 0]), warn=False)

    np.testing.assert_array_equal(captured["glc"], np.array([1, 0]))


def test_dn2dem_accepts_4d_input_and_preserves_shape(monkeypatch):
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()
    dn2dem_module = import_module("demregpy.dn2dem")
    captured = {}
    dn_4d = np.broadcast_to(dn, (1, 2, 3, dn.shape[0]))
    edn_4d = np.broadcast_to(edn, dn_4d.shape)
    nt = len(temps) - 1
    dem_norm0 = np.ones((1, 2, 3, nt))

    def fake_demmap(dd, ed, rmatrix, logt, dlogt, glc, **kwargs):
        captured["dd_shape"] = dd.shape
        captured["ed_shape"] = ed.shape
        captured["dem_norm0_shape"] = kwargs["dem_norm0"].shape
        nobs = dd.shape[0]
        nf = dd.shape[1]
        nt_local = logt.shape[0]
        return (
            np.arange(nobs * nt_local, dtype=float).reshape(nobs, nt_local),
            np.ones((nobs, nt_local)),
            np.full((nobs, nt_local), 2.0),
            np.arange(nobs, dtype=float),
            np.arange(nobs * nf, dtype=float).reshape(nobs, nf),
        )

    monkeypatch.setattr(dn2dem_module, "demmap", fake_demmap)

    dem, edem, elogt, chisq, dn_reg = dn2dem(
        dn_4d,
        edn_4d,
        tresp,
        tresp_logt,
        temps,
        dem_norm0=dem_norm0,
        warn=False,
    )

    assert captured["dd_shape"] == (6, 2)
    assert captured["ed_shape"] == (6, 2)
    assert captured["dem_norm0_shape"] == (6, nt)
    assert dem.shape == (1, 2, 3, nt)
    assert edem.shape == (1, 2, 3, nt)
    assert elogt.shape == (1, 2, 3, nt)
    assert chisq.shape == (1, 2, 3)
    assert dn_reg.shape == (1, 2, 3, 2)


def test_dn2dem_rejects_bad_gloci_scalar():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()

    with pytest.raises(ValueError, match="gloci scalar must be 0 or 1"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, gloci=2, warn=False)


def test_dn2dem_rejects_bad_gloci_shape():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()

    with pytest.raises(ValueError, match=r"gloci array must have shape \(2,\)"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, gloci=np.array([1, 0, 1]), warn=False)


def test_dn2dem_rejects_bad_gloci_values():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()

    with pytest.raises(ValueError, match="gloci array must contain only 0/1 values"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, gloci=np.array([1, 2]), warn=False)


def test_dn2dem_rejects_bad_dem_norm0_shape():
    dn, edn, tresp, tresp_logt, temps = _basic_dn2dem_inputs()

    with pytest.raises(ValueError, match=r"dem_norm0 must have shape"):
        dn2dem(dn, edn, tresp, tresp_logt, temps, dem_norm0=np.ones((2, 7)), warn=False)
