import numpy as np
import pytest
from importlib import import_module

from demregpy.demmap import dem_pix, dem_unwrap, demmap


def _basic_inputs():
    rmatrix = np.array(
        [
            [1.0, 0.5],
            [0.9, 0.6],
            [0.8, 0.7],
            [0.7, 0.8],
            [0.6, 0.9],
            [0.5, 1.0],
        ]
    )
    logt = np.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    dlogt = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    glc = np.zeros(2)
    dnin = np.array([1.0, 1.0])
    ednin = np.array([1.0, 1.0])
    return dnin, ednin, rmatrix, logt, dlogt, glc


def test_dem_pix_rejects_scalar_dem_norm0():
    dnin, ednin, rmatrix, logt, dlogt, glc = _basic_inputs()
    with pytest.raises(ValueError):
        dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, dem_norm0=0)


def test_dem_unwrap_rejects_scalar_dem_norm0():
    dnin, ednin, rmatrix, logt, dlogt, glc = _basic_inputs()
    dn = np.vstack([dnin, dnin])
    ed = np.vstack([ednin, ednin])
    with pytest.raises(ValueError):
        dem_unwrap(dn, ed, rmatrix, logt, dlogt, glc, dem_norm0=0)


def test_demmap_rejects_scalar_dem_norm0():
    dnin, ednin, rmatrix, logt, dlogt, glc = _basic_inputs()
    dd = np.vstack([dnin, dnin])
    ed = np.vstack([ednin, ednin])
    with pytest.raises(ValueError):
        demmap(dd, ed, rmatrix, logt, dlogt, glc, dem_norm0=0)


class _StopAtGSVD(RuntimeError):
    pass


def _smoothed_gloci_weight(dnin, rmatrix, glc):
    gdglc = np.nonzero(glc > 0)[0]
    emloci = np.zeros((rmatrix.shape[0], gdglc.shape[0]))
    for ee, idx in enumerate(gdglc):
        emloci[:, ee] = dnin[idx] / rmatrix[:, idx]

    dem_model = np.zeros(rmatrix.shape[0])
    for ttt in range(rmatrix.shape[0]):
        nz = np.nonzero(emloci[ttt, :])[0]
        dem_model[ttt] = np.min(emloci[ttt, nz]) if nz.size > 0 else 0.0

    dem_reg_lwght = np.convolve(dem_model[1:-1], np.ones(5) / 5.0)[1:-1] / np.max(dem_model)
    dem_reg_lwght[dem_reg_lwght <= 1e-8] = 1e-8
    return dem_reg_lwght


def test_dem_pix_uses_gloci_weighting_from_selected_filters(monkeypatch):
    dnin = np.array([10.0, 1.0, 12.0])
    ednin = np.ones(3)
    rmatrix = np.array(
        [
            [2.0, 10.0, 2.0],
            [2.5, 10.0, 2.4],
            [3.0, 10.0, 3.0],
            [4.0, 10.0, 4.0],
            [5.0, 10.0, 6.0],
            [6.0, 10.0, 12.0],
        ]
    )
    logt = np.linspace(5.0, 5.5, 6)
    dlogt = np.full(6, 0.1)
    glc = np.array([1, 0, 1])
    captured = []
    demmap_module = import_module("demregpy.demmap")

    def fake_dem_inv_gsvd_diag(A, bdiag):
        captured.append(np.array(bdiag, copy=True))
        raise _StopAtGSVD

    monkeypatch.setattr(demmap_module, "dem_inv_gsvd_diag", fake_dem_inv_gsvd_diag)

    with pytest.raises(_StopAtGSVD):
        dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, dem_norm0=None, warn=False)

    expected_weight = _smoothed_gloci_weight(dnin, rmatrix, glc)
    expected_ldiag = np.sqrt(dlogt) / np.sqrt(expected_weight)
    np.testing.assert_allclose(captured[0], expected_ldiag, rtol=1e-12, atol=0.0)


def test_dem_pix_prefers_user_dem_norm0_over_gloci(monkeypatch):
    dnin = np.array([10.0, 1.0, 12.0])
    ednin = np.ones(3)
    rmatrix = np.array(
        [
            [2.0, 10.0, 2.0],
            [2.5, 10.0, 2.4],
            [3.0, 10.0, 3.0],
            [4.0, 10.0, 4.0],
            [5.0, 10.0, 6.0],
            [6.0, 10.0, 12.0],
        ]
    )
    logt = np.linspace(5.0, 5.5, 6)
    dlogt = np.full(6, 0.1)
    glc = np.ones(3)
    dem_norm0 = np.array([2.0, 3.0, 4.0, 2.0, 3.0, 4.0])
    captured = []
    demmap_module = import_module("demregpy.demmap")

    def fake_dem_inv_gsvd_diag(A, bdiag):
        captured.append(np.array(bdiag, copy=True))
        raise _StopAtGSVD

    monkeypatch.setattr(demmap_module, "dem_inv_gsvd_diag", fake_dem_inv_gsvd_diag)

    with pytest.raises(_StopAtGSVD):
        dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, dem_norm0=dem_norm0, warn=False)

    expected_ldiag = np.sqrt(dlogt) / np.sqrt(dem_norm0)
    np.testing.assert_allclose(captured[0], expected_ldiag, rtol=1e-12, atol=0.0)


def test_dem_pix_honors_custom_dem_norm0_when_product_is_one():
    rmatrix = np.array(
        [
            [1.0, 0.5, 0.2],
            [0.9, 0.6, 0.3],
            [0.8, 0.7, 0.4],
            [0.7, 0.8, 0.5],
            [0.6, 0.9, 0.6],
            [0.5, 1.0, 0.7],
        ]
    )
    logt = np.linspace(5.0, 5.5, 6)
    dlogt = np.full(6, 0.1)
    glc = np.zeros(3)
    dnin = np.array([1.0, 2.0, 1.5])
    ednin = np.array([0.1, 0.2, 0.15])
    dem_norm0 = np.array([2.0, 0.5, 1.0, 1.0, 1.0, 1.0])

    dem_default = dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, dem_norm0=None, warn=False)[0]
    dem_weighted = dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc, dem_norm0=dem_norm0, warn=False)[0]

    assert not np.allclose(dem_default, dem_weighted, rtol=0.0, atol=1e-12)
