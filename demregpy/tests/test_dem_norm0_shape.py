import numpy as np
import pytest

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
