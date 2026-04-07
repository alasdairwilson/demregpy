import numpy as np

from demregpy.demmap import dem_reg_map


def test_dem_reg_map_handles_nmu_1():
    sigmaa = np.array([1.0, 0.5])
    sigmab = np.array([0.2, 0.3])
    U = np.eye(2)
    W = np.eye(2)
    data = np.array([1.0, 1.0])
    err = np.array([1.0, 1.0])

    lamb = dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak=1.0, nmu=1)
    assert np.isfinite(lamb)


def test_dem_reg_map_handles_zero_sigma_ratio():
    sigmaa = np.array([0.0, 1.0])
    sigmab = np.array([1.0, 1.0])
    U = np.eye(2)
    W = np.eye(2)
    data = np.array([1.0, 1.0])
    err = np.array([1.0, 1.0])

    lamb = dem_reg_map(sigmaa, sigmab, U, W, data, err, reg_tweak=1.0, nmu=10)
    assert np.isfinite(lamb)
