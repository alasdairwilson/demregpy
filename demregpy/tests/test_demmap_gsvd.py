import numpy as np

from demregpy.demmap import dem_inv_gsvd, dem_pix


def test_dem_inv_gsvd_handles_singular_b():
    # Typical usage: A is (nf, nt) with nf < nt, B is (nt, nt)
    A = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    )
    B = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    alpha, beta, U, V, W = dem_inv_gsvd(A, B)
    assert np.isfinite(alpha).all()
    assert np.isfinite(beta).all()
    assert np.isfinite(U).all()
    assert np.isfinite(V).all()
    assert np.isfinite(W).all()
    assert U.shape[0] == V.shape[0] == W.shape[1]


def test_dem_pix_accepts_zero_dn():
    nt = 6
    nf = 3
    rmatrix = np.ones((nt, nf))
    logt = np.linspace(1.0, 2.0, nt)
    dlogt = np.full(nt, 0.1)
    glc = np.ones(nf)
    dnin = np.array([0.0, 1.0, 2.0])
    ednin = np.ones(nf)

    dem, edem, elogt, chisq, dn_reg = dem_pix(dnin, ednin, rmatrix, logt, dlogt, glc)
    assert np.isfinite(dem).all()
    assert np.isfinite(edem).all()
    assert np.isfinite(elogt).all()
    assert np.isfinite(chisq)
    assert np.isfinite(dn_reg).all()
