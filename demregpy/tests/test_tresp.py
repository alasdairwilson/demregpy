import numpy as np

from demregpy.tresp import load_aia_response


def test_load_aia_response_returns_expected_shapes():
    channels, tresp_logt, trmatrix = load_aia_response()

    assert channels == ["A94", "A131", "A171", "A193", "A211", "A335"]
    assert tresp_logt.shape == (101,)
    assert trmatrix.shape == (101, 6)
    assert np.isfinite(tresp_logt).all()
    assert np.isfinite(trmatrix).all()
