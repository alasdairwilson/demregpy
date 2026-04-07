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

    with pytest.raises(ValueError, match="dem_norm0.*finite"):
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
