import numpy as np


def test_dn2dem_independent_of_response_units():
    # Scaling the response by s is only a change of units, so DEM * s and chi-squared
    # should come out the same for any s.
    from demregpy import dn2dem

    logt = np.arange(5.0, 7.51, 0.05)
    edges = 10.0 ** np.arange(5.6, 6.851, 0.05)
    peaks = np.linspace(5.85, 6.45, 8)
    resp = np.stack([np.exp(-0.5 * ((logt - p) / 0.12) ** 2) for p in peaks], axis=1)
    rng = np.random.default_rng(1)
    mid = np.log10(edges[:-1]) + 0.025
    dem_true = np.exp(-0.5 * ((mid - 6.15) / 0.15) ** 2)
    dt = 10 ** mid * np.log(10) * 0.05
    resp_mid = np.stack([np.interp(mid, logt, resp[:, i]) for i in range(len(peaks))], axis=1)
    dn = (dem_true * dt) @ resp_mid * (1 + 0.05 * rng.standard_normal(len(peaks)))

    results = []
    for scale in [1e-20, 1e-6, 1e6]:
        dem, _, _, chisq, _ = dn2dem(dn, 0.1 * dn, resp * scale, logt, edges)
        results.append((dem * scale, float(chisq)))
    for dem, chisq in results[1:]:
        assert np.allclose(dem, results[0][0], rtol=1e-3, atol=1e-6 * np.abs(results[0][0]).max())
        assert np.isclose(chisq, results[0][1], rtol=1e-3)

