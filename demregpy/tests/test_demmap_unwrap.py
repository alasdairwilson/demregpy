"""Correctness tests for :func:`demmap` and :func:`dem_unwrap` with small synthetic stacks."""

import numpy as np
import pytest

from demregpy.demmap import dem_unwrap, demmap


def _make_stack(nobs=5):
    """Build a small synthetic stack with *nobs* observations.

    Returns the pre-processed arrays that ``demmap`` / ``dem_unwrap`` expect
    (i.e. after the response matrix has already been interpolated, scaled,
    and multiplied by dlogT — the same pre-processing that ``dn2dem`` does).
    """
    # Temperature grid — 7 bins, 8 edges
    logtemps = np.linspace(5.7, 6.3, 8)
    temps = 10**logtemps
    dlogt = np.diff(logtemps)
    nt = len(dlogt)
    logt = logtemps[:-1] + 0.5 * dlogt

    # 6 synthetic Gaussian response curves
    nf = 6
    centers = np.array([5.75, 5.85, 5.95, 6.05, 6.15, 6.25])
    sigma = 0.08
    raw_tr = np.zeros((nt, nf))
    for i, c in enumerate(centers):
        raw_tr[:, i] = np.exp(-((logt - c) ** 2) / (2 * sigma**2))

    # Apply the same dlogT and scaling as dn2dem
    dlogTfac = 10.0**logt * np.log(10.0**dlogt)
    sclf = 1e15
    rmatrix = np.zeros((nt, nf))
    for i in range(nf):
        rmatrix[:, i] = raw_tr[:, i] * dlogTfac * sclf

    # Synthetic DEM — single Gaussian peak varied slightly per observation
    root2pi = (2.0 * np.pi) ** 0.5
    rng = np.random.default_rng(42)
    dem_models = np.zeros((nobs, nt))
    peak_centers = 5.95 + 0.05 * rng.standard_normal(nobs)
    for j in range(nobs):
        s = 0.12
        d = 4e22
        dem_models[j] = (d / (root2pi * s)) * np.exp(
            -((logt - peak_centers[j]) ** 2) / (2 * s**2)
        )

    # Forward-model counts: dn = rmatrix^T @ dem * dlogt  (approx)
    # Use the *raw* (unscaled) response for the forward model, then add noise
    dn_clean = np.zeros((nobs, nf))
    for j in range(nobs):
        for i in range(nf):
            dn_clean[j, i] = np.sum(raw_tr[:, i] * dem_models[j] * dlogTfac)
    edn = 0.1 * dn_clean + 1.0
    noise = rng.normal(0, 1, dn_clean.shape) * edn * 0.05
    dn_in = dn_clean + noise
    # Ensure non-negative
    dn_in = np.clip(dn_in, 0.1, None)

    glc = np.zeros(nf, dtype=int)  # self-normalised weighting

    return dn_in, edn, rmatrix, logt, dlogt, glc, dem_models


class TestDemmapCorrectness:
    """demmap should recover DEMs that reconstruct the input counts."""

    def test_output_shapes(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=5)
        dem, edem, elogt, chisq, dn_reg = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        nobs, nf = dn_in.shape
        nt = logt.shape[0]
        assert dem.shape == (nobs, nt)
        assert edem.shape == (nobs, nt)
        assert elogt.shape == (nobs, nt)
        assert chisq.shape == (nobs,)
        assert dn_reg.shape == (nobs, nf)

    def test_reconstruction_quality(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=5)
        dem, edem, elogt, chisq, dn_reg = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        ratio = dn_reg / dn_in
        assert np.all(ratio > 0.5), f"Reconstruction too low: min ratio {ratio.min():.3f}"
        assert np.all(ratio < 2.0), f"Reconstruction too high: max ratio {ratio.max():.3f}"

    def test_chisq_reasonable(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=5)
        _, _, _, chisq, _ = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        assert np.all(chisq > 0), "chi-squared must be positive"
        assert np.all(chisq < 10), f"chi-squared unreasonably large: max {chisq.max():.1f}"

    def test_dem_non_negative(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=5)
        dem, _, _, _, _ = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        assert np.all(dem >= 0), "Default demmap should return non-negative DEMs"

    def test_single_observation(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=1)
        dem, edem, elogt, chisq, dn_reg = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        assert dem.shape == (1, logt.shape[0])
        assert np.isfinite(dem).all()
        assert np.isfinite(chisq).all()


class TestDemUnwrapCorrectness:
    """dem_unwrap should produce the same results as demmap for small stacks."""

    def test_matches_demmap(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=3)
        kwargs = dict(
            reg_tweak=1.0, max_iter=10, rgt_fact=1.5,
            dem_norm0=None, nmu=42, warn=False, l_emd=False,
        )
        dem_m, edem_m, elogt_m, chisq_m, dnreg_m = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, **kwargs,
        )
        dem_u, edem_u, elogt_u, chisq_u, dnreg_u = dem_unwrap(
            dn_in, edn, rmatrix, logt, dlogt, glc, **kwargs,
        )
        np.testing.assert_allclose(dem_m, dem_u, rtol=1e-10)
        np.testing.assert_allclose(edem_m, edem_u, rtol=1e-10)
        np.testing.assert_allclose(elogt_m, elogt_u, rtol=1e-10)
        np.testing.assert_allclose(chisq_m, chisq_u, rtol=1e-10)
        np.testing.assert_allclose(dnreg_m, dnreg_u, rtol=1e-10)

    def test_output_shapes(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=4)
        dem, edem, elogt, chisq, dn_reg = dem_unwrap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        nobs, nf = dn_in.shape
        nt = logt.shape[0]
        assert dem.shape == (nobs, nt)
        assert chisq.shape == (nobs,)
        assert dn_reg.shape == (nobs, nf)

    def test_reconstruction_quality(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, _ = _make_stack(nobs=4)
        _, _, _, _, dn_reg = dem_unwrap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        ratio = dn_reg / dn_in
        assert np.all(ratio > 0.5), f"Reconstruction too low: min ratio {ratio.min():.3f}"
        assert np.all(ratio < 2.0), f"Reconstruction too high: max ratio {ratio.max():.3f}"


class TestDemmapWithGloci:
    """Test demmap with EM loci weighting enabled."""

    def test_gloci_all_filters(self):
        dn_in, edn, rmatrix, logt, dlogt, _, _ = _make_stack(nobs=3)
        glc = np.ones(dn_in.shape[1], dtype=int)
        dem, edem, elogt, chisq, dn_reg = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        assert np.all(np.isfinite(dem))
        assert np.all(chisq > 0)

    def test_gloci_subset(self):
        dn_in, edn, rmatrix, logt, dlogt, _, _ = _make_stack(nobs=3)
        nf = dn_in.shape[1]
        glc = np.zeros(nf, dtype=int)
        glc[:3] = 1  # use only first 3 filters for EM loci
        dem, edem, elogt, chisq, dn_reg = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc, nmu=42, warn=False,
        )
        assert np.all(np.isfinite(dem))
        assert np.all(chisq > 0)


class TestDemmapWithDemNorm0:
    """Test demmap with a user-supplied initial DEM shape."""

    def test_custom_dem_norm0(self):
        dn_in, edn, rmatrix, logt, dlogt, glc, dem_models = _make_stack(nobs=3)
        nobs = dn_in.shape[0]
        nt = logt.shape[0]
        # Use the true DEM shape as the initial guess
        dem_norm0 = dem_models / dem_models.max(axis=1, keepdims=True)
        dem, edem, elogt, chisq, dn_reg = demmap(
            dn_in, edn, rmatrix, logt, dlogt, glc,
            dem_norm0=dem_norm0, nmu=42, warn=False,
        )
        assert dem.shape == (nobs, nt)
        assert np.all(np.isfinite(dem))
        ratio = dn_reg / dn_in
        assert np.all(ratio > 0.5)
        assert np.all(ratio < 2.0)
