import numpy as np

from demregpy.synthetic import synthesize_counts


def test_synthesize_counts_returns_expected_shapes():
    logt = np.linspace(5.7, 6.3, 7)
    tresp = np.ones((7, 3))
    dem = np.arange(7, dtype=float) + 1.0

    synthetic = synthesize_counts(dem, logt, tresp, error_fraction=0.2)

    assert synthetic.dem.shape == (7,)
    assert synthetic.dn_clean.shape == (3,)
    assert synthetic.dn_in.shape == (3,)
    assert synthetic.edn_in.shape == (3,)
    np.testing.assert_allclose(synthetic.edn_in, 0.2 * synthetic.dn_clean)
    np.testing.assert_allclose(synthetic.dn_in, synthetic.dn_clean)


def test_synthesize_counts_noise_is_reproducible():
    logt = np.linspace(5.7, 6.3, 7)
    tresp = np.ones((7, 3))
    dem = np.arange(14, dtype=float).reshape(2, 7) + 1.0

    synth1 = synthesize_counts(dem, logt, tresp, noise_fraction=0.05, random_state=0)
    synth2 = synthesize_counts(dem, logt, tresp, noise_fraction=0.05, random_state=0)

    np.testing.assert_allclose(synth1.dn_in, synth2.dn_in)
    np.testing.assert_allclose(synth1.edn_in, synth2.edn_in)
