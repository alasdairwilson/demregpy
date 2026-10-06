# demregpy TODO

Follow-ups enabled by the standard-form solver (SVD of the weighted response times L⁻¹, λ solved
exactly by the discrepancy principle). See `changelog/24.*.rst` for the change itself.

## Before merging the solver change

- [x] Regenerate the golden outputs: `test_dem_pix_golden_outputs`, `test_dem_pix_golden_outputs_high_nf`,
      `test_aia_synoptic_central_pixel_golden`, `test_synth_golden_outputs`. The old values captured the
      λ-grid discretisation; the new ones match the old code run with a 1000-point grid to ~0.2%.
- [x] `test_synth_dn_ratio_close`: now uses 10% noise to match its 10% errors, with tolerance [0.75, 1.25]
      (passes 99.9% of 1,800 random draws). The old [0.85, 1.10] passed only because the data were
      noiseless and the 50-point grid happened to stop short of χ² = 1.
- [x] Rename the `changelog/24.*.rst` fragments to the real PR number.
- [ ] Check the original IDL `dem_reg_map.pro` for the same `maxx = max(sigs)` bound and tell
      Hannah/Kontar if it's there too.

## 1. Proper positivity instead of over-smoothing (biggest quality win)

The positivity loop raises the χ² target ×`rgt_fact` until the DEM is non-negative, so narrow DEMs
get over-smoothed (χ² 8–38, quiet-Sun peaks ~2× too low at AIA resolution).

- [ ] Solve min ‖A x − d‖² + λ‖L x‖² subject to x ≥ 0 directly: NNLS on the stacked system
      [A; √λ L] x ≈ [d; 0] (`scipy.optimize.nnls`; nt ~ 20, so cheap).
- [ ] Choose λ for the constrained problem by root-finding χ²(λ) = target (monotonic in practice; check).
- [ ] Keep the old loop behind a flag for comparison, then compare on synthetic truth (narrow / two-peak / hot-tail
      DEMs) and real AIA pixels: χ², peak recovery, EM, runtime.
- [ ] Known case: `test_dem_pix_golden_outputs_high_nf` exhausts the loop at `max_iter=10` (χ² = 1.5⁹ ≈ 38.4)
      and the golden records 4 slightly negative tail bins (−0.45% of peak). The old grid code reached
      positivity there only by overshooting to a coarse grid point (χ² 46). NNLS should give a non-negative
      DEM near χ² 1; regenerate that golden afterwards.

## 2. Vectorise across pixels (biggest speed win)

Each pixel is now one small SVD plus a scalar root solve.

- [ ] Batched `np.linalg.svd` on a stack `(npix, nf, nt)`.
- [ ] Vectorised Newton for λ: misfit(λ) = Σ (λ/(sᵢ²+λ))² cᵢ² + const, with an analytic derivative; 3–5
      iterations for all pixels at once, with a bracketed fallback.
- [ ] Keep per-pixel L where weighting is self-normalised or EM-loci (L differs per pixel; still batchable).
- [ ] Benchmark against the current process pool on AIA maps (expect 10–100×; verify).

## 3. Alternative λ criteria

- [ ] Add `method="discrepancy" | "gcv" | "lcurve"` to `dn2dem`. All three are a few lines given the
      filter factors sᵢ²/(sᵢ²+λ).
- [ ] GCV doesn't trust the absolute errors, which helps when errors are mostly systematic guesses
      (e.g. SDO/EVE full-disk lines).

## 4. Full uncertainties

- [ ] Return (optionally) the full DEM covariance L⁻¹ V F Σ⁻¹ … and the resolution matrix in closed form,
      not just the diagonal `edem`. Needed for honest errors on summed quantities (e.g. EM above some T).
- [ ] Consider exposing ∂DEM/∂λ for sensitivity checks.

## Later / ideas

- [ ] Joint regularisation in time for time series (smoothness across frames as well as in T).
- [ ] A guard or warning when the selected λ sits at an end of its bracket.
