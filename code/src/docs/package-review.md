# Package review

Review date: 2026-09-14. Scope: all source modules, both original test suites,
project metadata, dependencies, mathematical conventions, and rendering behavior.
The supplied workspace has no usable Git metadata, so no commit or tag was made.

## Findings and implemented fixes

| Priority | Original issue | Result |
| --- | --- | --- |
| Release blocker | Discovery pointed to missing `code/src`; README missing; placeholder description. | Explicit flat-layout discovery, README, meaningful metadata, MIT license chosen by owner, `complex-dynamics` distribution name. |
| Release blocker | Tests named `*_tests.py` were not found by standard pytest discovery. | Suites moved into `tests/test_*.py`, importlib collection configured. |
| High | Test calls used obsolete `num_iter`; one test expected orbit purity while implementation mutated. | Tests use `max_iterations`; numerical functions are pure by default. |
| High | `np.complex128(real, imag)` is not a valid scalar constructor. | Infinity and zero use valid scalar construction. |
| High | Evaluation changed caller arrays and converted invalid NaNs to infinity. | Separate outputs, explicit `out=`, NaN propagation and canonical poles. |
| High | Common factors and trailing zeros corrupted degree, roots, poles, and removable singularities. | Exact Gaussian-rational GCD at construction, trailing-zero trimming, zero-denominator validation. |
| High | Direct powers in reciprocal evaluation could overflow before a finite ratio formed. | Scaled Horner arithmetic retains mantissas and exponents separately until division. |
| High | Inverse projection lost finite points near the north pole through `1-Z` cancellation. | Northern-hemisphere inverse formula, unit-sphere validation, explicit invalid/pole handling. |
| High | Last convergence check could overwrite the first matching target. | Uniform initial/final checks with deterministic input-order ties. |
| High | `plot_divergence_times` overwrote data to set a color limit. | Explicit normalization limits, no input mutation. |
| High | NaN label comparison, absent basins, and zero-step normalization broke basin images. | NaN-aware masks, absent-basin handling, guarded normalization, palette sampling. |
| Medium | Boundary thickness result was discarded; dilation wrapped across edges. | Return value used, nonwrapping dilation, neighboring NaNs treated as one region. |
| Medium | Interactive calls opened blocking windows; import required PyVista even for 2-D. | Lazy PyVista imports, `show=False`, `off_screen` option; no private typing imports. |
| Medium | Single-point, repeated, and antipodal sphere orbits broke arc generation. | Explicit degenerate cases and deterministic great-circle choice for antipodes. |
| Medium | A convergence surface accepted axes but ignored them. | Supplied 3-D axes are respected and invalid axes rejected. |
| Medium | Notebook, compiler, test, and visualization packages were mandatory runtime dependencies. | NumPy/SymPy core; `plot`, `interactive`, and `latex` extras; development group; refreshed lockfile. |
| Medium | Incomplete annotations and prose-only docstrings. | Strict mypy configuration, concrete ndarray dtypes, NumPy-style public documentation, PEP 561 marker, Ruff enforcement. |

## API migration

Use `images = R(z)` to obtain images; use `R(z, out=z)` to mutate explicitly.
Calculator functions now preserve their input arrays. Call sites relying on their
side effects need to retain the returned orbit or evaluate explicitly. Scalar
inputs yield zero-dimensional arrays. Coefficient lists and real arrays are now
accepted and converted to complex128. `num_iter` is not a supported alias.

The chordal metric retains diameter 2, preserving existing tolerance semantics.
Invalid iteration counts are `-1`; unresolved convergence labels are NaN.
Interactive callers wishing to open a window must pass `show=True`. Scalar
plot functions still return figure/axes, with artists available through the axes.

## Numerical architecture and performance

`rational_map.py` owns polynomial validation, exact reduction, evaluation and
algebraic helpers. `calculator.py` owns geometry, meshes and finite-time dynamics.
`plot.py` renders precomputed arrays; it does not run dynamics. The small private
validation module keeps numerical parameter checks consistent.

Orbit storage is preallocated and costs O((n+1)N) complex numbers. Escape and
convergence iterate active points only. Convergence projects the active points
once per iteration and caches target coordinates, then checks targets in order:
O(kN) distance work with O(N+k) auxiliary storage, avoiding an N-by-k allocation.
The sequential iteration loop cannot be removed without changing the algorithm:
iteration n+1 depends on iteration n.

Scaled Horner evaluation costs O(dN). It uses float64 mantissas and integer
exponents, so results do not depend on whether `longdouble` has extra range;
[NumPy documents that this varies by platform](https://numpy.org/doc/stable/user/basics.types).
Exact common-factor reduction occurs once at construction, not during iteration.
For very high degrees, exact GCD cost can be substantial. Reuse map objects.

## Remaining limits and proposed core additions

1. **Structured iteration results.** Add counts, terminal values, and explicit
   converged/escaped/invalid/unresolved masks. The retained count-only escape API
   cannot distinguish an exit at the final step from a capped non-exit.
2. **Homogeneous projective iteration.** Retain normalized pairs [u:v] between
   iterations, including beyond float64 affine range. Current evaluation can
   form a finite ratio from enormous polynomials, but an unrepresentable final
   image still becomes infinity and loses information in later iterates.
3. **Fixed and periodic points with multipliers.** Solve P(z)-zQ(z)=0 with the
   infinity case handled in local coordinates; add primitive-period filtering
   and cycle multipliers before classifying attracting/repelling/neutral cycles.
4. **Critical points and spherical derivative.** Include critical poles and
   infinity, rather than only roots of the affine derivative. For degree d>=2,
   verify total critical multiplicity 2d-2. Critical orbits are a natural basis
   for parameter studies and basin exploration.
5. **Polynomial escape bounds and Julia approximations.** Add a documented
   sufficient bound per polynomial, strict escape semantics, backward iteration,
   and spherical derivative diagnostics. Do not label generic radius-exit
   boundaries as certified Julia sets.
6. **Conditioning diagnostics and higher precision.** Exact cancellation applies
   to supplied binary coefficients, not nearby intended symbolic coefficients.
   Polynomial evaluation near roots, repeated-root finding, and affine derivative
   coefficient arithmetic remain susceptible to cancellation and extreme scaling.
   NumPy roots are approximations, not certified roots. Offer an optional
   arbitrary-precision path and residual estimates when needed.
7. **Visualization topology and API consolidation.** Sphere meshes duplicate
   the azimuth seam and are not equal-area. Current boundary extraction uses
   rectangular adjacency, not seam wrapping or polar identification. Future
   sphere-specific boundary extraction should handle this topology. Unify the
   legacy `dynamic_plot_*` naming in a separately documented API transition.

`derivative()` is an affine derivative, not automatically a multiplier at
infinity. First entry into a target neighborhood does not certify a Fatou
component; radius exit does not certify divergence for a general rational map.
See [definitions and formulas](mathematics.md) and
[Beardon's publisher page](https://link.springer.com/book/9780387951515).

## Validation

Validation includes the migrated original suites plus regression tests for
poles, common factors, constant maps, derivatives, scalar/empty/strided/read-only
arrays, NaNs, extremes, metric broadcasting, stereographic round trips,
initial/final convergence, overlapping targets, input preservation, and plotting.
Interactive geometry is exercised in an isolated off-screen PyVista subprocess,
including a guard against implicit window display. This is an API/geometry test,
not a visual screenshot comparison.