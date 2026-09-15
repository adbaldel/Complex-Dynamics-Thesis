import numpy as np
import pytest

from complex_dynamics import (
    RationalMap,
    calculate_convergence_times,
    calculate_divergence_times,
    calculate_forward_orbit,
    chordal_distance,
    complex_to_sphere,
    sphere_to_complex,
    spherical_distance,
)


@pytest.mark.parametrize(
    "p,q",
    [
        ([], [1]),
        ([1], []),
        ([[1]], [1]),
        ([1], [0, 0]),
        ([np.nan], [1]),
        ([1], [np.inf]),
    ],
)
def test_invalid_coefficients(p, q):
    with pytest.raises(ValueError):
        RationalMap(p, q)


def test_reduction_trimming_and_removable_singularities():
    r = RationalMap([-1, 0, 1, 0], [-1, 1, 0])
    assert r.degree == 1
    np.testing.assert_allclose(r([1, -1, 2]), [2, 0, 3])
    np.testing.assert_array_equal(r.poles(), [np.inf])
    np.testing.assert_allclose(r.roots(), [-1])
    r.numerator.coef[:] = 0
    assert r(2) == 3


def test_near_common_factors_are_not_cancelled():
    r = RationalMap([-1, 1], [-1 + 1e-12, 1])
    assert r.degree == 1
    assert r(1) == 0


def test_zero_map_and_constant_derivative():
    r = RationalMap([0, 0], [0, 1])
    assert r.degree == 0
    np.testing.assert_array_equal(r([0, 1, np.inf]), [0, 0, 0])
    assert r.poles().size == 0
    with pytest.raises(ValueError):
        r.roots()
    assert RationalMap([3], [1]).derivative().degree == 0


def test_pure_and_explicit_inplace_evaluation():
    r = RationalMap([0, 2], [1])
    original = np.array([1, 2, 3, 4], dtype=np.complex128)
    view = original[::2]
    result = r(view)
    np.testing.assert_array_equal(original, [1, 2, 3, 4])
    np.testing.assert_array_equal(result, [2, 6])
    assert r(view, out=view) is view
    np.testing.assert_array_equal(original, [2, 2, 6, 4])
    original.flags.writeable = False
    np.testing.assert_array_equal(r(original), [4, 4, 12, 8])
    with pytest.raises(ValueError):
        r(original, out=original)
    with pytest.raises(ValueError):
        r([1], out=np.zeros(1))


def test_scalar_and_empty_shapes():
    r = RationalMap([1], [0, 1])
    assert r(2).shape == ()
    assert r(np.zeros((0, 2))).shape == (0, 2)
    assert calculate_forward_orbit(2, r, 2).shape == (3,)
    assert calculate_divergence_times(2, r, 2).shape == ()
    assert calculate_convergence_times(2, r, [0], 2)[0].shape == ()


def test_nan_precedes_infinity():
    values = np.array(
        [complex(np.nan, np.inf), complex(np.inf, np.nan), np.nan, np.inf]
    )
    result = RationalMap([1], [0, 1])(values)
    assert np.isnan(result[:3]).all()
    assert result[-1] == 0
    for coord in complex_to_sphere(values):
        assert np.isnan(coord[:3]).all()


@pytest.mark.parametrize(
    "p,q,point,expected",
    [
        ([0, 0, 1e-300], [1], 1e200, 1e100),
        ([1e300], [0, 0, 1], 1e200, 1e-100),
        ([1e308], [1e308], 1e308, 1),
        ([1], [0, 1], complex(1e308, 1e308), complex(5e-309, -5e-309)),
        ([1], [0, 0, 1], 1e150, 1e-300),
    ],
)
def test_extreme_representable_values(p, q, point, expected):
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result = RationalMap(p, q)(point)
    np.testing.assert_allclose(result, expected, rtol=2e-14, atol=0)


def test_derivative_against_closed_form():
    derivative = RationalMap([1, 0, 1], [0, 2]).derivative()
    z = np.array([1, 2, 3j])
    np.testing.assert_allclose(derivative(z), (1 - 1 / z**2) / 2)
    assert np.isinf(derivative(0))


def test_latex_custom_variable():
    pytest.importorskip("antlr4")
    r = RationalMap.from_latex(r"\frac{w+i}{2}", var="w")
    assert r.variable == "w"
    assert r(1) == 0.5 + 0.5j


@pytest.mark.parametrize(
    "z", [0, 1, 1j, -2 + 3j, 1e150 + 1e150j, 1e308, 1e-300, np.inf]
)
def test_sphere_round_trip(z):
    coordinates = complex_to_sphere(z)
    np.testing.assert_allclose(sum(x * x for x in coordinates), 1)
    result = sphere_to_complex(*coordinates)
    if np.isinf(z):
        assert np.isinf(result)
    else:
        np.testing.assert_allclose(result, z, rtol=2e-14, atol=0)


def test_sphere_validation_and_broadcasting():
    with pytest.raises(ValueError):
        sphere_to_complex(0, 0, 0)
    with pytest.raises(ValueError):
        sphere_to_complex(np.inf, 0, 1)
    assert np.isnan(sphere_to_complex(np.nan, 0, 1))
    assert chordal_distance(np.array([0, 1])[:, None], [0, 1, np.inf]).shape == (2, 3)


def test_metrics_random_formula_and_antipodes():
    rng = np.random.default_rng(42)
    z = rng.normal(size=100) + 1j * rng.normal(size=100)
    w = rng.normal(size=100) + 1j * rng.normal(size=100)
    expected = 2 * np.abs(z - w) / np.sqrt((1 + np.abs(z) ** 2) * (1 + np.abs(w) ** 2))
    np.testing.assert_allclose(chordal_distance(z, w), expected)
    antipodes = -1 / z.conj()
    np.testing.assert_allclose(spherical_distance(z, antipodes), np.pi)
    np.testing.assert_allclose(chordal_distance(1e308, -1e308), 4e-308, atol=0)


@pytest.mark.parametrize("iterations", [-1, 1.5, True])
def test_invalid_iteration_counts(iterations):
    r = RationalMap([0, 1], [1])
    for fn, args in [
        (calculate_forward_orbit, ()),
        (calculate_divergence_times, ()),
        (calculate_convergence_times, ([0],)),
    ]:
        with pytest.raises(ValueError):
            fn([0], r, *args, max_iterations=iterations)


def test_initial_final_and_overlapping_convergence():
    r = RationalMap([0, 0.5], [1])
    z = np.array([0, 2, np.nan], dtype=np.complex128)
    original = z.copy()
    times, targets = calculate_convergence_times(z, r, [1, 1.001, 0], 1, tolerance=0.01)
    np.testing.assert_array_equal(times, [0, 1, -1])
    np.testing.assert_allclose(targets, [0, 1, np.nan])
    np.testing.assert_array_equal(z, original)
    times, targets = calculate_convergence_times([1], r, [1, 1.001], 0, tolerance=0.01)
    np.testing.assert_array_equal(times, [0])
    np.testing.assert_array_equal(targets, [1])
    times, targets = calculate_convergence_times([2], r, [], 2)
    assert times[0] == 2 and np.isnan(targets[0])


def test_escape_boundary_and_input_preservation():
    r = RationalMap([0, 2], [1])
    z = np.array([2, 1, 0, np.nan], dtype=np.complex128)
    original = z.copy()
    np.testing.assert_array_equal(calculate_divergence_times(z, r, 2), [0, 1, 2, -1])
    np.testing.assert_array_equal(z, original)
    np.testing.assert_array_equal(calculate_divergence_times(z, r, 0), [0, 0, 0, -1])


def test_infinity_orbit_and_convergence():
    r = RationalMap([1], [0, 1])
    np.testing.assert_array_equal(
        calculate_forward_orbit(0, r, 3), [0, np.inf, 0, np.inf]
    )
    times, targets = calculate_convergence_times([0], r, [np.inf], 1)
    assert times[0] == 1 and np.isinf(targets[0])
