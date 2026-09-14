import numpy as np

from complex_dynamics.rational_map import RationalMap


class TestRationalMapDegree:
    """Tests checking if the degree is calculated correctly."""

    def test_degree(self):
        R = RationalMap(
            np.array([1, 0, 1], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )

        assert R.degree == 2


class TestRationalMapEvaluation:
    """Tests evaluating maps at finite points, poles, and infinity."""

    def test_identity_map(self):
        num = np.array([0, 1], dtype=np.complex128)
        denom = np.array([1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([0, 1, 2, -3 + 4j], dtype=np.complex128)
        expected = Z.copy()

        R(Z)

        assert np.allclose(Z, expected)

    def test_constant_map(self):
        num = np.array([5], dtype=np.complex128)
        denom = np.array([1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([0, 1, 100], dtype=np.complex128)

        R(Z)

        assert np.all(Z == 5)

    def test_reciprocal_map(self):
        num = np.array([1], dtype=np.complex128)
        denom = np.array([0, 1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([1, 2, 2 + 3j], dtype=np.complex128)
        expected = np.array([1, 1 / 2, 1 / (2 + 3j)], dtype=np.complex128)

        R(Z)

        assert np.allclose(Z, expected)

    def test_simple_polynomial_map(self):
        num = np.array([1, 0, 1], dtype=np.complex128)
        denom = np.array([1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([0, 1, 2j], dtype=np.complex128)

        R(Z)

        np.testing.assert_allclose(Z, [1, 2, -3])

    def test_pole_mapping_to_infinity(self):
        num = np.array([1], dtype=np.complex128)
        denom = np.array([0, 1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([0], dtype=np.complex128)

        R(Z)

        assert np.isinf(Z[0].real)

    def test_close_to_pole_mapping_to_infinity(self):
        num = np.array([1], dtype=np.complex128)
        denom = np.array([0, 0, 0, 1], dtype=np.complex128)
        R = RationalMap(num, denom)
        epsilon = 1e-250
        Z = np.array([0 + epsilon], dtype=np.complex128)

        R(Z)

        assert np.isinf(Z[0].real)

    def test_infinity_image_zero(self):
        num = np.array([1], dtype=np.complex128)
        denom = np.array([0, 1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([np.inf], dtype=np.complex128)

        R(Z)

        assert Z[0] == np.complex128(0, 0)

    def test_infinity_image_infinity(self):
        num = np.array([0, 0, 1], dtype=np.complex128)
        denom = np.array([1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([np.inf], dtype=np.complex128)

        R(Z)

        assert np.isinf(Z[0])

    def test_infinity_image_equal_degree(self):
        num = np.array([1, 2], dtype=np.complex128)
        denom = np.array([3, 4], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([np.inf], dtype=np.complex128)
        expected = np.array([0.5], dtype=np.complex128)

        R(Z)

        np.testing.assert_allclose(Z[0], expected)

    def test_large_numbers_precission(self):
        num = np.array([1, 1], dtype=np.complex128)
        denom = np.array([-1, 1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([1e250], dtype=np.complex128)
        expected = np.array([1], dtype=np.complex128)

        R(Z)

        assert np.allclose(Z, expected)

    def test_large_numbers_explode(self):
        num = np.array([0, 0, 0, 1], dtype=np.complex128)
        denom = np.array([1], dtype=np.complex128)
        R = RationalMap(num, denom)
        Z = np.array([1e100, 1e200], dtype=np.complex128)
        expected = np.array([1e300, np.inf], dtype=np.complex128)

        R(Z)

        assert np.allclose(Z, expected)


class TestSympyConversionsAndString:
    """Tests string representation and SymPy conversion helpers."""

    def test_str_formatting(self):
        # R(z) = (z^2 + 1) / (z - 1)
        num = np.array([1, 0, 1], np.complex128)
        denom = np.array([-1, 1], np.complex128)
        rmap = RationalMap(num, denom)

        latex_str = str(rmap)
        assert "z" in latex_str
        assert "frac" in latex_str


class TestFromLatex:
    """Tests parsing LaTeX strings into RationalMap instances."""

    def test_from_latex_basic(self):
        latex_str = r"\frac{z^2 + 1}{z - 1}"
        R = RationalMap.from_latex(latex_str)
        Z = np.array([2], np.complex128)

        # Verify evaluation on parsed object:
        # At z = 2: (2^2 + 1) / (2 - 1) = 5 / 1 = 5
        R(Z)
        np.testing.assert_allclose(Z, [5 + 0j])

    def test_from_latex_with_imaginary_unit(self):
        latex_str = r"\frac{z + i}{2}"
        R = RationalMap.from_latex(latex_str, imag_const="i")
        Z = np.array([0], dtype=np.complex128)

        # At z = 0: (0 + i) / 2 = 0.5i
        R(Z)
        np.testing.assert_allclose(Z, [0.5j])

    def test_from_latex_plain_polynomial(self):
        # Plain polynomial without fraction bar
        latex_str = r"z^2 + 2z + 1"
        R = RationalMap.from_latex(latex_str)
        Z = np.array([3], dtype=np.complex128)

        assert R.degree == 2
        # At z = 3: 3^2 + 2(3) + 1 = 16
        R(Z)
        np.testing.assert_allclose(Z, [16 + 0j])

    def test_from_latex_polynomial_with_imaginary_unit(self):
        # Polynomial with imaginary constant: z^2 + i
        latex_str = r"z^2 + i"
        R = RationalMap.from_latex(latex_str, imag_const="i")
        Z = np.array([1], dtype=np.complex128)

        # At z = 1: 1^2 + i = 1 + i
        R(Z)
        np.testing.assert_allclose(Z, [1 + 1j])
