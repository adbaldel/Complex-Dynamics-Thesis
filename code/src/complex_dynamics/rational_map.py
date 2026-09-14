import numpy as np
import sympy as sp
import sympy.parsing.latex as sp_pl
from numpy.polynomial import Polynomial
from numpy.typing import NDArray

infinity = np.complex128(np.inf, 0.0)


def _polynomial_to_sympy(polynomial: Polynomial, variable: str) -> sp.Poly:
    """
    Returns a SymPy expression of the given Polynomial.
    """
    expression = sp.Integer(0)

    for power, coefficient in enumerate(polynomial.coef):
        if np.iscomplexobj(coefficient) or isinstance(coefficient, complex):
            real_part = (
                int(coefficient.real)
                if float(coefficient.real).is_integer()
                else coefficient.real
            )
            imaginary_part = (
                int(coefficient.imag)
                if float(coefficient.imag).is_integer()
                else coefficient.imag
            )
            sympy_coefficient = (
                real_part
                if imaginary_part == 0
                else real_part + imaginary_part * sp.I
            )
        else:
            sympy_coefficient = (
                int(coefficient)
                if float(coefficient).is_integer()
                else float(coefficient)
            )

        expression += sp.sympify(sympy_coefficient) * (
            sp.Symbol(variable) ** power
        )

    return expression


def _sympy_to_coeffs(
    expression: sp.Expr, variable: str
) -> NDArray[np.complex128]:
    """
    Returns a Polynomial given a SymPy expression.
    """
    symbolic_polynomial = sp.Poly(expression, sp.Symbol(variable))
    coeffs_ascending = symbolic_polynomial.all_coeffs()[::-1]
    return np.asarray(coeffs_ascending, dtype=np.complex128)


def _sympy_to_polynomial(sym_expr: sp.Expr, var: str) -> Polynomial:
    """
    Returns a Polynomial given a SymPy expression.
    """
    return Polynomial(_sympy_to_coeffs(sym_expr, variable=var), symbol=var)


class RationalMap:
    """
    Represents a complex variable rational map.

    Uses the numpy.Polynomial class for representing the polynomials that define
    the rational map. That is, if the rational map is R = P / Q, P and Q will be
    numpy.polynomial.Polynomial.
    """

    def __init__(
        self,
        numerator_coefficients: NDArray[np.complex128],
        denominator_coefficients: NDArray[np.complex128],
        variable: str = "z",
    ) -> None:
        """
        Initializes a rational map with the given coefficients for its numerator
        and denominator.

        num_coef and denom_coef are lists of complex numbers that represent the
        coefficients of the numerator and denominator polynomials respectively.
        The coefficients in the lists have to be ordered from lowest order
        coefficient to highest.
        """
        if numerator_coefficients.dtype != np.complex128:
            raise TypeError(
                f"Z must have dtype=complex128, got {numerator_coefficients.dtype}. "
                f"Use np.array(..., dtype=np.complex128)."
            )
        if denominator_coefficients.dtype != np.complex128:
            raise TypeError(
                f"Z must have dtype=complex128, got {denominator_coefficients.dtype}. "
                f"Use np.array(..., dtype=np.complex128)."
            )

        self._variable = variable
        self._numerator = Polynomial(
            numerator_coefficients, symbol=self._variable
        )
        self._denominator = Polynomial(
            denominator_coefficients, symbol=self._variable
        )
        self._reverse_numerator = Polynomial(
            numerator_coefficients[::-1], symbol=self._variable
        )
        self._reverse_denominator = Polynomial(
            denominator_coefficients[::-1], symbol=self._variable
        )

        self._numerator_degree = self._numerator.degree()
        self._denominator_degree = self._denominator.degree()
        self._degree_difference = (
            self._numerator_degree - self._denominator_degree
        )
        self._degree = max(self._numerator_degree, self._denominator_degree)

        if self._numerator_degree < self._denominator_degree:
            self._infty_image = np.complex128(0, 0)
        elif self._numerator_degree > self._denominator_degree:
            self._infty_image = infinity
        else:
            self._infty_image = np.complex128(
                numerator_coefficients[-1] / denominator_coefficients[-1]
            )

    def __call__(self, Z: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """
        Evaluates the rational map on the given complex number array. Modifies
        the array itself.
        """
        if Z.dtype != np.complex128:
            raise TypeError(
                f"Z must have dtype=complex128, got {Z.dtype}. "
                f"Use np.array(..., dtype=np.complex128)."
            )

        # 1. Identify where the input is infinite
        is_input_inf = np.isinf(np.real(Z)) | np.isinf(np.imag(Z))
        # 2. Identify where the input is finite and small or big
        is_input_small = np.abs(Z) < 1
        is_input_big = (~is_input_small) & (~is_input_inf)

        # 3. Evaluate normally for small inputs
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            Z_small = Z[is_input_small]
            numerator = self._numerator(Z_small)
            denominator = self._denominator(Z_small)
            Z[is_input_small] = np.where(
                denominator != 0, numerator / denominator, infinity
            )

        # 4. Evaluate on w=1/z for big inputs
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            Z_big = Z[is_input_big]
            W = 1 / Z_big
            numerator = self._reverse_numerator(W)
            denominator = self._reverse_denominator(W)
            if self._degree_difference > 0:
                Z[is_input_big] = np.where(
                    denominator != 0,
                    np.power(Z_big, self._degree_difference)
                    * numerator
                    / denominator,
                    infinity,
                )
            elif self._degree_difference < 0:
                Z[is_input_big] = np.where(
                    denominator != 0,
                    numerator
                    / (np.power(Z_big, -self._degree_difference) * denominator),
                    infinity,
                )
            else:
                Z[is_input_big] = np.where(
                    denominator != 0,
                    numerator / denominator,
                    infinity,
                )

        # 5. Clean up outputs (Poles mapping to complex infinity) If the
        # calculation overflowed to inf/nan
        is_output_inf = (
            np.isinf(np.real(Z)) | np.isinf(np.imag(Z)) | np.isnan(np.real(Z))
        )
        Z[is_output_inf & ~is_input_inf] = infinity

        # 6. Evaluate infinite inputs
        Z[is_input_inf] = self._infty_image

        return Z

    def __str__(self) -> str:
        """
        Returns a LaTeX string representation of a rational map.
        """
        # 1. Convert both NumPy polynomials back to SymPy expressions
        num_expr = _polynomial_to_sympy(
            self._numerator, variable=self._variable
        )
        den_expr = _polynomial_to_sympy(
            self._denominator, variable=self._variable
        )

        # 2. Divide them to create a rational SymPy expression, then format to
        #    LaTeX
        rational_expr = num_expr / den_expr
        return sp.latex(rational_expr)

    @classmethod
    def from_latex(cls, latex_str: str, var: str = "z", imag_const: str = "i"):
        """
        Initializes a rational map (numerator, denominator) from a LaTeX string.
        """
        # 1. Parse LaTeX and handle imaginary constant
        expr = sp_pl.parse_latex(latex_str)
        i_sym = sp.Symbol(imag_const)
        expr = expr.subs(i_sym, sp.I)

        # 2. Split the expression into numerator and denominator For example,
        # \frac{z^2 + 1}{z - 1} becomes (z**2 + 1) and (z - 1)
        numer_expr, denom_expr = expr.as_numer_denom()
        num_coeffs = _sympy_to_coeffs(numer_expr, variable=var)
        denom_coeffs = _sympy_to_coeffs(denom_expr, variable=var)

        return cls(num_coeffs, denom_coeffs)

    @property
    def variable(self) -> str:
        """
        Returns the symbol used for representing the variable of the rational
        map.
        """
        return self._variable

    @property
    def numerator(self) -> Polynomial:
        """
        Returns the numerator polynomial of the rational map.
        """
        return self._numerator.copy()

    @property
    def denominator(self) -> Polynomial:
        """
        Returns the denominator polynomial of the rational map.
        """
        return self._denominator.copy()

    @property
    def degree(self) -> int:
        """
        Returns the degree of the rational map.
        """
        return self._degree

    def derivative(self) -> "RationalMap":
        """
        Returns the derivative of the rational map as a new RationalMap.

        If R = P / Q, the derivative is computed via the quotient rule: R' =
        (P'Q - PQ') / Q^2.
        """
        numerator_derivative = self._numerator.deriv()
        denominator_derivative = self._denominator.deriv()

        new_numerator = (
            numerator_derivative * self._denominator
            - self._numerator * denominator_derivative
        )
        new_denominator = self._denominator * self._denominator

        numerator_coefficients = new_numerator.coef.astype(np.complex128)
        denominator_coefficients = new_denominator.coef.astype(np.complex128)

        return RationalMap(
            numerator_coefficients,
            denominator_coefficients,
            variable=self._variable,
        )

    def poles(self, include_infinity: bool = True) -> NDArray[np.complex128]:
        """
        Returns the poles of the rational map as an array of complex numbers. It
        assumes that numerator and denominator have no common roots.

        The finite poles are the roots of the denominator polynomial, each
        appearing as many times as its multiplicity. If the numerator's degree
        is greater than the denominator's degree, infinity is also a pole of the
        map; when include_infinity is True (the default), it is appended to the
        returned array once for each order of the pole (i.e. the degree
        difference between numerator and denominator).
        """
        finite_poles = self._denominator.roots().astype(np.complex128)

        if include_infinity and self._degree_difference > 0:
            infinite_poles = np.full(
                self._degree_difference, infinity, dtype=np.complex128
            )
            return np.concatenate((finite_poles, infinite_poles))

        return finite_poles

    def roots(self, include_infinity: bool = True) -> NDArray[np.complex128]:
        """
        Returns the roots (zeros) of the rational map as an array of complex
        numbers. It assumes that numerator and denominator have no common roots.

        The finite roots are the roots of the numerator polynomial, each
        appearing as many times as its multiplicity. If the denominator's degree
        is greater than the numerator's degree, infinity is also a root of the
        map; when include_infinity is True (the default), it is appended to the
        returned array once for each order of the root (i.e. the degree
        difference between denominator and numerator).
        """
        finite_roots = self._numerator.roots().astype(np.complex128)

        if include_infinity and self._degree_difference < 0:
            infinite_roots = np.full(
                -self._degree_difference, infinity, dtype=np.complex128
            )
            return np.concatenate((finite_roots, infinite_roots))

        return finite_roots
