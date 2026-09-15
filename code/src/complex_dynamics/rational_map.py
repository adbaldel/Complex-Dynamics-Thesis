"""Rational maps on the Riemann sphere, with ascending polynomial coefficients."""

from __future__ import annotations

import numpy as np
import sympy as sp
from numpy.polynomial import Polynomial
from numpy.typing import ArrayLike, NDArray

infinity = np.complex128(complex(np.inf, 0))


def _coefficients(values: ArrayLike) -> NDArray[np.complex128]:
    """Validate and trim polynomial coefficients without changing the input."""
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(
            "Coefficients must be a nonempty, finite, one-dimensional array."
        )
    nonzero = np.flatnonzero(result)
    return result[: int(nonzero[-1]) + 1].copy() if nonzero.size else result[:1].copy()


def _expression(coefficients: NDArray[np.complex128], variable: sp.Symbol) -> sp.Expr:
    """Convert binary floating-point coefficients to exact Gaussian rationals."""
    return sum(
        (sp.Rational(float(c.real)) + sp.I * sp.Rational(float(c.imag))) * variable**i
        for i, c in enumerate(coefficients)
    )


def _sympy_to_coeffs(expression: sp.Expr, variable: str) -> NDArray[np.complex128]:
    """Convert a univariate symbolic polynomial to ascending coefficients."""
    return np.asarray(
        sp.Poly(expression, sp.Symbol(variable)).all_coeffs()[::-1], dtype=np.complex128
    )


def _ldexp_complex(
    values: NDArray[np.complex128], exponent: NDArray[np.int64]
) -> NDArray[np.complex128]:
    """Scale real and imaginary components without forming 2**exponent."""
    result = np.empty(values.shape, dtype=np.complex128)
    result.real = np.ldexp(values.real, exponent)
    result.imag = np.ldexp(values.imag, exponent)
    return result


def _normalize(
    values: NDArray[np.complex128],
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """Separate a complex mantissa and a base-two exponent."""
    _, power = np.frexp(np.maximum(np.abs(values.real), np.abs(values.imag)))
    exponent = np.asarray(power, dtype=np.int64)
    return _ldexp_complex(values, -exponent), exponent


def _scaled_polynomial(
    coefficients: NDArray[np.complex128], z: NDArray[np.complex128]
) -> tuple[NDArray[np.complex128], NDArray[np.int64]]:
    """Evaluate Horner's rule with separately stored binary exponents."""
    zm, ze = _normalize(z)
    cm, ce = _normalize(coefficients)
    mantissa = np.full(z.shape, cm[-1], dtype=np.complex128)
    exponent = np.full(z.shape, ce[-1], dtype=np.int64)
    for i in range(len(coefficients) - 2, -1, -1):
        mantissa, shift = _normalize(mantissa * zm)
        exponent = np.where(mantissa != 0, exponent + ze + shift, 0)
        # A zero summand must not force a nonzero summand to underflow.
        common = np.where(mantissa == 0, ce[i], np.maximum(exponent, ce[i]))
        if cm[i] == 0:
            common = exponent
        summed = _ldexp_complex(mantissa, exponent - common) + _ldexp_complex(
            np.full(z.shape, cm[i], dtype=np.complex128), ce[i] - common
        )
        mantissa, shift = _normalize(summed)
        exponent = np.where(mantissa != 0, common + shift, 0)
    return mantissa, exponent


def _scaled_ratio(
    p: NDArray[np.complex128], q: NDArray[np.complex128], z: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """Divide polynomial values before restoring their binary scale."""
    a, ae = _scaled_polynomial(p, z)
    b, be = _scaled_polynomial(q, z)
    result = _ldexp_complex(np.asarray(a / b, dtype=np.complex128), ae - be)
    result[(b == 0) & (a != 0)] = infinity
    return result


class RationalMap:
    r"""Represent a reduced rational map :math:`R=P/Q` on the Riemann sphere.

    Parameters
    ----------
    numerator_coefficients, denominator_coefficients : array_like
        Finite coefficients in ascending order: ``[1, 2, 3]`` means
        :math:`1+2z+3z^2`. The denominator must not be identically zero.
    variable : str, optional
        Symbol used for textual representations.

    Notes
    -----
    Exact common factors of the supplied binary floating-point coefficients
    are cancelled at construction. Nearly common factors are not cancelled.
    The degree is :math:`\max(\deg P,\deg Q)` after reduction; constant maps
    have degree zero. Classical Fatou/Julia theory assumes degree at least two.
    """

    def __init__(
        self,
        numerator_coefficients: ArrayLike,
        denominator_coefficients: ArrayLike,
        variable: str = "z",
    ) -> None:
        numerator = _coefficients(numerator_coefficients)
        denominator = _coefficients(denominator_coefficients)
        if not denominator.any():
            raise ValueError("The denominator polynomial must not be zero.")
        if not isinstance(variable, str) or not variable.isidentifier():
            raise ValueError("variable must be a valid identifier.")
        symbol = sp.Symbol(variable)
        p = sp.Poly(_expression(numerator, symbol), symbol, extension=sp.I)
        q = sp.Poly(_expression(denominator, symbol), symbol, extension=sp.I)
        common = sp.gcd(p, q)
        if common.degree() > 0:
            numerator = _coefficients(
                [complex(c) for c in p.exquo(common).all_coeffs()[::-1]]
            )
            denominator = _coefficients(
                [complex(c) for c in q.exquo(common).all_coeffs()[::-1]]
            )
        if not numerator.any():
            denominator = np.ones(1, dtype=np.complex128)
        num_values, den_values = numerator, denominator
        self._variable = variable
        self._numerator = Polynomial(num_values, symbol=variable)
        self._denominator = Polynomial(den_values, symbol=variable)
        self._degree_difference = int(
            self._numerator.degree() - self._denominator.degree()
        )
        self._degree = int(max(self._numerator.degree(), self._denominator.degree()))

    def __call__(
        self, Z: ArrayLike, *, out: NDArray[np.complex128] | None = None
    ) -> NDArray[np.complex128]:
        r"""Evaluate the map without changing the input unless ``out`` is supplied.

        Parameters
        ----------
        Z : array_like
            Points in :math:`\widehat{\mathbb C}`. Scalars produce 0-D arrays.
            An infinite component represents infinity; any NaN is invalid.
        out : ndarray of complex128, optional
            Writable destination of the same shape. May alias ``Z``.

        Returns
        -------
        ndarray of complex128
            Images with the shape of ``Z``. Poles and overflowing results are
            canonical ``inf+0j``. Invalid inputs or numerical 0/0 remain NaN.

        Notes
        -----
        Horner evaluation stores complex mantissas and binary exponents
        separately. Polynomial values may exceed float64 range internally;
        the ratio is formed before rescaling. This does not depend on platform
        long-double support. Severe cancellation remains ill-conditioned.
        """
        values = np.asarray(Z, dtype=np.complex128)
        if out is not None and (
            out.shape != values.shape
            or out.dtype != np.complex128
            or not out.flags.writeable
        ):
            raise ValueError("out must be writable complex128 with the input shape.")
        result = np.full(values.shape, complex(np.nan, np.nan), dtype=np.complex128)
        valid = ~np.isnan(values)
        infinite = valid & np.isinf(values)
        finite = valid & ~infinite
        p = np.asarray(self._numerator.coef, dtype=np.complex128)
        q = np.asarray(self._denominator.coef, dtype=np.complex128)
        with np.errstate(
            over="ignore", under="ignore", divide="ignore", invalid="ignore"
        ):
            result[finite] = _scaled_ratio(p, q, values[finite])
            if self._degree_difference > 0:
                result[infinite] = infinity
            elif self._degree_difference < 0:
                result[infinite] = 0
            else:
                result[infinite] = _scaled_ratio(
                    p[-1:], q[-1:], np.zeros(1, dtype=np.complex128)
                )[0]
        result[np.isinf(result)] = infinity
        if out is not None:
            np.copyto(out, result)
            return out
        return result

    def __str__(self) -> str:
        """Return a LaTeX representation of the reduced map."""
        symbol = sp.Symbol(self.variable)
        return str(
            sp.latex(
                _expression(
                    np.asarray(self._numerator.coef, dtype=np.complex128), symbol
                )
                / _expression(
                    np.asarray(self._denominator.coef, dtype=np.complex128), symbol
                )
            )
        )

    @classmethod
    def from_latex(
        cls, latex_str: str, var: str = "z", imag_const: str = "i"
    ) -> RationalMap:
        """Parse a rational expression using the optional ``latex`` extra.

        Parameters
        ----------
        latex_str : str
            Univariate rational expression in LaTeX.
        var : str, optional
            Variable symbol, retained by the resulting map.
        imag_const : str, optional
            Symbol interpreted as the imaginary unit; must differ from ``var``.

        Returns
        -------
        RationalMap
            Reduced map represented by the expression.
        """
        from sympy.parsing.latex import parse_latex

        if var == imag_const:
            raise ValueError("The variable and imaginary-unit symbols must differ.")
        expression = sp.cancel(
            parse_latex(latex_str, strict=True).subs(sp.Symbol(imag_const), sp.I)
        )
        num, den = expression.as_numer_denom()
        return cls(_sympy_to_coeffs(num, var), _sympy_to_coeffs(den, var), variable=var)

    @property
    def variable(self) -> str:
        """str: Symbol used to represent the variable."""
        return self._variable

    @property
    def numerator(self) -> Polynomial:
        """Polynomial: Independent copy of the reduced numerator."""
        return self._numerator.copy()

    @property
    def denominator(self) -> Polynomial:
        """Polynomial: Independent copy of the reduced denominator."""
        return self._denominator.copy()

    @property
    def degree(self) -> int:
        """int: Maximum polynomial degree after exact common-factor cancellation."""
        return self._degree

    def derivative(self) -> RationalMap:
        r"""Return the affine derivative :math:`R'=(P'Q-PQ')/Q^2`.

        Returns
        -------
        RationalMap
            Derivative in the finite coordinate, not the multiplier at infinity.
        """
        p, q = self._numerator, self._denominator
        return RationalMap(
            (p.deriv() * q - p * q.deriv()).coef, (q * q).coef, self.variable
        )

    def poles(self, include_infinity: bool = True) -> NDArray[np.complex128]:
        """Return poles of the reduced map, counted with multiplicity.

        Parameters
        ----------
        include_infinity : bool, optional
            Include infinity with multiplicity ``max(deg(P)-deg(Q), 0)``.

        Returns
        -------
        ndarray of complex128
            Numerical roots of Q, followed by any infinite poles.
        """
        finite = np.asarray(self._denominator.roots(), dtype=np.complex128)
        count = max(self._degree_difference, 0) if include_infinity else 0
        return np.concatenate((finite, np.full(count, infinity, dtype=np.complex128)))

    def roots(self, include_infinity: bool = True) -> NDArray[np.complex128]:
        """Return zeros of the reduced map, counted with multiplicity.

        Parameters
        ----------
        include_infinity : bool, optional
            Include infinity with multiplicity ``max(deg(Q)-deg(P), 0)``.

        Returns
        -------
        ndarray of complex128
            Numerical roots of P, followed by any infinite zeros.

        Raises
        ------
        ValueError
            If the map is identically zero (every point is a zero).
        """
        if not self._numerator.coef.any():
            raise ValueError("The zero map has infinitely many zeros.")
        finite = np.asarray(self._numerator.roots(), dtype=np.complex128)
        count = max(-self._degree_difference, 0) if include_infinity else 0
        return np.concatenate((finite, np.full(count, infinity, dtype=np.complex128)))
