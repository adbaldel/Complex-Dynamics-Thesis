"""Pure vectorized geometry and finite-time dynamics on the Riemann sphere.

Distances use the unit sphere: chordal distance ranges from 0 to 2. This is
twice the alternative normalized chordal metric with diameter 1. Broadcasting
is elementwise; use ``z[:, None]`` explicitly for pairwise distances.

For degree at least two, the Fatou set is the open set where the iterates form
a normal family in the spherical topology. Fatou components are its connected
components; the Julia set is its complement in the Riemann sphere. Finite-time
escape and target-entry tests do not certify membership in either set.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._validation import count as _count
from ._validation import positive as _positive
from .rational_map import RationalMap, infinity

_default_max_iterations = 100
_default_pixel_count = 400
_default_escape_radius = 2
_default_tolerance = 1e-2


def complex_to_sphere(
    W: ArrayLike,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    r"""Project the extended plane onto the unit sphere.

    Parameters
    ----------
    W : array_like
        Complex points; infinity denotes the north pole and NaN is invalid.

    Returns
    -------
    X, Y, Z : ndarray of float64
        Coordinates with the input shape, computed with scaling to avoid overflow.

    Notes
    -----
    For finite :math:`w=x+iy`, the coordinates are
    :math:`(2x,2y,|w|^2-1)/(1+|w|^2)`. Infinity maps to (0, 0, 1).
    """
    values = np.asarray(W, dtype=np.complex128)
    valid = ~np.isnan(values)
    infinite = valid & np.isinf(values)
    finite = valid & ~infinite
    X, Y, Z = (np.full(values.shape, np.nan, dtype=np.float64) for _ in range(3))
    X[infinite], Y[infinite], Z[infinite] = 0, 0, 1
    real, imag = values.real[finite], values.imag[finite]
    scale = np.maximum(1, np.maximum(np.abs(real), np.abs(imag)))
    x, y, inverse = real / scale, imag / scale, 1 / scale
    denominator = inverse**2 + x**2 + y**2
    X[finite] = 2 * x * inverse / denominator
    Y[finite] = 2 * y * inverse / denominator
    Z[finite] = (x**2 + y**2 - inverse**2) / denominator
    return X, Y, Z


def sphere_to_complex(
    X: ArrayLike, Y: ArrayLike, Z: ArrayLike
) -> NDArray[np.complex128]:
    r"""Project unit-sphere coordinates onto the extended complex plane.

    Parameters
    ----------
    X, Y, Z : array_like
        Broadcastable real coordinates on the unit sphere. NaNs propagate.

    Returns
    -------
    ndarray of complex128
        :math:`(X+iY)/(1-Z)`, with the north pole mapped to infinity.

    Raises
    ------
    ValueError
        If non-NaN coordinates are nonfinite or off the unit sphere.

    Notes
    -----
    The equivalent formula :math:`(1+Z)/(X-iY)` is used in the northern
    hemisphere to avoid subtractive cancellation near the north pole.
    """
    x, y, z = np.broadcast_arrays(
        np.asarray(X, dtype=np.float64),
        np.asarray(Y, dtype=np.float64),
        np.asarray(Z, dtype=np.float64),
    )
    invalid = np.isnan(x) | np.isnan(y) | np.isnan(z)
    norm = np.hypot(np.hypot(x, y), z)
    if np.any(
        ~invalid & (~np.isfinite(norm) | ~np.isclose(norm, 1, rtol=1e-12, atol=1e-12))
    ):
        raise ValueError("Coordinates must lie on the unit sphere.")
    result = np.full(x.shape, complex(np.nan, np.nan), dtype=np.complex128)
    north = ~invalid & (z > 0)
    south = ~invalid & ~north
    # Long intermediates retain tiny transverse coordinates near either pole.
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        result[south] = (x[south].astype(np.clongdouble) + 1j * y[south]) / (
            1 - z[south]
        )
        result[north] = (1 + z[north].astype(np.longdouble)) / (
            x[north].astype(np.clongdouble) - 1j * y[north]
        )
    pole = north & (x == 0) & (y == 0)
    result[pole | np.isinf(result)] = infinity
    return result


def chordal_distance(Z: ArrayLike, W: ArrayLike) -> NDArray[np.float64]:
    r"""Compute the unit-sphere chordal metric with NumPy broadcasting.

    Parameters
    ----------
    Z, W : array_like
        Points in the extended complex plane; NaNs propagate.

    Returns
    -------
    ndarray of float64
        :math:`2|z-w|/\sqrt{(1+|z|^2)(1+|w|^2)}` for finite inputs.
        Distance to infinity is :math:`2/\sqrt{1+|z|^2}`; the diameter is 2.
    """
    x, y, z = complex_to_sphere(Z)
    u, v, w = complex_to_sphere(W)
    return np.asarray(
        np.minimum(np.hypot(np.hypot(x - u, y - v), z - w), 2), dtype=np.float64
    )


def chordal_distance_to_point(Z: ArrayLike, a: complex) -> NDArray[np.float64]:
    """Compute chordal distances to one point (diameter 2).

    Parameters
    ----------
    Z : array_like
        Complex points.
    a : complex
        Target point, possibly infinity.

    Returns
    -------
    ndarray of float64
        Distances with the shape of Z; NaNs propagate.
    """
    return chordal_distance(Z, a)


def spherical_distance(Z: ArrayLike, W: ArrayLike) -> NDArray[np.float64]:
    r"""Compute geodesic distance on the unit sphere, in radians.

    Parameters
    ----------
    Z, W : array_like
        Broadcastable extended complex points.

    Returns
    -------
    ndarray of float64
        Values in :math:`[0,\pi]`, equal to :math:`2\arcsin(d/2)` where d is
        chordal distance. NaNs propagate.

    Notes
    -----
    An atan2 half-angle formula retains precision near antipodal points.
    """
    x, y, z = complex_to_sphere(Z)
    u, v, w = complex_to_sphere(W)
    difference = np.hypot(np.hypot(x - u, y - v), z - w)
    total = np.hypot(np.hypot(x + u, y + v), z + w)
    return np.asarray(2 * np.arctan2(difference, total), dtype=np.float64)


def spherical_distance_to_point(Z: ArrayLike, a: complex) -> NDArray[np.float64]:
    """Compute geodesic distances to one point, in radians.

    Parameters
    ----------
    Z : array_like
        Complex points.
    a : complex
        Target point, possibly infinity.

    Returns
    -------
    ndarray of float64
        Distances in [0, pi] with the shape of Z; NaNs propagate.
    """
    return spherical_distance(Z, a)


def create_mesh(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xpixels: int = _default_pixel_count,
    ypixels: int = _default_pixel_count,
) -> NDArray[np.complex128]:
    """Create a rectangular mesh including its endpoints.

    Parameters
    ----------
    xmin, xmax, ymin, ymax : float
        Finite ordered bounds; equal endpoints are allowed.
    xpixels, ypixels : int, optional
        Positive sample counts along each coordinate.

    Returns
    -------
    ndarray of complex128
        Shape (ypixels, xpixels), with rows increasing in imaginary part.
    """
    _count(xpixels, "xpixels", minimum=1)
    _count(ypixels, "ypixels", minimum=1)
    if not np.isfinite([xmin, xmax, ymin, ymax]).all() or xmin > xmax or ymin > ymax:
        raise ValueError("Mesh bounds must be finite and ordered.")
    x, y = np.meshgrid(
        np.linspace(xmin, xmax, xpixels), np.linspace(ymin, ymax, ypixels)
    )
    return np.asarray(x + 1j * y, dtype=np.complex128)


def create_projected_sphere_mesh(
    theta_min: float = 0,
    theta_max: float = 2 * np.pi,
    phi_min: float = 0,
    phi_max: float = np.pi,
    theta_pixels: int = _default_pixel_count,
    phi_pixels: int = _default_pixel_count,
) -> tuple[
    NDArray[np.complex128],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    """Sample the sphere uniformly in azimuth and colatitude.

    Parameters
    ----------
    theta_min, theta_max : float, optional
        Ordered azimuth bounds in radians, within [0, 2*pi].
    phi_min, phi_max : float, optional
        Ordered colatitude bounds in radians, within [0, pi].
    theta_pixels, phi_pixels : int, optional
        Positive sample counts. Endpoints and the azimuth seam are included.

    Returns
    -------
    mesh : ndarray of complex128
        Projected points, shape (phi_pixels, theta_pixels).
    X, Y, Z : ndarray of float64
        Unit-sphere coordinates with the same shape. Sampling is not equal-area.
    """
    if not (
        0 <= theta_min <= theta_max <= 2 * np.pi and 0 <= phi_min <= phi_max <= np.pi
    ):
        raise ValueError("Invalid azimuth or colatitude bounds.")
    angles = create_mesh(
        theta_min, theta_max, phi_min, phi_max, theta_pixels, phi_pixels
    )
    theta, phi = angles.real, angles.imag
    sine = np.sin(phi)
    sine[(phi == 0) | (phi == np.pi)] = 0
    X, Y, Z = sine * np.cos(theta), sine * np.sin(theta), np.cos(phi)
    return sphere_to_complex(X, Y, Z), X, Y, Z


def calculate_forward_orbit(
    Z: ArrayLike, R: RationalMap, max_iterations: int = _default_max_iterations
) -> NDArray[np.complex128]:
    r"""Compute :math:`(z,R(z),\ldots,R^n(z))` without mutating the input.

    Parameters
    ----------
    Z : array_like
        Initial points.
    R : RationalMap
        Map to iterate.
    max_iterations : int, optional
        Nonnegative number of applications of R.

    Returns
    -------
    ndarray of complex128
        Shape ``(max_iterations + 1, *Z.shape)``, including the initial points.
    """
    _count(max_iterations, "max_iterations")
    values = np.asarray(Z, dtype=np.complex128)
    orbit = np.empty((max_iterations + 1, *values.shape), dtype=np.complex128)
    orbit[0] = values
    for i in range(max_iterations):
        orbit[i + 1] = R(orbit[i])
    return orbit


def calculate_divergence_times(
    Z: ArrayLike,
    R: RationalMap,
    max_iterations: int = _default_max_iterations,
    escape_radius: float = _default_escape_radius,
) -> NDArray[np.int_]:
    """Count steps until first exit from an open disk, without input mutation.

    Parameters
    ----------
    Z : array_like
        Initial points; NaNs receive the sentinel -1.
    R : RationalMap
        Map to iterate.
    max_iterations : int, optional
        Nonnegative iteration limit.
    escape_radius : float, optional
        Positive finite disk radius. Points on the boundary have exited.

    Returns
    -------
    ndarray of int
        First n with abs(R**n(z)) >= escape_radius, capped at max_iterations.
        A capped count does not distinguish a last-step exit from no exit.

    Notes
    -----
    First exit is not proof of divergence for an arbitrary rational map.
    For example, 1/z sends infinity back to zero. A polynomial-specific escape
    bound is needed before interpreting these counts as escape to infinity.
    """
    _count(max_iterations, "max_iterations")
    _positive(escape_radius, "escape_radius")
    values = np.array(Z, dtype=np.complex128, copy=True)
    times = np.zeros(values.shape, dtype=np.int_)
    times[np.isnan(values)] = -1
    active = ~np.isnan(values) & (np.abs(values) < escape_radius)
    for _ in range(max_iterations):
        if not active.any():
            break
        values[active] = R(values[active])
        times[active] += 1
        times[np.isnan(values)] = -1
        active &= ~np.isnan(values) & (np.abs(values) < escape_radius)
    return times


def calculate_convergence_times(
    Z: ArrayLike,
    R: RationalMap,
    attractors: ArrayLike,
    max_iterations: int = _default_max_iterations,
    tolerance: float = _default_tolerance,
) -> tuple[NDArray[np.int_], NDArray[np.complex128]]:
    """Measure first entry into chordal neighborhoods of target points.

    Parameters
    ----------
    Z : array_like
        Initial points, left unchanged. Invalid points receive count -1.
    R : RationalMap
        Map to iterate.
    attractors : array_like
        One-dimensional target list, possibly empty, without NaNs. Infinity is
        allowed. These points are supplied by the caller, not certified attractors.
    max_iterations : int, optional
        Nonnegative iteration limit; initial and final states are checked.
    tolerance : float, optional
        Positive finite threshold for the chordal metric of diameter 2.
        First target in input order wins if neighborhoods overlap.

    Returns
    -------
    times : ndarray of int
        First entry counts; max_iterations for unresolved points, -1 for invalid.
    points : ndarray of complex128
        Matching targets; NaN for unresolved or invalid points. Shapes match Z.

    Notes
    -----
    Entering a neighborhood does not prove asymptotic convergence or membership
    in a Fatou component. Iteration is vectorized over active points. Targets
    are checked in sequence to keep memory linear in the number of points.
    """
    _count(max_iterations, "max_iterations")
    _positive(tolerance, "tolerance")
    targets = np.asarray(attractors, dtype=np.complex128)
    if targets.ndim != 1 or np.isnan(targets).any():
        raise ValueError("attractors must be a one-dimensional array without NaNs.")
    values = np.array(Z, dtype=np.complex128, copy=True)
    times = np.zeros(values.shape, dtype=np.int_)
    points = np.full(values.shape, complex(np.nan, np.nan), dtype=np.complex128)
    active = np.asarray(~np.isnan(values)).copy()
    times[~active] = -1
    target_coordinates = [complex_to_sphere(target) for target in targets]
    for step in range(max_iterations + 1):
        if not active.any():
            break
        indices = np.flatnonzero(active)
        x, y, z = complex_to_sphere(values.flat[indices])
        unmatched = np.ones(indices.size, dtype=bool)
        for target, (u, v, w) in zip(targets, target_coordinates, strict=True):
            match = unmatched & (np.hypot(np.hypot(x - u, y - v), z - w) < tolerance)
            points.flat[indices[match]] = target
            active.flat[indices[match]] = False
            unmatched[match] = False
        if step == max_iterations:
            break
        values[active] = R(values[active])
        times[active] += 1
        invalid = np.isnan(values)
        times[invalid] = -1
        active[invalid] = False
    return times, points
