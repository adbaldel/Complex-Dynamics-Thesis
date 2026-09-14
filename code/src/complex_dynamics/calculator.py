import numpy as np
from numpy.typing import NDArray

from complex_dynamics.rational_map import RationalMap

_default_max_iterations = 100
_default_pixel_count = 400
_default_escape_radius = 2
_default_tolerance = 1e-2


def complex_to_sphere(
    W: NDArray[np.complex128],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Forward stereographic projection: complex plane -> unit sphere.
    z = infinity maps to the north pole (0, 0, 1).
    """
    values = np.asarray(W)
    real = values.real
    imaginary = values.imag
    is_infinite = np.isinf(real) | np.isinf(imaginary)
    is_finite = ~is_infinite

    # NaN is an invalid complex value, while infinity is the north pole of the
    # Riemann sphere.  Keeping that distinction prevents numerical failures
    # from being silently interpreted as convergence to infinity.
    X = np.full(values.shape, np.nan, dtype=np.float64)
    Y = np.full(values.shape, np.nan, dtype=np.float64)
    Z = np.full(values.shape, np.nan, dtype=np.float64)
    X[is_infinite] = 0.0
    Y[is_infinite] = 0.0
    Z[is_infinite] = 1.0

    finite_real = real[is_finite]
    finite_imaginary = imaginary[is_finite]
    scale = np.maximum(
        1.0, np.maximum(np.abs(finite_real), np.abs(finite_imaginary))
    )
    scaled_real = finite_real / scale
    scaled_imaginary = finite_imaginary / scale
    inverse_scale = 1.0 / scale
    denominator = inverse_scale**2 + scaled_real**2 + scaled_imaginary**2

    X[is_finite] = 2.0 * scaled_real * inverse_scale / denominator
    Y[is_finite] = 2.0 * scaled_imaginary * inverse_scale / denominator
    Z[is_finite] = (
        scaled_real**2 + scaled_imaginary**2 - inverse_scale**2
    ) / denominator

    return X, Y, Z


def sphere_to_complex(
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
) -> NDArray[np.complex128]:
    """
    Inverse stereographic projection: unit sphere -> complex plane.
    The north pole (0, 0, 1) maps to infinity.
    """
    numerator = X + 1j * Y
    denominator = 1 - Z

    with np.errstate(divide="ignore", invalid="ignore"):
        W = np.where(denominator != 0, numerator / denominator, np.inf)

    return W


def chordal_distance(
    Z: NDArray[np.complex128], W: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """
    Returns the chordal distance from each point in Z to each point in W.

    Infinite values represent the point at infinity. NaN inputs produce NaN
    distances. The calculation is performed on stable unit-sphere coordinates
    to avoid overflow for very large finite complex values.
    """
    Z_x, Z_y, Z_z = complex_to_sphere(Z)
    W_x, W_y, W_z = complex_to_sphere(W)
    distance = np.hypot(np.hypot(Z_x - W_x, Z_y - W_y), Z_z - W_z)

    return distance


def chordal_distance_to_point(
    Z: NDArray[np.complex128], a: np.complex128
) -> NDArray[np.float64]:
    """
    Returns the chordal distance from each point in Z to the point a.

    Infinite values represent the point at infinity and NaN inputs produce NaN
    distances.
    """
    Z_x, Z_y, Z_z = complex_to_sphere(Z)
    a_x, a_y, a_z = complex_to_sphere(np.asarray(a))
    distance = np.hypot(np.hypot(Z_x - a_x, Z_y - a_y), Z_z - a_z)

    return distance


def spherical_distance(
    Z: NDArray[np.complex128], W: NDArray[np.complex128]
) -> NDArray[np.float64]:
    """
    Returns the spherical distance from each point in Z to each point in W.
    """
    distance = 2 * np.arcsin(chordal_distance(Z, W) / 2)

    return distance


def spherical_distance_to_point(
    Z: NDArray[np.complex128], a: np.complex128
) -> NDArray[np.float64]:
    """
    Returns the spherical distance from each point in Z to the point a.
    """
    distance = 2 * np.arcsin(chordal_distance_to_point(Z, a) / 2)

    return distance


def create_mesh(
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    xpixels: int = _default_pixel_count,
    ypixels: int = _default_pixel_count,
) -> NDArray[np.complex128]:
    """
    Returns a mesh representing the region of the extended complex plane that is
    contained in (xmin, xmax)x(ymin, ymax).
    """
    X, Y = np.meshgrid(
        np.linspace(xmin, xmax, xpixels), np.linspace(ymin, ymax, ypixels)
    )
    return X + (1j * Y)


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
    """
    Returns a mesh representing the projection into the extended complex plane
    of the riemann sphere contained in the region delimited by the polar angle
    interval (theta_min, theta_max) and the Azimuth angle interval (phi_min,
    phi_max); and the X, Y, Z 3D coordinates of each point in the mesh.
    """
    THETA, PHI = np.meshgrid(
        np.linspace(theta_min, theta_max, theta_pixels),
        np.linspace(phi_min, phi_max, phi_pixels),
    )

    # Convert to 3D Cartesian coordinates (Unit Sphere)
    X = np.sin(PHI) * np.cos(THETA)
    Y = np.sin(PHI) * np.sin(THETA)
    Z = np.cos(PHI)

    mesh = sphere_to_complex(X, Y, Z)

    # Inverse stereographic projection: Map 3D sphere coordinates to 2D complex
    # plane z.
    # North Pole Z=1 maps to infinity; South Pole Z=-1 maps to the origin (0,0)
    return mesh, X, Y, Z


def calculate_forward_orbit(
    Z: NDArray[np.complex128],
    R: RationalMap,
    max_iterations: int = _default_max_iterations,
) -> NDArray[np.complex128]:
    """
    Generates the defined number of iterations of forward orbit of the given
    points (Z).
    """
    forward_orbit = [Z.copy()]

    for i in range(max_iterations):
        R(Z)
        forward_orbit.append(Z.copy())

    return np.stack(forward_orbit)


def calculate_divergence_times(
    Z: NDArray[np.complex128],
    R: RationalMap,
    max_iterations: int = _default_max_iterations,
    escape_radius: float = _default_escape_radius,
) -> NDArray[np.int_]:
    """
    Returns an array that represents the iterations each point took to escape to
    infinity (has absolute value greater than the escape_radius). It also
    applies R to each point (modifying the given Z) until it escapes out of the
    escape radius.
    """
    divergence_times = np.zeros(Z.shape, dtype=int)
    active = np.abs(Z) < escape_radius

    for i in range(max_iterations):
        if not active.any():
            break

        # 1. Update active mask
        active = np.abs(Z) < escape_radius

        # 2. Iterate active points
        Z[active] = R(Z[active])
        divergence_times[active] += 1

    return divergence_times


def calculate_convergence_times(
    Z: NDArray[np.complex128],
    R: RationalMap,
    attractors: NDArray[np.complex128],
    max_iterations: int = _default_max_iterations,
    tolerance: float = _default_tolerance,
) -> tuple[NDArray[np.int_], NDArray[np.complex128]]:
    """
    Returns a tuple with two arrays. The first array represents the iterations
    each point took to converge to an attractor, and the second array represents
    the attractor to which each point has converged. It also applies R to each
    point (modifying the given Z) until converges to an attractor.
    """
    convergence_times = np.zeros(Z.shape, dtype=int)
    convergence_points = np.full(
        Z.shape, np.nan + 1j * np.nan, dtype=np.complex128
    )

    active = np.ones(Z.shape, dtype=bool)

    for i in range(max_iterations):
        if not active.any():
            break

        # 1. Update active mask
        for attractor in attractors:
            # Check if points have converged to any attractor
            just_converged = active & (
                chordal_distance_to_point(Z, attractor) < tolerance
            )
            # Note to which attractor they have converged
            convergence_points[just_converged] = attractor
            # Remove them from the active mask
            active[just_converged] = False

        # 2. Iterate active points
        Z[active] = R(Z[active])
        convergence_times[active] += 1

    # One final check to see if some point has converged in the last iteration
    for attractor in attractors:
        just_converged = active & (
            chordal_distance_to_point(Z, attractor) < tolerance
        )
        convergence_points[just_converged] = attractor

    return (convergence_times, convergence_points)
