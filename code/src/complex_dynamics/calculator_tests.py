import numpy as np

import complex_dynamics.calculator as dcalc
from complex_dynamics.rational_map import RationalMap


class TestCreateMesh:
    """Tests the rectangular complex-plane mesh generator."""

    def test_shape_and_dtype(self):
        mesh = dcalc.create_mesh(-1, 1, -2, 2, xpixels=10, ypixels=20)
        assert mesh.shape == (20, 10)
        assert mesh.dtype == np.complex128

    def test_corner_values(self):
        # Rows vary with y, columns vary with x: mesh[row, col] = x + iy
        mesh = dcalc.create_mesh(-1, 1, -2, 2, xpixels=3, ypixels=5)
        np.testing.assert_allclose(mesh[0, 0], -1 - 2j)
        np.testing.assert_allclose(mesh[0, -1], 1 - 2j)
        np.testing.assert_allclose(mesh[-1, 0], -1 + 2j)
        np.testing.assert_allclose(mesh[-1, -1], 1 + 2j)

    def test_default_pixel_counts(self):
        mesh = dcalc.create_mesh(-1, 1, -1, 1)
        assert mesh.shape == (400, 400)


class TestCreateProjectedSphereMesh:
    """Tests the inverse stereographic projection mesh generator."""

    def test_shape_and_dtype(self):
        (mesh, X, Y, Z) = dcalc.create_projected_sphere_mesh(
            theta_pixels=8, phi_pixels=6
        )
        assert mesh.shape == (6, 8)
        assert mesh.dtype == np.complex128

    def test_near_north_pole_is_large_magnitude(self):
        # Small phi -> near north pole (Z ~ 1) -> projects near infinity
        (mesh, X, Y, Z) = dcalc.create_projected_sphere_mesh(
            0, 2 * np.pi, 0.01, 0.02, theta_pixels=4, phi_pixels=4
        )
        assert np.all(np.abs(mesh) > 50)

    def test_near_south_pole_is_small_magnitude(self):
        # Large phi -> near south pole (Z ~ -1) -> projects near origin
        (mesh, X, Y, Z) = dcalc.create_projected_sphere_mesh(
            0,
            2 * np.pi,
            np.pi - 0.02,
            np.pi - 0.01,
            theta_pixels=4,
            phi_pixels=4,
        )
        assert np.all(np.abs(mesh) < 0.011)

    def test_equator_has_unit_magnitude(self):
        # phi = pi/2 -> Z = 0 on the sphere -> projects to the unit circle
        (mesh, X, Y, Z) = dcalc.create_projected_sphere_mesh(
            0, 2 * np.pi, np.pi / 2, np.pi / 2, theta_pixels=8, phi_pixels=1
        )
        np.testing.assert_allclose(np.abs(mesh), 1.0)


class TestChordalDistance:
    """Tests stable chordal-distance calculations on the Riemann sphere."""

    def test_huge_finite_values_do_not_overflow(self):
        Z = np.array([1e308 + 0j, -1e308 + 0j], dtype=np.complex128)
        W = np.array([-1e308 + 0j, 1e308 + 0j], dtype=np.complex128)

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            distance = dcalc.chordal_distance(Z, W)

        np.testing.assert_allclose(distance, [4e-308, 4e-308])

    def test_infinity_and_nan_are_distinguished(self):
        Z = np.array([np.inf + 0j, 0 + 0j, np.nan + 0j], dtype=np.complex128)

        distance = dcalc.chordal_distance_to_point(Z, np.inf + 0j)

        np.testing.assert_allclose(distance[:2], [0.0, 2.0])
        assert np.isnan(distance[2])

    def test_spherical_distance_is_safe_at_its_upper_bound(self):
        Z = np.array([0 + 0j], dtype=np.complex128)

        distance = dcalc.spherical_distance_to_point(Z, np.inf + 0j)

        np.testing.assert_allclose(distance, [np.pi])


class TestOrbit:
    """Tests forward orbit generation."""

    def test_identity_map_stays_constant(self):
        # R(z) = z -> every iterate equals the starting point
        R = RationalMap(
            np.array([0, 1], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([1 + 1j, -2 + 0j], dtype=np.complex128)

        result = dcalc.calculate_forward_orbit(mesh, R=R, max_iterations=3)

        assert result.shape == (4, 2)
        for step in result:
            np.testing.assert_allclose(step, mesh)

    def test_doubling_map_orbit_values(self):
        # R(z) = 2z -> orbit of z0 is z0, 2*z0, 4*z0, 8*z0, ...
        R = RationalMap(
            np.array([0, 2], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([1 + 0j], dtype=np.complex128)

        result = dcalc.calculate_forward_orbit(mesh, R=R, max_iterations=3)

        expected = np.array([[1 + 0j], [2 + 0j], [4 + 0j], [8 + 0j]])
        np.testing.assert_allclose(result, expected)

    def test_first_entry_is_not_aliased_to_later_mutations(self):
        # Regression test: md.calculate_forward_orbit()'s first snapshot must be a copy, not a
        # reference to the array that R() goes on to mutate in place.
        R = RationalMap(
            np.array([0, 2], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([1 + 0j], dtype=np.complex128)

        result = dcalc.calculate_forward_orbit(mesh, R=R, max_iterations=3)

        np.testing.assert_allclose(result[0], [1 + 0j])

    def test_does_not_mutate_input_mesh(self):
        R = RationalMap(
            np.array([0, 2], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([1 + 0j], dtype=np.complex128)
        mesh_copy = mesh.copy()

        dcalc.calculate_forward_orbit(mesh, R=R, max_iterations=5)

        np.testing.assert_allclose(mesh, mesh_copy)


class TestDivergenceRate:
    """Tests escape-time computation."""

    def test_point_that_escapes(self):
        # R(z) = 2z, escape_radius=2: z=1.5 -> 3.0 escapes after 1 iteration
        R = RationalMap(
            np.array([0, 2], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([1.5 + 0j], dtype=np.complex128)

        result = dcalc.calculate_divergence_times(
            mesh, R=R, num_iter=5, escape_radius=2
        )

        np.testing.assert_array_equal(result, [1])

    def test_point_already_outside_escape_radius(self):
        # Starts outside the escape radius -> never counted as active
        R = RationalMap(
            np.array([0, 2], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([5 + 0j], dtype=np.complex128)

        result = dcalc.calculate_divergence_times(
            mesh, R=R, num_iter=5, escape_radius=2
        )

        np.testing.assert_array_equal(result, [0])

    def test_point_that_never_escapes_hits_num_iter(self):
        # R(z) = z (identity) never escapes -> count saturates at num_iter
        R = RationalMap(
            np.array([0, 1], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([0.5 + 0j], dtype=np.complex128)

        result = dcalc.calculate_divergence_times(
            mesh, R=R, num_iter=10, escape_radius=2
        )

        np.testing.assert_array_equal(result, [10])

    def test_mixed_mesh_of_escaping_and_bounded_points(self):
        R = RationalMap(
            np.array([0, 2], dtype=np.complex128),
            np.array([1], dtype=np.complex128),
        )
        mesh = np.array([1.5 + 0j, 0.1 + 0j], dtype=np.complex128)

        result = dcalc.calculate_divergence_times(
            mesh, R=R, num_iter=5, escape_radius=2
        )

        # 1.5 escapes after 1 step; 0.1 doubles 5 times to 3.2, escaping on
        # the 5th iteration (0.1, 0.2, 0.4, 0.8, 1.6 all < 2; then -> 3.2)
        np.testing.assert_array_equal(result, [1, 5])


class TestConvergenceRate:
    """Tests convergence-time computation using Newton's method for z^2 - 1."""

    def _newton_map(self):
        # Newton's method iteration for f(z) = z^2 - 1: R(z) = (z^2+1)/(2z)
        return RationalMap(
            np.array([1, 0, 1], dtype=np.complex128),
            np.array([0, 2], dtype=np.complex128),
        )

    def test_converges_to_nearest_root(self):
        R = self._newton_map()
        attractors = np.array([1 + 0j, -1 + 0j])
        mesh = np.array([2 + 0j, -3 + 0j], dtype=np.complex128)

        conv_len, conv_point = dcalc.calculate_convergence_times(
            mesh, R=R, attractors=attractors, num_iter=20, tolerance=1e-6
        )

        np.testing.assert_allclose(conv_point, [1 + 0j, -1 + 0j])
        assert np.all(conv_len > 0)
        assert np.all(conv_len <= 20)

    def test_point_that_does_not_converge_keeps_default_and_full_iter_count(
        self,
    ):
        # A point that never gets within tolerance of any attractor should
        # run for the full num_iter and keep the sentinel default point.
        R = self._newton_map()
        attractors = np.array([100 + 100j])  # unreachable from this start
        mesh = np.array([2 + 0j], dtype=np.complex128)

        conv_len, conv_point = dcalc.calculate_convergence_times(
            mesh, R=R, attractors=attractors, num_iter=5, tolerance=1e-6
        )

        np.testing.assert_array_equal(conv_len, [5])
        np.testing.assert_allclose(conv_point, [np.nan + np.nan * 1j])

    def test_output_shapes_match_input_mesh(self):
        R = self._newton_map()
        attractors = np.array([1 + 0j, -1 + 0j])
        mesh = dcalc.create_mesh(-2, 2, -2, 2, xpixels=5, ypixels=5)

        conv_len, conv_point = dcalc.calculate_convergence_times(
            mesh, R=R, attractors=attractors, num_iter=10, tolerance=1e-3
        )

        assert conv_len.shape == mesh.shape
        assert conv_point.shape == mesh.shape
