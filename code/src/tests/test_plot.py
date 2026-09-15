import sys

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from complex_dynamics import create_projected_sphere_mesh, plot


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_2d_import_does_not_load_pyvista():
    assert "pyvista" not in sys.modules


def test_divergence_plot_preserves_data_and_axes():
    data = np.array([[1, 2], [3, 4]])
    original = data.copy()
    fig, ax = plt.subplots()
    assert plot.plot_divergence_times(data, max_iterations=10, ax=ax) == (fig, ax)
    np.testing.assert_array_equal(data, original)
    assert ax.images[0].get_clim() == (0, 10)


def test_unresolved_empty_and_zero_step_basins():
    cmap = ListedColormap(["red", "blue"])
    cmap.set_bad("gray")
    points = np.array([[0, np.nan], [0, 0]], dtype=np.complex128)
    image = plot._generate_convergence_image(
        np.zeros((2, 2), dtype=int), points, np.array([0, 1, 2]), cmap, 0.2, 0.7
    )
    assert np.isfinite(image).all()
    np.testing.assert_allclose(image[0, 1], cmap.get_bad())
    np.testing.assert_array_equal(image[..., 3], 1)


def test_nan_boundaries_and_dilation_do_not_wrap():
    assert not plot._find_regions_boundary(np.full((3, 3), np.nan)).any()
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    thick = plot._thicken_boundary_mask(mask, 2)
    assert thick.sum() == 3 and mask.sum() == 1
    assert not thick[-1, 0] and not thick[0, -1]


def test_public_thickness_changes_boundary():
    data = np.zeros((7, 7), dtype=int)
    data[3, 3] = 10
    _, thin = plot.plot_divergence_times_boundary(data, max_iterations=10, thickness=1)
    _, thick = plot.plot_divergence_times_boundary(data, max_iterations=10, thickness=2)
    assert (
        np.asarray(thick.images[0].get_array()).sum()
        > np.asarray(thin.images[0].get_array()).sum()
    )


def test_surface_respects_existing_axes_and_zero_limit():
    _, x, y, z = create_projected_sphere_mesh(theta_pixels=4, phi_pixels=4)
    data = np.zeros((4, 4), dtype=int)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    assert plot.plot_projected_sphere_convergence_times(
        data, data.astype(complex), x, y, z, np.array([0]), ax=ax
    ) == (fig, ax)
    plot.plot_projected_sphere_divergence_times(data, x, y, z, max_iterations=0)


def test_empty_orbits_rejected():
    with pytest.raises(ValueError):
        plot.plot_orbit(np.array([], dtype=complex))
