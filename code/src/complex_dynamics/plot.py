"""Rendering helpers for finite-time numerical data.

Matplotlib functions return figures and axes without displaying them. Interactive
functions require the ``interactive`` extra and return a PyVista plotter; use
``show=True`` to open a window. All plotting functions leave input data unchanged.
Pixel boundaries are visual approximations, not certified Julia sets.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from numpy.typing import NDArray

from . import calculator as dcalc
from ._validation import count as _count

if TYPE_CHECKING:
    import pyvista as pv

ColorLike = str | Sequence[float]
ColormapOptions = str | Colormap | list[str]


_default_orbit_line_color = "gray"
_default_zi_marker = "o"
_default_zi_color = "black"
_default_z0_color = "green"
_default_max_iterations = dcalc._default_max_iterations
_default_thickness = 1
_default_listed_colormap = cast(ListedColormap, mpl.colormaps["Set1"])
_default_divergence_times_colormap = mpl.colormaps["binary"]
_default_projected_sphere_divergence_times_colormap = mpl.colormaps["magma_r"]
_default_color_darkness_factor = 0.7
_default_color_peak_position = 0.2


def _generate_or_load_fig_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return existing figure and axes, or create a new pair."""
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()
    else:
        fig = cast(Figure, ax.get_figure())
        if not isinstance(fig, Figure):
            raise ValueError("ax must belong to a Figure.")

    return fig, ax


def _generate_arc_paths(
    points: NDArray[np.float64], *, resolution: int = 30
) -> pv.PolyData:
    """Connect sphere points by great-circle arcs with deterministic antipodes."""
    import pyvista as pv

    _count(resolution, "resolution", minimum=2)
    if len(points) < 2:
        return pv.PolyData(points)
    paths = []
    for start, end in zip(points[:-1], points[1:], strict=True):
        dot = float(np.clip(np.dot(start, end), -1, 1))
        angle = np.arccos(dot)
        if angle < 1e-12:
            continue
        tangent = end - dot * start
        norm = np.linalg.norm(tangent)
        if norm < 1e-12:
            # Antipodes admit many geodesics; choose a deterministic tangent.
            axis = np.eye(3)[np.argmin(np.abs(start))]
            tangent = np.cross(start, axis)
            norm = np.linalg.norm(tangent)
        tangent /= norm
        t = np.linspace(0, angle, resolution + 1)[:, None]
        path = np.cos(t) * start + np.sin(t) * tangent
        path[0], path[-1] = start, end
        paths.append(pv.lines_from_points(path))
    if not paths:
        return pv.PolyData(points[:1])
    return paths[0].merge(paths[1:]) if len(paths) > 1 else paths[0]


def _find_regions_boundary(regions: NDArray[Any]) -> NDArray[np.bool_]:
    """Mark neighboring unequal regions without wrapping at image edges."""
    if regions.ndim != 2:
        raise ValueError("regions must be a two-dimensional array.")
    boundary = np.zeros(regions.shape, dtype=np.bool_)
    for a, b in ((np.s_[:-1, :], np.s_[1:, :]), (np.s_[:, :-1], np.s_[:, 1:])):
        left, right = regions[a], regions[b]
        different = (left != right) & ~(np.isnan(left) & np.isnan(right))
        boundary[a] |= different
        boundary[b] |= different
    return boundary


def _thicken_boundary_mask(
    boundary_mask: NDArray[np.bool_], thickness: int
) -> NDArray[np.bool_]:
    """Dilate a mask in four directions without wrapping or input mutation."""
    _count(thickness, "thickness", minimum=1)
    result = boundary_mask.copy()
    for _ in range(thickness - 1):
        previous = result.copy()
        result[1:] |= previous[:-1]
        result[:-1] |= previous[1:]
        result[:, 1:] |= previous[:, :-1]
        result[:, :-1] |= previous[:, 1:]
    return result


def _generate_boundary_image(
    boundary_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Convert a Boolean boundary mask to a floating-point image."""
    boundary_image = np.zeros(boundary_mask.shape, dtype=np.float64)
    boundary_image[boundary_mask] = 1
    boundary_image[~boundary_mask] = 0

    return boundary_image


def _plot_boundary_image(
    boundary_image: NDArray[np.float64],
    *,
    ax: Axes | None = None,
    origin: Literal["upper", "lower"] = "lower",
    cmap: str | Colormap = _default_divergence_times_colormap,
    **imshow_keywords: Any,
) -> tuple[Figure, Axes]:
    """Plot the given boundary image with the default style."""
    fig, ax = _generate_or_load_fig_ax(ax)
    ax.imshow(boundary_image, origin=origin, cmap=cmap, **imshow_keywords)

    return fig, ax


def _plot_boundary_image_on_sphere(
    boundary_image: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    rcount: int = 400,
    ccount: int = 400,
    linewidth: int = 0,
    antialiased: bool = False,
    shade: bool = False,
    **plot_surface_keywords: Any,
) -> tuple[Figure, Axes]:
    """Plot the given boundary image on the given sphere with the default style."""
    # Apply color to boundary_image for painting the surface of the sphere
    cmap = _default_divergence_times_colormap
    facecolors = cmap(boundary_image)

    # Draw sphere
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    surface_ax = cast(Axes3D, ax)
    surface_ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        rcount=rcount,
        ccount=ccount,
        linewidth=linewidth,
        antialiased=antialiased,
        shade=shade,  # don't let matplotlib relight the colors
        **plot_surface_keywords,
    )
    surface_ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    return fig, ax


def _plot_boundary_image_on_dynamic_sphere(
    boundary_image: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    cmap: ColormapOptions | None = _default_divergence_times_colormap,
    show: bool = False,
    off_screen: bool = False,
) -> pv.Plotter:
    """Render a boundary mask on an interactive sphere."""
    import pyvista as pv

    grid = pv.StructuredGrid(X, Y, Z)
    # StructuredGrid flattens points in Fortran order internally
    grid.point_data["divergence"] = boundary_image.flatten(order="F")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(
        grid,
        scalars="divergence",
        cmap=mpl.colormaps[cmap] if isinstance(cmap, str) else cmap,
        clim=[0, 1],
        smooth_shading=True,
    )
    if show:
        plotter.show()

    return plotter


def _generate_convergence_image(
    convergence_times: NDArray[np.int_],
    convergence_points: NDArray[np.complex128],
    attractors: NDArray[np.complex128],
    listed_colormap: ListedColormap,
    color_peak_position: float,
    color_darkness_factor: float,
) -> NDArray[np.float64]:
    """Color target basins, safely handling absent and unresolved regions."""
    if convergence_times.shape != convergence_points.shape:
        raise ValueError("Convergence arrays must have the same shape.")
    if not 0 < color_peak_position < 1 or not 0 <= color_darkness_factor <= 1:
        raise ValueError("Invalid color peak or darkness factor.")
    # Check if there are points that do not converge to any attractor
    all_converge = not np.any(np.isnan(convergence_points))
    extended_attractors = attractors.tolist()
    if not all_converge:
        extended_attractors.append(np.nan)

    # Obtain color for each attractor (and for non converging points)
    color_list = _generate_color_list(
        attractors=attractors,
        all_converge=all_converge,
        listed_colormap=listed_colormap,
    )

    convergence_image = np.zeros((*convergence_times.shape, 4))
    for attractor, color in zip(extended_attractors, color_list, strict=True):
        attractor_mask = (
            np.isnan(convergence_points)
            if np.isnan(attractor)
            else convergence_points == attractor
        )
        if not attractor_mask.any():
            continue

        if np.isnan(attractor):
            convergence_image[attractor_mask, :] = color

        else:
            # Create color map from dark color to white going through the color
            nodes = [0.0, color_peak_position, 1.0]
            dark_color = color.copy()
            dark_color[:3] *= color_darkness_factor
            if len(dark_color) == 4:
                dark_color[3] = 1.0

            colors = [
                dark_color,
                color,
                np.ones(4),
            ]
            attractor_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "mycmap",
                [
                    (node, cast(tuple[float, float, float, float], tuple(color)))
                    for node, color in zip(nodes, colors, strict=True)
                ],
            )

            # Apply color map
            normalized_convergence_times = convergence_times[attractor_mask] / max(
                1, int(np.max(convergence_times[attractor_mask]))
            )
            convergence_image[attractor_mask] = attractor_cmap(
                normalized_convergence_times
            )

    return convergence_image


def _generate_color_list(
    attractors: NDArray[np.complex128],
    all_converge: bool,
    listed_colormap: mpl.colors.ListedColormap,
) -> list[NDArray[np.float64]]:
    """Sample a palette with one color for every target."""
    color_list = [
        np.asarray(
            listed_colormap(i),
            dtype=np.float64,
        )
        for i in range(len(attractors))
    ]

    if not all_converge:
        color_list.append(np.array(listed_colormap.get_bad()))

    return color_list


# --- Orbit plots --------------------------------------------------------------


def plot_orbit(
    orbit: NDArray[np.complex128],
    *,
    ax: Axes | None = None,
    color: str = _default_orbit_line_color,
    z0_color: str = _default_z0_color,
    z0_label: str = r"$z_0$",
    marker: str = _default_zi_marker,
    markerfacecolor: str = _default_zi_color,
    markeredgecolor: str = _default_zi_color,
    **plot_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render orbit.

    Parameters
    ----------
    orbit : ndarray of complex128
        One-dimensional orbit including its initial point.
    ax : matplotlib.axes.Axes, optional
        Existing axes; sphere surfaces require three-dimensional axes.
    color : color, optional
        Color of the orbit line.
    z0_color : color, optional
        Color of the initial point.
    z0_label : str, optional
        Legend label for the initial point.
    marker : str, optional
        Matplotlib point marker.
    markerfacecolor : color, optional
        Fill color of point markers.
    markeredgecolor : color, optional
        Edge color of point markers.
    **plot_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.
    """
    if orbit.ndim != 1 or orbit.size == 0:
        raise ValueError("orbit must be a nonempty one-dimensional array.")
    real_parts = np.real(orbit)
    imag_parts = np.imag(orbit)

    fig, ax = _generate_or_load_fig_ax(ax)
    ax.plot(
        real_parts,
        imag_parts,
        color=color,
        marker=marker,
        markerfacecolor=markerfacecolor,
        markeredgecolor=markeredgecolor,
        **plot_keywords,
    )
    ax.scatter(
        real_parts[0],
        imag_parts[0],
        color=z0_color,
        label=z0_label,
        zorder=2,
    )

    return fig, ax


def plot_orbit_on_dynamic_sphere(
    orbit: NDArray[np.complex128],
    *,
    sphere_color: ColorLike | None = "white",
    sphere_opacity: float | Sequence[float] | None = 0.3,
    arc_color: ColorLike | None = _default_orbit_line_color,
    arc_width: float | None = 2,
    zi_color: ColorLike | None = _default_zi_color,
    zi_point_size: float = 10,
    z0_color: ColorLike | None = _default_z0_color,
    z0_point_size: float = 16,
    show: bool = False,
    off_screen: bool = False,
) -> pv.Plotter:
    """Render orbit on dynamic sphere.

    Parameters
    ----------
    orbit : ndarray of complex128
        One-dimensional orbit including its initial point.
    sphere_color : color, optional
        Color of the reference sphere.
    sphere_opacity : float or sequence of float, optional
        Opacity of the reference sphere.
    arc_color : color, optional
        Color of geodesic orbit arcs.
    arc_width : float, optional
        Width of geodesic orbit arcs.
    zi_color : color, optional
        Color of orbit points.
    zi_point_size : float, optional
        Rendered size of orbit points.
    z0_color : color, optional
        Color of the initial point.
    z0_point_size : float, optional
        Rendered size of the initial point.
    show : bool, optional
        Open an interactive window when True; default False.
    off_screen : bool, optional
        Construct the PyVista plotter for off-screen rendering.

    Returns
    -------
    pyvista.Plotter
        Plotter retaining the rendered scene. Call close() when finished.
    """
    import pyvista as pv

    if orbit.ndim != 1 or orbit.size == 0 or np.isnan(orbit).any():
        raise ValueError("orbit must be nonempty, one-dimensional, and without NaNs.")
    X, Y, Z = dcalc.complex_to_sphere(orbit)
    points = np.column_stack([X, Y, Z])

    plotter = pv.Plotter(off_screen=off_screen)

    # Reference sphere
    sphere = pv.Sphere(radius=1.0)
    plotter.add_mesh(
        sphere, color=sphere_color, opacity=sphere_opacity, smooth_shading=True
    )

    # Orbit path connecting z_i -> z_{i+1}
    orbit_arcs = _generate_arc_paths(points)
    plotter.add_mesh(orbit_arcs, color=arc_color, line_width=arc_width)

    # z_i points
    plotter.add_mesh(
        pv.PolyData(points),
        color=zi_color,
        point_size=zi_point_size,
        render_points_as_spheres=True,
    )

    # z_0, highlighted
    plotter.add_mesh(
        pv.PolyData(points[0]),
        color=z0_color,
        point_size=z0_point_size,
        render_points_as_spheres=True,
    )

    if show:
        plotter.show()

    return plotter


def plot_orbits(
    orbits: NDArray[np.complex128],
    *,
    ax: Axes | None = None,
    color: str = _default_orbit_line_color,
    z0_color: str = _default_z0_color,
    z0i_symbol: str = "z",
    marker: str = _default_zi_marker,
    markerfacecolor: str = _default_zi_color,
    markeredgecolor: str = _default_zi_color,
    **plot_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render orbits.

    Parameters
    ----------
    orbits : ndarray of complex128
        Orbits with shape (steps, points), including initial points.
    ax : matplotlib.axes.Axes, optional
        Existing axes; sphere surfaces require three-dimensional axes.
    color : color, optional
        Color of the orbit line.
    z0_color : color, optional
        Color of the initial point.
    z0i_symbol : str, optional
        Symbol in initial-point legend labels.
    marker : str, optional
        Matplotlib point marker.
    markerfacecolor : color, optional
        Fill color of point markers.
    markeredgecolor : color, optional
        Edge color of point markers.
    **plot_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.
    """
    fig, ax = _generate_or_load_fig_ax(ax)

    if orbits.ndim != 2 or orbits.shape[0] == 0:
        raise ValueError("orbits must have shape (steps, points), with steps > 0.")
    for i in range(orbits.shape[1]):
        real_parts = np.real(orbits[:, i])
        imag_parts = np.imag(orbits[:, i])

        ax.plot(
            real_parts,
            imag_parts,
            color=color,
            marker=marker,
            markerfacecolor=markerfacecolor,
            markeredgecolor=markeredgecolor,
            **plot_keywords,
        )
        ax.scatter(
            real_parts[0],
            imag_parts[0],
            color=z0_color,
            label=rf"${z0i_symbol}_{{0_{i}}}$",
            zorder=2,
        )

    return fig, ax


# --- Divergence times plots ----------------------------------------------------


def plot_divergence_times(
    divergence_times: NDArray[np.int_],
    *,
    max_iterations: int = _default_max_iterations,
    ax: Axes | None = None,
    origin: Literal["upper", "lower"] = "lower",
    cmap: str | Colormap = _default_divergence_times_colormap,
    **imshow_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render divergence times.

    Parameters
    ----------
    divergence_times : ndarray of int
        Two-dimensional first-exit counts; see calculate_divergence_times.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    ax : matplotlib.axes.Axes, optional
        Existing axes; sphere surfaces require three-dimensional axes.
    origin : {'upper', 'lower'}, optional
        Location of array row zero in the image.
    cmap : str or Colormap, optional
        Color map for the rendered scalar data.
    **imshow_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.
    """
    _count(max_iterations, "max_iterations")
    if "norm" not in imshow_keywords:
        imshow_keywords.setdefault("vmin", 0)
        imshow_keywords.setdefault("vmax", max(1, max_iterations))

    # Show image
    fig, ax = _generate_or_load_fig_ax(ax)
    ax.imshow(
        divergence_times,
        origin=origin,
        cmap=cmap,
        **imshow_keywords,
    )

    return fig, ax


def plot_divergence_times_boundary(
    divergence_times: NDArray[np.int_],
    *,
    max_iterations: int = _default_max_iterations,
    thickness: int = _default_thickness,
    ax: Axes | None = None,
    **imshow_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render divergence times boundary.

    Parameters
    ----------
    divergence_times : ndarray of int
        Two-dimensional first-exit counts; see calculate_divergence_times.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    thickness : int, optional
        Positive number of pixels in the boundary dilation radius plus one.
    ax : matplotlib.axes.Axes, optional
        Existing axes; sphere surfaces require three-dimensional axes.
    **imshow_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.

    Notes
    -----
    Boundaries approximate transitions between finite-time pixel labels. They
    do not certify a Julia set. Image edges do not wrap at the sphere seam.
    """
    # Find boundary
    regions = divergence_times == max_iterations
    boundary_mask = _find_regions_boundary(regions)
    boundary_mask = _thicken_boundary_mask(boundary_mask, thickness)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image(boundary_image, ax=ax, **imshow_keywords)


def plot_projected_sphere_divergence_times(
    divergence_times: NDArray[np.int_],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    max_iterations: int = _default_max_iterations,
    rcount: int = 400,
    ccount: int = 400,
    linewidth: int = 0,
    antialiased: bool = False,
    shade: bool = False,
    **plot_surface_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render projected sphere divergence times.

    Parameters
    ----------
    divergence_times : ndarray of int
        Two-dimensional first-exit counts; see calculate_divergence_times.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    rcount : int, optional
        Maximum surface samples along rows.
    ccount : int, optional
        Maximum surface samples along columns.
    linewidth : int, optional
        Width of surface edges.
    antialiased : bool, optional
        Whether surface edges are antialiased.
    shade : bool, optional
        Whether lighting modifies the supplied surface colors.
    **plot_surface_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.
    """
    # Normalize colors for applying in the surface of the sphere
    cmap = _default_projected_sphere_divergence_times_colormap
    _count(max_iterations, "max_iterations")
    facecolors = cmap(divergence_times / max(1, max_iterations))

    # Draw sphere
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    surface_ax = cast(Axes3D, ax)
    surface_ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=facecolors,
        rcount=rcount,
        ccount=ccount,
        linewidth=linewidth,
        antialiased=antialiased,
        shade=shade,  # don't let matplotlib relight the colors
        **plot_surface_keywords,
    )
    surface_ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    return fig, ax


def plot_projected_sphere_divergence_time_boundary(
    divergence_times: NDArray[np.int_],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    max_iterations: int = _default_max_iterations,
    thickness: int = _default_thickness,
    **plot_surface_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render projected sphere divergence time boundary.

    Parameters
    ----------
    divergence_times : ndarray of int
        Two-dimensional first-exit counts; see calculate_divergence_times.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    thickness : int, optional
        Positive number of pixels in the boundary dilation radius plus one.
    **plot_surface_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.

    Notes
    -----
    Boundaries approximate transitions between finite-time pixel labels. They
    do not certify a Julia set. Image edges do not wrap at the sphere seam.
    """
    # Find boundary
    regions = divergence_times == max_iterations
    boundary_mask = _find_regions_boundary(regions)
    boundary_mask = _thicken_boundary_mask(boundary_mask, thickness)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image_on_sphere(
        boundary_image, X, Y, Z, **plot_surface_keywords
    )


def dynamic_plot_projected_sphere_divergence_times(
    divergence_times: NDArray[np.int_],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    max_iterations: int = _default_max_iterations,
    cmap: ColormapOptions | None = _default_divergence_times_colormap,
    show: bool = False,
    off_screen: bool = False,
) -> pv.Plotter:
    """Render projected sphere divergence times.

    Parameters
    ----------
    divergence_times : ndarray of int
        Two-dimensional first-exit counts; see calculate_divergence_times.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    cmap : str or Colormap, optional
        Color map for the rendered scalar data.
    show : bool, optional
        Open an interactive window when True; default False.
    off_screen : bool, optional
        Construct the PyVista plotter for off-screen rendering.

    Returns
    -------
    pyvista.Plotter
        Plotter retaining the rendered scene. Call close() when finished.
    """
    import pyvista as pv

    grid = pv.StructuredGrid(X, Y, Z)
    # StructuredGrid flattens points in Fortran order internally
    grid.point_data["divergence"] = divergence_times.flatten(order="F")

    plotter = pv.Plotter(off_screen=off_screen)
    plotter.add_mesh(
        grid,
        scalars="divergence",
        cmap=mpl.colormaps[cmap] if isinstance(cmap, str) else cmap,
        clim=[0, max_iterations],
        smooth_shading=True,
    )
    if show:
        plotter.show()

    return plotter


def dynamic_plot_projected_sphere_divergence_times_boundary(
    divergence_times: NDArray[np.int_],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    max_iterations: int = _default_max_iterations,
    thickness: int = _default_thickness,
    cmap: ColormapOptions | None = _default_divergence_times_colormap,
    show: bool = False,
    off_screen: bool = False,
) -> pv.Plotter:
    """Render projected sphere divergence times boundary.

    Parameters
    ----------
    divergence_times : ndarray of int
        Two-dimensional first-exit counts; see calculate_divergence_times.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    thickness : int, optional
        Positive number of pixels in the boundary dilation radius plus one.
    cmap : str or Colormap, optional
        Color map for the rendered scalar data.
    show : bool, optional
        Open an interactive window when True; default False.
    off_screen : bool, optional
        Construct the PyVista plotter for off-screen rendering.

    Returns
    -------
    pyvista.Plotter
        Plotter retaining the rendered scene. Call close() when finished.

    Notes
    -----
    Boundaries approximate transitions between finite-time pixel labels. They
    do not certify a Julia set. Image edges do not wrap at the sphere seam.
    """
    # Find boundary
    regions = divergence_times == max_iterations
    boundary_mask = _find_regions_boundary(regions)
    boundary_mask = _thicken_boundary_mask(boundary_mask, thickness)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image_on_dynamic_sphere(
        boundary_image, X, Y, Z, cmap=cmap, show=show, off_screen=off_screen
    )


# --- Convergence times plots ---------------------------------------------------


def plot_convergence_times(
    convergence_times: NDArray[np.int_],
    convergence_points: NDArray[np.complex128],
    attractors: NDArray[np.complex128],
    *,
    max_iterations: int = _default_max_iterations,
    listed_colormap: ListedColormap = _default_listed_colormap,
    color_peak_position: float = _default_color_peak_position,
    color_darkness_factor: float = _default_color_darkness_factor,
    ax: Axes | None = None,
    origin: Literal["upper", "lower"] = "lower",
    **imshow_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render convergence times.

    Parameters
    ----------
    convergence_times : ndarray of int
        Two-dimensional first-entry counts; unresolved entries remain capped.
    convergence_points : ndarray of complex128
        Assigned targets; NaN marks unresolved or invalid points.
    attractors : ndarray of complex128
        One-dimensional target list, in color assignment order.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    listed_colormap : ListedColormap, optional
        Palette sampled across the target list.
    color_peak_position : float, optional
        Position of the saturated basin color, strictly between zero and one.
    color_darkness_factor : float, optional
        Initial color intensity multiplier in [0, 1].
    ax : matplotlib.axes.Axes, optional
        Existing axes; sphere surfaces require three-dimensional axes.
    origin : {'upper', 'lower'}, optional
        Location of array row zero in the image.
    **imshow_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.
    """
    convergence_image = _generate_convergence_image(
        convergence_times,
        convergence_points,
        attractors=attractors,
        listed_colormap=listed_colormap,
        color_peak_position=color_peak_position,
        color_darkness_factor=color_darkness_factor,
    )

    fig, ax = _generate_or_load_fig_ax(ax)
    ax.imshow(convergence_image, origin=origin, **imshow_keywords)

    return fig, ax


def plot_convergence_points_boundary(
    convergence_points: NDArray[np.complex128],
    *,
    thickness: int = _default_thickness,
    ax: Axes | None = None,
    **imshow_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render convergence points boundary.

    Parameters
    ----------
    convergence_points : ndarray of complex128
        Assigned targets; NaN marks unresolved or invalid points.
    thickness : int, optional
        Positive number of pixels in the boundary dilation radius plus one.
    ax : matplotlib.axes.Axes, optional
        Existing axes; sphere surfaces require three-dimensional axes.
    **imshow_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.

    Notes
    -----
    Boundaries approximate transitions between finite-time pixel labels. They
    do not certify a Julia set. Image edges do not wrap at the sphere seam.
    """
    # Find boundary
    boundary_mask = _find_regions_boundary(convergence_points)
    boundary_mask = _thicken_boundary_mask(boundary_mask, thickness)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image(boundary_image, ax=ax, **imshow_keywords)


def plot_projected_sphere_convergence_times(
    convergence_times: NDArray[np.int_],
    convergence_points: NDArray[np.complex128],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    attractors: NDArray[np.complex128],
    *,
    max_iterations: int = _default_max_iterations,
    listed_colormap: ListedColormap = _default_listed_colormap,
    color_peak_position: float = _default_color_peak_position,
    color_darkness_factor: float = _default_color_darkness_factor,
    ax: Axes | None = None,
    rcount: int = 400,
    ccount: int = 400,
    linewidth: int = 0,
    antialiased: bool = False,
    shade: bool = False,
    **plot_surface_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render projected sphere convergence times.

    Parameters
    ----------
    convergence_times : ndarray of int
        Two-dimensional first-entry counts; unresolved entries remain capped.
    convergence_points : ndarray of complex128
        Assigned targets; NaN marks unresolved or invalid points.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    attractors : ndarray of complex128
        One-dimensional target list, in color assignment order.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    listed_colormap : ListedColormap, optional
        Palette sampled across the target list.
    color_peak_position : float, optional
        Position of the saturated basin color, strictly between zero and one.
    color_darkness_factor : float, optional
        Initial color intensity multiplier in [0, 1].
    ax : matplotlib.axes.Axes, optional
        Existing axes; sphere surfaces require three-dimensional axes.
    rcount : int, optional
        Maximum surface samples along rows.
    ccount : int, optional
        Maximum surface samples along columns.
    linewidth : int, optional
        Width of surface edges.
    antialiased : bool, optional
        Whether surface edges are antialiased.
    shade : bool, optional
        Whether lighting modifies the supplied surface colors.
    **plot_surface_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.
    """
    convergence_image = _generate_convergence_image(
        convergence_times,
        convergence_points,
        attractors=attractors,
        listed_colormap=listed_colormap,
        color_peak_position=color_peak_position,
        color_darkness_factor=color_darkness_factor,
    )

    # Draw sphere
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
    else:
        fig = cast(Figure, ax.get_figure())
        if not isinstance(fig, Figure):
            raise ValueError("ax must belong to a Figure.")
        if not hasattr(ax, "plot_surface"):
            raise ValueError("ax must be a three-dimensional axes.")
    surface_ax = cast(Axes3D, ax)
    surface_ax.plot_surface(
        X,
        Y,
        Z,
        facecolors=convergence_image,
        rcount=rcount,
        ccount=ccount,
        linewidth=linewidth,
        antialiased=antialiased,
        shade=shade,  # don't let matplotlib relight the colors
        **plot_surface_keywords,
    )
    surface_ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    return fig, ax


def plot_projected_sphere_convergence_points_boundary(
    convergence_points: NDArray[np.complex128],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    thickness: int = _default_thickness,
    **plot_surface_keywords: Any,
) -> tuple[Figure, Axes]:
    """Render projected sphere convergence points boundary.

    Parameters
    ----------
    convergence_points : ndarray of complex128
        Assigned targets; NaN marks unresolved or invalid points.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    thickness : int, optional
        Positive number of pixels in the boundary dilation radius plus one.
    **plot_surface_keywords
        Additional keyword arguments passed to the Matplotlib rendering method.

    Returns
    -------
    figure : matplotlib.figure.Figure
        Figure containing the plot.
    axes : matplotlib.axes.Axes
        Axes containing the artists; inspect axes.images or axes.collections.

    Notes
    -----
    Boundaries approximate transitions between finite-time pixel labels. They
    do not certify a Julia set. Image edges do not wrap at the sphere seam.
    """
    # Find boundary
    boundary_mask = _find_regions_boundary(convergence_points)
    boundary_mask = _thicken_boundary_mask(boundary_mask, thickness)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image_on_sphere(
        boundary_image, X, Y, Z, **plot_surface_keywords
    )


def dynamic_plot_projected_sphere_convergence_times(
    convergence_times: NDArray[np.int_],
    convergence_points: NDArray[np.complex128],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    attractors: NDArray[np.complex128],
    *,
    max_iterations: int = _default_max_iterations,
    listed_colormap: ListedColormap = _default_listed_colormap,
    color_peak_position: float = _default_color_peak_position,
    color_darkness_factor: float = _default_color_darkness_factor,
    show: bool = False,
    off_screen: bool = False,
) -> pv.Plotter:
    """Render projected sphere convergence times.

    Parameters
    ----------
    convergence_times : ndarray of int
        Two-dimensional first-entry counts; unresolved entries remain capped.
    convergence_points : ndarray of complex128
        Assigned targets; NaN marks unresolved or invalid points.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    attractors : ndarray of complex128
        One-dimensional target list, in color assignment order.
    max_iterations : int, optional
        Iteration cap used to calculate the data, not a divergence certificate.
    listed_colormap : ListedColormap, optional
        Palette sampled across the target list.
    color_peak_position : float, optional
        Position of the saturated basin color, strictly between zero and one.
    color_darkness_factor : float, optional
        Initial color intensity multiplier in [0, 1].
    show : bool, optional
        Open an interactive window when True; default False.
    off_screen : bool, optional
        Construct the PyVista plotter for off-screen rendering.

    Returns
    -------
    pyvista.Plotter
        Plotter retaining the rendered scene. Call close() when finished.
    """
    import pyvista as pv

    convergence_image = _generate_convergence_image(
        convergence_times,
        convergence_points,
        attractors=attractors,
        listed_colormap=listed_colormap,
        color_peak_position=color_peak_position,
        color_darkness_factor=color_darkness_factor,
    )

    grid = pv.StructuredGrid(X, Y, Z)
    # StructuredGrid flattens points in Fortran order internally
    # VTK's direct RGBA colour mode expects 8-bit colour channels.  Passing
    # normalized floats makes an alpha of 1.0 behave like 1/255, so the rear
    # side of the sphere incorrectly shows through the front.
    rgba = np.rint(convergence_image.reshape(-1, 4, order="F") * 255).astype(np.uint8)
    grid.point_data["colors_rgba"] = rgba

    plotter = pv.Plotter(off_screen=off_screen)
    sphere = pv.Sphere(radius=0.99)
    plotter.add_mesh(sphere, color="white", opacity=1.0, smooth_shading=True)
    plotter.add_mesh(grid, scalars="colors_rgba", rgba=True)
    if show:
        plotter.show()

    return plotter


def dynamic_plot_projected_sphere_convergence_points_boundary(
    convergence_points: NDArray[np.complex128],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    thickness: int = _default_thickness,
    cmap: ColormapOptions | None = _default_divergence_times_colormap,
    show: bool = False,
    off_screen: bool = False,
) -> pv.Plotter:
    """Render projected sphere convergence points boundary.

    Parameters
    ----------
    convergence_points : ndarray of complex128
        Assigned targets; NaN marks unresolved or invalid points.
    X : ndarray of float64
        Unit-sphere x coordinates, matching the data grid.
    Y : ndarray of float64
        Unit-sphere y coordinates, matching the data grid.
    Z : ndarray of float64
        Unit-sphere z coordinates, matching the data grid.
    thickness : int, optional
        Positive number of pixels in the boundary dilation radius plus one.
    cmap : str or Colormap, optional
        Color map for the rendered scalar data.
    show : bool, optional
        Open an interactive window when True; default False.
    off_screen : bool, optional
        Construct the PyVista plotter for off-screen rendering.

    Returns
    -------
    pyvista.Plotter
        Plotter retaining the rendered scene. Call close() when finished.

    Notes
    -----
    Boundaries approximate transitions between finite-time pixel labels. They
    do not certify a Julia set. Image edges do not wrap at the sphere seam.
    """
    # Find boundary
    boundary_mask = _find_regions_boundary(convergence_points)
    boundary_mask = _thicken_boundary_mask(boundary_mask, thickness)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image_on_dynamic_sphere(
        boundary_image, X, Y, Z, cmap=cmap, show=show, off_screen=off_screen
    )
