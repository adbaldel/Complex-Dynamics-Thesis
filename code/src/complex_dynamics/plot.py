from typing import Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, ListedColormap
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pyvista.plotting._typing import ColorLike, ColormapOptions

import complex_dynamics.calculator as dcalc

_default_orbit_line_color = "gray"
_default_zi_marker = "o"
_default_zi_color = "black"
_default_z0_color = "green"
_default_max_iterations = dcalc._default_max_iterations
_default_thickness = 1
_default_listed_colormap = mpl.colormaps["Set1"]
_default_divergence_times_colormap = mpl.colormaps["binary"]
_default_projected_sphere_divergence_times_colormap = mpl.colormaps["magma_r"]
_default_color_darkness_factor = 0.7
_default_color_peak_position = 0.2


def _generate_or_load_fig_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    """
    If the given axes exists returns its figure and itself. Otherwise, it
    returns a figure with one default axes.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()
    else:
        fig = ax.get_figure()

    return fig, ax


def _generate_arc_paths(
    points: NDArray[np.float64], *, resolution: int = 30
) -> pv.PolyData:
    """
    Builds a polyline that follows great-circle arcs (on the unit sphere)
    between consecutive orbit points, rather than straight chords through
    the sphere's interior.
    """
    arcs = [
        pv.CircularArc(
            pointa=points[i],
            pointb=points[i + 1],
            center=(0.0, 0.0, 0.0),
            resolution=resolution,
        )
        for i in range(len(points) - 1)
    ]
    return arcs[0].merge(arcs[1:]) if len(arcs) > 1 else arcs[0]


def _find_regions_boundary(regions: NDArray) -> NDArray[np.bool]:
    """
    Given a divergence times mesh, it returns a mask for the mesh that is true
    on boundary points between divergent and convergent points, and false
    everywhere else.
    """
    # Find boundaries by comparing each pixel to its neighbors.
    up = np.roll(regions, shift=-1, axis=0)
    down = np.roll(regions, shift=1, axis=0)
    left = np.roll(regions, shift=-1, axis=1)
    right = np.roll(regions, shift=1, axis=1)

    boundary_mask = (
        (regions != up)
        | (regions != down)
        | (regions != left)
        | (regions != right)
    )

    # Remove edge artifacts
    boundary_mask[0, :] = False
    boundary_mask[-1, :] = False
    boundary_mask[:, 0] = False
    boundary_mask[:, -1] = False

    return boundary_mask


def _thicken_boundary_mask(
    boundary_mask: NDArray[np.bool], thickness: int
) -> NDArray[np.bool]:
    """
    Given a boundary mask, it thickens that boundary mask by the number of
    points specified. It also returns that same boundary mask modified.
    """
    # We iteratively expand the mask in all 4 directions based on the thickness
    # parameter
    for _ in range(thickness - 1):
        boundary_mask = (
            boundary_mask
            | np.roll(boundary_mask, 1, axis=0)
            | np.roll(boundary_mask, -1, axis=0)
            | np.roll(boundary_mask, 1, axis=1)
            | np.roll(boundary_mask, -1, axis=1)
        )

    return boundary_mask


def _generate_boundary_image(
    boundary_mask: NDArray[np.bool],
) -> NDArray[np.float64]:
    """
    Generates a boundary image from a boundary mask.
    """
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
    **imshow_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the given boundary image with the default style.
    """
    fig, ax = _generate_or_load_fig_ax(ax)
    im = ax.imshow(boundary_image, origin=origin, cmap=cmap, **imshow_keywords)

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
    **plot_surface_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the given boundary image on the given sphere with the default style.
    """
    # Apply color to boundary_image for painting the surface of the sphere
    cmap = _default_divergence_times_colormap
    facecolors = cmap(boundary_image)

    # Draw sphere
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
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
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    return fig, ax


def _plot_boundary_image_on_dynamic_sphere(
    boundary_image: NDArray[np.float64],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    cmap: ColormapOptions | None = _default_divergence_times_colormap,
) -> pv.Plotter:
    """
    Plots the given boundary image on a dynamic sphere with the given
    coordinates with the default style.
    """
    grid = pv.StructuredGrid(X, Y, Z)
    # StructuredGrid flattens points in Fortran order internally
    grid.point_data["divergence"] = boundary_image.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="divergence",
        cmap=cmap,
        clim=[0, 1],
        smooth_shading=True,
    )
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
    """
    Generates an image assinging a different color to each attractor and shading
    between a dark version of the color and white depending on how many
    iterations it took for each point to converge to their respective attractor.
    For points that do not converge to any attractor it assigns to all of them
    the same color.
    """
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
    for attractor, color in zip(extended_attractors, color_list):
        attractor_mask = convergence_points == attractor

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
                (1.0, 1.0, 1.0, 1.0),
            ]
            attractor_cmap = mpl.colors.LinearSegmentedColormap.from_list(
                "mycmap",
                list(zip(nodes, colors)),
            )

            # Apply color map
            normalized_convergence_times = convergence_times[
                attractor_mask
            ] / np.max(convergence_times[attractor_mask])
            convergence_image[attractor_mask] = attractor_cmap(
                normalized_convergence_times
            )

    return convergence_image


def _generate_color_list(
    attractors: NDArray[np.complex128],
    all_converge: bool,
    listed_colormap: mpl.colors.ListedColormap,
) -> list[NDArray[np.float64]]:
    """
    Returns a color list based on the given color map with the number of colors
    needed for plotting the convergence rates associated with the convergence
    points given.
    """
    color_list = [
        np.array(color) for color in listed_colormap.colors[: len(attractors)]
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
    **plot_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the orbit of a point with the default style. Returns the figure and
    axes.
    """
    real_parts = np.real(orbit)
    imag_parts = np.imag(orbit)

    fig, ax = _generate_or_load_fig_ax(ax)
    (line,) = ax.plot(
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
) -> pv.Plotter:
    """
    Plots the orbit of a point with the default style on a dynamic sphere.
    Returns the pyvista Plotter.
    """
    X, Y, Z = dcalc.complex_to_sphere(orbit)
    points = np.column_stack([X, Y, Z])

    plotter = pv.Plotter()

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
    **plot_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the orbits of an array of points with the default style. Returns the
    figure and axes.
    """
    fig, ax = _generate_or_load_fig_ax(ax)

    for i in range(len(orbits[0])):
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
    **imshow_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the divergence times mesh in the default style. Returns the figure and
    axes.

    In order to know if a point in the mesh has diverged, it needs to know the
    number of iterations used to calculate the divergence times. This can be set
    using the keyword argument max_iterations.
    """
    # Set a pixel to max_iterations to ensure the color map knows the maximum
    divergence_times[0, 0] = max_iterations

    # Show image
    fig, ax = _generate_or_load_fig_ax(ax)
    im = ax.imshow(
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
    **imshow_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the boundary of the divergence times mesh in the default style. The
    thickness of the boundary is modifiable with the thickness keyword argument.
    Returns the figure and axes.

    In order to know if a point in the mesh has diverged, it needs to know the
    number of iterations used to calculate the divergence times. This can be set
    using the keyword argument max_iterations.
    """
    # Find boundary
    regions = divergence_times == max_iterations
    boundary_mask = _find_regions_boundary(regions)
    if thickness > 1:
        _thicken_boundary_mask(boundary_mask, thickness - 1)

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
    **plot_surface_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the projected divergence times mesh in a sphere with the default
    style. Returns the figure and axes.

    In order to know if a point in the mesh has diverged, it needs to know the
    number of iterations used to calculate the divergence times. This can be set
    using the keyword argument max_iterations.
    """
    # Normalize colors for applying in the surface of the sphere
    cmap = _default_projected_sphere_divergence_times_colormap
    facecolors = cmap(divergence_times / max_iterations)

    # Draw sphere
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
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
    ax.set_box_aspect([1, 1, 1])
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
    **plot_surface_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the boundary of the projected divergence times mesh in a sphere with
    the default style. The thickness of the boundary is modifiable with the
    thickness keyword argument. Returns the figure and axes.

    In order to know if a point in the mesh has diverged, it needs to know the
    number of iterations used to calculate the divergence times. This can be set
    using the keyword argument max_iterations.
    """
    # Find boundary
    regions = divergence_times == max_iterations
    boundary_mask = _find_regions_boundary(regions)
    if thickness > 1:
        _thicken_boundary_mask(boundary_mask, thickness - 1)

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
) -> pv.Plotter:
    """
    Plots the projected divergence times mesh in a pyvista sphere with the
    default style. Returns the pyvista Plotter.

    In order to know if a point in the mesh has diverged, it needs to know the
    number of iterations used to calculate the divergence times. This can be set
    using the keyword argument max_iterations.
    """
    grid = pv.StructuredGrid(X, Y, Z)
    # StructuredGrid flattens points in Fortran order internally
    grid.point_data["divergence"] = divergence_times.flatten(order="F")

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        scalars="divergence",
        cmap=cmap,
        clim=[0, max_iterations],
        smooth_shading=True,
    )
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
) -> pv.Plotter:
    """
    Plots the boundary of the projected divergence times mesh in a pyvista
    sphere with the default style. The thickness of the boundary is modifiable
    with the thickness keyword argument. Returns the pyvista Plotter.

    In order to know if a point in the mesh has diverged, it needs to know the
    number of iterations used to calculate the divergence times. This can be set
    using the keyword argument max_iterations.
    """
    # Find boundary
    regions = divergence_times == max_iterations
    boundary_mask = _find_regions_boundary(regions)
    if thickness > 1:
        _thicken_boundary_mask(boundary_mask, thickness - 1)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image_on_dynamic_sphere(
        boundary_image, X, Y, Z, cmap=cmap
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
    **imshow_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the convergence times mesh in the default style. Returns the figure
    and axes.

    It needs to know to which attractor each point has converged.
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
    im = ax.imshow(convergence_image, origin=origin, **imshow_keywords)

    return fig, ax


def plot_convergence_points_boundary(
    convergence_points: NDArray[np.complex128],
    *,
    thickness: int = _default_thickness,
    ax: Axes | None = None,
    **imshow_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the boundary of the convergence points mesh in the default style. The
    thickness of the boundary is modifiable with the thickness keyword argument.
    Returns the figure and axes.
    """
    # Find boundary
    boundary_mask = _find_regions_boundary(convergence_points)
    if thickness > 1:
        _thicken_boundary_mask(boundary_mask, thickness - 1)

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
    **plot_surface_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the projected convergence times mesh in a sphere with the default
    style. Returns the figure and axes.
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
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(
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
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    return fig, ax


def plot_projected_sphere_convergence_points_boundary(
    convergence_points: NDArray[np.complex128],
    X: NDArray[np.float64],
    Y: NDArray[np.float64],
    Z: NDArray[np.float64],
    *,
    thickness: int = _default_thickness,
    **plot_surface_keywords,
) -> tuple[Figure, Axes]:
    """
    Plots the boundary of the projected convergence points mesh in a sphere with
    the default style. The thickness of the boundary is modifiable with the
    thickness keyword argument. Returns the figure and axes.
    """
    # Find boundary
    boundary_mask = _find_regions_boundary(convergence_points)
    if thickness > 1:
        _thicken_boundary_mask(boundary_mask, thickness - 1)

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
) -> pv.Plotter:
    """
    Plots the projected convergence times mesh in a pyvista sphere with the
    default style. Returns the pyvista Plotter.
    """
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
    rgba = np.rint(convergence_image.reshape(-1, 4, order="F") * 255).astype(
        np.uint8
    )
    grid.point_data["colors_rgba"] = rgba

    plotter = pv.Plotter()
    sphere = pv.Sphere(radius=0.99)
    plotter.add_mesh(sphere, color="white", opacity=1.0, smooth_shading=True)
    plotter.add_mesh(grid, scalars="colors_rgba", rgba=True)
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
) -> pv.Plotter:
    """
    Plots the boundary of the projected convergence points mesh in a pyvista
    sphere with the default style. The thickness of the boundary is modifiable
    with the thickness keyword argument. Returns the pyvista Plotter.
    """
    # Find boundary
    boundary_mask = _find_regions_boundary(convergence_points)
    if thickness > 1:
        _thicken_boundary_mask(boundary_mask, thickness - 1)

    # Generate boundary image
    boundary_image = _generate_boundary_image(boundary_mask)

    # Call plot divergence times function with the fake data
    return _plot_boundary_image_on_dynamic_sphere(
        boundary_image, X, Y, Z, cmap=cmap
    )
