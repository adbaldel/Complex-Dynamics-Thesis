import importlib.util
import os
import subprocess
import sys

import pytest


def test_interactive_geometry_and_no_implicit_show():
    if importlib.util.find_spec("pyvista") is None:
        pytest.skip("optional interactive dependency is not installed")
    # Isolate VTK and its rendering context from the 2-D import tests.
    code = """
import numpy as np
import pyvista as pv
from complex_dynamics import create_projected_sphere_mesh
from complex_dynamics import plot

def unexpected_show(*args, **kwargs):
    raise AssertionError("A plotting function displayed implicitly")
pv.Plotter.show = unexpected_show
_, x, y, z = create_projected_sphere_mesh(theta_pixels=5, phi_pixels=5)
counts = np.zeros((5, 5), dtype=int)
points = np.zeros((5, 5), dtype=complex)
for orbit in ([1], [1, 1], [0, np.inf], [0, 1, np.inf]):
    scene = plot.plot_orbit_on_dynamic_sphere(np.asarray(orbit, dtype=complex), off_screen=True)
    scene.close()
for fn in (plot.dynamic_plot_projected_sphere_divergence_times, plot.dynamic_plot_projected_sphere_divergence_times_boundary):
    scene = fn(counts, x, y, z, off_screen=True)
    scene.close()
scene = plot.dynamic_plot_projected_sphere_convergence_times(counts, points, x, y, z, np.array([0]), off_screen=True)
scene.close()
scene = plot.dynamic_plot_projected_sphere_convergence_points_boundary(points, x, y, z, off_screen=True)
scene.close()
arc = plot._generate_arc_paths(np.array([[0., 0., -1.], [0., 0., 1.]]))
np.testing.assert_allclose(np.linalg.norm(arc.points, axis=1), 1, atol=1e-7)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, "MPLBACKEND": "Agg"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
