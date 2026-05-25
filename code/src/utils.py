import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class DynamicsPlotter:
    """
    A unified class for plotting complex dynamics structures such as orbits,
    Julia sets, and Fatou basins.
    
    This class maintains a consistent coordinate system (window) for the complex 
    plane, ensuring that boundaries align perfectly with generated arrays from 
    the dynamics module.
    
    Attributes:
        x_min (float): The minimum real coordinate.
        x_max (float): The maximum real coordinate.
        y_min (float): The minimum imaginary coordinate.
        y_max (float): The maximum imaginary coordinate.
        figsize (tuple): Tuple specifying the figure dimensions.
        use_latex (bool): Whether to use LaTeX for text rendering.
    """
    
    def __init__(self, x_min=-2.0, x_max=2.0, y_min=-2.0, y_max=2.0, figsize=(6, 6), use_latex=False):
        """
        Initializes the plotting window for the complex plane.

        Args:
            x_min (float): Minimum real value.
            x_max (float): Maximum real value.
            y_min (float): Minimum imaginary value.
            y_max (float): Maximum imaginary value.
            figsize (tuple): Figure size for matplotlib.
            use_latex (bool): If True, uses LaTeX engine for text rendering (slower but professional).
                              If False, uses standard matplotlib text rendering (faster for previews).
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.figsize = figsize
        self.use_latex = use_latex
        self.extent = [self.x_min, self.x_max, self.y_min, self.y_max]

        self._setup_latex_style()

    def _setup_latex_style(self) -> None:
        """Configures matplotlib fonts and styling, optionally using LaTeX."""
        params = {
            "axes.labelsize": 12,           # Standard size for thesis
            "font.size": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.grid": False,             # Grid off by default, enabled specifically for orbits
            "grid.alpha": 0.4,
            "grid.linestyle": "--",
            "grid.color": "gray"
        }
        
        if self.use_latex:
            params.update({
                "text.usetex": True,            # Use LaTeX to write all text
                "font.family": "serif",         # Use serif fonts
                "font.serif": ["Computer Modern Roman"],
            })
        else:
            params.update({
                "text.usetex": False,           # Use standard matplotlib text rendering
                "font.family": "sans-serif",    # Standard UI-friendly font
            })
            
        plt.rcParams.update(params)

    def _setup_figure(self, title=None, show_axis=True, show_axis_labels=True, show_grid=False):
        """Internal helper to setup the figure, axes labels, and title."""
        fig, ax = plt.subplots(figsize=self.figsize)
        if title:
            ax.set_title(title, fontsize=14)
            
        if show_axis:
            if show_axis_labels:
                ax.set_xlabel(r"$\mathrm{Re}(z)$")
                ax.set_ylabel(r"$\mathrm{Im}(z)$")
            if show_grid:
                ax.grid(True)
        else:
            ax.axis('off')
            
        return fig, ax
    
    def add_points(self, ax, points, label=None, marker='.', color='black', s=50, zorder=10, annotate_label=True, text_offset=(0, -10), fontsize=16, **kwargs):
        """
        Auxiliary method to overlay complex points on an existing axis, 
        with options for standard legends or direct plot annotation.
        
        Args:
            ax (matplotlib.axes.Axes): The axis to plot on.
            points (list or np.ndarray): Complex numbers to plot.
            label (str, optional): The text label for the points.
            marker (str): Matplotlib marker style.
            color (str): Matplotlib color.
            s (int): Marker size.
            zorder (int): Drawing order (higher is on top).
            annotate_label (bool): If True, places the label text directly next to the point on the plot.
                                   If False, adds the label to a standard legend box.
            text_offset (tuple): The (x, y) offset in points (pixels) for the annotation. 
                                 Default (0, -12) places it directly below.
            fontsize (int): Font size for the annotated text.
            **kwargs: Additional arguments passed to ax.scatter.
        """
        r_real = [np.real(p) for p in points]
        r_imag = [np.imag(p) for p in points]
        
        # Plot the marker
        ax.scatter(r_real, r_imag, color=color, marker=marker, s=s, label=label if not annotate_label else None, zorder=zorder, **kwargs)
        
        if label:
            if annotate_label:
                # Place the text directly on the plot for each point
                for x, y in zip(r_real, r_imag):
                    ax.annotate(
                        label, 
                        (x, y), 
                        xytext=text_offset, 
                        textcoords='offset points', 
                        ha='center',     # Horizontally center the text under the point
                        va='top',        # Vertically align the top of the text with the offset
                        color=color, 
                        fontsize=fontsize, 
                        zorder=zorder + 1
                    )
            else:
                # Fallback to standard matplotlib legend behavior
                ax.legend()
        
        return ax

    def plot_orbit(self, orbit_data, connect_lines=True, marker='o', color='black', title="Orbit", show_axis=True, show_axis_labels=False, show_grid=True):
        """
        Plots the forward orbit of a point on the complex plane.

        Args:
            orbit_data (np.ndarray): 1D array of complex numbers representing the orbit.
            connect_lines (bool): If True, draws lines between successive points.
            marker (str): Matplotlib marker style for the orbit points.
            color (str): Color of the markers and lines.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.
            show_grid (bool): If True, displays a background grid.
            
        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid)
        
        real_parts = np.real(orbit_data)
        imag_parts = np.imag(orbit_data)
        
        if connect_lines:
            ax.plot(real_parts, imag_parts, color=color, alpha=0.5, linestyle='-', linewidth=1)
            
        ax.scatter(real_parts, imag_parts, color=color, marker=marker, zorder=5)
        
        ax.set_xlim(self.x_min, self.x_max)
        ax.set_ylim(self.y_min, self.y_max)
        
        plt.tight_layout()
        return fig, ax

    def plot_escape_time_fractal(self, escape_data, cmap='binary', title="Escape Time Fractal", show_axis=True, show_axis_labels=False, show_colorbar=False, colorbar_label="Iterations to escape"):
        """
        Plots an escape time fractal (e.g., a Julia set) as a 2D image.

        Args:
            escape_data (np.ndarray): 2D integer array containing escape times.
            cmap (str): Matplotlib colormap string.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.
            show_colorbar (bool): If True, displays a colorbar.
            colorbar_label (str): Label for the colorbar (if shown).

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        # origin='lower' ensures y_min is at the bottom, y_max at the top
        im = ax.imshow(escape_data, extent=self.extent, origin='lower', cmap=cmap)
        
        if show_colorbar:
            fig.colorbar(im, ax=ax, label=colorbar_label)
        
        plt.tight_layout()
        return fig, ax

    def plot_julia_set_from_escape_times(self, escape_data, max_iter, color='black', thickness=1, title="Julia Set (Escape Boundary)", show_axis=True, show_axis_labels=False):
        """
        Extracts and plots the Julia set from escape time data (typically for polynomials).
        
        For polynomials, the filled Julia set consists of points that do not escape 
        to infinity (i.e., those that reach max_iter). The Julia set is the boundary 
        of this filled set.

        Args:
            escape_data (np.ndarray): 2D integer array containing escape times.
            max_iter (int): The maximum iterations used when generating the escape data.
            color (str): Matplotlib color string for the Julia set.
            thickness (int): Pixel thickness of the Julia Set. Defaults to 1.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        # Points that never escaped are considered "inside" the filled Julia set
        is_inside = (escape_data == max_iter)
        
        # Find boundaries by comparing each pixel to its neighbors.
        up = np.roll(is_inside, shift=-1, axis=0)
        down = np.roll(is_inside, shift=1, axis=0)
        left = np.roll(is_inside, shift=-1, axis=1)
        right = np.roll(is_inside, shift=1, axis=1)
        
        # A point is on the boundary if its inside/outside status differs from any neighbor
        boundary_mask = (is_inside != up) | (is_inside != down) | \
                        (is_inside != left) | (is_inside != right)
        
        # THICKEN THE BOUNDARY (Dilation)
        # We iteratively expand the mask in all 4 directions based on the thickness parameter
        if thickness > 1:
            for _ in range(thickness - 1):
                boundary_mask = boundary_mask | np.roll(boundary_mask, 1, axis=0) | np.roll(boundary_mask, -1, axis=0) | \
                                np.roll(boundary_mask, 1, axis=1) | np.roll(boundary_mask, -1, axis=1)
        
        # Remove edge artifacts
        boundary_mask[0, :] = False
        boundary_mask[-1, :] = False
        boundary_mask[:, 0] = False
        boundary_mask[:, -1] = False
        
        # Create an RGBA image: transparent everywhere, chosen color on boundaries
        import matplotlib.colors as mcolors
        rgba_color = mcolors.to_rgba(color)
        
        img = np.zeros((*escape_data.shape, 4))
        img[boundary_mask] = rgba_color
        
        ax.imshow(img, extent=self.extent, origin='lower')
        
        plt.tight_layout()
        return fig, ax

    def plot_fatou_basins(self, basin_data, cmap='Set1', title="Fatou Basins of Attraction", show_axis=True, show_axis_labels=True):
        """
        Plots the Fatou basins of attraction. 

        Args:
            basin_data (np.ndarray): 2D integer array containing basin indices.
            cmap (str or Colormap): Matplotlib colormap string. Defaults to 'Set1'.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        min_val = int(np.min(basin_data))
        max_val = int(np.max(basin_data))
        
        # Custom logic to ensure qualitative colormaps map sequentially 
        # and don't "stretch" across the min and max values.
        if isinstance(cmap, str):
            try:
                base_cmap = plt.get_cmap(cmap)
                if hasattr(base_cmap, 'colors'):
                    color_list = []
                    # -1 usually means "did not converge". We give it a dark grey background.
                    if min_val < 0:
                        color_list.append((1.0, 1.0, 1.0, 1.0)) 
                        start_idx = 0
                    else:
                        start_idx = min_val
                        
                    # Grab the first N colors sequentially from the colormap
                    for i in range(start_idx, max_val + 1):
                        color_list.append(base_cmap.colors[i % len(base_cmap.colors)])
                        
                    cmap = mcolors.ListedColormap(color_list)
            except Exception:
                pass # Fallback to default behavior if anything fails
        
        # BoundaryNorm guarantees that each integer gets exactly one distinct color bin
        bounds = np.arange(min_val, max_val + 2) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N if hasattr(cmap, 'N') else len(bounds)-1)
        
        # We use a discrete mapping approach via imshow
        ax.imshow(basin_data, extent=self.extent, origin='lower', cmap=cmap, norm=norm)
            
        plt.tight_layout()
        return fig, ax

    def plot_julia_set_from_basins(self, basin_data, color='black', thickness=1, title="Julia Set (Basin Boundaries)", show_axis=True, show_axis_labels=False):
        """
        Extracts and plots the Julia set as the boundary of the Fatou basins.
        
        In complex dynamics, the Julia set is precisely the common boundary 
        of the Fatou basins. This method finds the boundaries numerically 
        by detecting adjacent pixels belonging to different basins.

        Args:
            basin_data (np.ndarray): 2D integer array containing basin indices.
            color (str): Matplotlib color string for the Julia set.
            thickness (int): Pixel thickness of the Julia Set. Defaults to 1.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        # Find boundaries by comparing each pixel to its neighbors.
        # np.roll shifts the array to easily compare adjacent pixels.
        up = np.roll(basin_data, shift=-1, axis=0)
        down = np.roll(basin_data, shift=1, axis=0)
        left = np.roll(basin_data, shift=-1, axis=1)
        right = np.roll(basin_data, shift=1, axis=1)
        
        # A point is on the boundary if its basin index differs from any neighbor
        boundary_mask = (basin_data != up) | (basin_data != down) | \
                        (basin_data != left) | (basin_data != right)
        
        # THICKEN THE BOUNDARY (Dilation)
        # We iteratively expand the mask in all 4 directions based on the thickness parameter
        if thickness > 1:
            for _ in range(thickness - 1):
                boundary_mask = boundary_mask | np.roll(boundary_mask, 1, axis=0) | np.roll(boundary_mask, -1, axis=0) | \
                                np.roll(boundary_mask, 1, axis=1) | np.roll(boundary_mask, -1, axis=1)
        
        # Remove edge artifacts caused by the wrap-around behavior of np.roll
        boundary_mask[0, :] = False
        boundary_mask[-1, :] = False
        boundary_mask[:, 0] = False
        boundary_mask[:, -1] = False
        
        # Create an RGBA image: transparent everywhere, chosen color on boundaries
        import matplotlib.colors as mcolors
        rgba_color = mcolors.to_rgba(color)
        
        # Initialize empty transparent image (height, width, 4 channels for RGBA)
        img = np.zeros((*basin_data.shape, 4))
        # Apply color only to the boundary pixels
        img[boundary_mask] = rgba_color
        
        ax.imshow(img, extent=self.extent, origin='lower')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_combined_fatou_basins_and_julia_set(self, basin_data, cmap='Set1', julia_color='black', julia_thickness=1, title="Fatou Basins and Julia Set", show_axis=True, show_axis_labels=True):
        """
        Plots the Fatou basins and overlays the Julia set (basin boundaries) on a single image.

        Args:
            basin_data (np.ndarray): 2D integer array containing basin indices.
            cmap (str or Colormap): Matplotlib colormap string. Defaults to 'Set1'.
            julia_color (str): Matplotlib color string for the Julia set. Defaults to 'black'.
            julia_thickness (int): Pixel thickness of the Julia Set. Defaults to 1.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        # 1. Setup the single figure and axis
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        # --- Plot Fatou Basins (Background) ---
        min_val = int(np.min(basin_data))
        max_val = int(np.max(basin_data))
        
        if isinstance(cmap, str):
            try:
                base_cmap = plt.get_cmap(cmap)
                if hasattr(base_cmap, 'colors'):
                    color_list = []
                    if min_val < 0:
                        color_list.append((1.0, 1.0, 1.0, 1.0)) 
                        start_idx = 0
                    else:
                        start_idx = min_val
                        
                    for i in range(start_idx, max_val + 1):
                        color_list.append(base_cmap.colors[i % len(base_cmap.colors)])
                        
                    cmap = mcolors.ListedColormap(color_list)
            except Exception:
                pass 
        
        bounds = np.arange(min_val, max_val + 2) - 0.5
        norm = mcolors.BoundaryNorm(bounds, cmap.N if hasattr(cmap, 'N') else len(bounds)-1)
        
        # Plot the basins
        ax.imshow(basin_data, extent=self.extent, origin='lower', cmap=cmap, norm=norm)

        # --- Plot Julia Set (Foreground Overlay) ---
        up = np.roll(basin_data, shift=-1, axis=0)
        down = np.roll(basin_data, shift=1, axis=0)
        left = np.roll(basin_data, shift=-1, axis=1)
        right = np.roll(basin_data, shift=1, axis=1)
        
        boundary_mask = (basin_data != up) | (basin_data != down) | \
                        (basin_data != left) | (basin_data != right)

        # THICKEN THE BOUNDARY (Dilation)
        # We iteratively expand the mask in all 4 directions based on the thickness parameter
        if julia_thickness > 1:
            for _ in range(julia_thickness - 1):
                boundary_mask = boundary_mask | np.roll(boundary_mask, 1, axis=0) | np.roll(boundary_mask, -1, axis=0) | \
                                np.roll(boundary_mask, 1, axis=1) | np.roll(boundary_mask, -1, axis=1)
        
        # Remove edge artifacts
        boundary_mask[0, :] = False
        boundary_mask[-1, :] = False
        boundary_mask[:, 0] = False
        boundary_mask[:, -1] = False
        
        rgba_color = mcolors.to_rgba(julia_color)
        img_julia = np.zeros((*basin_data.shape, 4))
        img_julia[boundary_mask] = rgba_color
        
        # Plot the Julia set on top (the transparency of the 0 array handles the overlay)
        ax.imshow(img_julia, extent=self.extent, origin='lower')
            
        plt.tight_layout()
        return fig, ax

    def plot_fatou_basins_shaded(self, basin_data, times_data, cmap='Set1', shading_power=0.5, dark_start=0.5, color_peak=0.5, title="Shaded Fatou Basins", show_axis=True, show_axis_labels=True):
        """
        Plots Fatou basins shaded dynamically by their rate of convergence.
        Uses a two-phase gradient: Dark -> Pure Color -> White.

        Args:
            basin_data (np.ndarray): 2D integer array containing basin indices.
            times_data (np.ndarray): 2D integer array containing convergence iteration counts.
            cmap (str): Matplotlib colormap string. Defaults to 'Set1'.
            shading_power (float): Curves the overall distribution of the gradient.
            dark_start (float): Brightness multiplier for the fastest converging points (0.0 to 1.0).
            color_peak (float): The normalized time [0.0 to 1.0] where the color reaches 100% purity.
                                e.g., 0.5 means the pure color hits halfway through the gradient.
            title (str, optional): Title of the plot.
            show_axis (bool): If True, shows axes and tick marks.
            show_axis_labels (bool): If True, shows "Re(z)" and "Im(z)" labels.

        Returns:
            fig, ax: The matplotlib figure and axes objects.
        """
        fig, ax = self._setup_figure(title, show_axis, show_axis_labels, show_grid=False)
        
        min_val = int(np.min(basin_data))
        max_val = int(np.max(basin_data))
        
        # 1. Extract base colors for each basin
        base_cmap = plt.get_cmap(cmap)
        color_list = []
        
        if min_val < 0:
            color_list.append(np.array([0.0, 0.0, 0.0]))  # Solid black for non-converging points
            start_idx = 0
        else:
            start_idx = min_val
            
        for i in range(start_idx, max_val + 1):
            color_list.append(np.array(base_cmap.colors[i % len(base_cmap.colors)][:3]))
            
        # 2. Compute the normalized time 't'
        max_time = np.max(times_data)
        if max_time == 0: 
            max_time = 1  
            
        # t scales from 0.0 (fastest) to 1.0 (slowest)
        t = (times_data / max_time) ** shading_power
        
        # 3. Apply colors and shading vectorially (Two-Phase Interpolation)
        img = np.zeros((*basin_data.shape, 3))
        
        for val, color in zip(range(min_val, max_val + 1), color_list):
            basin_mask = (basin_data == val)
            
            if val < 0:
                for channel in range(3):
                    img[basin_mask, channel] = color[channel]
                continue
            
            # Split the basin into two masks based on the color_peak
            phase1_mask = basin_mask & (t <= color_peak)
            phase2_mask = basin_mask & (t > color_peak)
            
            # Local interpolation factors for each phase
            # t1 goes from 0.0 to 1.0 during Phase 1
            t1 = t[phase1_mask] / color_peak if color_peak > 0 else np.zeros_like(t[phase1_mask])
            # t2 goes from 0.0 to 1.0 during Phase 2
            t2 = (t[phase2_mask] - color_peak) / (1.0 - color_peak) if color_peak < 1 else np.zeros_like(t[phase2_mask])
            
            darkened_color = color * dark_start
            
            for channel in range(3):
                # Phase 1: Interpolate from darkened_color to pure color
                img[phase1_mask, channel] = darkened_color[channel] + t1 * (color[channel] - darkened_color[channel])
                
                # Phase 2: Interpolate from pure color to pure white (1.0)
                img[phase2_mask, channel] = color[channel] + t2 * (1.0 - color[channel])
                
        # Clip just to be safe with matplotlib's RGB expectations
        img = np.clip(img, 0, 1)
        
        ax.imshow(img, extent=self.extent, origin='lower')
            
        plt.tight_layout()
        return fig, ax

    def save_figure(self, fig, filename, dpi=300, bbox_inches='tight', **kwargs):
        """
        Saves the matplotlib figure to a file.
        
        This is highly recommended when exporting figures for LaTeX. 
        'tight' bounding box removes unnecessary white space around the image.
        
        Args:
            fig (matplotlib.figure.Figure): The figure object to save.
            filename (str): Output filename/path (e.g., 'report/images/julia.pdf').
            dpi (int): Resolution of the image. 300 is standard for print quality.
            bbox_inches (str): Bounding box style.
            **kwargs: Additional arguments passed to fig.savefig.
        """
        fig.savefig(filename, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
        print(f"Saved figure to: {filename}")