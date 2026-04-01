import numpy as np
import matplotlib.pyplot as plt

def iterate(R, z0, max_iter):
    zn = [z0]

    for i in range(max_iter):
        next_z = R(zn[-1])
        zn.append(next_z)
    
    return zn

def plot_sequence_dark(zn, center: complex, zoom):
    zn = np.array(zn)
    
    plt.figure(figsize=(10, 10))
    plt.style.use('dark_background')

    # 1. Use a brighter colormap like 'plasma', 'spring', or 'autumn'
    # 'plasma' is great because it goes from bright purple to yellow
    colors = np.linspace(0, 1, len(zn))
    
    # 2. Add a very thin line connecting the points so you can see the 'flow'
    plt.plot(zn.real, zn.imag, color='white', linewidth=0.5, alpha=0.3)

    # 3. Increase size (s) and remove transparency (alpha)
    plt.scatter(zn.real, zn.imag, c=colors, cmap='viridis', s=30, alpha=1.0, edgecolors='none')

    # Highlight start and end with distinct colors
    plt.scatter(zn.real[0], zn.imag[0], color='white', s=50, label='Start (z0)', zorder=5)
    plt.scatter(zn.real[-1], zn.imag[-1], color='red', s=50, label='End', zorder=5)

    plt.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    plt.axvline(0, color='white', linewidth=0.5, alpha=0.3)
    
    # Labels and Grid
    plt.title(f"Möbius Iterations (Zoom: {zoom})")
    plt.legend(scatterpoints=1)
    plt.grid(alpha=0.1)

    # --- Zoom Settings ---
    plt.xlim(center.real - zoom, center.real + zoom)
    plt.ylim(center.imag - zoom, center.imag + zoom)
    plt.gca().set_aspect('equal', adjustable='box')
    
    plt.show()