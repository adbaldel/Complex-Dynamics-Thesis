**Use both**, but for completely different parts of the image.

Because a fractal contains a mix of smooth 3D surfaces and infinitely complex pixel-level details, the best file format depends entirely on what your background looks like.

Here is the breakdown for a LaTeX document:

### 1. For the Fractal Sphere Itself: Use PNG (at 600 DPI)

While LaTeX loves PDFs because they are vector-based and scale infinitely without losing quality, a 3D fractal surface plot is a bad candidate for a pure PDF export.

* **The Problem with PDF here:** If you export a $2000 \times 2000$ grid mesh as a PDF, Matplotlib will write every single one of those 4 million tiny 3D polygons into the file as individual mathematical vector shapes. Your PDF file size will balloon to 50MB+ or 100MB+, and when someone tries to open your paper in Adobe Reader or a web browser, the page will visibly stutter, lag, or freeze as it tries to draw millions of shapes in real time.
* **The Fix:** Export the plot as a high-resolution **PNG** at `dpi=600`. At 600 DPI, the human eye cannot see the individual pixels on a printed page or standard monitor, and the file size will remain tiny (a few megabytes), ensuring your paper opens instantly.

### 2. For the Labels, Text, and Layout: Use PDF

If your figure includes math equations, text labels, arrows, or axis lines, you want those to be crystal-clear vectors so they match the rest of your LaTeX fonts perfectly.

### How to get the best of both worlds in LaTeX

The professional way to handle this is to generate a clean, label-free, high-DPI **PNG** of the sphere from Python, and then use LaTeX itself to overlay the vector text and math symbols.

You can do this cleanly using the `overpic` package in your LaTeX document:

```latex
\usepackage{overpic}
\usepackage{xcolor}

\begin{figure}[htbp]
    \centering
    % Load the high-res PNG image
    \begin{overpic}[width=0.7\textwidth]{julia_riemann_sphere.png}
        % Overlay vector LaTeX text directly onto the image coordinates (0-100 scale)
        \put(50, 92){\color{white}\makebox(0,0){\Large $z_{n+1} = z_n^2 + c$}}
        \put(50, 5){\color{white}\makebox(0,0){South Pole ($z=0$)}}
        \put(50, 82){\color{white}\makebox(0,0){North Pole ($z=\infty$)}}
    \end{overpic}
    \caption{Julia set mapped onto the Riemann Sphere using inverse stereographic projection.}
    \label{fig:riemann_julia}
\end{figure}

```

This ensures your fractal renders smoothly without lagging the reader's computer, while your equations and descriptions remain perfectly crisp, infinitely scalable vectors.