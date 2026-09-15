# Mathematical conventions

For a reduced rational map $R=P/Q$, its degree is
$d=\max(\deg P,\deg Q)$. Trailing zero coefficients do not contribute to degree.
The value at infinity is zero if $\deg P<\deg Q$, infinity if $\deg P>\deg Q$,
and the ratio of leading coefficients otherwise. Common factors must be
cancelled before using these rules. The zero map is degree zero and has every
point as a zero, so its `roots()` method raises rather than returning a finite list.

For degree $d\ge2$, the Fatou set $F(R)$ is the open set where the family of
iterates $\{R^n\}_{n\ge0}$ is normal in the spherical topology. A Fatou component
is a connected component of this open set. The Julia set is
$J(R)=\widehat{\mathbb C}\setminus F(R)$. Attracting basins are only part of the
Fatou set: parabolic basins, Siegel disks, and Herman rings also occur. Pixel
boundaries of computed basins or first-exit regions are not definitions of $J(R)$.

For finite $z=x+iy$, stereographic coordinates on the unit sphere are

$$S(z)=\frac{(2x,2y,|z|^2-1)}{1+|z|^2},\qquad S(\infty)=(0,0,1).$$

The inverse is $(X+iY)/(1-Z)$, using $(1+Z)/(X-iY)$ in the northern hemisphere
to avoid cancellation. Coordinates are checked against unit norm with relative
and absolute tolerances $10^{-12}$. NaNs propagate. At huge finite magnitudes,
the third coordinate may round to 1; the small transverse coordinates still
retain information needed for the inverse projection.

The package's chordal metric is Euclidean distance on the unit sphere:

$$d_c(z,w)=\frac{2|z-w|}{\sqrt{1+|z|^2}\sqrt{1+|w|^2}},\qquad
 d_c(z,\infty)=\frac{2}{\sqrt{1+|z|^2}}.$$

Its diameter is 2. The normalized convention $\chi=d_c/2$ has diameter 1.
The spherical geodesic distance is $d_s=2\arcsin(d_c/2)$, in $[0,\pi]$.
Implementation uses $2\operatorname{atan2}(\|S(z)-S(w)\|,\|S(z)+S(w)\|)$,
which is more accurate near antipodes. Broadcasting is elementwise.

An attracting fixed point has multiplier of modulus below 1 in a local
coordinate. At infinity, use $w=1/z$ and the local map $1/R(1/w)$ for a fixed
infinity; the affine derivative `R.derivative()` alone does not give that
multiplier. The software does not currently certify attractors or periodic cycles.

For $z^2+c$, $r=\max(2,|c|)$ is a sufficient escape bound once $|z|>r$.
Equality must be treated carefully: $z=2$ is fixed for $z^2-2$. The package uses
an open-disk **first-exit** convention and does not infer polynomial divergence
from equality at an arbitrary radius.

Background: A. F. Beardon, *Iteration of Rational Functions*, Springer, 1991;
R. L. Devaney, *An Introduction to Chaotic Dynamical Systems*.
