# Neural-network lesson scope

Keep this explainer focused on one scalar input, one hidden layer of ReLU
neurons, and one linear output. An affine function may introduce the setup.

Preserve the three-part sequence:

1. Functions produced by a shallow ReLU network with a univariate input.
2. Hidden-layer width and the number of joints. A width of m permits at most
   m distinct joints and m + 1 linear pieces; coincidences, zero weights,
   cancellation, and inactive units can reduce the realized count.
3. Universal approximation by widening the same shallow architecture, using
   a piecewise linear sine approximation and local joint insertion.

Scrolling drives the mathematical changes in all three parts. Preserve the
course design guide, blacksmith title illustration, and the readable mobile
and reduced-motion versions.
