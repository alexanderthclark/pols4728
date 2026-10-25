# Neural-network lesson scope

Keep the original construction focused on one scalar input, one hidden layer
of ReLU neurons, and one linear output. An affine function introduces the setup.
The instructor-authorized extension follows it with sigmoid classification and
a binary, two-input, two-ReLU-hidden-neuron learned representation.

Preserve the five-part sequence:

1. Functions produced by a shallow ReLU network with a univariate input.
2. Hidden-layer width and the number of joints. A width of m permits at most
   m distinct joints and m + 1 linear pieces; coincidences, zero weights,
   cancellation, and inactive units can reduce the realized count.
3. Universal approximation by widening the same shallow architecture, using
   a piecewise linear sine approximation and local joint insertion.
4. Beck, King, and Zeng (2000), Equation 3: sigmoid hidden activations and a
   sigmoid classification head. Open with a centered equation in the paper's
   notation, using the function name sigmoid and no diagrams. Briefly state
   that BKZ train a network with sigmoid activations. Next compare that model
   with logistic regression side by side, without plots or a nested-sigmoid
   tangent comparison. Then show how additional hidden neurons permit more
   complicated shapes using fixed one-input constructions in the course notation.
5. A fitted binary network: move the same observations into their actual two
   hidden activations before revealing the fitted head's straight boundary.

The binary example is a documented new fit inspired by the notes' adjacent
curved classes, not the unavailable original three-class fit. Use its frozen
fitted weights consistently in both spaces. Never substitute a different
classifier, arbitrary point destinations, or a projection of a wider hidden
layer. Distinguish perfect training classification from generalization and
from exact probability estimates. The sigmoid constructions are not learned.

Scrolling drives the mathematical changes in all five parts. Preserve the
course design guide, blacksmith title illustration, and the readable mobile
and reduced-motion versions.

Keep the related diagrams together in one pinned visual stage with a
scene-specific heading, equation, and concise caption. Do not add a separate
explanation column. Keep the bias caption synchronized with the bias node's
removal and explain that omitting intercepts from the drawing does not change
the model. Retain the full prose as the print and no-JavaScript reading route.
Use the local LaTeX renderer for stage equations, with compact line breaks
where needed. Typeset new mathematical plot and network labels with local
KaTeX, including axis labels and parameter subscripts. Captions should add
information rather than repeat labels.

Keep the input-x slider in the original univariate ReLU scenes. Hide it in the
classification extension, whose complete story is controlled by scrolling.
The boundary must remain hidden for the entire representation-movement scene,
including under reduced motion and when scrolling backward from its reveal.
