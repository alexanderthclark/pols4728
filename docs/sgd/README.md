# Training neural networks

A scroll-driven POLS 4728 lesson. It constructs a step approximation from two separate ReLUs, transforms their input weights and hidden biases, flips one output contribution, adds them, and applies the shared output bias. It then uses the same step target throughout training and comparison.

The experiment uses 2,048 evenly spaced noiseless observations and plain minibatch SGD. Each epoch shuffles all observations into 32 batches of 64. Both shallow widths (2 and 12) train for 300 epochs with matched batch order and learning-rate schedule. Seed 1 is replayed for each width; all eight fixed seeds appear in the results table. The first epoch shows every update, then later epochs accelerate through checkpoints. Animation interpolates only between actual before/after parameters at a recorded update.

`model.mjs` defines the model, backpropagation, deterministic data, shuffle, and SGD. `experiment.json` records reproducible traces and results. `construction.mjs` renders the two-neuron construction; `views.mjs` renders training and comparison. `scenes.mjs` supplies the narrative; `build-page.mjs` generates the static reading route with local KaTeX.

```sh
node docs/sgd/generate.mjs
node docs/sgd/check.mjs
node docs/sgd/build-page.mjs
```

Serve `docs/` over HTTP. Review forward and reverse scrolling, keyboard steps, narrow-screen rendering, and reduced motion. Scroll is the complete route through the lesson. The static reading route also works without JavaScript.

The step construction converges pointwise and in integrated absolute error, not uniformly across the jump. The standard continuous-target UAT is discussed only in the methods qualification; no unrelated target function is introduced. Extra width helps these particular training runs; width 2 already has the capacity to represent a sharp ramp.
