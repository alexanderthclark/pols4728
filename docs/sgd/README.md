# Training neural networks

A 26-scene, scroll-driven POLS 4728 lesson. It constructs a step approximation from two separate ReLUs, transforms their input weights and hidden biases, flips one output contribution, adds them, and applies the shared output bias. It then uses the same step target throughout training and comparison.

The main SGD chapter opens with a transition and a diagram partitioning the actual 2,048 observations into 32 disjoint batches of 64. It selects the first batch, follows one observation's prediction and squared-error loss, adds the 64 point losses, and divides by 64 to obtain the mean batch loss. It then differentiates this loss, demonstrates one parameter's gradient step, and applies the update to the whole model. The inactive first neuron and its zero gradients remain explicit.

The next batch uses the learned weights to compute a new loss, gradient, and update. The animation then speeds through batches 3–32, completes the first epoch, reshuffles the same observations, and performs the first update of epoch two. Learned weights remain continuous across these scenes. A lap-style counter at the bottom right shows the current epoch and the number of completed batch updates in that pass; it advances to the next epoch only after all 32 updates are complete.

`batches.html` remains a separate, focused seven-scene introduction to batches and epochs. Its small teaching example groups 12 numbered observations into three batches of four, follows one complete pass, then reshuffles the same observations for another epoch. It ends by connecting these definitions to the actual experiment's 2,048 observations and 32 batches; it does not include the loss and gradient calculations or the training counter.

The experiment uses 2,048 evenly spaced noiseless observations and plain minibatch SGD. Each epoch shuffles all observations into 32 batches of 64. Both shallow widths (2 and 12) train for 300 epochs with matched batch order and learning-rate schedule. Seed 1 is replayed for each width; all eight fixed seeds appear in the results table. The first two epochs are recorded in full so the transition between passes preserves the exact model and batch order. Later training accelerates through recorded checkpoints. Animation interpolates only between actual before/after parameters at a recorded update.

`model.mjs` defines the model, backpropagation, deterministic data, shuffle, and SGD. `experiment.json` records reproducible traces and results. `construction.mjs` renders the two-neuron construction; `batch-epoch.mjs` renders the focused batches-and-epochs introduction; `sgd-detail.mjs` renders the partition and numerical batch walkthrough; `views.mjs` renders training and comparison. `epoch-counter.mjs` derives the counter from completed updates. `scenes.mjs` supplies the narrative and procedure labels; `build-page.mjs` generates both pages and their complete static reading routes with local KaTeX.

```sh
node docs/sgd/generate.mjs
node docs/sgd/check.mjs
node docs/sgd/build-page.mjs
node --test tests/sgd.test.mjs
```

Serve `docs/` over HTTP. Review forward and reverse scrolling, keyboard steps, narrow-screen rendering, and reduced motion. Scroll is the complete route through the lesson. The static reading route also works without JavaScript.

The step construction converges pointwise and in integrated absolute error, not uniformly across the jump. The standard continuous-target UAT is discussed only in the methods qualification; no unrelated target function is introduced. Extra width helps these particular training runs; width 2 already has the capacity to represent a sharp ramp.
