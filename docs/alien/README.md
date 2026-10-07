# How an alien learns basketball

A twelve-stage scrollytelling explanation of expectation, bias, variance, and MSE, based on the Fall 2026 course notes’ Knicks–Spurs alien example. The opening title screen shows a classic alien stick figure scratching its head while watching basketball. Its registered monochrome illustration is transparent. Titles and narrative remain selectable HTML text; quantitative labels remain real SVG text.

## The argument

One alien observes the original five games and fits a decision stump with cutoff −1.5. This fits all training labels but misclassifies a Knicks loss at −1. The story then introduces independently trained aliens and holds their test input fixed at +0.5. Each alien represents a different training dataset passed through the same algorithm.

The twelve stages develop the argument in order:

1. Observe five final scores and define the input and binary outcome.
2. Fit the midpoint cutoff between the closest observed loss and win.
3. Test a new game and compare the fitted cutoff with the true boundary.
4. Compare three independent training samples and their fitted rules.
5. Ask every alien about the same half-point Knicks lead.
6. Repeat training 1,000 times, with one mark for each fitted rule.
7. Gather predictions at zero and one and estimate their expectation.
8. Measure bias from the true outcome to the average prediction.
9. Measure variance from individual predictions to their average.
10. Measure MSE from individual predictions to the true outcome.
11. Decompose that MSE into squared bias, variance, and zero outcome noise.
12. Explore different training-sample sizes, fixed test inputs, and fresh samples.

The finite average estimates expectation over training samples. At +0.5 the original simulation has 532 win predictions and 468 loss predictions: mean 0.532, bias −0.468, squared bias 0.219024, variance 0.248976, and MSE 0.468. Outcome noise is zero conditional on the final score differential.

Let q be the probability a newly trained alien is wrong at the fixed binary input. Squared bias is q², variance is q(1−q), and MSE is q. At a winning input bias is −q; at a losing input it is +q. For q below one half, reducing q reduces both squared bias and variance: predictions become more accurate and more unanimous. The learning rule and model complexity remain fixed. At +0.5 the default sequence for 5, 10, 25, 100, and 500 games gives estimated q values 0.468, 0.406, 0.369, 0.102, and 0.000, so both components fall in that displayed sequence. Finite batches can fluctuate. If q exceeds one half, improving it toward one half initially raises variance while reducing squared bias; this binary-prediction qualification is explained in the explorer.

## The visual argument

One persistent SVG accompanies the narrative. The opening table displays outcomes as plain zeroes and ones. The first fitted cutoff and its comparison with the true boundary share one number line. All observed locations lie on that line; the repeated +1 is labeled “2 games.” A short cutoff tick and midpoint bracket locate −1.5. The new-game comparison adds a short dashed true-boundary tick and a narrow hatched error interval, with no vertical data dimension. Three aligned sample rows then show how training data change the fitted rule; the sample selector gives access to all 1,000 worked samples. The default rows are Aliens 1, 2, and 16. Alien 16 sees five wins, uses cutoff −∞, and remains visible as other samples are selected. Infinite cutoffs use an outward arrow instead of a finite boundary line. A shared vertical test line at +0.5 makes the fixed-input comparison explicit, and arrows end at plain zeroes and ones in a prediction column.

The population contains exactly 1,000 keyed marks, one for each alien in training-sample order. The same marks move from a crowd into prediction piles at zero and one; their identities remain stable through expectation, bias, variance, MSE, and the decomposition. Circles represent win predictions and squares represent loss predictions, alongside numerical labels and counts. A mean marker and true-outcome marker provide the changing reference points. The prediction axis is explicitly distinguished from the earlier score-differential axis.

The decomposition uses exact proportional segment widths on a squared-error scale from zero to one. Squared bias and variance add to the MSE width; outcome noise contributes zero width. Readouts round the worked components to approximately 0.219 and 0.249, while expandable prose retains the exact values.

Phones use compact figure coordinates and shorter displayed formulas, with the substantive explanation and exact values retained in the narrative. The stack column count adapts to the larger prediction group so that all 1,000 marks remain visible even when every alien gives the same answer. The title image supplies character artwork; miniature SVG alien glyphs identify the fitted rules in the sample diagrams.

## Simulation and explorer

`samples.json` contains exactly the five-game samples from the notes. They were generated with NumPy’s legacy `RandomState` using seed 1. In each repetition, draw five Knicks integers first, then five Spurs integers, independently from 0 through 100 inclusive. Add 0.5 to each Knicks score. The half-point convention prevents ties. For mixed outcomes, put the cutoff halfway between the largest losing differential and smallest winning differential. At equality with a fitted cutoff, predict win. For samples with only wins, use cutoff −∞; for only losses, use +∞. Retain the corresponding constant prediction for every finite input.

The explorer varies games per alien and the fixed test point; drawing fresh samples advances the random seed. New samples use a deterministic xorshift generator, not NumPy’s generator. Thus fresh runs are not intended to reproduce the seed-1 notes, although they use the same stated score support and fitting rule. Restore the worked example to return to the exact original fixture. Variance always uses the number of aliens as its divisor; the empirical MSE decomposition holds exactly up to floating-point rounding.

## Run and validate

From the repository root:

```sh
python3 -m http.server 8873 --directory docs
node --check docs/alien/story.js
node --test tests/*.test.mjs
```

Open `http://localhost:8873/alien/`. All paths are relative, with no external runtime libraries or external font requests. Mathematics reuses the repository’s locally served STIX Two Math font. The page supports native keyboard-operable selection and controls, visible focus treatments, reduced motion, Previous/Next navigation, a text explanation usable without JavaScript, and responsive desktop and phone layouts.

`model.mjs` holds the fitting rule, prediction, simulations, and error calculations. `geometry.mjs` provides DOM-independent crowd positions, centered prediction stacks, and proportional decomposition segments. `story.js` draws the persistent figure and handles scrolling and controls.

The five tests in `tests/alien-model.test.mjs` check the original simulation, constant-class samples, the illustrative classification error, score support, and the decomposition at win and loss test points and across sample sizes. The seven tests in `tests/alien-geometry.test.mjs` check the 468/532 split, preserved identities, centered stacks, collision-free worked layouts, proportional widths, zero components, and invalid geometry. All 26 tests in the repository pass. The existing Pages workflow includes these checks.

Browser verification covers all twelve stages at 1440×1000, 600×750, 390×844, and 325×703. The intermediate-width layout was also inspected with Alien 11 selected, whose sample includes a repeated differential. Explorer checks cover twelve combinations: training sizes 5, 25, and 500, each at fixed inputs −20.5, −0.5, +0.5, and +20.5. Every checked combination retains exactly 1,000 marks with no target position outside the figure. No console errors were observed in the desktop run. Enter-key activation of Next was verified; reduced-motion support is implemented but was not verified in the browser. Visual inspection does not establish accessibility compliance.

## Artwork and publication

`assets/alien-basketball.png` is recorded with its description, usage, and SHA-256 in `design/assets.json`. Image-generation prompts are kept outside version control. Its title composition follows the sparse monochrome explanatory-illustration direction in the course design guide. Miniature alien glyphs are drawn directly in SVG as schematic markers in the learning diagram.

The course landing page links to this story. Merging to `main` triggers the repository’s existing GitHub Pages publication workflow.
