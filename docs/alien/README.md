# Can an alien tell who won?

A twelve-stage scrollytelling explanation of expectation, bias, variance, and MSE, based on the Fall 2026 course notes’ Knicks–Spurs alien example. The cover states the prediction task: given Knicks points minus Spurs points, predict whether the Knicks won. A classic alien stick figure scratches its head while watching basketball. Its registered monochrome illustration is transparent. Titles and narrative remain selectable HTML text; quantitative labels remain real SVG text.

## The argument

Alien 1 observes the original five games and fits a decision stump with cutoff −1.5. This fits all training labels but misclassifies a Knicks loss at −1. The same alien and history remain visible when two other illustrative histories are introduced. Alien 2 initially learns cutoff +1; Alien 3 sees five wins, uses cutoff −∞, and always predicts a win. Their presentation labels 1, 2, and 3 remain stable when Alien 2 receives new histories. The history controls change only Alien 2.

All three then answer the same question at +0.5. A separate controlled simulation begins in stage 6: 1,000 independent synthetic five-game samples, each fitted by the same learning procedure and evaluated at that fixed input. The illustrative original history is not inserted into this population. Numerical expectations and error components refer to the stated synthetic score distribution.

The twelve stages develop the argument in order:

1. Establish the classification task, observe five final scores, and define the input and binary target.
2. Fit the midpoint cutoff between the closest observed loss and win.
3. Test a new game and compare the fitted cutoff with the true boundary.
4. Compare the original history with two illustrative alternatives and their fitted rules.
5. Apply the three rules to the same half-point final difference.
6. Begin 1,000 independent synthetic training repetitions, with one mark for each fitted rule.
7. Gather predictions at zero and one and estimate their expectation.
8. Measure bias from the true outcome to the average prediction.
9. Measure variance from individual predictions to their average.
10. Measure MSE from individual predictions to the true outcome and introduce the wrong-answer probability q.
11. Connect squared bias and variance through q and account for zero outcome noise.
12. Compare current results with a five-game reference while varying sample size, fixed input, or fresh samples.

The finite average estimates expectation over training datasets drawn in the same way. At +0.5 the worked simulation has 532 win predictions and 468 loss predictions: mean 0.532, bias −0.468, squared bias 0.219024, variance 0.248976, and MSE 0.468. No individual fitted rule predicts 0.532. Outcome noise is zero conditional on the final score differential. The formal expectation notation and its definitions appear in a disclosure after the concrete average is explained.

Let q be the probability a newly trained alien is wrong at the fixed binary input. Squared bias is q², variance is q(1−q), and MSE is q. At a winning input bias is −q; at a losing input it is +q. For q below one half, reducing q reduces both squared bias and variance: predictions become more accurate and more unanimous. The learning rule and model complexity remain fixed. At +0.5 the default sequence for 5, 10, 25, 100, and 500 games gives estimated q values 0.468, 0.406, 0.369, 0.102, and 0.000, so both components fall in that displayed sequence. Finite batches can fluctuate. If q exceeds one half, improving it toward one half initially raises variance while reducing squared bias; this binary-prediction qualification is explained in the explorer.

## The visual argument

One persistent SVG accompanies the narrative. The opening table displays outcomes as plain zeroes and ones. The first fitted cutoff and its comparison with the true boundary share one number line. All observed locations lie on that line; the repeated +1 is labeled “2 games.” A short cutoff tick and midpoint bracket locate −1.5. The new-game comparison adds a short dashed true-boundary tick and a narrow hatched error interval, with no vertical data dimension.

The three illustrative aliens have distinct monochrome poses. Stage 4 presents each history as five sorted numerical score differences, rather than placing the observations on a crowded spatial axis. The buttons “Give Alien 2 new games” and “Restore the three histories” update the middle history or restore its initial state. Aliens 1 and 3 retain their games. Stage 5 switches to aligned learned-rule axes, zoomed to −3 through +3, with a common test line at +0.5. Cutoffs outside the view, including infinite cutoffs, use outward arrows and retain their numerical labels. Arrows from the fitted rules end at plain zeroes and ones in a prediction column.

The controlled population contains exactly 1,000 keyed marks, one for each fitted synthetic sample. The same marks move from a crowd into prediction piles at zero and one; their identities remain stable through expectation, bias, variance, MSE, and the decomposition. Circles represent win predictions and squares represent loss predictions, alongside numerical labels and counts. A compact context line states the training sample size, fixed score difference, and actual outcome on every statistical frame. The prediction axis is explicitly distinguished from the earlier score-differential axis.

Bias uses an arrow from the actual outcome to the mean. Variance measures the signed differences between individual predictions and that mean before squaring. MSE changes the reference to the actual outcome and labels the squared error for each prediction group. These frames preserve the same fitted predictions; only the reference point and calculation change.

The decomposition uses exact proportional segment widths on a squared-error scale from zero to one. Squared bias and variance add to the MSE width; outcome noise contributes zero width. Readouts round the worked components to approximately 0.219 and 0.249, while expandable prose retains the exact values.

Phones use compact figure coordinates and shorter displayed formulas, with the substantive explanation and exact values retained in the narrative. Their history diagrams use an Alien column with numbers beside the poses, giving the illustrations and numerical labels separate space. The stack column count adapts to the larger prediction group so that all 1,000 marks remain visible even when every alien gives the same answer; nearly unanimous desktop stacks reserve space above the marks for the mean label. The title image supplies character artwork; miniature SVG alien glyphs identify the fitted rules in the sample diagrams.

## Simulation and explorer

`samples.json` contains exactly the five-game samples from the notes. They were generated with NumPy’s legacy `RandomState` using seed 1. In each repetition, draw five Knicks integers first, then five Spurs integers, independently from 0 through 100 inclusive. Add 0.5 to each Knicks score. The half-point convention prevents ties. For mixed outcomes, put the cutoff halfway between the largest losing differential and smallest winning differential. At equality with a fitted cutoff, predict win. For samples with only wins, use cutoff −∞; for only losses, use +∞. Retain the corresponding constant prediction for every finite input.

The history controls cycle Alien 2 through worked five-game samples, leaving the original illustrative history and chosen all-win history fixed. These controls do not change the 1,000-sample population or its statistics.

The explorer varies games per alien and the fixed test point; drawing fresh samples advances the random seed. New samples use a deterministic xorshift generator, not NumPy’s generator. Thus fresh runs are not intended to reproduce the seed-1 notes, although they use the same stated score support and fitting rule. Restore the worked example to return to the exact original fixture, +0.5 test input, five-game setting, and explorer seed 10. Variance always uses the number of aliens as its divisor; the empirical MSE decomposition holds exactly up to floating-point rounding.

A live comparison table retains the worked five-game reference alongside the current squared bias, variance, and percentage of wrong answers. Both columns use the currently selected fixed test input, including losing inputs. The reference is recomputed from the unchanged five-game fixture when that input changes. This makes joint movement of the components visible without requiring the reader to remember earlier frames. The status text reports the wrong-answer count and the estimate of q. The prose distinguishes the default improvement sequence from fluctuations in finite simulations and explains why variance can initially rise when q exceeds one half.

## Run and validate

From the repository root:

```sh
python3 -m http.server 8873 --directory docs
node --check docs/alien/story.js
node --test tests/*.test.mjs
```

Open `http://localhost:8873/alien/`. All paths are relative, with no external runtime libraries or external font requests. Mathematics reuses the repository’s locally served STIX Two Math font. The page supports native keyboard-operable buttons and explorer controls, visible focus treatments, reduced motion, Previous/Next navigation, a text explanation usable without JavaScript, and responsive desktop and phone layouts.

`model.mjs` holds the fitting rule, prediction, simulations, and error calculations. `geometry.mjs` provides DOM-independent crowd positions, centered prediction stacks, and proportional decomposition segments. `story.js` draws the persistent figure and handles scrolling and controls.

The five tests in `tests/alien-model.test.mjs` check the original simulation, constant-class samples, the illustrative classification error, score support, and the decomposition at win and loss test points and across sample sizes. The seven tests in `tests/alien-geometry.test.mjs` check the 468/532 split, preserved identities, centered stacks, collision-free worked layouts, proportional widths, zero components, and invalid geometry. All 26 repository tests pass. The existing Pages workflow includes these checks.

Browser review covered all twelve stages at 1440×1000, 600×750, and 325×703, with additional focused checks at 390×844. The tablet and narrow-phone sequences have no horizontal overflow. Changing and restoring Alien 2’s history preserved Aliens 1 and 3, including their poses and games. Finite cutoffs on both sides of the zoomed view and the all-win infinite cutoff were checked. A static-layer refresh preserves unchanged identity labels when the SVG is redrawn, while the population retains its animated movement.

The reference comparison was verified at 5, 10, 25, and 500 games and inputs +0.5, −0.5, and −20.5. Both columns use the selected test input; drawing new samples leaves the reference unchanged, and restoring the worked example also restores the default seed. Unanimous groups retain all 1,000 marks, with the mean label clear of the stack on desktop and phone. Enter-key activation of Next was verified, and no browser console errors were observed. Reduced motion is supported by the stylesheet; visual inspection alone does not establish accessibility compliance.

## Artwork and publication

`assets/alien-basketball.png` is recorded with its description, usage, and SHA-256 in `design/assets.json`. Image-generation prompts are kept outside version control. Its title composition follows the sparse monochrome explanatory-illustration direction in the course design guide. Miniature alien glyphs are drawn directly in SVG as schematic markers in the learning diagram.

The course landing page links to this story. Merging to `main` triggers the repository’s existing GitHub Pages publication workflow.
