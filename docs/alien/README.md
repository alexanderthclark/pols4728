# Learning basketball from five games

A ten-stage scrollytelling explanation of expectation, bias, variance, and MSE, based on the Fall 2026 course notes’ Knicks–Spurs alien example. The separate title screen shows a classic alien stick figure scratching its head while watching basketball. Its registered monochrome illustration is transparent; the title and all teaching content remain selectable HTML text.

## The argument

One alien observes the original five games and fits a decision stump with cutoff −1.5. This fits all training labels but misclassifies a Knicks loss at −1. The story then introduces 1,000 independently trained aliens and holds their test input fixed at +0.5. Each alien is a concrete representation of a different training dataset passed through the same algorithm.

The finite average estimates expectation over training samples. At +0.5 the original simulation has 532 win predictions and 468 loss predictions: mean 0.532, bias −0.468, squared bias 0.219024, variance 0.248976, and MSE 0.468. Outcome noise is zero conditional on the final score differential. This demonstrates bias and variance within one rule, without claiming a complexity tradeoff.

The illustration grid shows only the first 48 aliens on desktop, or 24 on phones, with this limitation labeled directly. Calculations use all 1,000. Selecting an alien reveals its five training differentials and its learned rule. Win predictions use circle badges and the numeral 1; loss predictions use square badges and 0. The title image and the code-native miniature alien glyphs serve different roles: the title is character artwork; the miniature glyphs mark independently fitted rules.

## Simulation and explorer

`samples.json` contains exactly the five-game samples from the notes. They were generated with NumPy’s legacy `RandomState` using seed 1. In each repetition, draw five Knicks integers first, then five Spurs integers, independently from 0 through 100 inclusive. Add 0.5 to each Knicks score. The half-point convention prevents ties. For mixed outcomes, put the cutoff halfway between the largest losing differential and smallest winning differential. At equality with a fitted cutoff, predict win. For samples with only one outcome class, predict that class everywhere.

The explorer varies games per alien, fixed test point, and random seed. New samples use a deterministic xorshift generator, not NumPy’s generator. Thus fresh runs are not intended to reproduce the seed-1 notes, although they use the same stated score support and fitting rule. Restore the worked example to return to the exact original fixture. Variance always uses the number of aliens as its divisor; the empirical MSE decomposition holds exactly up to floating-point rounding.

## Run and validate

From the repository root:

```sh
python3 -m http.server 8873 --directory docs
node --check docs/alien/story.js
node --test tests/*.test.mjs
```

Open `http://localhost:8873/alien/`. All paths are relative, with no external runtime libraries or external font requests. Mathematics reuses the repository’s locally served STIX Two Math font. The page includes Previous/Next buttons, keyboard-accessible sample selection and controls, reduced-motion support, a text explanation usable without JavaScript, and responsive desktop and phone layouts.

`model.mjs` holds the fitting rule, prediction, simulations, and error calculations. `story.js` draws the figures and handles scrolling and controls. `tests/alien-model.test.mjs` checks the original simulation, constant-class samples, the illustrative classification error, score support, and the decomposition at win and loss test points and across sample sizes. The existing Pages workflow includes these checks.

Browser verification covers all ten scenes at 1440×1000, 390×844, and 320×740, including navigation, alien inspection, explorer settings, reset, page errors, and horizontal overflow. Visual inspection complements these checks; it does not establish accessibility compliance.

## Artwork and publication

`assets/alien-basketball.png` is recorded with its description, usage, and SHA-256 in `design/assets.json`. Image-generation prompts are kept outside version control. Its title composition follows the sparse monochrome explanatory-illustration direction in the course design guide. Miniature alien glyphs are drawn directly in SVG as schematic markers in the learning diagram.

The course landing page links to this story. Merging to `main` triggers the repository’s existing GitHub Pages publication workflow.
