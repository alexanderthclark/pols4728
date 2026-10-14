# Shapley values for credit assignment

“Shapley values for credit assignment” is a nineteen-stage scrolling explanation of Shapley values, beginning with three voters. All three support a proposal that needs two votes to pass. The coalition diagram grows from the empty set and singletons into the complete Hasse diagram, then connects joining contributions to the six equally likely accounting orders.

A separate opening page places the real-text title above a registered monochrome illustration of three torso-free stick figures putting their ballots into one box labeled VOTE at once. “A voting application” appears at the bottom of the cover. The cover and story share the course guide’s ink, muted, and blue palette, with the full illustration preserved on desktop and phones. Serif headings use regular weight; prose introduces definitions in italics and leaves numerical conclusions in regular text. The masthead reads “Shapley values.” **Start the explanation** leads to the first teaching scene; **Back to the beginning** returns to the cover. The cover does not add a scene or change the nineteen-stage sequence.

The first eight frames use plain-language groups, outcomes, paths, and weights. A concluding bridge names the existing node numbers as the value function `v(S)`: one for coalitions with at least two voters, zero otherwise. Entering that frame replaces each pass/fail payoff row with `v(group) = payoff`; returning to an earlier frame restores the familiar labels. Only then do six formula frames introduce `N`, `n`, and `i` and connect the general Shapley formula to the diagram, explaining the summation range, marginal contribution, arrangements before the joining voter, arrangements after, division by all orders, and the completed weighted sum. The complete diagram stays visible, and selecting a coalition changes the example edge and its counts. Math uses native MathML, without an external rendering dependency.

At “Put the pieces together,” click a weight or **See the six paths behind these weights**. The optional panel displays all six Hasse diagrams in a three-column grid on wide screens, two columns on medium screens, and a scrolling column on phones. It highlights paths containing the selected joining edge and emphasizes that edge with a thicker arrow. The other paths remain faded. A’s two zero-contribution terms select different pairs of paths.

The interactive explorer also supports B and C, coalition and edge inspection, and order replay. The player selector and inspection result stay beneath the diagram in the main scene. **More exploration** opens replay and detailed edge controls; choosing an order or edge closes the panel so the result is visible. Calculation terms open the six-path explanation for the selected voter. Closing either panel returns keyboard focus to its trigger; opening one pauses any active replay. Reduced-motion preferences are respected by the story.

## Single-column presentation

One full-width visual stage stays in view while scrolling changes the picture,
equation, title, and caption. Each of the nineteen captions contains 16–31 words.
The title and caption sit beneath the visual, with Previous/Next controls and a
step count. Left/Right arrow keys also navigate while a form control or dialog
is not active. The original opening title page remains the entry point.

The six-order scene visits accounting paths as the reader scrolls through it;
its compact buttons also select any order. Entering the value-function scene
names the existing coalition values and puts the rule beneath the diagram.
Formula scenes keep the picture and equation together, highlighting the current
factor. Short phone screens prioritize the equation in these scenes.

The active caption retains its original controls and MathML nodes. Original
sections provide the scroll track and stable hash destinations; links focus the
new caption heading. Optional context, methods, sources, and longer exploration
controls open in a keyboard-accessible native dialog. The state selector retains its selection between visits; moving controls does not
recreate their DOM nodes. A restrained caption transition
is disabled when reduced motion is requested. Annotations retain stable positions in the visual.
Without JavaScript, the source captions and equations remain readable in order.

Browser review covers all nineteen scenes at 1440×900, 1366×650, 768×900,
390×844, 320×700, and 320×568, including diagram/caption separation, detail-panel
focus, keyboard navigation, coalition selection, replay, and the state selector.

## Electoral College example

Four closing scenes change the value function to a weighted voting game with
50 states and DC as players. The 2024/2028 allocation has 538 electoral votes and
a quota of 270. A cumulative tally illustrates one accounting order in which
California is pivotal, followed by a three-map sequence:

1. A geographic Albers USA state map labels the electoral votes, with Alaska
   and Hawaii in insets and callouts for small northeastern jurisdictions.
2. State shapes morph into a continuous electoral-vote cartogram. The mainland
   keeps shared borders; each state's area is proportional to its electoral votes.
3. Shapes morph into a cartogram of adjusted votes, `a_i = 538 × phi_i`.
   These adjusted votes still sum to 538. Equivalently,
   `a_i = w_i × [phi_i / (w_i / 538)]`.

The map uses the same coordinate system and on-screen scale in every phase.
Space for captions, the final equation, and controls is reserved across the
sequence, including short phones; layout changes cannot magnify or hide the
small power premium. An optional dashed electoral-area outline helps compare
the final shapes. State selection, labels, and numeric readouts stay linked.
Forward/reverse morphs, replay, interruption, resizing, and reduced motion are
supported. Phone layouts prioritize state abbreviations and the numeric selector.

The boundary data come from Census-derived U.S. Atlas 3.0.1. The published page
imports local precomputed geometry and requires no external map service.
`scripts/generate-electoral-map.py` rebuilds the geometry offline from the
vendored topology, exact voting-power data, and a continuous area-cartogram
solver. Full provenance, dependencies, and the source license are documented in
[`data/shapley/README.md`](../../data/shapley/README.md).

The geometric approximation has less than 0.1% relative area error per state.
Every state polygon is valid, and the generator verifies no overlaps across
41 sampled endpoints/intermediate frames. Tiny offshore islands are omitted;
Rhode Island uses its mainland outline. Alaska and Hawaii remain separate,
shape-preserving insets using the same vote-area scale as the mainland.

The voting-power calculation itself is exact. `exactWeightedVotingPower` counts
coalitions by size and electoral total, then weights pivotal coalitions by the
factorial arrangements before and after the joining state. Integer arithmetic
preserves exact counts and the denominator is 51!. The browser uses precomputed
exact results; it does not estimate power by permutation simulation.

California's 54 electoral votes become about 58.12 adjusted votes (a 7.64%
relative area increase), while Texas's 40 become 41.66. Florida and New York
also exceed proportional power. Every other state and DC falls below it. These
are state-bloc indices under uniform accounting orders, including a bloc
simplification for Maine and Nebraska. They do not incorporate partisan lean,
swing-state probabilities, population, or individual voters' power.

Sources: [National Archives allocations](https://www.archives.gov/electoral-college/allocation),
[Shapley and Shubik (1954)](https://doi.org/10.2307/1951053),
[Mann and Shapley (1962)](https://www.rand.org/pubs/research_memoranda/RM3158.html),
and [Dougenik, Chrisman, and Niemeyer (1985)](https://doi.org/10.1111/j.0033-0124.1985.00075.x).

## Run locally

From the repository root:

```sh
python3 -m http.server 8765 --directory docs
```

Open `http://localhost:8765/shapley/`. A web server is required because the page imports JavaScript modules. No package installation or build step is needed. Prose fonts are requested from Google Fonts, with local fallbacks; the math font is served locally.

## Files and checks

- `index.html` contains the opening page, narrative, and page structure.
- `style.css` styles the opening page, story, Hasse diagram, and optional panel.
- `assets/voters-ballot-box.png` is the registered opening illustration; the title and application label remain real text in the page.
- `game.mjs` calculates coalition values, joining edges, all accounting orders, edge frequencies, and Shapley values.
- `story.js` handles scroll stages, caption placement, navigation, detail dialogs, and the three-voter explorer.
- `presentation.css` gives the complete story its responsive single-column layout.
- `electoral-college.mjs` provides exact weighted voting calculations and the current allocation/results.
- `electoral-view.js` and `electoral.css` draw the pivotal tally.
- `electoral-map-view.js` and `electoral-map.css` animate the geographic/electoral/adjusted maps.
- `electoral-map-data.mjs` stores the compatible local map geometry.
- `weight-view.js` draws the six-diagram explanation from the game’s orders and edges.
- `formula-view.js` and `formula.css` connect the final formula to selected edges, factorial counts, and complete orders.
- `fonts/` contains the locally served STIX Two Math font and its SIL Open Font License, from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/stixtwomath).

Run the mathematical regression checks with Node.js:

```sh
node --test tests/shapley-game.test.mjs tests/electoral-college.test.mjs tests/electoral-map.test.mjs
```

These checks cover all joining-edge frequencies, the partition of orders for each voter, the distinction between zero-contribution edges, the independent arrangements before and after a joining voter, and a weighted-voting example with unequal Shapley values. Browser checks cover the original explorer and optional weight panel, formula stages and coalition selection, and phone layouts.

## Extending the example

`defineGame` takes player names and a coalition-value function. `matchingOrders` identifies paths using a specific joining edge; it does not group edges merely because their contributions are equal. For three voters, a joining edge after a coalition of size *k* occurs in *k!* × *(2 − k)!* of the six orders. Dividing that count by six gives its Shapley weight.

The current story copy, coalition geometry, and six-diagram panel are tailored to three voters. Another three-player game can reuse the model and paths, but its narrative and outcome labels must be adapted. More players require a new layout and an approach to the growing number of orders.

## Hosting

All asset paths are relative, so the site can be served under `/pols4728/shapley/`. GitHub Pages publishes the `docs/` directory using `.github/workflows/pages.yml`.

The workflow checks syntax and the mathematical tests on site changes in branches and pull requests. A successful check on `main` then publishes the site automatically. Manual deployment is available from the workflow’s **Run workflow** control when `main` is selected; other branches only run checks. The `github-pages` deployment environment permits `main`.

After this branch is merged, the first successful **Check and publish course site** run will make the story available at `https://alexanderthclark.github.io/pols4728/shapley/`, with a course landing page at `https://alexanderthclark.github.io/pols4728/`. No custom domain, paid hosting, build step, or repository secrets are required.
