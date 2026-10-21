# Shapley values for credit assignment

“Shapley values for credit assignment” is a twenty-stage scrolling explanation of Shapley values, beginning with three voters. Each can vote yes or no; the yes voters form a coalition. The proposal passes with two or three yes votes and fails with zero or one. The coalition diagram grows from the empty set and singletons into the complete Hasse diagram, then connects joining contributions to the six equally likely accounting orders.

A separate opening page places the real-text title above a registered monochrome illustration of three torso-free stick figures putting their ballots into one box labeled VOTE at once. “A voting application” appears at the bottom of the cover. The cover and story share the course guide’s ink, muted, and blue palette, with the full illustration preserved on desktop and phones. Serif headings use regular weight; prose introduces definitions in italics and leaves numerical conclusions in regular text. The masthead reads “Shapley values.” **Start the explanation** leads to the first teaching scene; **Back to the beginning** returns to the cover. The cover does not add a scene or change the twenty-stage sequence.

The first eight frames use plain-language groups, outcomes, paths, and weights. The opening frame shows three neutral players, each with the choice “Yes or no,” above two rule lines: “2 or 3 yes votes → Pass” and “0 or 1 yes votes → Fail.” Its caption identifies the yes voters as the coalition; it shows no particular voting outcome. The next frame introduces the empty coalition and singletons. The pair level unfolds across two scroll steps: first A joins B or C, revealing {A, B}, {A, C}, and their two joining arrows under “Add A to coalition.” The next step fills in the remaining pair and the other joining arrows, with that annotation removed before the full Hasse diagram appears. Both pair steps use the same title without a paragraph. A concluding bridge names the existing node numbers as the value function `v(S)`: one for coalitions with at least two voters, zero otherwise. Entering that frame replaces each pass/fail payoff row with `v(group) = payoff`; returning to an earlier frame restores the familiar labels. Only then do six formula frames introduce `N`, `n`, and `i` and connect the general Shapley formula to the diagram, explaining the summation range, marginal contribution, arrangements before the joining voter, arrangements after, division by all orders, and the completed weighted sum. The complete diagram stays visible, and selecting a coalition changes the example edge and its counts. Math uses native MathML, without an external rendering dependency.

At “Average A’s contributions,” click a weight or **See the six paths behind these weights**. The optional panel displays all six Hasse diagrams in a three-column grid on wide screens, two columns on medium screens, and a scrolling column on phones. It highlights paths containing the selected joining edge and emphasizes that edge with a thicker arrow. The other paths remain faded. A’s two zero-contribution terms select different pairs of paths.

The eighth scene keeps the short Hasse-diagram explanation and A/B/C selector beneath the diagram, without a visible heading. The highlighted joining edges and calculation update together, and each voter receives one-third. The inline definition identifies a *Hasse diagram*: each arrow adds one voter to a coalition. Calculation terms open the six-path explanation for the selected voter. Closing that panel returns keyboard focus to its trigger; opening it pauses any active playback. The earlier scenes demonstrate order changes; this scene keeps the voter comparison beside the diagram. Reduced-motion preferences are respected by the story.

## Single-column presentation

One full-width visual stage stays in view while scrolling changes the picture,
equation, title, and caption. Each scene’s caption stays brief.
The title and caption sit beneath the visual. Scrolling moves between scenes;
Left/Right arrow keys also navigate while a form control or dialog is not active.
The original opening title page remains the entry point.

The six-order scene visits accounting paths as the reader scrolls through it;
its compact buttons also select any order. Entering the value-function scene
names the existing coalition values and puts the rule beneath the diagram.
Formula scenes begin with “A formula for A’s Shapley value” and keep the picture
and equation together. On larger screens, the summation scene uses the available
height for a larger diagram and places the equation and divider below it.
The explanation of each highlighted term uses the same
blue as that term. “Count orders before A” shows the arrangements without a
separate numerical factorial calculation; the other worked calculations retain
their matching blue. Short phone screens prioritize the equation in these scenes.

The active caption retains its original controls and MathML nodes. Original
sections provide the scroll track and stable hash destinations; links focus the
new caption heading. Optional context, methods, and sources
open in a keyboard-accessible native dialog. Background scrolling pauses while a detail dialog is open and resumes when it closes. Moving controls does not recreate their DOM nodes. A restrained caption transition
only introduces changed headings and summaries: repeated text stays visible as the
diagram advances or reverses. The transition is disabled when reduced motion is
requested. Annotations retain stable positions in the visual.
Without JavaScript, the source captions and equations remain readable in order.

Browser review should cover all twenty scenes at 1440×900, 1063×736, 768×900,
390×844, 320×700, and 320×568, including diagram/caption separation, detail-panel
focus, keyboard navigation, coalition selection, and replay.

## Electoral College example

Five closing scenes change the value function to a weighted voting game with
50 states and DC as players. The first is a title transition at
`#electoral-introduction`: a monochrome cartoon above the real-text title
“Application: US Electoral College.” Two equally sized round, torso-free figures use one
ballot box. A surfer with a surfboard puts in many ballots; a figure in a cowboy
hat holds exactly three. The illustration has a white background and black lines,
with no labels, state outlines, or partisan colors. The original opening cover
remains separate and unchanged.

The transition artwork shares the opening title graphic’s width and responsive
height limits, leaving white space around the illustration instead of filling
the visual stage.

The 2024/2028 allocation has 538 electoral votes and a quota of 270.
The following tally shows California’s marginal contribution to a 250-vote coalition:
its 54 votes raise the total to 304, crossing the threshold. Three maps follow:

1. A geographic Albers USA state map labels the electoral votes, with Alaska
   and Hawaii in insets. Only California and Wyoming have numeric labels.
2. State shapes morph into a square-cell electoral-vote cartogram. Its 538
   complete squares each represent one electoral vote, grouped into connected
   state shapes in an approximate U.S. layout.
3. Boundary cells grow or shrink into a cartogram of adjusted votes, `a_i = 538 × phi_i`.
   These adjusted votes still sum to 538. Equivalently,
   `a_i = w_i × [phi_i / (w_i / 538)]`.

The map uses the same coordinate system and on-screen scale in every phase.
Space for captions, the final equation, and controls is reserved across the
sequence, including short phones; layout changes cannot magnify or hide the
small power premium. The final scene adds a full viewport of scroll space so
the stage remains pinned while the Shapley adjustment plays, then releases
naturally to the footer. Electoral scenes use neutral grays throughout, with
California and Wyoming marked by a darker fill and thicker outline, avoiding
party colors. Their electoral-area outlines stay visible in the final view. A persistent
three-stage control navigates Geography, Electoral votes, and Shapley votes,
updating the story caption and hash together. Their fixed labels update from
54 and 3 electoral votes to 58.1 and 2.9 adjusted votes during the second morph.
Forward/reverse morphs, replay, interruption, resizing, and reduced motion are
supported. California and Wyoming are the only labeled states. Their numeric
labels stay visible at every screen size; all state names and voting values
remain in hover titles and accessible labels. Removing the state selector and
readout gives the map more space. A single frame crops unused margins around
all three endpoints together, keeping their screen scale identical.

The boundary data come from Census-derived U.S. Atlas 3.0.1. The published page
imports local precomputed geometry and requires no external map service.
The geographic geometry comes from `scripts/generate-electoral-map.py` and the
vendored topology. The square-cell layout is generated separately, then
`electoral-map-geometry.mjs` places both maps in the same coordinate frame.
Full provenance, dependencies, and the source license are documented in
[`data/shapley/README.md`](../../data/shapley/README.md).

Whole and fractional squares use the same area unit in both cartograms.
California adds four complete squares and a fraction of a fifth; Texas adds
one and a fraction, while Florida and New York each add a fraction. Every
other state and DC loses a fraction of one boundary square. Interpolating one
dimension of each changing square preserves the national filled area throughout
the second morph. Independent tests check exact state counts, connected shapes,
fractional areas, the common scale, and no cell overlap across 41 sampled frames.
Tiny offshore islands are omitted from the geographic map; Rhode Island uses
its mainland outline. Alaska and Hawaii remain separate insets.

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
and [U.S. Atlas](https://github.com/topojson/us-atlas).

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
- `assets/electoral-ballot-characters.png` is the registered Electoral College transition illustration; “Application: US Electoral College” remains real text below the art.
- `game.mjs` calculates coalition values, joining edges, all accounting orders, edge frequencies, and Shapley values.
- `story.js` handles scroll stages, caption placement, navigation, detail dialogs, and the three-voter explorer.
- `presentation.css` gives the complete story its responsive single-column layout.
- `electoral-college.mjs` provides exact weighted voting calculations and the current allocation/results.
- `electoral-view.js` and `electoral.css` draw the coalition tally.
- `electoral-map-view.js` and `electoral-map.css` animate the geographic/electoral/adjusted maps.
- `electoral-map-data.mjs` stores the local geographic map geometry.
- `electoral-tile-data.mjs` stores complete and fractional square-cell cartograms.
- `electoral-map-geometry.mjs` aligns map endpoints and calculates exposed cell edges.
- `weight-view.js` draws the six-diagram explanation from the game’s orders and edges.
- `formula-view.js` and `formula.css` connect the final formula to selected edges, factorial counts, and complete orders.
- `fonts/` contains the locally served STIX Two Math font and its SIL Open Font License, from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/stixtwomath).

Run the mathematical regression checks with Node.js:

```sh
node --test tests/shapley-game.test.mjs tests/electoral-college.test.mjs tests/electoral-map.test.mjs tests/electoral-tile.test.mjs
```

These checks cover all joining-edge frequencies, the partition of orders for each voter, the distinction between zero-contribution edges, the independent arrangements before and after a joining voter, and a weighted-voting example with unequal Shapley values. Browser checks cover the original explorer and optional weight panel, formula stages and coalition selection, and phone layouts.

## Extending the example

`defineGame` takes player names and a coalition-value function. `matchingOrders` identifies paths using a specific joining edge; it does not group edges merely because their contributions are equal. For three voters, a joining edge after a coalition of size *k* occurs in *k!* × *(2 − k)!* of the six orders. Dividing that count by six gives its Shapley weight.

The current story copy, coalition geometry, and six-diagram panel are tailored to three voters. Another three-player game can reuse the model and paths, but its narrative and outcome labels must be adapted. More players require a new layout and an approach to the growing number of orders.

## Hosting

All asset paths are relative, so the site can be served under `/pols4728/shapley/`. GitHub Pages publishes the `docs/` directory using `.github/workflows/pages.yml`.

The workflow checks syntax and the mathematical tests on site changes in branches and pull requests. A successful check on `main` then publishes the site automatically. Manual deployment is available from the workflow’s **Run workflow** control when `main` is selected; other branches only run checks. The `github-pages` deployment environment permits `main`.

After this branch is merged, the first successful **Check and publish course site** run will make the story available at `https://alexanderthclark.github.io/pols4728/shapley/`, with a course landing page at `https://alexanderthclark.github.io/pols4728/`. No custom domain, paid hosting, build step, or repository secrets are required.
