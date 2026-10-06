# Shapley values, step by step

“Who gets the credit?” is a fourteen-stage scrolling explanation of Shapley values for three voters. All three support a proposal that needs two votes to pass. The coalition diagram grows from the empty set and singletons into the complete Hasse diagram, then connects joining contributions to the six equally likely accounting orders.

The first eight frames use plain-language groups, outcomes, paths, and weights. A concluding bridge names the existing node numbers as the value function `v(S)`: one for coalitions with at least two voters, zero otherwise. Only then do six formula frames introduce `N`, `n`, and `i` and connect the general Shapley formula to the diagram, explaining the summation range, marginal contribution, arrangements before the joining voter, arrangements after, division by all orders, and the completed weighted sum. The complete diagram stays visible, and selecting a coalition changes the example edge and its counts. Math uses native MathML, without an external rendering dependency.

At “Put the pieces together,” click a weight or **See the six paths behind these weights**. The optional panel displays all six Hasse diagrams in a three-column grid on wide screens, two columns on medium screens, and a scrolling column on phones. It highlights paths containing the selected joining edge and emphasizes that edge with a thicker arrow. The other paths remain faded. A’s two zero-contribution terms select different pairs of paths.

The final explorer also supports B and C, coalition and edge inspection, and order replay. Calculation terms open the same explanation for the selected voter. Closing the panel returns keyboard focus to its trigger; opening it pauses any active replay. Reduced-motion preferences are respected by the story.

## Run locally

From the repository root:

```sh
python3 -m http.server 8765 --directory docs
```

Open `http://localhost:8765/shapley/`. A web server is required because the page imports JavaScript modules. No package installation or build step is needed. Fonts are requested from Google Fonts, with local fallbacks.

## Files and checks

- `index.html` contains the narrative and page structure.
- `style.css` styles the story, Hasse diagram, and optional panel.
- `game.mjs` calculates coalition values, joining edges, all accounting orders, edge frequencies, and Shapley values.
- `story.js` handles scroll stages and the final explorer.
- `weight-view.js` draws the six-diagram explanation from the game’s orders and edges.
- `formula-view.js` and `formula.css` connect the final formula to selected edges, factorial counts, and complete orders.
- `fonts/` contains the locally served STIX Two Math font and its SIL Open Font License, from [Google Fonts](https://github.com/google/fonts/tree/main/ofl/stixtwomath).

Run the mathematical regression checks with Node.js:

```sh
node --test tests/shapley-game.test.mjs
```

These checks cover all joining-edge frequencies, the partition of orders for each voter, the distinction between zero-contribution edges, the independent arrangements before and after a joining voter, and a weighted-voting example with unequal Shapley values. Browser checks cover the original explorer and optional weight panel, formula stages and coalition selection, and phone layouts.

## Extending the example

`defineGame` takes player names and a coalition-value function. `matchingOrders` identifies paths using a specific joining edge; it does not group edges merely because their contributions are equal. For three voters, a joining edge after a coalition of size *k* occurs in *k!* × *(2 − k)!* of the six orders. Dividing that count by six gives its Shapley weight.

The current story copy, coalition geometry, and six-diagram panel are tailored to three voters. Another three-player game can reuse the model and paths, but its narrative and outcome labels must be adapted. More players require a new layout and an approach to the growing number of orders.

## Hosting

All asset paths are relative, so the site can be served under `/pols4728/shapley/`. GitHub Pages publishes the `docs/` directory using `.github/workflows/pages.yml`.

The workflow checks syntax and the mathematical tests on site changes in branches and pull requests. A successful check on `main` then publishes the site automatically. Manual deployment is available from the workflow’s **Run workflow** control when `main` is selected; other branches only run checks. The `github-pages` deployment environment permits `main`.

After this branch is merged, the first successful **Check and publish course site** run will make the story available at `https://alexanderthclark.github.io/pols4728/shapley/`, with a course landing page at `https://alexanderthclark.github.io/pols4728/`. No custom domain, paid hosting, build step, or repository secrets are required.
