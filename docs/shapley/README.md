# Shapley values, step by step

“Who gets the credit?” is an eight-stage scrolling explanation of Shapley values for three voters. All three support a proposal that needs two votes to pass. The coalition diagram grows from the empty set and singletons into the complete Hasse diagram, then connects joining contributions to the six equally likely accounting orders.

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

Run the mathematical regression checks with Node.js:

```sh
node --test tests/shapley-game.test.mjs
```

These checks cover all joining-edge frequencies, the partition of orders for each voter, the distinction between zero-contribution edges, and a weighted-voting example with unequal Shapley values. Browser checks also cover frame-seven access, switching weights, B’s explanation, keyboard closing and focus return, and the phone layout.

## Extending the example

`defineGame` takes player names and a coalition-value function. `matchingOrders` identifies paths using a specific joining edge; it does not group edges merely because their contributions are equal. For three voters, a joining edge after a coalition of size *k* occurs in *k!* × *(2 − k)!* of the six orders. Dividing that count by six gives its Shapley weight.

The current story copy, coalition geometry, and six-diagram panel are tailored to three voters. Another three-player game can reuse the model and paths, but its narrative and outcome labels must be adapted. More players require a new layout and an approach to the growing number of orders.

## Hosting

All asset paths are relative, so the site can be served under a project path such as `/pols4728/shapley/`. The `docs/` layout is ready for a future GitHub Pages deployment. This branch adds the source; it does not enable Pages or change the repository’s current deployment settings.
