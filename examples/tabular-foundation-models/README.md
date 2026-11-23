# Bear interaction diagnostics

This experiment isolates difficulties in finding a simple interaction amid irrelevant features. It evaluates XGBoost only. No TabPFN scores are generated or inferred. Settings are fixed in advance for explanation, without hyperparameter tuning; these results are not a claim about the best attainable XGBoost performance.

## Reproduce

Use Python 3.12 and install the pinned requirements:

```sh
python -m pip install -r examples/tabular-foundation-models/requirements.txt
python examples/tabular-foundation-models/bear_diagnostics.py
```

The command writes per-seed records to `results/bear-records.csv`, complete results to `results/bear-results.json`, and the equivalent JavaScript module to `docs/tabular-foundation-models/bear-results.mjs`. Paths are resolved from the script, so it can run from any working directory. `--n`, `--noise-columns` (default 10), `--seeds`, `--depths`, `--output-dir`, and `--web-output` support separate exploratory runs; use separate output paths to retain the published run.

## Data-generating process

Let `B` indicate a brown bear and `A` indicate lying down. Both are independent Bernoulli(0.5) draws. Ten additional columns are independent Uniform(0,1) noise, independent of both binary variables and the standard logistic outcome error `E`:

```python
y = (-1 + 2 * (B == A) + E > 0).astype(int)
```

Population survival probability is `sigmoid(1) = 0.7310586` when `B == A`, and `sigmoid(-1) = 0.2689414` otherwise. Either binary variable by itself leaves survival probability at 0.5. The logistic outcome error is distinct from the irrelevant input columns. The example is a simulated prediction problem.

For each seed 0 through 11, generate 10,000 rows using `bear_simulation.py`. The original draw order B, A, the first three noise columns, then E is preserved. Seven further noise columns are appended from an independent `SeedSequence([seed, 4728, 20261123])` random stream, drawn one full column at a time. This preserves the original labels, signal columns, and first three distractors when extending the earlier design from three to ten noise columns. Use an unstratified 80/20 `train_test_split` with `random_state=seed`, giving 8,000 training and 2,000 test rows. All variants within a seed share exactly the same labels and split indices.

## Comparisons

1. Bear type and strategy only.
2. Add ten continuous noise columns (12 input features in total).
3. Replace the same noise draws by indicators `noise > 0.5`.
4. Keep continuous noise and append the product `B * A`.
5. Keep the original continuous inputs but reduce XGBoost's histogram bin count from 256 to 2.

For each version, fit 200 boosted trees at learning rate 0.1, using histogram tree construction, two CPU threads, and depths 2 and 6 separately. The random state equals the data seed. Other XGBoost settings retain the defaults of version 3.0.5. The bin count is 256 except in the explicitly labeled two-bin case. There is no early stopping or selection using test outcomes.

ROC AUC measures ranking quality; larger is better. Brier score is the mean squared error of predicted probabilities; smaller is better. Summaries report means and sample standard deviations across the 12 seeds. These SDs are not confidence intervals. Paired deltas subtract the original continuous-noise case at the same seed and depth; positive AUC deltas and negative Brier deltas indicate improvement.

`first_root_feature` records the root split of the first boosted tree, not all trees. The `product` category is separate from both original signal and noise. This diagnostic does not make a causal claim about individual attention heads or identify TabPFN's internal algorithm.

## Interpretation

A depth-two tree can represent the population relationship, but greedy construction evaluates one split at a time. Continuous distractors offer many candidate thresholds, creating opportunities for accidental training gains. Removing or coarsening distractors, or making the interaction available as a feature, can help. Binarization also changes the inputs' marginal distributions and removes the cue that only useful variables are binary; it is not a perfectly isolated intervention on one internal mechanism.

Increasing depth can help discover the interaction, although it can also fit outcome noise. The depth-six results are included so the lesson does not equate a depth-two diagnostic with a universal limit of boosted trees. A matched TabPFN comparison would require running a specified checkpoint on the same splits and variants.

## References

- [XGBoost: learning tree structure](https://xgboost.readthedocs.io/en/stable/tutorials/model.html#learn-the-tree-structure).
- [Donick and Lera (2021), Uncovering feature interdependencies in high-noise environments with stepwise lookahead decision forests](https://pmc.ncbi.nlm.nih.gov/articles/PMC8085031/).
- [Christoph Molnar, Tabular Foundation Models](https://tabularfoundationmodels.com/).
