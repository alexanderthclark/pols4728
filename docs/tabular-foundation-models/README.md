# Tabular foundation models

Buildless scroll-driven course lesson for POLS 4728, following Christoph Molnar's *Tabular Foundation Models*. White textbook layout, Latin Modern type, course palette, torso-less hero illustration. The bear-survival task is the running example throughout the lesson: the true depth-two data-generating tree, labeled context, a worked training episode, shared network weights, actual row encodings and cross-attention calculations, exact Bayesian updating, and seven scenes diagnosing the interaction for XGBoost.

Run `npm run build` to regenerate the no-JavaScript reading version, `npm test` for Bayesian arithmetic and reversible visual-state checks, and `npm run dev` for a local preview. Source files in this directory are served directly. All core teaching points are reached by native scrolling; chapter links offer shortcuts. `?motion=reduce` or system reduced-motion preference uses complete visual states at logical thresholds.

## Numerical example

Two possible bear-survival worlds, with equal prior mass. World A assigns survival probability p = logistic(1) to matching pairs (black-and-fight or brown-and-lie-down), and 1 − p to mismatches; world B reverses them. These are the same population probabilities used in the later simulation. Outcomes are independent conditional on the shared world and fixed features.

The Bayesian sequence uses the miniature network's actual four-row context throughout: Black/Fight/Survived, Black/Lie down/Died, Brown/Fight/Died, and Brown/Lie down/Survived. Every observed outcome is p under A and 1 − p under B, including the deaths. Each supplies a likelihood ratio exp(1), so all four give posterior odds exp(4), posterior probability 98.2014%, and predictive survival 72.2747% for the Brown/Lie down query. Reversing all four labels gives 27.7253% predictive survival. These are compared directly with the fixed network's measured 74.0281% and 23.5539% outputs, illustrating approximation error. An expected-log-loss curve explains why the ideal prediction is the conditional probability under the task generator. This identifies the population optimum, not guaranteed convergence of a finite neural network.

No TabPFN model is run. Animation interpolation is visual interpolation between exact results, not fractional-data Bayesian updating. The core training and inference chapter uses outputs from the saved miniature network.

## Sources and attribution

Molnar's book (https://tabularfoundationmodels.com/) is the main reference, particularly chapters 2–5, with application and limitation context from chapters 7 and 12. The text is original exposition, and the two-world example and visual diagrams are original. The book is CC BY-NC 4.0. The PFN paper and TabPFN Nature paper are linked in the lesson. Version-specific limits and benchmark claims are deliberately omitted. Source references checked 2026-11-22.

## Design choices

Course design reference: the existing POLS 4728 Open textbook guide. Navy denotes the model and hypothesis A; rust denotes weight updates, held-out evaluation, and contrasting hypothesis B (direct labels distinguish these uses); teal denotes targets and predictive probabilities. Table grids are retained to convey rows and columns. Narrow screens use compact diagrams with larger labels rather than shrinking desktop SVGs. Locally bundled Latin Modern and KaTeX license files accompany their assets.


## Trained miniature PFN

`mini_pfn.py` trains a small educational model across two opposed bear-survival rules, with one hidden rule shared by all rows in each task. It receives raw context B/L/Y and query B/L, with four context rows. A 32-dimensional row encoder and query encoder feed one cross-attention head and an MLP decoder. Neither the matching-pair indicator nor world identity nor a Bayesian answer enters training. The actual run uses 4,000 Adam steps, 256 generated tasks per step, and 7,617 learned parameters.

`mini-pfn-results.mjs` is generated from the saved run. It contains validation checkpoints, independent test metrics, all trained parameters, and complete A/B context traces. The displayed query is the same in both; changing only context labels changes prediction from 0.740281 to 0.235539. Exact Bayesian targets are 0.722747 and 0.277253. Independent test log losses are 0.630063 for the network, 0.628353 for Bayes, and 0.693147 when ignoring context. The validation curve uses the same 2,048 independent tasks at each checkpoint. This miniature is not TabPFN, is not an irrelevant-feature benchmark, and is not intended to reproduce production architectures.

The no-JavaScript version includes the mechanism and Bayesian diagrams, plus separate before/after snapshots of the recorded first Adam update. The training chapter uses that update’s own context and query; the update averages 256 task losses. Final-checkpoint predictions are reserved for the inference chapter. The numerical tests independently reconstruct both forward passes from the exported parameters, including attention softmax, value mixtures, decoder logits, and probabilities.

## Bear experiment

The bear chapter separates population probabilities, measured XGBoost results, and a proposed explanation for TabPFN. `bear-results.mjs` is generated by `../../examples/tabular-foundation-models/bear_diagnostics.py`; the SVG plot consumes its exact per-seed records and means. Its oracle reference is the population AUC logistic(1) for this balanced symmetric DGP, not an upper bound on every finite test sample. The reading version includes depth-two and depth-six results, seed standard deviations, Brier scores, and reproduction links. The simulation uses independent binary bear/strategy features, three uniform distractors, and logistic outcome noise. It yields 73.1059% versus 26.8941% survival by response match. No TabPFN inference or attention measurement is represented as an observed result; the separate miniature PFN trace is explicitly identified.

The plotted variants hold labels and splits fixed within each seed: no distractors; continuous distractors; binarized distractors; two histogram bins; and an added feature product. Full CSV/JSON results and dependency versions are tracked with the script. `npm test` checks reported summaries against all per-seed records, exact chart data, retained scene routing, and deterministic reversible visual states.

Authoritative source is this directory in `alexanderthclark/pols4728`, branch `codex/tabular-foundation-models`. The hosted preview is a copy of these assets. No homepage link is added by this change.
