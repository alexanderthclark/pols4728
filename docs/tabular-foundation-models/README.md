# Tabular foundation models

Buildless scroll-driven course lesson for POLS 4728, following Christoph Molnar's *Tabular Foundation Models* and Hollmann et al.'s Nature 2025 paper. White textbook layout, Latin Modern type, course palette, torso-less hero illustration. An opening ATUS prediction comparison motivates testing pretrained models on a real task. The bear-survival task is then the running mechanism example: the true depth-two data-generating tree, a measured tree/forest/TabICLv2 comparison with distractors, labeled context, a worked training episode, shared network weights, actual row encodings and cross-attention calculations, exact Bayesian updating, and seven scenes diagnosing the interaction for XGBoost.

Run `npm run build` to regenerate the no-JavaScript reading version, `npm test` for Bayesian arithmetic and reversible visual-state checks, and `npm run dev` for a local preview. Source files in this directory are served directly. All core teaching points are reached by native scrolling; chapter links offer shortcuts. `?motion=reduce` or system reduced-motion preference uses complete visual states at logical thresholds.

## Numerical example

Two possible bear-survival worlds, with equal prior mass. World A assigns survival probability p = logistic(1) to matching pairs (black-and-fight or brown-and-lie-down), and 1 − p to mismatches; world B reverses them. These are the same population probabilities used in the later simulation. Outcomes are independent conditional on the shared world and fixed features.

The Bayesian sequence uses the miniature network's actual four-row context throughout: Black/Fight/Survived, Black/Lie down/Died, Brown/Fight/Died, and Brown/Lie down/Survived. Every observed outcome is p under A and 1 − p under B, including the deaths. Each supplies a likelihood ratio exp(1), so all four give posterior odds exp(4), posterior probability 98.2014%, and predictive survival 72.2747% for the Brown/Lie down query. Reversing all four labels gives 27.7253% predictive survival. These are compared directly with the fixed network's measured 74.0281% and 23.5539% outputs, illustrating approximation error. The expected-log-loss derivation is retained only as optional background in the reading version, outside the applied scroll sequence.

No TabPFN model is run. Animation interpolation is visual interpolation between exact results, not fractional-data Bayesian updating. The core training and inference chapter uses outputs from the saved miniature network.

## Sources and attribution

Molnar's book (https://tabularfoundationmodels.com/) is the main reference, particularly chapters 2–5, with application and limitation context from chapters 7 and 12. The text is original exposition, and the two-world example and visual diagrams are original. The book is CC BY-NC 4.0. The PFN paper and TabPFN Nature paper are linked in the lesson. The Nature paper’s generator, architecture, and sample-efficiency result appear in three scroll scenes; all version-specific results are labeled as evidence from the 2025 paper. Its main benchmark scope is distinguished from current software limits. Source references checked 2026-11-22.

## Nature 2025 integration

`nature-scenes.mjs` and `nature.mjs` connect the miniature PFN to the full paper. The original bear dependency-graph analogy is retained in optional reading background; the main pretraining subsection instead uses varied anonymous SCMs. The architecture scene distinguishes within-row feature attention and across-row sample attention; it shows individual cells for clarity and notes that the paper groups two features per token by default. It is a permitted-information-flow illustration, not an internal trace. Query labels remain withheld and query rows read context rows only.

The Fig. 5a sample-efficiency scene uses fixed quantitative bar geometry: default CatBoost with 100% of training rows versus TabPFN with approximately 50%, at matching aggregate predictive performance. This is not an AUC chart or a per-dataset guarantee. The paper’s distractor experiment uses shuffled copies of original features, unlike our independently generated uniform columns. The bear hypothesis diagram supplies all columns without labeling relevant ones in advance. It explicitly distinguishes joint signal from marginal association.

The reading guide records benchmark scope (29 classification/28 regression datasets, at most 10,000 rows/500 features/10 classes), evaluation and hardware qualifications, regression distributions, and the default four/eight inference variants. Optional product features are identified as disabled by default. The paper is CC BY 4.0; the diagrams and prose are original adaptations with attribution.

## Design choices

Course design reference: the existing POLS 4728 Open textbook guide. Navy denotes the model and hypothesis A; rust denotes weight updates, held-out evaluation, and contrasting hypothesis B (direct labels distinguish these uses); teal denotes targets and predictive probabilities. Table grids are retained to convey rows and columns. Narrow screens use compact diagrams with larger labels rather than shrinking desktop SVGs. Locally bundled Latin Modern and KaTeX license files accompany their assets.


## Trained miniature PFN

`mini_pfn.py` trains a small educational model across two opposed bear-survival rules, with one hidden rule shared by all rows in each task. It receives raw context B/L/Y and query B/L, with four context rows. A 32-dimensional row encoder and query encoder feed one cross-attention head and an MLP decoder. Neither the matching-pair indicator nor world identity nor a Bayesian answer enters training. The actual run uses 4,000 Adam steps, 256 generated tasks per step, and 7,617 learned parameters.

`mini-pfn-results.mjs` is generated from the saved run. It contains validation checkpoints, independent test metrics, all trained parameters, and complete A/B context traces. The displayed query is the same in both; changing only context labels changes prediction from 0.740281 to 0.235539. Exact Bayesian targets are 0.722747 and 0.277253. Independent test log losses are 0.630063 for the network, 0.628353 for Bayes, and 0.693147 when ignoring context. The validation curve uses the same 2,048 independent tasks at each checkpoint. This miniature is not TabPFN, is not an irrelevant-feature benchmark, and is not intended to reproduce production architectures.

The no-JavaScript version includes the mechanism and Bayesian diagrams, plus separate before/after snapshots of the recorded first Adam update. The training chapter uses that update’s own context and query; the update averages 256 task losses. Final-checkpoint predictions are reserved for the inference chapter. The numerical tests independently reconstruct both forward passes from the exported parameters, including attention softmax, value mixtures, decoder logits, and probabilities.

## Bear experiment

The bear chapter separates population probabilities, measured XGBoost results, and a proposed explanation for TabPFN. `bear-results.mjs` is generated by `../../examples/tabular-foundation-models/bear_diagnostics.py`; the SVG plot consumes its exact per-seed records and means. Its oracle reference is the population AUC logistic(1) for this balanced symmetric DGP, not an upper bound on every finite test sample. The reading version includes depth-two and depth-six results, seed standard deviations, Brier scores, and reproduction links. The simulation uses independent binary bear/strategy features, ten uniform distractors, and logistic outcome noise. It yields 73.1059% versus 26.8941% survival by response match. No TabPFN inference or attention measurement is represented as an observed result; the separate miniature PFN trace is explicitly identified.

The plotted variants hold labels and splits fixed within each seed: no distractors; continuous distractors; binarized distractors; two histogram bins; and an added feature product. Full CSV/JSON results and dependency versions are tracked with the script. `npm test` checks reported summaries against all per-seed records, exact chart data, retained scene routing, and deterministic reversible visual states.

Authoritative source is this directory in `alexanderthclark/pols4728`, branch `main`. The hosted preview is a copy of these assets. No homepage link is added by this change.

## Opening ATUS comparison

Two scroll beats precede the formal mechanism. A fixed average-precision scale shows the seven-candidate AutoGluon OOF ensemble and all four prespecified TabICLv2/TabDPT feature variants. The larger-training comparison explicitly distinguishes AutoGluon’s 136,146 fitted cases from the foundation models’ 6,468-row contexts. The next beat changes to the same 100-positive/100-negative training sample for every learner. Both use the unchanged 44,885-case official test set (526 positives).

Only weights selected from the 200-row OOF predictions define the low-data AutoGluon baseline. The inherited fixed three-model average is excluded. Feature ranking and learned preprocessing are redone using the 200 cases. Three of four foundation variants have higher observed AP, but the leading TabDPT100 difference has paired interval [-0.00205,0.00864], which includes zero. The leading feature variant is a descriptive maximum across the four reported fixed variants, not a training-selected pipeline. These are two different sample-size/class-balance regimes, not a controlled learning curve or a 200-label acquisition experiment: balancing consulted the full training pool’s labels.

`atus-results.mjs` records exact aggregate measurements, seeds, contexts and scoring conventions. No student model, per-person prediction, or withheld test outcome is published here. These measurements run TabICLv2 and TabDPT; they are distinct from the TabPFN paper evidence and the miniature teaching network. The static reading version retains both charts and the protocol qualifications.


## Early bear model comparison

Immediately after the true data-generating tree, three scroll scenes compare a depth-two CART tree, a 500-tree random forest, and a pretrained TabICLv2 checkpoint. Each successive scene retains the previous model rows. Only the continuous-noise condition is compared. Each learner receives the same two signal columns and ten independent continuous distractors. Scrolling reveals the active model’s measured seed results and then its mean AUC; previous model rows remain visible. The same scale and population-oracle reference are retained throughout, including reverse scrolling and reduced motion.

`bear-comparison-results.mjs` is generated by `../../examples/tabular-foundation-models/bear_model_comparison.py`. Every learner and representation shares the same 8,000 training and 2,000 test encounters within each of 12 fixed seeds. The ten-column regime appends seven independent distractors while preserving the original outcomes, first three noise columns, and split. Signal-only control measurements are reused only after validating their unchanged inputs and settings; all noisy-input fits are rerun. Settings are fixed rather than selected on test results. `bear-comparison-scenes.mjs` separates the recorded TabICLv2 inference from the later XGBoost diagnostic, the miniature teaching network, and the TabPFN paper. The reading version retains every chart and a continuous-noise-only table with exact means and between-seed standard deviations. The full measured benchmark, including its signal-only control condition, remains in the reproducibility files. The checkpoint revision, SHA-256, dependency versions, all per-seed results, and reproduction instructions accompany the script.


## Publication visibility

This lesson is intentionally unlisted: do not add homepage, navigation, or sitemap links. Both HTML entry points include `noindex, nofollow, nosnippet`; `build.mjs` preserves the reading page directive. Search-engine exclusion is not authentication. The public GitHub repository still exposes the source, and GitHub Pages on this repository does not provide native password protection. The separate Sites preview retains its existing sign-in requirement.


## Abstract overview before the worked bear model

Three scenes immediately follow the opening model rankings: shared pretraining across tables, the labeled context and hidden-target query as prediction inputs, and fixed-weight encoding, information exchange and decoding. The examples are abstract schematics; the bear task returns only after in-context learning has been defined. Original diagrams follow Molnar’s introductory, in-context-learning and pretraining chapters. Context rows move into a stable pretrained network, and changing the context changes the computation while its weights stay fixed. Schematic outputs are not benchmark scores or measured attention traces. The later miniature PFN supplies the recorded numerical example.


## Broad pretraining and a new task

Four scenes after slide 16 form a distinct subsection: varied illustrative SCMs and model-specific data sources; a generated table split into labeled context and hidden-target queries; shared-weight learning across tasks; and the measured TabICLv2 bear benchmark with fixed pretrained weights. TabPFN 2025 and TabICLv2 use broad synthetic generators; TabDPT uses curated real OpenML tables. The four exact seeded SCM examples are original illustrations, not recovered checkpoint-training samples. The reading version preserves every generated-table example. The return to the saved Mini-PFN checkpoint is explicit; it was not retrained on the broader SCM gallery.

`pretraining-data.mjs` defines four audited generators with varying schemas; `pretraining.mjs` keeps query truth outside model input and separates its unnumbered schematic query output from the saved twelve-seed AUC. The main sequence omits Adam/gradient formulas and the expected-log-loss proof. Numerical training details, equations, and removed teaching illustrations remain in reading background. In the binary examples, qθ(Y=1 | inputs) denotes the model's predicted probability of class 1 (survival for the bear task).
