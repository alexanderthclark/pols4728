# LLM corrections: storyboard for review

This is a working scrollytelling storyboard, not the finished lesson. Open
`storyboard.html` through a local HTTP server rooted at `docs/`. It uses the
course's existing fonts, KaTeX, palette, and reversible scroll convention.

## Scope

Expert-coded score X; fixed automated score Q; fully observed downstream outcome
Y. A simple random sample S receives expert coding; A is the full corpus.
Predictions are fixed before the audit, so this introductory case requires
neither learned sampling probabilities nor cross-fitting. The general DSL
estimator has the same correction structure; its doubly robust theory and
optional prediction-improvement stage are deferred.

The target mean is the expert-coded average, not the already observed average
automated score. A representative corpus supports interpretation as a broader
population average; the expert audit does not itself repair corpus selection.
The regression target is the descriptive regression of Y on X with an intercept.

## Sequence

1. Introduce the measurement problem.
2. Walk through the paper's LLM-only four-step procedure.
3. Compare its classical supervised-learning procedure.
4. Introduce DSL's four steps, skipping optional prediction improvement.
5. Correct a mean: the full-corpus prediction average plus the audit's paired
   expert-minus-prediction average. The 40% to 34% binary example is an
   illustrative teaching example, separate from the 40-post regression corpus.
6. Reuse the 40-post example: the correct overall mean can coexist with the
   wrong regression slope.
7. Reveal the original 16-post random audit.
8. Correct the X, X-squared, and XY moments separately, then assemble the slope.
9. Compare automated, expert-only, DSL, and full-expert benchmark estimates.
10. Distinguish population regression uncertainty from audit-only variation.

The regression animation must not imply that DSL recovers every post's expert
score. Full labels are shown only to explain/evaluate this synthetic example.
All feasible corrections use the original 16 audited labels. Corrected raw
moments are design-unbiased; the slope is a nonlinear ratio and need not be
exactly unbiased in this small sample.

## Source and numerical provenance

Naoki Egami, Musashi Hinck, Brandon M. Stewart, and Hanying Wei,
[Using Large Language Model Annotations for the Social Sciences: A General
Framework of Using Predicted Variables in Downstream Analyses](https://naokiegami.com/paper/dsl_ss.pdf).
The numbered procedures are in sections 3.2 and 4.1; optional prediction
improvement is explicit in footnote 7 of the current PDF. The mean correction
specializes equation (5); the regressor-moment correction specializes equation
(8), section 4.3.3. These are paraphrases and an explicit OLS specialization.

The synthetic example reuses `dsl_biased_regressor.ipynb` created November 16,
2026. Its current contents were read before preparing this storyboard, including
the later population-standard-error additions. It is not an empirical result
from the paper.

Run `python3 verify_example.py` to reproduce the original point estimates and
linearized population SEs and regenerate `example-results.json`. This uses only
the Python standard library and the original 16 audited post IDs. The balanced
corpus is exactly the notebook's construction. The original NumPy audit seed
was 2026; 5,000-audit simulation seed was 1701.

| Method | Slope | Population SE, linearized |
| --- | ---: | ---: |
| Full-expert benchmark | 2.000 | — |
| Automated | 3.111 | — |
| Expert-only audit | 1.706 | 1.158 |
| DSL | 2.061 | 0.802 |

The 5,000-audit RMSEs (expert-only 0.979, DSL 0.194) are retained notebook outputs,
not rerun by this small verification script. They condition on the same 40
posts and are not population regression standard errors. The latest notebook
also checks fresh-corpus coverage and power; those additional experiments are
outside this first storyboard. Precision gains depend on the predictions and
sampling design and are not guaranteed.

## Review

Review narrative ordering, the two distinct targets, and the moment-by-moment
animation before expanding this into the final lesson. The course homepage
and published main branch are unchanged.
