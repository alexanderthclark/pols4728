(learning)=
# Learning

## What is machine learning?

Machine learning is the practice of building algorithms that improve with experience.
A useful working definition is Mitchell's formulation: a program learns from experience `E`
for task `T` under performance measure `P` when performance improves with more experience.

This framing is useful for social-science workflows because it forces three explicit choices:
what we want to predict, how we will measure success, and what data will count as learning input.

## Well-posed learning problems

A well-posed problem clearly specifies:

- `T` (task): what the model should do.
- `P` (performance): how success is evaluated.
- `E` (experience): what data and feedback are used during training.

### Example: wage prediction

- `T`: predict wages from observed covariates (education, experience, sector).
- `P`: mean squared error on held-out data.
- `E`: labeled observations with wages and predictors.

### Example: board-game play

- `T`: choose strong game moves.
- `P`: win rate against benchmark opponents.
- `E`: supervised game records and/or self-play trajectories.

## Types of learning tasks

### Supervised learning

Supervised learning uses labeled outcomes and is the dominant mode in this course.

- Regression: continuous outcome (wage, turnout share, risk score).
- Classification: categorical outcome (approve/deny, winner/loser, label class).

### Unsupervised learning

Unsupervised learning has no target label and focuses on structure discovery.

- Clustering similar observations.
- Dimensionality reduction for high-dimensional data.
- Topic/theme discovery in text corpora.

## Performance measures

Different metrics encode different priorities. For binary classification, define positive as the
policy-relevant event (e.g., high risk).

|                       | Predicted Positive | Predicted Negative |
|-----------------------|--------------------|--------------------|
| Actually Positive     | True Positive      | False Negative     |
| Actually Negative     | False Positive     | True Negative      |

Common metrics:

- Precision: `TP / (TP + FP)`
- Recall: `TP / (TP + FN)`
- F1 score: harmonic mean of precision and recall
- ROC/AUC: threshold-free ranking performance summary

```{figure} ../assets/scatter_precision_recall_f1.svg
:width: 85%
:align: center

Precision, recall, and F1 as alternative evaluation views.
```

No metric is universally best; metric choice is part of the substantive research design.

## Experience and training

For reproducible model evaluation, separate:

- training data (fit parameters),
- validation data (tune decisions),
- test data (final generalization estimate).

This prevents optimistic performance claims from accidental leakage.

## Summary

Machine learning in social science is less about model novelty and more about disciplined choices:
clear task definition, aligned metrics, and valid evaluation splits.
