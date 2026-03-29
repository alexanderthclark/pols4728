(evaluating_predictive_models)=
# Evaluating Predictive Models

Evaluation is the center of predictive modeling. Fit quality on training data is not enough; we need unbiased estimates of out-of-sample performance.

## Canon objectives

By the end of this lecture, you should be able to:

- choose an evaluation design that matches the data-generating process,
- separate model selection from final performance reporting,
- identify and eliminate leakage pathways in reproducible pipelines.

## Training versus test

Training error is optimistic because parameters are chosen to minimize it. Generalization performance must be estimated on data not used for fitting.

## Holdout validation

Simple workflow:

1. Split data into train and validation sets.
2. Fit candidate models on train.
3. Compare validation metrics.

```{figure} ../assets/images/line_simple_holdout_validation_k.pdf
:width: 80%
:align: center

Simple holdout validation workflow.
```

Holdout is fast but high-variance in smaller samples.

## K-fold cross-validation

K-fold CV averages across many train/validation splits.

$$
\widehat{R}_{\mathrm{CV}}=\frac{1}{K}\sum_{k=1}^K
\frac{1}{|I_k|}\sum_{i\in I_k}\ell\left(y_i,\hat f^{(-k)}(x_i)\right).
$$

Use CV for hyperparameter tuning and model comparison.

Typical defaults in this course are `K=5` or `K=10`, with stratification for classification.

## Nested cross-validation

Nested CV separates tuning from performance estimation:

- Inner loop: select hyperparameters.
- Outer loop: estimate generalization of that tuning process.

Use nested CV when model search is broad or sample size is limited.

## The wrong way and the right way

Wrong:

- preprocess (scale/select features/impute) on full data,
- then cross-validate model.

Right:

- build one pipeline containing preprocessing + model,
- run CV on the full pipeline so each fold's preprocessing is learned from training-fold data only.

Common leakage sources:

- scaling or imputation fit on full data before splitting,
- feature selection done once globally before CV,
- using test performance to choose model family or threshold.

## Production evaluation checklist

- Declare split design before fitting.
- Keep a final untouched test set.
- Align metric choice with deployment objective.
- Report uncertainty (fold variability, bootstrap intervals, or repeated CV).
- Document every transformation inside the resampling loop.
- Freeze the final specification before touching test data.

Reliable model development is mostly reliable evaluation.
