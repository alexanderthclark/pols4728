"""Compare resampling choices using untouched validation folds and test data.

Install: python -m pip install scikit-learn imbalanced-learn
Run: python imblearn_demo.py
"""
from collections import Counter

from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split


X, y = make_classification(
    n_samples=4000, n_features=10, n_informative=5,
    n_redundant=2, weights=[0.97, 0.03], flip_y=0,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42,
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# Each validation fold and the test set have only 30 positives; AP can be noisy.

# A sampler changes BOTH X and y. Use imblearn's Pipeline.
# GridSearchCV calls fit on each training fold; validation is not resampled.
pipeline = Pipeline([
    ("sample", "passthrough"),
    ("forest", RandomForestClassifier(
        n_estimators=100, criterion="entropy", random_state=42, n_jobs=1,
    )),
])
common = {
    "forest__max_depth": [4, None],
    "forest__min_samples_leaf": [1, 5],
}
grid = [
    {**common, "sample": ["passthrough"]},
    {**common,
     "sample": [RandomUnderSampler(random_state=42)],
     "sample__sampling_strategy": [0.1, 0.25, 1.0]},
]
search = GridSearchCV(
    pipeline, grid, cv=cv, scoring="average_precision",
    n_jobs=-1, refit=True, error_score="raise",
)
search.fit(X_train, y_train)

# A separate comparison: the forest creates a fresh balanced draw per tree.
# Explicit sampling settings avoid ambiguity across imblearn versions.
balanced_search = GridSearchCV(
    BalancedRandomForestClassifier(
        n_estimators=100, criterion="entropy", sampling_strategy="all",
        replacement=True, bootstrap=False,
        random_state=42, n_jobs=1,
    ),
    {"max_depth": [4, None], "min_samples_leaf": [1, 5]},
    cv=cv, scoring="average_precision", n_jobs=-1,
    refit=True, error_score="raise",
)
balanced_search.fit(X_train, y_train)

print("Training counts:", Counter(y_train))
print("Untouched test counts:", Counter(y_test))
print("Best CV AP by sampling choice (mean ± fold SD):")
results = search.cv_results_
for ratio in [None, 0.1, 0.25, 1.0]:
    candidates = [i for i, params in enumerate(results["params"])
                  if params.get("sample__sampling_strategy") == ratio]
    best = max(candidates, key=lambda i: results["mean_test_score"][i])
    label = "No resampling" if ratio is None else f"Minority/majority={ratio:g}"
    print(f"  {label}: {results['mean_test_score'][best]:.4f} ± "
          f"{results['std_test_score'][best]:.4f}")
spread = balanced_search.cv_results_["std_test_score"][balanced_search.best_index_]
print(f"  Balanced forest: {balanced_search.best_score_:.4f} ± {spread:.4f}")
for label, result in [
    ("Resampling pipeline", search),
    ("Balanced forest", balanced_search),
]:
    print(label, "parameters:", result.best_params_)

# Choose the method on CV results, then evaluate that choice once on test data.
selected = max([search, balanced_search], key=lambda result: result.best_score_)
score = selected.predict_proba(X_test)[:, 1]
print("Selected model test AP:", round(average_precision_score(y_test, score), 4))
print("Selected model test ROC AUC:", round(roc_auc_score(y_test, score), 4))
# AP and ROC AUC assess scores, not a chosen classification threshold.
# Tune thresholds or calibrate probabilities using separate training-only folds.
