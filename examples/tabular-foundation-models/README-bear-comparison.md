# Bear survival: fit three learners before explaining the foundation model

This is an actual held-out comparison of a greedy depth-two decision tree, a random forest, and the public TabICLv2 checkpoint. It is separate from the lesson's XGBoost diagnostics and from the miniature two-world teaching network.

Each of 12 fixed simulation seeds generates 10,000 encounters using the existing `bear_diagnostics.py:simulate` function, preserving its draw order. An unstratified 80/20 split gives every learner exactly the same 8,000 training encounters and 2,000 test encounters. Each method runs first with the two binary signal features and then with those same features plus three independent Uniform(0,1) columns. The labels, split, signal features, and noise draws are paired within each seed. No feature selector receives advance knowledge of the signal columns in the five-feature representation. All hyperparameters and the seed set were fixed before this comparison; no test labels select a model or setting.

The true conditional survival probabilities are logistic(1) = 0.7310586 when bear type and response match, and logistic(−1) = 0.2689414 otherwise. Because the signal variables are independent and balanced, neither has a population marginal association with survival. The population AUC of the true conditional probability is 0.7310586; a finite test-sample AUC may exceed that reference.

The CART learner uses Gini splits, depth2, all candidate features, and default minimum leaf sizes. The forest uses 500 bootstrap trees, unrestricted depth, `max_features="sqrt"`, and default minimum leaf sizes. The foundation-model procedure uses four TabICLv2 preprocessing ensemble members with fixed pretrained weights, CPU full precision, and native package preprocessing. All three use the data seed as their learner random state. These are fixed procedures rather than tuned best-in-family scores. Forest feature subsampling follows the ordinary square-root rule in both representations; therefore its number of candidate features also changes when columns are added.

The plot reports mean held-out ROC AUC and the separate seed results. Variation across seeds describes repeated independently simulated datasets; it is not a confidence interval. The paired changes isolate adding these distractors for the specified learners and sample size. They do not establish a general ranking on other tasks.

## Reproduce

Install the versions recorded in `comparison-results/bear-comparison-results.json` under `metadata.versions`. The existing simulation module also imports XGBoost. Fetch the pinned public checkpoint once:

```python
from huggingface_hub import hf_hub_download

checkpoint = hf_hub_download(
    repo_id="jingang/TabICL",
    filename="tabicl-classifier-v2-20260212.ckpt",
    revision="4dcd344ece2c00be9e831fdd35bed57b5ad83e19",
)
print(checkpoint)
```

From the repository root:

```bash
python examples/tabular-foundation-models/bear_model_comparison.py
```

The script verifies checkpoint SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` before inference. Its default path is the standard Hugging Face cache path for the pinned revision. Use `--checkpoint` to provide the returned download path if a custom cache directory is configured. The output directory contains per-seed CSV records, JSON summaries, paired differences, timings, and probability hashes. Interrupted work resumes from the same-protocol progress JSON without rerunning complete cells. Each foundation-model fit runs in a fresh subprocess to release native CPU inference buffers between cells. `bear_tabicl_cpu_batching.py` bounds independent column groups, rows, and ensemble members at 8, 32, and 1 batch elements; it never partitions an attention sequence or removes an encounter. All final foundation-model measurements use this adapter. Its source hash and execution settings are recorded in the metadata. The generated website ES module contains those same aggregate measurements and per-seed metrics; it contains no individual encounter outcomes or probabilities. `comparison-results/execution-validation.json` records exact held-out AUC agreement between the native and bounded execution for seed0 with distractors, with Brier difference below 3e-9. `check_bear_tabicl_cpu_batching.py` checks native versus bounded attention outputs and populated caches in 12 small-network configurations; maximum observed output error was 2.09e-7, within the stated floating-point tolerance.

TabICLv2 is described by Qu et al. (2026), [*TabICLv2: A better, faster, scalable, and open tabular foundation model*](https://arxiv.org/abs/2602.11139). [Its public checkpoint card](https://huggingface.co/jingang/TabICL) identifies the BSD3-Clause license. [The implementation](https://github.com/soda-inria/tabicl) provides the inference code.
