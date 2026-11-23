# Bear survival: fit three learners before explaining the foundation model

This is an actual held-out comparison of a greedy depth-two decision tree, a random forest, and the public TabICLv2 checkpoint. It is separate from the lesson's XGBoost diagnostics and from the miniature two-world teaching network.

Each of 12 fixed simulation seeds generates 10,000 encounters using the shared `bear_simulation.py:simulate` function. The original draw order for bear type, response, the first three noise columns, and outcome noise is preserved exactly. Seven additional independent noise columns are appended from a separate seeded random-number stream. An unstratified 80/20 split gives every learner exactly the same 8,000 training encounters and 2,000 test encounters. Each method runs first with the two binary signal features and then with those same features plus ten independent Uniform(0,1) columns. The labels, split, signal features, and noise draws are paired within each seed. No feature selector receives advance knowledge of the signal columns in the twelve-feature representation. All hyperparameters and the seed set were fixed before this comparison; no test labels select a model or setting.

The true conditional survival probabilities are logistic(1) = 0.7310586 when bear type and response match, and logistic(−1) = 0.2689414 otherwise. Because the signal variables are independent and balanced, neither has a population marginal association with survival. The population AUC of the true conditional probability is 0.7310586; a finite test-sample AUC may exceed that reference.

The CART learner uses Gini splits, depth2, all candidate features, and default minimum leaf sizes. The forest uses 500 bootstrap trees, unrestricted depth, `max_features="sqrt"`, and default minimum leaf sizes. The foundation-model procedure uses four TabICLv2 preprocessing ensemble members with fixed pretrained weights, CPU full precision, and native package preprocessing. All three use the data seed as their learner random state. These are fixed procedures rather than tuned best-in-family scores. Forest feature subsampling follows the ordinary square-root rule in both representations: the twelve-column task normally offers three randomly selected candidate features at each split. No depth cap is applied to the forest.

The plot reports mean held-out ROC AUC and the separate seed results for the ten-distractor condition only. The signal-only control remains in the archived reproducibility output. Variation across seeds describes repeated independently simulated datasets; it is not a confidence interval. The paired changes isolate adding these distractors for the specified learners and sample size. They do not establish a general ranking on other tasks.

## Reproduce

Install the versions recorded in `comparison-results/bear-comparison-results.json` under `metadata.versions`. Fetch the pinned public checkpoint once:

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
python examples/tabular-foundation-models/bear_model_comparison.py --noise-columns 10
```

The published update reuses the 36 unchanged signal-only model/seed cells from the original three-distractor experiment and refits all 36 ten-distractor cells. Reuse is explicitly recorded per cell and in `metadata.signal_only_reuse`; the original JSON has SHA-256 `a42155237a72a3c561c58aafb0f63b9a9d09aec62fdf2e3b804f5510de50a204`. Before reusing a cell, the runner checks its settings, dependency versions, checkpoint, CPU execution adapter, row counts, feature count, and per-seed oracle test AUC. The original controls used stage batch limits 8/32/1, while the twelve-column task uses 1/8/1 to bound peak CPU memory; this difference is recorded, and only independent batch dimensions change. The shared generator retains the original signal-only inputs and outcomes bitwise. Per-seed hashes of signal values, outcomes, split indices, original noise columns, and current noise columns are recorded in `metadata.paired_task_hashes`.

The default command above recomputes every cell from scratch. To reproduce the published update while reusing the same unchanged controls, first obtain the original immutable results:

```python
from urllib.request import urlretrieve

urlretrieve(
    "https://raw.githubusercontent.com/alexanderthclark/pols4728/049971dd5fb93911d41e17e8b8ce3c37de3a9bce/examples/tabular-foundation-models/comparison-results/bear-comparison-results.json",
    "bear-three-noise-results.json",
)
```

```bash
python examples/tabular-foundation-models/bear_model_comparison.py --noise-columns 10 --reuse-signal-results bear-three-noise-results.json
```

Use a fresh `--output-dir` when changing the execution protocol; an existing progress checkpoint must match all current metadata. The added columns use `default_rng(SeedSequence([seed, 4728, 20261123]))`; extra columns are drawn column-by-column so larger extensions retain the same earlier distractors.

Published metadata records SHA-256 hashes of the runner, shared generator, execution adapters, and current parity evidence files. Each parity report also identifies its checker source; the end-to-end cache report verifies and records the pinned checkpoint and dependency versions.

The script verifies checkpoint SHA-256 `bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0` before inference. Its default path is the standard Hugging Face cache path for the pinned revision. Use `--checkpoint` to provide the returned download path if a custom cache directory is configured. The output directory contains per-seed CSV records, JSON summaries, paired differences, timings, and probability hashes. Interrupted work resumes from the same-protocol progress JSON without rerunning complete cells. A standard-library-only controller runs the scikit-learn phase to completion and lets that process exit before launching the foundation-model fits in fresh subprocesses. This releases forest and inference allocations between phases. `bear_tabicl_cpu_batching.py` bounds independent column groups, rows, and ensemble members at 1, 8, and 1 batch elements; it never partitions an attention sequence or removes an encounter. The newly fitted ten-distractor foundation-model cells also use `bear_tabicl_disk_cache.py`: each native ensemble-member cache is written to an ordinary temporary directory inside `--output-dir`, then memory-mapped one member at a time. This avoids retaining and concatenating all four native caches at once. The directory is removed after prediction. Use an output directory on a disk-backed filesystem rather than a RAM-backed temporary filesystem. Preprocessing views, member order, cached attention calculations, class permutations, per-member logits, and native final logit averaging are unchanged. `comparison-results/disk-cache-check.json` records an end-to-end twelve-feature, four-member parity check against ordinary native cached execution: probabilities, AUC and Brier score agreed exactly on 512 training and 64 test rows. Its source hash and execution settings are recorded in the metadata. The generated website ES module contains those same aggregate measurements and per-seed metrics; it contains no individual encounter outcomes or probabilities. `comparison-results/execution-validation.json` records exact held-out AUC agreement between the native and bounded execution for seed0 with the original three distractors, with Brier difference below 3e-9. `check_bear_tabicl_cpu_batching.py` checks native versus bounded attention outputs and populated caches in 12 small-network configurations with twelve input features and the current 1/8/1 execution limits; maximum observed output error was 2.38e-7, within the stated floating-point tolerance.

TabICLv2 is described by Qu et al. (2026), [*TabICLv2: A better, faster, scalable, and open tabular foundation model*](https://arxiv.org/abs/2602.11139). [Its public checkpoint card](https://huggingface.co/jingang/TabICL) identifies the BSD3-Clause license. [The implementation](https://github.com/soda-inria/tabicl) provides the inference code.
