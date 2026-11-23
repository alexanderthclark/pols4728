"""Compare fixed learners on the paired bear-survival simulation.

All learners get the same 8,000 training rows and 2,000 held-out rows per seed.
The two representations preserve the signal, labels, and split and differ only
by adding ten independent Uniform(0, 1) distractors. No model is tuned here.
"""
from __future__ import annotations

import argparse
import csv
from contextlib import nullcontext
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REVISION = '4dcd344ece2c00be9e831fdd35bed57b5ad83e19'
FILENAME = 'tabicl-classifier-v2-20260212.ckpt'
CHECKPOINT_SHA256 = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'
MODEL_LABELS = {
    'tree_depth2': 'Decision tree · depth 2',
    'random_forest': 'Random forest',
    'tabicl_v2': 'TabICLv2',
}
CASE_LABELS = {
    'signal_only': 'Bear type and strategy only',
    'continuous_noise': 'Add ten continuous noise columns',
}


def digest(path):
    with Path(path).open('rb') as stream:
        h = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=10_000)
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(12)))
    parser.add_argument('--noise-columns', type=int, default=10)
    parser.add_argument('--reuse-signal-results', type=Path, help='Reuse unchanged signal-only cells from a compatible saved run; all distractor cells are still refitted.')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--forest-threads', type=int, default=6)
    parser.add_argument('--checkpoint', type=Path, default=Path.home() / '.cache/huggingface/hub/models--jingang--TabICL/snapshots' / REVISION / FILENAME)
    parser.add_argument('--output-dir', type=Path, default=HERE / 'comparison-results')
    parser.add_argument('--web-output', type=Path, default=REPO / 'docs/tabular-foundation-models/bear-comparison-results.mjs')
    parser.add_argument('--prepare-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--summarize-only', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--tabicl-cell', nargs=2, metavar=('SEED', 'CASE'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.n < 100 or args.noise_columns < 1 or args.threads < 1 or args.forest_threads < 1 or len(set(args.seeds)) != len(args.seeds) or not args.seeds or min(args.seeds) < 0:
        parser.error('n >= 100, positive noise-column/thread counts, and distinct nonnegative seeds are required')
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[variable] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = '1'

    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import brier_score_loss, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier
    from bear_simulation import simulate

    CASE_LABELS['continuous_noise'] = f'Add {args.noise_columns} continuous noise columns'
    if args.tabicl_cell:
        import torch
        from tabicl import TabICLClassifier
        torch.set_num_threads(args.threads)
    if digest(args.checkpoint) != CHECKPOINT_SHA256:
        raise ValueError('The checkpoint hash differs from the pinned public TabICLv2 checkpoint')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        'experiment': 'Bear survival: fixed learners with and without continuous distractors',
        'n_noise': args.noise_columns, 'n_features': 2 + args.noise_columns,
        'n_total': args.n, 'n_train': int(args.n * .8), 'n_test': args.n - int(args.n * .8),
        'seeds': args.seeds, 'tuned': False, 'test_size': .2, 'split_stratified': False,
        'split_random_state': 'same integer as data seed', 'metric': 'ROC AUC',
        'uncertainty': 'Dots are separate simulated datasets; SD is sample standard deviation across seeds, not a confidence interval.',
        'paired_design': 'Within each seed every learner and representation shares B, A, all noise draws, Y, and train/test indices. Only the independent distractor columns are added.',
        'dgp': f'B,A independently Bernoulli(0.5); Z1,...,Z{args.noise_columns} independently Uniform(0,1); E independently standard Logistic; Y=1[-1+2*1[B=A]+E>0].',
        'nested_distractors': 'Original B/L/first-three-noise/outcome-error draws retain their original order. Additional columns use default_rng(SeedSequence([seed,4728,20261123])) and are drawn column-by-column after the original task is fixed.',
        'p_survival_match': float(1 / (1 + np.exp(-1))),
        'p_survival_mismatch': float(1 / (1 + np.exp(1))),
        'population_oracle_auc': float(1 / (1 + np.exp(-1))),
        'oracle_note': 'Population AUC of the true conditional probability, not an upper bound on every finite test-sample AUC.',
        'model_settings': {
            'tree_depth2': {'criterion': 'gini', 'splitter': 'best', 'max_depth': 2, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': None, 'random_state': 'data seed'},
            'random_forest': {'n_estimators': 500, 'criterion': 'gini', 'max_depth': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'bootstrap': True, 'n_jobs': args.forest_threads, 'random_state': 'data seed'},
            'tabicl_v2': {'n_estimators': 4, 'batch_size': 1, 'kv_cache': True, 'device': 'cpu', 'n_jobs': args.threads, 'random_state': 'data seed', 'use_amp': False, 'use_fa3': False, 'offload_mode': False, 'allow_auto_download': False, 'checkpoint_version': FILENAME, 'other_settings': f'tabicl {version("tabicl")} defaults'},
        },
        'checkpoint': {'repo_id': 'jingang/TabICL', 'revision': REVISION, 'filename': FILENAME, 'sha256': CHECKPOINT_SHA256},
        'versions': {name: version(name) for name in ('numpy', 'scikit-learn', 'torch', 'tabicl', 'xgboost')},
        'python': platform.python_version(),
        'cpu_execution': {'adapter_module': 'examples/tabular-foundation-models/bear_tabicl_cpu_batching.py', 'adapter_sha256': digest(HERE / 'bear_tabicl_cpu_batching.py'), 'stage_batch_limits': {'column': 1, 'row': 8, 'icl': 1}, 'attention_sequences_split': False, 'fresh_subprocess_per_foundation_fit': True, 'parity_evidence': 'examples/tabular-foundation-models/comparison-results/execution-validation.json', 'disk_cache_adapter_module': 'examples/tabular-foundation-models/bear_tabicl_disk_cache.py', 'disk_cache_adapter_sha256': digest(HERE / 'bear_tabicl_disk_cache.py'), 'disk_cache_parity_evidence': 'examples/tabular-foundation-models/comparison-results/disk-cache-check.json', 'controller': 'stdlib-only parent; sklearn phase exits before fresh per-cell foundation workers'},
        'simulation_source': 'examples/tabular-foundation-models/bear_simulation.py:simulate',
        'simulation_source_sha256': digest(HERE / 'bear_simulation.py'),
        'script_source': 'examples/tabular-foundation-models/bear_model_comparison.py',
        'script_sha256': digest(__file__),
    }
    metadata['cpu_execution'].update({
        'stage_batch_parity_evidence': 'examples/tabular-foundation-models/comparison-results/cpu-batching-check.json',
        'stage_batch_parity_sha256': digest(HERE / 'comparison-results/cpu-batching-check.json'),
        'disk_cache_parity_sha256': digest(HERE / 'comparison-results/disk-cache-check.json'),
    })
    metadata['paired_task_hashes'] = []
    for seed in args.seeds:
        variants, outcomes = simulate(seed, args.n, n_noise=args.noise_columns)
        legacy, legacy_outcomes = simulate(seed, args.n, n_noise=3)
        if not np.array_equal(outcomes, legacy_outcomes) or not np.array_equal(variants['signal_only'], legacy['signal_only']):
            raise ValueError('Adding distractors changed the original signal task')
        if args.noise_columns >= 3 and not np.array_equal(variants['continuous_noise'][:, :5], legacy['continuous_noise']):
            raise ValueError('Adding distractors changed the original first three columns')
        train, test = train_test_split(np.arange(args.n), test_size=.2, random_state=seed)
        def array_hash(array, dtype):
            return sha256(np.asarray(array, dtype=dtype).tobytes()).hexdigest()
        metadata['paired_task_hashes'].append({
            'seed': seed,
            'signal_sha256': array_hash(variants['signal_only'], '<i8'),
            'outcomes_sha256': array_hash(outcomes, '<i8'),
            'train_indices_sha256': array_hash(train, '<i8'),
            'test_indices_sha256': array_hash(test, '<i8'),
            'original_three_noise_sha256': array_hash(legacy['continuous_noise'][:, 2:], '<f8'),
            'current_noise_sha256': array_hash(variants['continuous_noise'][:, 2:], '<f8'),
        })
    reused_records = []
    metadata['signal_only_reuse'] = None
    if args.reuse_signal_results:
        source = json.loads(args.reuse_signal_results.read_text())
        old = source['metadata']
        for key in ('n_total', 'n_train', 'n_test', 'test_size', 'split_stratified',
                    'split_random_state', 'tuned', 'versions', 'checkpoint',
                    'model_settings'):
            if old.get(key) != metadata.get(key):
                raise ValueError(f'Reuse source has incompatible {key}')
        for key in ('adapter_module', 'adapter_sha256', 'attention_sequences_split', 'fresh_subprocess_per_foundation_fit'):
            if old['cpu_execution'].get(key) != metadata['cpu_execution'].get(key):
                raise ValueError(f'Reuse source has incompatible CPU execution {key}')
        if not set(args.seeds).issubset(old['seeds']):
            raise ValueError('Reuse source does not contain every requested seed')
        source_digest = digest(args.reuse_signal_results)
        for seed in args.seeds:
            variants, y = simulate(seed, args.n, n_noise=args.noise_columns)
            _, test = train_test_split(np.arange(args.n), test_size=.2, random_state=seed)
            oracle = np.where(variants['signal_only'][test, 0] == variants['signal_only'][test, 1],
                              metadata['p_survival_match'], metadata['p_survival_mismatch'])
            oracle_auc = float(roc_auc_score(y[test], oracle))
            for model_id in MODEL_LABELS:
                matches = [r for r in source['records'] if r['seed'] == seed
                           and r['id'] == 'signal_only' and r['model'] == model_id]
                if len(matches) != 1:
                    raise ValueError(f'Reuse source has missing or repeated signal cell {seed}/{model_id}')
                record = matches[0]
                if record['n_features'] != 2 or record['n_train'] != metadata['n_train'] or record['n_test'] != metadata['n_test']:
                    raise ValueError('Reuse source has incompatible signal input shape')
                if abs(record['oracle_test_auc'] - oracle_auc) > 1e-15:
                    raise ValueError('Reuse source has different outcomes or split')
                reused_records.append({**record, 'reused_signal_source_sha256': source_digest})
        metadata['signal_only_reuse'] = {
            'source_sha256': source_digest,
            'source_filename': args.reuse_signal_results.name,
            'source_noise_columns': old.get('n_noise', 3),
            'cells_reused': len(reused_records),
            'source_stage_batch_limits': old['cpu_execution']['stage_batch_limits'],
            'current_stage_batch_limits': metadata['cpu_execution']['stage_batch_limits'],
            'execution_note': 'Unchanged controls used native cached storage and stage limits 8/32/1. Current distractor fits use the same per-member cached calculations with disk-streamed native caches and independent stage limits 1/8/1. Small-network stage/cache parity and end-to-end four-member classifier parity are recorded; no attention sequence, row, feature or ensemble member is removed.',
            'reason': 'Adding independent distractor columns leaves the two-column signal-only inputs, outcomes, and splits unchanged. Model settings, versions, checkpoint, execution adapter, and per-seed oracle test scores were checked before reuse.',
        }
    progress_path = args.output_dir / 'bear-comparison-progress.json'
    if progress_path.exists():
        previous = json.loads(progress_path.read_text())
        if previous['metadata'] != metadata:
            raise ValueError('Saved progress has a different protocol; use a fresh output directory')
        records = previous['records']
    else:
        records = reused_records
    for seed in args.seeds:
        if args.summarize_only:
            continue
        if args.tabicl_cell and seed != int(args.tabicl_cell[0]):
            continue
        variants, y = simulate(seed, args.n, n_noise=args.noise_columns)
        train, test = train_test_split(np.arange(args.n), test_size=.2, random_state=seed)
        oracle_probability = np.where(variants['signal_only'][:, 0] == variants['signal_only'][:, 1], metadata['p_survival_match'], metadata['p_survival_mismatch'])
        oracle_auc = float(roc_auc_score(y[test], oracle_probability[test]))
        for case_id in CASE_LABELS:
            if args.tabicl_cell and case_id != args.tabicl_cell[1]:
                continue
            X = variants[case_id]
            for model_id in MODEL_LABELS:
                if args.prepare_only and model_id == 'tabicl_v2':
                    continue
                if args.tabicl_cell and model_id != 'tabicl_v2':
                    continue
                if any(r['seed'] == seed and r['id'] == case_id and r['model'] == model_id for r in records):
                    continue
                if model_id == 'tabicl_v2' and not args.tabicl_cell:
                    # A fresh interpreter releases all native inference buffers
                    # after this fixed cell, without changing its computation.
                    subprocess.run([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--tabicl-cell', str(seed), case_id], check=True)
                    records = json.loads(progress_path.read_text())['records']
                    continue
                print(json.dumps({'phase': 'start', 'seed': seed, 'id': case_id, 'model': model_id}), flush=True)
                started = time.perf_counter()
                if model_id == 'tree_depth2':
                    model = DecisionTreeClassifier(max_depth=2, criterion='gini', random_state=seed)
                elif model_id == 'random_forest':
                    model = RandomForestClassifier(n_estimators=500, max_features='sqrt', bootstrap=True, n_jobs=args.forest_threads, random_state=seed)
                else:
                    model = TabICLClassifier(n_estimators=4, batch_size=1, kv_cache=True, device='cpu', n_jobs=args.threads, random_state=seed, model_path=args.checkpoint, checkpoint_version=FILENAME, use_amp=False, use_fa3=False, offload_mode=False, allow_auto_download=False)
                if model_id == 'tabicl_v2':
                    from bear_tabicl_cpu_batching import cpu_batched_tabicl
                    scope = cpu_batched_tabicl(column_batch_size=1, row_batch_size=8, icl_batch_size=1)
                else:
                    scope = nullcontext()
                with scope:
                    if model_id == 'tabicl_v2':
                        from bear_tabicl_disk_cache import disk_backed_kv_cache
                        with TemporaryDirectory(prefix='bear-tabicl-kv-', dir=args.output_dir) as cache_dir:
                            with disk_backed_kv_cache(cache_dir):
                                model.fit(X[train], y[train])
                                fit_seconds = time.perf_counter() - started
                                probability = model.predict_proba(X[test])[:, 1]
                    else:
                        model.fit(X[train], y[train])
                        fit_seconds = time.perf_counter() - started
                        probability = model.predict_proba(X[test])[:, 1]
                if not np.isfinite(probability).all():
                    raise ValueError('Nonfinite held-out probabilities')
                record = {'model': model_id, 'id': case_id, 'seed': seed, 'n_features': X.shape[1], 'n_train': len(train), 'n_test': len(test), 'auc': float(roc_auc_score(y[test], probability)), 'brier': float(brier_score_loss(y[test], probability)), 'oracle_test_auc': oracle_auc, 'fit_seconds': fit_seconds, 'total_seconds': time.perf_counter() - started, 'prediction_sha256': sha256(np.asarray(probability, dtype='<f8').tobytes()).hexdigest()}
                if model_id == 'tree_depth2':
                    root = int(model.tree_.feature[0])
                    record['root_feature'] = root
                    record['root_kind'] = 'signal' if 0 <= root < 2 else ('noise' if root >= 2 else 'leaf')
                records.append(record)
                temporary = progress_path.with_suffix('.json.tmp')
                temporary.write_text(json.dumps({'metadata': metadata, 'records': records}, indent=2, allow_nan=False) + '\n')
                temporary.replace(progress_path)
                print(json.dumps({'phase': 'complete', **record}), flush=True)
                del model
                if args.tabicl_cell:
                    return
    if args.tabicl_cell or args.prepare_only:
        return
    cases, paired_deltas = [], []
    for model_id, model_label in MODEL_LABELS.items():
        for case_id, case_label in CASE_LABELS.items():
            rows = [r for r in records if r['model'] == model_id and r['id'] == case_id]
            assert len(rows) == len(args.seeds)
            case = {'model': model_id, 'model_label': model_label, 'id': case_id, 'label': case_label, 'n_seeds': len(rows)}
            for metric in ('auc', 'brier'):
                values = [r[metric] for r in rows]
                case[f'mean_{metric}'] = float(np.mean(values))
                case[f'sd_{metric}'] = float(np.std(values, ddof=1)) if len(values) > 1 else None
            cases.append(case)
        baseline = {r['seed']: r for r in records if r['model'] == model_id and r['id'] == 'signal_only'}
        noisy = [r for r in records if r['model'] == model_id and r['id'] == 'continuous_noise']
        values = [{'seed': r['seed'], 'auc_delta': r['auc'] - baseline[r['seed']]['auc'], 'brier_delta': r['brier'] - baseline[r['seed']]['brier']} for r in noisy]
        paired_deltas.append({'model': model_id, 'comparison': 'continuous_noise minus signal_only', 'mean_auc_delta': float(np.mean([r['auc_delta'] for r in values])), 'sd_auc_delta': float(np.std([r['auc_delta'] for r in values], ddof=1)) if len(values) > 1 else None, 'per_seed': values})
    result = {'metadata': metadata, 'cases': cases, 'records': records, 'paired_deltas': paired_deltas}
    text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    (args.output_dir / 'bear-comparison-results.json').write_text(text + '\n')
    with (args.output_dir / 'bear-comparison-records.csv').open('w', newline='') as stream:
        fields = sorted({key for row in records for key in row})
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    args.web_output.parent.mkdir(parents=True, exist_ok=True)
    args.web_output.write_text('// Generated by examples/tabular-foundation-models/bear_model_comparison.py.\n// Actual held-out measurements; fixed settings, no test-label selection.\nexport const bearComparisonResults = ' + text + ';\n')
    for case in cases:
        print(f"{case['model']:15} {case['id']:18} AUC={case['mean_auc']:.6f} SD={case['sd_auc']}")


def run():
    # The orchestration parent never imports NumPy, sklearn or Torch. The sklearn
    # preparation phase exits before any foundation worker starts, releasing all
    # native forest allocations instead of retaining them beside a large model.
    worker_flags = {'--tabicl-cell', '--prepare-only', '--summarize-only', '--help', '-h'}
    if worker_flags.intersection(sys.argv[1:]):
        main()
        return
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument('--seeds', type=int, nargs='+', default=list(range(12)))
    options.add_argument('--output-dir', type=Path, default=HERE/'comparison-results')
    args, _ = options.parse_known_args()
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    subprocess.run(command+['--prepare-only'], check=True)
    progress = args.output_dir/'bear-comparison-progress.json'
    for seed in args.seeds:
        for case_id in CASE_LABELS:
            current = json.loads(progress.read_text())['records']
            if any(r['seed'] == seed and r['id'] == case_id and r['model'] == 'tabicl_v2' for r in current):
                continue
            subprocess.run(command+['--tabicl-cell', str(seed), case_id], check=True)
    subprocess.run(command+['--summarize-only'], check=True)


if __name__ == '__main__':
    run()
