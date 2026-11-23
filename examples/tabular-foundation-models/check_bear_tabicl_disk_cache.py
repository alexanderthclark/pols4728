"""Check native versus streamed four-member TabICLv2 inference on 12 columns."""
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import json

import numpy as np
import torch
from sklearn.metrics import brier_score_loss, roc_auc_score
from tabicl import TabICLClassifier

from bear_simulation import simulate
from bear_tabicl_cpu_batching import cpu_batched_tabicl
from bear_tabicl_disk_cache import disk_backed_kv_cache

HERE = Path(__file__).resolve().parent
CHECKPOINT = Path.home() / '.cache/huggingface/hub/models--jingang--TabICL/snapshots/4dcd344ece2c00be9e831fdd35bed57b5ad83e19/tabicl-classifier-v2-20260212.ckpt'
CHECKPOINT_SHA256 = 'bdc7dbd5e4ff21f8f0456fcf90c6b7cdf72dbea960f2d05b19bec19f9b3d4ed0'


def digest(path):
    with Path(path).open('rb') as stream:
        result = sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def classifier():
    return TabICLClassifier(n_estimators=4, batch_size=1, kv_cache=True,
                           device='cpu', n_jobs=2, random_state=4728,
                           model_path=CHECKPOINT,
                           checkpoint_version=CHECKPOINT.name,
                           use_amp=False, use_fa3=False,
                           offload_mode=False, allow_auto_download=False)


def main():
    if digest(CHECKPOINT) != CHECKPOINT_SHA256:
        raise ValueError('The checkpoint differs from the pinned public TabICLv2 weights')
    torch.set_num_threads(2)
    variants, y = simulate(4728, 576, n_noise=10)
    X = variants['continuous_noise']
    with cpu_batched_tabicl(column_batch_size=1, row_batch_size=8, icl_batch_size=1):
        native = classifier().fit(X[:512], y[:512])
        reference = native.predict_proba(X[512:])
        reference_classes = native.classes_.copy()
    del native
    gc.collect()
    with TemporaryDirectory(prefix='bear-tabicl-kv-check-', dir=HERE/'comparison-results') as cache_dir:
        with cpu_batched_tabicl(column_batch_size=1, row_batch_size=8, icl_batch_size=1), disk_backed_kv_cache(cache_dir):
            streamed = classifier().fit(X[:512], y[:512])
            actual = streamed.predict_proba(X[512:])
            np.testing.assert_array_equal(streamed.classes_, reference_classes)
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-6)
    result = {
        'passed': True, 'n_features': 12, 'n_train': 512, 'n_test': 64,
        'n_estimators': 4, 'batch_size': 1, 'cache_mode': 'kv',
        'stage_batch_limits': {'column': 1, 'row': 8, 'icl': 1},
        'average_logits': bool(streamed.average_logits),
        'max_probability_abs_error': float(np.max(np.abs(actual-reference))),
        'native_auc': float(roc_auc_score(y[512:], reference[:, 1])),
        'streamed_auc': float(roc_auc_score(y[512:], actual[:, 1])),
        'native_brier': float(brier_score_loss(y[512:], reference[:, 1])),
        'streamed_brier': float(brier_score_loss(y[512:], actual[:, 1])),
        'checkpoint': {'repo_id': 'jingang/TabICL', 'revision': CHECKPOINT.parent.name,
                       'filename': CHECKPOINT.name, 'sha256': CHECKPOINT_SHA256},
        'versions': {name: version(name) for name in ('numpy', 'scikit-learn', 'torch', 'tabicl')},
        'script_sha256': digest(__file__),
        'simulation_source_sha256': digest(HERE/'bear_simulation.py'),
        'disk_adapter_sha256': digest(HERE/'bear_tabicl_disk_cache.py'),
        'cpu_adapter_sha256': digest(HERE/'bear_tabicl_cpu_batching.py'),
        'tolerance': {'rtol': 2e-5, 'atol': 2e-6},
        'note': 'Native class permutations, normalization, four ensemble views, per-member logits and final probability aggregation are unchanged. Only native cache storage is streamed.',
    }
    path = HERE/'comparison-results'/'disk-cache-check.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
