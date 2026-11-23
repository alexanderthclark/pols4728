"""Disk-backed native TabICL 2.2.0 classifier ensemble caches.

Use ``with disk_backed_kv_cache(cache_dir):`` around fit and every prediction,
with ``batch_size=1`` and CPU inference. Native training views, cache contents,
attention, feature/class permutations and prediction aggregation are unchanged.
Each ensemble member is saved before the next is computed; concatenation is
never used. Prediction memory-maps one member and releases it before the next.

The supplied directory is retained after the context exits. Native estimator
save/load is supported, but saves contain references to these external files,
not their contents: retain the directory and this importable module. Cache files
are trusted local torch saves; do not replace them with untrusted files. This
process-global temporary patch is not intended for concurrent classifier work.
"""
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from tabicl import TabICLClassifier
from tabicl._model.kv_cache import KVCache, KVCacheEntry, TabICLCache


def _cache_bytes(cache):
    total = 0
    for part in (cache.col_cache, cache.icl_cache):
        if part is not None:
            for entry in part.kv.values():
                for tensor in (entry.key, entry.value):
                    if tensor is not None:
                        total += tensor.numel() * tensor.element_size()
    if cache.row_repr is not None:
        total += cache.row_repr.numel() * cache.row_repr.element_size()
    return total


@dataclass(frozen=True)
class DiskTabICLCache:
    """Metadata-only proxy for one normalization method's native caches.

    ``col_cache`` exposes only an empty dtype marker for the native estimator's
    load-time dtype inspection. Actual cache tensors come from ``slice_batch``.
    Only one-member slices are accepted, ensuring no accidental materialization
    of the whole ensemble. ``to`` records a lazy device/dtype conversion.
    """
    paths: tuple[str, ...]
    train_shape: tuple[int, int, int]
    num_classes: int
    tensor_bytes: int
    stored_dtype: torch.dtype
    cache_type: str
    device: str = "cpu"
    dtype: torch.dtype | None = None

    @property
    def col_cache(self):
        marker = torch.empty(0, dtype=self.dtype or self.stored_dtype)
        return KVCache(kv={0: KVCacheEntry(key=marker, value=marker)})

    def cache_size_mb(self):
        """Logical stored tensor size, excluding file-format overhead."""
        return self.tensor_bytes // (1024 * 1024)

    def is_empty(self):
        return not self.paths

    def to(self, device, dtype=None):
        return replace(self, device=str(torch.device(device)),
                       dtype=self.dtype if dtype is None else dtype)

    def slice_batch(self, start, end):
        if not (0 <= start < end <= len(self.paths)) or end - start != 1:
            raise ValueError("Disk-backed TabICL caches require one-member slices (batch_size=1).")
        cache = torch.load(self.paths[start], map_location="cpu", mmap=True,
                           weights_only=False)
        if not isinstance(cache, TabICLCache) or cache.train_shape[0] != 1:
            raise ValueError("Invalid native one-member TabICL cache file.")
        if self.device != "cpu" or self.dtype is not None:
            cache = cache.to(self.device, dtype=self.dtype)
        return cache


@contextmanager
def disk_backed_kv_cache(cache_dir):
    """Temporarily use disk-backed native classifier caches on CPU.

    Files are placed in a unique subdirectory of ``cache_dir``. They survive
    context exit for estimator reuse/save/load; callers manage their lifetime.
    CPU stage batching can be nested with this context in either order.
    """
    if version("tabicl") != "2.2.0":
        raise RuntimeError("This disk cache adapter is validated for tabicl==2.2.0 only.")
    if getattr(TabICLClassifier, "_disk_cache_adapter_active", False):
        raise RuntimeError("Nested disk cache adapters are not supported.")
    root = Path(cache_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    original_build = TabICLClassifier._build_kv_cache
    original_forward = TabICLClassifier._batch_forward_with_cache

    def build(classifier):
        if classifier.batch_size != 1:
            raise ValueError("Disk-backed TabICL caches require batch_size=1.")
        if torch.device(classifier.device_).type != "cpu":
            raise ValueError("This disk cache adapter is validated for CPU inference only.")
        run_dir = root / uuid4().hex
        run_dir.mkdir()
        # This is exactly the native ensemble transformation, including all
        # normalization, feature shuffles and class shuffles already fitted.
        train_data = classifier.ensemble_generator_.transform(X=None, mode="train")
        classifier.model_kv_cache_ = OrderedDict()
        for norm_index, (norm_method, (Xs, ys)) in enumerate(train_data.items()):
            paths = []
            total_bytes = 0
            first_metadata = None
            # Native batch_size=1 np.array_split produces these same views.
            for member in range(Xs.shape[0]):
                X_batch = torch.from_numpy(Xs[member:member + 1]).float().to(classifier.device_)
                y_batch = torch.from_numpy(ys[member:member + 1]).float().to(classifier.device_)
                try:
                    with torch.no_grad():
                        classifier.model_.forward_with_cache(
                            X_train=X_batch, y_train=y_batch,
                            use_cache=False, store_cache=True,
                            cache_mode=classifier.cache_mode_,
                            inference_config=classifier.inference_config_)
                    cache = classifier.model_._cache
                    path = run_dir / f"norm_{norm_index:03d}_member_{member:04d}.pt"
                    torch.save(cache, path)
                    total_bytes += _cache_bytes(cache)
                    if first_metadata is None:
                        first_entry = next(iter(cache.col_cache.kv.values()))
                        first_metadata = (tuple(cache.train_shape), cache.num_classes,
                                          first_entry.key.dtype, cache.cache_type)
                        del first_entry
                    paths.append(str(path))
                    del cache
                finally:
                    classifier.model_.clear_cache()
                    del X_batch, y_batch
            if first_metadata is None:
                raise ValueError("Native ensemble generator returned no members.")
            shape, classes, dtype, cache_type = first_metadata
            classifier.model_kv_cache_[norm_method] = DiskTabICLCache(
                paths=tuple(paths), train_shape=(len(paths), shape[1], shape[2]),
                num_classes=classes, tensor_bytes=total_bytes,
                stored_dtype=dtype, cache_type=cache_type)
        classifier.disk_cache_directory_ = str(run_dir)

    def forward(classifier, Xs, kv_cache):
        if not isinstance(kv_cache, DiskTabICLCache):
            return original_forward(classifier, Xs, kv_cache)
        if classifier.batch_size != 1:
            raise ValueError("Disk-backed TabICL caches require batch_size=1.")
        if Xs.shape[0] != len(kv_cache.paths):
            raise ValueError("Test ensemble member count does not match the native cache.")
        # Same per-member operations as native _batch_forward_with_cache.
        # Its locals and model._cache retain the previous cache through the next
        # load, so explicitly release both after each native forward completes.
        outputs = []
        classifier.model_.clear_cache()
        for member in range(Xs.shape[0]):
            cache_subset = kv_cache.slice_batch(member, member + 1)
            X_batch = torch.from_numpy(Xs[member:member + 1]).float().to(classifier.device_)
            try:
                with torch.no_grad():
                    out = classifier.model_.forward_with_cache(
                        X_test=X_batch, cache=cache_subset,
                        return_logits=True if classifier.average_logits else False,
                        softmax_temperature=classifier.softmax_temperature,
                        inference_config=classifier.inference_config_)
                outputs.append(out.float().cpu().numpy())
                del out
            finally:
                classifier.model_.clear_cache()
                del cache_subset, X_batch
        return np.concatenate(outputs, axis=0)

    TabICLClassifier._build_kv_cache = build
    TabICLClassifier._batch_forward_with_cache = forward
    TabICLClassifier._disk_cache_adapter_active = True
    try:
        yield root
    finally:
        TabICLClassifier._build_kv_cache = original_build
        TabICLClassifier._batch_forward_with_cache = original_forward
        del TabICLClassifier._disk_cache_adapter_active
