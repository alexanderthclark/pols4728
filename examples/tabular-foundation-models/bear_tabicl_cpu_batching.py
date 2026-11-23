"""Bound TabICL 2.2.0 CPU activations by batching independent stage inputs.

TabICL's released inference manager batches accelerator inputs but executes the
whole stage at once on CPU. This temporary process-local adapter applies the
same stage batch/cache utilities on CPU. It never splits an attention sequence,
changes model weights, drops features, or subsamples rows. Column-stage batches
contain independent feature groups; row-stage batches contain independent rows.
Dataset-wise ICL attention remains intact inside each ensemble member.

Use one adapter context around fit and prediction, with sklearn batch_size=1:

    with cpu_batched_tabicl(column_batch_size=8, row_batch_size=32):
        classifier.fit(X_context, y_context)
        probabilities = classifier.predict_proba(X_test_chunk)

The adapter is tested against the original implementation by the adjacent
check_bear_tabicl_cpu_batching.py script. No installed package files are modified.
"""
from collections import OrderedDict
from contextlib import contextmanager
from importlib.metadata import version
import math

import torch
from tabicl._model.inference import InferenceManager
from tabicl._model.kv_cache import KVCache


@contextmanager
def cpu_batched_tabicl(column_batch_size=8, row_batch_size=32, icl_batch_size=1):
    """Temporarily cap independent CPU batch elements in each model stage."""
    if version("tabicl") != "2.2.0":
        raise RuntimeError("This CPU adapter is validated for tabicl==2.2.0 only.")
    limits = {"tf_col": column_batch_size, "tf_row": row_batch_size, "tf_icl": icl_batch_size}
    if any(not isinstance(limit, int) or limit < 1 for limit in limits.values()):
        raise ValueError("All stage batch limits must be positive integers.")
    if getattr(InferenceManager, "_cpu_batch_adapter_active", False):
        raise RuntimeError("Nested CPU batching adapters are not supported.")
    original = InferenceManager.__call__

    def batched_call(manager, forward_fn, inputs, auto_batch=True, output_repeat=1):
        if not auto_batch or manager.exe_device.type != "cpu":
            return original(manager, forward_fn, inputs, auto_batch=auto_batch, output_repeat=output_repeat)
        if not manager._is_configured:
            raise RuntimeError("Inference manager must be configured before execution.")
        first = next(iter(inputs.values()))
        if not isinstance(first, torch.Tensor) or first.ndim < 3:
            raise ValueError("Stage input must start with a tensor of rank at least three.")
        batch_shape = tuple(first.shape[:-2])
        limit = limits[manager.enc_name]
        if math.prod(batch_shape) <= limit:
            return original(manager, forward_fn, inputs, auto_batch=False, output_repeat=output_repeat)
        split_sizes = manager.compute_split_sizes(batch_shape, limit)
        # Record empty caches before any chunk populates the original cache.
        stores = {name for name, value in inputs.items()
                  if isinstance(value, KVCache) and not value.is_populated()}
        output = None
        iterator = manager.create_multidim_batches(OrderedDict(inputs), batch_shape, split_sizes)
        for batch, indices in iterator:
            for name in stores:
                batch[name] = KVCache()
            chunk = manager._run_forward(forward_fn, batch)
            if not isinstance(chunk, torch.Tensor):
                raise TypeError("TabICL stage forward must return a tensor.")
            if output is None:
                output = torch.empty(batch_shape + tuple(chunk.shape[len(batch_shape):]),
                                     dtype=chunk.dtype, device=chunk.device)
            output[indices] = chunk
            for name in stores:
                part = batch[name]
                if part.is_populated():
                    target = inputs[name]
                    if not target.is_populated():
                        target.preallocate(part, batch_shape, device=manager.exe_device)
                    target[indices] = part
            del chunk, batch
        return output

    InferenceManager.__call__ = batched_call
    InferenceManager._cpu_batch_adapter_active = True
    try:
        yield
    finally:
        InferenceManager.__call__ = original
        del InferenceManager._cpu_batch_adapter_active
