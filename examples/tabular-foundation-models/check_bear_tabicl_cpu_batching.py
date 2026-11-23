"""Compare original and bounded-memory TabICL inference; no model training."""
import copy
import json
from pathlib import Path
from itertools import product

import torch
from tabicl._model.tabicl import TabICL
from tabicl import InferenceConfig
from bear_tabicl_cpu_batching import cpu_batched_tabicl


def tensors(cache):
    result = {}
    for name in ("col_cache", "icl_cache"):
        part = getattr(cache, name)
        if part is not None:
            for layer, entry in part.kv.items():
                result[f"{name}.{layer}.key"] = entry.key
                result[f"{name}.{layer}.value"] = entry.value
    if cache.row_repr is not None:
        result["row_repr"] = cache.row_repr
    return result


def main():
    torch.set_num_threads(2)
    torch.manual_seed(4728)
    config = InferenceConfig()
    for stage in (config.COL_CONFIG, config.ROW_CONFIG, config.ICL_CONFIG):
        stage.update({"device": "cpu", "use_amp": False, "use_fa3": False, "offload": False})
    checks = []
    # Small randomly initialized networks exercise the actual attention and cache
    # code. No fit method, gradient update, or pretrained checkpoint is used.
    for groups, ssmax in product((False, "same"), (False, "qassmax-mlp-elementwise")):
        model = TabICL(max_classes=2, embed_dim=16, col_num_blocks=2,
                       col_nhead=2, col_num_inds=5, col_feature_group=groups,
                       col_feature_group_size=3, col_target_aware=True,
                       col_ssmax=ssmax, row_num_blocks=2, row_nhead=2,
                       row_num_cls=2, icl_num_blocks=2, icl_nhead=2,
                       icl_ssmax=ssmax, zero_init=False).eval()
        X_train = torch.randn(2, 19, 11)
        X_test = torch.randn(2, 7, 11)
        y = torch.randint(0, 2, (2, 19))
        X = torch.cat([X_train, X_test], dim=1)
        with torch.inference_mode():
            ordinary = model(X.clone(), y.clone(), inference_config=config)
            with cpu_batched_tabicl(column_batch_size=3, row_batch_size=5, icl_batch_size=1):
                bounded = model(X.clone(), y.clone(), inference_config=config)
            torch.testing.assert_close(ordinary, bounded, rtol=2e-5, atol=2e-6)
            checks.append({"grouping": groups, "attention_scaling": ssmax, "mode": "uncached",
                           "max_abs_error": float((ordinary-bounded).abs().max())})
            for mode in ("kv", "repr"):
                model.clear_cache()
                model.forward_with_cache(X_train=X_train.clone(), y_train=y.clone(),
                                         store_cache=True, use_cache=False,
                                         cache_mode=mode, inference_config=config)
                reference_cache = copy.deepcopy(model._cache)
                ordinary_cached = model.forward_with_cache(X_test=X_test.clone(),
                                                            cache=reference_cache,
                                                            inference_config=config)
                model.clear_cache()
                with cpu_batched_tabicl(column_batch_size=3, row_batch_size=5, icl_batch_size=1):
                    model.forward_with_cache(X_train=X_train.clone(), y_train=y.clone(),
                                             store_cache=True, use_cache=False,
                                             cache_mode=mode, inference_config=config)
                    bounded_cache = copy.deepcopy(model._cache)
                    bounded_cached = model.forward_with_cache(X_test=X_test.clone(),
                                                               cache=bounded_cache,
                                                               inference_config=config)
                torch.testing.assert_close(ordinary_cached, bounded_cached, rtol=2e-5, atol=2e-6)
                a, b = tensors(reference_cache), tensors(bounded_cache)
                assert a.keys() == b.keys()
                for key in a:
                    torch.testing.assert_close(a[key], b[key], rtol=2e-5, atol=2e-6)
                checks.append({"grouping": groups, "attention_scaling": ssmax, "mode": mode,
                               "max_abs_error": float((ordinary_cached-bounded_cached).abs().max()),
                               "cache_tensors_compared": len(a),
                               "max_cache_abs_error": max(float((a[k]-b[k]).abs().max()) for k in a)})
    result = {"passed": True, "torch_version": torch.__version__,
              "tolerance": {"rtol": 2e-5, "atol": 2e-6}, "checks": checks}
    path = Path(__file__).with_name("comparison-results") / "cpu-batching-check.json"
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
