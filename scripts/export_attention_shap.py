#!/usr/bin/env python3
"""Enumerate exact token Shapley values for the two Infinite Jest examples.

Run with the same pinned dependencies as export_attention_model.py:
    python scripts/export_attention_shap.py --output docs/attention/data/infinite-jest-shap.json

The players are the 13 non-special token positions. Missing players are replaced
by [MASK] token IDs; sequence length, position IDs, [CLS]/[SEP], and the all-one
attention mask stay fixed. Every coalition runs the complete frozen classifier.
No grouping hierarchy, sampling, gradients, attention attribution, or retraining
is used. The JSON preserves all coalition scores for independent verification.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import time


MODEL_ID = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
REVISION = "714eb0fa89d2f80546fda750413ed43d93601a13"
EXAMPLES = (
    ("negative", "Good and Infinite Jest don't belong in the same sentence"),
    ("positive", "Infinite Jest wasn't just good. It was spectacular."),
)


def exact_shapley(coalition_scores: list[float], feature_count: int) -> list[float]:
    """Ordinary Shapley values; bit i says whether player i is present."""
    if feature_count < 1 or len(coalition_scores) != 1 << feature_count:
        raise ValueError("Need one finite score for each coalition of all players")
    if not all(math.isfinite(value) for value in coalition_scores):
        raise ValueError("Coalition scores must be finite")
    weights = [1.0 / (feature_count * math.comb(feature_count - 1, size))
               for size in range(feature_count)]
    return [math.fsum(
        weights[bin(mask).count("1")] * (coalition_scores[mask | (1 << i)] - coalition_scores[mask])
        for mask in range(1 << feature_count) if not mask & (1 << i)
    ) for i in range(feature_count)]


def display_words(text: str, offsets: list[list[int]], values: list[float]) -> list[dict]:
    """Sum token credits for display, without redefining the coalition game."""
    groups = []
    covered = set()
    for match in re.finditer(r"\w+(?:['’]\w+)*|[^\w\s]", text):
        start, end = match.span()
        positions = [i for i, (left, right) in enumerate(offsets)
                     if right > left and left < end and right > start]
        if covered.intersection(positions):
            raise ValueError("A tokenizer piece crosses a display-word boundary")
        covered.update(positions)
        groups.append({"text": match.group(), "start": start, "end": end,
                       "token_positions": positions,
                       "shap_value": math.fsum(values[i] for i in positions)})
    expected = {i for i, (left, right) in enumerate(offsets) if right > left}
    if covered != expected:
        raise ValueError("Display grouping must cover every text token exactly once")
    return groups


def export(output_path: Path, reference_path: Path, batch_size: int = 64,
           threads: int = 2, local_files_only: bool = False) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    torch.set_num_threads(threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=False, use_fast=True,
        local_files_only=local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, revision=REVISION, trust_remote_code=False, use_safetensors=True,
        attn_implementation="eager", torch_dtype=torch.float32,
        local_files_only=local_files_only,
    ).to("cpu").eval()
    labels = [model.config.id2label[i] for i in range(model.config.num_labels)]
    assert labels == ["NEGATIVE", "POSITIVE"], labels
    assert tokenizer.mask_token_id is not None
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    assert reference["model"]["id"] == MODEL_ID
    assert reference["model"]["revision"] == REVISION
    reference_examples = {example["text"]: example for example in reference["examples"]}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output_path.parent / ".shap-checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    examples = []
    started = time.monotonic()

    for example_id, text in EXAMPLES:
        encoded = tokenizer(text, return_offsets_mapping=True, return_special_tokens_mask=True)
        token_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        special = encoded["special_tokens_mask"]
        positions = [i for i, is_special in enumerate(special) if not is_special]
        assert len(positions) == 13, positions
        assert token_ids == reference_examples[text]["token_ids"]
        assert encoded["attention_mask"] == [1] * len(token_ids)
        total = 1 << len(positions)
        ids = torch.tensor(token_ids, dtype=torch.long)
        position_tensor = torch.tensor(positions, dtype=torch.long)
        feature_bits = torch.arange(len(positions), dtype=torch.long)
        scores: list[float] = []
        checkpoint_path = checkpoint_dir / f"{example_id}.json"
        checkpoint_identity = {"model": MODEL_ID, "revision": REVISION, "token_ids": token_ids,
                               "mask_token_id": tokenizer.mask_token_id,
                               "torch": torch.__version__,
                               "transformers": importlib.metadata.version("transformers"),
                               "batch_size": batch_size}
        if checkpoint_path.exists():
            checkpoint = json.loads(checkpoint_path.read_text())
            if checkpoint.get("identity") == checkpoint_identity:
                scores = checkpoint["scores"]
        assert len(scores) <= total and (len(scores) % batch_size == 0 or len(scores) == total)
        example_started = time.monotonic()
        starting_count = len(scores)
        with torch.inference_mode():
            for start in range(starting_count, total, batch_size):
                stop = min(start + batch_size, total)
                masks = torch.arange(start, stop, dtype=torch.long)
                present = ((masks[:, None] >> feature_bits[None, :]) & 1).bool()
                batch_ids = ids.repeat(stop - start, 1)
                batch_ids[:, position_tensor] = torch.where(
                    present, ids[position_tensor][None, :], tokenizer.mask_token_id)
                attention_mask = torch.ones_like(batch_ids)
                logits = model(input_ids=batch_ids, attention_mask=attention_mask).logits
                margins = logits[:, 1] - logits[:, 0]
                assert torch.isfinite(margins).all()
                scores.extend(margins.tolist())
                if stop % (batch_size * 8) == 0 or stop == total:
                    checkpoint_path.write_text(json.dumps({"identity": checkpoint_identity,
                                                           "scores": scores}, allow_nan=False))
                    elapsed = time.monotonic() - example_started
                    rate = (stop - starting_count) / elapsed
                    print(json.dumps({"example": example_id, "coalitions": stop, "total": total,
                                      "elapsed_seconds": round(elapsed, 1),
                                      "remaining_seconds": round((total-stop)/rate, 1)}), flush=True)

        feature_values = exact_shapley(scores, len(positions))
        values = [0.0] * len(token_ids)
        for position, value in zip(positions, feature_values):
            values[position] = value
        expected_margin = (reference_examples[text]["logits"][1]
                           - reference_examples[text]["logits"][0])
        efficiency_error = abs(scores[0] + math.fsum(values) - scores[-1])
        reference_error = abs(scores[-1] - expected_margin)
        assert efficiency_error < 1e-10, efficiency_error
        assert reference_error < 2e-4, reference_error
        groups = display_words(text, offsets, values)
        assert abs(math.fsum(g["shap_value"] for g in groups) - math.fsum(values)) < 1e-10
        good_position = next(i for i, token in enumerate(tokenizer.convert_ids_to_tokens(token_ids))
                             if token == "good")
        good_bit = positions.index(good_position)
        full_without_good = (total - 1) ^ (1 << good_bit)
        examples.append({
            "id": example_id, "text": text,
            "tokens": tokenizer.convert_ids_to_tokens(token_ids),
            "token_ids": token_ids, "token_offsets": offsets,
            "special_tokens_mask": special, "feature_positions": positions,
            "shap_values": values, "display_words": groups,
            "baseline_margin": scores[0], "full_margin": scores[-1],
            "baseline_positive_probability": 1 / (1 + math.exp(-scores[0])),
            "positive_probability": 1 / (1 + math.exp(-scores[-1])),
            "coalition_count": total, "coalition_margins": scores,
            "illustrative_comparison": {
                "position": good_position, "token": "good",
                "without_margin": scores[full_without_good],
                "with_margin": scores[-1],
                "marginal_gain": scores[-1] - scores[full_without_good],
                "shap_value": values[good_position],
                "note": "This is one marginal contribution with every other token present; SHAP averages over every coalition of the other tokens.",
            },
            "validation": {"efficiency_error": efficiency_error,
                           "original_margin_error": reference_error,
                           "display_sum_error": abs(math.fsum(g["shap_value"] for g in groups)
                                                    - math.fsum(values))},
        })

    payload = {
        "schema_version": 1,
        "model": {"id": MODEL_ID, "revision": REVISION, "labels": labels,
                  "url": f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}"},
        "method": {
            "name": "exact-token-shapley",
            "score": "positive-minus-negative logit margin",
            "score_equation": "s(x) = z_positive(x) - z_negative(x) = log(p_positive(x) / p_negative(x))",
            "features": "Each non-special input token position is one player.",
            "mask_token": tokenizer.mask_token, "mask_token_id": tokenizer.mask_token_id,
            "masking": "Replace absent token IDs with [MASK]; preserve sequence length, positions, special tokens, and an all-one attention mask. Run the entire model for every coalition.",
            "baseline": "Model margin with all non-special positions replaced by [MASK]. This baseline need not be neutral and can depend on sequence length.",
            "coalition_indexing": "Array index is a bitmask. Bit i is 1 when feature_positions[i] retains its original token, and 0 when replaced by [MASK].",
            "formula": "phi_i = sum_{S subset N\\{i}} (1 / (n * binom(n-1, |S|))) * (v(S union {i}) - v(S))",
            "display_grouping": "Word labels sum the exact subword token Shapley values. They are not separately computed whole-word coalition Shapley values.",
            "scope": "Attributions describe this classifier's whole-input sentiment under this masking rule, not an intrinsic word sentiment or a causal explanation of its internal attention.",
            "approximation": "All coalitions enumerated; only model floating-point arithmetic is approximate. No Partition/Owen hierarchy or sampling.",
        },
        "inference": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "packages": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors", "numpy")},
            "device": "cpu", "dtype": "float32", "attention_implementation": "eager",
            "model_mode": "eval", "seed": 0, "threads": threads, "batch_size": batch_size,
            "deterministic_algorithms": True, "shapley_accumulation": "Python double with math.fsum",
            "reference_file": reference_path.name,
            "reference_sha256": hashlib.sha256(reference_bytes).hexdigest(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        },
        "examples": examples,
    }
    raw = (json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode()
    output_path.write_bytes(raw)
    print(json.dumps({"output": str(output_path), "bytes": len(raw),
                      "sha256": hashlib.sha256(raw).hexdigest(),
                      "examples": [{k: e[k] for k in ("id", "baseline_margin", "full_margin", "display_words", "validation")}
                                   for e in examples]}, indent=2), flush=True)
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).resolve().parents[1] / "docs/attention/data/infinite-jest-attention.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error("batch size and threads must be positive")
    export(args.output, args.reference, args.batch_size, args.threads, args.local_files_only)
